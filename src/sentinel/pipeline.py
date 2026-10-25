"""End-to-end operations: train a project, monitor a batch, retrain with a champion/challenger gate."""
import time
import numpy as np
import pandas as pd
from .serving import prepare
from .registry import Registry
from .datasets import file_hash
from .planner import apply_plan
from collections.abc import Callable
from .drift import drift_report, performance_check
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from .training import train_clustering, train_forecast, train_supervised


def train_project(reg: Registry, project: str, df: pd.DataFrame, plan: dict, dataset: dict,
                  progress: Callable[[str], None], models: list[str] | None = None) -> dict:
    clean, log = apply_plan(df, plan)
    for line in log:
        progress(line)
    if len(clean) < 20:
        raise ValueError("fewer than 20 usable rows after cleaning")
    if plan["task"] == "clustering":
        res = train_clustering(clean, plan, progress)
    elif plan["task"] == "forecasting":
        res = train_forecast(clean, plan, progress)
    else:
        res = train_supervised(clean, plan, progress, models=models)
    res["report"]["cleaning_log"] = log
    meta = reg.register(project, res, plan, {**dataset, "rows": len(df), "hash": file_hash(df)})
    progress(f"Registered {meta['model_id']}" + (" as champion" if reg.champion(project) == meta["model_id"] else ""))
    return meta


def primary_metric(task: str) -> str:
    return {"classification": "roc_auc", "regression": "rmse"}.get(task, "")


def evaluate_on(model, meta: dict, df: pd.DataFrame) -> float | None:
    tgt, task = meta["target"], meta["task"]
    if tgt not in df.columns or task not in ("classification", "regression"):
        return None
    X, _ = prepare(df.drop(columns=[tgt]), meta["schema"])
    y = df[tgt]
    if task == "classification":
        proba = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        y = y.astype(str).str.strip()
        if len(classes) == 2 and meta.get("positive_class") in classes:
            yb = (y == meta["positive_class"]).astype(int)
            return float(roc_auc_score(yb, proba[:, classes.index(meta["positive_class"])])) if yb.nunique() == 2 else None
        return float(roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_))
    return float(mean_squared_error(pd.to_numeric(y, errors="coerce"), model.predict(X)) ** 0.5)


def monitor_batch(reg: Registry, project: str, raw: pd.DataFrame, floor: float | None = None) -> dict:
    mid = reg.champion(project)
    if not mid:
        raise KeyError("project has no champion")
    meta, model = reg.meta(mid), reg.load(mid)
    if meta["task"] == "forecasting":
        raise ValueError("drift monitoring applies to classification, regression and clustering models")
    clean, _ = apply_plan(raw, meta["plan"], inference=True)
    if meta["target"] in clean.columns:
        clean[meta["target"]] = clean[meta["target"]].astype("string").str.strip() if meta["task"] == "classification"             else pd.to_numeric(clean[meta["target"]], errors="coerce")
    X, _ = prepare(clean.drop(columns=[meta["target"]], errors="ignore"), meta["schema"])
    cur_scores = None
    if meta["task"] == "classification" and meta.get("positive_class"):
        classes = [str(c) for c in model.classes_]
        if meta["positive_class"] in classes:
            cur_scores = model.predict_proba(X)[:, classes.index(meta["positive_class"])]
    elif meta["task"] == "regression":
        cur_scores = model.predict(X)
    rep = drift_report(reg.reference(mid), X, meta["schema"], reg.reference_scores(mid), cur_scores,
                       meta["report"].get("importance"))
    rep["model_id"] = mid
    metric = primary_metric(meta["task"])
    cur = evaluate_on(model, meta, clean)
    if cur is not None:
        ref = meta["report"]["metrics"][metric]
        if meta["task"] == "classification":
            perf = performance_check(metric, cur, ref, floor if floor is not None else max(0.6, ref - 0.1))
        else:  # lower RMSE is better: express as a negative so "drop" keeps its meaning
            perf = performance_check("neg_rmse", -cur, -ref, -(floor if floor is not None else ref * 1.25))
            perf.update(metric="rmse", current=round(cur, 4), reference=round(ref, 4))
        rep["performance"] = perf
    rep["recommendation"] = ("retrain" if (rep.get("performance", {}).get("action") == "retrain" or rep["status"] == "major")
                             else "watch" if rep["status"] == "moderate" else "none")
    rep["ts"] = time.time()
    return rep


def retrain(reg: Registry, project: str, new_data: pd.DataFrame, progress: Callable[[str], None],
            min_gain: float = 0.005, floor: float | None = None) -> dict:
    """Train a challenger on recent labelled data and promote it only if it beats the champion on held-out recent rows."""
    champ_id = reg.champion(project)
    champ_meta, champ = reg.meta(champ_id), reg.load(champ_id)
    plan = champ_meta["plan"]
    if champ_meta["task"] not in ("classification", "regression"):
        raise ValueError("retraining with a gate needs a supervised model")
    if plan["target"] not in new_data.columns:
        raise ValueError("new data must include the target column to retrain")
    clean, _ = apply_plan(new_data, {**plan, "fitted": {}})
    strat = clean[plan["target"]] if champ_meta["task"] == "classification" else None
    fit_part, gate_part = train_test_split(clean, test_size=0.3, random_state=7, stratify=strat)
    progress(f"Challenger trains on {len(fit_part)} recent rows; gate evaluates on {len(gate_part)} held-out recent rows")
    meta = train_project(reg, project, fit_part, {**plan, "drop_duplicates": False},
                         {"id": "retrain", "parent": champ_id}, progress, models=[champ_meta["report"]["best_model"]])
    chal = reg.load(meta["model_id"])
    c_score, n_score = evaluate_on(champ, champ_meta, gate_part), evaluate_on(chal, meta, gate_part)
    higher_better = champ_meta["task"] == "classification"
    gain = (n_score - c_score) if higher_better else (c_score - n_score)
    floor_ok = True if floor is None else (n_score >= floor if higher_better else n_score <= floor)
    promote = gain >= min_gain and floor_ok
    reason = (f"challenger {primary_metric(champ_meta['task'])} {n_score:.4f} vs champion {c_score:.4f} on recent data"
              f" (gain {gain:+.4f}, required {min_gain})")
    if promote:
        reg.promote(project, meta["model_id"], "retrain gate passed: " + reason)
    progress(("Promoted: " if promote else "Kept champion: ") + reason)
    return {"challenger": meta["model_id"], "champion_before": champ_id, "promoted": promote,
            "champion_score": round(c_score, 4), "challenger_score": round(n_score, 4), "gain": round(gain, 4),
            "reason": reason, "floor_ok": floor_ok}


def forecast(reg: Registry, project: str, horizon: int) -> list[dict]:
    from .training import recursive_forecast
    mid = reg.champion(project)
    meta = reg.meta(mid)
    if meta["task"] != "forecasting":
        raise ValueError("not a forecasting project")
    series = reg.series(mid)
    vals = recursive_forecast(reg.load(mid), series, horizon)
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=pd.infer_freq(series.index) or "MS")[1:]
    return [{"date": str(d.date()), "value": round(float(np.exp(v) if meta.get("use_log") else v), 2)} for d, v in zip(idx, vals)]
