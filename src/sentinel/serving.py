"""Online and batch prediction with input validation and model-agnostic local explanations."""
import time
import numpy as np
import pandas as pd


class InputError(ValueError):
    pass


def prepare(records: list[dict] | pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[list[str]]]:
    """Coerce to the training schema; collect warnings per row instead of failing on recoverable problems."""
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    if df.empty:
        raise InputError("no rows to score")
    warnings = [[] for _ in range(len(df))]
    unknown = [c for c in df.columns if c not in schema]
    for c, spec in schema.items():
        if c not in df.columns:
            df[c] = np.nan
            for w in warnings:
                w.append(f"{c}: missing, imputed")
            continue
        if spec["type"] == "numeric":
            raw = df[c]
            df[c] = pd.to_numeric(raw, errors="coerce")
            bad = df[c].isna() & raw.notna()
            for i in np.flatnonzero(bad.to_numpy()):
                warnings[i].append(f"{c}: '{raw.iloc[i]}' is not a number, imputed")
            std = spec["std"] or 1.0
            z = (df[c] - spec["mean"]).abs() / std
            for i in np.flatnonzero((z > 4).to_numpy()):
                warnings[i].append(f"{c}: {df[c].iloc[i]:g} is far outside the training range "
                                   f"[{spec['p01']:g}, {spec['p99']:g}]")
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
            known = set(spec["values"])
            for i, v in enumerate(df[c].tolist()):
                if v is not np.nan and v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) not in known:
                    warnings[i].append(f"{c}: unseen category '{v}'")
            df[c] = df[c].map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    if unknown and warnings:
        warnings[0].append(f"ignored unknown columns: {', '.join(unknown[:10])}")
    return df[list(schema)], warnings


def predict(model, meta: dict, df: pd.DataFrame) -> list[dict]:
    task = meta["task"]
    if task == "classification":
        proba = model.predict_proba(df)
        classes = [str(c) for c in model.classes_]
        out = []
        for p in proba:
            i = int(np.argmax(p))
            row = {"prediction": classes[i], "confidence": round(float(p[i]), 4),
                   "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)}}
            if meta.get("positive_class") in classes:
                row["score"] = round(float(p[classes.index(meta["positive_class"])]), 4)
            out.append(row)
        return out
    if task == "regression":
        return [{"prediction": round(float(v), 4)} for v in model.predict(df)]
    if task == "clustering":
        return [{"segment": int(v)} for v in model.predict(df)]
    raise InputError(f"use /forecast for {task} models")


def _score(model, meta, df) -> np.ndarray:
    if meta["task"] == "classification":
        proba = model.predict_proba(df)
        classes = [str(c) for c in model.classes_]
        pos = meta.get("positive_class")
        return proba[:, classes.index(pos)] if pos in classes else proba.max(axis=1)
    return np.asarray(model.predict(df), dtype=float)


def explain(model, meta: dict, row: pd.DataFrame, top: int = 6) -> list[dict]:
    """Ablation explanation: how much the score moves when each feature is replaced by its typical value."""
    if meta["task"] not in ("classification", "regression"):
        return []
    base = float(_score(model, meta, row)[0])
    variants = []
    for c, spec in meta["schema"].items():
        alt = row.copy()
        alt[c] = spec["median"] if spec["type"] == "numeric" else spec["mode"]
        variants.append(alt)
    scores = _score(model, meta, pd.concat(variants, ignore_index=True))
    contrib = [{"feature": c, "value": None if pd.isna(row[c].iloc[0]) else (row[c].iloc[0].item()
                                                                              if hasattr(row[c].iloc[0], "item") else row[c].iloc[0]),
                "contribution": round(base - float(s), 4)} for c, s in zip(meta["schema"], scores)]
    return sorted(contrib, key=lambda r: -abs(r["contribution"]))[:top]


def score_records(model, meta: dict, records, with_explanations: bool = False) -> dict:
    from .planner import apply_plan
    t0 = time.perf_counter()
    raw = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
    if raw.empty:
        raise InputError("no rows to score")
    fixed, _ = apply_plan(raw.drop(columns=[meta.get("target")], errors="ignore"), meta["plan"], inference=True)
    df, warns = prepare(fixed, meta["schema"])
    preds = predict(model, meta, df)
    for p, w in zip(preds, warns):
        p["warnings"] = w
    if with_explanations and len(df) <= 20:
        for i, p in enumerate(preds):
            p["explanation"] = explain(model, meta, df.iloc[[i]])
    return {"model_id": meta["model_id"], "predictions": preds, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
