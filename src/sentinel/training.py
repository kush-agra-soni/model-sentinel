"""Model training: leakage-safe pipelines, cross-validated model selection and holdout evaluation."""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections.abc import Callable
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
                             silhouette_score)

warnings.filterwarnings("ignore", category=UserWarning)


def feature_columns(df: pd.DataFrame, plan: dict) -> tuple[list[str], list[str]]:
    feats = [c for c in df.columns if c != plan.get("target") and plan["columns"].get(c, {}).get("role") != "time"]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and plan["columns"].get(c, {}).get("kind", "numeric") != "categorical"]
    cat = [c for c in feats if c not in num]
    return num, cat


def preprocessor(num: list[str], cat: list[str], scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", Pipeline(cat_steps), cat)],
                             remainder="drop", verbose_feature_names_out=False)


def candidates(task: str) -> dict:
    from lightgbm import LGBMClassifier, LGBMRegressor
    if task == "classification":
        return {
            "logistic_regression": (LogisticRegression(max_iter=2000, C=1.0), True),
            "random_forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0), False),
            "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0), False),
            "lightgbm": (LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.9,
                                        colsample_bytree=0.9, random_state=0, verbose=-1), False),
        }
    return {
        "ridge": (Ridge(alpha=1.0), True),
        "random_forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0), False),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0), False),
        "lightgbm": (LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.9,
                                   colsample_bytree=0.9, random_state=0, verbose=-1), False),
    }


def _cls_metrics(y_true, proba, classes, positive) -> dict:
    pred = classes[np.argmax(proba, axis=1)]
    out = {"accuracy": accuracy_score(y_true, pred), "f1_macro": f1_score(y_true, pred, average="macro")}
    if len(classes) == 2:
        pos_idx = list(classes).index(positive)
        p = proba[:, pos_idx]
        yb = (np.asarray(y_true) == positive).astype(int)
        base = yb.mean()
        brier = brier_score_loss(yb, p)
        out.update(roc_auc=roc_auc_score(yb, p), precision=precision_score(yb, (p >= 0.5).astype(int), zero_division=0),
                   recall=recall_score(yb, (p >= 0.5).astype(int)), f1=f1_score(yb, (p >= 0.5).astype(int)),
                   brier=brier, brier_skill=1 - brier / max(base * (1 - base), 1e-9))
    else:
        out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
    return {k: round(float(v), 4) for k, v in out.items()}


def _reg_metrics(y, pred) -> dict:
    return {k: round(float(v), 4) for k, v in {"rmse": mean_squared_error(y, pred) ** 0.5, "mae": mean_absolute_error(y, pred),
                                                "r2": r2_score(y, pred)}.items()}


def train_supervised(df: pd.DataFrame, plan: dict, progress: Callable[[str], None] = lambda m: None,
                     models: list[str] | None = None, test_size: float = 0.2, cv_folds: int = 5) -> dict:
    task, target = plan["task"], plan["target"]
    num, cat = feature_columns(df, plan)
    X, y = df[num + cat], df[target]
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    progress(f"Split {len(X_tr)} training / {len(X_te)} holdout rows; {len(num)} numeric and {len(cat)} categorical features")
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=0) if task == "classification" else KFold(cv_folds, shuffle=True, random_state=0)
    scoring = ("roc_auc" if y.nunique() == 2 else "roc_auc_ovr") if task == "classification" else "neg_root_mean_squared_error"
    board = []
    pool = candidates(task)
    if models:
        pool = {k: v for k, v in pool.items() if k in models}
    y_cv = y_tr if task != "classification" or y.nunique() != 2 else (y_tr == plan["positive_class"]).astype(int)
    for name, (est, scale) in pool.items():
        t0 = time.time()
        pipe = Pipeline([("prep", preprocessor(num, cat, scale)), ("model", est)])
        scores = cross_val_score(pipe, X_tr, y_cv, cv=cv, scoring=scoring, n_jobs=1)
        cv_score = float(np.mean(scores)) * (-1 if scoring.startswith("neg_") else 1)
        board.append({"model": name, "cv_score": round(cv_score, 4), "cv_std": round(float(np.std(scores)), 4),
                      "metric": "roc_auc" if task == "classification" else "rmse", "seconds": round(time.time() - t0, 1)})
        progress(f"{name}: CV {board[-1]['metric']} {cv_score:.4f} +/- {np.std(scores):.4f} ({board[-1]['seconds']} s)")
    board.sort(key=lambda r: -r["cv_score"] if task == "classification" else r["cv_score"])
    best = board[0]["model"]
    est, scale = candidates(task)[best]
    pipe = Pipeline([("prep", preprocessor(num, cat, scale)), ("model", est)])
    pipe.fit(X_tr, y_tr)
    progress(f"Best model: {best}; evaluating on the holdout set")

    report = {"leaderboard": board, "best_model": best, "features": {"numeric": num, "categorical": cat},
              "train_rows": len(X_tr), "test_rows": len(X_te)}
    if task == "classification":
        classes = pipe.classes_
        proba = pipe.predict_proba(X_te)
        report["metrics"] = _cls_metrics(y_te, proba, classes, plan.get("positive_class"))
        dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
        report["baseline"] = _cls_metrics(y_te, dummy.predict_proba(X_te), dummy.classes_, plan.get("positive_class"))
        pred = classes[np.argmax(proba, axis=1)]
        report["confusion"] = {"labels": [str(c) for c in classes],
                               "matrix": confusion_matrix(y_te, pred, labels=classes).tolist()}
        if len(classes) == 2:
            p = proba[:, list(classes).index(plan["positive_class"])]
            yb = (y_te == plan["positive_class"]).astype(int).to_numpy()
            bins = np.clip((p * 10).astype(int), 0, 9)
            report["calibration"] = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "predicted": round(float(p[bins == b].mean()), 3),
                                      "observed": round(float(yb[bins == b].mean()), 3), "n": int((bins == b).sum())}
                                     for b in range(10) if (bins == b).any()]
        scoring_imp = "roc_auc" if len(classes) == 2 else "accuracy"
        ref_scores = proba[:, list(classes).index(plan["positive_class"])] if len(classes) == 2 else None
    else:
        pred = pipe.predict(X_te)
        report["metrics"] = _reg_metrics(y_te, pred)
        report["baseline"] = _reg_metrics(y_te, DummyRegressor().fit(X_tr, y_tr).predict(X_te))
        scoring_imp, ref_scores = "neg_root_mean_squared_error", pred
    sample = min(len(X_te), 2000)
    imp = permutation_importance(pipe, X_te.iloc[:sample], y_te.iloc[:sample], n_repeats=5, random_state=0,
                                 scoring=scoring_imp, n_jobs=1)
    report["importance"] = sorted(({"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
                                   for f, m, s in zip(X.columns, imp.importances_mean, imp.importances_std)),
                                  key=lambda r: -r["importance"])[:20]
    progress("Computed permutation importance")
    return {"pipeline": pipe, "report": report, "reference": X_tr.sample(min(len(X_tr), 5000), random_state=0),
            "reference_scores": None if ref_scores is None else np.asarray(ref_scores)[:5000]}


def train_clustering(df: pd.DataFrame, plan: dict, progress: Callable[[str], None] = lambda m: None,
                     k_range: range = range(2, 9)) -> dict:
    num, cat = feature_columns(df, plan)
    X = df[num + cat]
    prep = preprocessor(num, cat, True)
    Z = prep.fit_transform(X)
    board = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(len(Z), 3000), random_state=0)
        board.append({"k": k, "silhouette": round(float(sil), 4), "inertia": round(float(km.inertia_), 1)})
        progress(f"k={k}: silhouette {sil:.3f}")
    best_k = max(board, key=lambda r: r["silhouette"])["k"]
    pipe = Pipeline([("prep", preprocessor(num, cat, True)), ("model", KMeans(n_clusters=best_k, n_init=10, random_state=0))])
    labels = pipe.fit_predict(X)
    prof = X.assign(segment=labels).groupby("segment")[num].mean().round(2) if num else pd.DataFrame()
    sizes = pd.Series(labels).value_counts().sort_index()
    report = {"leaderboard": board, "best_model": f"kmeans_k{best_k}", "k": best_k,
              "metrics": {"silhouette": max(r["silhouette"] for r in board), "clusters": best_k},
              "segments": [{"segment": int(s), "size": int(sizes[s]), **{c: float(prof.loc[s, c]) for c in num}}
                           for s in sizes.index], "features": {"numeric": num, "categorical": cat}, "train_rows": len(X)}
    return {"pipeline": pipe, "report": report, "reference": X.sample(min(len(X), 5000), random_state=0),
            "reference_scores": None}


def make_lag_frame(series: pd.Series, lags=(1, 2, 3, 6, 12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.to_numpy()}, index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["month"] = series.index.month if hasattr(series.index, "month") else 0
    df["t"] = np.arange(len(df))
    return df


def train_forecast(df: pd.DataFrame, plan: dict, progress: Callable[[str], None] = lambda m: None,
                   horizon: int = 12) -> dict:
    time_col = next(c for c, e in plan["columns"].items() if e["role"] == "time")
    s = df.sort_values(time_col).set_index(pd.to_datetime(df.sort_values(time_col)[time_col]))[plan["target"]].astype(float)
    use_log = bool((s > 0).all())
    ys = np.log(s) if use_log else s
    frame = make_lag_frame(ys).dropna()
    train, test = frame.iloc[:-horizon], frame.iloc[-horizon:]
    feats = [c for c in frame.columns if c != "y"]
    models = {"ridge": Ridge(alpha=1.0), "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=200, random_state=0)}
    board = []
    fitted = {}
    for name, m in models.items():
        m.fit(train[feats], train["y"])
        preds = recursive_forecast(m, ys.iloc[:-horizon], horizon)
        pred_lvl = np.exp(preds) if use_log else preds
        mets = _reg_metrics(s.iloc[-horizon:].to_numpy(), pred_lvl)
        mets["mape"] = round(float(np.mean(np.abs((s.iloc[-horizon:].to_numpy() - pred_lvl) / s.iloc[-horizon:].to_numpy())) * 100), 2)
        board.append({"model": name, **mets})
        fitted[name] = m
        progress(f"{name}: holdout MAPE {mets['mape']}%")
    naive = s.iloc[-horizon - 12:-12].to_numpy() if len(s) > horizon + 12 else np.repeat(s.iloc[-horizon - 1], horizon)
    base = _reg_metrics(s.iloc[-horizon:].to_numpy(), naive)
    base["mape"] = round(float(np.mean(np.abs((s.iloc[-horizon:].to_numpy() - naive) / s.iloc[-horizon:].to_numpy())) * 100), 2)
    board.sort(key=lambda r: r["mape"])
    best = board[0]["model"]
    final = type(fitted[best])(**fitted[best].get_params()).fit(frame[feats], frame["y"])
    future = recursive_forecast(final, ys, horizon)
    idx = pd.date_range(s.index[-1], periods=horizon + 1, freq=pd.infer_freq(s.index) or "MS")[1:]
    report = {"leaderboard": board, "best_model": best, "metrics": {k: v for k, v in board[0].items() if k != "model"},
              "baseline": base, "baseline_name": "seasonal naive (same month last year)", "log_transform": use_log,
              "history": [{"date": str(d.date()), "value": float(v)} for d, v in s.items()],
              "holdout": [{"date": str(d.date()), "actual": float(a)} for d, a in s.iloc[-horizon:].items()],
              "forecast": [{"date": str(d.date()), "value": round(float(np.exp(v) if use_log else v), 2)} for d, v in zip(idx, future)],
              "train_rows": len(train), "test_rows": len(test), "features": {"numeric": feats, "categorical": []}}
    return {"pipeline": final, "report": report, "reference": frame[feats], "reference_scores": None,
            "series": ys, "use_log": use_log}


def recursive_forecast(model, history: pd.Series, horizon: int) -> np.ndarray:
    hist = history.copy()
    freq = pd.infer_freq(hist.index) or "MS"
    out = []
    for _ in range(horizon):
        nxt = hist.index[-1] + pd.tseries.frequencies.to_offset(freq)
        ext = pd.concat([hist, pd.Series([np.nan], index=[nxt])])
        row = make_lag_frame(ext).iloc[[-1]].drop(columns="y")
        v = float(model.predict(row)[0])
        out.append(v)
        hist = pd.concat([hist, pd.Series([v], index=[nxt])])
    return np.array(out)
