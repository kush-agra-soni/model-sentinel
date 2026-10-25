"""Data, prediction and performance drift against a model's training reference."""
import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE, PSI_MAJOR = 0.1, 0.25


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    rv = ref.astype(str).value_counts(normalize=True)
    cv = cur.astype(str).value_counts(normalize=True)
    cats = rv.index.union(cv.index)
    r = np.clip(rv.reindex(cats, fill_value=0).to_numpy(), 1e-4, None)
    c = np.clip(cv.reindex(cats, fill_value=0).to_numpy(), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(p: float) -> str:
    return "major" if p >= PSI_MAJOR else "moderate" if p >= PSI_MODERATE else "stable"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, schema: dict) -> list[dict]:
    rows = []
    for c, spec in schema.items():
        if c not in current.columns or c not in reference.columns:
            continue
        if spec["type"] == "numeric":
            r = pd.to_numeric(reference[c], errors="coerce").to_numpy(dtype=float)
            x = pd.to_numeric(current[c], errors="coerce").to_numpy(dtype=float)
            p = psi(r, x)
            rr, xx = r[~np.isnan(r)], x[~np.isnan(x)]
            pval = float(stats.ks_2samp(rr, xx).pvalue) if len(rr) > 5 and len(xx) > 5 else 1.0
            rows.append({"feature": c, "type": "numeric", "psi": round(p, 4), "test": "KS", "p_value": round(pval, 5),
                         "ref_mean": round(float(np.nanmean(r)), 4), "cur_mean": round(float(np.nanmean(x)), 4),
                         "level": level(p)})
        else:
            p = psi_categorical(reference[c], current[c])
            cats = reference[c].astype(str).value_counts().index.union(current[c].astype(str).value_counts().index)
            table = np.array([reference[c].astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(),
                              current[c].astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()])
            table = table[:, table.sum(axis=0) > 0]
            pval = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else 1.0
            new = sorted(set(current[c].dropna().astype(str)) - set(reference[c].dropna().astype(str)))[:5]
            rows.append({"feature": c, "type": "categorical", "psi": round(p, 4), "test": "chi2", "p_value": round(pval, 5),
                         "new_categories": new, "level": level(p)})
    return sorted(rows, key=lambda r: -r["psi"])


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, schema: dict, ref_scores=None, cur_scores=None,
                 importance: list[dict] | None = None) -> dict:
    feats = feature_drift(reference, current, schema)
    imp = {i["feature"]: max(i["importance"], 0) for i in (importance or [])}
    total_imp = sum(imp.values()) or 1
    weighted = sum(f["psi"] * imp.get(f["feature"], 0) for f in feats) / total_imp if imp else float(np.mean([f["psi"] for f in feats] or [0]))
    pred = None
    if ref_scores is not None and cur_scores is not None:
        pred = {"psi": round(psi(np.asarray(ref_scores, float), np.asarray(cur_scores, float)), 4),
                "ref_mean": round(float(np.mean(ref_scores)), 4), "cur_mean": round(float(np.mean(cur_scores)), 4)}
        pred["level"] = level(pred["psi"])
    n_major = sum(f["level"] == "major" for f in feats)
    n_mod = sum(f["level"] == "moderate" for f in feats)
    status = "major" if n_major or (pred and pred["level"] == "major") or weighted >= PSI_MAJOR else \
        "moderate" if n_mod or (pred and pred["level"] == "moderate") or weighted >= PSI_MODERATE else "stable"
    return {"status": status, "rows": len(current), "features_major": n_major, "features_moderate": n_mod,
            "importance_weighted_psi": round(float(weighted), 4), "prediction_drift": pred, "features": feats}


def performance_check(metric_name: str, current_value: float, reference_value: float, floor: float,
                      max_relative_drop: float = 0.05) -> dict:
    drop = (reference_value - current_value) / reference_value if reference_value else 0.0
    below_floor = current_value < floor
    degraded = drop > max_relative_drop
    return {"metric": metric_name, "current": round(current_value, 4), "reference": round(reference_value, 4),
            "floor": floor, "relative_drop": round(drop, 4), "below_floor": below_floor, "degraded": degraded,
            "action": "retrain" if below_floor or degraded else "none"}


def simulate_drift(df: pd.DataFrame, schema: dict, strength: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Shift numeric features and re-weight categories, for demos and tests."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c, spec in schema.items():
        if c not in out:
            continue
        if spec["type"] == "numeric":
            out[c] = pd.to_numeric(out[c], errors="coerce") + strength * (spec["std"] or 1) * 0.8
        elif len(spec["values"]) > 1:
            rare = spec["values"][-1]
            mask = rng.random(len(out)) < 0.35 * strength
            out.loc[mask, c] = rare
    return out
