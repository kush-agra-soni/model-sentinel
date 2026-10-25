"""Automatic EDA and data-quality scorecard."""
import re
import numpy as np
import pandas as pd

ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|customerid|^key$)", re.I)


def infer_kind(s: pd.Series, n_rows: int) -> str:
    """numeric | categorical | datetime | text | id | constant | numeric_as_text"""
    nn = s.dropna()
    if nn.empty or nn.nunique() <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        if ID_NAME.search(str(s.name)) and nn.nunique() > 0.95 * len(nn):
            return "id"
        return "categorical" if nn.nunique() <= 10 and (nn % 1 == 0).all() and nn.nunique() < 0.05 * n_rows + 3 else "numeric"
    sample = nn.astype(str).str.strip().head(500)
    num = pd.to_numeric(sample, errors="coerce")
    if num.notna().mean() > 0.95:
        return "numeric_as_text"
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        try:
            pd.to_datetime(sample, errors="raise")
            return "datetime"
        except (ValueError, TypeError):
            pass
    uniq = nn.nunique()
    if uniq > 0.95 * len(nn) and (ID_NAME.search(str(s.name)) or sample.str.len().mean() < 40):
        return "id"
    if ID_NAME.search(str(s.name)) and uniq > 0.3 * len(nn) and uniq > 50:  # IDs repeated across stacked extracts
        return "id"
    if sample.str.len().mean() > 60 and uniq > 0.5 * len(nn):
        return "text"
    return "categorical"


def profile(df: pd.DataFrame, target: str | None = None, max_cats: int = 12) -> dict:
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        kind = infer_kind(s, n)
        missing = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum() if s.dtype == object else s.isna().sum())
        info = {"name": c, "kind": kind, "dtype": str(s.dtype), "missing": missing, "missing_pct": round(missing / max(n, 1) * 100, 2),
                "unique": int(s.nunique(dropna=True)), "is_target": c == target}
        if kind in ("numeric", "numeric_as_text"):
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x):
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
                counts, edges = np.histogram(x, bins=min(20, max(5, x.nunique())))
                info.update(mean=round(float(x.mean()), 4), std=round(float(x.std()), 4), min=float(x.min()),
                            p50=float(x.median()), max=float(x.max()), skew=round(float(x.skew()), 3),
                            outliers=out, hist={"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]})
                if kind == "numeric_as_text":
                    info["unparseable"] = int(pd.to_numeric(s.astype(str).str.strip().replace("", np.nan), errors="coerce").isna().sum() - s.isna().sum())
        elif kind in ("categorical", "constant"):
            vc = s.astype(str).str.strip().value_counts().head(max_cats)
            info["top"] = {str(k): int(v) for k, v in vc.items()}
            raw = s.dropna().astype(str)
            variants = raw.str.strip().str.lower().nunique()
            info["case_or_space_variants"] = int(raw.nunique() - variants)
        elif kind == "datetime":
            d = pd.to_datetime(s, errors="coerce").dropna()
            if len(d):
                info.update(min=str(d.min()), max=str(d.max()))
        cols.append(info)

    num = df[[c["name"] for c in cols if c["kind"] == "numeric"]]
    corr = []
    if num.shape[1] >= 2:
        m = num.corr(numeric_only=True)
        for i, a in enumerate(m.columns):
            for b in m.columns[i + 1:]:
                v = m.loc[a, b]
                if pd.notna(v) and abs(v) >= 0.85:
                    corr.append({"a": a, "b": b, "r": round(float(v), 3)})
    dups = int(df.duplicated().sum())
    return {"rows": n, "columns": len(df.columns), "duplicate_rows": dups, "column_profiles": cols,
            "high_correlations": sorted(corr, key=lambda x: -abs(x["r"]))[:15],
            "target_balance": _balance(df, target) if target in df.columns else None,
            "quality": quality(cols, n, dups)}


def _balance(df: pd.DataFrame, target: str) -> dict:
    s = df[target]
    if s.nunique() <= 20:
        vc = s.astype(str).value_counts(normalize=True)
        return {"type": "classes", "share": {k: round(float(v), 4) for k, v in vc.items()}}
    x = pd.to_numeric(s, errors="coerce")
    return {"type": "numeric", "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4)}


def quality(cols: list[dict], n: int, dups: int) -> dict:
    total_cells = max(1, n * len(cols))
    completeness = 1 - sum(c["missing"] for c in cols) / total_cells
    parse_issues = sum(c.get("unparseable", 0) for c in cols)
    validity = 1 - parse_issues / total_cells
    uniqueness = 1 - dups / max(n, 1)
    variants = sum(1 for c in cols if c.get("case_or_space_variants", 0) > 0)
    consistency = 1 - variants / max(1, len(cols))
    base = 100 * (0.4 * completeness + 0.2 * validity + 0.2 * uniqueness + 0.2 * consistency)
    issues = []
    for c in cols:
        if c["missing_pct"] > 0:
            issues.append({"column": c["name"], "issue": "missing values", "detail": f"{c['missing_pct']}% missing",
                           "severity": "high" if c["missing_pct"] > 30 else "medium" if c["missing_pct"] > 5 else "low"})
        if c["kind"] == "numeric_as_text":
            issues.append({"column": c["name"], "issue": "numbers stored as text",
                           "detail": f"{c.get('unparseable', 0)} values not parseable", "severity": "medium"})
        if c["kind"] == "constant":
            issues.append({"column": c["name"], "issue": "constant column", "detail": "carries no information", "severity": "low"})
        if c["kind"] == "id":
            issues.append({"column": c["name"], "issue": "identifier", "detail": "unique per row; not a feature", "severity": "low"})
        if c.get("case_or_space_variants"):
            issues.append({"column": c["name"], "issue": "inconsistent labels",
                           "detail": f"{c['case_or_space_variants']} values differ only in case/spacing", "severity": "medium"})
        if c.get("outliers", 0) > 0:
            issues.append({"column": c["name"], "issue": "extreme outliers", "detail": f"{c['outliers']} beyond 3 IQR", "severity": "low"})
    if dups:
        issues.append({"column": "*", "issue": "duplicate rows", "detail": f"{dups} exact duplicates", "severity": "medium"})
    by_type: dict[str, int] = {}
    for i in issues:  # cap each issue type so e.g. many legitimate outliers cannot dominate the score
        by_type[i["issue"]] = by_type.get(i["issue"], 0) + {"high": 8, "medium": 4, "low": 1}[i["severity"]]
    penalty = sum(min(v, 12) for v in by_type.values())
    score = round(max(0.0, base - penalty), 1)
    return {"score": score, "completeness": round(completeness, 4), "validity": round(validity, 4),
            "uniqueness": round(uniqueness, 4), "consistency": round(consistency, 4), "issues": issues}
