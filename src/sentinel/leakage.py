"""Statistical leakage screen: a feature that predicts the target almost perfectly on its own is suspicious."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


def single_feature_scores(df: pd.DataFrame, target: str, task: str, max_rows: int = 5000) -> list[dict]:
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    y = df[target]
    out = []
    for c in df.columns:
        if c == target:
            continue
        x = df[c]
        if pd.api.types.is_numeric_dtype(x):
            X = x.fillna(x.median() if x.notna().any() else 0).to_numpy().reshape(-1, 1)
        else:
            codes = x.astype(str).astype("category").cat.codes
            X = codes.to_numpy().reshape(-1, 1)
        try:
            if task == "classification":
                yy = y.astype(str)
                if yy.nunique() < 2 or yy.value_counts().min() < 3:
                    continue
                cv = StratifiedKFold(3, shuffle=True, random_state=0)
                proba = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=0), X, yy, cv=cv,
                                          method="predict_proba")
                classes = np.unique(yy)
                score = roc_auc_score(yy, proba[:, 1]) if len(classes) == 2 else roc_auc_score(
                    yy, proba, multi_class="ovr")
                metric = "AUC"
            else:
                yy = pd.to_numeric(y, errors="coerce").fillna(0)
                pred = cross_val_predict(DecisionTreeRegressor(max_depth=4, random_state=0), X, yy,
                                         cv=KFold(3, shuffle=True, random_state=0))
                score, metric = r2_score(yy, pred), "R2"
        except ValueError:
            continue
        out.append({"column": c, "metric": metric, "score": round(float(score), 4)})
    return sorted(out, key=lambda r: -r["score"])


def leakage_suspects(scores: list[dict], threshold_auc: float = 0.98, threshold_r2: float = 0.97) -> list[dict]:
    return [s for s in scores if (s["metric"] == "AUC" and s["score"] >= threshold_auc) or
            (s["metric"] == "R2" and s["score"] >= threshold_r2)]
