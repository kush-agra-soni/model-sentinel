"""Local model registry: versioned artifacts, a champion per project, promotion history and rollback."""
import json
import time
import uuid
import joblib
import shutil
import threading
import numpy as np
import pandas as pd
from pathlib import Path


def input_schema(reference: pd.DataFrame, num: list[str], cat: list[str]) -> dict:
    schema = {}
    for c in num:
        x = pd.to_numeric(reference[c], errors="coerce").dropna()
        schema[c] = {"type": "numeric", "mean": float(x.mean()) if len(x) else 0.0, "std": float(x.std()) if len(x) > 1 else 0.0,
                     "p01": float(x.quantile(0.01)) if len(x) else 0.0, "p99": float(x.quantile(0.99)) if len(x) else 0.0,
                     "median": float(x.median()) if len(x) else 0.0}
    for c in cat:
        vc = reference[c].astype(str).value_counts()
        schema[c] = {"type": "categorical", "values": vc.index[:200].tolist(), "mode": str(vc.index[0]) if len(vc) else ""}
    return schema


class Registry:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, object] = {}

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _index(self) -> dict:
        p = self._index_path()
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"projects": {}}

    def _save_index(self, idx: dict) -> None:
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        tmp.replace(self._index_path())

    def register(self, project: str, result: dict, plan: dict, dataset: dict, promote_if_first: bool = True) -> dict:
        with self._lock:
            idx = self._index()
            proj = idx["projects"].setdefault(project, {"versions": [], "champion": None, "history": []})
            version = len(proj["versions"]) + 1
            mid = f"{project}-v{version}-{uuid.uuid4().hex[:6]}"
            d = self.root / mid
            d.mkdir(parents=True)
            joblib.dump(result["pipeline"], d / "model.joblib")
            result["reference"].to_pickle(d / "reference.pkl")
            if result.get("reference_scores") is not None:
                np.save(d / "reference_scores.npy", result["reference_scores"])
            feats = result["report"]["features"]
            meta = {"model_id": mid, "project": project, "version": version, "created": time.time(),
                    "task": plan["task"], "target": plan.get("target"), "positive_class": plan.get("positive_class"),
                    "plan": plan, "dataset": dataset, "report": result["report"],
                    "schema": input_schema(result["reference"], feats["numeric"], feats["categorical"])
                    if plan["task"] != "forecasting" else {}}
            if plan["task"] == "forecasting":
                result["series"].to_pickle(d / "series.pkl")
                meta["use_log"] = result["use_log"]
            (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            proj["versions"].append(mid)
            if promote_if_first and proj["champion"] is None:
                proj["champion"] = mid
                proj["history"].append({"ts": time.time(), "action": "promote", "model_id": mid, "reason": "first model"})
            self._save_index(idx)
            return meta

    def projects(self) -> dict:
        return self._index()["projects"]

    def meta(self, mid: str) -> dict:
        p = self.root / mid / "meta.json"
        if not p.exists():
            raise KeyError(mid)
        return json.loads(p.read_text(encoding="utf-8"))

    def load(self, mid: str):
        with self._lock:
            if mid not in self._cache:
                if not (self.root / mid / "model.joblib").exists():
                    raise KeyError(mid)
                self._cache[mid] = joblib.load(self.root / mid / "model.joblib")
            return self._cache[mid]

    def reference(self, mid: str) -> pd.DataFrame:
        return pd.read_pickle(self.root / mid / "reference.pkl")

    def reference_scores(self, mid: str):
        p = self.root / mid / "reference_scores.npy"
        return np.load(p) if p.exists() else None

    def series(self, mid: str) -> pd.Series:
        return pd.read_pickle(self.root / mid / "series.pkl")

    def champion(self, project: str) -> str | None:
        return self.projects().get(project, {}).get("champion")

    def promote(self, project: str, mid: str, reason: str) -> None:
        with self._lock:
            idx = self._index()
            proj = idx["projects"].get(project)
            if not proj or mid not in proj["versions"]:
                raise KeyError(mid)
            proj["champion"] = mid
            proj["history"].append({"ts": time.time(), "action": "promote", "model_id": mid, "reason": reason})
            self._save_index(idx)

    def rollback(self, project: str) -> str:
        with self._lock:
            idx = self._index()
            proj = idx["projects"].get(project)
            promos = [h["model_id"] for h in (proj or {}).get("history", []) if h["action"] in ("promote", "rollback")]
            prev = next((m for m in reversed(promos[:-1]) if m != proj["champion"]), None) if proj else None
            if not prev:
                raise ValueError("no earlier champion to roll back to")
            t0 = time.perf_counter()
            proj["champion"] = prev
            proj["history"].append({"ts": time.time(), "action": "rollback", "model_id": prev,
                                    "reason": f"rolled back in {(time.perf_counter() - t0) * 1000:.1f} ms"})
            self._save_index(idx)
            return prev

    def delete_project(self, project: str) -> None:
        with self._lock:
            idx = self._index()
            proj = idx["projects"].pop(project, None)
            if proj:
                for mid in proj["versions"]:
                    shutil.rmtree(self.root / mid, ignore_errors=True)
                    self._cache.pop(mid, None)
            self._save_index(idx)
