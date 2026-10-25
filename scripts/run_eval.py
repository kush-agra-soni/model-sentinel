"""End-to-end evaluation on every built-in dataset (LLM planning when available) plus the bank drift scenario.

Usage: python scripts/run_eval.py [--no-llm]"""
import sys
import json
import time
import argparse
import tempfile
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from sentinel.api import create_app  # noqa: E402
from sentinel.config import Settings, get_settings  # noqa: E402


def wait(c, jid):
    while (j := c.get(f"/jobs/{jid}").json())["status"] == "running":
        time.sleep(0.3)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s = get_settings()
    data = Path(tempfile.mkdtemp())
    (data / "datasets").mkdir()
    for f in (ROOT / "data" / "datasets").glob("*"):
        (data / "datasets" / f.name).write_bytes(f.read_bytes())
    c = TestClient(create_app(Settings(s.llm_base_url, s.llm_api_key, s.llm_model, data), llm_enabled=not a.no_llm))
    results = {}
    for ds, kw, rows_to in [("telco_churn", {}, 1.0), ("bank_marketing", {}, 0.4), ("titanic", {}, 1.0),
                            ("breast_cancer", {}, 1.0), ("diabetes", {}, 1.0), ("wine", {}, 1.0),
                            ("customer_segments", {}, 1.0), ("airline_passengers", {}, 1.0)]:
        if a.only and ds not in a.only:
            continue
        t = time.time()
        r = c.post("/plan", json={"dataset_id": ds, "use_llm": not a.no_llm, **kw}).json()
        plan_s = round(time.time() - t, 1)
        t = time.time()
        j = wait(c, c.post("/train", json={"project": ds.replace("_", "-"), "dataset_id": ds, "plan": r["plan"],
                                           "rows_to": rows_to}).json()["job_id"])
        if j["status"] != "done":
            print(ds, "FAILED", j["error"])
            continue
        meta = c.get(f"/models/{j['result']['model_id']}").json()
        rep = meta["report"]
        results[ds] = {"task": meta["task"], "plan_source": r["plan"]["source"], "plan_seconds": plan_s,
                       "llm_error": r["llm_error"], "leakage": [(x["column"], x["source"]) for x in r["leakage"]],
                       "dropped": [k for k, v in r["plan"]["columns"].items() if v["role"] == "drop"],
                       "best_model": rep["best_model"], "metrics": rep.get("metrics"), "baseline": rep.get("baseline"),
                       "train_seconds": round(time.time() - t, 1), "quality": r["profile_quality"]}
        print(ds, json.dumps(results[ds]), flush=True)

    if a.only:
        (ROOT / "reports" / "eval_partial.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        return
    # serving latency on the telco champion
    rec = c.get("/datasets/telco_churn/preview", params={"rows": 1}).json()
    one = [dict(zip(rec["columns"], rec["rows"][0]))]
    lat = []
    for _ in range(100):
        t = time.perf_counter()
        c.post("/projects/telco-churn/predict", json={"records": one})
        lat.append((time.perf_counter() - t) * 1000)
    results["serving"] = {"online_p50_ms": round(float(np.percentile(lat, 50)), 1),
                          "online_p95_ms": round(float(np.percentile(lat, 95)), 1)}
    print("serving", results["serving"])

    # bank drift scenario: trained on the first 40% (time order); monitor later periods; retrain with gate
    drift = []
    for a_, b_ in [(0.0, 0.4), (0.4, 0.7), (0.7, 1.0)]:
        rep = c.post("/projects/bank-marketing/monitor", json={"dataset_id": "bank_marketing", "rows_from": a_, "rows_to": b_}).json()
        drift.append({"rows": f"{a_:.0%}-{b_:.0%}", "status": rep["status"], "weighted_psi": rep["importance_weighted_psi"],
                      "prediction_psi": rep["prediction_drift"]["psi"], "auc": rep["performance"]["current"],
                      "recommendation": rep["recommendation"]})
    rt = wait(c, c.post("/projects/bank-marketing/retrain", json={"dataset_id": "bank_marketing", "rows_from": 0.4, "rows_to": 0.7}).json()["job_id"])["result"]
    after = c.post("/projects/bank-marketing/monitor", json={"dataset_id": "bank_marketing", "rows_from": 0.7, "rows_to": 1.0}).json()
    t = time.perf_counter()
    rb = c.post("/projects/bank-marketing/rollback").json()
    results["bank_drift"] = {"monitoring": drift, "retrain": {k: rt[k] for k in ("promoted", "champion_score", "challenger_score", "gain")},
                             "auc_on_last_30pct_after_retrain": after["performance"]["current"],
                             "rollback_ms": round((time.perf_counter() - t) * 1000, 1), "rolled_back_to": rb.get("champion")}
    print("bank_drift", json.dumps(results["bank_drift"]))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eval.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
