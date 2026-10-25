import io
import pandas as pd
from conftest import churn_frame, seasonal_series, upload_frame, wait_job


def build_and_train(c, ds_id, project="churn", train_to=1.0, **plan_kw):
    r = c.post("/plan", json={"dataset_id": ds_id, **plan_kw}).json()
    jid = c.post("/train", json={"project": project, "dataset_id": ds_id, "plan": r["plan"], "rows_to": train_to,
                                 "models": ["logistic_regression"] if r["plan"]["task"] == "classification" else None}).json()
    j = wait_job(c, jid["job_id"])
    assert j["status"] == "done", j.get("error")
    return r, j


def test_datasets_upload_preview_profile(client):
    ids = {d["id"] for d in client.get("/datasets").json()}
    assert {"telco_churn", "bank_marketing", "breast_cancer", "customer_segments"} <= ids
    uid = upload_frame(client, churn_frame(400))
    assert any(d["id"] == uid for d in client.get("/datasets").json())
    pv = client.get(f"/datasets/{uid}/preview", params={"rows": 5}).json()
    assert pv["shape"] == [400, 8] and len(pv["rows"]) == 5
    prof = client.post("/profile", json={"dataset_id": uid, "target": "churn"}).json()
    assert prof["quality"]["score"] < 100 and prof["rows"] == 400
    assert client.post("/datasets/upload", files={"file": ("x.csv", b"onlyonecolumn\n1\n")}).status_code == 422
    assert client.get("/datasets/nope/preview").status_code == 404


def test_plan_combines_llm_and_statistical_leakage(client):
    df = churn_frame(800)
    df["churn_flag_copy"] = (df["churn"] == "Yes").astype(int)
    uid = upload_frame(client, df)
    r = client.post("/plan", json={"dataset_id": uid, "target": "churn"}).json()
    by = {x["column"]: x["source"] for x in r["leakage"]}
    assert by == {"cancellation_fee_charged": "llm", "churn_flag_copy": "statistics"}
    assert r["plan"]["columns"]["customer_id"]["role"] == "drop" and r["plan"]["source"] == "llm"
    assert r["single_feature_scores"][0]["score"] >= 0.98
    client.chat.fail = True
    r2 = client.post("/plan", json={"dataset_id": uid, "target": "churn"}).json()
    assert r2["plan"]["source"] == "rules" and "down" in r2["llm_error"]


def test_train_predict_audit_and_card(client):
    uid = upload_frame(client, churn_frame(1500))
    r, j = build_and_train(client, uid, target="churn")
    mid = j["result"]["model_id"]
    proj = client.get("/projects").json()[0]
    assert proj["project"] == "churn" and proj["champion"] == mid and proj["metrics"]["roc_auc"] > 0.7
    rec = churn_frame(5, seed=9).drop(columns=["churn"]).to_dict("records")
    out = client.post("/projects/churn/predict", json={"records": rec[:1], "explain": True}).json()
    assert out["predictions"][0]["explanation"] and out["model_id"] == mid
    batch = client.post("/projects/churn/predict", json={"records": rec}).json()
    assert len(batch["predictions"]) == 5
    csv = client.post("/projects/churn/predict/csv",
                      files={"file": ("b.csv", churn_frame(50, seed=4).drop(columns=["churn"]).to_csv(index=False).encode())})
    res = pd.read_csv(io.StringIO(csv.text))
    assert {"prediction", "score", "warnings"} <= set(res.columns) and len(res) == 50
    kinds = [a["kind"] for a in client.get("/audit", params={"project": "churn"}).json()]
    assert kinds == ["csv", "batch", "online"]
    card = client.get(f"/models/{mid}/card").text
    assert "# Model card" in card and "leakage" in card and "| roc_auc |" in card
    assert client.post("/projects/nope/predict", json={"records": rec[:1]}).status_code == 404


def test_monitor_retrain_promote_rollback(client):
    uid = upload_frame(client, pd.concat([churn_frame(1500), churn_frame(1500, seed=5, shift=1.5)], ignore_index=True))
    build_and_train(client, uid, target="churn", train_to=0.5)
    first = client.get("/projects").json()[0]["champion"]
    stable = client.post("/projects/churn/monitor", json={"dataset_id": uid, "rows_from": 0.0, "rows_to": 0.4}).json()
    drifted = client.post("/projects/churn/monitor", json={"dataset_id": uid, "rows_from": 0.5, "rows_to": 1.0}).json()
    assert stable["status"] == "stable" and drifted["status"] == "major" and drifted["recommendation"] == "retrain"
    sim = client.post("/projects/churn/monitor", json={"dataset_id": uid, "rows_from": 0.0, "rows_to": 0.4,
                                                      "simulate_drift": 1.5}).json()
    assert sim["status"] == "major"
    assert [h["status"] for h in client.get("/projects/churn/monitoring").json()] == ["stable", "major", "major"]
    assert client.get(f"/monitoring/{drifted['id']}").json()["status"] == "major"
    jid = client.post("/projects/churn/retrain", params={"min_gain": 0.0},
                      json={"dataset_id": uid, "rows_from": 0.5, "rows_to": 1.0}).json()["job_id"]
    j = wait_job(client, jid)
    assert j["status"] == "done" and "challenger" in j["result"]
    if j["result"]["promoted"]:
        rb = client.post("/projects/churn/rollback").json()
        assert rb["champion"] == first and rb["seconds"] < 1
    client.post("/projects/churn/promote", json={"model_id": first})
    assert client.post("/projects/churn/promote", json={"model_id": "x"}).status_code == 404
    assert client.delete("/projects/churn").status_code == 200 and client.get("/projects").json() == []


def test_forecast_and_clustering_projects(client):
    uid = upload_frame(client, seasonal_series())
    build_and_train(client, uid, project="sales", target="sales", task="forecasting", time_column="date")
    fc = client.get("/projects/sales/forecast", params={"horizon": 6}).json()
    assert len(fc) == 6 and all(v["value"] > 0 for v in fc)
    assert client.post("/projects/sales/predict", json={"records": [{"x": 1}]}).status_code == 422
    assert client.get("/projects/sales/forecast", params={"horizon": 99}).status_code == 422
    build_and_train(client, "customer_segments", project="segments")
    seg = client.post("/projects/segments/predict", json={"records": [{"annual_income_k": 120, "spending_score": 80,
                                                                       "visits_per_month": 3, "avg_basket": 140, "age": 55}]}).json()
    assert "segment" in seg["predictions"][0]


def test_validation_errors(client):
    assert client.post("/train", json={"project": "Bad Name!", "dataset_id": "wine", "plan": {}}).status_code == 422
    assert client.post("/train", json={"project": "ok", "dataset_id": "wine", "plan": {"task": "magic", "columns": {}}}).status_code == 422
    assert client.post("/projects/ok/monitor", json={"dataset_id": "wine"}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404 and client.get("/models/nope").status_code == 404
