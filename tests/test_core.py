import json
import time
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from sentinel.llm import LLMError
from sentinel.registry import Registry
from sentinel.serving import score_records
from sentinel.profile import infer_kind, profile
from sentinel.datasets import customer_segments, load
from conftest import FakeChat, churn_frame, seasonal_series
from sentinel.pipeline import monitor_batch, retrain, train_project
from sentinel.leakage import leakage_suspects, single_feature_scores
from sentinel.drift import drift_report, performance_check, psi, simulate_drift
from sentinel.training import train_clustering, train_forecast, train_supervised
from sentinel.planner import PlanError, apply_plan, llm_plan, parse_json_lenient, rule_plan, validate_plan


def plan_for(df, target, **kw):
    return validate_plan(rule_plan(profile(df, target), target, **kw), list(df.columns))


# ---------- profiling ----------
def test_kind_inference():
    df = churn_frame(300)
    kinds = {c: infer_kind(df[c], len(df)) for c in df}
    assert kinds["customer_id"] == "id" and kinds["total_charges"] == "numeric_as_text"
    assert kinds["tenure"] == "numeric" and kinds["contract"] == "categorical"
    assert infer_kind(pd.Series(["2024-01-05", "2024-02-06"] * 20), 40) == "datetime"
    assert infer_kind(pd.Series([1] * 50), 50) == "constant"


def test_profile_quality_issues_and_scoring():
    p = profile(churn_frame(500), "churn")
    issues = {(i["column"], i["issue"]) for i in p["quality"]["issues"]}
    assert ("total_charges", "numbers stored as text") in issues and ("customer_id", "identifier") in issues
    assert ("contract", "inconsistent labels") in issues
    assert 0 < p["quality"]["score"] < 100 and p["target_balance"]["type"] == "classes"
    many_outliers = pd.DataFrame({f"x{i}": np.r_[np.zeros(98), [1e6, -1e6]] + np.arange(100) for i in range(30)})
    assert profile(many_outliers)["quality"]["score"] >= 85  # per-type penalty cap


# ---------- planning & cleaning ----------
def test_rule_plan_decisions():
    plan = plan_for(churn_frame(500), "churn")
    cols = plan["columns"]
    assert plan["task"] == "classification" and plan["positive_class"] == "Yes"
    assert cols["customer_id"]["role"] == "drop" and cols["total_charges"]["fixes"] == ["to_numeric"]
    assert cols["contract"]["fixes"] == ["strip", "lower"]
    assert plan_for(customer_segments(), None)["task"] == "clustering"


def test_validate_plan_rejects_bad_plans():
    df = churn_frame(200)
    good = plan_for(df, "churn")
    for mutate, msg in [(lambda p: p.update(task="magic"), "task"), (lambda p: p.update(target="nope"), "target"),
                        (lambda p: p["columns"]["tenure"].update(fixes=["rm -rf"]), "unsupported"),
                        (lambda p: p["columns"]["tenure"].update(role="boss"), "invalid")]:
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(PlanError, match=msg):
            validate_plan(bad, list(df.columns))
    fc = plan_for(seasonal_series(), "sales", task="regression")
    fc["task"] = "forecasting"
    with pytest.raises(PlanError, match="time column"):
        validate_plan(fc, ["date", "sales"])


def test_apply_plan_fixes_and_inference_mode():
    df = churn_frame(500)
    plan = plan_for(df, "churn")
    plan["columns"]["monthly_charges"]["fixes"] = ["clip_outliers"]
    clean, log = apply_plan(df, plan)
    assert "customer_id" not in clean and clean["total_charges"].dtype.kind == "f"
    assert set(clean["contract"].dropna()) == {"month-to-month", "one year", "two year"}
    assert any("unparseable" in line for line in log) and "clip:monthly_charges" in plan["fitted"]
    lo, hi = plan["fitted"]["clip:monthly_charges"]
    extreme = df.head(3).copy()
    extreme["monthly_charges"] = [1e9, -1e9, 50]
    inf, _ = apply_plan(extreme, plan, inference=True)
    assert inf["monthly_charges"].tolist()[:2] == [hi, lo] and len(inf) == 3
    dated = pd.DataFrame({"signup": ["2024-01-05", "2024-06-30"] * 20, "x": range(40), "y": [0, 1] * 20})
    p2 = plan_for(dated, "y")
    out, _ = apply_plan(dated, p2)
    assert {"signup_year", "signup_month", "signup_weekday"} <= set(out.columns)


def test_categorical_ints_become_strings_consistently():
    df = pd.DataFrame({"senior": [0, 1] * 50, "x": np.arange(100.0), "y": ["a", "b", "b", "a"] * 25})
    plan = plan_for(df, "y")
    assert plan["columns"]["senior"]["kind"] == "categorical"
    clean, _ = apply_plan(df, plan)
    inf, _ = apply_plan(df.drop(columns="y").astype({"senior": float}), plan, inference=True)
    assert set(clean["senior"]) == set(inf["senior"]) == {"0", "1"}


def test_llm_plan_merges_validates_and_flags_leakage():
    df = churn_frame(300)
    prof = profile(df, "churn")
    draft = plan_for(df, "churn")
    plan = llm_plan(draft, prof, "churn data", FakeChat())
    assert plan["source"] == "llm" and plan["columns"]["cancellation_fee_charged"]["role"] == "drop"
    assert plan["columns"]["cancellation_fee_charged"]["reason"].startswith("leakage")
    assert "not_a_column" not in plan["columns"] and plan["notes"] == ["checked by fake planner"]
    with pytest.raises(LLMError):
        llm_plan(draft, prof, "", FakeChat(fail=True))


def test_parse_json_lenient_repairs_truncation():
    assert parse_json_lenient('{"leakage": [{"column": "d", "reason": "x"}], "columns": {"a": {"role": "fe')["leakage"][0]["column"] == "d"
    assert parse_json_lenient('<think>hmm</think>```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json_lenient("no braces at all")


def test_leakage_screen_finds_planted_leak():
    df = churn_frame(800)
    clean, _ = apply_plan(df, plan_for(df, "churn"))
    sus = leakage_suspects(single_feature_scores(clean, "churn", "classification"))
    assert [s["column"] for s in sus] == ["cancellation_fee_charged"]


# ---------- training ----------
def test_classification_beats_baseline_on_breast_cancer():
    df = load("breast_cancer", Path(tempfile.mkdtemp()))
    plan = plan_for(df, "malignant")
    res = train_supervised(*apply_plan(df, plan)[:1], plan, models=["logistic_regression", "hist_gradient_boosting"])
    r = res["report"]
    assert r["metrics"]["roc_auc"] > 0.97 > r["baseline"]["roc_auc"]
    assert r["confusion"]["labels"] and r["calibration"] and r["importance"]
    assert len(r["leaderboard"]) == 2 and r["leaderboard"][0]["cv_score"] >= r["leaderboard"][1]["cv_score"]


def test_regression_and_multiclass():
    tmp = Path(tempfile.mkdtemp())
    d = load("diabetes", tmp)
    plan = plan_for(d, "progression")
    r = train_supervised(apply_plan(d, plan)[0], plan, models=["ridge"])["report"]
    assert plan["task"] == "regression" and r["metrics"]["rmse"] < r["baseline"]["rmse"]
    w = load("wine", tmp)
    plan = plan_for(w, "cultivar")
    r = train_supervised(apply_plan(w, plan)[0], plan, models=["logistic_regression"])["report"]
    assert r["metrics"]["accuracy"] > 0.9 and len(r["confusion"]["labels"]) == 3


def test_clustering_finds_segments():
    df = customer_segments()
    plan = plan_for(df, None)
    r = train_clustering(apply_plan(df, plan)[0], plan)["report"]
    assert r["k"] in (3, 4) and r["metrics"]["silhouette"] > 0.4 and sum(s["size"] for s in r["segments"]) == len(df)


def test_forecasting_beats_seasonal_naive():
    df = seasonal_series()
    plan = plan_for(df, "sales", task="forecasting", time_column="date")
    res = train_forecast(apply_plan(df, plan)[0], plan)
    r = res["report"]
    assert r["metrics"]["mape"] < r["baseline"]["mape"] and len(r["forecast"]) == 12


# ---------- registry & serving ----------
@pytest.fixture
def trained(tmp_path):
    reg = Registry(tmp_path / "models")
    df = churn_frame(1500)
    plan = plan_for(df, "churn")
    plan["columns"]["cancellation_fee_charged"]["role"] = "drop"
    meta = train_project(reg, "churn", df, plan, {"id": "synthetic"}, lambda m: None, models=["logistic_regression"])
    return reg, meta, df


def test_registry_champion_promote_rollback(trained):
    reg, meta, df = trained
    assert reg.champion("churn") == meta["model_id"] and meta["schema"]["contract"]["type"] == "categorical"
    m2 = train_project(reg, "churn", df, reg.meta(meta["model_id"])["plan"], {}, lambda m: None, models=["random_forest"])
    assert reg.champion("churn") == meta["model_id"]  # not auto-promoted
    reg.promote("churn", m2["model_id"], "test")
    assert reg.champion("churn") == m2["model_id"]
    assert reg.rollback("churn") == meta["model_id"] and reg.champion("churn") == meta["model_id"]
    with pytest.raises(KeyError):
        reg.promote("churn", "missing", "x")


def test_serving_validation_explanations_and_latency(trained):
    reg, meta, df = trained
    model = reg.load(meta["model_id"])
    rec = df.drop(columns=["churn"]).iloc[0].to_dict()
    rec.update(tenure=5000, contract=" Lifetime ", monthly_charges="lots")
    del rec["support_calls"]
    p = score_records(model, meta, [rec], with_explanations=True)["predictions"][0]
    w = " | ".join(p["warnings"])
    assert "tenure: 5000 is far outside" in w and "unseen category 'lifetime'" in w
    assert "support_calls: missing" in w and "monthly_charges: 'lots' is not a number" in w
    assert p["prediction"] in ("Yes", "No") and 0 <= p["score"] <= 1 and p["explanation"][0]["feature"] == "tenure"
    clean_row = df.drop(columns=["churn"]).iloc[[3]]
    t = time.perf_counter()
    for _ in range(50):
        score_records(model, meta, clean_row)
    assert (time.perf_counter() - t) / 50 < 0.05
    assert score_records(model, meta, df.drop(columns=["churn"]).iloc[3:4])["predictions"][0]["warnings"] == []


# ---------- drift & retraining ----------
def test_psi_and_drift_report():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.02 and psi(a, rng.normal(1, 1, 5000)) > 0.25
    ref = pd.DataFrame({"x": a, "c": rng.choice(["a", "b"], 5000)})
    schema = {"x": {"type": "numeric", "std": 1.0, "values": []}, "c": {"type": "categorical", "values": ["a", "b"]}}
    assert drift_report(ref, ref.sample(1000, random_state=1), schema)["status"] == "stable"
    rep = drift_report(ref, simulate_drift(ref.sample(1000, random_state=1), schema, 1.0), schema)
    assert rep["status"] == "major" and rep["features"][0]["level"] == "major"
    pc = performance_check("roc_auc", 0.6, 0.8, floor=0.65)
    assert pc["below_floor"] and pc["action"] == "retrain"


def test_monitoring_and_gated_retraining(trained):
    reg, meta, df = trained
    same = monitor_batch(reg, "churn", churn_frame(800, seed=1))
    assert same["status"] == "stable" and same["recommendation"] == "none" and same["performance"]["current"] > 0.7
    shifted = churn_frame(1500, seed=2, shift=1.5)
    drifted = monitor_batch(reg, "churn", shifted)
    assert drifted["status"] == "major" and drifted["recommendation"] == "retrain"
    out = retrain(reg, "churn", shifted, lambda m: None, min_gain=0.0)
    assert out["challenger"] != meta["model_id"] and isinstance(out["promoted"], bool)
    blocked = retrain(reg, "churn", churn_frame(800, seed=3), lambda m: None, min_gain=0.5)
    assert blocked["promoted"] is False and reg.champion("churn") != blocked["challenger"]


def test_missing_categoricals_train_cleanly():
    df = churn_frame(600)
    df.loc[::9, "contract"] = None
    df.loc[::11, "contract"] = "   "
    plan = plan_for(df, "churn")
    clean, _ = apply_plan(df, plan)
    assert not any(v is pd.NA for v in clean["contract"])
    assert train_supervised(clean, plan, models=["logistic_regression"])["report"]["metrics"]["roc_auc"] > 0.6


def test_llm_plan_ignores_invalid_values_without_failing():
    class Bad:
        def __call__(self, msgs):
            return json.dumps({"columns": {"customer_id": {"role": "identifier", "kind": "string", "fixes": ["strip", "evil"]},
                                           "tenure": {"role": "drop", "reason": "test"}}})
    df = churn_frame(300)
    plan = llm_plan(plan_for(df, "churn"), profile(df, "churn"), "", Bad())
    assert plan["columns"]["customer_id"]["role"] == "drop" and plan["columns"]["tenure"]["role"] == "drop"
    assert any("Ignored invalid" in n for n in plan["notes"])
