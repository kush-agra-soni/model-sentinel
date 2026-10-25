import os
import json
import time
import httpx
import pandas as pd
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8005')}")
st.set_page_config(page_title="Tabular ML Platform", layout="wide")
ss = st.session_state


def api(method, path, **kw):
    return httpx.request(method, f"{BACKEND}{path}", timeout=kw.pop("timeout", 600), **kw)


def ok(r):
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text)
        return None
    return r.json()


try:
    health = api("GET", "/health", params={"check_llm": True}).json()
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND}")
    st.stop()

st.title("Tabular ML Platform")
st.caption(f"LLM planner: {'connected' if health.get('llm') else 'unavailable (rule-based planning only)'}  |  "
           f"{health['projects']} project(s)")
tabs = st.tabs(["Data", "Plan", "Train", "Predict", "Monitor", "Governance"])
datasets = api("GET", "/datasets").json()
ds_label = {f"{d['title']}  ({d['id']})": d for d in datasets}


def wait_job(jid: str, box) -> dict:
    while True:
        j = api("GET", f"/jobs/{jid}").json()
        box.code("\n".join(e["msg"] for e in j["log"][-25:]) or "starting...", language=None)
        if j["status"] != "running":
            return j
        time.sleep(0.8)


# ======================= Data =======================
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Dataset", list(ds_label), key="ds_pick")
    ds = ds_label[pick]
    c1.caption(f"{ds['description']}  Source: {ds['source']}")
    with c2:
        up = st.file_uploader("or upload a CSV", type=["csv"])
        if up and st.button("Add dataset"):
            r = ok(api("POST", "/datasets/upload", files={"file": (up.name, up.getvalue())}))
            if r:
                st.success(f"Added {r['rows']} rows")
                st.rerun()
    ss.dataset = ds
    if st.button("Profile dataset", type="primary"):
        with st.spinner("Profiling..."):
            ss.profile = ok(api("POST", "/profile", json={"dataset_id": ds["id"], "target": ds.get("target")}))
            ss.profile_ds = ds["id"]
    if ss.get("profile") and ss.get("profile_ds") == ds["id"]:
        p = ss.profile
        q = p["quality"]
        m = st.columns(6)
        m[0].metric("Quality score", f"{q['score']:.0f} / 100")
        m[1].metric("Rows", f"{p['rows']:,}")
        m[2].metric("Columns", p["columns"])
        m[3].metric("Completeness", f"{q['completeness']:.1%}")
        m[4].metric("Duplicate rows", p["duplicate_rows"])
        m[5].metric("Issues", len(q["issues"]))
        if q["issues"]:
            st.dataframe(pd.DataFrame(q["issues"]), hide_index=True, use_container_width=True)
        cols = pd.DataFrame([{k: c.get(k) for k in ("name", "kind", "missing_pct", "unique", "mean", "min", "max", "outliers")}
                             for c in p["column_profiles"]])
        st.dataframe(cols, hide_index=True, use_container_width=True)
        g1, g2 = st.columns(2)
        col = g1.selectbox("Distribution of", [c["name"] for c in p["column_profiles"] if c.get("hist") or c.get("top")])
        cp = next(c for c in p["column_profiles"] if c["name"] == col)
        if cp.get("hist"):
            e = cp["hist"]["edges"]
            g1.bar_chart(pd.DataFrame({"count": cp["hist"]["counts"]}, index=[f"{e[i]:.3g}" for i in range(len(e) - 1)]))
        else:
            g1.bar_chart(pd.Series(cp["top"], name="count"))
        if p.get("target_balance") and p["target_balance"]["type"] == "classes":
            g2.markdown("**Target balance**")
            g2.bar_chart(pd.Series(p["target_balance"]["share"], name="share"))
        if p["high_correlations"]:
            g2.markdown("**Highly correlated pairs** (|r| >= 0.85)")
            g2.dataframe(pd.DataFrame(p["high_correlations"]), hide_index=True)
        with st.expander("Preview rows"):
            pv = api("GET", f"/datasets/{ds['id']}/preview").json()
            st.dataframe(pd.DataFrame(pv["rows"], columns=pv["columns"]), use_container_width=True)

# ======================= Plan =======================
with tabs[1]:
    ds = ss.get("dataset", datasets[0])
    st.caption(f"Dataset: {ds['title']}. The rule planner proposes a cleaning plan; the LLM reviews it for leakage and "
               "data issues; every suggestion is validated before use. Edit anything before training.")
    pv = api("GET", f"/datasets/{ds['id']}/preview", params={"rows": 1}).json()
    c1, c2, c3, c4 = st.columns(4)
    tasks = ["(auto)", "classification", "regression", "clustering", "forecasting"]
    task = c1.selectbox("Task", tasks, index=tasks.index(ds["task"]) if ds.get("task") in tasks else 0)
    target = c2.selectbox("Target", ["(auto)"] + pv["columns"],
                          index=(pv["columns"].index(ds["target"]) + 1) if ds.get("target") in pv["columns"] else 0)
    tcol = c3.selectbox("Time column (forecasting)", ["(none)"] + pv["columns"],
                        index=(pv["columns"].index(ds["time_column"]) + 1) if ds.get("time_column") in pv["columns"] else 0)
    use_llm = c4.toggle("Ask the LLM to review", value=bool(health.get("llm")))
    if st.button("Build plan", type="primary"):
        with st.spinner("Planning (the LLM review can take up to a minute)..."):
            r = ok(api("POST", "/plan", json={"dataset_id": ds["id"], "use_llm": use_llm,
                                               "target": None if target == "(auto)" else target,
                                               "task": None if task == "(auto)" else task,
                                               "time_column": None if tcol == "(none)" else tcol}))
        if r:
            ss.plan_resp, ss.plan_ds = r, ds["id"]
    if ss.get("plan_resp") and ss.get("plan_ds") == ds["id"]:
        r = ss.plan_resp
        plan = r["plan"]
        st.markdown(f"**Task:** {plan['task']}  |  **Target:** {plan.get('target')}  |  **Positive class:** "
                    f"{plan.get('positive_class')}  |  **Plan source:** {plan.get('source')}")
        if r.get("llm_error"):
            st.warning(f"LLM review unavailable ({r['llm_error']}); using the rule-based plan.")
        if r["leakage"]:
            st.error("Possible target leakage (dropped): " + "; ".join(
                f"{x['column']} ({x['source']}: {x.get('reason') or x.get('metric', '') + ' ' + str(x.get('score', ''))})"
                for x in r["leakage"]))
        for n in plan.get("notes", []):
            st.info(n)
        rows = [{"column": c, "role": e["role"], "kind": e["kind"], "fixes": ", ".join(e["fixes"]), "reason": e["reason"]}
                for c, e in plan["columns"].items()]
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True, disabled=["column", "reason"],
                                column_config={"role": st.column_config.SelectboxColumn(options=["feature", "drop", "target", "time"]),
                                               "kind": st.column_config.SelectboxColumn(options=["numeric", "categorical", "datetime", "text"])},
                                key=f"plan_ed_{ds['id']}")
        for row in edited.itertuples():
            plan["columns"][row.column].update(role=row.role, kind=row.kind,
                                               fixes=[f.strip() for f in str(row.fixes).split(",") if f.strip()])
        if r.get("single_feature_scores"):
            with st.expander("Single-feature predictive power (leakage screen)"):
                st.dataframe(pd.DataFrame(r["single_feature_scores"]), hide_index=True)
        ss.plan = plan

# ======================= Train =======================
with tabs[2]:
    ds = ss.get("dataset", datasets[0])
    if not ss.get("plan") or ss.get("plan_ds") != ds["id"]:
        st.info("Build a plan for the selected dataset in the Plan tab first.")
    else:
        c1, c2, c3 = st.columns([2, 2, 2])
        project = c1.text_input("Project name", value=ds["id"].replace("_", "-")[:30])
        all_models = {"classification": ["logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm"],
                      "regression": ["ridge", "random_forest", "hist_gradient_boosting", "lightgbm"]}.get(ss.plan["task"], [])
        models = c2.multiselect("Candidate models", all_models, default=all_models) if all_models else None
        rng = c3.slider("Train on this share of rows (in file order)", 0.0, 1.0, (0.0, 1.0), 0.05,
                        help="For time-ordered data, train on early rows and monitor later ones.")
        if st.button("Train", type="primary"):
            jid = ok(api("POST", "/train", json={"project": project, "dataset_id": ds["id"], "plan": ss.plan,
                                                  "models": models or None, "rows_from": rng[0], "rows_to": rng[1]}))
            if jid:
                j = wait_job(jid["job_id"], st.empty())
                if j["status"] == "done":
                    ss.last_model = j["result"]["model_id"]
                    st.success(f"Trained {j['result']['model_id']} ({j['result']['best_model']})")
                else:
                    st.error(j["error"])
        mid = ss.get("last_model")
        if mid:
            meta = api("GET", f"/models/{mid}").json()
            rep = meta["report"]
            st.subheader(f"{mid}: {rep['best_model']}")
            st.markdown("**Leaderboard (cross-validation)**")
            st.dataframe(pd.DataFrame(rep["leaderboard"]), hide_index=True, use_container_width=True)
            if rep.get("baseline") is not None:
                st.markdown("**Holdout metrics vs a naive baseline**")
                st.dataframe(pd.DataFrame({"model": rep["metrics"], "baseline": rep["baseline"]}), use_container_width=True)
            g1, g2 = st.columns(2)
            if rep.get("importance"):
                g1.markdown("**Permutation importance**")
                g1.bar_chart(pd.DataFrame(rep["importance"]).set_index("feature")["importance"].head(12))
            if rep.get("confusion"):
                g2.markdown("**Confusion matrix (rows = actual)**")
                g2.dataframe(pd.DataFrame(rep["confusion"]["matrix"], index=rep["confusion"]["labels"],
                                          columns=rep["confusion"]["labels"]))
            if rep.get("calibration"):
                g2.markdown("**Calibration**")
                g2.line_chart(pd.DataFrame(rep["calibration"]).set_index("bin")[["predicted", "observed"]])
            if rep.get("segments"):
                st.markdown(f"**Segments (k = {rep['k']}, silhouette {rep['metrics']['silhouette']})**")
                st.dataframe(pd.DataFrame(rep["segments"]), hide_index=True, use_container_width=True)
            if rep.get("forecast"):
                hist = pd.DataFrame(rep["history"]).set_index("date")["value"].rename("history")
                fc = pd.DataFrame(rep["forecast"]).set_index("date")["value"].rename("forecast")
                st.line_chart(pd.concat([hist, fc], axis=1))
                st.caption(f"Holdout MAPE {rep['metrics']['mape']}% vs {rep['baseline']['mape']}% for the {rep['baseline_name']}")
            with st.expander("Cleaning log"):
                st.code("\n".join(rep.get("cleaning_log", [])), language=None)

# ======================= Predict =======================
projects = api("GET", "/projects").json()
proj_names = [p["project"] for p in projects]
with tabs[3]:
    if not projects:
        st.info("Train a model first.")
    else:
        pname = st.selectbox("Project", proj_names, key="pred_proj")
        proj = next(p for p in projects if p["project"] == pname)
        meta = api("GET", f"/models/{proj['champion']}").json()
        st.caption(f"Champion {proj['champion']} ({meta['task']}, target {meta['target']})")
        if meta["task"] == "forecasting":
            h = st.slider("Months ahead", 1, 36, 12)
            fc = api("GET", f"/projects/{pname}/forecast", params={"horizon": h}).json()
            st.line_chart(pd.DataFrame(fc).set_index("date"))
        else:
            with st.form("predict"):
                rec = {}
                cols = st.columns(3)
                for i, (c, spec) in enumerate(meta["schema"].items()):
                    box = cols[i % 3]
                    if spec["type"] == "numeric":
                        rec[c] = box.number_input(c, value=float(round(spec["median"], 4)))
                    else:
                        opts = spec["values"][:50]
                        rec[c] = box.selectbox(c, opts, index=opts.index(spec["mode"]) if spec["mode"] in opts else 0)
                go = st.form_submit_button("Predict", type="primary")
            if go:
                out = ok(api("POST", f"/projects/{pname}/predict", json={"records": [rec], "explain": True}))
                if out:
                    p = out["predictions"][0]
                    m = st.columns(3)
                    m[0].metric("Prediction", str(p.get("prediction", p.get("segment"))))
                    if "confidence" in p:
                        m[1].metric("Confidence", f"{p['confidence']:.1%}")
                    m[2].metric("Latency", f"{out['latency_ms']} ms")
                    for w in p["warnings"]:
                        st.warning(w)
                    if p.get("explanation"):
                        st.markdown("**Why: contribution of each feature vs. its typical value**")
                        st.bar_chart(pd.DataFrame(p["explanation"]).set_index("feature")["contribution"])
            st.markdown("**Batch scoring**")
            f = st.file_uploader("CSV with the same columns", type=["csv"], key="batch_csv")
            if f and st.button("Score file"):
                r = api("POST", f"/projects/{pname}/predict/csv", files={"file": (f.name, f.getvalue())})
                if r.status_code == 200:
                    st.success(f"Scored in {r.headers.get('x-latency-ms')} ms")
                    st.download_button("Download predictions", r.content, file_name="predictions.csv")
                else:
                    st.error(r.json()["detail"])

# ======================= Monitor =======================
with tabs[4]:
    if not projects:
        st.info("Train a model first.")
    else:
        pname = st.selectbox("Project", proj_names, key="mon_proj")
        proj = next(p for p in projects if p["project"] == pname)
        c1, c2 = st.columns(2)
        dsm = ds_label[c1.selectbox("Monitor data from", list(ds_label), key="mon_ds")]
        rng = c1.slider("Rows (share of the file, in order)", 0.0, 1.0, (0.7, 1.0), 0.05, key="mon_rng")
        sim = c2.slider("Add simulated drift (0 = none)", 0.0, 2.0, 0.0, 0.25)
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Check for drift", type="primary"):
            with st.spinner("Comparing with the training reference..."):
                ss.drift = ok(api("POST", f"/projects/{pname}/monitor", json={"dataset_id": dsm["id"], "rows_from": rng[0],
                                                                              "rows_to": rng[1], "simulate_drift": sim}))
        if b2.button("Retrain on these rows"):
            jid = ok(api("POST", f"/projects/{pname}/retrain", json={"dataset_id": dsm["id"], "rows_from": rng[0], "rows_to": rng[1]}))
            if jid:
                j = wait_job(jid["job_id"], st.empty())
                if j["status"] == "done":
                    res = j["result"]
                    (st.success if res["promoted"] else st.info)(("Promoted " if res["promoted"] else "Kept champion. ") + res["reason"])
                else:
                    st.error(j["error"])
        if b3.button("Roll back champion"):
            r = ok(api("POST", f"/projects/{pname}/rollback"))
            if r:
                st.success(f"Champion is now {r['champion']} ({r['seconds'] * 1000:.1f} ms)")
        if ss.get("drift"):
            d = ss.drift
            color = {"stable": st.success, "moderate": st.warning, "major": st.error}[d["status"]]
            color(f"Drift: {d['status']} | importance-weighted PSI {d['importance_weighted_psi']} | "
                  f"{d['features_major']} major, {d['features_moderate']} moderate features | recommendation: {d['recommendation']}")
            if d.get("prediction_drift"):
                pdft = d["prediction_drift"]
                st.caption(f"Prediction drift PSI {pdft['psi']} (mean score {pdft['ref_mean']} -> {pdft['cur_mean']})")
            if d.get("performance"):
                pf = d["performance"]
                st.caption(f"Performance on labelled rows: {pf['metric']} {pf['current']} vs {pf['reference']} at training "
                           f"(floor {pf['floor']}) -> {pf['action']}")
            st.dataframe(pd.DataFrame(d["features"]), hide_index=True, use_container_width=True)
        hist = api("GET", f"/projects/{pname}/monitoring").json()
        if hist:
            st.markdown("**Monitoring history**")
            h = pd.DataFrame(hist)
            h["ts"] = pd.to_datetime(h["ts"], unit="s").dt.strftime("%m-%d %H:%M:%S")
            st.dataframe(h, hide_index=True, use_container_width=True)
        st.markdown("**Model versions**")
        rows = []
        for v in proj["versions"]:
            m = api("GET", f"/models/{v}").json()
            rows.append({"model_id": v, "champion": v == proj["champion"], "algorithm": m["report"]["best_model"],
                         **{k: m["report"].get("metrics", {}).get(k) for k in ("roc_auc", "rmse", "silhouette", "mape")}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        pick = st.selectbox("Promote a version", [r["model_id"] for r in rows])
        if st.button("Promote"):
            ok(api("POST", f"/projects/{pname}/promote", json={"model_id": pick}))
            st.rerun()

# ======================= Governance =======================
with tabs[5]:
    if projects:
        pname = st.selectbox("Project", proj_names, key="gov_proj")
        proj = next(p for p in projects if p["project"] == pname)
        mid = st.selectbox("Model", proj["versions"][::-1])
        card = api("GET", f"/models/{mid}/card").text
        st.markdown(card)
        st.download_button("Download model card", card, file_name=f"{mid}.md")
        st.markdown("**Promotion history**")
        h = pd.DataFrame(proj["history"])
        if not h.empty:
            h["ts"] = pd.to_datetime(h["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(h, hide_index=True, use_container_width=True)
    st.markdown("**Prediction audit log**")
    audit = api("GET", "/audit").json()
    if audit:
        a = pd.DataFrame(audit)
        a["ts"] = pd.to_datetime(a["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(a, hide_index=True, use_container_width=True)
