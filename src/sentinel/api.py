"""Tabular ML Platform API."""
import io
import re
import time
from . import llm
import pandas as pd
from .store import Store
from pathlib import Path
from .profile import profile
from .registry import Registry
from .drift import simulate_drift
from collections.abc import Callable
from pydantic import BaseModel, Field
from .config import Settings, get_settings
from .serving import InputError, score_records
from .datasets import CATALOG, DatasetError, load
from fastapi.responses import PlainTextResponse, Response
from .leakage import leakage_suspects, single_feature_scores
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from .pipeline import forecast, monitor_batch, retrain, train_project
from .planner import PlanError, apply_plan, llm_plan, rule_plan, validate_plan

PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class PlanRequest(BaseModel):
    dataset_id: str
    target: str | None = None
    task: str | None = None
    time_column: str | None = None
    use_llm: bool = True


class TrainRequest(BaseModel):
    project: str = Field(pattern=PROJECT_RE.pattern)
    dataset_id: str
    plan: dict
    models: list[str] | None = None
    rows_from: float = Field(0.0, ge=0, le=1)
    rows_to: float = Field(1.0, ge=0, le=1)


class PredictRequest(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=10000)
    explain: bool = False


class SliceRequest(BaseModel):
    dataset_id: str
    rows_from: float = Field(0.0, ge=0, le=1)
    rows_to: float = Field(1.0, ge=0, le=1)
    simulate_drift: float = Field(0.0, ge=0, le=3)


class PromoteRequest(BaseModel):
    model_id: str


def create_app(settings: Settings | None = None, chat: Callable | None = None, llm_enabled: bool = True) -> FastAPI:
    settings = settings or get_settings()
    reg = Registry(settings.models_dir)
    store = Store(settings.db_path)
    uploads_dir = settings.data_dir / "uploads"
    if llm_enabled and chat is None:
        chat = lambda msgs: llm.complete(settings, msgs, max_tokens=3000)  # noqa: E731
    app = FastAPI(title="Tabular ML Platform", version="1.0.0")
    app.state.registry, app.state.store = reg, store
    frames: dict[str, pd.DataFrame] = {}

    def frame(dataset_id: str) -> pd.DataFrame:
        if dataset_id in frames:
            return frames[dataset_id]
        if dataset_id in CATALOG:
            try:
                df = load(dataset_id, settings.datasets_dir)
            except DatasetError as e:
                raise HTTPException(502, str(e)) from e
        else:
            up = store.upload(dataset_id)
            if not up:
                raise HTTPException(404, "dataset not found")
            df = pd.read_csv(up["path"])
        frames[dataset_id] = df
        return df

    def sliced(dataset_id: str, a: float, b: float) -> pd.DataFrame:
        df = frame(dataset_id)
        if b <= a:
            raise HTTPException(422, "rows_to must be greater than rows_from")
        return df.iloc[int(len(df) * a):int(len(df) * b)]

    def champion(project: str) -> tuple[dict, object]:
        mid = reg.champion(project)
        if not mid:
            raise HTTPException(404, "project not found or has no champion")
        return reg.meta(mid), reg.load(mid)

    def read_csv_upload(data: bytes) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as e:
            raise HTTPException(422, f"Could not read CSV: {e}") from e
        if df.empty or df.shape[1] < 2:
            raise HTTPException(422, "CSV needs at least 2 columns and 1 row")
        return df

    # ---------- health ----------
    @app.get("/health")
    def health(check_llm: bool = False) -> dict:
        out = {"status": "ok", "projects": len(reg.projects())}
        if check_llm:
            out["llm"] = llm.ping(settings) if llm_enabled else False
        return out

    # ---------- datasets ----------
    @app.get("/datasets")
    def datasets() -> list[dict]:
        out = [{"id": d.id, "title": d.title, "task": d.task, "target": d.target, "description": d.description,
                "source": d.source, "time_column": d.time_column, "builtin": True} for d in CATALOG.values()]
        out += [{"id": u["id"], "title": u["name"], "task": None, "target": None,
                 "description": f"Uploaded CSV, {u['rows']} rows x {u['cols']} columns", "source": "upload",
                 "time_column": None, "builtin": False} for u in store.uploads()]
        return out

    @app.post("/datasets/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(413, "CSV larger than 200 MB")
        df = read_csv_upload(data)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        path = uploads_dir / f"{int(time.time() * 1000)}.csv"
        path.write_bytes(data)
        return {"id": store.add_upload(file.filename or "upload.csv", path, len(df), df.shape[1]),
                "rows": len(df), "columns": list(df.columns)}

    @app.get("/datasets/{dataset_id}/preview")
    def preview(dataset_id: str, rows: int = 20) -> dict:
        df = frame(dataset_id)
        return {"columns": list(df.columns), "rows": df.head(min(rows, 200)).astype(object).where(df.head(min(rows, 200)).notna(), None).values.tolist(),
                "shape": list(df.shape)}

    @app.post("/profile")
    def do_profile(req: PlanRequest) -> dict:
        df = frame(req.dataset_id)
        tgt = req.target or (CATALOG[req.dataset_id].target if req.dataset_id in CATALOG else None)
        return profile(df, tgt)

    # ---------- planning ----------
    @app.post("/plan")
    def plan(req: PlanRequest) -> dict:
        df = frame(req.dataset_id)
        info = CATALOG.get(req.dataset_id)
        target = req.target if req.target is not None else (info.target if info else None)
        task = req.task or (info.task if info else None)
        tcol = req.time_column or (info.time_column if info else None)
        prof = profile(df, target)
        try:
            draft = validate_plan(rule_plan(prof, target, task, tcol), list(df.columns))
        except PlanError as e:
            raise HTTPException(422, str(e)) from e
        out = {"plan": draft, "llm_error": None, "leakage": [], "profile_quality": prof["quality"]["score"]}
        if req.use_llm and chat is not None:
            try:
                out["plan"] = llm_plan(draft, prof, info.description if info else "", chat)
            except (llm.LLMError, PlanError, ValueError, TypeError) as e:
                out["llm_error"] = str(e)[:300]
        p = out["plan"]
        if p["task"] in ("classification", "regression"):
            clean, _ = apply_plan(df, {**p, "fitted": {}})
            scores = single_feature_scores(clean, p["target"], p["task"])
            out["single_feature_scores"] = scores[:15]
            for s in leakage_suspects(scores):
                if p["columns"][s["column"]]["role"] == "feature":
                    p["columns"][s["column"]].update(role="drop", reason=f"leakage: predicts the target almost perfectly on its own ({s['metric']} {s['score']})")
                out["leakage"].append({**s, "source": "statistics"})
        for c, e in p["columns"].items():
            if e["reason"].startswith("leakage") and not any(x["column"] == c for x in out["leakage"]):
                out["leakage"].append({"column": c, "source": "llm", "reason": e["reason"]})
        return out

    # ---------- training & registry ----------
    @app.post("/train")
    def train(req: TrainRequest) -> dict:
        df = sliced(req.dataset_id, req.rows_from, req.rows_to)
        try:
            plan = validate_plan({**req.plan, "fitted": {}}, list(df.columns))
        except PlanError as e:
            raise HTTPException(422, str(e)) from e
        jid = store.start_job("train", req.project, lambda progress: _summ(train_project(
            reg, req.project, df, plan, {"id": req.dataset_id, "rows_from": req.rows_from, "rows_to": req.rows_to},
            progress, req.models)))
        return {"job_id": jid}

    def _summ(meta: dict) -> dict:
        return {"model_id": meta["model_id"], "metrics": meta["report"].get("metrics"),
                "best_model": meta["report"]["best_model"]}

    @app.get("/jobs/{jid}")
    def job(jid: str) -> dict:
        j = store.job(jid)
        if not j:
            raise HTTPException(404, "job not found")
        return j

    @app.get("/jobs")
    def jobs() -> list[dict]:
        return store.jobs()

    @app.get("/projects")
    def projects() -> list[dict]:
        out = []
        for name, p in reg.projects().items():
            champ = reg.meta(p["champion"]) if p["champion"] else None
            out.append({"project": name, "champion": p["champion"], "versions": p["versions"], "history": p["history"],
                        "task": champ["task"] if champ else None, "target": champ["target"] if champ else None,
                        "metrics": champ["report"].get("metrics") if champ else None})
        return out

    @app.get("/models/{mid}")
    def model(mid: str) -> dict:
        try:
            return reg.meta(mid)
        except KeyError as e:
            raise HTTPException(404, "model not found") from e

    @app.post("/projects/{project}/promote")
    def promote(project: str, req: PromoteRequest) -> dict:
        try:
            reg.promote(project, req.model_id, "manual promotion")
        except KeyError as e:
            raise HTTPException(404, "model not in project") from e
        return {"champion": req.model_id}

    @app.post("/projects/{project}/rollback")
    def rollback(project: str) -> dict:
        t0 = time.perf_counter()
        try:
            mid = reg.rollback(project)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return {"champion": mid, "seconds": round(time.perf_counter() - t0, 4)}

    @app.delete("/projects/{project}")
    def delete_project(project: str) -> dict:
        if project not in reg.projects():
            raise HTTPException(404, "project not found")
        reg.delete_project(project)
        return {"deleted": project}

    # ---------- serving ----------
    @app.post("/projects/{project}/predict")
    def predict(project: str, req: PredictRequest) -> dict:
        meta, model = champion(project)
        if meta["task"] == "forecasting":
            raise HTTPException(422, "use /forecast for forecasting projects")
        try:
            out = score_records(model, meta, req.records, req.explain)
        except InputError as e:
            raise HTTPException(422, str(e)) from e
        n_warn = sum(len(p["warnings"]) for p in out["predictions"])
        store.audit(project, meta["model_id"], "online" if len(req.records) == 1 else "batch", len(req.records),
                    out["latency_ms"], n_warn, {"first": out["predictions"][0].get("prediction", out["predictions"][0].get("segment"))})
        return out

    @app.post("/projects/{project}/predict/csv")
    async def predict_csv(project: str, file: UploadFile = File(...)) -> Response:
        meta, model = champion(project)
        df = read_csv_upload(await file.read())
        try:
            out = score_records(model, meta, df)
        except InputError as e:
            raise HTTPException(422, str(e)) from e
        res = df.copy()
        preds = out["predictions"]
        key = "prediction" if "prediction" in preds[0] else "segment"
        res[key] = [p[key] for p in preds]
        if "score" in preds[0]:
            res["score"] = [p["score"] for p in preds]
        res["warnings"] = ["; ".join(p["warnings"]) for p in preds]
        store.audit(project, meta["model_id"], "csv", len(df), out["latency_ms"], sum(len(p["warnings"]) for p in preds), {})
        return Response(res.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="predictions.csv"',
                                 "X-Latency-Ms": str(out["latency_ms"])})

    @app.get("/projects/{project}/forecast")
    def get_forecast(project: str, horizon: int = 12) -> list[dict]:
        if not 1 <= horizon <= 60:
            raise HTTPException(422, "horizon must be 1-60")
        try:
            return forecast(reg, project, horizon)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except KeyError as e:
            raise HTTPException(404, "project not found") from e

    # ---------- monitoring & retraining ----------
    def monitor_df(project: str, df: pd.DataFrame, source: str, strength: float = 0.0) -> dict:
        meta, _ = champion(project)
        if strength:
            df = simulate_drift(df, meta["schema"], strength)
        try:
            rep = monitor_batch(reg, project, df)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        rep["id"] = store.add_monitoring(project, source, rep)
        return rep

    @app.post("/projects/{project}/monitor")
    def monitor_slice(project: str, req: SliceRequest) -> dict:
        df = sliced(req.dataset_id, req.rows_from, req.rows_to)
        src = f"{req.dataset_id}[{req.rows_from:.0%}-{req.rows_to:.0%}]" + (f" +drift x{req.simulate_drift}" if req.simulate_drift else "")
        return monitor_df(project, df, src, req.simulate_drift)

    @app.post("/projects/{project}/monitor/csv")
    async def monitor_csv(project: str, file: UploadFile = File(...)) -> dict:
        return monitor_df(project, read_csv_upload(await file.read()), file.filename or "upload")

    @app.get("/projects/{project}/monitoring")
    def monitoring(project: str) -> list[dict]:
        return store.monitoring(project)

    @app.get("/monitoring/{report_id}")
    def monitoring_report(report_id: int) -> dict:
        r = store.monitoring_report(report_id)
        if not r:
            raise HTTPException(404, "report not found")
        return r

    @app.post("/projects/{project}/retrain")
    def do_retrain(project: str, req: SliceRequest, min_gain: float = 0.005) -> dict:
        champion(project)
        df = sliced(req.dataset_id, req.rows_from, req.rows_to)
        jid = store.start_job("retrain", project, lambda progress: retrain(reg, project, df, progress, min_gain))
        return {"job_id": jid}

    # ---------- governance ----------
    @app.get("/audit")
    def audit(project: str | None = None) -> list[dict]:
        return store.audit_log(project)

    @app.get("/models/{mid}/card", response_class=PlainTextResponse)
    def model_card(mid: str) -> str:
        try:
            m = reg.meta(mid)
        except KeyError as e:
            raise HTTPException(404, "model not found") from e
        rep, plan = m["report"], m["plan"]
        drops = [f"- `{c}`: {e['reason']}" for c, e in plan["columns"].items() if e["role"] == "drop"]
        mets = "\n".join(f"| {k} | {v} | {rep.get('baseline', {}).get(k, '')} |" for k, v in (rep.get("metrics") or {}).items())
        imp = "\n".join(f"{i + 1}. {x['feature']} ({x['importance']})" for i, x in enumerate(rep.get("importance", [])[:8]))
        return (f"# Model card: {mid}\n\n- Project: {m['project']} (version {m['version']})\n- Task: {m['task']}"
                f"\n- Target: {m['target']}" + (f" (positive class: {m['positive_class']})" if m.get("positive_class") else "") +
                f"\n- Algorithm: {rep['best_model']}\n- Training rows: {rep.get('train_rows')}, holdout rows: {rep.get('test_rows')}"
                f"\n- Dataset: {m['dataset']}\n- Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(m['created']))}"
                f"\n\n## Metrics (holdout)\n\n| Metric | Model | Baseline |\n|---|---|---|\n{mets}"
                f"\n\n## Most important features (permutation)\n\n{imp or 'n/a'}"
                f"\n\n## Data decisions\n\n" + ("\n".join(drops) or "- none") +
                f"\n\nPlan source: {plan.get('source')}. " + " ".join(plan.get("notes", [])) +
                "\n\n## Intended use and limitations\n\nTrained on the dataset above only; monitor drift before relying on "
                "it for new populations. Predictions include warnings for unseen categories and out-of-range inputs.\n")

    return app
