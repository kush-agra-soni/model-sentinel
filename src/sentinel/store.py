"""SQLite store for background jobs, uploaded datasets, the prediction audit log and monitoring history."""
import json
import time
import uuid
import sqlite3
import threading
import traceback
from pathlib import Path
from collections.abc import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, project TEXT, status TEXT, log TEXT, result TEXT,
                                 error TEXT, created REAL, finished REAL);
CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, name TEXT, path TEXT, rows INTEGER, cols INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, project TEXT, model_id TEXT,
                                  kind TEXT, rows INTEGER, latency_ms REAL, warnings INTEGER, summary TEXT);
CREATE TABLE IF NOT EXISTS monitoring (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, project TEXT, model_id TEXT,
                                       source TEXT, status TEXT, weighted_psi REAL, prediction_psi REAL,
                                       metric REAL, recommendation TEXT, report TEXT);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.execute("UPDATE jobs SET status='failed', error='Interrupted by restart' WHERE status='running'")
        self._db.commit()
        self._lock = threading.RLock()
        self.live: dict[str, dict] = {}

    def _exec(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _rows(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    # ---------- jobs ----------
    def start_job(self, kind: str, project: str, fn: Callable[[Callable[[str], None]], dict]) -> str:
        jid = f"{kind[:5]}-{uuid.uuid4().hex[:8]}"
        state = {"id": jid, "kind": kind, "project": project, "status": "running", "log": [], "result": None,
                 "error": None, "created": time.time(), "finished": None}
        with self._lock:
            self.live[jid] = state
        self._exec("INSERT INTO jobs (id, kind, project, status, log, created) VALUES (?,?,?,?,?,?)",
                   (jid, kind, project, "running", "[]", state["created"]))

        def progress(msg: str) -> None:
            with self._lock:
                state["log"].append({"ts": time.time(), "msg": str(msg)})

        def run():
            try:
                state["result"] = fn(progress)
                state["status"] = "done"
            except Exception as e:
                state["status"], state["error"] = "failed", f"{type(e).__name__}: {e}"
                progress(f"ERROR {state['error']}")
                progress(traceback.format_exc()[-800:])
            state["finished"] = time.time()
            self._exec("UPDATE jobs SET status=?, log=?, result=?, error=?, finished=? WHERE id=?",
                       (state["status"], json.dumps(state["log"]), json.dumps(state["result"], default=str),
                        state["error"], state["finished"], jid))

        threading.Thread(target=run, daemon=True, name=jid).start()
        return jid

    def job(self, jid: str) -> dict | None:
        with self._lock:
            if jid in self.live:
                s = self.live[jid]
                return {**s, "log": list(s["log"])}
        r = self._rows("SELECT * FROM jobs WHERE id=?", (jid,))
        if not r:
            return None
        d = r[0]
        d["log"] = json.loads(d["log"] or "[]")
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def jobs(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT id, kind, project, status, error, created, finished FROM jobs ORDER BY created DESC LIMIT ?",
                          (limit,))

    # ---------- uploads ----------
    def add_upload(self, name: str, path: Path, rows: int, cols: int) -> str:
        uid = "upload-" + uuid.uuid4().hex[:8]
        self._exec("INSERT INTO uploads VALUES (?,?,?,?,?,?)", (uid, name, str(path), rows, cols, time.time()))
        return uid

    def uploads(self) -> list[dict]:
        return self._rows("SELECT * FROM uploads ORDER BY created DESC")

    def upload(self, uid: str) -> dict | None:
        r = self._rows("SELECT * FROM uploads WHERE id=?", (uid,))
        return r[0] if r else None

    def delete_upload(self, uid: str) -> bool:
        return self._exec("DELETE FROM uploads WHERE id=?", (uid,)).rowcount > 0

    # ---------- audit & monitoring ----------
    def audit(self, project: str, model_id: str, kind: str, rows: int, latency_ms: float, warnings: int, summary: dict):
        self._exec("INSERT INTO audit (ts, project, model_id, kind, rows, latency_ms, warnings, summary) VALUES (?,?,?,?,?,?,?,?)",
                   (time.time(), project, model_id, kind, rows, latency_ms, warnings, json.dumps(summary, default=str)))

    def audit_log(self, project: str | None = None, limit: int = 200) -> list[dict]:
        if project:
            return self._rows("SELECT * FROM audit WHERE project=? ORDER BY id DESC LIMIT ?", (project, limit))
        return self._rows("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))

    def add_monitoring(self, project: str, source: str, rep: dict) -> int:
        perf = rep.get("performance") or {}
        return self._exec(
            "INSERT INTO monitoring (ts, project, model_id, source, status, weighted_psi, prediction_psi, metric, "
            "recommendation, report) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rep["ts"], project, rep["model_id"], source, rep["status"], rep["importance_weighted_psi"],
             (rep.get("prediction_drift") or {}).get("psi"), perf.get("current"), rep["recommendation"],
             json.dumps(rep, default=str))).lastrowid

    def monitoring(self, project: str) -> list[dict]:
        out = self._rows("SELECT id, ts, model_id, source, status, weighted_psi, prediction_psi, metric, recommendation "
                         "FROM monitoring WHERE project=? ORDER BY id", (project,))
        return out

    def monitoring_report(self, mid: int) -> dict | None:
        r = self._rows("SELECT report FROM monitoring WHERE id=?", (mid,))
        return json.loads(r[0]["report"]) if r else None
