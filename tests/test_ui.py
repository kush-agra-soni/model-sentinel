"""Renders the UI headlessly against an in-process backend with a trained project."""
import time
import httpx
import pytest
import socket
import uvicorn
import threading
from conftest import ROOT, FakeChat, churn_frame


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from sentinel.api import create_app
    from sentinel.config import Settings

    app = create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path_factory.mktemp("ui")), chat=FakeChat())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    c = httpx.Client(base_url=url, timeout=120)
    uid = c.post("/datasets/upload", files={"file": ("churn.csv", churn_frame(1200).to_csv(index=False).encode())}).json()["id"]
    plan = c.post("/plan", json={"dataset_id": uid, "target": "churn"}).json()["plan"]
    jid = c.post("/train", json={"project": "churn", "dataset_id": uid, "plan": plan, "models": ["logistic_regression"]}).json()["job_id"]
    while c.get(f"/jobs/{jid}").json()["status"] == "running":
        time.sleep(0.1)
    mp = pytest.MonkeyPatch()
    mp.setenv("BACKEND_URL", url)
    yield url
    mp.undo()
    server.should_exit = True


def test_ui_renders_tabs_and_predicts(backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Data", "Plan", "Train", "Predict", "Monitor", "Governance"]
    next(b for b in at.button if b.label == "Profile dataset").click().run()
    assert not at.exception and any(m.label == "Quality score" for m in at.metric)
    predict = next(b for b in at.button if b.label == "Predict")
    predict.click().run()
    assert not at.exception
    assert any(m.label == "Prediction" for m in at.metric)
    assert any("Model card" in m.value for m in at.markdown)
