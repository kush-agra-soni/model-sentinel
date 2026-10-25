import sys
import json
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "launcher")]


def churn_frame(n: int = 1500, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Synthetic churn data with an ID, a numbers-as-text column, messy labels and a planted leak."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n) + shift * 20
    monthly = rng.normal(65, 20, n).round(2) + shift * 25
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    logit = -0.04 * tenure + 0.02 * (monthly - 65) + np.where(contract == "Month-to-month", 1.2, -0.8)
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    total = (tenure * monthly).round(2).astype(str)
    total[rng.choice(n, min(10, n // 5), replace=False)] = " "
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "tenure": tenure, "monthly_charges": monthly, "total_charges": total,
        "contract": [c if i % 7 else f" {c.upper()} " for i, c in enumerate(contract)],
        "support_calls": rng.poisson(2 + shift, n),
        "cancellation_fee_charged": np.where(churn == "Yes", "yes", "no"),  # leak: only known after churn
        "churn": churn,
    })


def seasonal_series(months: int = 96) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=months, freq="MS")
    t = np.arange(months)
    y = 200 + 3 * t + 40 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 6, months)
    return pd.DataFrame({"date": idx, "sales": y.round(1)})


class FakeChat:
    """Planner LLM stand-in: flags a configured leak column and adds a note."""

    def __init__(self, leak: str | None = "cancellation_fee_charged", fail: bool = False):
        self.leak, self.fail, self.calls = leak, fail, 0

    def __call__(self, msgs):
        self.calls += 1
        if self.fail:
            from sentinel.llm import LLMError
            raise LLMError("down")
        out = {"leakage": [{"column": self.leak, "reason": "only known after the outcome"}] if self.leak else [],
               "notes": ["checked by fake planner"], "columns": {"not_a_column": {"role": "drop"}}}
        return json.dumps(out)


def wait_job(c, jid, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from sentinel.api import create_app
    from sentinel.config import Settings

    chat = FakeChat()
    c = TestClient(create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path), chat=chat))
    c.chat = chat
    return c


def upload_frame(c, df: pd.DataFrame, name: str = "data.csv") -> str:
    r = c.post("/datasets/upload", files={"file": (name, df.to_csv(index=False).encode())})
    assert r.status_code == 200, r.text
    return r.json()["id"]
