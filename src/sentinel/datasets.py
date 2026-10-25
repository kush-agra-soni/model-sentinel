"""Built-in datasets: some bundled with scikit-learn (offline), some downloaded once and checksum-verified."""
import io
import httpx
import hashlib
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    title: str
    task: str  # classification | regression | clustering | forecasting
    target: str | None
    description: str
    source: str
    url: str | None = None
    sha256: str | None = None
    time_column: str | None = None


CATALOG = {d.id: d for d in [
    DatasetInfo("telco_churn", "Telco customer churn", "classification", "Churn",
                "7,043 telecom customers; predict who cancels. Messy numeric column (TotalCharges) and an ID column.",
                "IBM sample data",
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
                "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"),
    DatasetInfo("bank_marketing", "Bank marketing (2008-2010)", "classification", "y",
                "41,188 phone-campaign contacts in time order across the 2008 financial crisis (real drift). Contains a "
                "known leak: call duration is only known after the outcome.",
                "UCI ML Repository (CC BY 4.0)", "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
                "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"),
    DatasetInfo("titanic", "Titanic survival", "classification", "survived",
                "891 passengers. Contains a leak: the 'alive' column restates the target.", "seaborn-data",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
                "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2"),
    DatasetInfo("airline_passengers", "Airline passengers (1949-1960)", "forecasting", "passengers",
                "Monthly international airline passengers; trend plus yearly seasonality.", "seaborn-data",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
                "237d834127d9c6355630d8f443a7a2377b5925923010009b59809ba0b67f4fac", time_column="date"),
    DatasetInfo("breast_cancer", "Breast cancer diagnosis", "classification", "malignant",
                "569 tumours, 30 numeric measurements (bundled, offline).", "scikit-learn / UCI"),
    DatasetInfo("diabetes", "Diabetes progression", "regression", "progression",
                "442 patients, 10 baseline variables; predict disease progression after one year (bundled).",
                "scikit-learn"),
    DatasetInfo("wine", "Wine cultivar", "classification", "cultivar",
                "178 wines, 13 chemical measurements, 3 classes (bundled).", "scikit-learn / UCI"),
    DatasetInfo("customer_segments", "Customer segments (synthetic)", "clustering", None,
                "1,200 synthetic shoppers with income, spend, visits and basket size; find natural segments.",
                "generated (seeded)"),
]}


class DatasetError(RuntimeError):
    pass


def _download(info: DatasetInfo, cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(info.url).name
    if path.exists():
        data = path.read_bytes()
    else:
        try:
            r = httpx.get(info.url, timeout=60, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise DatasetError(f"Could not download {info.title}: {e}") from e
        data = r.content
    digest = hashlib.sha256(data).hexdigest()
    if digest != info.sha256:
        path.unlink(missing_ok=True)
        raise DatasetError(f"Checksum mismatch for {info.title} (got {digest[:12]}...)")
    if not path.exists():
        path.write_bytes(data)
    return data


def load(dataset_id: str, cache: Path) -> pd.DataFrame:
    info = CATALOG.get(dataset_id)
    if info is None:
        raise DatasetError(f"Unknown dataset {dataset_id!r}")
    if dataset_id == "telco_churn":
        return pd.read_csv(io.BytesIO(_download(info, cache)))
    if dataset_id == "titanic":
        return pd.read_csv(io.BytesIO(_download(info, cache)))
    if dataset_id == "bank_marketing":
        outer = zipfile.ZipFile(io.BytesIO(_download(info, cache)))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
        name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
        return pd.read_csv(io.BytesIO(inner.read(name)), sep=";")
    if dataset_id == "airline_passengers":
        df = pd.read_csv(io.BytesIO(_download(info, cache)))
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B")
        return df[["date", "passengers"]]
    from sklearn import datasets as skd
    if dataset_id == "breast_cancer":
        b = skd.load_breast_cancer(as_frame=True)
        df = b.frame.rename(columns={"target": "malignant"})
        df["malignant"] = (df["malignant"] == 0).astype(int)  # sklearn encodes benign as 1
        return df
    if dataset_id == "diabetes":
        return skd.load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})
    if dataset_id == "wine":
        return skd.load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
    if dataset_id == "customer_segments":
        return customer_segments()
    raise DatasetError(f"No loader for {dataset_id}")


def customer_segments(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = {"budget": (28, 25, 2, 18), "family": (62, 55, 4, 64), "premium": (120, 82, 3, 140), "occasional": (55, 20, 1, 35)}
    rows = []
    for (inc, spend, visits, basket), share in zip(centers.values(), (0.3, 0.3, 0.15, 0.25)):
        k = int(n * share)
        rows.append(pd.DataFrame({
            "annual_income_k": rng.normal(inc, inc * 0.12, k).round(1),
            "spending_score": np.clip(rng.normal(spend, 7, k), 1, 100).round(0),
            "visits_per_month": np.clip(rng.normal(visits, 0.8, k), 0, None).round(1),
            "avg_basket": np.clip(rng.normal(basket, basket * 0.15, k), 1, None).round(2),
            "age": np.clip(rng.normal(30 + inc / 4, 8, k), 18, 80).round(0),
        }))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "customer_id", [f"C{10000 + i}" for i in range(len(df))])
    return df


def file_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]
