# Tabular ML Platform

An end-to-end platform for machine learning on tabular data. You bring a CSV, or pick a built-in dataset, and it covers the whole lifecycle:

- **Profiling:** a data-quality scorecard.
- **Cleaning plan:** proposed by rules and reviewed by an LLM, which catches target leakage that statistics alone miss.
- **Training:** cross-validated models compared against a naive baseline.
- **Model registry:** versioned models with a champion.
- **Predictions:** served with input validation and per-prediction explanations.
- **Monitoring and retraining:** drift monitoring on new data, and retraining that promotes a new model only if it is measurably better. Rollback takes milliseconds.

It supports classification, regression, clustering (segmentation) and time-series forecasting. A one-click launcher sets everything up on Windows.

![Train](docs/screenshots/train.png)

## Contents

- [Quick start](#quick-start)
- [Features](#features)
- [Results](#results)
- [How it works](#how-it-works)
- [API](#api)
- [Development](#development)
- [Design decisions](#design-decisions)

## Quick start

**Requirements:** Windows 10/11 with Python 3.10 or newer ([python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked). An OpenAI-compatible LLM endpoint is optional: without one, planning is rule-based.

1. Download or clone this repository.
2. Double-click `launcher.exe`.

The first run takes about 2 minutes. The launcher creates `.venv/`, installs `requirements.txt`, creates `.env`, starts the API and UI, and opens `http://localhost:8505`. Closing the launcher window stops everything.

Optional `.env` for LLM-reviewed cleaning plans:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=your-key
LLM_MODEL=qwen3.6-35b-a3b
```

A typical walk-through:

1. **Data:** pick *Telco customer churn* and click **Profile dataset**.
2. **Plan:** click **Build plan**.
3. **Train:** click **Train**.
4. **Predict:** fill in the form and click **Predict**.

For drift, pick *Bank marketing*, train on the first 40% of rows, then use the **Monitor** tab on later rows. The data is in time order, so later rows reflect real economic change.

On Linux or macOS, run `./launch.sh`.

## Features

| Area | What it does |
|---|---|
| Data | 8 built-in datasets, each downloaded once with its checksum verified, or bundled. You can also upload CSV files. |
| Profiling | Column type inference (numeric, categorical, datetime, identifier, numbers stored as text, constant), distributions, correlations, target balance, and a 0–100 quality score with an issue list (missing values, inconsistent labels, duplicates, outliers, identifiers) |
| Cleaning plan | Rules propose each column's role, type and fixes. An LLM reviews the plan for leakage and data problems. Every suggestion is validated against a fixed allowlist of operations, and you can edit the plan in the UI. |
| Leakage detection | Two independent screens. **Statistics:** any single feature that predicts the target almost perfectly. **LLM:** columns whose meaning shows they are recorded after the outcome. |
| Training | Leakage-safe scikit-learn pipelines (imputation, scaling and encoding fitted inside cross-validation); logistic or ridge regression, random forest, histogram gradient boosting and LightGBM; 5-fold cross-validation; holdout metrics against a naive baseline; permutation importance; confusion matrix; calibration |
| Other tasks | Clustering: k-means with the number of clusters chosen by silhouette score, and segment profiles. Forecasting: lag and seasonal features, recursive multi-step forecasts, and a seasonal-naive baseline. |
| Registry | Versioned models, each with its plan, input schema, reference sample and metrics. One champion per project, with promotion history, rollback and model cards. |
| Serving | Online and CSV batch predictions. Inputs are validated (missing columns, unseen categories, out-of-range values) and warnings are returned instead of errors. Single predictions include a feature-contribution explanation. Every request is written to an audit log. |
| Monitoring | Data drift per feature (PSI with KS or chi-squared tests), weighted by feature importance; prediction drift; and performance on labelled new data, judged against both an absolute floor and the original score. Each check ends in a recommendation (none / watch / retrain). |
| Retraining | A challenger is trained on recent data and must beat the champion on held-out recent rows by a minimum margin. Otherwise the champion stays, and rollback takes milliseconds. |

### Screenshots

| Data quality | Cleaning plan with leakage findings |
|---|---|
| ![Data](docs/screenshots/data.png) | ![Plan](docs/screenshots/plan.png) |

| Prediction with explanation | Drift monitoring |
|---|---|
| ![Predict](docs/screenshots/predict.png) | ![Monitor](docs/screenshots/monitor.png) |

## Results

Measured with `scripts/run_eval.py`, using an LLM-reviewed plan for every dataset (Qwen3.6-35B-A3B). Metrics are on a 20% holdout.

| Dataset | Task | Leakage found | Best model | Result | Naive baseline |
|---|---|---|---|---|---|
| Telco customer churn (7,043 rows) | classification | none (ID dropped) | logistic regression | ROC AUC 0.842 | 0.500 |
| Titanic (891 rows) | classification | `alive` (restates the target) | random forest | ROC AUC 0.845 | 0.500 |
| Bank marketing, first 40% (16,475 rows) | classification | `duration` (known only after the call) | logistic regression | ROC AUC 0.592 | 0.500 |
| Breast cancer (569 rows) | classification | none | logistic regression | ROC AUC 0.997 | 0.500 |
| Wine (178 rows, 3 classes) | classification | none | logistic regression | accuracy 1.00 | 0.39 |
| Diabetes (442 rows) | regression | none | ridge | RMSE 53.8 | 73.2 |
| Customer segments (1,200 rows) | clustering | none (ID dropped) | k-means, k = 3 | silhouette 0.49 | - |
| Airline passengers (144 months) | forecasting | none | ridge on lags | MAPE 5.2% | 10.0% (seasonal naive) |

**Leakage: statistics and the LLM catch different things.** The single-feature screen flags Titanic's `alive` (AUC 1.0 on its own). The bank's `duration` scores only 0.80 on its own, which looks like any strong feature, but the LLM recognises from its meaning that call duration is known only after the outcome. Two lessons from development:

- An earlier prompt let the LLM call correlated measurements "leakage", and it dropped good breast-cancer features. The prompt now separates leakage from redundancy.
- Invalid LLM suggestions are now ignored one by one instead of rejecting the whole review.

**Drift: bank marketing (2008–2010).** The model was trained on the first 40% of contacts in time order, then monitored on later periods:

| Rows monitored | Drift status | Importance-weighted PSI | Prediction PSI | AUC on new labels | Recommendation |
|---|---|---|---|---|---|
| 0–40% (training period) | stable | 0.001 | 0.003 | 0.624 | none |
| 40–70% | major | 2.22 | 1.76 | 0.528 | retrain |
| 70–100% | major | 2.18 | 2.75 | 0.671 | retrain |

The economic indicators (employment variation rate, Euribor, number employed) shifted by PSI values above 5. The retrained challenger beat the champion on held-out recent rows by **+0.137 AUC (0.520 → 0.657)** and was promoted automatically. Rolling back took 4 ms.

**Serving:**

- one prediction: 9 ms median in-process; 22 ms median and 27 ms at the 95th percentile through the HTTP API
- a 1,000-row batch: 20 ms

**A bug found by monitoring.** Before release, the drift check raised prediction drift on data from the training period itself. The cause was categorical integer columns: they were encoded as numbers in training but as text when serving. All categorical values are now cast to text in one shared cleaning step used by training, serving and monitoring.

## How it works

```
CSV / built-in dataset
   -> profile (types, quality, issues)
   -> rule plan -> LLM review (leakage, fixes) -> validated plan (user-editable)
   -> apply plan (row-wise fixes; clip bounds learned once)          shared by training, serving, monitoring
   -> scikit-learn pipeline (impute + scale + one-hot inside CV) -> leaderboard -> holdout report
   -> registry (model, plan, schema, reference sample, metrics; champion per project)
   -> serve (validate -> predict -> explain -> audit)
   -> monitor (PSI / KS / chi2, prediction drift, performance floor) -> retrain challenger -> gate -> promote / keep / rollback
```

| Module | Responsibility |
|---|---|
| `datasets.py` | Built-in catalog, checksum-verified downloads, synthetic segments |
| `profile.py` | Type inference, EDA statistics, quality scorecard |
| `planner.py` | Rule planner, LLM review with validation and truncated-JSON repair, plan application |
| `leakage.py` | Single-feature predictive-power screen |
| `training.py` | Supervised, clustering and forecasting trainers |
| `registry.py` | Versioned artifacts, champion, promotion history, rollback |
| `serving.py` | Input validation, prediction, ablation explanations |
| `drift.py`, `pipeline.py` | Drift metrics, monitoring, gated retraining |
| `api.py`, `store.py` | REST API, jobs, uploads, audit and monitoring history (SQLite) |

## API

Interactive documentation is available at `http://localhost:8005/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/datasets`, `/datasets/{id}/preview` | Catalog and uploads, preview rows |
| POST | `/datasets/upload` | Upload a CSV |
| POST | `/profile` | EDA and quality scorecard |
| POST | `/plan` | Rule plus LLM cleaning plan, with leakage findings |
| POST | `/train` | Train a project (background job; optional row range for time-ordered data) |
| GET | `/jobs/{id}` | Job status and log |
| GET | `/projects`, `/models/{id}`, `/models/{id}/card` | Projects, model metadata, model card (Markdown) |
| POST | `/projects/{p}/predict`, `/projects/{p}/predict/csv` | Online or batch predictions |
| GET | `/projects/{p}/forecast?horizon=` | Forecast future periods |
| POST | `/projects/{p}/monitor`, `/projects/{p}/monitor/csv` | Drift and performance check |
| GET | `/projects/{p}/monitoring` | Monitoring history |
| POST | `/projects/{p}/retrain` | Challenger training and promotion gate |
| POST | `/projects/{p}/promote`, `/projects/{p}/rollback` | Champion management |
| GET | `/audit` | Prediction audit log |

## Development

```
.venv/Scripts/python -m pytest              # 33 tests, offline (bundled and synthetic data, fake LLM)
.venv/Scripts/python scripts/run_eval.py    # live evaluation on all built-in datasets
powershell -File scripts/build_launcher.ps1
```

## Design decisions

- **In-house registry, explanations and drift.** MLflow, SHAP and Evidently were left out to keep installation small and reliable on Windows. The ablation explanation is model-agnostic and fast for the feature counts typical of tabular business data.
- **Rows are never silently dropped at serving time.** Missing or unexpected inputs are imputed or ignored and reported as warnings, so callers can decide what to do.
- **Promotion requires evidence.** Drift alone recommends retraining, but a new model replaces the champion only after it wins on recent held-out data.
- **Time order matters.** Training and monitoring accept row ranges, so time-ordered data can be split without leaking the future into training.
