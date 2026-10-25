"""Cleaning plans: a rule-based planner, an LLM planner that refines it, strict validation, and application."""
import re
import copy
import json
import numpy as np
import pandas as pd
from collections.abc import Callable
from .llm import LLMError, parse_json

TASKS = ("classification", "regression", "clustering", "forecasting")
ROLES = ("feature", "drop", "target", "time")
KINDS = ("numeric", "categorical", "datetime", "text")
FIXES = ("to_numeric", "strip", "lower", "parse_datetime", "clip_outliers", "log1p")
TARGET_NAMES = re.compile(r"^(target|label|class|y|churn|outcome|survived|default|fraud|is_fraud|price|sales)$", re.I)


class PlanError(ValueError):
    pass


def rule_plan(prof: dict, target: str | None = None, task: str | None = None,
              time_column: str | None = None) -> dict:
    cols = {c["name"]: c for c in prof["column_profiles"]}
    if target is None and task != "clustering":
        target = next((n for n in cols if TARGET_NAMES.match(n)), None)
    if task is None:
        if target is None:
            task = "clustering"
        elif time_column:
            task = "forecasting"
        else:
            t = cols[target]
            task = "classification" if t["kind"] in ("categorical", "constant") or t["unique"] <= 20 else "regression"
    plan = {"target": target, "task": task, "time_column": time_column, "positive_class": None,
            "drop_duplicates": prof["duplicate_rows"] > 0 and task != "forecasting", "columns": {}, "notes": [],
            "source": "rules"}
    for name, c in cols.items():
        entry = {"role": "feature", "kind": "numeric", "fixes": [], "reason": ""}
        if name == target:
            entry.update(role="target", kind="categorical" if task == "classification" else "numeric",
                         reason="prediction target")
            if c["kind"] == "numeric_as_text":
                entry["fixes"] = ["to_numeric"]
        elif name == time_column:
            entry.update(role="time", kind="datetime", fixes=["parse_datetime"], reason="time index")
        elif c["kind"] == "id":
            entry.update(role="drop", kind="categorical", reason="identifier: unique per row, would let the model memorise")
        elif c["kind"] == "constant":
            entry.update(role="drop", kind="categorical", reason="constant: carries no information")
        elif c["kind"] == "text":
            entry.update(role="drop", kind="text", reason="free text: not used by tabular models")
        elif c["kind"] == "numeric_as_text":
            entry.update(kind="numeric", fixes=["to_numeric"], reason="numbers stored as text")
        elif c["kind"] == "datetime":
            entry.update(kind="datetime", fixes=["parse_datetime"], reason="expanded to year / month / weekday")
        elif c["kind"] == "categorical":
            entry.update(kind="categorical", fixes=["strip", "lower"] if c.get("case_or_space_variants") else ["strip"],
                         reason="categorical" + (" with inconsistent labels" if c.get("case_or_space_variants") else ""))
        else:
            fixes = ["clip_outliers"] if c.get("outliers", 0) > 0.01 * prof["rows"] else []
            entry.update(kind="numeric", fixes=fixes, reason="numeric" + (" with extreme outliers" if fixes else ""))
        if entry["role"] == "feature" and c["missing_pct"] > 60:
            entry.update(role="drop", reason=f"{c['missing_pct']}% missing")
        plan["columns"][name] = entry
    if task == "classification" and target:
        share = (prof.get("target_balance") or {}).get("share") or {}
        if len(share) == 2:
            plan["positive_class"] = min(share, key=share.get)  # minority class is usually the event of interest
    return plan


def compact_profile(prof: dict) -> str:
    lines = [f"rows={prof['rows']} duplicate_rows={prof['duplicate_rows']}"]
    for c in prof["column_profiles"]:
        bits = [f"{c['name']}: kind={c['kind']} missing={c['missing_pct']}% unique={c['unique']}"]
        if "top" in c:
            bits.append("top=" + ", ".join(list(c["top"])[:6]))
        if "min" in c:
            bits.append(f"range=[{c['min']}, {c['max']}]")
        lines.append("  ".join(bits))
    return "\n".join(lines)


PLANNER_SYSTEM = f"""You are a senior data scientist reviewing a cleaning plan for a tabular ML dataset.
You get the dataset description, a column profile and a draft plan. Return ONLY compact JSON (no indentation),
with the keys in this order:
{{"leakage": [{{"column": "<name>", "reason": "<why it would not be known at prediction time or restates the target>"}}],
  "target": "<column or null>", "task": one of {list(TASKS)},
  "positive_class": "<class label or null>",
  "columns": {{"<column>": {{"role": one of {list(ROLES)}, "kind": one of {list(KINDS)},
                             "fixes": subset of {list(FIXES)}, "reason": "<short reason>"}}}},
  "notes": ["<important observation>"]}}
Only include columns whose plan you change. Leakage means information that would NOT be available at prediction
time (recorded after the outcome) or that restates the target. Correlation or redundancy between features is NOT
leakage: never drop a feature only because it is correlated with another feature. Drop identifiers. Keep
everything else."""


def parse_json_lenient(text: str):
    """parse_json, falling back to closing the brackets of a reply that was cut off mid-object."""
    try:
        out = parse_json(text)
        if isinstance(out, dict):
            return out
    except LLMError:
        pass
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = body.find("{")
    if start < 0:
        raise LLMError("Model did not return JSON")
    body = body[start:]
    for cut in range(len(body), max(len(body) - 4000, 0), -1):
        chunk = body[:cut].rstrip().rstrip(",")
        stack, in_str, esc = [], False, False
        for ch in chunk:
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            continue
        try:
            return json.loads(chunk + "".join(reversed(stack)))
        except json.JSONDecodeError:
            continue
    raise LLMError("Model did not return valid JSON")


def llm_plan(draft: dict, prof: dict, description: str, chat: Callable[[list[dict]], str]) -> dict:
    msgs = [{"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Dataset: {description or 'user-provided CSV'}\n\nProfile:\n{compact_profile(prof)}\n\n"
                                        f"Draft plan:\n{json.dumps({k: draft[k] for k in ('target', 'task', 'positive_class')})}\n"
                                        + json.dumps({k: {'role': v['role'], 'kind': v['kind'], 'fixes': v['fixes']}
                                                      for k, v in draft['columns'].items()})}]
    out = parse_json_lenient(chat(msgs))
    if not isinstance(out, dict):
        raise LLMError("planner returned no object")
    plan = copy.deepcopy(draft)
    for key in ("target", "task", "positive_class"):
        if out.get(key) not in (None, ""):
            plan[key] = out[key]
    ignored = []
    for name, ch in (out.get("columns") or {}).items():
        if name not in plan["columns"] or not isinstance(ch, dict):
            continue
        e = plan["columns"][name]
        if ch.get("role") in ROLES:
            e["role"] = ch["role"]
        elif "role" in ch:
            ignored.append(f"{name}.role={ch['role']!r}")
        if ch.get("kind") in KINDS:
            e["kind"] = ch["kind"]
        elif "kind" in ch:
            ignored.append(f"{name}.kind={ch['kind']!r}")
        if isinstance(ch.get("fixes"), list):
            e["fixes"] = [f for f in ch["fixes"] if f in FIXES]
        if ch.get("reason"):
            e["reason"] = str(ch["reason"])[:300]
    for leak in out.get("leakage") or []:
        name = leak.get("column") if isinstance(leak, dict) else None
        if name in plan["columns"] and plan["columns"][name]["role"] != "target":
            plan["columns"][name].update(role="drop", reason="leakage: " + str(leak.get("reason", ""))[:200])
    plan["notes"] = [str(n)[:300] for n in (out.get("notes") or [])][:8]
    if ignored:
        plan["notes"].append("Ignored invalid suggestions: " + ", ".join(ignored[:6]))
    plan["source"] = "llm"
    return validate_plan(plan, list(plan["columns"]))


def validate_plan(plan: dict, columns: list[str]) -> dict:
    if plan.get("task") not in TASKS:
        raise PlanError(f"task must be one of {TASKS}")
    if plan["task"] != "clustering":
        if plan.get("target") not in columns:
            raise PlanError(f"target column {plan.get('target')!r} not found")
    targets = [c for c, e in plan["columns"].items() if e["role"] == "target"]
    if plan["task"] != "clustering":
        for c in targets:
            if c != plan["target"]:
                plan["columns"][c]["role"] = "drop"
        plan["columns"][plan["target"]]["role"] = "target"
    for name, e in plan["columns"].items():
        if name not in columns:
            raise PlanError(f"unknown column {name!r}")
        if e.get("role") not in ROLES or e.get("kind") not in KINDS:
            raise PlanError(f"invalid role/kind for {name!r}")
        bad = [f for f in e.get("fixes", []) if f not in FIXES]
        if bad:
            raise PlanError(f"unsupported fixes for {name!r}: {bad}")
    if plan["task"] == "forecasting" and not any(e["role"] == "time" for e in plan["columns"].values()):
        raise PlanError("forecasting needs a time column")
    if not any(e["role"] == "feature" for e in plan["columns"].values()) and plan["task"] != "forecasting":
        raise PlanError("plan keeps no feature columns")
    return plan


def apply_plan(df: pd.DataFrame, plan: dict, inference: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Row-wise fixes; imputation, scaling and encoding are learned inside the model pipeline.

    In training mode, clip bounds are learned and stored in plan["fitted"]; in inference mode (serving and
    monitoring) the stored bounds are reused, duplicates are kept and rows are never dropped."""
    fitted = plan.setdefault("fitted", {})
    log = [f"start: {len(df)} rows x {df.shape[1]} columns"]
    out = df.copy()
    drop = [c for c, e in plan["columns"].items() if e["role"] == "drop" and c in out]
    if drop:
        out = out.drop(columns=drop)
        log.append(f"dropped {len(drop)} column(s): {', '.join(drop)}")
    for c, e in plan["columns"].items():
        if c not in out:
            continue
        for f in e.get("fixes", []):
            s = out[c]
            if f == "strip" and s.dtype == object:
                out[c] = s.astype("string").str.strip().replace("", pd.NA).astype(object)
            elif f == "lower" and s.dtype == object:
                out[c] = s.astype("string").str.lower().astype(object)
            elif f == "to_numeric":
                before = s.isna().sum()
                out[c] = pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")
                added = int(out[c].isna().sum() - before)
                log.append(f"{c}: converted to numbers" + (f" ({added} unparseable values set to missing)" if added else ""))
            elif f == "parse_datetime":
                out[c] = pd.to_datetime(s, errors="coerce")
            elif f == "clip_outliers" and pd.api.types.is_numeric_dtype(s):
                key = f"clip:{c}"
                if key not in fitted or not inference:
                    if inference and key not in fitted:
                        continue
                    fitted[key] = [float(v) for v in s.quantile([0.005, 0.995])]
                lo, hi = fitted[key]
                n = int(((s < lo) | (s > hi)).sum())
                out[c] = s.clip(lo, hi)
                if n and not inference:
                    log.append(f"{c}: clipped {n} extreme values to the 0.5-99.5 percentile range")
            elif f == "log1p" and pd.api.types.is_numeric_dtype(s):
                out[c] = np.log1p(s.clip(lower=0))
                log.append(f"{c}: log-transformed")
    for c, e in plan["columns"].items():
        if e["role"] == "feature" and e["kind"] == "datetime" and c in out:
            d = pd.to_datetime(out[c], errors="coerce")
            out[f"{c}_year"], out[f"{c}_month"], out[f"{c}_weekday"] = d.dt.year, d.dt.month, d.dt.dayofweek
            out = out.drop(columns=[c])
            log.append(f"{c}: expanded to year, month, weekday")
    for c, e in plan["columns"].items():  # categories are always strings: identical encoding in training and serving
        if e["role"] == "feature" and e["kind"] == "categorical" and c in out:
            out[c] = out[c].map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA
                                else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
    for c in out.columns:  # pandas NA from string ops confuses scikit-learn imputers: use NaN
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: np.nan if v is pd.NA else v)
    if inference:
        return out.reset_index(drop=True), log
    if plan.get("drop_duplicates"):
        n = len(out)
        out = out.drop_duplicates()
        if n - len(out):
            log.append(f"removed {n - len(out)} duplicate rows")
    tgt = plan.get("target")
    if tgt and tgt in out and plan["task"] != "clustering":
        n = len(out)
        if plan["task"] == "classification":
            out[tgt] = out[tgt].astype("string").str.strip()
        else:
            out[tgt] = pd.to_numeric(out[tgt], errors="coerce")
        out = out[out[tgt].notna()]
        if n - len(out):
            log.append(f"removed {n - len(out)} rows without a target value")
    log.append(f"result: {len(out)} rows x {out.shape[1]} columns")
    return out.reset_index(drop=True), log
