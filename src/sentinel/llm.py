import re
import json
import time
import httpx
from .config import Settings


class LLMError(RuntimeError):
    pass


def complete(s: Settings, messages: list[dict], max_tokens: int = 1200) -> str:
    payload = {"model": s.llm_model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    last = None
    for attempt in range(3):
        try:
            r = httpx.post(f"{s.llm_base_url}/chat/completions", json=payload,
                           headers={"Authorization": f"Bearer {s.llm_api_key}"},
                           timeout=httpx.Timeout(s.llm_timeout, connect=15))
            if r.status_code == 200:
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
            last = LLMError(f"LLM server returned {r.status_code}: {r.text[:200]}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except httpx.HTTPError as e:
            last = LLMError(f"Cannot reach LLM server: {e}")
        time.sleep(2 * (attempt + 1))
    raise last or LLMError("LLM call failed")


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    cand = (m.group(1) if m else text).strip()
    starts = sorted((i, o, c) for o, c in (("{", "}"), ("[", "]")) if (i := cand.find(o)) >= 0)
    for i, o, c in starts:
        j = cand.rfind(c)
        if j > i:
            try:
                return json.loads(cand[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("Model did not return valid JSON")


def ping(s: Settings) -> bool:
    try:
        return httpx.get(f"{s.llm_base_url}/models", headers={"Authorization": f"Bearer {s.llm_api_key}"},
                         timeout=10).status_code == 200
    except httpx.HTTPError:
        return False
