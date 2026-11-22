"""The text step: a language model reads a calendar note and labels the day.

Every model gets the same instructions, the same note and the same output schema, at
temperature 0 with a fixed seed. The schema constrains decoding to
{"day_type": "workday" | "saturday_like" | "sunday_like"}. An answer that is not valid JSON
with one of those labels counts as wrong when the labels are scored; downstream, the
pipeline falls back to the weekday for that day, as a production pipeline would have to.
A call that fails in transport is retried and then stops the run instead.

Backends:
- "ollama": a local Ollama server (native /api/chat, which honours `format` and `think`);
- "openai": any OpenAI-compatible chat-completions endpoint (vLLM, llama.cpp server,
  LM Studio, Mistral's API), with a JSON-schema response format.

Answers are cached in labels/<llm>.jsonl, keyed by the hash of the model, the prompt version,
the instructions and the note, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

from .config import DAY_TYPES, Config, LLMSpec

SYSTEM = (
    "You estimate how national electricity demand behaves on a given day in a European country.\n"
    "Classify the day by comparing its total electricity demand with ordinary weeks around it:\n"
    "- workday: demand like an ordinary working day of the same weekday;\n"
    "- saturday_like: demand clearly below an ordinary working day, close to an ordinary Saturday;\n"
    "- sunday_like: demand close to an ordinary Sunday.\n"
    "A Saturday or Sunday with nothing special is saturday_like or sunday_like respectively.\n"
    "Use the calendar note, the day of the week, and what you know about the country's customs "
    "(bridge days between a holiday and the weekend, regional holidays and how much of the country "
    "they cover, the Christmas and New Year period, holiday seasons).\n"
    'Answer with JSON only, in the form {"day_type": "<label>"}.'
)

SCHEMA = {
    "type": "object",
    "properties": {"day_type": {"type": "string", "enum": list(DAY_TYPES)}},
    "required": ["day_type"],
    "additionalProperties": False,
}


def prompt_hash(version: str) -> str:
    return hashlib.sha256(f"{version}\n{SYSTEM}\n{json.dumps(SCHEMA, sort_keys=True)}".encode()).hexdigest()[:16]


def cache_key(spec: LLMSpec, version: str, note: str) -> str:
    return hashlib.sha256(f"{spec.backend}|{spec.model}|{prompt_hash(version)}|{note}".encode()).hexdigest()


def parse_answer(text: str) -> str | None:
    """The label in a model's answer, or None if the answer is not valid."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    label = obj.get("day_type") if isinstance(obj, dict) else None
    return label if label in DAY_TYPES else None


class OllamaBackend:
    def __init__(self, spec: LLMSpec, base_url: str | None = None, session=None, timeout: float = 600):
        import requests

        self.spec = spec
        self.base_url = (base_url or spec.base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip(
            "/"
        )
        if not self.base_url.startswith("http"):
            self.base_url = "http://" + self.base_url
        self.session = session or requests.Session()
        self.timeout = timeout

    def ask(self, note: str, seed: int) -> str:
        body = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": note}],
            "stream": False,
            "format": SCHEMA,
            "options": {"temperature": 0, "seed": seed, "num_ctx": 2048, "num_predict": 64},
            "keep_alive": "30m",
        }
        if self.spec.think is not None:
            body["think"] = self.spec.think
        r = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def provenance(self) -> dict:
        out = {"backend": "ollama", "model": self.spec.model, "base_url": self.base_url}
        try:
            tags = self.session.get(f"{self.base_url}/api/tags", timeout=30).json().get("models", [])
            match = [m for m in tags if m.get("name") == self.spec.model or m.get("model") == self.spec.model]
            if match:
                out["digest"] = match[0].get("digest")
                out["details"] = match[0].get("details")
            out["server_version"] = self.session.get(f"{self.base_url}/api/version", timeout=30).json().get("version")
            show = self.session.post(f"{self.base_url}/api/show", json={"model": self.spec.model}, timeout=60).json()
            licence = show.get("license") or ""
            out["licence_text_start"] = licence.strip()[:300]
        except Exception as e:  # provenance is best effort; the answers are what is scored
            out["provenance_error"] = str(e)
        return out


class OpenAIBackend:
    def __init__(self, spec: LLMSpec, session=None, timeout: float = 600):
        import requests

        if not spec.base_url:
            raise ValueError(f"{spec.key}: backend 'openai' needs base_url")
        self.spec = spec
        self.session = session or requests.Session()
        self.timeout = timeout
        self.headers = {}
        if spec.api_key_env:
            key = os.environ.get(spec.api_key_env)
            if not key:
                raise ValueError(f"{spec.key}: environment variable {spec.api_key_env} is not set")
            self.headers["Authorization"] = f"Bearer {key}"

    def ask(self, note: str, seed: int) -> str:
        body = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": note}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": 64,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "day_type", "schema": SCHEMA, "strict": True},
            },
        }
        r = self.session.post(
            f"{self.spec.base_url.rstrip('/')}/chat/completions", json=body, headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def provenance(self) -> dict:
        return {"backend": "openai", "model": self.spec.model, "base_url": self.spec.base_url}


class RuleBackend:
    """A stand-in that answers from the note with fixed rules, for tests and dry runs."""

    def __init__(self, spec: LLMSpec):
        self.spec = spec

    def ask(self, note: str, seed: int) -> str:
        line = next(x for x in note.splitlines() if x.startswith("On this date:"))
        weekday = next(x for x in note.splitlines() if x.startswith("Date:")).split()[1]
        if "nationwide public holiday" in line:
            label = "sunday_like"
        elif weekday == "Saturday":
            label = "saturday_like"
        elif weekday == "Sunday":
            label = "sunday_like"
        else:
            label = "workday"
        return json.dumps({"day_type": label})

    def provenance(self) -> dict:
        return {"backend": "rule", "model": self.spec.model}


def backend_for(spec: LLMSpec):
    if spec.backend == "ollama":
        return OllamaBackend(spec)
    if spec.backend == "openai":
        return OpenAIBackend(spec)
    if spec.backend == "rule":
        return RuleBackend(spec)
    raise ValueError(f"{spec.key}: unknown backend {spec.backend!r}")


def _ask_with_retries(backend, text: str, seed: int, attempts: int = 3) -> str:
    """A call that fails in transport (server down, timeout) is retried and then stops the run:
    it says nothing about the model, so it must not be scored as a wrong answer."""
    last = None
    for attempt in range(attempts):
        try:
            return backend.ask(text, seed)
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"the language model server failed {attempts} times: {last}") from last


def labels_path(cfg: Config, llm: str) -> Path:
    return cfg.labels_dir / f"{llm}.jsonl"


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def annotate(cfg: Config, llm: str, items: list[tuple[str, date, str, str]], backend=None, log_every: int = 25) -> Path:
    """Ask one language model about every (country, date, reason, note) not yet answered.

    Appends one JSON line per answer to labels/<llm>.jsonl.
    """
    spec = cfg.llms[llm]
    backend = backend or backend_for(spec)
    path = labels_path(cfg, llm)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["country"], r["date"], r["cache_key"]) for r in read_labels(path)}
    todo = []
    for country, d, reason, text in items:
        key = cache_key(spec, cfg.prompt_version, text)
        if (country, d.isoformat(), key) not in done:
            todo.append((country, d, reason, text, key))
    print(f"{llm}: {len(items) - len(todo)} answers cached, {len(todo)} to ask")
    phash = prompt_hash(cfg.prompt_version)
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as out:
        for i, (country, d, reason, text, key) in enumerate(todo, 1):
            t0 = time.perf_counter()
            raw = _ask_with_retries(backend, text, cfg.seed)
            label = parse_answer(raw)
            row = {
                "llm": llm,
                "model": spec.model,
                "country": country,
                "date": d.isoformat(),
                "reason": reason,
                "label": label,
                "valid": label is not None,
                "raw": raw[:500],
                "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_hash": phash,
                "cache_key": key,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % log_every == 0 or i == len(todo):
                rate = (time.perf_counter() - started) / i
                print(f"{llm}: {i}/{len(todo)} answered, {rate:.1f} s per answer")
    provenance = backend.provenance()
    provenance.update({"prompt_hash": phash, "prompt_version": cfg.prompt_version, "system_prompt": SYSTEM})
    prov_path = cfg.labels_dir / f"{llm}.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def latest_labels(cfg: Config, llm: str) -> dict[tuple[str, str], dict]:
    """The answer for each (country, date) under the current prompt version and model."""
    spec = cfg.llms[llm]
    phash = prompt_hash(cfg.prompt_version)
    out = {}
    for row in read_labels(labels_path(cfg, llm)):
        if row.get("prompt_hash") == phash and row.get("model") == spec.model:
            out[(row["country"], row["date"])] = row
    return out
