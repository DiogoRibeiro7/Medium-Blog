"""Shared pieces of the blind-tester experiment: configuration, questions, prompts, answer parsing and
the client for the GitHub Models inference API.

The experiment puts the same multiple-choice question to a model in four roles:

  coder     answers the question; its answer is what the team's function would return.
  blind     a test engineer decides what the unit test should expect before seeing the coder's work.
  anchored  a test engineer decides what the test should expect after reading the coder's answer
            (MetaGPT's order: the tester sees the code first).
  warned    as anchored, but told that the coder may be wrong and to work the answer out itself.

The coder and the blind tester answer every sampled question. The anchored and warned testers answer
every question on which the blind tester disagreed with the coder, plus a fixed random share of all
questions (the check set), which lets the analysis test the assumption that seeing the coder's answer
never turns an agreement into a disagreement.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LETTERS = "ABCD"
ROLES = ("coder", "blind", "anchored", "warned")


def load_config(path: Path | None = None) -> dict:
    """config.json, or the file named by BLIND_TESTER_CONFIG (the tests use a fast synthetic one)."""
    import os
    path = path or Path(os.environ.get("BLIND_TESTER_CONFIG", HERE / "config.json"))
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    qid: int          # row index in the MMLU test file
    subject: str
    question: str
    choices: tuple    # four option texts
    answer: int       # index of the keyed answer, 0..3


def fetch_mmlu(cfg: dict, cache: Path = HERE / "data") -> tuple[list[Question], dict]:
    """Download the MMLU test split at the configured revision and return every row in file order,
    with provenance (resolved commit, file SHA-256). The file is cached in data/, never committed."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    m = cfg["mmlu"]
    info = HfApi().dataset_info(m["repo_id"], revision=m["revision"])
    sha = info.sha
    path = Path(hf_hub_download(m["repo_id"], m["filename"], repo_type="dataset", revision=sha,
                                cache_dir=str(cache)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table = pq.read_table(path).to_pylist()
    rows = [Question(i, r["subject"], r["question"], tuple(r["choices"]), int(r["answer"]))
            for i, r in enumerate(table)]
    return rows, {"repo_id": m["repo_id"], "filename": m["filename"], "revision": sha,
                  "sha256": digest, "rows": len(rows)}


def load_questions(cfg: dict, questions_file: Path | None = None) -> tuple[list[Question], dict]:
    """MMLU by default; tests pass a JSON file of synthetic questions with the same fields."""
    if questions_file is None:
        return fetch_mmlu(cfg)
    raw = json.loads(Path(questions_file).read_text())
    rows = [Question(int(r["qid"]), r["subject"], r["question"], tuple(r["choices"]), int(r["answer"]))
            for r in raw]
    return rows, {"source": str(questions_file), "rows": len(rows)}


def sample_order(rows: list[Question], cfg: dict) -> list[Question]:
    """The questions in the order in which every model works through them: a seeded random draw
    without replacement. A model with a smaller quota answers a prefix of the same order, so its
    questions are a random subset of the larger models' questions."""
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(rows))[: cfg["n_questions"]]
    return [rows[i] for i in idx]


def in_check_set(qid: int, cfg: dict) -> bool:
    """A fixed, seed-dependent share of questions gets the anchored and warned roles whatever the
    blind tester said. Hash-based so that it does not depend on run order."""
    h = hashlib.sha256(f"{cfg['seed']}:{qid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < cfg["check_share"]


# ---------------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------------

SYSTEM = {
    "coder": (
        "You are the engineer in a small software team. The team is building a function that answers "
        "the multiple-choice question below by returning one option. Your job is to decide which option "
        "the function returns."
    ),
    "blind": (
        "You are the test engineer in a small software team. The engineer will build a function that "
        "answers the multiple-choice question below by returning one option. You have not seen the "
        "engineer's work. Decide which option the unit test should expect the function to return."
    ),
    "anchored": (
        "You are the test engineer in a small software team. The engineer has built a function that "
        "answers the multiple-choice question below by returning one option; the engineer's answer is "
        "shown below the question. Decide which option the unit test should expect the function to return."
    ),
    "warned": (
        "You are the test engineer in a small software team. The engineer has built a function that "
        "answers the multiple-choice question below by returning one option; the engineer's answer is "
        "shown below the question. The engineer may be wrong. Work out the answer yourself before you "
        "decide; if you disagree with the engineer, the test should expect your answer. Decide which "
        "option the unit test should expect the function to return."
    ),
}

INSTRUCTION = ("Give at most three short sentences of reasoning, then end with a line of the form "
               "'ANSWER: X', where X is A, B, C or D.")


def format_question(q: Question) -> str:
    subject = q.subject.replace("_", " ")
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q.choices))
    return f"Question ({subject}):\n{q.question}\n\n{opts}"


def messages(role: str, q: Question, coder_text: str | None = None) -> list[dict]:
    body = format_question(q)
    if role in ("anchored", "warned"):
        if coder_text is None:
            raise ValueError(f"role {role} needs the coder's response")
        body += f"\n\nThe engineer's answer:\n{coder_text.strip()}"
    return [{"role": "system", "content": SYSTEM[role]},
            {"role": "user", "content": f"{body}\n\n{INSTRUCTION}"}]


_ANSWER = re.compile(r"ANSWER\s*[:\-]\s*\**\s*\(?\s*([ABCD])\b", re.I)


def parse_answer(text: str | None) -> str | None:
    """The letter on the last 'ANSWER: X' line, or None if there is none."""
    if not text:
        return None
    found = _ANSWER.findall(text)
    return found[-1].upper() if found else None


# ---------------------------------------------------------------------------------------------------
# GitHub Models client
# ---------------------------------------------------------------------------------------------------

class QuotaExhausted(Exception):
    """The model's daily request limit is used up; try again in a later run."""


class Rejected(Exception):
    """The service refused this prompt for good (e.g. a content filter). Recorded, not retried."""


class Client:
    def __init__(self, endpoint: str, token: str, timeout: float = 120.0, sleep=time.sleep):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.sleep = sleep

    def _request(self, method: str, path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.endpoint + path, data=data, method=method, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def catalog(self) -> list[dict]:
        return self._request("GET", "/catalog/models")

    def chat(self, model: str, msgs: list[dict], temperature: float, max_tokens: int,
             long_wait: float = 300.0, attempts: int = 6) -> dict:
        """One chat completion. Short 429s (per-minute limits) and transient errors are retried;
        a 429 asking to wait longer than `long_wait` seconds means the daily quota is gone."""
        payload = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        delay = 5.0
        for attempt in range(attempts):
            try:
                return self._request("POST", "/inference/chat/completions", payload)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code == 429:
                    wait = _retry_after(e.headers, body)
                    if wait is not None and wait > long_wait:
                        raise QuotaExhausted(f"{model}: wait {wait:.0f}s: {body[:300]}") from None
                    self.sleep((wait or delay) + 1)
                    delay = min(delay * 2, 120)
                    continue
                if e.code == 400 and ("content_filter" in body or "content management policy" in body):
                    raise Rejected(f"content_filter: {body[:300]}") from None
                if e.code in (500, 502, 503, 504) and attempt < attempts - 1:
                    self.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"HTTP {e.code} from {model}: {body[:500]}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < attempts - 1:
                    self.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"network error calling {model}: {e}") from None
        raise RuntimeError(f"{model}: still rate limited after {attempts} attempts")


def _retry_after(headers, body: str) -> float | None:
    for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-timeremaining"):
        v = headers.get(name) if headers is not None else None
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    m = re.search(r"wait (\d+) seconds", body)
    return float(m.group(1)) if m else None


def resolve_models(cfg: dict, catalog: list[dict]) -> list[dict]:
    """Match each configured model to the first candidate present in the catalog (case-insensitive)
    and attach its rate-limit tier. Models with no candidate in the catalog are returned with id None."""
    by_id = {m["id"].lower(): m for m in catalog}
    out = []
    for spec in cfg["models"]:
        hit = next((by_id[c.lower()] for c in spec["candidates"] if c.lower() in by_id), None)
        tier = (hit or {}).get("rate_limit_tier", "low")
        tier = tier if tier in cfg["tiers"] else "high"
        out.append({**spec, "id": hit["id"] if hit else None, "tier": tier,
                    "n_questions": cfg["tiers"][tier]["n_questions"],
                    "rpm": cfg["tiers"][tier]["rpm"], "per_day": cfg["tiers"][tier]["per_day"]})
    return out


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
