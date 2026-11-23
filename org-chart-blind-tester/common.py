"""Shared pieces of the blind-tester experiment: configuration, questions, prompts, answer parsing and
the client for a local Ollama server.

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
import os
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
    """config.json, or the file named by BLIND_TESTER_CONFIG (the tests use a small synthetic one)."""
    path = path or Path(os.environ.get("BLIND_TESTER_CONFIG", HERE / "config.json"))
    return json.loads(Path(path).read_text())


def model_specs(cfg: dict, keys=None) -> list[dict]:
    specs = [dict(m, shards=m.get("shards", cfg["shards"])) for m in cfg["models"]]
    return [s for s in specs if not keys or s["key"] in keys]


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
    """Download the MMLU test split at the pinned revision, check its digest, and return every row in
    file order. The file is cached in data/, never committed."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    m = cfg["mmlu"]
    path = Path(hf_hub_download(m["repo_id"], m["filename"], repo_type="dataset", revision=m["revision"],
                                cache_dir=str(cache)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if m.get("sha256") and digest != m["sha256"]:
        raise RuntimeError(f"MMLU file digest {digest} differs from the pinned {m['sha256']}")
    table = pq.read_table(path).to_pylist()
    rows = [Question(i, r["subject"], r["question"], tuple(r["choices"]), int(r["answer"]))
            for i, r in enumerate(table)]
    return rows, {"repo_id": m["repo_id"], "filename": m["filename"], "revision": m["revision"],
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
    """The sampled questions in a fixed order: a seeded random draw without replacement."""
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(rows))[: cfg["n_questions"]]
    return [rows[i] for i in idx]


def in_check_set(qid: int, cfg: dict) -> bool:
    """A fixed, seed-dependent share of questions gets the anchored and warned roles whatever the
    blind tester said. Hash-based so that it does not depend on run order or sharding."""
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
# Ollama client (native /api/chat, which honours `think` and a fixed seed)
# ---------------------------------------------------------------------------------------------------

class Ollama:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 900.0, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep = sleep

    def _request(self, method: str, path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def chat(self, spec: dict, msgs: list[dict], cfg: dict, attempts: int = 4) -> dict:
        body = {"model": spec["model"], "messages": msgs, "stream": False, "keep_alive": "60m",
                "options": {"temperature": cfg["temperature"], "seed": cfg["seed"],
                            "num_predict": cfg["max_tokens"], "num_ctx": cfg["num_ctx"]}}
        if spec.get("think") is not None:
            body["think"] = spec["think"]
        delay = 5.0
        for attempt in range(attempts):
            try:
                r = self._request("POST", "/api/chat", body)
                return {"text": (r.get("message") or {}).get("content") or "",
                        "finish_reason": r.get("done_reason"), "model_reported": r.get("model"),
                        "usage": {"prompt_tokens": r.get("prompt_eval_count"),
                                  "completion_tokens": r.get("eval_count"),
                                  "seconds": round((r.get("total_duration") or 0) / 1e9, 2)}}
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code >= 500 and attempt < attempts - 1:
                    self.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} from Ollama for {spec['model']}: {msg}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < attempts - 1:
                    self.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"cannot reach Ollama for {spec['model']}: {e}") from None
        raise RuntimeError("unreachable")

    def provenance(self, spec: dict) -> dict:
        out = {"model": spec["model"], "base_url": self.base_url}
        try:
            out["server_version"] = self._request("GET", "/api/version").get("version")
            tags = self._request("GET", "/api/tags").get("models", [])
            hit = [m for m in tags if spec["model"] in (m.get("name"), m.get("model"))]
            if hit:
                out["digest"] = hit[0].get("digest")
                out["size"] = hit[0].get("size")
                out["details"] = hit[0].get("details")
            show = self._request("POST", "/api/show", {"model": spec["model"]})
            out["licence_start"] = (show.get("license") or "").strip()[:300]
        except Exception as e:  # noqa: BLE001 - provenance is best effort; the answers are what is scored
            out["provenance_error"] = repr(e)[:300]
        return out


# ---------------------------------------------------------------------------------------------------
# Runs on disk: runs/<key>.shard<i>of<k>.jsonl, one JSON line per call
# ---------------------------------------------------------------------------------------------------

def shard_path(runs_dir: Path, key: str, shard: int, shards: int) -> Path:
    return runs_dir / f"{key}.shard{shard}of{shards}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_model_runs(runs_dir: Path, key: str) -> list[dict]:
    recs = []
    for p in sorted(runs_dir.glob(f"{key}.shard*.jsonl")):
        recs.extend(read_jsonl(p))
    return recs
