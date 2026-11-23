"""A stand-in for the GitHub Models API, with simulated agents whose parameters are known, so that the
runner can be tested against quotas, refusals and restarts, and the analysis against a ground truth.

Each simulated model answers from a one-factor-plus-model-factor probit:
    Z = sqrt(rho_cross) F_item + sqrt(rho_same - rho_cross) G_item,model + sqrt(1 - rho_same) e
and is wrong when Z exceeds its threshold. Two roles of the same model then have error correlation
rho_same; roles of different models have rho_cross. An anchored tester adopts the coder's answer with
probability a, otherwise it gives the answer it would have given blind.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy import stats

LETTERS = "ABCD"

MODELS = {
    # id: (tier, q, rho_same, rho_cross, a_anchored, a_warned)
    "openai/gpt-4o-mini": ("low", 0.20, 0.80, 0.40, 0.90, 0.50),
    "openai/gpt-4o": ("high", 0.15, 0.80, 0.40, 0.95, 0.60),
    "meta/Llama-3.3-70B-Instruct": ("low", 0.30, 0.70, 0.40, 0.80, 0.40),
    "mistral-ai/mistral-small-2503": ("low", 0.25, 0.75, 0.40, 0.85, 0.45),
}
ATTRACT = 0.6          # probability that an erring agent picks the item's most tempting wrong option
UNPARSEABLE = 0.01     # share of responses with no 'ANSWER:' line
FILTERED_EVERY = 97    # every 97th question is refused by the content filter (coder role)


def unit(*parts) -> np.random.Generator:
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def simulated_answer(model: str, qid: int, role: str, key: int) -> int:
    tier, q, rs, rc, *_ = MODELS[model]
    f = unit("F", qid).standard_normal()
    g = unit("G", qid, model).standard_normal()
    e = unit("e", qid, model, role).standard_normal()
    z = np.sqrt(rc) * f + np.sqrt(rs - rc) * g + np.sqrt(1 - rs) * e
    if z <= stats.norm.ppf(1 - q):
        return key
    wrongs = [i for i in range(4) if i != key]
    attractor = wrongs[unit("attractor", qid).integers(0, 3)]
    r = unit("pick", qid, model, role)
    if r.random() < ATTRACT:
        return attractor
    rest = [w for w in wrongs if w != attractor]
    return rest[r.integers(0, 2)]


class FakeModels:
    def __init__(self, questions: list[dict], system_prompts: dict, quota: int):
        self.key = {int(r["qid"]): int(r["answer"]) for r in questions}
        self.system = {v: k for k, v in system_prompts.items()}
        self.quota = quota
        self.used = {m: 0 for m in MODELS}
        self.minute_429_sent = set()
        self.lock = threading.Lock()
        self.server = None

    def new_day(self):
        with self.lock:
            self.used = {m: 0 for m in MODELS}

    def respond(self, payload: dict) -> tuple[int, dict, dict]:
        model = payload["model"]
        if model not in MODELS:
            return 404, {}, {"error": {"code": "unknown_model"}}
        msgs = payload["messages"]
        role = self.system[msgs[0]["content"]]
        user = msgs[1]["content"]
        qid = int(re.search(r"Synthetic question (\d+)", user).group(1))
        with self.lock:
            if self.used[model] == 3 and model not in self.minute_429_sent:
                self.minute_429_sent.add(model)
                return 429, {"Retry-After": "0"}, {"error": {"message": "Rate limit per minute"}}
            if self.used[model] >= self.quota:
                return 429, {"Retry-After": "86000"}, {"error": {"code": "RateLimitReached", "message":
                        f"Rate limit of {self.quota} per 86400s exceeded for UserByModelByDay. "
                        "Please wait 86000 seconds before retrying."}}
            self.used[model] += 1
        if role == "coder" and qid % FILTERED_EVERY == 0:
            return 400, {}, {"error": {"code": "content_filter", "message": "filtered"}}
        key = self.key[qid]
        tier, q, rs, rc, a_anch, a_warn = MODELS[model]
        if role in ("anchored", "warned"):
            seen = re.findall(r"ANSWER:\s*([ABCD])", user.split("The engineer's answer:")[1])
            defer = unit("defer", qid, model, role).random() < (a_anch if role == "anchored" else a_warn)
            if seen and defer:
                ans = LETTERS.index(seen[-1])
            else:
                ans = simulated_answer(model, qid, "blind", key)
        else:
            ans = simulated_answer(model, qid, role, key)
        if unit("unparseable", qid, model, role).random() < UNPARSEABLE:
            text = "I think it is probably the second option, but I am not sure."
        else:
            text = f"Some short reasoning about the question.\nANSWER: {LETTERS[ans]}"
        return 200, {}, {"model": model + "-sim", "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112}}

    def catalog(self) -> list[dict]:
        return [{"id": m, "rate_limit_tier": t[0]} for m, t in MODELS.items()]

    def start(self) -> str:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, headers, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/catalog/models":
                    self._send(200, {}, fake.catalog())
                else:
                    self._send(404, {}, {})

            def do_POST(self):
                assert self.headers["Authorization"].startswith("Bearer ")
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/inference/chat/completions":
                    self._send(*fake.respond(payload))
                else:
                    self._send(404, {}, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        if self.server:
            self.server.shutdown()
