"""A stand-in for a local Ollama server, with simulated agents whose parameters are known, so that the
runner can be tested against interruptions, restarts and server errors, and the analysis against a
ground truth.

Each simulated model answers from a probit with a shared item factor, a family factor and a model factor:
    Z = sqrt(rho_cross) F_item + sqrt(rho_family - rho_cross) H_item,family
        + sqrt(rho_same - rho_family) G_item,model + sqrt(1 - rho_same) e
and is wrong when Z exceeds its threshold. Two roles of the same model then have error correlation
rho_same, two models of one family rho_family, and models of different families rho_cross. An anchored
tester adopts the coder's answer with probability a, otherwise it gives the answer it would have given blind.
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
RHO_CROSS = 0.40
MODELS = {
    # tag: (family, q, rho_same, rho_family, a_anchored, a_warned)
    "llama3.1:8b": ("Llama", 0.30, 0.80, RHO_CROSS, 0.90, 0.50),
    "qwen3:8b": ("Qwen", 0.22, 0.75, 0.60, 0.80, 0.40),
    "qwen3.5:9b": ("Qwen", 0.18, 0.75, 0.60, 0.85, 0.45),
    "ministral-3:8b": ("Ministral", 0.28, 0.70, RHO_CROSS, 0.95, 0.60),
}
ATTRACT = 0.6          # probability that an erring agent picks the item's most tempting wrong option
UNPARSEABLE = 0.01     # share of responses with no 'ANSWER:' line
FLAKY = "qwen3:8b"     # returns one HTTP 500 at its first call, to exercise the retry


def unit(*parts) -> np.random.Generator:
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def simulated_answer(model: str, qid: int, role: str, key: int) -> int:
    family, q, rs, rf, *_ = MODELS[model]
    f = unit("F", qid).standard_normal()
    h = unit("H", qid, family).standard_normal()
    g = unit("G", qid, model).standard_normal()
    e = unit("e", qid, model, role).standard_normal()
    z = np.sqrt(RHO_CROSS) * f + np.sqrt(rf - RHO_CROSS) * h + np.sqrt(rs - rf) * g + np.sqrt(1 - rs) * e
    if z <= stats.norm.ppf(1 - q):
        return key
    wrongs = [i for i in range(4) if i != key]
    attractor = wrongs[unit("attractor", qid).integers(0, 3)]
    r = unit("pick", qid, model, role)
    if r.random() < ATTRACT:
        return attractor
    rest = [w for w in wrongs if w != attractor]
    return rest[r.integers(0, 2)]


class FakeOllama:
    def __init__(self, questions: list[dict], system_prompts: dict):
        self.key = {int(r["qid"]): int(r["answer"]) for r in questions}
        self.system = {v: k for k, v in system_prompts.items()}
        self.calls = {m: 0 for m in MODELS}
        self.flaked = False
        self.lock = threading.Lock()
        self.server = None

    def chat(self, payload: dict) -> tuple[int, dict]:
        model = payload["model"]
        if model not in MODELS:
            return 404, {"error": f"model '{model}' not found"}
        assert payload["stream"] is False and payload["options"]["temperature"] == 0
        with self.lock:
            self.calls[model] += 1
            if model == FLAKY and not self.flaked:
                self.flaked = True
                return 500, {"error": "llama runner process has terminated"}
        msgs = payload["messages"]
        role = self.system[msgs[0]["content"]]
        user = msgs[1]["content"]
        qid = int(re.search(r"Synthetic question (\d+)", user).group(1))
        key = self.key[qid]
        family, q, rs, rf, a_anch, a_warn = MODELS[model]
        if role in ("anchored", "warned"):
            seen = re.findall(r"ANSWER:\s*([ABCD])", user.split("The engineer's answer:")[1])
            defer = unit("defer", qid, model, role).random() < (a_anch if role == "anchored" else a_warn)
            ans = LETTERS.index(seen[-1]) if (seen and defer) else simulated_answer(model, qid, "blind", key)
        else:
            ans = simulated_answer(model, qid, role, key)
        if unit("unparseable", qid, model, role).random() < UNPARSEABLE:
            text = "I think it is probably the second option, but I am not sure."
        else:
            text = f"Some short reasoning about the question.\nANSWER: {LETTERS[ans]}"
        return 200, {"model": model, "message": {"role": "assistant", "content": text}, "done": True,
                     "done_reason": "stop", "prompt_eval_count": 180, "eval_count": 14,
                     "total_duration": 1_200_000_000}

    def start(self) -> str:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m, "model": m, "digest": f"sha256:fake-{i}", "size": 1,
                                                 "details": {"quantization_level": "Q4_K_M"}}
                                                for i, m in enumerate(MODELS)]})
                else:
                    self._send(404, {})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/chat":
                    self._send(*fake.chat(payload))
                elif self.path == "/api/show":
                    self._send(200, {"license": "fake licence"})
                else:
                    self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        if self.server:
            self.server.shutdown()
