"""Decode the 144 prompts with one model under every rule, and score the human continuations.

Runs in GitHub Actions, one model per job (see .github/workflows/decoding-loops.yml), because the
models are downloaded from the Hugging Face Hub at pinned revisions:

    python generate_runs.py --model pythia-410m     # writes results/gen_pythia-410m.json.gz
    python generate_runs.py --smoke                 # tiny random model, no download, for CI tests
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

import lm_backend as B
import loops_core as L

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 20261126

# key -> (Hub name, revision, parameters in millions, family, batch size). Pythia's final checkpoint is
# the branch step143000; the commit each revision resolves to is recorded in the output.
MODELS = {
    "pythia-70m": ("EleutherAI/pythia-70m", "step143000", 70, "Pythia", 48),
    "pythia-160m": ("EleutherAI/pythia-160m", "step143000", 160, "Pythia", 48),
    "pythia-410m": ("EleutherAI/pythia-410m", "step143000", 410, "Pythia", 36),
    "pythia-1b": ("EleutherAI/pythia-1b", "step143000", 1000, "Pythia", 24),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", "step143000", 1400, "Pythia", 24),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "main", 135, "SmolLM2", 48),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "main", 360, "SmolLM2", 36),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "main", 1700, "SmolLM2", 24),
}


def resolved_sha(name, revision):
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name, revision=revision).sha
    except Exception as e:  # pragma: no cover - network
        return f"unresolved ({type(e).__name__})"


def smoke_model():
    """A tiny random GPT-NeoX and a byte-level stand-in tokenizer: exercises the whole path offline."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=259, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64
    )
    model = B.HFModel(GPTNeoXForCausalLM(cfg))

    class Bytes:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))

        def decode(self, ids):
            return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    model.tok = Bytes()
    return model


def run(key, smoke=False):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    passages = L.make_prompts()
    if smoke:
        model, name, revision, params, family, batch = smoke_model(), "random-tiny-neox", "-", 0, "smoke", 3
        passages, n_new = passages[:: len(passages) // 6][:6], 32
        sha = "-"
    else:
        name, revision, params, family, batch = MODELS[key]
        model = B.HFModel.load(name, revision)
        n_new = L.N_NEW
        sha = resolved_sha(name, revision)
    tok = model.tok
    prompts = [tok.encode(p.prompt, add_special_tokens=False) for p in passages]
    refs = [tok.encode(p.reference, add_special_tokens=False)[:n_new] for p in passages]
    out = {
        "model": key if not smoke else "smoke",
        "name": name,
        "revision": revision,
        "sha": sha,
        "params_m": params,
        "family": family,
        "n_new": n_new,
        "torch": torch.__version__,
        "passages": [{"book": p.book, "index": p.index} for p in passages],
        "prompt_tokens": prompts,
        "rules": {},
        "seconds": {},
    }
    for r_i, (rname, rule) in enumerate(L.RULES.items()):
        t0 = time.time()
        toks, lps, mgs = [], [], []
        for b_i in range(0, len(prompts), batch):
            rng = np.random.default_rng([SEED, r_i, b_i])
            d = L.decode(model, prompts[b_i : b_i + batch], rule, n_new, rng)
            toks += d.tokens.tolist()
            lps += np.round(d.logprob, 5).tolist()
            mgs += np.round(d.margin, 5).tolist()
        out["rules"][rname] = {
            "tokens": toks,
            "logprob": lps,
            "margin": mgs,
            "text": [tok.decode(t) for t in toks],
        }
        out["seconds"][rname] = round(time.time() - t0, 1)
        print(f"{out['model']} {rname}: {out['seconds'][rname]} s", flush=True)
    t0 = time.time()
    lp, mg = [], []
    for b_i in range(0, len(prompts), batch):
        a, m = model.score(prompts[b_i : b_i + batch], refs[b_i : b_i + batch])
        for row_a, row_m, r in zip(a, m, refs[b_i : b_i + batch], strict=True):
            lp.append(np.round(row_a[: len(r)], 5).tolist())
            mg.append(np.round(row_m[: len(r)], 5).tolist())
    out["reference"] = {"tokens": refs, "logprob": lp, "margin": mg, "text": [tok.decode(r) for r in refs]}
    out["seconds"]["reference"] = round(time.time() - t0, 1)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"gen_{out['model']}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke and not a.model:
        ap.error("give --model or --smoke")
    run(a.model, smoke=a.smoke)
