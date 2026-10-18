"""Retrain a multilingual bigram model on its own output, generation after generation.

Policies (what the next model is trained on):
  replace     only the text the previous model generated
  accumulate  the original corpus plus every generated text so far
  anchor      a fresh random slice of the original corpus (share alpha)
              plus generated text (share 1 - alpha)

Decoding (how the model writes the text used for the next round):
  tau    temperature, p ** (1 / tau) renormalised in every context
  top_p  nucleus sampling threshold (Holtzman et al., 2020)

Every generation's text has the same number of tokens as the training corpus.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import bigram as B
import corpus as C

DOC_LEN = 200          # tokens per generated document and per corpus block
TEST_FRAC = 0.10       # share of corpus blocks held out for evaluation
SPLIT_SEED = 20261018  # the train/test split is fixed across all runs
LAMBDA, DELTA = 0.7, 0.5  # interpolation weight and add-delta for held-out cross-entropy


@dataclass(frozen=True)
class Config:
    name: str
    policy: str          # replace | accumulate | anchor
    alpha: float = 0.0   # share of fresh real text (anchor only)
    tau: float = 1.0
    top_p: float = 1.0


CONFIGS = [
    Config("replace", "replace"),
    Config("accumulate", "accumulate"),
    Config("anchor10", "anchor", alpha=0.10),
    Config("replace_tau08", "replace", tau=0.8),
    Config("replace_topp95", "replace", top_p=0.95),
    Config("accumulate_tau08", "accumulate", tau=0.8),
    Config("anchor10_tau08", "anchor", alpha=0.10, tau=0.8),
    Config("accumulate_topp95", "accumulate", top_p=0.95),
    Config("anchor10_topp95", "anchor", alpha=0.10, top_p=0.95),
]


class Setup:
    def __init__(self, data_dir: Path):
        self.c = C.load(data_dir)
        self.V = self.c.V
        rng = np.random.default_rng(SPLIT_SEED)
        self.train, self.test = C.split_blocks(self.c, DOC_LEN, TEST_FRAC, rng)
        self.N = sum(len(b) for b in self.train)
        self.n_docs = self.N // DOC_LEN
        self.real = B.count_docs(self.train, self.V)
        self.is_word = self.c.lang_of >= 0
        self.L = len(self.c.langs)
        # held-out evaluation material
        t = np.concatenate(self.test)
        tw = t[self.is_word[t]]
        self.test_words = tw
        self.test_word_lang = self.c.lang_of[tw]
        self.test_prev = np.concatenate([b[:-1] for b in self.test])
        self.test_next = np.concatenate([b[1:] for b in self.test])
        # original model's word types per language (reference vocabulary)
        cols0 = np.unique(self.real.cols)
        cols0 = cols0[self.is_word[cols0]]
        self.types0 = np.bincount(self.c.lang_of[cols0], minlength=self.L)

    def metrics(self, m: B.Counts) -> dict:
        V, lang_of = self.V, self.c.lang_of
        col_n = np.bincount(m.cols, weights=m.n, minlength=V)
        row_n = np.bincount(m.rows, weights=m.n, minlength=V)
        alive = col_n > 0
        words_alive = np.flatnonzero(alive & self.is_word)
        types_lang = np.bincount(lang_of[words_alive], minlength=self.L)
        # unigram over words
        pw = col_n[self.is_word]
        pw = pw[pw > 0] / pw.sum()
        gini_simpson = float(1 - np.sum(pw ** 2))
        shannon = float(-np.sum(pw * np.log2(pw)))
        # missing mass: share of held-out word tokens the model can no longer write
        miss = ~alive[self.test_words]
        miss_lang = np.bincount(self.test_word_lang, weights=miss, minlength=self.L) / \
            np.bincount(self.test_word_lang, minlength=self.L)
        # held-out cross-entropy (bits per token), interpolated bigram/unigram
        code = self.test_prev * V + self.test_next
        pos = np.searchsorted(m.codes, code)
        pos = np.minimum(pos, len(m.codes) - 1)
        c_vw = np.where(m.codes[pos] == code, m.n[pos], 0.0)
        r = row_n[self.test_prev]
        p_bi = np.divide(c_vw, r, out=np.zeros_like(c_vw), where=r > 0)
        p_uni = (col_n[self.test_next] + DELTA) / (col_n.sum() + DELTA * V)
        xent = float(-np.mean(np.log2(LAMBDA * p_bi + (1 - LAMBDA) * p_uni)))
        # entropy rate of the fitted chain
        rows = m.rows
        pr = m.n / row_n[rows]
        h_row = np.bincount(rows, weights=-pr * np.log2(pr), minlength=V)
        ent_rate = float(np.sum(row_n / row_n.sum() * h_row))
        # conditional Gini-Simpson: 1 - sum_w p(w|v)^2, averaged over contexts by weight
        gs_row = 1 - np.bincount(rows, weights=pr ** 2, minlength=V)
        cond_gs = float(np.sum(row_n / row_n.sum() * np.where(row_n > 0, gs_row, 0.0)))
        return dict(
            types=int(len(words_alive)),
            types_lang=types_lang.tolist(),
            bigram_types=int(len(m.codes)),
            hapax_bigram_share=float(m.n[m.n == 1].sum() / m.n.sum()),
            gini_simpson=gini_simpson,
            shannon_bits=shannon,
            missing_mass=float(miss.mean()),
            missing_mass_lang=miss_lang.tolist(),
            xent_bits=xent,
            entropy_rate_bits=ent_rate,
            cond_gini_simpson=cond_gs,
        )


def run(setup: Setup, cfg: Config, seed: int, T: int) -> list[dict]:
    rng = np.random.default_rng([seed, zlib.crc32(cfg.name.encode())])
    model = setup.real
    out = []
    for t in range(T + 1):
        rec = setup.metrics(model)
        s = B.make_sampler(model, tau=cfg.tau, top_p=cfg.top_p)
        n_syn = setup.n_docs if cfg.policy != "anchor" else int(round((1 - cfg.alpha) * setup.n_docs))
        gen = B.generate(s, n_syn, DOC_LEN, rng)
        words = gen.ravel()
        wl = setup.c.lang_of[words]
        wl = wl[wl >= 0]
        rec["lang_share_output"] = (np.bincount(wl, minlength=setup.L) / len(wl)).tolist()
        rec["t"] = t
        out.append(rec)
        if t == T:
            break
        g = B.count_matrix(gen, setup.V)
        if cfg.policy == "replace":
            model = g
        elif cfg.policy == "accumulate":
            model = B.add(model, g)
        elif cfg.policy == "anchor":
            k = setup.n_docs - n_syn
            pick = rng.choice(len(setup.train), size=k, replace=False)
            model = B.add(g, B.count_docs([setup.train[i] for i in pick], setup.V))
        else:
            raise ValueError(cfg.policy)
    return out


_SETUP = None


def _init(data_dir):
    global _SETUP
    _SETUP = Setup(Path(data_dir))


def _job(args):
    cfg, seed, T = args
    return cfg.name, seed, run(_SETUP, cfg, seed, T)


def main(data_dir: Path, out_dir: Path, T: int = 50, seeds: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg, s, T) for cfg in CONFIGS for s in range(seeds)]
    results = {cfg.name: {"config": asdict(cfg), "runs": [None] * seeds} for cfg in CONFIGS}
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init,
                             initargs=(str(data_dir),)) as ex:
        for name, seed, recs in ex.map(_job, jobs):
            results[name]["runs"][seed] = recs
    st = Setup(data_dir)
    meta = dict(N=st.N, n_docs=st.n_docs, doc_len=DOC_LEN, V=st.V, langs=st.c.langs,
                types0=st.types0.tolist(), T=T, seeds=seeds, lam=LAMBDA, delta=DELTA)
    (out_dir / "bigram_loop.json").write_text(json.dumps({"meta": meta, "results": results}))
    return meta, results


if __name__ == "__main__":
    import sys
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    main(data, Path("results"))
