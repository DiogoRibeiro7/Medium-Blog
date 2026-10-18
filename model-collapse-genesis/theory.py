"""Closed-form results for a categorical (unigram) model retrained on its own samples,
each checked against Monte Carlo.

Notation: p_t is the model at generation t, N the sample size per generation,
H(p) = 1 - sum_i p_i^2 the Gini-Simpson index (the probability that two
independent draws are different words).

1. replace      E[H(p_t)] = H(p_0) (1 - 1/N)^t                 (Wright-Fisher)
2. one step     E[types kept] = sum_i 1 - (1 - p_i)^N,
                E[mass lost] = sum_i p_i (1 - p_i)^N  ~  E[n_1 / N] (Good-Turing)
3. accumulate   E[H(p_inf)] / H(p_0) = prod_{k>=2} (1 - 1/(N k^2))
                                     = sin(pi z) / (pi z (1 - z^2)),  z = 1/sqrt(N)
4. anchor       stationary E[H] / H(p_0) = 1 - 1/(N a (2 - a) + 1 - a),
                a = share of each generation drawn fresh from p_0
5. temperature  (no sampling noise) p_t proportional to p_0 ** (tau ** -t)
6. nucleus      (no sampling noise) iterating top-p truncation stops only when
                every surviving word has probability > 1 - p, so at most
                ceil(1/(1 - p)) - 1 words survive
7. branching    at a context shared by several languages, temperature tau moves
                the mass of language L to  m_L^b exp((1 - b) H_b(q_L)), b = 1/tau,
                with H_b the Renyi entropy of L's continuation distribution q_L
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import corpus as C


def gini_simpson(p):
    return 1.0 - np.sum(p ** 2, axis=-1)


def draw(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial(n, P[r]) for every row r at once, by inverse-CDF sampling.

    Equivalent to rng.multinomial(n, P) but much faster when n is small
    relative to the number of categories.
    """
    reps, K = P.shape
    cdf = np.cumsum(P, axis=1)
    cdf /= cdf[:, -1:]
    cdf[:, -1] = 1.0
    cdf += np.arange(reps)[:, None]
    u = 1e-9 + rng.random((reps, n)) * (1 - 2e-9) + np.arange(reps)[:, None]
    idx = np.searchsorted(cdf.ravel(), u.ravel())
    return np.bincount(idx, minlength=reps * K).reshape(reps, K).astype(float)


def unigram(c: C.Corpus, code: str) -> np.ndarray:
    s = c.streams[code]
    w = s[c.lang_of[s] >= 0]
    _, n = np.unique(w, return_counts=True)
    return np.sort(n)[::-1].astype(float)


def check_replace(p0, N, T, reps, rng):
    p = np.tile(p0, (reps, 1))
    h0 = gini_simpson(p0)
    out = {}
    for t in range(1, T + 1):
        p = draw(p, N, rng) / N
        if t in (10, 100, 250, 500, 1000):
            h = gini_simpson(p)
            out[t] = dict(mc=float(h.mean() / h0), se=float(h.std(ddof=1) / np.sqrt(reps) / h0),
                          formula=float((1 - 1 / N) ** t))
    return out


def check_one_step(counts, reps, rng):
    N = int(counts.sum())
    p = counts / N
    kept_f = float(np.sum(1 - (1 - p) ** N))
    lost_f = float(np.sum(p * (1 - p) ** N))
    gt_f = float(np.sum(p * (1 - p) ** (N - 1)))
    kept, lost, gt = [], [], []
    for _ in range(reps):
        x = rng.multinomial(N, p)
        kept.append((x > 0).sum())
        lost.append(p[x == 0].sum())
        gt.append((x == 1).sum() / N)
    n1 = int((counts == 1).sum())
    return dict(N=N, types=len(counts), hapax=n1, hapax_share_types=n1 / len(counts),
                good_turing_n1_over_N=n1 / N,
                kept_formula=kept_f, kept_mc=float(np.mean(kept)),
                mass_lost_formula=lost_f, mass_lost_mc=float(np.mean(lost)),
                n1_over_N_next_formula=gt_f, n1_over_N_next_mc=float(np.mean(gt)),
                hapax_survival_approx=float(1 - np.exp(-1)))


def euler_product(N):
    z = 1 / np.sqrt(N)
    return float(np.sin(np.pi * z) / (np.pi * z * (1 - z * z)))


def check_accumulate(p0, N, T, reps, rng):
    cnt = np.tile(N * p0, (reps, 1))
    for _ in range(T):
        p = cnt / cnt.sum(axis=1, keepdims=True)
        cnt = cnt + draw(p, N, rng)
    p = cnt / cnt.sum(axis=1, keepdims=True)
    h = gini_simpson(p) / gini_simpson(p0)
    return dict(N=N, T=T, mc=float(h.mean()), se=float(h.std(ddof=1) / np.sqrt(reps)),
                formula=euler_product(N),
                replace_same_T=float((1 - 1 / N) ** T))


def check_anchor(p0, N, a, T, burn, reps, rng):
    k_real = int(round(a * N))
    a_eff = k_real / N
    p = np.tile(p0, (reps, 1))
    hs = []
    for t in range(T):
        real = draw(np.broadcast_to(p0, p.shape), k_real, rng)
        syn = draw(p, N - k_real, rng)
        p = (real + syn) / N
        if t >= burn:
            hs.append(gini_simpson(p))
    h = np.concatenate(hs) / gini_simpson(p0)
    formula = 1 - 1 / (N * a_eff * (2 - a_eff) + 1 - a_eff)
    return dict(N=N, alpha=a_eff, mc=float(h.mean()), formula=float(formula),
                replace_after_T=float((1 - 1 / N) ** T))


def temperature_path(p0, tau, T):
    lp = np.log(p0)
    out = []
    for t in range(T + 1):
        e = tau ** (-t)
        q = np.exp(e * (lp - lp.max()))
        q /= q.sum()
        q = q[q > 0]
        out.append(dict(t=t, eff_types=float(np.exp(-np.sum(q * np.log(q)))),
                        top_word_share=float(q.max())))
    return out


def iterate_nucleus(p0, top_p, max_steps=10_000):
    p = np.sort(p0)[::-1]
    for step in range(max_steps):
        cs = np.cumsum(p)
        before = cs - p
        keep = before < top_p * (1 - 1e-12)
        if keep.all():
            return dict(top_p=top_p, survivors=int(len(p)), steps=step,
                        bound=int(np.ceil(1 / (1 - top_p)) - 1),
                        smallest=float(p[-1]))
        p = p[keep]
        p = p / p.sum()
    raise RuntimeError("no convergence")


def branching(c: C.Corpus, context: str, tau: float):
    """Language masses after the context `context` before and after temperature tau."""
    b = 1 / tau
    ctx = c.vocab.index(f"pun|{context}")
    m, H, mass_after = {}, {}, {}
    for li, code in enumerate(c.langs):
        s = c.streams[code]
        nxt = s[1:][s[:-1] == ctx]
        _, n = np.unique(nxt, return_counts=True)
        m[code] = float(n.sum())
        q = n / n.sum()
        H[code] = float(np.log(np.sum(q ** b)) / (1 - b))       # Renyi entropy, nats
        mass_after[code] = float(n.sum() ** b * np.sum(q ** b))  # proportional to m^b exp((1-b) H_b)
    tot, tot2 = sum(m.values()), sum(mass_after.values())
    return {code: dict(share_before=m[code] / tot, share_after=mass_after[code] / tot2,
                       renyi_bits=H[code] / np.log(2)) for code in c.langs}


def nucleus_cut(c: C.Corpus, context: str, top_p: float):
    """Share of each language's continuation mass that nucleus sampling removes."""
    ctx = c.vocab.index(f"pun|{context}")
    nxt_all, lang_all = [], []
    for li, code in enumerate(c.langs):
        s = c.streams[code]
        nxt_all.append(s[1:][s[:-1] == ctx])
    nxt = np.concatenate(nxt_all)
    u, n = np.unique(nxt, return_counts=True)
    order = np.argsort(-n, kind="stable")
    u, n = u[order], n[order].astype(float)
    p = n / n.sum()
    keep = (np.cumsum(p) - p) < top_p
    lang = c.lang_of[u]
    out = {}
    for li, code in enumerate(c.langs):
        sel = lang == li
        out[code] = dict(mass=float(p[sel].sum()),
                         cut_share_of_own_mass=float(p[sel & ~keep].sum() / p[sel].sum()))
    return out


def main(data_dir: Path, out_dir: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    c = C.load(data_dir)
    en = unigram(c, "en")
    p_en = en / en.sum()
    res = {}
    res["replace_N1000"] = check_replace(p_en, 1000, 1000, 400, rng)
    res["one_step"] = {code: check_one_step(unigram(c, code), 200, rng) for code in c.langs}
    res["accumulate"] = [check_accumulate(p_en, N, 400, 2000, rng) for N in (4, 25, 100)]
    res["anchor"] = [check_anchor(p_en, 50, a, 400, 200, 2000, rng) for a in (0.1, 0.3)]
    res["temperature_en_tau08"] = temperature_path(p_en, 0.8, 20)
    pooled = np.concatenate([unigram(c, code) for code in c.langs])
    res["nucleus"] = {code: [iterate_nucleus(unigram(c, code) / unigram(c, code).sum(), tp)
                             for tp in (0.9, 0.95)] for code in c.langs}
    res["nucleus"]["pooled"] = [iterate_nucleus(pooled / pooled.sum(), tp) for tp in (0.9, 0.95)]
    res["branching_tau08"] = {ctx: branching(c, ctx, 0.8) for ctx in (".", ",", ";")}
    res["nucleus_cut_095"] = {ctx: nucleus_cut(c, ctx, 0.95) for ctx in (".", ",")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "theory_checks.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    r = main(data, Path("results"))
    print(json.dumps(r, indent=1))
