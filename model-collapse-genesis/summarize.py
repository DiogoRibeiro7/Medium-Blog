"""Print every number quoted in the article from the saved results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import corpus as C
import experiment as E


def arr(res, name, key, idx=None):
    runs = res[name]["runs"]
    return np.asarray([[rec[key] if idx is None else rec[key][idx] for rec in r] for r in runs],
                      dtype=float)


def fmt(a, t, pct=False, d=1):
    m, lo, hi = a[:, t].mean(), a[:, t].min(), a[:, t].max()
    if pct:
        return f"{100 * m:.{d}f}% [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"
    return f"{m:,.{d}f} [{lo:,.{d}f}, {hi:,.{d}f}]"


def euler_R(n):
    n = np.asarray(n, dtype=float)
    z = 1 / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sin(np.pi * z) / (np.pi * z * (1 - z * z))
    return np.where(n > 1, r, 0.5)


def main(data_dir: Path, results: Path):
    d = json.loads((results / "bigram_loop.json").read_text())
    meta, res = d["meta"], d["results"]
    tp = results / "theory_checks.json"
    th = json.loads(tp.read_text()) if tp.exists() else None
    langs = meta["langs"]
    T = meta["T"]
    p = print

    p("== corpus")
    c = C.load(data_dir)
    tot = 0
    for code in c.langs:
        s = c.streams[code]
        w = s[c.lang_of[s] >= 0]
        _, n = np.unique(w, return_counts=True)
        tot += len(s)
        p(f"{C.LANG_NAMES[code]:10s} tokens {len(s):6d} words {len(w):6d} types {len(n):5d} "
          f"hapax {np.sum(n == 1):5d} ({np.mean(n == 1):.1%} of types) n1/N {np.sum(n == 1) / len(w):.2%}")
    p(f"all tokens {tot:,}  vocabulary incl. punctuation {c.V:,}")
    p(f"train tokens N={meta['N']:,} docs={meta['n_docs']}  types0={meta['types0']} sum={sum(meta['types0']):,}")
    st = E.Setup(data_dir)
    row_n = np.bincount(st.real.rows, weights=st.real.n, minlength=st.V)
    n = row_n[row_n > 0]
    p(f"test tokens {sum(len(b) for b in st.test):,}; contexts {len(n):,}; median context count {np.median(n):.0f}; "
      f"share seen once {np.mean(n == 1):.1%}; <=2 {np.mean(n <= 2):.1%}; bigram types {len(st.real.codes):,}; "
      f"bigram tokens that are singletons {st.real.n[st.real.n == 1].sum() / st.real.n.sum():.1%}")

    n_blocks = len(st.train) + len(st.test)
    p(f"held-out blocks {len(st.test)} of {n_blocks} ({len(st.test) / n_blocks:.1%})")
    col_n0 = np.bincount(st.real.cols, weights=st.real.n, minlength=st.V)
    N_words = col_n0[st.is_word].sum()
    dead = np.flatnonzero((col_n0 > 0) & st.is_word & (row_n == 0))
    p(f"word tokens in the first model N_w={N_words:,.0f}; word types with no successor {len(dead)}")
    # openings after a full stop, whole text of each translation
    dot = c.vocab.index("pun|.")
    for code in c.langs:
        s_ = c.streams[code]
        nxt = s_[1:][s_[:-1] == dot]
        u, n_ = np.unique(nxt, return_counts=True)
        k = np.argmax(n_)
        p(f"after '.' in {C.LANG_NAMES[code]:10s}: {len(nxt):,} words, most common "
          f"'{c.vocab[u[k]].split('|')[1]}' {n_[k]:,} ({n_[k] / len(nxt):.1%})")
    # one-round survival formula on each language's training counts
    for li, code in enumerate(c.langs):
        cnt = col_n0[(c.lang_of == li) & (col_n0 > 0)]
        NL = cnt.sum()
        kept = np.sum(1 - (1 - cnt / NL) ** NL) / len(cnt)
        p(f"one-round formula on training counts, {C.LANG_NAMES[code]:10s}: {kept:.1%} of {len(cnt)} types")

    p("\n== replace, temperature 1")
    ty = arr(res, "replace", "types")
    for t in (1, 2, 5, 10, 20, 30, 50):
        p(f"t={t:2d} types {fmt(ty, t, d=0)}  retained {ty[:, t].mean() / ty[:, 0].mean():.1%}")
    for li, code in enumerate(langs):
        a = arr(res, "replace", "types_lang", li) / meta["types0"][li]
        mm = arr(res, "replace", "missing_mass_lang", li)
        p(f"  {C.LANG_NAMES[code]:10s} retained t1 {a[:, 1].mean():.1%} t10 {a[:, 10].mean():.1%} "
          f"t50 {a[:, 50].mean():.1%} | missing mass t0 {mm[:, 0].mean():.1%} t50 {mm[:, 50].mean():.1%}")
    for key in ("gini_simpson", "shannon_bits", "cond_gini_simpson", "entropy_rate_bits",
                "bigram_types", "missing_mass", "xent_bits", "hapax_bigram_share"):
        a = arr(res, "replace", key)
        p(f"  {key:20s} t0 {a[:, 0].mean():.4f}  t1 {a[:, 1].mean():.4f}  t10 {a[:, 10].mean():.4f}  "
          f"t50 {a[:, 50].mean():.4f}  ratio50 {a[:, 50].mean() / a[:, 0].mean():.4f}")
    gs0 = arr(res, "replace", "gini_simpson")[:, 0].mean()
    gsr = arr(res, "replace", "gini_simpson")
    p(f"  Gini-Simpson t0 {gsr[:, 0].mean():.5f} t50 {gsr[:, 50].mean():.5f}; Wright-Fisher prediction "
      f"with word tokens N_w: {gs0 * (1 - 1 / N_words) ** 50:.5f} ((1-1/N_w)^50 = {(1 - 1 / N_words) ** 50:.5f})")
    sh = np.stack([arr(res, "replace", "lang_share_output", li) for li in range(len(langs))])
    per_seed = np.abs(sh[:, :, 50] - sh[:, :, 0]).max(axis=0)
    p(f"  max |language share change| t0->t50: seed means {np.max(np.abs(sh[:, :, 50].mean(1) - sh[:, :, 0].mean(1))):.3f}; "
      f"single seeds max {per_seed.max():.3f}, median {np.median(per_seed):.3f}")
    # per-context prediction, conditional Gini-Simpson
    pr = st.real.n / row_n[st.real.rows]
    gs_row = 1 - np.bincount(st.real.rows, weights=pr ** 2, minlength=st.V)
    w = row_n / row_n.sum()
    cg = arr(res, "replace", "cond_gini_simpson")
    for t in (1, 2, 5, 10):
        pred = np.sum(w * gs_row * (1 - 1 / np.maximum(row_n, 1)) ** t)
        p(f"  cond GS t={t}: predicted (1-1/n_v)^t {pred:.4f}  simulated {cg[:, t].mean():.4f}")

    p("\n== data policies, temperature 1")
    for name in ("replace", "accumulate", "anchor10"):
        ty = arr(res, name, "types")
        mm = arr(res, name, "missing_mass")
        xe = arr(res, name, "xent_bits")
        cg = arr(res, name, "cond_gini_simpson")
        p(f"{name:11s} types t50 {fmt(ty, 50, d=0)} ({ty[:, 50].mean() / ty[:, 0].mean():.1%})  "
          f"missing t0 {mm[:, 0].mean():.1%} t10 {mm[:, 10].mean():.1%} t50 {fmt(mm, 50, pct=True)}  "
          f"xent t0 {xe[:, 0].mean():.2f} t10 {xe[:, 10].mean():.2f} t50 {xe[:, 50].mean():.2f}  "
          f"condGS t50/t0 {cg[:, 50].mean() / cg[:, 0].mean():.4f}")
    pred_acc = np.sum(w * gs_row * euler_R(np.maximum(row_n, 1))) / np.sum(w * gs_row)
    p(f"accumulate: predicted limit of conditional Gini-Simpson ratio (Euler product per context) {pred_acc:.4f}")
    for nv in (2, 3, 5, 10, 100):
        p(f"  R({nv}) = {float(euler_R(nv)):.4f}")
    a = arr(res, "anchor10", "types")
    p(f"anchor10 types t20 {a[:, 20].mean():,.0f} t30 {a[:, 30].mean():,.0f} t50 {a[:, 50].mean():,.0f}")

    p("\n== decoding")
    en = langs.index("en")
    fi = langs.index("fi")
    for name in ("replace", "replace_tau08", "replace_topp95", "accumulate_tau08", "anchor10_tau08",
                 "accumulate_topp95", "anchor10_topp95"):
        e = arr(res, name, "lang_share_output", en)
        f = arr(res, name, "lang_share_output", fi)
        ty = arr(res, name, "types")
        gs = arr(res, name, "gini_simpson")
        mm = arr(res, name, "missing_mass")
        first90 = [int(np.argmax(r > 0.9)) if np.any(r > 0.9) else None for r in e]
        p(f"{name:18s} EN t0 {e[:, 0].mean():.1%} t5 {e[:, 5].mean():.1%} t10 {e[:, 10].mean():.1%} "
          f"t20 {e[:, 20].mean():.1%} t50 {fmt(e, 50, pct=True)} | FI t0 {f[:, 0].mean():.1%} t50 {f[:, 50].mean():.1%} "
          f"| types t10 {ty[:, 10].mean():,.0f} t20 {ty[:, 20].mean():,.0f} t50 {ty[:, 50].mean():,.0f} "
          f"| GS t50 {gs[:, 50].mean():.3f} | missing t50 {mm[:, 50].mean():.1%} | first gen EN>90%: {first90}")

    p("\n== temperature 0.8, replace: when each language disappears (first generation with zero share)")
    for seed in range(meta["seeds"]):
        order = []
        for li, code in enumerate(langs):
            a = arr(res, "replace_tau08", "lang_share_output", li)[seed]
            z = np.flatnonzero(a == 0)
            order.append((int(z[0]) if len(z) else None, C.LANG_NAMES[code]))
        p(f"  seed {seed}: " + ", ".join(f"{n} {g}" for g, n in sorted(order, key=lambda x: (x[0] is None, x[0]))))
    e = arr(res, "replace_tau08", "lang_share_output", en)
    p(f"  English share exactly 100% from generation: seed mean {int(np.argmax(e.mean(0) == 1.0))}, "
      f"per seed {[int(np.argmax(r == 1.0)) for r in e]}")

    if th is None:
        return
    p("\n== closed forms vs Monte Carlo")
    for t, v in th["replace_N1000"].items():
        p(f"replace N=1000 t={t}: MC {v['mc']:.4f} (se {v['se']:.4f})  formula {v['formula']:.4f}")
    for code, v in th["one_step"].items():
        p(f"one step {code}: kept formula {v['kept_formula']:.1f} ({v['kept_formula'] / v['types']:.1%}) "
          f"MC {v['kept_mc']:.1f} of {v['types']} | "
          f"mass lost formula {v['mass_lost_formula']:.4f} MC {v['mass_lost_mc']:.4f} | "
          f"n1/N of new sample formula {v['n1_over_N_next_formula']:.4f} MC {v['n1_over_N_next_mc']:.4f} | "
          f"Good-Turing n1/N of corpus {v['good_turing_n1_over_N']:.4f}")
    for v in th["accumulate"]:
        p(f"accumulate N={v['N']}: MC {v['mc']:.4f} (se {v['se']:.4f}) formula {v['formula']:.4f} "
          f"| replace after {v['T']} gens {v['replace_same_T']:.2e}")
    for v in th["anchor"]:
        p(f"anchor N={v['N']} a={v['alpha']}: MC {v['mc']:.4f} formula {v['formula']:.4f} "
          f"| replace after 400 gens {v['replace_after_T']:.2e}")
    for rec in th["temperature_en_tau08"]:
        if rec["t"] in (0, 1, 2, 5, 10, 20):
            p(f"temperature 0.8, English unigram, t={rec['t']}: effective types {rec['eff_types']:.1f} "
              f"top word share {rec['top_word_share']:.3f}")
    for code, v in th["nucleus"].items():
        p(f"iterated nucleus {code}: " + "; ".join(
            f"p={x['top_p']} survivors {x['survivors']} (bound {x['bound']}) after {x['steps']} steps"
            for x in v))
    for ctx, v in th["branching_tau08"].items():
        p(f"branching after '{ctx}' at tau 0.8: " + "; ".join(
            f"{k} {x['share_before']:.3f}->{x['share_after']:.3f} (Renyi {x['renyi_bits']:.2f} bits)"
            for k, x in v.items()))
    for ctx, v in th["nucleus_cut_095"].items():
        p(f"nucleus 0.95 after '{ctx}': " + "; ".join(
            f"{k} mass {x['mass']:.3f} cut {x['cut_share_of_own_mass']:.1%}" for k, x in v.items()))


if __name__ == "__main__":
    import sys
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    main(data, Path("results"))
