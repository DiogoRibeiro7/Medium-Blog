"""Every number in the article, printed in order, plus the arrays the figures use (results/).

python letters_numbers.py > output.txt
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

import letters_core as L

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 20261124
SHUFFLE_DRAWS = 10_000
NOISE_DRAWS = 1_000
TEMPERATURES = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03]
VARIANTS = [(a, w, b) for a in L.ALPHABETS for w in L.WEIGHTINGS for b in (False, True)]
MAIN = (
    "standard",
    "text",
    False,
)  # the post's choices: standard letters, running text, inside words
SPACE = ("standard", "text", True)


def show(nodes, sym):
    return L.spell(nodes, sym).replace(L.BOUNDARY, "␣")


def loops_text(s, sym):
    parts = [show(lp, sym) for lp in s.loops] + [f"end:{sym[d]}" for d in s.dead_ends]
    return " / ".join(parts)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def header(t):
    print(f"\n== {t}")


def main():
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    counts, freqs = {}, {}
    for code in L.LANGUAGES:
        freqs[code] = L.word_frequencies(code)
        counts[code] = L.count_language(code, freqs[code])
    chains = {(code, *v): L.chain(counts[code], code, *v) for code in L.LANGUAGES for v in VARIANTS}

    header("Inputs")
    print(f"wordfreq {L.wordfreq_version()}, CLDR 49; {len(L.LANGUAGES)} languages")
    kinds = Counter(L.wordlist_kind(c) for c in L.LANGUAGES)
    print(f"word lists: {kinds['large']} large, {kinds['small']} small")
    for code, (name, _loc, script) in L.LANGUAGES.items():
        c = counts[code][("standard", "text")]
        cd = counts[code][("standard", "dictionary")]
        unseen = "".join(L.unseen_letters(code, counts[code]))
        print(
            f"{code:3s} {name:17s} {script:8s} list={L.wordlist_kind(code)} letters={len(c.symbols) - 1}"
            f" kept={pct(c.kept)} words_text={c.words} words_dict={cd.words}" + (f" unseen={unseen}" if unseen else "")
        )

    header("Norvig's table (Google Books, inside words)")
    print(f"letters {L.NORVIG_LETTERS:,} - words {L.NORVIG_WORDS:,} = pairs {L.NORVIG_LETTERS - L.NORVIG_WORDS:,}")
    nv = L.NORVIG_TOP50
    for b1, others in (
        ("er", ("en", "es", "ed")),
        ("re", ("ri", "ro", "ra")),
    ):
        print(f"{b1} {nv[b1]}B vs " + ", ".join(f"{o} {nv[o]}B" for o in others))
    en = counts["en"][("standard", "text")]
    keys, nv_s, wf_s, sigma, settled = L.norvig_comparison(en)
    agree = sum(v[0] == v[1] for v in settled.values())
    print(f"letters whose arrow Norvig's top 50 settles: {len(settled)}; same arrow in wordfreq: {agree}")
    print("arrows (Norvig, wordfreq): " + " ".join(f"{a}->{v[0]}/{v[1]}" for a, v in settled.items()))
    lr = np.log(wf_s / nv_s)
    worst = np.argsort(-np.abs(lr))[:3]
    print(f"SD of log share ratio over the 50 pairs: {sigma:.3f} (a typical factor of {np.exp(sigma):.2f})")
    print("largest differences: " + ", ".join(f"{keys[k]} x{np.exp(lr[k]):.2f}" for k in worst))
    print(
        f"rank correlation of the 50 shares: "
        f"{np.corrcoef(np.argsort(np.argsort(nv_s)), np.argsort(np.argsort(wf_s)))[0, 1]:.3f}"
    )

    header("English, as in the post (standard letters, running text, inside words)")
    sym, m, f, s, order = chains[("en", *MAIN)]
    row = m / m.sum(axis=1, keepdims=True)
    ix = {c: k for k, c in enumerate(sym)}
    print("map: " + " ".join(f"{sym[i]}{sym[f[i]]}" for i in range(len(sym))))
    print(f"loops: {loops_text(s, sym)}; attractors {s.n_attractors()}; longest path to the loop {s.depth.max()}")
    deepest = [sym[i] for i in np.flatnonzero(s.depth == s.depth.max())]
    path = [deepest[0]]
    while s.depth[ix[path[-1]]] > 0:
        path.append(sym[f[ix[path[-1]]]])
    print(f"deepest start(s): {''.join(deepest)}; path {'->'.join(path)}")
    for a, b in (
        ("q", "u"),
        ("x", "p"),
        ("e", "r"),
        ("r", "e"),
        ("t", "h"),
        ("h", "e"),
    ):
        print(f"P({b}|{a}) = {pct(row[ix[a], ix[b]])}")
    second_e = sym[int(np.argsort(-m[ix["e"]])[1])]
    print(f"e: r {pct(row[ix['e'], ix['r']])}, next {second_e} {pct(row[ix['e'], ix[second_e]])}")
    into = Counter(sym[f[i]] for i in range(len(sym)))
    print("in-degree: " + ", ".join(f"{k} {v}" for k, v in into.most_common(6)))
    lw = order / order.sum()
    top = np.argsort(-lw)[:3]
    print("most frequent letters: " + ", ".join(f"{sym[k]} {pct(lw[k], 2)}" for k in top))

    header("Baselines")
    for n in (26, 31, 36):
        p = L.p_one_loop_uniform(n)
        print(f"uniform random map, n={n}: P(one loop) = {float(p):.4f}; sqrt(pi/2n) = {np.sqrt(np.pi / (2 * n)):.4f}")
    shuffle = {}
    for code in L.LANGUAGES:
        for b in (False, True):
            sym, m, f, s, order = chains[(code, "standard", "text", b)]
            att = L.shuffled_targets(f, rng, SHUFFLE_DRAWS)
            ind = np.bincount(f[f >= 0], minlength=len(sym))
            shuffle[(code, b)] = {
                "p_one": float((att == 1).mean()),
                "mean": float(att.mean()),
                "observed": s.n_attractors(),
                "top_in": int(ind.max()),
                "n": len(sym),
                "p_uniform": float(L.p_one_loop_uniform(len(sym))),
            }
    for b in (False, True):
        rows = [shuffle[(c, b)] for c in L.LANGUAGES]
        p1 = np.array([r["p_one"] for r in rows])
        obs1 = sum(r["observed"] == 1 for r in rows)
        share_top = np.array([r["top_in"] / r["n"] for r in rows])
        print(
            f"{'with boundary' if b else 'inside words'}: observed one attractor in {obs1} of 36; "
            f"expected under shuffle {p1.sum():.1f}; under uniform maps "
            f"{sum(r['p_uniform'] for r in rows):.1f}; shuffle P(one) median {np.median(p1):.2f} "
            f"(range {p1.min():.2f}-{p1.max():.2f}); most popular target gets a median "
            f"{pct(np.median(share_top), 0)} of the arrows"
        )
    e = shuffle[("en", False)]
    print(f"English inside words: shuffle P(one) = {e['p_one']:.3f}, mean attractors {e['mean']:.2f}")
    for code in L.LANGUAGES:
        sym, m, f, s, order = chains[(code, *MAIN)]
        att = L.shuffled_targets(f, rng, SHUFFLE_DRAWS // 5, allow_self=False)
        shuffle[(code, False)]["p_one_noself"] = float((att == 1).mean())
    p_ns = np.array([shuffle[(c, False)]["p_one_noself"] for c in L.LANGUAGES])
    print(
        f"inside words, shuffle with no letter pointing at itself: expected one attractor in {p_ns.sum():.1f} "
        f"of 36; P(one) median {np.median(p_ns):.2f}; English {shuffle[('en', False)]['p_one_noself']:.3f}"
    )

    header("36 languages, inside words (standard letters, running text)")
    multi, fixed_points = [], []
    for code in L.LANGUAGES:
        sym, m, f, s, order = chains[(code, *MAIN)]
        sizes = sorted(s.basin_sizes().tolist(), reverse=True)
        print(f"{code:3s} {loops_text(s, sym):24s} attractors={s.n_attractors()} basins={sizes}")
        if s.n_attractors() > 1:
            multi.append(code)
        fixed_points += [f"{code}:{sym[lp[0]]}" for lp in s.loops if len(lp) == 1]
    print(f"one attractor: {36 - len(multi)} of 36; more than one: {len(multi)} ({', '.join(multi)})")
    kinds = Counter()
    for code in multi:
        sym, m, f, s, order = chains[(code, *MAIN)]
        labels = L.classify_attractors(s)
        kinds.update({"with a fixed point": "fixed point" in labels, "with a dead end": "dead end" in labels,
                      "with two loops of 2+ letters": labels.count("loop") >= 2})  # fmt: skip
    print("of these: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    sym, m, f, s, order = chains[("en", *MAIN)]
    vowels = set("aeiou")
    kinds_en = Counter(
        ("V" if sym[i] in vowels else "C") + ("V" if sym[f[i]] in vowels else "C") for i in range(len(sym))
    )
    cc = [f"{sym[i]}{sym[f[i]]}" for i in range(len(sym)) if sym[i] not in vowels and sym[f[i]] not in vowels]
    print(f"English arrows by vowel (aeiou) and consonant: {dict(sorted(kinds_en.items()))}; CC: {' '.join(cc)}")
    mg = L.margins(m)
    for a in "lseot":
        k = sym.index(a)
        sec = sym[int(np.argsort(-m[k])[1])]
        print(
            f"en {a}: {a}{sym[f[k]]} beats {a}{sec} by a factor {mg[k]:.2f}; {a}{a} share {pct(m[k, k] / m[k].sum())}"
        )
    print(f"fixed points (a letter whose likeliest successor is itself): {', '.join(fixed_points)}")
    has_top = 0
    for code in L.LANGUAGES:
        sym, m, f, s, order = chains[(code, *MAIN)]
        att = L.main_attractor(s)
        has_top += att[0] == "loop" and int(np.argmax(order)) in att[1]
    print(f"languages whose main loop contains their most frequent letter: {has_top} of 36")
    final_like = {}
    for code in L.LANGUAGES:
        c = counts[code][("standard", "text")]
        share = c.pairs[1:, 1:].sum(axis=1) / c.letters[1:]
        low = [c.symbols[1 + i] for i in np.flatnonzero(share < 0.01)]
        if low:
            final_like[code] = (low, [float(x) for x in share[share < 0.01]])
    for code, (low, sh) in final_like.items():
        print(f"{code}: letters followed by another letter in under 1% of uses: {''.join(low)} (max {pct(max(sh), 3)})")
    sym, m, f, s, order = chains[("he", *MAIN)]
    k = sym.index("ם")
    print(f"he: arrow from final mem {sym[k]}->{sym[f[k]]} rests on {pct(m[k].sum() / order[k], 3)} of its uses")

    header("36 languages with a word boundary (standard letters, running text)")
    words = {}
    for code, (name, *_) in L.LANGUAGES.items():
        sym, m, f, s, order = chains[(code, *SPACE)]
        gw = L.greedy_word(f)
        g = L.spell(gw, sym) if gw is not None else None
        letters = set(sym[1:])
        kept = [w for w in sorted(freqs[code], key=freqs[code].get, reverse=True) if w and set(w) <= letters]
        rank = kept.index(g) + 1 if g in kept else None
        words[code] = {
            "greedy": g,
            "rank": rank,
            "top": kept[0],
            "top3": kept[:3],
            "attractors": s.n_attractors(),
        }
        print(
            f"{code:3s} {name:17s} loops {loops_text(s, sym):18s} greedy word {g!s:6s} rank {rank!s:5s} "
            f"most frequent: {' '.join(kept[:3])}"
        )
    top1 = sum(v["rank"] == 1 for v in words.values())
    top10 = sum(v["rank"] is not None and v["rank"] <= 10 for v in words.values())
    rare = [c for c, v in words.items() if v["rank"] is None or v["rank"] > 1000]
    print(
        f"greedy word is the most frequent word: {top1} of 36; in the top 10: {top10}; "
        f"ranked below 1,000: {len(rare)} ({', '.join(rare)})"
    )
    multi_b = [c for c, v in words.items() if v["attractors"] > 1]
    print(f"more than one attractor with boundary: {len(multi_b)} ({', '.join(multi_b)})")
    sym, m, f, s, order = chains[("en", *SPACE)]
    ix = {c: k for k, c in enumerate(sym)}
    row = m / m.sum(axis=1, keepdims=True)
    for a, b in (("_", "t"), ("t", "h"), ("h", "e"), ("e", "_"), ("e", "r")):
        print(f"en P({b}|{a}) = {pct(row[ix[a], ix[b]])}")
    for code, letters in (("es", "_ed"), ("de", "_den"), ("fi", "_ka"), ("he", "_ה"), ("ar", "_ال")):
        sym, m, f, s, order = chains[(code, *SPACE)]
        ix = {c: k for k, c in enumerate(sym)}
        row = m / m.sum(axis=1, keepdims=True)
        for a in dict.fromkeys(letters):
            top = np.argsort(-row[ix[a]])[:3]
            print(f"{code} after {a}: " + ", ".join(f"{sym[k]} {pct(row[ix[a], k])}" for k in top))

    header("Alphabet: folding accents and positional forms")
    for b in (False, True):
        changed = []
        for code in L.LANGUAGES:
            s0, s1 = (
                chains[(code, "standard", "text", b)],
                chains[(code, "folded", "text", b)],
            )
            if {show(x, s0[0]) for x in s0[3].loops} != {show(x, s1[0]) for x in s1[3].loops}:
                changed.append(f"{code}: {loops_text(s0[3], s0[0])} -> {loops_text(s1[3], s1[0])}")
        print(f"{'with boundary' if b else 'inside words'}: loops change in {len(changed)} of 36")
        for line in changed:
            print("  " + line)
    for code in ("vi", "he", "el", "fi"):
        print(
            f"{code}: letters {len(counts[code][('standard', 'text')].symbols) - 1} standard, "
            f"{len(counts[code][('folded', 'text')].symbols) - 1} folded"
        )

    header("Weighting: running text against a dictionary (standard letters)")
    for b in (False, True):
        changed = []
        for code in L.LANGUAGES:
            s0, s1 = (
                chains[(code, "standard", "text", b)],
                chains[(code, "standard", "dictionary", b)],
            )
            if {show(x, s0[0]) for x in s0[3].loops} != {show(x, s1[0]) for x in s1[3].loops}:
                changed.append(f"{code}: {loops_text(s0[3], s0[0])} -> {loops_text(s1[3], s1[0])}")
        print(f"{'with boundary' if b else 'inside words'}: loops change in {len(changed)} of 36")
        for line in changed:
            print("  " + line)
    sym, m, f, s, order = chains[("en", "standard", "dictionary", True)]
    ix = {c: k for k, c in enumerate(sym)}
    row = m / m.sum(axis=1, keepdims=True)
    print(f"en dictionary with boundary: {loops_text(s, sym)}; basins {sorted(s.basin_sizes().tolist(), reverse=True)}")
    for a, b in (("_", "s"), ("_", "t"), ("e", "r"), ("e", "_"), ("s", "_")):
        print(f"en dictionary P({b}|{a}) = {pct(row[ix[a], ix[b]])}")
    tb = sum(len(L.tied_rows(chains[(c, "standard", "dictionary", b)][1])) for c in L.LANGUAGES for b in (0, 1))
    print(f"rows with tied dictionary counts (broken by running text): {tb}")

    header("All eight choices together")
    distinct = {}
    for code in L.LANGUAGES:
        sets = {frozenset(show(x, chains[(code, *v)][0]) for x in chains[(code, *v)][3].loops) for v in VARIANTS}
        distinct[code] = len(sets)
    hist = Counter(distinct.values())
    print("distinct sets of loops across the 8 choices: " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
    print(f"English: {distinct['en']}; most: {max(distinct, key=distinct.get)} {max(distinct.values())}")
    single_all = [c for c in L.LANGUAGES if all(chains[(c, *v)][3].n_attractors() == 1 for v in VARIANTS)]
    print(f"one attractor under all 8 choices: {len(single_all)} ({', '.join(single_all)})")

    header("Robustness to the corpus")
    print(f"noise: every count times exp({sigma:.3f} Z), {NOISE_DRAWS} draws per language")
    survive, draws_en = {}, {}
    for b in (False, True):
        for code in L.LANGUAGES:
            sym, m, f, s, order = chains[(code, "standard", "text", b)]
            loops, n_att = L.perturbed(m, sigma, rng, NOISE_DRAWS, order, 0 if b else None)
            if code == "en":
                draws_en[b] = (loops, n_att)
            ref = L.loop_set(s)
            r = m.copy()
            mg = L.margins(r)
            loop_nodes = sorted({v for lp in s.loops for v in lp})
            survive[(code, b)] = {
                "same_loops": float(np.mean([x == ref for x in loops])),
                "one_attractor": float((n_att == 1).mean()),
                "observed_one": s.n_attractors() == 1,
                "min_margin": float(np.nanmin(mg[loop_nodes])),
            }
        same = np.array([survive[(c, b)]["same_loops"] for c in L.LANGUAGES])
        print(
            f"{'with boundary' if b else 'inside words'}: loops unchanged in a median {pct(np.median(same), 0)} "
            f"of draws; at least 95% in {(same >= 0.95).sum()} of 36; below 50% in {(same < 0.5).sum()}"
        )
        fragile = sorted(L.LANGUAGES, key=lambda c: survive[(c, b)]["same_loops"])[:5]
        print("  most fragile: " + ", ".join(f"{c} {pct(survive[(c, b)]['same_loops'], 0)}" for c in fragile))
    for b in (False, True):
        sym = chains[("en", "standard", "text", b)][0]
        loops, n_att = draws_en[b]
        alt = Counter(" / ".join(sorted(show(x, sym) for x in lp)) for lp in loops)
        print(
            f"English {'with boundary' if b else 'inside words'}, outcomes under noise: "
            + ", ".join(f"{k} {pct(v / NOISE_DRAWS, 0)}" for k, v in alt.most_common(4))
            + f"; one attractor {pct((n_att == 1).mean(), 0)}"
        )
    sym, m, f, s, order = chains[("en", *MAIN)]
    mg = L.margins(m)
    print(
        f"English inside words: unchanged {pct(survive[('en', False)]['same_loops'], 1)}; "
        f"margins e {mg[sym.index('e')]:.2f}, r {mg[sym.index('r')]:.2f}"
    )
    sym, m, f, s, order = chains[("en", *SPACE)]
    mg = L.margins(m)
    print(
        f"English with boundary: unchanged {pct(survive[('en', True)]['same_loops'], 1)}; margins "
        + ", ".join(f"{sym[i]} {mg[i]:.2f}" for i in s.loops[0])
    )

    header("Randomness instead of the maximum (standard letters, running text, with boundary)")
    temp, worst_res = {}, 0.0
    for code in L.LANGUAGES:
        sym, m, f, s, order = chains[(code, *SPACE)]
        assert L.is_irreducible(m)
        main_loop = L.main_attractor(s)[1]
        out = []
        for t in TEMPERATURES:
            pi, res = L.stationary(L.tempered(m, t))
            worst_res = max(worst_res, res)
            pi = np.clip(pi, 0, None)
            out.append(
                {
                    "T": t,
                    "mass": float(pi[list(main_loop)].sum()),
                    "H": L.entropy_rate(L.tempered(m, t), pi / pi.sum()),
                }
            )
        temp[code] = out
    print(f"every stationary solve has residual |pi P - pi| below 1e-12: {worst_res < 1e-12}")
    en = temp["en"]
    print("English, share of time on the loop _the: " + ", ".join(f"T={r['T']} {pct(r['mass'], 0)}" for r in en))
    print("English, bits per symbol: " + ", ".join(f"T={r['T']} {r['H']:.2f}" for r in en))
    m1 = np.array([temp[c][0]["mass"] for c in L.LANGUAGES])
    print(
        f"at T=1 the word loop takes a median {pct(np.median(m1), 0)} of the time "
        f"(range {pct(m1.min(), 0)}-{pct(m1.max(), 0)})"
    )
    cold = {c: temp[c][-1]["mass"] for c in L.LANGUAGES}
    low = sorted(cold, key=cold.get)[:3]
    print(
        f"at T={TEMPERATURES[-1]} the loop takes a median {pct(np.median(list(cold.values())), 0)}; lowest: "
        + ", ".join(f"{c} {pct(cold[c], 0)}" for c in low)
    )
    for c in low[:2]:
        sym, m, f, s, order = chains[(c, *SPACE)]
        mg = L.margins(m)
        k = min(L.main_attractor(s)[1], key=lambda i: mg[i])
        print(f"  {c}: closest call on the loop {sym[k]}->{sym[f[k]]}, margin {mg[k]:.3f}")
    half = []
    for c in L.LANGUAGES:
        ts = [r["T"] for r in temp[c] if r["mass"] >= 0.5]
        half.append(max(ts) if ts else None)
    print(f"highest T in the grid where the loop takes half the time: median {np.median([h for h in half if h]):.2f}")
    for code in multi_b:
        sym, m, f, s, order = chains[(code, *SPACE)]
        masses = []
        for t in (1.0, 0.2, 0.05):
            pi = np.clip(L.stationary(L.tempered(m, t))[0], 0, None)
            masses.append(f"T={t}: " + ", ".join(f"{show(lp, sym)} {pct(pi[list(lp)].sum(), 1)}" for lp in s.loops))
        print(f"{code}: " + "; ".join(masses))
    for code in ("en", "fr", "fi"):
        sym, m, f, s, order = chains[(code, *SPACE)]
        print(f"text from the {L.LANGUAGES[code][0]} chain:")
        for t in (1.0, 0.5, 0.25):
            g = np.random.default_rng(SEED)
            print(f"  T={t}: {L.generate(L.tempered(m, t), sym, g, 70)!r}")
        print(f"  greedy: {L.generate(L.tempered(m, 1.0), sym, rng, 40, greedy=True)!r}")

    # arrays for figures.py
    keep = {
        "chains": {
            f"{k[0]}|{k[1]}|{k[2]}|{int(k[3])}": {"sym": v[0], "f": v[2].tolist(), "depth": v[3].depth.tolist(),
                                                  "basin": v[3].basin.tolist(), "order": v[4].tolist(),
                                                  "loops": [list(x) for x in v[3].loops]}
            for k, v in chains.items()
        },
        "shuffle": {f"{k[0]}|{int(k[1])}": v for k, v in shuffle.items()},
        "survive": {f"{k[0]}|{int(k[1])}": v for k, v in survive.items()},
        "temp": temp,
        "words": words,
        "sigma": sigma,
    }  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    with open(RESULTS / "counts.pkl", "wb") as fh:
        pickle.dump(counts, fh)


if __name__ == "__main__":
    main()
