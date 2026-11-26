"""Every number in the article, from the generations in results/ (written by generate_runs.py).

python loops_numbers.py > output.txt         # after the full run (artifacts gen-* in results/)
python loops_numbers.py --smoke              # the smoke run's file only
"""

import gzip
import json
import sys
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

import loops_core as L

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NGRAM_STEPS = 200
MEMORISED = 32  # a greedy continuation sharing this many leading tokens with the book is "memorised"
FUNCTION_TOP = 100  # "function words": the 100 most frequent word types of the 18 books


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def header(t):
    print(f"\n== {t}")


def load(smoke=False):
    files = sorted(RESULTS.glob("gen_*.json.gz"))
    files = [f for f in files if (f.name == "gen_smoke.json.gz") == smoke]
    runs = []
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return sorted(runs, key=lambda r: (r["family"], r["params_m"]))


def summarise(tokens, logprob):
    """Loop statistics of a set of continuations."""
    tails = [L.periodic_tail(t) for t in tokens]
    looped = [t for t in tails if t.looped]
    n = len(tokens)
    k = len(looped)
    lo, hi = L.wilson(k, n)
    lp = np.concatenate([np.asarray(x) for x in logprob])
    return {
        "n": n,
        "looped": k,
        "rate": k / n if n else float("nan"),
        "ci": (lo, hi),
        "onset_median": float(np.median([t.start for t in looped])) if looped else float("nan"),
        "period_median": float(np.median([t.period for t in looped])) if looped else float("nan"),
        "rep4": float(np.mean([L.repeated_ngram_share(t, 4) for t in tokens])),
        "logprob": float(lp.mean()),
        "tails": tails,
    }


def reinforcement(tokens, logprob, margin, tails, max_cycles=6):
    """Mean probability and top-2 gap of the chosen tokens before the loop and in its 1st, 2nd... cycle."""
    pre_p, pre_m = [], []
    cyc_p = [[] for _ in range(max_cycles)]
    cyc_m = [[] for _ in range(max_cycles)]
    for lp, mg, t in zip(logprob, margin, tails, strict=True):
        if not t.looped:
            continue
        p = np.exp(np.asarray(lp))
        m = np.asarray(mg)
        if t.start > 0:
            pre_p.append(p[: t.start].mean())
            pre_m.append(m[: t.start].mean())
        for c in range(max_cycles):
            a, b = t.start + c * t.period, t.start + (c + 1) * t.period
            if b > len(p):
                break
            cyc_p[c].append(p[a:b].mean())
            cyc_m[c].append(m[a:b].mean())
    return {
        "pre_p": float(np.mean(pre_p)) if pre_p else float("nan"),
        "pre_m": float(np.mean(pre_m)) if pre_m else float("nan"),
        "cycle_p": [float(np.mean(x)) if x else float("nan") for x in cyc_p],
        "cycle_m": [float(np.mean(x)) if x else float("nan") for x in cyc_m],
        "cycle_n": [len(x) for x in cyc_p],
    }


@cache
def book_words():
    words = []
    for b in L.BOOKS:
        words += L.tokenize_words(L.read_book(b))
    return words


def alpha_words(text):
    return [w.lower() for w in L.tokenize_words(text) if w.isalpha()]


def word_loops(texts, vocab):
    """Loops at the level of words: the repeating block (one period), the words before it, and the index."""
    blocks, prefixes, which = [], [], []
    for i, t in enumerate(texts):
        w = alpha_words(t)
        codes = np.array([vocab.setdefault(x, len(vocab)) for x in w], dtype=np.int64)
        tail = L.periodic_tail(codes, max_period=40, min_repeats=3, min_tokens=12)
        if tail.looped:
            blocks.append(w[tail.start : tail.start + tail.period])
            prefixes.append(w[: tail.start])
            which.append(i)
    return blocks, prefixes, which


def copied(block, source, k=4):
    """Does some run of k consecutive words of the repeating block (read round the cycle) occur in source?"""
    if not block:
        return False
    k = min(k, len(block))
    cyc = block + block[: k - 1]
    grams = {tuple(source[i : i + k]) for i in range(len(source) - k + 1)}
    return any(tuple(cyc[i : i + k]) in grams for i in range(len(block)))


def counting_loop(text):
    """A loop in the characters once every digit is masked (verse numbers that count on)."""
    chars = np.frombuffer(L.mask_digits(text).encode("utf-32-le"), dtype=np.uint32)
    return L.periodic_tail(chars, max_period=600, min_tokens=60).looped


def repeated_twice(tokens):
    """A weaker criterion: the continuation ends in a block of up to 128 tokens that occurs twice in a row."""
    return L.periodic_tail(np.asarray(tokens), max_period=128, min_repeats=2, min_tokens=48).looped


def share_in(words, top):
    return sum(w in top for w in words) / len(words) if words else float("nan")


def ngram_ladder(passages):
    words = book_words()
    out = {}
    for n in (1, 2, 3, 4):
        chain = L.NGramChain(words, n)
        tails = []
        for p in passages:
            cont = chain.continue_from(L.tokenize_words(p.prompt), NGRAM_STEPS)
            tails.append(L.periodic_tail(np.array([chain.code[w] for w in cont])))
        looped = [t for t in tails if t.looped]
        example = chain.continue_from(L.tokenize_words(passages[0].prompt), 40)
        out[n] = {
            "rate": len(looped) / len(tails),
            "onset": float(np.median([t.start for t in looped])) if looped else float("nan"),
            "period": float(np.median([t.period for t in looped])) if looped else float("nan"),
            "example": " ".join(example),
        }
    return out, len(words)


def main(smoke=False):
    runs = load(smoke)
    if not runs:
        sys.exit("no generations in results/")
    passages = L.make_prompts()
    summary = {"models": {}, "rules": list(L.RULES)}

    header("Runs")
    for r in runs:
        print(
            f"{r['model']:13s} {r['name']} rev {r['revision']} ({str(r['sha'])[:12]}), {r['params_m']}M, "
            f"{len(r['prompt_tokens'])} prompts x {r['n_new']} tokens, torch {r['torch']}"
        )
    prompt_len = [len(p) for p in runs[0]["prompt_tokens"]]
    print(f"prompt length in tokens ({runs[0]['model']}): median {np.median(prompt_len):.0f}")

    header("Loops: share of continuations ending in a repeating block (95% Wilson interval)")
    stats = {}
    for r in runs:
        rows = []
        for rule in L.RULES:
            d = r["rules"][rule]
            s = summarise(d["tokens"], d["logprob"])
            stats[(r["model"], rule)] = s
            rows.append(f"{rule} {pct(s['rate'], 0)} [{pct(s['ci'][0], 0)}-{pct(s['ci'][1], 0)}]")
        ref = summarise(r["reference"]["tokens"], r["reference"]["logprob"])
        stats[(r["model"], "human")] = ref
        print(f"{r['model']:13s} " + "; ".join(rows) + f"; human text {pct(ref['rate'], 0)}")

    header("Greedy loops: where they start and how long the repeating block is (tokens)")
    for r in runs:
        s = stats[(r["model"], "greedy")]
        tails = [t for t in s["tails"] if t.looped]
        short = sum(t.period <= 4 for t in tails)
        print(
            f"{r['model']:13s} looped {s['looped']}/{s['n']}; onset median {s['onset_median']:.0f}; "
            f"period median {s['period_median']:.0f}; periods <= 4 tokens: {short}"
        )

    header("Greedy loops: block length (tokens) and share of greedy output inside the loop")
    for r in runs:
        s = stats[(r["model"], "greedy")]
        per = [t.period for t in s["tails"] if t.looped]
        b1 = sum(p <= 4 for p in per)
        b2 = sum(5 <= p <= 16 for p in per)
        b3 = sum(p >= 17 for p in per)
        inside = sum(r["n_new"] - t.start for t in s["tails"] if t.looped) / (s["n"] * r["n_new"])
        print(f"{r['model']:13s} periods 1-4: {b1}, 5-16: {b2}, 17-64: {b3}; tokens inside a loop {pct(inside, 0)}")

    header("Greedy loops by book (all models pooled)")
    by_book = {}
    for r in runs:
        for t, ps in zip(stats[(r["model"], "greedy")]["tails"], r["passages"], strict=True):
            by_book.setdefault(ps["book"], []).append(t.looped)
    for b, v in sorted(by_book.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{b:20s} {sum(v)}/{len(v)} ({pct(np.mean(v), 0)})")

    header("The same prompts across sizes (greedy)")
    for fam in sorted({r["family"] for r in runs}):
        fr = [r for r in runs if r["family"] == fam]
        flags = np.array([[t.looped for t in stats[(r["model"], "greedy")]["tails"]] for r in fr])
        all_ = int(flags.all(axis=0).sum())
        none = int((~flags).all(axis=0).sum())
        print(
            f"{fam} ({len(fr)} sizes): prompts that loop at every size {all_}; at no size {none}; of {flags.shape[1]}"
        )
    summary["greedy_onset"] = {
        r["model"]: [t.start if t.looped else -1 for t in stats[(r["model"], "greedy")]["tails"]] for r in runs
    }
    summary["books"] = [ps["book"] for ps in runs[0]["passages"]]

    header("Size against rule")
    for fam in sorted({r["family"] for r in runs}):
        fr = [r for r in runs if r["family"] == fam]
        small, large = fr[0]["model"], fr[-1]["model"]
        g_s, g_l = stats[(small, "greedy")]["rate"], stats[(large, "greedy")]["rate"]
        line = f"{fam}: greedy {small} {pct(g_s, 0)} -> {large} {pct(g_l, 0)}"
        for rule in ("T0.75", "T1", "top-p0.9"):
            a, b = stats[(small, rule)]["rate"], stats[(large, rule)]["rate"]
            line += f"; {rule} at {small} {pct(a, 0)}, at {large} {pct(b, 0)}"
        print(line)
    for rule in L.RULES:
        rates = [stats[(r["model"], rule)]["rate"] for r in runs]
        print(f"{rule}: loop rate across the {len(runs)} models {pct(min(rates), 0)}-{pct(max(rates), 0)}")

    header("Repetition and probability (mean over continuations)")
    for r in runs:
        parts = []
        for rule in [*L.RULES, "human"]:
            s = stats[(r["model"], rule)]
            parts.append(f"{rule} rep4 {pct(s['rep4'], 0)} lp {s['logprob']:.2f}")
        print(f"{r['model']:13s} " + "; ".join(parts))

    header("Memorisation: greedy continuations that reproduce the book")
    for r in runs:
        pref = [
            L.common_prefix(g, h) for g, h in zip(r["rules"]["greedy"]["tokens"], r["reference"]["tokens"], strict=True)
        ]
        mem = sum(p >= MEMORISED for p in pref)
        print(
            f"{r['model']:13s} first {MEMORISED}+ tokens identical to the book: {mem}/{len(pref)}; longest {max(pref)}"
        )
        for i in [i for i, p in enumerate(pref) if p >= MEMORISED]:
            print(f"    #{i} {passages[i].book}: {pref[i]} tokens; {r['rules']['greedy']['text'][i][:100]!r}")

    header("Greedy continuations that do not end in a loop: what they are")
    for r in runs:
        d = r["rules"]["greedy"]
        tails = stats[(r["model"], "greedy")]["tails"]
        rest = [i for i, t in enumerate(tails) if not t.looped]
        mem = [i for i in rest if L.common_prefix(d["tokens"][i], r["reference"]["tokens"][i]) >= MEMORISED]
        count = [i for i in rest if i not in mem and counting_loop(d["text"][i])]
        twice = [i for i in rest if i not in mem + count and repeated_twice(d["tokens"][i])]
        other = [i for i in rest if i not in mem + count + twice]
        rep = np.mean([L.repeated_ngram_share(d["tokens"][i], 4) for i in other]) if other else float("nan")
        print(
            f"{r['model']:13s} not looped {len(rest)}: recite the book {len(mem)}, count (digits masked) {len(count)}, "
            f"a block of up to 128 tokens repeated twice {len(twice)}, other {len(other)} (repeated 4-grams "
            f"{pct(rep, 0)})"
        )

    header("Self-reinforcement inside greedy loops: chosen-token probability and top-2 gap (nats)")
    for r in runs:
        s = stats[(r["model"], "greedy")]
        d = r["rules"]["greedy"]
        re_ = reinforcement(d["tokens"], d["logprob"], d["margin"], s["tails"])
        summary["models"].setdefault(r["model"], {})["reinforcement"] = re_
        cyc = ", ".join(
            f"c{i + 1} {p:.2f}/{m:.1f} (n={k})"
            for i, (p, m, k) in enumerate(zip(re_["cycle_p"], re_["cycle_m"], re_["cycle_n"], strict=True))
            if k
        )
        print(f"{r['model']:13s} before the loop {re_['pre_p']:.2f}/{re_['pre_m']:.1f}; {cyc}")

    header("Probability of the chosen token by position (mean over prompts): tokens 1, 16, 64, 128, 256")
    for r in runs:
        parts = []
        curves = {}
        for rule in ("greedy", "T0.5", "T1", "human"):
            lp = r["reference"]["logprob"] if rule == "human" else r["rules"][rule]["logprob"]
            n = min(len(x) for x in lp)
            curve = np.exp(np.array([x[:n] for x in lp])).mean(axis=0)
            curves[rule] = [round(float(v), 4) for v in curve]
            parts.append(f"{rule} " + "/".join(f"{curve[i - 1]:.2f}" for i in (1, 16, 64, 128, 256) if i <= n))
        summary["models"].setdefault(r["model"], {})["position_p"] = curves
        print(f"{r['model']:13s} " + "; ".join(parts))

    header("Loops that count: characters with every digit masked (max period 600 characters)")
    for r in runs:
        texts = r["rules"]["greedy"]["text"]
        masked = [L.periodic_tail(np.frombuffer(L.mask_digits(t).encode("utf-32-le"), dtype=np.uint32),
                                  max_period=600, min_tokens=60).looped for t in texts]  # fmt: skip
        tok = [t.looped for t in stats[(r["model"], "greedy")]["tails"]]
        bible = [i for i, ps in enumerate(r["passages"]) if ps["book"] == "bible-kjv"]
        bt, bm = sum(tok[i] for i in bible), sum(masked[i] for i in bible)
        print(
            f"{r['model']:13s} token loops {sum(tok)}/{len(tok)}; with digits masked {sum(masked)}/{len(masked)}; "
            f"Bible prompts: tokens {bt}/{len(bible)}, masked {bm}/{len(bible)}"
        )
    r0 = runs[0]
    for i in [i for i, ps in enumerate(r0["passages"]) if ps["book"] == "bible-kjv"][:2]:
        print(f"{r0['model']} Bible #{i}: {r0['rules']['greedy']['text'][i][-160:]!r}")
    for r in runs:  # a loop that only digit masking finds: the words repeat, the numbers count on
        texts = r["rules"]["greedy"]["text"]
        tails = stats[(r["model"], "greedy")]["tails"]
        for i in [i for i, ps in enumerate(r["passages"]) if ps["book"] == "bible-kjv"]:
            m = L.periodic_tail(np.frombuffer(L.mask_digits(texts[i]).encode("utf-32-le"), dtype=np.uint32),
                                max_period=600, min_tokens=60)  # fmt: skip
            if m.looped and not tails[i].looped:
                print(f"{r['model']} counting loop, Bible #{i}, period {m.period} characters: {texts[i][-240:]!r}")
                break

    header("Examples: the last 120 characters of greedy continuations that loop (prompts 0, 40, 80, 120 or next)")
    for r in runs:
        d = r["rules"]["greedy"]
        tails = stats[(r["model"], "greedy")]["tails"]
        for k0 in (0, 40, 80, 120):
            k = next((i for i in range(k0, len(tails)) if tails[i].looped), None)
            if k is not None:
                print(f"{r['model']:13s} #{k} [{passages[k].book}] p={tails[k].period} {d['text'][k][-120:]!r}")

    header("One prompt in full (#0, Emma): the prompt, the book, and the first 400 characters of each continuation")
    print(f"prompt: {passages[0].prompt!r}")
    print(f"book:   {passages[0].reference[:400]!r}")
    for r in runs:
        print(f"{r['model']:13s} greedy {r['rules']['greedy']['text'][0][:400]!r}")
    for r in (runs[-1], *[x for x in runs if x["family"] == "Pythia"][-1:]):
        for rule in ("T0.5", "T1", "top-p0.9"):
            print(f"{r['model']:13s} {rule} {r['rules'][rule]['text'][0][:400]!r}")

    header(f"Which words loop: share of function words (the {FUNCTION_TOP} commonest words of the 18 books)")
    counts = Counter(w.lower() for w in book_words() if w.isalpha())
    top = {w for w, _ in counts.most_common(FUNCTION_TOP)}
    print("the commonest 20: " + " ".join(w for w, _ in counts.most_common(20)))
    pooled_block, pooled_human = Counter(), Counter()
    for r in runs:
        vocab = {}
        blocks, prefixes, which = word_loops(r["rules"]["greedy"]["text"], vocab)
        k = len(which)
        from_prompt = sum(copied(b, alpha_words(passages[i].prompt)) for b, i in zip(blocks, which, strict=True))
        from_before = sum(copied(b, p) for b, p in zip(blocks, prefixes, strict=True))
        bw = [w for b in blocks for w in b]
        pw = [w for p in prefixes for w in p]
        hw = [w for t in r["reference"]["text"] for w in alpha_words(t)]
        sw = [w for t in r["rules"]["T1"]["text"] for w in alpha_words(t)]
        only = sum(all(w in top for w in b) for b in blocks)
        none = sum(not any(w in top for w in b) for b in blocks)
        pooled_block.update(bw)
        pooled_human.update(hw)
        print(
            f"{r['model']:13s} word loops {k}/{len(r['rules']['greedy']['text'])}; function-word share: loop block "
            f"{pct(share_in(bw, top), 0)}, greedy text before the loop {pct(share_in(pw, top), 0)}, sampled T1 "
            f"{pct(share_in(sw, top), 0)}, human {pct(share_in(hw, top), 0)}; blocks of function words only {only}, "
            f"with none {none}; block shares 4 consecutive words with the prompt {from_prompt}, with the greedy text "
            f"before the loop {from_before}"
        )
    nb, nh = sum(pooled_block.values()), sum(pooled_human.values())
    ratio = {w: (c / nb) / ((pooled_human[w] + 1) / nh) for w, c in pooled_block.items() if c >= 20}
    best = sorted(ratio, key=lambda w: -ratio[w])[:15]
    print("over-represented in loop blocks against human text (all models, words seen 20+ times in blocks): "
          + ", ".join(f"{w} x{ratio[w]:.1f}" for w in best))  # fmt: skip
    commonest = [w for w, _ in pooled_block.most_common(15)]
    print("commonest words in loop blocks: " + ", ".join(f"{w} {pct(pooled_block[w] / nb)}" for w in commonest))

    if not smoke:
        header("Word n-gram chains (memory of n - 1 words), greedy, 200 words after each prompt")
        ladder, n_words = ngram_ladder(passages)
        print(f"trained on the 18 books: {n_words:,} word and punctuation tokens")
        for n, v in ladder.items():
            print(
                f"n={n}: loops {pct(v['rate'], 0)}; onset median {v['onset']:.0f} words; period median "
                f"{v['period']:.0f}; e.g. {v['example'][:90]!r}"
            )
        summary["ngram"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "example"} for k, v in ladder.items()}

    for r in runs:
        m = summary["models"].setdefault(r["model"], {})
        m.update({"family": r["family"], "params_m": r["params_m"]})
        for rule in [*L.RULES, "human"]:
            s = stats[(r["model"], rule)]
            m[rule] = {k: v for k, v in s.items() if k != "tails"}
            m[rule]["onsets"] = [t.start for t in s["tails"] if t.looped]
            m[rule]["periods"] = [t.period for t in s["tails"] if t.looped]
    (RESULTS / ("summary_smoke.json" if smoke else "summary.json")).write_text(json.dumps(summary))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
