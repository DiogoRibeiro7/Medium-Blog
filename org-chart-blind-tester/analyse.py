"""Blind testers, anchored testers and testers from another model family: what each lets through.

Reads the responses in runs/*.jsonl (written by run.py), scores them against the MMLU answer key,
prints every number and draws every figure. It calls no model.

Sections
  1  Coverage: questions complete per model, unparseable answers, content-filter refusals.
  2  Error rates of the coder and of the blind tester (Wilson intervals).
  3  Error correlation between the coder and a blind tester: same model, and every other model.
     The correlation is the tetrachoric correlation of the two error indicators, the rho of the
     one-factor probit model in the org-chart article.
  4  Anchoring: how often a tester that has read the coder's answer adopts it on the questions where,
     blind, it would have disagreed; with and without a warning that the coder may be wrong. The check
     set tests the assumption that reading the answer never creates a disagreement.
  5  What each design lets through: the coder is wrong and the test expects the coder's answer, so the
     test passes. Anchored, warned, blind same-model, and blind testers from other families, with
     paired bootstrap intervals.
  6  The org-chart article's Model 1 against the measurements.

Run:  python analyse.py > results/output.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

from common import HERE, LETTERS, in_check_set, load_config, load_questions, parse_answer_how, read_model_runs, sample_order
from run import question_complete

B_BOOT = 2000

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
})


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ci(lo_hi, d: int = 1) -> str:
    return f"[{100 * lo_hi[0]:.{d}f}, {100 * lo_hi[1]:.{d}f}]"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


# ---------------------------------------------------------------------------------------------------
# Tetrachoric correlation of two error indicators (vectorised over bootstrap replicates)
# ---------------------------------------------------------------------------------------------------

def bvn_upper(h, k, rho):
    """P(X > h, Y > k) for a standard bivariate normal with correlation rho, via Owen's T.
    Arrays broadcast; h and k must be non-zero."""
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(rho, float))
    # P(X > h, Y > k) = P(X < -h, Y < -k) = Phi2(-h, -k; rho)
    a, b = -h, -k
    s = np.sqrt(np.clip(1 - rho * rho, 1e-15, None))
    t1 = special.owens_t(a, (b - rho * a) / (a * s))
    t2 = special.owens_t(b, (a - rho * b) / (b * s))
    beta = np.where(a * b > 0, 0.0, 0.5)
    out = 0.5 * (stats.norm.cdf(a) + stats.norm.cdf(b)) - t1 - t2 - beta
    return np.clip(out, 0.0, 1.0)


def tetrachoric(q1, q2, p11, iters: int = 60):
    """rho such that two latent normals, each above its threshold with probability q1, q2, are both
    above with probability p11. Bisection on [-0.999, 0.999]; values outside the attainable range
    are returned at the bound. NaN when a margin is 0 or 1."""
    q1, q2, p11 = (np.asarray(x, float) for x in (q1, q2, p11))
    ok = (q1 > 0) & (q1 < 1) & (q2 > 0) & (q2 < 1)
    h = stats.norm.ppf(1 - np.where(ok, q1, 0.5))
    k = stats.norm.ppf(1 - np.where(ok, q2, 0.5))
    h = np.where(h == 0, 1e-9, h)
    k = np.where(k == 0, 1e-9, k)
    lo = np.full(np.broadcast(q1, q2, p11).shape, -0.999)
    hi = np.full_like(lo, 0.999)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f = bvn_upper(h, k, mid) - p11
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return np.where(ok, (lo + hi) / 2, np.nan)


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def load(cfg, runs_dir: Path, questions_file: Path | None):
    rows, source = load_questions(cfg, questions_file)
    order = sample_order(rows, cfg)
    key = {q.qid: LETTERS[q.answer] for q in order}
    models = {}
    for spec in cfg["models"]:
        recs = read_model_runs(runs_dir, spec["key"])
        if not recs:
            continue
        for r in recs:  # score with the current parser, whatever the run recorded
            r["answer"], r["parsed_by"] = parse_answer_how(r.get("text"))
        prov = [json.loads(p.read_text())
                for p in sorted(runs_dir.glob(f"{spec['key']}.shard*.provenance.json"))]
        done = {(r["qid"], r["role"]): r for r in recs}
        n_target = max((r["position"] for r in recs), default=-1) + 1
        qs = [q for q in order if question_complete(q, done, cfg)]
        tab = {}
        for q in qs:
            tab[q.qid] = {role: done[(q.qid, role)] for role in ("coder", "blind", "anchored", "warned")
                          if (q.qid, role) in done}
        ids = sorted({r["model"] for r in recs})
        digests = sorted({p.get("digest") or "?" for p in prov})
        versions = sorted({p.get("server_version") or "?" for p in prov})
        models[spec["key"]] = {"spec": spec, "tab": tab, "ids": ids, "digests": digests, "versions": versions,
                               "records": recs, "positions_started": n_target}
    return order, key, source, models


def arrays(m: dict, key: dict, qids: list[int]) -> dict:
    """Per-question arrays for one model on the given questions. Missing answers are None."""
    tab = m["tab"]

    def ans(role):
        return np.array([(tab[q].get(role) or {}).get("answer") for q in qids], dtype=object)

    k = np.array([key[q] for q in qids], dtype=object)
    return {"qids": np.array(qids), "key": k, "coder": ans("coder"), "blind": ans("blind"),
            "anchored": ans("anchored"), "warned": ans("warned"),
            "has_anchored": np.array(["anchored" in tab[q] for q in qids]),
            "has_warned": np.array(["warned" in tab[q] for q in qids])}


def passes(coder, tester, key):
    """The coder is wrong and the test expects exactly the coder's answer: the test passes on a wrong
    answer. A coder with no parseable answer never passes (the function returns nothing a test can match)."""
    valid = np.array([c is not None for c in coder])
    return valid & (coder != key) & (tester == coder)


def wrong(ans, key):
    return np.array([a is None for a in ans]) | (ans != key)


def valid(ans):
    return np.array([a is not None for a in ans])


def plain(x: np.ndarray, idx: np.ndarray):
    """Mean, 95% bootstrap interval and the bootstrap replicates."""
    reps = x[idx].mean(axis=1)
    return float(x.mean()), (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))), reps


def followup_rates(cwrong, cright, stay, seen, agree, check, idx):
    """Misses and false alarms of a tester asked on every disagreement but, among agreements, only on
    the random check set. On disagreements both are counted directly. On agreements the coder's
    correctness is known for every question, and the check set estimates how often the tester stays
    with the coder's answer, separately for agreements where the coder is wrong and where it is right.
    Returns ((miss, ci, reps), (false alarm, ci, reps))."""
    stay = stay & seen

    def est(cw, cr, st, ag, ch):
        n = ag.shape[-1]
        dis = ~ag
        miss_d = (dis & cw & st).sum(-1)
        fa_d = (dis & cr & ~st).sum(-1)
        def retention(mask):
            k = (ag & ch & mask & st).sum(-1)
            m = (ag & ch & mask).sum(-1)
            return np.where(m > 0, k / np.maximum(m, 1), 1.0)
        r_w, r_r = retention(cw), retention(cr)
        miss = (miss_d + r_w * (ag & cw).sum(-1)) / n
        fa = (fa_d + (1 - r_r) * (ag & cr).sum(-1)) / n
        return miss, fa

    m0, f0 = est(cwrong[None], cright[None], stay[None], agree[None], check[None])
    mr, fr = est(cwrong[idx], cright[idx], stay[idx], agree[idx], check[idx])
    q = lambda r: (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))  # noqa: E731
    return (float(m0[0]), q(mr), mr), (float(f0[0]), q(fr), fr)


def boot_idx(n: int, rng) -> np.ndarray:
    return rng.integers(0, n, size=(B_BOOT, n))


def boot_mean(x: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    v = x.astype(float)[idx].mean(axis=1)
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))


def rho_with_ci(e1: np.ndarray, e2: np.ndarray, idx: np.ndarray):
    e1, e2 = e1.astype(float), e2.astype(float)
    point = float(tetrachoric(e1.mean(), e2.mean(), (e1 * e2).mean()))
    b1, b2 = e1[idx], e2[idx]
    reps = tetrachoric(b1.mean(1), b2.mean(1), (b1 * b2).mean(1))
    reps = reps[np.isfinite(reps)]
    lohi = (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))) if len(reps) else (np.nan, np.nan)
    return point, lohi


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--questions-file", type=Path)
    ap.add_argument("--fig-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--summary", type=Path, default=HERE / "results" / "summary.json")
    args = ap.parse_args(argv)

    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"] + 1)
    order, key, source, models = load(cfg, args.runs_dir, args.questions_file)
    if not models:
        print(f"no runs in {args.runs_dir}", file=sys.stderr)
        return 2
    keys = list(models)
    summary = {"questions_source": source, "models": {}, "pairs": {}}
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    # 1. Coverage ---------------------------------------------------------------------------------
    header("1. Coverage")
    print(f"Questions: {source}")
    print(f"Sampled order: {len(order)} questions, seed {cfg['seed']}; check set share {cfg['check_share']}")
    for k in keys:
        m = models[k]
        recs = m["records"]
        n_complete = len(m["tab"])
        unparsed = sum(r["answer"] is None for r in recs)
        fallback = sum(r["parsed_by"] == "fallback" for r in recs)
        truncated = sum(r.get("finish_reason") == "length" for r in recs)
        secs = [r["usage"]["seconds"] for r in recs if (r.get("usage") or {}).get("seconds")]
        by_role = {role: sum(r["role"] == role for r in recs) for role in ("coder", "blind", "anchored", "warned")}
        spec = m["spec"]
        print(f"  {k:<15} {spec.get('developer', '')}, {', '.join(m['ids'])}; digest {', '.join(d[:19] for d in m['digests'])}; "
              f"Ollama {', '.join(m['versions'])}")
        print(f"  {'':<15} {n_complete} questions complete; "
              f"{len(recs)} responses ({', '.join(f'{r} {n}' for r, n in by_role.items())}); "
              f"no answer found {unparsed}, answer read from 'the answer is X' {fallback}, cut at max_tokens {truncated}"
              + (f"; median {np.median(secs):.1f} s per call" if secs else ""))
        if len(m["digests"]) > 1:
            print(f"  {'':<15} WARNING: shards ran different weights")
        summary["models"][k] = {"ids": m["ids"], "digests": m["digests"], "complete": n_complete,
                                "responses": len(recs), "unparseable": unparsed, "fallback_parse": fallback,
                                "truncated": truncated}

    # 2. Error rates ------------------------------------------------------------------------------
    header("2. Error rates, on the answers that parsed (missing answers are counted separately)")
    A = {}
    for k in keys:
        qids = list(models[k]["tab"])
        a = arrays(models[k], key, qids)
        A[k] = a
        n = len(qids)
        out = {}
        for role in ("coder", "blind"):
            v = valid(a[role])
            e = wrong(a[role], a["key"]) & v
            out[role] = (int(e.sum()), int(v.sum()), n - int(v.sum()))
        (ec, nc, mc), (eb, nb, mb) = out["coder"], out["blind"]
        print(f"  {k:<15} n={n:4d}  coder {pct(ec / nc)} {ci(wilson(ec, nc))} (no answer {mc})   "
              f"blind tester {pct(eb / nb)} {ci(wilson(eb, nb))} (no answer {mb})")
        summary["models"][k].update(q_coder=ec / nc, q_blind=eb / nb, n=n, missing_coder=mc, missing_blind=mb)

    # 3. Error correlation ------------------------------------------------------------------------
    header("3. Error correlation between a coder and a blind tester (tetrachoric, 95% bootstrap interval)")
    print("Rows: the coder's model. Columns: the blind tester's model, on the questions both completed.")
    R = np.full((len(keys), len(keys)), np.nan)
    pair_stats = {}
    for i, kc in enumerate(keys):
        for j, kt in enumerate(keys):
            common = sorted(set(models[kc]["tab"]) & set(models[kt]["tab"]),
                            key=lambda q: [o.qid for o in order].index(q))
            if len(common) < 20:
                continue
            ac, at = arrays(models[kc], key, common), arrays(models[kt], key, common)
            ps_all = passes(ac["coder"], at["blind"], ac["key"])
            idx_all = boot_idx(len(common), rng)
            # the correlation is a property of answers, so it is estimated where both parsed
            ok = valid(ac["coder"]) & valid(at["blind"])
            ec, et = wrong(ac["coder"][ok], ac["key"][ok]), wrong(at["blind"][ok], at["key"][ok])
            idx = boot_idx(int(ok.sum()), rng)
            rho, rho_ci = rho_with_ci(ec, et, idx)
            both = ec & et
            same_opt = both & (ac["coder"][ok] == at["blind"][ok])
            R[i, j] = rho
            relation = ("same model" if kc == kt else "same family"
                        if models[kc]["spec"].get("family") == models[kt]["spec"].get("family") else "other family")
            pair_stats[(kc, kt)] = {"relation": relation, "n": int(ok.sum()), "n_common": len(common),
                                    "rho": rho, "rho_ci": rho_ci,
                                    "q_coder": float(ec.mean()), "q_tester": float(et.mean()),
                                    "both_wrong": float(both.mean()), "pass": float(ps_all.mean()),
                                    "pass_ci": boot_mean(ps_all, idx_all),
                                    "same_option_share": float(same_opt.sum() / both.sum()) if both.sum() else float("nan")}
    for (kc, kt), s in pair_stats.items():
        tag = f"{s['relation']:<12}"
        print(f"  coder {kc:<15} tester {kt:<15} [{tag}] n={s['n']:4d}  rho {s['rho']:+.2f} "
              f"[{s['rho_ci'][0]:+.2f}, {s['rho_ci'][1]:+.2f}]  both wrong {pct(s['both_wrong'])}  "
              f"same wrong option {pct(s['same_option_share'], 0) if np.isfinite(s['same_option_share']) else '-'} of those")
        summary["pairs"][f"{kc}|{kt}"] = {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in s.items()}

    print("Average correlation by relation (coder against blind tester):")
    for rel in ("same model", "same family", "other family"):
        v = [s["rho"] for s in pair_stats.values() if s["relation"] == rel and np.isfinite(s["rho"])]
        if v:
            print(f"  {rel:<12} {np.mean(v):+.2f} over {len(v)} pairs (range {min(v):+.2f} to {max(v):+.2f})")
            summary.setdefault("relations", {})[rel] = {"mean_rho": float(np.mean(v)), "pairs": len(v),
                                                        "min": float(min(v)), "max": float(max(v))}

    # 4. Anchoring --------------------------------------------------------------------------------
    header("4. Anchoring: the tester has read the coder's answer")
    anch = {}
    for k in keys:
        a = A[k]
        cvalid = np.array([c is not None for c in a["coder"]])
        dis = cvalid & (a["blind"] != a["coder"])
        chk = np.array([in_check_set(int(q), cfg) for q in a["qids"]])
        res = {}
        for role in ("anchored", "warned"):
            have = a[f"has_{role}"]
            d = dis & have
            adopted = d & (a[role] == a["coder"])
            agree_chk = chk & cvalid & (a["blind"] == a["coder"]) & have
            broke = agree_chk & (a[role] != a["coder"])
            # among disagreements, did the tester at least end on the right answer?
            right_after = d & (a[role] == a["key"])
            res[role] = {"disagreements": int(d.sum()), "adopted": int(adopted.sum()),
                         "a": float(adopted.sum() / d.sum()) if d.sum() else float("nan"),
                         "a_ci": wilson(int(adopted.sum()), int(d.sum())),
                         "check_agreements": int(agree_chk.sum()), "check_broken": int(broke.sum()),
                         "right_after": int(right_after.sum())}
        anch[k] = res
        r1, r2 = res["anchored"], res["warned"]
        print(f"  {k:<15} blind tester disagreed with the coder on {r1['disagreements']} questions.")
        print(f"  {'':<15} anchored tester adopted the coder's answer on {r1['adopted']} "
              f"({pct(r1['a'])} {ci(r1['a_ci'])}); warned tester on {r2['adopted']} ({pct(r2['a'])} {ci(r2['a_ci'])})")
        print(f"  {'':<15} check set: of {r1['check_agreements']} questions where blind tester and coder agreed, "
              f"reading the answer changed the anchored tester's answer in {r1['check_broken']} "
              f"and the warned tester's in {r2['check_broken']}")
        summary["models"][k].update(anchoring={kk: {k2: (list(v2) if isinstance(v2, tuple) else v2)
                                                    for k2, v2 in vv.items()} for kk, vv in res.items()})

    # 5. What each design lets through ------------------------------------------------------------
    header("5. Misses and false alarms of one tester after the coder")
    print("Miss: the coder is wrong and the test expects the coder's answer, so the test passes.\n"
          "False alarm: the coder is right and the test expects something else, so the test fails.\n"
          "Rates are shares of all questions. Anchored and warned testers were asked on every disagreement and\n"
          "on the random check set; on agreements outside it, the check set gives how often they stay with the\n"
          "coder's answer (separately where the coder is wrong and where it is right).")
    designs = {}
    for k in keys:
        a = A[k]
        n = len(a["qids"])
        idx = boot_idx(n, rng)
        cvalid = valid(a["coder"])
        cwrong = cvalid & (a["coder"] != a["key"])
        cright = cvalid & (a["coder"] == a["key"])
        agree = cvalid & (a["blind"] == a["coder"])
        check = np.array([in_check_set(int(q), cfg) for q in a["qids"]])
        rows = {}
        for role in ("anchored", "warned"):
            mi, fa = followup_rates(cwrong, cright, a[role] == a["coder"], a[f"has_{role}"], agree, check, idx)
            rows[role] = (mi, fa, None, None)
        bm = (cwrong & (a["blind"] == a["coder"])).astype(float)
        bf = (cright & (a["blind"] != a["coder"])).astype(float)
        rows["blind (same model)"] = (plain(bm, idx), plain(bf, idx), bm, bf)
        print(f"  coder {k} ({n} questions; coder wrong on {pct(cwrong.mean())}):")
        print(f"    {'tester':<24} {'misses':<22} false alarms")
        dk = {}
        q_w, q_r = cwrong.mean(), cright.mean()
        for name, (mi, fa, _, _) in rows.items():
            print(f"    {name:<24} {pct(mi[0])} {ci(mi[1]):<15} {pct(fa[0])} {ci(fa[1]):<15} "
                  f"= {pct(mi[0] / q_w, 0)} of the coder's errors pass, {pct(fa[0] / q_r, 0)} of its right answers fail")
            dk[name] = {"pass": mi[0], "ci": list(mi[1]), "false_alarm": fa[0], "false_alarm_ci": list(fa[1]),
                        "pass_share_of_errors": mi[0] / q_w, "alarm_share_of_right": fa[0] / q_r}
        # what reordering buys: anchored minus blind, same model, paired over the same bootstrap draws
        d_point = rows["anchored"][0][0] - rows["blind (same model)"][0][0]
        d_reps = rows["anchored"][0][2] - rows["blind (same model)"][0][2]
        lo, hi = np.quantile(d_reps, [0.025, 0.975])
        print(f"    reordering removes {pct(d_point)} [{pct(lo)}, {pct(hi)}] of misses "
              f"(anchored minus blind, same model)")
        dk["reorder_gain"] = {"value": d_point, "ci": [float(lo), float(hi)]}
        for kt in keys:
            if kt == k or (k, kt) not in pair_stats:
                continue
            s = pair_stats[(k, kt)]
            common = [q for q in a["qids"] if q in models[kt]["tab"]]
            ac, at = arrays(models[k], key, common), arrays(models[kt], key, common)
            cv = valid(ac["coder"])
            cw, cr = cv & (ac["coder"] != ac["key"]), cv & (ac["coder"] == ac["key"])
            om = (cw & (at["blind"] == ac["coder"])).astype(float)
            of = (cr & (at["blind"] != ac["coder"])).astype(float)
            sm = (cw & (ac["blind"] == ac["coder"])).astype(float)
            sf = (cr & (ac["blind"] != ac["coder"])).astype(float)
            idx2 = boot_idx(len(common), rng)
            mi, fa = plain(om, idx2), plain(of, idx2)
            diff = sm[idx2].mean(1) - om[idx2].mean(1)
            lo, hi = np.quantile(diff, [0.025, 0.975])
            label = f"blind {kt}" + (" (same family)" if s["relation"] == "same family" else "")
            d_miss, d_fa = sm.mean() - om.mean(), of.mean() - sf.mean()
            ratio = d_fa / d_miss if d_miss > 0 and d_fa > 0 else float("nan")
            print(f"    {label:<24} {pct(mi[0])} {ci(mi[1]):<15} {pct(fa[0])} {ci(fa[1]):<15} "
                  f"vs same-model blind: {pct(d_miss)} fewer misses [{pct(lo)}, {pct(hi)}], "
                  f"{pct(d_fa)} more false alarms; break-even cost ratio {ratio:.2f}; tester wrong on {pct(s['q_tester'])}")
            dk[f"blind {kt}"] = {"relation": s["relation"], "pass": mi[0], "ci": list(mi[1]),
                                 "false_alarm": fa[0], "false_alarm_ci": list(fa[1]), "n": len(common),
                                 "same_model_on_common": float(sm.mean()), "minus_same": float(sm.mean() - om.mean()),
                                 "minus_same_ci": [float(lo), float(hi)], "q_tester": s["q_tester"],
                                 "more_false_alarms": float(d_fa), "break_even_ratio": float(ratio),
                                 "pass_share_of_errors": float(om.sum() / cw.sum()) if cw.sum() else float("nan")}
        designs[k] = dk
        summary["models"][k]["designs"] = dk

    print("\nAverages over coder models (misses / false alarms):")
    pooled = {}
    for name in ("anchored", "warned", "blind (same model)"):
        v = [designs[k][name] for k in keys if name in designs[k]]
        pooled[name] = v
    for rel in ("same family", "other family"):
        pooled[f"blind, {rel}"] = [d for k in keys for nm, d in designs[k].items()
                                   if nm.startswith("blind ") and isinstance(d, dict) and d.get("relation") == rel]
    ratios = [d["break_even_ratio"] for k in keys for nm, d in designs[k].items()
              if nm.startswith("blind ") and isinstance(d, dict) and d.get("relation") == "other family"
              and np.isfinite(d.get("break_even_ratio", np.nan))]
    if ratios:
        print(f"Break-even cost ratio, other-family blind tester against same-model blind tester: median "
              f"{np.median(ratios):.2f} (range {min(ratios):.2f} to {max(ratios):.2f}, {len(ratios)} pairs). The\n"
              "other-family tester costs less whenever a missed error costs more than this many false alarms.")
        summary["break_even_ratio"] = {"median": float(np.median(ratios)), "min": float(min(ratios)),
                                       "max": float(max(ratios)), "pairs": len(ratios)}
    for name, v in pooled.items():
        if v:
            mi = np.mean([d["pass"] for d in v])
            fa = np.mean([d["false_alarm"] for d in v])
            print(f"  {name:<24} {pct(mi)} / {pct(fa)}  ({len(v)} coder-tester pairs)")
            summary.setdefault("pooled_designs", {})[name] = {"pass": float(mi), "false_alarm": float(fa), "pairs": len(v)}

    # 6. Model 1 against the measurements --------------------------------------------------------
    header("6. The org-chart article's Model 1, fed the measured rates")
    print("Model 1 counts an error as caught only if the reviewer is right; on multiple choice a reviewer that is\n"
          "wrong in a different way also flags it. 'Both wrong' is Model 1's residual; 'passes' is what a test lets through.")
    for k in keys:
        s = pair_stats.get((k, k))
        if not s:
            continue
        a_hat = anch[k]["anchored"]["a"]
        pred_anch = a_hat * s["q_coder"] + (1 - a_hat) * s["both_wrong"] if np.isfinite(a_hat) else float("nan")
        print(f"  {k:<15} q={pct(s['q_coder'])}, rho={s['rho']:+.2f}, a={pct(a_hat)}: Model 1 blind residual "
              f"{pct(s['both_wrong'])} (passes {pct(s['pass'])}); anchored residual {pct(pred_anch)} "
              f"(measured passes {pct(designs[k]['anchored']['pass'])})")

    # 7. Where reordering matters most ------------------------------------------------------------
    header("7. What reordering buys, by question difficulty")
    print("Difficulty of a question for one model: how many of the other models' coders got it wrong.\n"
          "Misses as a share of the questions in each group; anchored minus blind is what reordering removes.")
    groups = (("easy", lambda d, m: d <= m // 3), ("middle", lambda d, m: (d > m // 3) & (d < m - m // 3)),
              ("hard", lambda d, m: d >= m - m // 3))
    pooled_gain = {g: [] for g, _ in groups}
    for k in keys:
        a = A[k]
        others = [kt for kt in keys if kt != k]
        if not others:
            continue
        wrong_by = np.zeros(len(a["qids"]), dtype=int)
        for kt in others:
            tab = models[kt]["tab"]
            for i, q in enumerate(a["qids"]):
                r = (tab.get(int(q)) or {}).get("coder")
                if r is not None and r.get("answer") is not None and r["answer"] != key[int(q)]:
                    wrong_by[i] += 1
        m = len(others)
        cvalid = valid(a["coder"])
        cwrong = cvalid & (a["coder"] != a["key"])
        cright = cvalid & (a["coder"] == a["key"])
        agree = cvalid & (a["blind"] == a["coder"])
        check = np.array([in_check_set(int(q), cfg) for q in a["qids"]])
        line = []
        for g, f in groups:
            sel = np.where(f(wrong_by, m))[0]
            if len(sel) < 10:
                continue
            idx = boot_idx(len(sel), rng)
            (am, _, ar), _ = followup_rates(cwrong[sel], cright[sel], (a["anchored"] == a["coder"])[sel],
                                            a["has_anchored"][sel], agree[sel], check[sel], idx)
            bm = (cwrong[sel] & (a["blind"][sel] == a["coder"][sel])).astype(float)
            gain = am - bm.mean()
            lo, hi = np.quantile(ar - bm[idx].mean(1), [0.025, 0.975])
            pooled_gain[g].append(gain)
            line.append(f"{g} n={len(sel)}: coder wrong {pct(cwrong[sel].mean(), 0)}, anchored {pct(am)}, "
                        f"blind {pct(bm.mean())}, gain {pct(gain)} [{pct(lo)}, {pct(hi)}]")
            summary["models"][k].setdefault("by_difficulty", {})[g] = {
                "n": int(len(sel)), "coder_wrong": float(cwrong[sel].mean()), "anchored_miss": am,
                "blind_miss": float(bm.mean()), "gain": gain, "gain_ci": [float(lo), float(hi)]}
        pad = "\n  " + " " * 16
        print(f"  {k:<15} " + pad.join(line) if line else f"  {k:<15} too few questions")
    for g, v in pooled_gain.items():
        if v:
            print(f"  mean gain over models, {g}: {pct(np.mean(v))}")

    args.summary.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    draw(keys, models, R, pair_stats, anch, designs, args.fig_dir)
    return 0


# ---------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------

def draw(keys, models, R, pair_stats, anch, designs, fig_dir: Path) -> None:
    labels = keys

    # Figure 1: misses and false alarms of each design, per coder model
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.8 * len(keys) + 2.0), sharey=True)
    y = np.arange(len(keys))[::-1]
    styles = [("anchored", ORANGE, "o", "Read the coder's answer first"),
              ("warned", PURPLE, "D", "Read it, told the coder may be wrong"),
              ("blind (same model)", BLUE, "s", "Blind, same model")]
    for ax, metric, cikey, title in ((axes[0], "pass", "ci", "Misses: coder wrong, test passes"),
                                     (axes[1], "false_alarm", "false_alarm_ci", "False alarms: coder right, test fails")):
        for off, (name, col, mk, lab) in zip((0.24, 0.08, -0.08), styles):
            for yi, k in zip(y, keys):
                d = designs[k].get(name)
                if not d:
                    continue
                v, (lo, hi) = 100 * d[metric], d[cikey]
                ax.errorbar(v, yi + off, xerr=[[v - 100 * lo], [100 * hi - v]], fmt=mk, color=col, ms=5.5,
                            capsize=0, lw=1.3, label=lab if (yi == y[0] and ax is axes[0]) else None)
        seen = set()
        for yi, k in zip(y, keys):
            for name, d in designs[k].items():
                if not (name.startswith("blind ") and name != "blind (same model)" and isinstance(d, dict)):
                    continue
                rel = d.get("relation", "other family")
                col, lab = (AQUA, "Blind, another family") if rel == "other family" else \
                    (BLUE_RAMP[0], "Blind, same family, other model")
                ax.plot(100 * d[metric], yi - 0.24, "^", color=col, ms=5, alpha=0.85,
                        label=lab if (rel not in seen and ax is axes[0]) else None)
                seen.add(rel)
        ax.set_xlim(left=0)
        ax.set_xlabel("Share of all questions (%)")
        ax.set_title(title, loc="left", fontsize=10.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(keys)
    fig.legend(loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("One tester after the coder, by how the tester was set up", x=0.01, ha="left",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(fig_dir / "fig1_designs.png")
    plt.close(fig)

    # Figure 4: misses against false alarms, every design and pair
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    seen = set()
    for k in keys:
        for name, d in designs[k].items():
            if not isinstance(d, dict) or "false_alarm" not in d:
                continue
            if name == "anchored":
                col, mk, lab = ORANGE, "o", "Read the coder's answer first"
            elif name == "warned":
                col, mk, lab = PURPLE, "D", "Read it, told the coder may be wrong"
            elif name == "blind (same model)":
                col, mk, lab = BLUE, "s", "Blind, same model"
            elif d.get("relation") == "same family":
                col, mk, lab = BLUE_RAMP[0], "^", "Blind, same family, other model"
            else:
                col, mk, lab = AQUA, "^", "Blind, another family"
            ax.plot(100 * d["false_alarm"], 100 * d["pass"], mk, color=col, ms=6, alpha=0.85,
                    label=lab if lab not in seen else None)
            seen.add(lab)
    ax.set_xlabel("False alarms: coder right, test fails (% of questions)")
    ax.set_ylabel("Misses: coder wrong, test passes (% of questions)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Every coder and tester pairing: what passes against what is blocked", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_tradeoff.png")
    plt.close(fig)

    # Figure 2: error-correlation matrix
    fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 2.6, 1.0 * len(keys) + 1.6))
    im = ax.imshow(R, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(keys)):
        for j in range(len(keys)):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                        color="white" if R[i, j] > 0.6 else INK, fontweight="bold" if i == j else None)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Blind tester's model")
    ax.set_ylabel("Coder's model")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error correlation (tetrachoric)")
    ax.set_title("Error correlation, coder against blind tester", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_error_correlation.png")
    plt.close(fig)

    # Figure 3: anchoring
    fig, ax = plt.subplots(figsize=(8.0, 0.6 * len(keys) + 1.6))
    y = np.arange(len(keys))[::-1]
    for off, role, col, lab in ((0.15, "anchored", ORANGE, "Read the coder's answer"),
                                (-0.15, "warned", PURPLE, "Read it, told it may be wrong")):
        for yi, k in zip(y, keys):
            r = anch[k][role]
            if not np.isfinite(r["a"]):
                continue
            ax.barh(yi + off, 100 * r["a"], height=0.28, color=col, label=lab if yi == y[0] else None)
            ax.plot([100 * r["a_ci"][0], 100 * r["a_ci"][1]], [yi + off] * 2, color=INK2, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Of the questions where the blind tester disagreed, share where the tester adopted the coder's answer (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Reading the answer first", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_anchoring.png")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
