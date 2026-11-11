"""Part 4 of the actuarial-tariff-calibration series: two insurers pricing the same freMTPL2 portfolio.

Every number quoted in the article is printed by this script. Run:

    python market_numbers.py > output.txt     # a few minutes, single-threaded; writes figure_data.json

The data, the cleaning, the 60/20/20 split and the three frequency models (GLM-A with bonus-malus linear, GLM-B with
bonus-malus in bands, Poisson boosting) are those of Part 3 and are imported from part3/audit_numbers.py, so the
models here are the models audited there.

The market. Two insurers quote every policy of a sample; the premium of insurer j for policy i is its exposure times
its rate times its level, a pure premium with no expense or profit loading. A policyholder chooses insurer A with
probability 1 / (1 + (P_A / P_B) ** beta). With beta = infinity the cheaper quote always wins; beta = 0 is a market
that ignores price. Market outcomes are computed in expectation over these choice probabilities, with the observed
claims of each policy. The loss ratio of an insurer is the claims it receives divided by the claims its premiums
were meant to pay, so a loss ratio of 1.10 is a loss of 10 percent of premium.

Sections
1. The tariffs: the three models of Part 3 and transformations of the boosted model with exactly its ranking
   (powers rescaled to the training total, and level shifts).
2. A double lift chart and the ordered Lorenz curve with relativities (Frees, Meyers and Cummings 2014).
3. The relativity Gini matrix and the mini-max choice.
4. One-period markets on the test sample: shares and loss ratios for pairs of tariffs.
5. Where the business goes: GLM-A against GLM-B by bonus-malus band.
6. Sensitivity to beta, and the market as a whole.
7. Twins: the same model fitted on two halves of the training sample (estimation noise only).
8. Rate reviews: each insurer rescales its level by its own loss ratio, round after round, on the calibration
   sample; the final levels are then applied to the test sample.
9. Same ranking, different dispersion: powers of the boosted model against GLM-B.
10. Uncertainty: a bootstrap of the test sample and ten refitted splits; losses capped at 100,000 instead of counts.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "part3"))
import audit_numbers as p3  # noqa: E402
from tariffcal import gini, lorenz, premium_gini  # noqa: E402

SEED = p3.SEED                    # the split of Part 3
TWIN_SEED = 20140601
N_BOOT = 500
N_SPLITS = 10
ROUNDS = 40
BETAS = (0.0, 2.0, 5.0, 10.0, 20.0, np.inf)
POWERS = (0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
BM_EDGES = p3.BM_EDGES
BM_LABELS = p3.BM_LABELS
INF = np.inf


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def bstr(beta: float) -> str:
    return "inf" if np.isinf(beta) else f"{beta:g}"


# ---------------------------------------------------------------------------------------------------
# Market tools
# ---------------------------------------------------------------------------------------------------
def choice(PA: np.ndarray, PB: np.ndarray, beta: float) -> np.ndarray:
    """Probability that each policyholder chooses insurer A."""
    if np.isinf(beta):
        return np.where(PA < PB, 1.0, np.where(PA > PB, 0.0, 0.5))
    return 1.0 / (1.0 + (PA / PB) ** beta)


def market(ra, rb, e, N, beta, la=1.0, lb=1.0) -> dict:
    """One period: rates ra, rb (claims per policy-year) at levels la, lb; observed claims N."""
    PA, PB = e * ra * la, e * rb * lb
    w = choice(PA, PB, beta)
    pa, pb = (w * PA).sum(), ((1 - w) * PB).sum()
    ca, cb = (w * N).sum(), ((1 - w) * N).sum()
    return {"share": float((w * e).sum() / e.sum()), "lrA": float(ca / pa) if pa > 0 else np.nan,
            "lrB": float(cb / pb) if pb > 0 else np.nan, "mkt": float((ca + cb) / (pa + pb)),
            "premA": float(pa), "premB": float(pb), "w": w}


def rate_review(ra, rb, e, N, beta, rounds=ROUNDS) -> list[dict]:
    """Each round both insurers multiply their level by their own loss ratio of the round (full credibility)."""
    la = lb = 1.0
    path = []
    for t in range(rounds):
        m = market(ra, rb, e, N, beta, la, lb)
        path.append({"round": t + 1, "la": la, "lb": lb, "share": m["share"], "lrA": m["lrA"], "lrB": m["lrB"],
                     "mkt": m["mkt"]})
        if m["share"] < 1e-3 or m["share"] > 1 - 1e-3:
            break
        la, lb = la * m["lrA"], lb * m["lrB"]
    return path


def relativity_gini(base, score, e, N) -> float:
    """Gini of the ordered Lorenz curve of Frees, Meyers and Cummings (2014).

    Policies are sorted by the relativity score / base; the horizontal axis is the share of the base premium,
    the vertical axis the share of claims. Relativities are rounded to 10 significant digits so that scores which
    differ from the base by a constant factor form a single block instead of an order set by rounding noise.
    """
    rel = np.round(score / base, 10 - int(np.floor(np.log10(np.max(score / base)))))
    return gini(rel, e * base, N)


def double_lift(ra, rb, e, N, groups=10) -> list[dict]:
    """CAS Monograph 5 double lift chart: sort by ra / rb, equal-exposure groups, each average over the overall."""
    order = np.argsort(ra / rb, kind="stable")
    cum = np.cumsum(e[order]) - 0.5 * e[order]
    g = np.minimum((groups * cum / e.sum()).astype(int), groups - 1)
    lab = np.empty_like(g)
    lab[order] = g
    ex = np.bincount(lab, weights=e, minlength=groups)
    act = np.bincount(lab, weights=N, minlength=groups) / ex / (N.sum() / e.sum())
    pa = np.bincount(lab, weights=e * ra, minlength=groups) / ex / ((e * ra).sum() / e.sum())
    pb = np.bincount(lab, weights=e * rb, minlength=groups) / ex / ((e * rb).sum() / e.sum())
    lo = [float((ra / rb)[lab == j].min()) for j in range(groups)]
    hi = [float((ra / rb)[lab == j].max()) for j in range(groups)]
    return [{"actual": float(act[j]), "A": float(pa[j]), "B": float(pb[j]), "lo": lo[j], "hi": hi[j],
             "claims": float(N[lab == j].sum())} for j in range(groups)]


def tariffs(d: dict, models: dict, tr: np.ndarray, samples: dict) -> dict:
    """The three fitted models and transformations of the boosted model, each evaluated on every sample.

    Powers of the boosted rate are rescaled so that they collect the training claims, the convention of Parts 1
    and 2; they rank every policy exactly as the boosted model does.
    """
    pred = {name: p3.predict(models, d, idx) for name, idx in samples.items()}
    t = {k: {s: pred[s][k] for s in samples} for k in p3.MODELS}
    g_tr = p3.predict(models, d, tr)["GBM"]
    for pw in POWERS:
        if pw == 1.0:
            continue
        c = d["N"][tr].sum() / (d["e"][tr] * g_tr ** pw).sum()
        t[f"GBM^{pw:g}"] = {s: c * pred[s]["GBM"] ** pw for s in samples}
    for lv in (0.97, 1.03):
        t[f"GBM x{lv:g}"] = {s: lv * pred[s]["GBM"] for s in samples}
    return t


PAIRS = [("GLM-A", "GLM-B"), ("GLM-B", "GBM"), ("GLM-A", "GBM"), ("GBM^0.5", "GLM-B"), ("GBM^1.25", "GLM-B"),
         ("GBM^2", "GLM-B"), ("GBM x0.97", "GLM-B"), ("GBM x1.03", "GLM-B"), ("GBM^1.25", "GBM"),
         ("GBM^0.8", "GBM"), ("GBM", "GBM")]


def main() -> None:
    print("Part 4: two insurers pricing the same freMTPL2 portfolio.")
    d = p3.load()
    n = d["e"].size
    rng = np.random.default_rng(SEED)
    tr, ca, te = p3.split(n, rng)
    models = p3.fit_all(d, tr)
    samples = {"ca": ca, "te": te}
    T = tariffs(d, models, tr, samples)
    e, N = d["e"][te], d["N"][te]
    ec, Nc = d["e"][ca], d["N"][ca]

    # -----------------------------------------------------------------------------------------------
    header("1. The tariffs (test sample)")
    print("Same split and models as Part 3: training 406,794 policies, calibration 135,598, test 135,599.")
    print(f"{'tariff':<11}{'Gini':>8}{'premium Gini':>14}{'balance':>9}{'P90/P10 of rate':>17}")
    for k in ["GLM-A", "GLM-B", "GBM", "GBM^0.5", "GBM^0.8", "GBM^1.25", "GBM^2", "GBM x0.97", "GBM x1.03"]:
        r = T[k]["te"]
        q = np.quantile(r, [0.1, 0.9])
        print(f"{k:<11}{gini(r, e, N):8.4f}{premium_gini(r, e):14.4f}{(e * r).sum() / N.sum():9.4f}{q[1] / q[0]:17.2f}")
    print("(Every power and level of the boosted rate ranks the policies exactly as the boosted model does.)")

    # -----------------------------------------------------------------------------------------------
    header("2. Double lift charts and the ordered Lorenz curve with relativities (test sample)")
    dl = {}
    for a, b in (("GLM-A", "GLM-B"), ("GBM^2", "GLM-B")):
        dl[f"{a}|{b}"] = rows = double_lift(T[a]["te"], T[b]["te"], e, N)
        print(f"{a} against {b}: deciles of {a} / {b} (equal exposure); actual, {a} and {b} relative to their averages")
        for j, r_ in enumerate(rows):
            print(f"  {j + 1:2d}  ratio {r_['lo']:.3f}-{r_['hi']:.3f}  claims {r_['claims']:5.0f}  actual {r_['actual']:.3f}"
                  f"  {a} {r_['A']:.3f}  {b} {r_['B']:.3f}")
        errA = np.mean([abs(r_["A"] / r_["actual"] - 1) for r_ in rows])
        errB = np.mean([abs(r_["B"] / r_["actual"] - 1) for r_ in rows])
        print(f"  mean absolute percent error over the deciles: {a} {100 * errA:.1f}, {b} {100 * errB:.1f}")
    lor = {}
    for base, score in (("GLM-A", "GLM-B"), ("GLM-B", "GLM-A"), ("GLM-B", "GBM"), ("GBM", "GLM-B")):
        P, S = T[base]["te"], T[score]["te"]
        x, y = lorenz(S / P, e * P, N)
        lor[f"{base}|{score}"] = {"x": x, "y": y}
        lrA = (e * P).sum()
        cut = {}
        for lv in (0.97, 1.0, 1.03):
            m = market(P, S, e, N, INF, 1.0, lv)
            keep = m["w"] > 0.5
            xs, ys = (e * P)[~keep].sum() / lrA, N[~keep].sum() / N.sum()
            cut[lv] = (xs, ys)
            via_curve = (1 - ys) / (1 - xs) * N.sum() / lrA
            print(f"  base {base}, competitor {score} at level {lv:.2f}: competitor takes {100 * xs:.1f}% of the base"
                  f" premium and {100 * ys:.1f}% of the claims; base keeps loss ratio {m['lrA']:.4f}"
                  f" (from the curve {via_curve:.4f})")
        lor[f"{base}|{score}"]["cut"] = {f"{k:g}": v for k, v in cut.items()}
        print(f"  relativity Gini, base {base}, score {score}: {relativity_gini(P, S, e, N):.4f}")

    # -----------------------------------------------------------------------------------------------
    header("3. The relativity Gini matrix (rows: base premium; columns: competing score) and the mini-max choice")
    names = ["GLM-A", "GLM-B", "GBM", "GBM^0.5", "GBM^2", "GBM x0.97"]
    print(" " * 11 + "".join(f"{s:>11}" for s in names) + f"{'max':>9}")
    mx = {}
    for b in names:
        row = [relativity_gini(T[b]["te"], T[s]["te"], e, N) if s != b else 0.0 for s in names]
        mx[b] = max(row)
        print(f"{b:<11}" + "".join(f"{v:11.4f}" for v in row) + f"{mx[b]:9.4f}")
    print(f"mini-max choice (smallest maximal Gini): {min(mx, key=mx.get)}")
    print("Row GBM x0.97 equals row GBM: the relativity Gini does not change when a premium is multiplied by a constant.")

    # -----------------------------------------------------------------------------------------------
    header("4. One-period markets, cheapest quote wins (test sample)")
    print(f"{'A':>10} vs {'B':<9}{'share A':>9}{'LR A':>8}{'LR B':>8}{'market LR':>11}{'Gini A':>8}{'Gini B':>8}"
          f"{'rel Gini A|B':>14}{'rel Gini B|A':>14}")
    one = {}
    for a, b in PAIRS:
        m = market(T[a]["te"], T[b]["te"], e, N, INF)
        one[f"{a}|{b}"] = {k: v for k, v in m.items() if k != "w"}
        ga, gb_ = gini(T[a]["te"], e, N), gini(T[b]["te"], e, N)
        rab = relativity_gini(T[a]["te"], T[b]["te"], e, N) if a != b else 0.0
        rba = relativity_gini(T[b]["te"], T[a]["te"], e, N) if a != b else 0.0
        lrA = f"{m['lrA']:8.3f}" if np.isfinite(m["lrA"]) else f"{'-':>8}"
        lrB = f"{m['lrB']:8.3f}" if np.isfinite(m["lrB"]) else f"{'-':>8}"
        print(f"{a:>10} vs {b:<9}{m['share']:9.3f}{lrA}{lrB}{m['mkt']:11.3f}{ga:8.4f}{gb_:8.4f}{rab:14.4f}{rba:14.4f}")
    print("(share A: share of exposure won by A; rel Gini A|B: base A, competing score B, i.e. how much B can take from A.)")
    agree_rel = agree_gini = ties = total = 0
    for a, b in PAIRS:
        if a == b:
            continue
        o = one[f"{a}|{b}"]
        total += 1
        a_worse = o["lrA"] > o["lrB"]
        rab = relativity_gini(T[a]["te"], T[b]["te"], e, N)
        rba = relativity_gini(T[b]["te"], T[a]["te"], e, N)
        agree_rel += int((rab > rba) == a_worse)
        ga, gb_ = gini(T[a]["te"], e, N), gini(T[b]["te"], e, N)
        if abs(ga - gb_) < 1e-9:
            ties += 1
        else:
            agree_gini += int((ga < gb_) == a_worse)
    print(f"Of {total} pairs, the insurer with the higher loss ratio is the one with the larger relativity Gini against it"
          f" in {agree_rel}; the one with the lower stand-alone Gini in {agree_gini} of the {total - ties} pairs whose"
          f" Gini indices differ ({ties} pairs have identical Gini indices).")

    # -----------------------------------------------------------------------------------------------
    header("5. Where the business goes: GLM-A against GLM-B by bonus-malus band (test sample, cheapest wins)")
    band = np.digitize(d["bm"][te], BM_EDGES)
    m = market(T["GLM-A"]["te"], T["GLM-B"]["te"], e, N, INF)
    w = m["w"]
    bands = []
    for j, lab in enumerate(BM_LABELS):
        s = band == j
        sa = (w[s] * e[s]).sum() / e[s].sum()
        pa = (w[s] * e[s] * T["GLM-A"]["te"][s]).sum()
        pb = ((1 - w[s]) * e[s] * T["GLM-B"]["te"][s]).sum()
        lra = (w[s] * N[s]).sum() / pa if pa > 0 else np.nan
        lrb = ((1 - w[s]) * N[s]).sum() / pb if pb > 0 else np.nan
        bands.append({"band": lab, "shareA": float(sa), "lrA": float(lra), "lrB": float(lrb),
                      "claims": float(N[s].sum())})
        print(f"  {lab:<8} exposure share won by GLM-A {sa:.3f}; loss ratio of GLM-A's book {lra:.3f},"
              f" of GLM-B's book {lrb:.3f}; claims {N[s].sum():.0f}")
    print(f"  GLM-A's share of exposure {m['share']:.3f}, of claims {(w * N).sum() / N.sum():.3f}")

    # -----------------------------------------------------------------------------------------------
    header("6. Price sensitivity: shares and loss ratios as beta varies (test sample)")
    sens = {}
    for a, b in [("GLM-A", "GLM-B"), ("GLM-B", "GBM"), ("GBM^2", "GLM-B"), ("GBM x0.97", "GLM-B")]:
        for beta in BETAS:
            m = market(T[a]["te"], T[b]["te"], e, N, beta)
            sens[f"{a}|{b}|{bstr(beta)}"] = {k: v for k, v in m.items() if k != "w"}
            print(f"  {a:>9} vs {b:<6} beta {bstr(beta):>4}: share A {m['share']:.3f}, LR A {m['lrA']:.3f},"
                  f" LR B {m['lrB']:.3f}, market {m['mkt']:.3f}")

    # -----------------------------------------------------------------------------------------------
    header("7. Twins: the same specification fitted on two random halves of the training sample")
    rt = np.random.default_rng(TWIN_SEED)
    perm = rt.permutation(tr)
    h1, h2 = np.sort(perm[: tr.size // 2]), np.sort(perm[tr.size // 2:])
    m1, m2 = p3.fit_all(d, h1), p3.fit_all(d, h2)
    twins = {}
    for k in ("GLM-B", "GBM"):
        a_ = {s: p3.predict(m1, d, idx)[k] for s, idx in samples.items()}
        b_ = {s: p3.predict(m2, d, idx)[k] for s, idx in samples.items()}
        ratio = a_["te"] / b_["te"]
        q = np.quantile(ratio, [0.1, 0.5, 0.9])
        twins[k] = {"a": a_, "b": b_, "ratio_q": q}
        print(f"  {k}: Gini {gini(a_['te'], e, N):.4f} and {gini(b_['te'], e, N):.4f}; balance"
              f" {(e * a_['te']).sum() / N.sum():.4f} and {(e * b_['te']).sum() / N.sum():.4f};"
              f" ratio of the twins' rates: 10th percentile {q[0]:.3f}, median {q[1]:.3f}, 90th {q[2]:.3f}")
        for beta in BETAS:
            mm = market(a_["te"], b_["te"], e, N, beta)
            sens[f"twins {k}|{bstr(beta)}"] = {kk: v for kk, v in mm.items() if kk != "w"}
            print(f"    beta {bstr(beta):>4}: share A {mm['share']:.3f}, LR A {mm['lrA']:.3f}, LR B {mm['lrB']:.3f},"
                  f" market {mm['mkt']:.3f}, premium collected / claims {1 / mm['mkt']:.3f}")

    # -----------------------------------------------------------------------------------------------
    header(f"8. Rate reviews on the calibration sample (up to {ROUNDS} rounds), final levels applied to the test sample")
    reviews = {}
    review_pairs = [("GLM-A", "GLM-B"), ("GLM-B", "GBM"), ("GBM^1.25", "GBM"), ("GBM", "GBM")]
    for beta in (INF, 10.0):
        for a, b in review_pairs:
            path = rate_review(T[a]["ca"], T[b]["ca"], ec, Nc, beta)
            last = path[-1]
            mt = market(T[a]["te"], T[b]["te"], e, N, beta, last["la"], last["lb"])
            exit_round = next((p_["round"] for p_ in path if p_["share"] < 0.01 or p_["share"] > 0.99), None)
            reviews[f"{a}|{b}|{bstr(beta)}"] = {"path": path, "test": {k: v for k, v in mt.items() if k != "w"}}
            print(f"  {a:>9} vs {b:<6} beta {bstr(beta):>4}: rounds {len(path)}; round 1 share A {path[0]['share']:.3f}"
                  f" LR A {path[0]['lrA']:.3f} LR B {path[0]['lrB']:.3f}; final levels A {last['la']:.3f} B {last['lb']:.3f},"
                  f" share A {last['share']:.3f}; one side below 1% share from round {exit_round};"
                  f" test: share A {mt['share']:.3f}, market LR {mt['mkt']:.3f}")
        for k in ("GLM-B", "GBM"):
            tw = twins[k]
            path = rate_review(tw["a"]["ca"], tw["b"]["ca"], ec, Nc, beta)
            last = path[-1]
            mt = market(tw["a"]["te"], tw["b"]["te"], e, N, beta, last["la"], last["lb"])
            reviews[f"twins {k}|{bstr(beta)}"] = {"path": path, "test": {kk: v for kk, v in mt.items() if kk != "w"}}
            print(f"  twins {k:<6} beta {bstr(beta):>4}: rounds {len(path)}; final levels A {last['la']:.3f}"
                  f" B {last['lb']:.3f}, share A {last['share']:.3f}; test: share A {mt['share']:.3f},"
                  f" LR A {mt['lrA']:.3f}, LR B {mt['lrB']:.3f}, market LR {mt['mkt']:.3f},"
                  f" average level weighted by share {mt['share'] * last['la'] + (1 - mt['share']) * last['lb']:.3f}")

    # -----------------------------------------------------------------------------------------------
    header("9. Same ranking, different dispersion: powers of the boosted rate (Gini 0.3218) against GLM-B (0.2924)")
    powers = {}
    for pw in POWERS:
        k = "GBM" if pw == 1.0 else f"GBM^{pw:g}"
        row = {"premium_gini": premium_gini(T[k]["te"], e),
               "relAB": relativity_gini(T[k]["te"], T["GLM-B"]["te"], e, N),
               "relBA": relativity_gini(T["GLM-B"]["te"], T[k]["te"], e, N)}
        print(f"  power {pw:<5g} relativity Gini with base GBM^{pw:g} and score GLM-B {row['relAB']:.4f};"
              f" base GLM-B and score GBM^{pw:g} {row['relBA']:.4f}")
        for beta in (INF, 10.0):
            m0 = market(T[k]["te"], T["GLM-B"]["te"], e, N, beta)
            path = rate_review(T[k]["ca"], T["GLM-B"]["ca"], ec, Nc, beta)
            last = path[-1]
            mt = market(T[k]["te"], T["GLM-B"]["te"], e, N, beta, last["la"], last["lb"])
            row[bstr(beta)] = {"share0": m0["share"], "lrA0": m0["lrA"], "lrB0": m0["lrB"], "rounds": len(path),
                               "la": last["la"], "lb": last["lb"], "share_end": mt["share"], "mkt_end": mt["mkt"]}
            print(f"  power {pw:<5g} premium Gini {row['premium_gini']:.4f} beta {bstr(beta):>4}: one period share"
                  f" {m0['share']:.3f}, LR {m0['lrA']:.3f} against GLM-B {m0['lrB']:.3f}; after review: levels"
                  f" {last['la']:.3f} and {last['lb']:.3f}, test share {mt['share']:.3f}")
        powers[f"{pw:g}"] = row

    # -----------------------------------------------------------------------------------------------
    header(f"10. Uncertainty: paired bootstrap of the test sample ({N_BOOT} resamples) and {N_SPLITS} refitted splits")
    key_pairs = [("GLM-A", "GLM-B"), ("GLM-B", "GBM"), ("GBM^0.5", "GLM-B"), ("GBM^1.25", "GLM-B"), ("GBM^2", "GLM-B"),
                 ("GBM^1.25", "GBM")]
    rb = np.random.default_rng(SEED + 4)
    boot = {f"{a}|{b}": {"lrA": np.empty(N_BOOT), "lrB": np.empty(N_BOOT), "share": np.empty(N_BOOT)}
            for a, b in key_pairs}
    for i_ in range(N_BOOT):
        i = rb.integers(0, e.size, e.size)
        for a, b in key_pairs:
            m = market(T[a]["te"][i], T[b]["te"][i], e[i], N[i], INF)
            for kk in ("lrA", "lrB", "share"):
                boot[f"{a}|{b}"][kk][i_] = m[kk]
    for a, b in key_pairs:
        bt = boot[f"{a}|{b}"]
        diff = bt["lrA"] - bt["lrB"]
        print(f"  {a:>9} vs {b:<6}: LR A sd {bt['lrA'].std(ddof=1):.3f}, LR B sd {bt['lrB'].std(ddof=1):.3f},"
              f" LR A - LR B {one[f'{a}|{b}']['lrA'] - one[f'{a}|{b}']['lrB']:+.3f} (sd {diff.std(ddof=1):.3f},"
              f" positive in {np.mean(diff > 0):.3f}); LR A above 1 in {np.mean(bt['lrA'] > 1):.3f},"
              f" LR B above 1 in {np.mean(bt['lrB'] > 1):.3f}")

    # Losses capped at 100,000 instead of claim counts: premium = rate x average capped cost per claim (training).
    sev = d["loss_capped"][tr].sum() / d["N"][tr].sum()
    Lc = d["loss_capped"][te]
    print(f"  Capped losses: average capped cost per claim in the training sample {sev:,.0f} euros;"
          f" test balance of each tariff in money: " + ", ".join(
              f"{k} {(e * T[k]['te'] * sev).sum() / Lc.sum():.4f}" for k in ("GLM-A", "GLM-B", "GBM")))
    money = {}
    for a, b in key_pairs:
        m = market(T[a]["te"] * sev, T[b]["te"] * sev, e, Lc, INF)
        rb2 = np.random.default_rng(SEED + 5)
        da = np.empty(N_BOOT)
        for i_ in range(N_BOOT):
            i = rb2.integers(0, e.size, e.size)
            mm = market(T[a]["te"][i] * sev, T[b]["te"][i] * sev, e[i], Lc[i], INF)
            da[i_] = mm["lrA"] - mm["lrB"]
        money[f"{a}|{b}"] = {"lrA": m["lrA"], "lrB": m["lrB"]}
        print(f"    {a:>9} vs {b:<6}: money LR A {m['lrA']:.3f}, LR B {m['lrB']:.3f}, difference {m['lrA'] - m['lrB']:+.3f}"
              f" (sd {da.std(ddof=1):.3f}, positive in {np.mean(da > 0):.3f})")

    print()
    print("Refitted splits (cheapest wins; one-period market on each test sample; rate review on each calibration sample):")
    rs = np.random.default_rng(SEED + 2)
    reps = []
    for s_ in range(N_SPLITS):
        tr_s, ca_s, te_s = p3.split(n, rs)
        mod = p3.fit_all(d, tr_s)
        Ts = tariffs(d, mod, tr_s, {"ca": ca_s, "te": te_s})
        es, Ns, ecs, Ncs = d["e"][te_s], d["N"][te_s], d["e"][ca_s], d["N"][ca_s]
        row = {}
        for a, b in key_pairs:
            m = market(Ts[a]["te"], Ts[b]["te"], es, Ns, INF)
            path = rate_review(Ts[a]["ca"], Ts[b]["ca"], ecs, Ncs, INF)
            mt = market(Ts[a]["te"], Ts[b]["te"], es, Ns, INF, path[-1]["la"], path[-1]["lb"])
            row[f"{a}|{b}"] = {"share": m["share"], "lrA": m["lrA"], "lrB": m["lrB"], "mkt": m["mkt"],
                               "share_end": mt["share"], "ga": gini(Ts[a]["te"], es, Ns), "gb": gini(Ts[b]["te"], es, Ns)}
        reps.append(row)
        print(f"  split {s_ + 1:2d}: " + "; ".join(
            f"{a} vs {b} share {row[f'{a}|{b}']['share']:.3f} LR {row[f'{a}|{b}']['lrA']:.3f}/{row[f'{a}|{b}']['lrB']:.3f}"
            f" end {row[f'{a}|{b}']['share_end']:.3f}" for a, b in key_pairs))
    for a, b in key_pairs:
        k = f"{a}|{b}"
        sh = np.array([r[k]["share"] for r in reps])
        la = np.array([r[k]["lrA"] for r in reps])
        lb_ = np.array([r[k]["lrB"] for r in reps])
        mk = np.array([r[k]["mkt"] for r in reps])
        se = np.array([r[k]["share_end"] for r in reps])
        gd = np.array([r[k]["ga"] - r[k]["gb"] for r in reps])
        print(f"  {a:>9} vs {b:<6}: share A mean {sh.mean():.3f} [{sh.min():.3f}, {sh.max():.3f}]; LR A mean {la.mean():.3f}"
              f" [{la.min():.3f}, {la.max():.3f}]; LR B mean {lb_.mean():.3f} [{lb_.min():.3f}, {lb_.max():.3f}];"
              f" LR A > LR B in {np.sum(la > lb_)} of {N_SPLITS}; market LR mean {mk.mean():.3f}; share of A after review"
              f" below 0.5 in {np.sum(se < 0.5)} of {N_SPLITS} (range [{se.min():.3f}, {se.max():.3f}]);"
              f" Gini A - Gini B mean {gd.mean():+.4f}")

    # -----------------------------------------------------------------------------------------------
    out = {
        "double_lift": dl,
        "lorenz": {k: {"x": [round(float(v), 5) for v in val["x"][:: max(1, val["x"].size // 2000)]] + [1.0],
                       "y": [round(float(v), 5) for v in val["y"][:: max(1, val["y"].size // 2000)]] + [1.0],
                       "cut": {c: [round(float(z), 5) for z in xy] for c, xy in val["cut"].items()}}
                   for k, val in lor.items()},
        "bands": bands,
        "sens": {k: {kk: round(float(vv), 6) for kk, vv in v.items()} for k, v in sens.items()},
        "reviews": {k: [{kk: round(float(vv), 6) for kk, vv in p_.items()} for p_ in v["path"]] for k, v in reviews.items()},
        "powers": {k: {"premium_gini": round(float(v["premium_gini"]), 6), "relAB": round(float(v["relAB"]), 6),
                       "relBA": round(float(v["relBA"]), 6),
                       **{b_: {kk: round(float(vv), 6) for kk, vv in v[b_].items()} for b_ in ("inf", "10")}}
                   for k, v in powers.items()},
        "twin_ratio": {k: {"q": [round(float(x), 4) for x in v["ratio_q"]],
                           "hist": np.histogram(np.log(v["a"]["te"] / v["b"]["te"]), bins=np.linspace(-1.2, 1.2, 97))[0].tolist()}
                       for k, v in twins.items()},
    }
    (HERE / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
