"""Print every number used in the article, in the order the article uses them, and save the arrays the figures need.

python prevalence_numbers.py > output.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

import prevalence_core as C

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEED = 20261123
RNG = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(16)]
Y, N = C.FIELD_POS, C.FIELD_N
RAW = Y / N
WEIGHT_FACTOR = C.PUBLISHED_WEIGHTED / RAW  # v1 population weighting relative to the raw rate
CLAIM_UNWEIGHTED = C.PUBLISHED_RANGE[0] / WEIGHT_FACTOR


def pct(v, d=1):
    return f"{100 * float(v):.{d}f}%"


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def summary(name, post):
    q = post.quantile([0.025, 0.05, 0.5, 0.95, 0.975])
    lo, hi = post.shortest()
    print(
        f"{name}: median {pct(q[2], 2)}; central 95% {pct(q[0], 2)} to {pct(q[4], 2)};"
        f" central 90% {pct(q[1], 2)} to {pct(q[3], 2)}; shortest 95% {pct(lo, 2)} to {pct(hi, 2)}"
    )
    print(
        f"    P(prevalence < 0.25%) {pct(post.prob_below(0.0025))}; P(< 0.5%) {pct(post.prob_below(0.005))};"
        f" P(>= 1.2%) {pct(1 - post.prob_below(0.012))}; P(>= {pct(CLAIM_UNWEIGHTED, 2)}) "
        f"{pct(1 - post.prob_below(CLAIM_UNWEIGHTED))}"
    )
    return q


def survey():
    head("1. The survey and the published claim")
    print(f"{Y} positives in {N:,} tests: raw rate {pct(RAW, 2)}")
    print(
        f"v1 published: raw 95% CI {pct(C.PUBLISHED_RAW[0], 2)} to {pct(C.PUBLISHED_RAW[1], 2)};"
        f" population-weighted {pct(C.PUBLISHED_WEIGHTED, 2)}; weighted and test-adjusted range"
        f" {pct(C.PUBLISHED_RANGE[0], 2)} to {pct(C.PUBLISHED_RANGE[1], 2)}"
    )
    print(
        f"Weighting factor (weighted / raw) {WEIGHT_FACTOR:.3f}; the bottom of the published range, unweighted:"
        f" {pct(CLAIM_UNWEIGHTED, 2)}"
    )
    e, lo, hi = C.PUBLISHED_IJE_UNWEIGHTED
    print(f"2021 journal version, unweighted and test-adjusted: {pct(e)} (95% CI {pct(lo)} to {pct(hi)})")
    ys, ns = C.V1_SPEC
    yd, nd = C.V1_SENS
    print(
        f"v1 validation: {ys}/{ns} known negatives negative (specificity {pct(ys / ns, 2)});"
        f" {yd}/{nd} known positives positive (sensitivity {pct(yd / nd, 1)})"
    )
    ys2, ns2 = C.pooled(C.SPEC_STUDIES)
    yd2, nd2 = C.pooled(C.SENS_STUDIES)
    print(
        f"v2 validation, pooled: {ys2:,}/{ns2:,} (specificity {pct(ys2 / ns2, 2)}) in {len(C.SPEC_STUDIES)} studies;"
        f" {yd2}/{nd2} (sensitivity {pct(yd2 / nd2, 1)}) in {len(C.SENS_STUDIES)} studies"
    )


def false_alarms():
    head("2. Before the update: could the positives all be false?")
    for spec in (0.995, 0.99, 0.985):
        print(
            f"If nobody were infected and specificity were {pct(spec, 1)}:"
            f" expected false positives {N * (1 - spec):.1f}"
        )
    print(f"Specificity at which false positives alone average {Y}: {pct(1 - Y / N, 2)}")
    for name, (ys, ns) in (("v1", C.V1_SPEC), ("v2 pooled", C.pooled(C.SPEC_STUDIES))):
        a, b = ns - ys + 1.0, ys + 1.0  # false-positive rate ~ Beta(fp + 1, tn + 1)
        lo, hi = stats.beta.ppf([0.025, 0.975], b, a)
        pp = stats.betabinom(N, a, b)
        print(
            f"{name}: specificity 95% interval {pct(lo, 2)} to {pct(hi, 2)}; with nobody infected,"
            f" P(>= {Y} positives) = {pct(pp.sf(Y - 1), 1)}; expected positives {pp.mean():.1f}"
            f" (90% range {int(pp.ppf(0.05))} to {int(pp.ppf(0.95))})"
        )
    rng = RNG[0]
    fp_rate = rng.beta(C.V1_SPEC[1] - C.V1_SPEC[0] + 1.0, C.V1_SPEC[0] + 1.0, 20_000)
    fp_counts = rng.binomial(N, fp_rate)
    np.savez_compressed(RES / "false_alarms.npz", fp_counts=fp_counts)


def posteriors():
    head("3. The update: three routes to a posterior")
    out = {}
    p1, s1, d1 = C.pooled_posterior(C.V1_SPEC, C.V1_SENS, RNG[1])
    out["v1"] = summary("Version 1 data, pooled (399/401, 103/122)", p1)
    pis, specs = C.joint_draws(Y, N, s1, d1, p1, RNG[2])
    print(f"    correlation of prevalence and specificity in the joint posterior: {np.corrcoef(pis, specs)[0, 1]:.2f}")
    sp_q = np.quantile(specs, [0.025, 0.5, 0.975])
    print(f"    posterior specificity: median {pct(sp_q[1], 2)}, 95% {pct(sp_q[0], 2)} to {pct(sp_q[2], 2)}")
    print("    Gelman and Carpenter (2020), same model and data: shortest 95% interval 0 to 1.8%")
    p2, s2, d2 = C.pooled_posterior(C.pooled(C.SPEC_STUDIES), C.pooled(C.SENS_STUDIES), RNG[3])
    out["v2"] = summary("Version 2 data, pooled across studies (3308/3324, 130/157)", p2)
    print("    2021 journal version (pooled, unweighted, test-adjusted): 1.2%, 95% CI 0.7% to 1.8%")
    pis2, specs2 = C.joint_draws(Y, N, s2, d2, p2, RNG[4])
    np.savez_compressed(
        RES / "posteriors.npz",
        grid=p1.grid,
        v1=p1.mass,
        v2=p2.mass,
        pis_v1=pis,
        specs_v1=specs,
        pis_v2=pis2,
        specs_v2=specs2,
    )
    return p2


def heterogeneity():
    head("4. Checking the pooling assumption: do the 13 specificity studies agree?")
    for (y, n), i in zip(C.SPEC_STUDIES, range(1, 14), strict=True):
        print(f"  study {i:>2}: {n - y} false positives in {n:,}")
    ppc = C.pooled_ppc(RNG[5])
    print(
        f"Posterior predictive check under one shared specificity: statistic {ppc['t_obs']:.1f},"
        f" p = {ppc['p_value']:.3f}"
    )
    print(
        f"Under pooling, chance of no false positive in the {ppc['largest_n']:,}-sample study:"
        f" {pct(ppc['p_zero_in_largest'])}"
    )


def hierarchical():
    head("5. Letting specificity and sensitivity vary between studies (hierarchical model)")
    print(
        "Priors as Gelman and Carpenter: mu ~ Normal(4, 2) on the logit scale; between-study sd ~ half-Normal(0, tau)"
    )
    keep, sigmas = {}, {}
    for tau in (1.0, 0.5, 0.3, 0.2, 0.1):
        post, spec, sens, hs, hd = C.hierarchical_posterior(tau, tau, RNG[6], k=200_000 if tau == 1.0 else 100_000)
        print(
            f"tau = {tau}: hyper-grid edge mass {C.hyper_edge_mass(hs):.1e} (spec), {C.hyper_edge_mass(hd):.1e} (sens)"
        )
        summary(f"    prevalence, tau {tau}", post)
        sig_s = np.sum(hs.mass, axis=0)
        sig_med = float(np.interp(0.5, np.cumsum(sig_s), hs.sigma))
        sp = np.quantile(spec, [0.025, 0.5, 0.975])
        se = np.quantile(sens, [0.025, 0.5, 0.975])
        print(f"    between-study sd of logit specificity: posterior median {sig_med:.2f}")
        sigmas[tau] = sig_med
        print(
            f"    a new lab's specificity before the survey: median {pct(sp[1], 2)},"
            f" 95% {pct(sp[0], 2)} to {pct(sp[2], 2)};"
            f" sensitivity median {pct(se[1])}, 95% {pct(se[0])} to {pct(se[2])}"
        )
        keep[tau] = post
    print(
        "Gelman and Carpenter (2020): weak priors (tau 1) 0.3%, 1.7%, 35.4% (2.5%, median, 97.5%);"
        " stronger priors 0.3%, 1.2%, 2.1%"
    )
    np.savez_compressed(RES / "hierarchical.npz", grid=keep[1.0].grid, **{f"tau_{t}": p.mass for t, p in keep.items()})
    return keep, sigmas


def calibration_check():
    head("6. Checking the engine: does the posterior cover the truth as often as it says?")
    u = C.calibration(RNG[7])
    c90 = np.mean((u > 0.05) & (u < 0.95))
    c50 = np.mean((u > 0.25) & (u < 0.75))
    print(
        f"{u.size} simulated surveys from the model's own priors: 90% intervals cover {pct(c90)}, 50% cover {pct(c50)};"
        f" uniformity of posterior quantiles, KS p = {stats.kstest(u, 'uniform').pvalue:.2f}"
    )


def decision_one(v2, hier):
    head("7. Decision 1: could the survey support acting on a prevalence of 2.5% to 4.2%?")
    print(
        f"Claim threshold in unweighted terms (bottom of the published range / weighting factor):"
        f" {pct(CLAIM_UNWEIGHTED, 2)}"
    )
    rows = (
        ("pooled, v2", v2),
        ("hierarchical, tau 0.3", hier[0.3]),
        ("hierarchical, tau 0.2", hier[0.2]),
        ("hierarchical, tau 1", hier[1.0]),
    )
    for name, post in rows:
        print(
            f"  {name:>22}: P(prevalence >= {pct(CLAIM_UNWEIGHTED, 2)}) {pct(1 - post.prob_below(CLAIM_UNWEIGHTED))};"
            f" P(< 0.5%) {pct(post.prob_below(0.005))}; P(< 0.25%) {pct(post.prob_below(0.0025))}"
        )
    print("Bayes-optimal point estimate under linear loss with under-estimates k times as costly: the k/(k+1) quantile")
    for name, post in rows[:3]:
        qs = post.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(
            f"  {name:>22}: 10% {pct(qs[0], 2)}, 25% {pct(qs[1], 2)}, 50% {pct(qs[2], 2)}, 75% {pct(qs[3], 2)},"
            f" 90% {pct(qs[4], 2)}; weighted (x{WEIGHT_FACTOR:.2f}): 10% {pct(WEIGHT_FACTOR * qs[0], 2)},"
            f" 50% {pct(WEIGHT_FACTOR * qs[2], 2)}, 90% {pct(WEIGHT_FACTOR * qs[4], 2)}"
        )


SPLITS = (100, 200, 300, 401, 600, 800, 1000, 1300, 1600, 2000, 2500)
SCENARIOS = (0.005, 0.012, 0.03)
BUDGET = C.FIELD_N + C.V1_SPEC[1] + C.V1_SENS[1]
TRUE_SPEC, TRUE_SENS = 0.995, 0.83


def decision_two():
    head("8. Decision 2: how to split a fixed number of tests between validation and the survey")
    print(
        f"Budget {BUDGET:,} tests (Santa Clara: {C.FIELD_N:,} survey + {C.V1_SPEC[1]} known negatives"
        f" + {C.V1_SENS[1]} known positives); known positives held at {C.V1_SENS[1]}"
    )
    print(
        f"Same-lab validation; true specificity {pct(TRUE_SPEC, 1)}, sensitivity {pct(TRUE_SENS, 0)}; 300 simulated"
        " surveys per split; posterior 5% and 95% quantiles of prevalence"
    )
    widths = np.empty((len(SCENARIOS), len(SPLITS)))
    resolve = np.empty_like(widths)
    for i, pi in enumerate(SCENARIOS):
        rng = RNG[8 + i]
        for j, ns in enumerate(SPLITS):
            q05, q95 = C.design_widths(
                ns, C.V1_SENS[1], BUDGET - ns - C.V1_SENS[1], pi, TRUE_SPEC, TRUE_SENS, rng, reps=300
            )
            widths[i, j] = np.median(q95 - q05)
            resolve[i, j] = np.mean(q05 > pi / 2)
        best = int(np.argmin(widths[i]))
        sc = SPLITS.index(401)
        print(f"  true prevalence {pct(pi, 1)}:")
        print("    known negatives: " + ", ".join(f"{s}" for s in SPLITS))
        print("    median 90% width: " + ", ".join(f"{100 * w:.2f}" for w in widths[i]) + " (points)")
        print("    P(lower 5% bound above half the truth): " + ", ".join(f"{100 * r:.0f}%" for r in resolve[i]))
        print(
            f"    narrowest at {SPLITS[best]} known negatives ({pct(SPLITS[best] / BUDGET, 0)} of the budget):"
            f" {100 * widths[i, best]:.2f} points against {100 * widths[i, sc]:.2f} at Santa Clara's 401"
            f" ({100 * (1 - widths[i, best] / widths[i, sc]):.0f}% narrower)"
        )
    print("Larger budgets at a true prevalence of 0.5% (share of tests on known negatives 10%, 25%, 40%):")
    big = {}
    for mult in (2, 4, 8):
        b = BUDGET * mult
        row = []
        for share in (0.10, 0.25, 0.40):
            ns = int(round(share * b))
            q05, q95 = C.design_widths(
                ns, C.V1_SENS[1] * mult, b - ns - C.V1_SENS[1] * mult, 0.005, TRUE_SPEC, TRUE_SENS, RNG[11], reps=200
            )
            row.append((np.median(q95 - q05), np.mean(q05 > 0.0025)))
        big[mult] = row
        print(
            f"  {b:,} tests: "
            + "; ".join(
                f"{int(s * 100)}%: width {100 * w:.2f} points, resolved {100 * r:.0f}%"
                for s, (w, r) in zip((0.10, 0.25, 0.40), row, strict=True)
            )
        )
    np.savez_compressed(
        RES / "design.npz",
        splits=np.array(SPLITS),
        scenarios=np.array(SCENARIOS),
        widths=widths,
        resolve=resolve,
        budget=BUDGET,
    )


def floor(sigma_est):
    head("9. Validation in other labs has a floor; validation in the survey's own lab does not")
    print(f"Between-lab sd {sigma_est:.2f} on the logit scale is the posterior median under tau 0.3")
    sens_yn = C.pooled(C.SENS_STUDIES)
    sizes = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
    same = []
    for n_neg in sizes:
        post = C.same_lab(n_neg, TRUE_SPEC, sens_yn, RNG[12], k=50_000)
        lo, hi = post.quantile([0.025, 0.975])
        same.append((lo, hi))
        print(
            f"  same lab, {n_neg:>6,} known negatives: 95% interval {pct(lo, 2)} to {pct(hi, 2)}"
            f" (width {100 * (hi - lo):.2f} points)"
        )
    floors = {}
    for sigma in (0.25, 0.5, round(sigma_est, 2), 1.0):
        post = C.other_lab_floor(sigma, TRUE_SPEC, sens_yn, RNG[13], k=50_000)
        lo, hi = post.quantile([0.025, 0.975])
        floors[sigma] = (lo, hi)
        sp = C.special.expit(C.special.logit(TRUE_SPEC) + sigma * np.array([-1.96, 1.96]))
        print(
            f"  other labs only, between-lab sd {sigma} (new lab's specificity 95% {pct(sp[0], 2)} to {pct(sp[1], 2)}):"
            f" 95% interval {pct(lo, 2)} to {pct(hi, 2)} (width {100 * (hi - lo):.2f} points), however many samples"
        )
    np.savez_compressed(
        RES / "floor.npz",
        sizes=np.array(sizes),
        same=np.array(same),
        sigmas=np.array(list(floors)),
        floors=np.array(list(floors.values())),
    )


def main():
    RES.mkdir(exist_ok=True)
    print(
        "Companion numbers for the article on estimating prevalence from an imperfect test (Santa Clara, April 2020)."
    )
    print(f"Seed {SEED}. NumPy {np.__version__}.")
    survey()
    false_alarms()
    v2 = posteriors()
    heterogeneity()
    hier, sigmas = hierarchical()
    calibration_check()
    decision_one(v2, hier)
    decision_two()
    floor(sigmas[0.3])


if __name__ == "__main__":
    main()
