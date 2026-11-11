"""Part 2 of the actuarial-tariff-calibration series: the mechanism, in a simulation with a known true premium.

Every number quoted in the article is printed by this script. Run:

    python mechanism_numbers.py > output.txt      # a few minutes, single-threaded; no downloads

A synthetic motor portfolio is drawn from a known claim-frequency function, with unobserved heterogeneity and
exposure that depends on the driver's age. Because the true frequency mu(x) is known, population quantities can be
computed without sampling noise from a large pool of policies: the expected Poisson deviance of a rate r in excess
of the truth is the Bregman divergence 2 sum e [mu log(mu / r) - (mu - r)], and the recalibrated rate is the
isotonic regression of mu on r.

Sections
1. The portfolio and its true frequency.
2. One ranking, many prices: powers and levels of the true frequency. The Gini of the claims is unchanged; the
   Gini of the premium itself, miscalibration and deviance are not.
3. The geometry: in-sample isotonic recalibration is the greatest convex minorant of the ordered Lorenz curve,
   so it can only raise the Gini; discrimination is a Kullback-Leibler divergence over the isotonic blocks.
4. Four fitted models (a binned Poisson GLM, a Poisson gradient-boosted model, a claim classifier, and a boosted
   model fitted without the exposure offset): Gini, balance and the Murphy decomposition on a test sample and in
   the population.
5. Isotonic recalibration on a separate calibration sample: what it repairs, what it costs, and how many
   premium levels it leaves.
6. Model selection on test samples of 170,000 policies: how often the Gini and the deviance choose each model.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # scikit-learn's boosting uses OpenMP; one thread keeps sums in order

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tariffcal import (StepRecalibration, gini, isotonic_recalibration, lorenz, lower_convex_hull,  # noqa: E402
                       murphy, poisson_deviance, premium_gini)

SEED = 19600523  # the first day of the CAS meeting at which Bailey and Simon's paper was presented
SS = np.random.SeedSequence(SEED)
RNG_TRAIN, RNG_CAL, RNG_TEST, RNG_POOL, RNG_NOISE, RNG_REP = (np.random.default_rng(s) for s in SS.spawn(6))

N_TRAIN, N_CAL, N_TEST, N_POOL = 500_000, 200_000, 500_000, 2_000_000
N_REP, N_REP_SIZE = 400, 170_000

REGION_P = np.array([0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.05])
REGION_EFFECT = np.array([0.0, 0.10, -0.08, 0.18, -0.15, 0.05, 0.25, -0.20])
FEATURES = ("driver age", "bonus-malus", "vehicle power", "vehicle age", "log10 density", "region")
HET_SHAPE = 2.0  # unobserved heterogeneity: Gamma with mean 1 and variance 1 / HET_SHAPE


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------------------------------
# The portfolio
# ---------------------------------------------------------------------------------------------------
def simulate_policies(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Covariates and exposure of n policies. Young drivers have worse bonus-malus and more part-year policies."""
    age = 18 + 72 * rng.beta(2.0, 3.2, n)
    young = age < 26
    bm_excess = np.where(rng.random(n) < np.where(young, 0.75, 0.30), rng.gamma(1.5, np.where(young, 25, 15), n), 0.0)
    bm = np.clip(50 + bm_excess, 50, 230).round()
    power = np.clip(rng.poisson(2.5, n) + 4, 4, 15).astype(float)
    vage = np.clip(rng.gamma(2.0, 3.5, n), 0, 25).round()
    logdens = np.clip(rng.normal(2.6, 0.8, n), 0, 4.5)
    region = rng.choice(8, n, p=REGION_P).astype(float)
    full_year = rng.random(n) < np.where(young, 0.35, 0.60)
    exposure = np.where(full_year, 1.0, rng.uniform(0.03, 1.0, n))
    return np.column_stack([age, bm, power, vage, logdens, region]), exposure


def true_frequency(X: np.ndarray) -> np.ndarray:
    """The true expected claims per policy-year: young drivers, bonus-malus, density, region, vehicle age, and an
    interaction between youth and vehicle power that a main-effects GLM cannot represent."""
    age, bm, power, vage, logdens, region = X.T
    eta = (np.log(0.065)
           + 0.9 * np.exp(-(age - 18) / 6.0) + 0.25 * np.maximum(age - 70, 0) / 10
           + 1.6 * np.log(bm / 50)
           + 0.10 * logdens
           + REGION_EFFECT[region.astype(int)]
           - 0.025 * vage
           + 0.012 * (power - 6) + 0.06 * (power - 6) * (age < 30))
    return np.exp(eta)


def draw_claims(exposure: np.ndarray, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = rng.gamma(HET_SHAPE, 1.0 / HET_SHAPE, exposure.size)
    return rng.poisson(exposure * mu * theta).astype(float)


def sample(n: int, rng: np.random.Generator) -> dict:
    X, e = simulate_policies(n, rng)
    mu = true_frequency(X)
    return {"X": X, "e": e, "mu": mu, "N": draw_claims(e, mu, rng)}


# ---------------------------------------------------------------------------------------------------
# Population quantities from the known truth
# ---------------------------------------------------------------------------------------------------
def bregman(mu: np.ndarray, r: np.ndarray, e: np.ndarray) -> float:
    """Exposure-weighted Poisson Bregman divergence 2 sum e [mu log(mu / r) - (mu - r)]; infinite if r = 0 < mu."""
    if np.any((r <= 0) & (mu > 0)):
        return float("inf")
    return float(2.0 * np.sum(e * (mu * np.log(mu / r) - (mu - r))))


def population(rate: np.ndarray, pool: dict) -> dict:
    """Gini, balance and the Murphy decomposition of a rate in the population, per 1,000 policy-years.

    excess = expected deviance of the rate minus that of the true frequency = MCB + lost discrimination,
    MCB = divergence of the rate from its recalibration E[mu | rate], DSC = divergence of the recalibration from
    the flat rate, lost = DSC of the truth minus DSC of the rate.
    """
    e, mu = pool["e"], pool["mu"]
    rc = isotonic_recalibration(rate, e, e * mu).predict(rate)
    flat = np.full_like(rc, (e * mu).sum() / e.sum())
    per = e.sum() / 1000.0
    return {
        "gini": gini(rate, e, e * mu),
        "premium_gini": premium_gini(rate, e),
        "balance": (e * rate).sum() / (e * mu).sum(),
        "excess": bregman(mu, rate, e) / per,
        "MCB": bregman(rc, rate, e) / per,
        "DSC": bregman(rc, flat, e) / per,
        "lost": bregman(mu, rc, e) / per,
    }


def where_it_counts(rate: np.ndarray, pool: dict, top: float = 0.1) -> dict:
    """Share of the population DSC and of the Gini area contributed by the top `top` share of exposure (by rate).

    DSC contributions are per-policy Bregman divergences of the recalibrated rate from the flat rate; the Gini
    contribution is twice the area between the diagonal and the concentration curve over the last `top` of the
    horizontal axis.
    """
    e, mu = pool["e"], pool["mu"]
    rc = isotonic_recalibration(rate, e, e * mu).predict(rate)
    c = (e * mu).sum() / e.sum()
    contrib = 2.0 * e * (rc * np.log(rc / c) - (rc - c))
    order = np.argsort(rate, kind="stable")
    cum = np.cumsum(e[order]) / e.sum()
    in_top = np.zeros(rate.size, bool)
    in_top[order[cum > 1 - top]] = True
    cw, cc = lorenz(rate, e, e * mu)
    grid = np.linspace(0.0, 1.0, 200_001)
    gap = grid - np.interp(grid, cw, cc)
    total_area = np.trapezoid(gap, grid)
    top_area = np.trapezoid(gap[grid >= 1 - top], grid[grid >= 1 - top])
    per = e.sum() / 1000.0
    return {"dsc_top": contrib[in_top].sum() / per, "dsc_rest": contrib[~in_top].sum() / per,
            "gini_top": 2 * top_area, "gini_rest": 2 * (total_area - top_area)}


def rebalance(rate: np.ndarray, e: np.ndarray, target: float) -> np.ndarray:
    """Scale a rate so that its total premium on exposure e equals target."""
    return rate * target / (e * rate).sum()


# ---------------------------------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------------------------------
GLM_BANDS = {0: [21, 26, 31, 41, 51, 61, 71], 1: [51, 61, 71, 81, 101], 2: [5, 6, 7, 8, 9, 11],
             3: [2, 5, 10, 15], 4: [1, 2, 3, 4]}


def glm_design(X: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(X))]
    for j, edges in GLM_BANDS.items():
        b = np.digitize(X[:, j], edges)
        cols += [(b == k).astype(float) for k in range(1, len(edges) + 1)]
    cols += [(X[:, 5] == k).astype(float) for k in range(1, 8)]
    return np.column_stack(cols)


def fit_glm(X, e, N) -> np.ndarray:
    """Poisson GLM with log link and offset log(exposure), main effects of banded factors, by IRLS."""
    D = glm_design(X)
    b = np.zeros(D.shape[1])
    b[0] = np.log(N.sum() / e.sum())
    for _ in range(100):
        m = e * np.exp(D @ b)
        z = D @ b + (N - m) / m
        b_new = np.linalg.solve(D.T @ (m[:, None] * D), D.T @ (m * z))
        done = np.max(np.abs(b_new - b)) < 1e-12
        b = b_new
        if done:
            break
    return b


def boosting_kwargs() -> dict:
    return dict(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=200, early_stopping=True,
                validation_fraction=0.2, n_iter_no_change=20, random_state=7, categorical_features=[5])


def fit_models(train: dict) -> dict:
    X, e, N = train["X"], train["e"], train["N"]
    models = {"glm": fit_glm(X, e, N)}
    models["gbm"] = HistGradientBoostingRegressor(loss="poisson", **boosting_kwargs()).fit(X, N / e, sample_weight=e)
    models["clf"] = HistGradientBoostingClassifier(**boosting_kwargs()).fit(X, (N > 0).astype(int))
    models["nooffset"] = HistGradientBoostingRegressor(loss="poisson", **boosting_kwargs()).fit(X, N)
    return models


def predict(models: dict, X: np.ndarray) -> dict:
    """Raw rates. The classifier's probability of at least one claim is used as a score; it is rebalanced later."""
    return {
        "glm": np.exp(glm_design(X) @ models["glm"]),
        "gbm": models["gbm"].predict(X),
        "clf": models["clf"].predict_proba(X)[:, 1],
        "nooffset": models["nooffset"].predict(X),
    }


MIN_STEP_EXPOSURE = 1000.0  # policy-years per step of the recalibrated tariff (about 100 expected claims)

LABELS = {
    "truth": "true frequency",
    "glm": "Poisson GLM, banded main effects",
    "gbm": "Poisson boosting",
    "clf": "claim classifier, rebalanced",
    "nooffset": "boosting without exposure offset",
}


def main() -> None:
    print("Part 2: rank invariance, auto-calibration and the Murphy decomposition, in a simulation with known truth.")
    print(f"Seed {SEED}; train {N_TRAIN:,}, calibration {N_CAL:,}, test {N_TEST:,} policies; population pool {N_POOL:,}.")

    train, cal, test = sample(N_TRAIN, RNG_TRAIN), sample(N_CAL, RNG_CAL), sample(N_TEST, RNG_TEST)
    Xp, ep = simulate_policies(N_POOL, RNG_POOL)
    pool = {"X": Xp, "e": ep, "mu": true_frequency(Xp)}

    # -----------------------------------------------------------------------------------------------
    header("1. The portfolio and its true frequency")
    e, mu, N, X = test["e"], test["mu"], test["N"], test["X"]
    young = X[:, 0] < 26
    print(f"Test sample: {N_TEST:,} policies, {e.sum():,.0f} policy-years, {N.sum():,.0f} claims, "
          f"frequency {N.sum() / e.sum():.4f} per policy-year.")
    print(f"Policies with no claim: {np.mean(N == 0):.4f}; with two or more: {np.mean(N >= 2):.4f}.")
    print(f"Drivers under 26: {young.mean():.3f} of policies; full-year policies among them {np.mean(e[young] == 1):.3f},"
          f" among the others {np.mean(e[~young] == 1):.3f}; mean exposure {e[young].mean():.3f} vs {e[~young].mean():.3f}.")
    pm = pool["mu"]
    q = np.quantile(pm, [0.01, 0.5, 0.99])
    print(f"True frequency in the pool: 1st percentile {q[0]:.4f}, median {q[1]:.4f}, 99th percentile {q[2]:.4f};"
          f" mean (exposure-weighted) {(ep * pm).sum() / ep.sum():.4f}.")
    print(f"Unobserved heterogeneity: Gamma with mean 1 and variance {1 / HET_SHAPE:.2f}.")
    pop_truth = population(pm, pool)
    print(f"Population Gini of the true frequency (the ceiling for any rate on these covariates): {pop_truth['gini']:.4f}")
    print(f"Population discrimination (DSC) of the true frequency: {pop_truth['DSC']:.3f} per 1,000 policy-years")
    mt = murphy(mu, e, N)
    print(f"Test sample, true frequency: Gini {gini(mu, e, N):.4f}; deviance per 1,000 policy-years "
          f"{mt['D'] / (e.sum() / 1000):.2f}; UNC {mt['UNC'] / (e.sum() / 1000):.2f}; DSC {mt['DSC'] / (e.sum() / 1000):.2f};"
          f" MCB {mt['MCB'] / (e.sum() / 1000):.3f}; balance {(e * mu).sum() / N.sum():.4f}")
    print(f"  (the truth's MCB on a sample is not zero: the isotonic fit also follows the sample's noise)")

    # -----------------------------------------------------------------------------------------------
    header("2. One ranking, many prices: powers and levels of the true frequency (population)")
    total = (ep * pm).sum()
    print(f"{'rate':<22}{'Gini':>8}{'prem Gini':>11}{'balance':>9}{'excess':>9}{'MCB':>8}{'DSC':>8}{'lost':>7}")
    family = {}
    for g in (0.5, 0.8, 1.0, 1.25, 2.0):
        family[g] = rebalance(pm ** g, ep, total)
    family["level 0.9"] = 0.9 * pm
    family["level 1.1"] = 1.1 * pm
    family_pop = {}
    for k, r in family.items():
        p = family_pop[k] = population(r, pool)
        name = f"truth^{k}" if not isinstance(k, str) else f"truth, {k}"
        print(f"{name:<22}{p['gini']:8.4f}{p['premium_gini']:11.4f}{p['balance']:9.4f}{p['excess']:9.3f}"
              f"{p['MCB']:8.3f}{p['DSC']:8.3f}{p['lost']:7.3f}")
    print("(per 1,000 policy-years; excess = MCB + lost; DSC of the truth is the most any rate can reach)")

    # -----------------------------------------------------------------------------------------------
    header("3. The geometry of isotonic recalibration (test sample, in-sample fit)")
    models = fit_models(train)
    raw_test = predict(models, X)
    raw_pool = predict(models, Xp)
    raw_cal = predict(models, cal["X"])
    raw_train = predict(models, train["X"])
    k_clf = train["N"].sum() / (train["e"] * raw_train["clf"]).sum()  # the classifier's level, set on training data
    for d in (raw_test, raw_pool, raw_cal):
        d["clf"] = d["clf"] * k_clf
    test_rates = {"truth": mu, "glm": raw_test["glm"], "gbm": raw_test["gbm"],
                  "clf": raw_test["clf"], "nooffset": raw_test["nooffset"],
                  "truth^2": rebalance(mu ** 2, e, N.sum())}
    for k in ("clf", "truth^2", "glm"):
        r = test_rates[k]
        m = murphy(r, e, N)
        cw, cc = lorenz(r, e, N)
        hx, hy = lower_convex_hull(cw, cc)
        cw_rc, cc_rc = lorenz(m["rc"], e, N)
        gap = np.max(np.abs(np.interp(cw_rc, hx, hy) - cc_rc))
        gap2 = np.max(np.abs(np.interp(hx, cw_rc, cc_rc) - hy))
        # discrimination as a Kullback-Leibler divergence over the isotonic blocks
        blocks, inv = np.unique(m["rc"], return_inverse=True)
        cb = np.bincount(inv, weights=N)
        eb = np.bincount(inv, weights=e)
        nz = cb > 0
        kl = np.sum((cb[nz] / N.sum()) * np.log((cb[nz] / N.sum()) / (eb[nz] / e.sum())))
        print(f"{k:<8} Gini {gini(r, e, N):.4f} -> recalibrated {gini(m['rc'], e, N):.4f}; isotonic blocks {m['blocks']};"
              f" Lorenz of recalibration vs convex minorant: max gap {max(gap, gap2):.1e};"
              f" DSC {m['DSC']:.2f} vs 2 x claims x KL {2 * N.sum() * kl:.2f}")

    # -----------------------------------------------------------------------------------------------
    header("4. Four fitted models: test sample and population")
    tr = raw_train
    print("Total balance on the training sample (premium / claims): "
          f"GLM {(train['e'] * tr['glm']).sum() / train['N'].sum():.6f}, "
          f"boosting {(train['e'] * tr['gbm']).sum() / train['N'].sum():.6f}, "
          f"no offset {(train['e'] * tr['nooffset']).sum() / train['N'].sum():.4f}")
    print(f"Boosting iterations at early stopping: Poisson {models['gbm'].n_iter_}, classifier {models['clf'].n_iter_},"
          f" no offset {models['nooffset'].n_iter_}")
    print(f"Claim classifier: level factor set on training data {k_clf:.4f}; mean predicted probability "
          f"{raw_test['clf'].mean() / k_clf:.4f};"
          f" share of test policies with a claim {np.mean(N > 0):.4f}")
    per = e.sum() / 1000.0
    print()
    print("Test sample (per 1,000 policy-years; D = UNC - DSC + MCB):")
    print(f"{'model':<34}{'Gini':>8}{'balance':>9}{'D':>9}{'MCB':>8}{'DSC':>8}{'blocks':>8}{'prem Gini':>11}")
    for k in ("truth", "glm", "gbm", "clf", "nooffset"):
        r = test_rates[k]
        m = murphy(r, e, N)
        print(f"{LABELS[k]:<34}{gini(r, e, N):8.4f}{(e * r).sum() / N.sum():9.4f}{m['D'] / per:9.2f}"
              f"{m['MCB'] / per:8.3f}{m['DSC'] / per:8.2f}{m['blocks']:8d}{premium_gini(r, e):11.4f}")
    print(f"UNC (flat rate) on the test sample: {murphy(mu, e, N)['UNC'] / per:.2f}")
    print("Gap between the concentration Gini and the premium's own Gini on the test sample: " + "; ".join(
        f"{k} {gini(test_rates[k], e, N) - premium_gini(test_rates[k], e):.4f}" for k in ("truth", "glm", "gbm", "nooffset", "clf")))
    print()
    pool_rates = {"truth": pm, "glm": raw_pool["glm"], "gbm": raw_pool["gbm"],
                  "clf": raw_pool["clf"], "nooffset": raw_pool["nooffset"]}
    print("Population (per 1,000 policy-years; excess = expected deviance above the truth = MCB + lost):")
    print(f"{'model':<34}{'Gini':>8}{'prem Gini':>11}{'balance':>9}{'excess':>9}{'MCB':>8}{'DSC':>8}{'lost':>7}")
    pops = {}
    for k, r in pool_rates.items():
        p = pops[k] = population(r, pool)
        print(f"{LABELS[k]:<34}{p['gini']:8.4f}{p['premium_gini']:11.4f}{p['balance']:9.4f}{p['excess']:9.3f}"
              f"{p['MCB']:8.3f}{p['DSC']:8.3f}{p['lost']:7.3f}")
    fitted = ("glm", "gbm", "clf", "nooffset")
    print("Order of the fitted models by population Gini:   ", " > ".join(sorted(fitted, key=lambda k: -pops[k]["gini"])))
    print("Order of the fitted models by population deviance:", " > ".join(sorted(fitted, key=lambda k: pops[k]["excess"])))
    print("Order of the fitted models by population DSC:     ", " > ".join(sorted(fitted, key=lambda k: -pops[k]["DSC"])))
    print()
    print("Where the discrimination and the Gini come from: the most expensive 10% of exposure against the rest")
    wc = {}
    for k in ("truth", "glm", "gbm"):
        w = wc[k] = where_it_counts(pool_rates[k], pool)
        print(f"  {LABELS[k]:<34} DSC top {w['dsc_top']:.3f} + rest {w['dsc_rest']:.3f};"
              f" Gini top {w['gini_top']:.4f} + rest {w['gini_rest']:.4f}")
    print(f"  boosting minus GLM: DSC top {wc['gbm']['dsc_top'] - wc['glm']['dsc_top']:+.3f},"
          f" rest {wc['gbm']['dsc_rest'] - wc['glm']['dsc_rest']:+.3f};"
          f" Gini top {wc['gbm']['gini_top'] - wc['glm']['gini_top']:+.4f}, rest {wc['gbm']['gini_rest'] - wc['glm']['gini_rest']:+.4f}")
    yo = Xp[:, 0] < 30
    hp = Xp[:, 2] >= 8
    for k in ("glm", "gbm"):
        r = pool_rates[k]
        m = yo & hp
        print(f"  {LABELS[k]}: premium / expected claims for drivers under 30 with power class 8 or more"
              f" ({ep[m].sum() / ep.sum():.3f} of exposure): {(ep[m] * r[m]).sum() / (ep[m] * pm[m]).sum():.3f}")
    print(f"Mean exposure of drivers under 26 relative to the others: {ep[Xp[:, 0] < 26].mean() / ep[Xp[:, 0] >= 26].mean():.3f}")
    print()
    p125 = population(rebalance(pm ** 1.25, ep, total), pool)
    print(f"For comparison, truth^1.25: Gini {p125['gini']:.4f}, excess {p125['excess']:.3f}, MCB {p125['MCB']:.3f}")
    young_p = Xp[:, 0] < 26
    for k in ("nooffset", "clf", "gbm", "glm"):
        r = pool_rates[k]
        print(f"  {LABELS[k]}: premium / expected claims for drivers under 26 "
              f"{(ep[young_p] * r[young_p]).sum() / (ep[young_p] * pm[young_p]).sum():.3f}, others "
              f"{(ep[~young_p] * r[~young_p]).sum() / (ep[~young_p] * pm[~young_p]).sum():.3f}")

    # -----------------------------------------------------------------------------------------------
    header("5. Isotonic recalibration on a separate calibration sample (fitted on calibration, judged in population)")
    ec, Nc = cal["e"], cal["N"]
    print(f"Calibration sample: {N_CAL:,} policies, {ec.sum():,.0f} policy-years, {Nc.sum():,.0f} claims.")
    cal_rates = {"glm": raw_cal["glm"], "gbm": raw_cal["gbm"], "clf": raw_cal["clf"], "nooffset": raw_cal["nooffset"],
                 "truth^1.25": cal["mu"] ** 1.25}
    pool_raw = {"glm": raw_pool["glm"], "gbm": raw_pool["gbm"], "clf": raw_pool["clf"], "nooffset": raw_pool["nooffset"],
                "truth^1.25": pm ** 1.25}
    print()
    print("Plain isotonic (pool-adjacent-violators) step tariff:")
    for k in cal_rates:
        st = StepRecalibration(0.0).fit(cal_rates[k], ec, Nc)
        r = st.predict(pool_raw[k])
        zero = ep[r == 0].sum() / ep.sum()
        print(f"  {LABELS.get(k, k):<34} steps {st.steps:3d}; smallest step {st.exposure_.min():8.1f} policy-years;"
              f" lowest rate {st.rates_[0]:.4f}; population exposure charged nothing {zero:.4f}")
    print()
    print(f"Steps merged to at least {MIN_STEP_EXPOSURE:,.0f} policy-years each (per 1,000 policy-years in population):")
    print(f"{'model':<34}{'steps':>6}{'Gini':>8}{'balance':>9}{'excess':>9}{'MCB':>8}{'DSC':>8}   before: Gini, excess, MCB")
    recal_pool, recal_pop = {}, {}
    for k in cal_rates:
        st = StepRecalibration(MIN_STEP_EXPOSURE).fit(cal_rates[k], ec, Nc)
        r = recal_pool[k] = st.predict(pool_raw[k])
        p = recal_pop[k] = population(r, pool)
        before = pops[k] if k in pops else p125
        print(f"{LABELS.get(k, k):<34}{st.steps:6d}{p['gini']:8.4f}{p['balance']:9.4f}{p['excess']:9.3f}{p['MCB']:8.3f}"
              f"{p['DSC']:8.3f}   {before['gini']:.4f}, {before['excess']:.3f}, {before['MCB']:.3f}")
    print()
    print("Minimum exposure per step against population excess deviance (steps in brackets):")
    for k in ("gbm", "clf", "truth^1.25"):
        cells = []
        for mx in (250, 500, 1000, 2000, 4000, 8000):
            st = StepRecalibration(mx).fit(cal_rates[k], ec, Nc)
            cells.append(f"{mx:,}: {population(st.predict(pool_raw[k]), pool)['excess']:.3f} ({st.steps})")
        print(f"  {LABELS.get(k, k):<34}" + "; ".join(cells))
    print()
    print("Order of the recalibrated fitted models by population Gini:   ",
          " > ".join(sorted(fitted, key=lambda k: -recal_pop[k]["gini"])))
    print("Order of the recalibrated fitted models by population deviance:",
          " > ".join(sorted(fitted, key=lambda k: recal_pop[k]["excess"])))

    # -----------------------------------------------------------------------------------------------
    header(f"6. Choosing between two models on {N_REP} test samples of {N_REP_SIZE:,} policies")
    pairs = [("gbm", "truth^1.25"), ("gbm", "glm"), ("glm", "clf"), ("gbm", "nooffset"), ("gbm", "truth^1.25 recal")]
    rates_for_rep = {"glm": pool_rates["glm"], "gbm": pool_rates["gbm"], "clf": pool_rates["clf"],
                     "nooffset": pool_rates["nooffset"], "truth^1.25": rebalance(pm ** 1.25, ep, total),
                     "truth^1.25 recal": recal_pool["truth^1.25"]}
    needed = sorted({m for p in pairs for m in p})
    g_rep = {k: np.empty(N_REP) for k in needed}
    d_rep = {k: np.empty(N_REP) for k in needed}
    for i in range(N_REP):
        idx = RNG_REP.integers(0, N_POOL, N_REP_SIZE)
        e_i, mu_i = ep[idx], pm[idx]
        N_i = draw_claims(e_i, mu_i, RNG_REP)
        for k in needed:
            r = rates_for_rep[k][idx]
            g_rep[k][i] = gini(r, e_i, N_i)
            d_rep[k][i] = poisson_deviance(N_i, e_i * r) / (e_i.sum() / 1000.0)
    print("For each pair (A, B): population Gini and excess deviance of each, and the share of test samples in which")
    print("each criterion prefers A; mean and standard deviation of the paired differences A - B.")
    for a, b in pairs:
        pa = pops[a]
        pb = pops.get(b) or (recal_pop["truth^1.25"] if b == "truth^1.25 recal" else p125)
        dg = g_rep[a] - g_rep[b]
        dd = d_rep[a] - d_rep[b]
        print(f"A = {a:<10} B = {b:<17} pop Gini {pa['gini']:.4f} vs {pb['gini']:.4f}; pop excess {pa['excess']:.3f} vs"
              f" {pb['excess']:.3f}")
        print(f"    Gini prefers A in {np.mean(dg > 0):.3f}; deviance prefers A in {np.mean(dd < 0):.3f};"
              f" Gini difference {dg.mean():+.4f} (sd {dg.std(ddof=1):.4f});"
              f" deviance difference {dd.mean():+.3f} (sd {dd.std(ddof=1):.3f})")
    for k in needed:
        print(f"  {k:<17} Gini across samples: mean {g_rep[k].mean():.4f}, sd {g_rep[k].std(ddof=1):.4f}")
    sd_single = [g_rep[k].std(ddof=1) for k in needed]
    sd_pair = [(g_rep[a] - g_rep[b]).std(ddof=1) for a, b in pairs]
    print(f"Ratio of a single model's Gini sd to a paired difference's sd: {min(sd_single) / max(sd_pair):.1f}"
          f" to {max(sd_single) / min(sd_pair):.1f}")
    small = slice(0, 3000)
    corp = {}
    for k in ("clf", "gbm", "nooffset"):
        rc = murphy(test_rates[k], e, N)["rc"]
        o = np.argsort(test_rates[k], kind="stable")
        keep = np.r_[True, np.diff(rc[o]) != 0]
        corp[k] = {"rate": [round(float(x), 6) for x in test_rates[k][o][keep]],
                   "recal": [round(float(x), 6) for x in rc[o][keep]]}
    out = {"family": {str(k): {m: round(float(x), 6) for m, x in v.items()} for k, v in family_pop.items()},
           "recalibrated": {k: {m: round(float(x), 6) for m, x in v.items()} for k, v in recal_pop.items()},
           "small_sample": {"clf": [round(float(x), 6) for x in test_rates["clf"][small]],
                            "e": [round(float(x), 6) for x in e[small]], "N": [int(x) for x in N[small]]},
           "corp": corp,
           "gini": {k: [round(float(x), 6) for x in v] for k, v in g_rep.items()},
           "deviance": {k: [round(float(x), 4) for x in v] for k, v in d_rep.items()},
           "population": {k: {m: round(float(x), 6) for m, x in v.items()} for k, v in
                          {**pops, "truth^1.25": p125, "truth^1.25 recal": recal_pop["truth^1.25"]}.items()}}
    (Path(__file__).parent / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
