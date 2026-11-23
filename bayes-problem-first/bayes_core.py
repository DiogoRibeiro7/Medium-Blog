"""Models, samplers and checks for the problem-first Bayesian course.

Two real problems and one check:
- the BNT162b2 vaccine trial (Polack et al. 2020): 8 of 170 cases in the vaccine group, with the
  protocol's Beta(0.700102, 1) prior on the share of cases that fall in the vaccine group;
- the Space Shuttle O-ring flights before Challenger (Dalal, Fowlkes and Hoadley 1989): a binomial
  logistic regression of thermally distressed O-rings on launch temperature;
- simulation-based calibration (Talts et al. 2018), which checks a sampler without any algebra.

Everything is NumPy and SciPy. The sampler is a random-walk Metropolis written out in full so that
every step can be read; in a course, PyMC, Stan or brms would play its role.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special, stats

# ----------------------------------------------------------------------------------------------- vaccine trial

PROTOCOL_PRIOR = (0.700102, 1.0)  # Beta prior on theta, from the trial protocol
VACCINE_CASES = 8
TOTAL_CASES = 170  # 8 vaccine + 162 placebo, primary efficacy population
# Surveillance time, 1000 person-years (Polack et al. 2020, Table 3): vaccine 2.214, placebo 2.222.
SURV_RATIO = 2.214 / 2.222
VE_BAR = 0.30  # the efficacy the trial had to beat
LOOKS = (32, 62, 92, 120, 164)  # planned analyses, by total cases
THRESHOLD_INTERIM = 0.995  # posterior probability of VE > 30% needed at an interim look
THRESHOLD_FINAL = 0.986  # and at the final look


def ve_from_theta(theta, r=1.0):
    """Vaccine efficacy from theta, the share of cases in the vaccine arm.

    ``r`` is the ratio of surveillance time, vaccine to placebo: theta = r(1 - VE) / (1 + r(1 - VE)).
    The design used r = 1; the published estimate adjusts for the observed ratio.
    """
    theta = np.asarray(theta, dtype=float)
    return 1.0 - theta / (r * (1.0 - theta))


def theta_from_ve(ve, r=1.0):
    ve = np.asarray(ve, dtype=float)
    return r * (1.0 - ve) / (1.0 + r * (1.0 - ve))


THETA_BAR = float(theta_from_ve(VE_BAR))  # VE > 30% exactly when theta < THETA_BAR


def conjugate(x, n, a, b):
    """Beta(a, b) prior and x of n cases in the vaccine arm: the posterior is Beta(a + x, b + n - x)."""
    return a + x, b + n - x


def prob_ve_above(x, n, a, b, ve=VE_BAR, r=1.0):
    """Posterior probability that VE exceeds ``ve`` (vectorised over x and n)."""
    pa, pb = conjugate(np.asarray(x, float), np.asarray(n, float), a, b)
    return stats.beta.cdf(float(theta_from_ve(ve, r)), pa, pb)


def ve_quantiles(x, n, a, b, q, r=1.0):
    """Posterior quantiles of VE; VE falls as theta rises, so quantile q of VE is quantile 1 - q of theta."""
    pa, pb = conjugate(np.asarray(x, float), np.asarray(n, float), a, b)
    q = np.asarray(q, float)
    return ve_from_theta(stats.beta.ppf(1.0 - q, pa, pb), r)


def success_boundary(n, a=PROTOCOL_PRIOR[0], b=PROTOCOL_PRIOR[1], threshold=THRESHOLD_INTERIM):
    """Largest number of vaccine cases among n that still gives P(VE > 30%) above the threshold (-1 if none)."""
    xs = np.arange(n + 1)
    ok = prob_ve_above(xs, n, a, b) > threshold
    return int(xs[ok].max()) if ok.any() else -1


def grid_posterior(x, n, a, b, m=20_000):
    """Grid approximation on theta: m midpoints of (0, 1), prior times likelihood, normalised to sum to 1."""
    theta = (np.arange(m) + 0.5) / m
    logp = (a - 1 + x) * np.log(theta) + (b - 1 + n - x) * np.log1p(-theta)
    w = np.exp(logp - logp.max())
    return theta, w / w.sum()


def grid_update(theta, w, case_in_vaccine_arm):
    """One case arrives: multiply by its likelihood (theta or 1 - theta) and normalise again."""
    w = w * (theta if case_in_vaccine_arm else 1.0 - theta)
    return w / w.sum()


def weighted_quantile(values, weights, q):
    """Quantiles of a discrete distribution on sorted ``values`` (interpolated on the cumulative weights)."""
    order = np.argsort(values)
    v, c = np.asarray(values)[order], np.cumsum(np.asarray(weights)[order])
    c = c - 0.5 * np.asarray(weights)[order]
    return np.interp(q, c / c[-1], v)


def log_post_logit(z, x, n, a, b):
    """Log posterior of z = logit(theta), Jacobian included: (a + x) log theta + (b + n - x) log(1 - theta)."""
    return -(a + x) * np.logaddexp(0.0, -z) - (b + n - x) * np.logaddexp(0.0, z)


def metropolis_theta(x, n, a, b, *, rng, iters, step, start, burn=0, thin=1, data_weight=1.0):
    """Random-walk Metropolis on logit(theta), run for many problems at once.

    ``x`` and ``n`` may be arrays: each entry is a separate chain. ``data_weight`` multiplies the likelihood
    (2.0 is the double-counting bug used in the calibration check). Returns draws of theta with shape
    (kept iterations, number of chains) and the acceptance rate.
    """
    x = np.atleast_1d(np.asarray(x, float))
    n = np.broadcast_to(np.asarray(n, float), x.shape)
    z = np.broadcast_to(np.asarray(start, float), x.shape).copy()

    def lp(zz):
        return log_post_logit(zz, data_weight * x, data_weight * n, a, b)

    cur = lp(z)
    kept, accepted = [], 0
    for i in range(iters):
        prop = z + step * rng.standard_normal(z.shape)
        new = lp(prop)
        acc = np.log(rng.random(z.shape)) < new - cur
        z = np.where(acc, prop, z)
        cur = np.where(acc, new, cur)
        accepted += int(acc.sum())
        if i >= burn and (i - burn) % thin == 0:
            kept.append(special.expit(z))
    return np.array(kept), accepted / (iters * x.size)


def split_rhat(chains):
    """Split R-hat (Gelman et al. 2013) for draws of shape (iterations, chains)."""
    n = chains.shape[0] // 2
    halves = np.concatenate([chains[:n], chains[n : 2 * n]], axis=1)
    means, vars_ = halves.mean(axis=0), halves.var(axis=0, ddof=1)
    w, bvar = vars_.mean(), n * means.var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + bvar / n) / w))


def random_orders(rng, k, x=VACCINE_CASES, n=TOTAL_CASES):
    """k random orders of arrival of the n cases, x of them in the vaccine arm: a (k, n) array of 0/1."""
    base = np.zeros(n, dtype=np.int8)
    base[:x] = 1
    return np.array([rng.permutation(base) for _ in range(k)])


def false_success_rate(rng, looks, thresholds, *, ve=VE_BAR, trials=200_000, a=PROTOCOL_PRIOR[0], b=PROTOCOL_PRIOR[1]):
    """Share of simulated trials with true VE = ``ve`` that the decision rule would call a success.

    Cases arrive one at a time, each in the vaccine arm with probability theta(ve). The rule declares
    success at the first look whose vaccine-case count is at or below that look's boundary.
    """
    theta = float(theta_from_ve(ve))
    nmax = max(looks)
    cum = np.cumsum(rng.random((trials, nmax)) < theta, axis=1, dtype=np.int16)
    hit = np.zeros(trials, dtype=bool)
    for look, thr in zip(looks, thresholds, strict=True):
        hit |= cum[:, look - 1] <= success_boundary(look, a, b, thr)
    return float(hit.mean())


# ----------------------------------------------------------------------------------------------- O-rings

RINGS = 6  # field-joint O-rings per flight (three per booster)
CENTRE, SCALE = 70.0, 10.0  # temperature is centred at 70 F and measured in tens of degrees
LAUNCH_F = 31.0  # temperature used for the Challenger launch by Dalal et al. (1989)


@dataclass(frozen=True)
class Flights:
    temp: np.ndarray  # launch temperature, F
    distressed: np.ndarray  # O-rings with thermal distress, of 6

    @property
    def s(self):
        return (self.temp - CENTRE) / SCALE


def load_flights(path: Path) -> Flights:
    """The 23 flights with a known O-ring count from vcd::SpaceShuttle (flight 4 has no count and is dropped)."""
    temp, dist = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["nFailures"] in ("", "NA"):
                continue
            temp.append(float(row["Temperature"]))
            dist.append(int(row["nFailures"]))
    return Flights(np.array(temp), np.array(dist))


@dataclass(frozen=True)
class NormalPrior:
    """Independent normal priors on the intercept (log-odds at 70 F) and the slope per 10 F."""

    name: str
    a_mean: float
    a_sd: float
    b_mean: float
    b_sd: float

    def draw(self, rng, k):
        return rng.normal(self.a_mean, self.a_sd, k), rng.normal(self.b_mean, self.b_sd, k)

    def logpdf(self, a, b):
        return stats.norm.logpdf(a, self.a_mean, self.a_sd) + stats.norm.logpdf(b, self.b_mean, self.b_sd)


VAGUE = NormalPrior("vague", 0.0, 10.0, 0.0, 10.0)
# At 70 F, a few per cent of rings in distress, but anything from about 0.2% to 50% is allowed;
# temperature may push either way, by up to a factor of about e^2 in the odds per 10 F.
WEAKLY_INFORMATIVE = NormalPrior("weakly informative", -3.0, 1.5, 0.0, 1.0)


def loglik(a, b, flights: Flights):
    """Binomial log likelihood of (a, b) arrays (any shape), summed over flights."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    eta = a[..., None] + b[..., None] * flights.s
    y = flights.distressed
    return np.sum(y * -np.logaddexp(0.0, -eta) + (RINGS - y) * -np.logaddexp(0.0, eta), axis=-1)


@dataclass(frozen=True)
class Grid2D:
    a: np.ndarray
    b: np.ndarray
    w: np.ndarray  # posterior weights, shape (len(a), len(b)), summing to 1


def grid_posterior_2d(flights: Flights, prior: NormalPrior, a_range=(-9.0, 2.0), b_range=(-5.0, 2.5), m=500):
    a = np.linspace(*a_range, m)
    b = np.linspace(*b_range, m)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    lp = loglik(aa, bb, flights) + prior.logpdf(aa, bb)
    w = np.exp(lp - lp.max())
    return Grid2D(a, b, w / w.sum())


def grid_edge_mass(g: Grid2D):
    """Posterior mass in the outermost row and column of the grid: near zero if the grid is wide enough."""
    w = g.w
    return float(w[0].sum() + w[-1].sum() + w[:, 0].sum() + w[:, -1].sum())


def grid_draws(g: Grid2D, rng, k):
    """Draw k (a, b) pairs from the grid posterior, jittered within each cell."""
    idx = rng.choice(g.w.size, size=k, p=g.w.ravel())
    i, j = np.unravel_index(idx, g.w.shape)
    da, db = g.a[1] - g.a[0], g.b[1] - g.b[0]
    return g.a[i] + da * (rng.random(k) - 0.5), g.b[j] + db * (rng.random(k) - 0.5)


def metropolis_2d(flights: Flights, prior: NormalPrior, *, rng, chains, iters, burn, step, start=(-3.0, 0.0)):
    """Random-walk Metropolis on (a, b), ``chains`` chains at once from overdispersed starts."""
    a = start[0] + rng.normal(0, 1.0, chains)
    b = start[1] + rng.normal(0, 1.0, chains)
    cur = loglik(a, b, flights) + prior.logpdf(a, b)
    out_a, out_b, acc_total = [], [], 0
    for i in range(iters):
        pa, pb = a + step[0] * rng.standard_normal(chains), b + step[1] * rng.standard_normal(chains)
        new = loglik(pa, pb, flights) + prior.logpdf(pa, pb)
        acc = np.log(rng.random(chains)) < new - cur
        a, b, cur = np.where(acc, pa, a), np.where(acc, pb, b), np.where(acc, new, cur)
        acc_total += int(acc.sum())
        if i >= burn:
            out_a.append(a.copy())
            out_b.append(b.copy())
    return np.array(out_a), np.array(out_b), acc_total / (iters * chains)


def p_at(a, b, temp_f):
    """Chance that one O-ring shows thermal distress at temperature ``temp_f`` (F)."""
    return special.expit(np.asarray(a) + np.asarray(b) * (temp_f - CENTRE) / SCALE)


def mle(flights: Flights, iters=50):
    """Maximum likelihood (a, b) by Newton's method, for reference."""
    X = np.column_stack([np.ones_like(flights.s), flights.s])
    beta = np.zeros(2)
    for _ in range(iters):
        p = special.expit(X @ beta)
        grad = X.T @ (flights.distressed - RINGS * p)
        hess = -(X * (RINGS * p * (1 - p))[:, None]).T @ X
        beta = beta - np.linalg.solve(hess, grad)
    return beta


def prior_extreme_share(prior: NormalPrior, rng, k=400_000, temp_f=CENTRE, lo=0.01, hi=0.99):
    """Share of prior draws whose implied chance at ``temp_f`` is below ``lo`` or above ``hi``."""
    a, b = prior.draw(rng, k)
    p = p_at(a, b, temp_f)
    return float(np.mean((p < lo) | (p > hi)))


def posterior_quantile_of_truth(g: Grid2D, a_true, b_true):
    """Posterior CDF at the true values (Cook, Gelman and Rubin 2006): uniform if the inference is right."""
    pa, pb = g.w.sum(axis=1), g.w.sum(axis=0)
    return float(np.interp(a_true, g.a, np.cumsum(pa))), float(np.interp(b_true, g.b, np.cumsum(pb)))


# ----------------------------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class SamplerSetup:
    name: str
    iters: int
    step: float
    start: str  # "data" (logit of the observed share) or "half" (theta = 0.5)
    burn: int
    thin: int
    data_weight: float


SAMPLERS = (
    SamplerSetup("correct", iters=2_500, step=0.35, start="data", burn=500, thin=20, data_weight=1.0),
    SamplerSetup("stopped too early", iters=200, step=0.1, start="half", burn=0, thin=1, data_weight=1.0),
    SamplerSetup("data counted twice", iters=2_500, step=0.25, start="data", burn=500, thin=20, data_weight=2.0),
)


def sbc(setup: SamplerSetup, rng, sims=2_000, n=TOTAL_CASES, a=PROTOCOL_PRIOR[0], b=PROTOCOL_PRIOR[1], draws=99):
    """Simulation-based calibration of a theta sampler.

    For each simulation: draw theta from the prior, draw x ~ Binomial(n, theta), run the sampler, and record
    the rank of the true theta among the last ``draws`` posterior draws (0 to ``draws``), and whether the
    central 90% interval of those draws contains it. Correct inference gives uniform ranks and 90% coverage.
    """
    theta = rng.beta(a, b, sims)
    x = rng.binomial(n, theta)
    start = special.logit((x + 0.5) / (n + 1.0)) if setup.start == "data" else np.zeros(sims)
    out, acc = metropolis_theta(
        x,
        n,
        a,
        b,
        rng=rng,
        iters=setup.iters,
        step=setup.step,
        start=start,
        burn=setup.burn,
        thin=setup.thin,
        data_weight=setup.data_weight,
    )
    post = out[-draws:]
    ranks = (post < theta).sum(axis=0)
    lo, hi = np.quantile(post, [0.05, 0.95], axis=0)
    covered = (theta >= lo) & (theta <= hi)
    return ranks, covered, acc


def rank_uniformity(ranks, draws=99, bins=20):
    """Chi-square test of uniform ranks over ``bins`` equal bins of the draws + 1 possible ranks."""
    counts = np.bincount(ranks * bins // (draws + 1), minlength=bins)
    return counts, float(stats.chisquare(counts).pvalue)
