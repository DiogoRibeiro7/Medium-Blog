"""Prevalence from an imperfect test: the Santa Clara serosurvey, worked through to two decisions.

Model (Gelman and Carpenter 2020): a person tests positive with probability
    p = pi * sens + (1 - pi) * (1 - spec),
where pi is the prevalence, spec the specificity (true-negative rate) and sens the sensitivity.
Validation studies give known negatives (for spec) and known positives (for sens).

Posteriors for pi are computed on a grid by integrating spec and sens out with draws:
    p(pi | data) is proportional to prior(pi) * E[ Binomial(y | n, p(pi, spec, sens)) ],
where the expectation is over the distribution of (spec, sens) given the validation data alone. Because the
field data depend on the hyperparameters only through the target study's spec and sens, this is exact for the
pooled and for the hierarchical model; only Monte Carlo error remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import special, stats

# ----------------------------------------------------------------------------------------------- data
# Bendavid et al. (2020), version 1 (11 April 2020) and version 2 (27 April 2020), as transcribed by
# Gelman and Carpenter (2020), github.com/bob-carpenter/diagnostic-testing, src/specificity-santa-clara.

FIELD_POS, FIELD_N = 50, 3330

V1_SPEC = (399, 401)  # manufacturer 369/371 + Stanford 30/30 known negatives testing negative
V1_SENS = (103, 122)  # manufacturer 78/85 + Stanford 25/37 known positives testing positive

# Version 2: 13 specificity studies (negatives correctly negative, tested) and 3 sensitivity studies.
SPEC_STUDIES = (
    (368, 371),
    (30, 30),
    (70, 70),
    (1102, 1102),
    (300, 300),
    (311, 311),
    (500, 500),
    (198, 200),
    (99, 99),
    (29, 31),
    (146, 150),
    (105, 108),
    (50, 52),
)
SENS_STUDIES = ((78, 85), (27, 37), (25, 35))

# Published figures used for comparison
PUBLISHED_RAW = (0.0111, 0.0197)  # v1, exact binomial 95% CI of 50/3330
PUBLISHED_WEIGHTED = 0.0281  # v1 population-weighted prevalence
PUBLISHED_RANGE = (0.0249, 0.0416)  # v1, weighted and test-adjusted, across three test scenarios
PUBLISHED_IJE_UNWEIGHTED = (0.012, 0.007, 0.018)  # 2021 journal version: estimate, 95% CI, unweighted and adjusted


def pooled(studies):
    y = sum(s[0] for s in studies)
    n = sum(s[1] for s in studies)
    return y, n


# ----------------------------------------------------------------------------------------------- grid


def prevalence_grid(fine_to=0.05, n_fine=2001, n_coarse=381):
    """Nodes on [0, 1], fine below ``fine_to``, with trapezoid widths for integration."""
    g = np.concatenate([np.linspace(0.0, fine_to, n_fine), np.linspace(fine_to, 1.0, n_coarse)[1:]])
    w = np.empty_like(g)
    d = np.diff(g)
    w[0], w[-1] = d[0] / 2, d[-1] / 2
    w[1:-1] = (d[:-1] + d[1:]) / 2
    return g, w


@dataclass(frozen=True)
class Posterior:
    grid: np.ndarray
    mass: np.ndarray  # probability of each node (density times width), sums to 1

    def cdf(self, x):
        c = np.cumsum(self.mass)
        return float(np.interp(x, self.grid, c))

    def quantile(self, q):
        c = np.cumsum(self.mass)
        return np.interp(q, c, self.grid)

    def prob_below(self, x):
        return self.cdf(x)

    def shortest(self, level=0.95):
        """Shortest interval holding ``level`` of the mass (the posterior is skewed with a bound at zero)."""
        c = np.cumsum(self.mass)
        lo_q = np.linspace(0, 1 - level, 401)
        lo, hi = np.interp(lo_q, c, self.grid), np.interp(lo_q + level, c, self.grid)
        k = int(np.argmin(hi - lo))
        return float(lo[k]), float(hi[k])


def field_loglik(y, n, fp, sens, grid):
    """Binomial log likelihood of y positives in n at each (draw, grid node): shape (draws, nodes)."""
    p = fp[:, None] + grid[None, :] * (sens - fp)[:, None]
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return y * np.log(p) + (n - y) * np.log1p(-p)


def posterior_pi(y, n, spec_draws, sens_draws, grid=None, chunk=4000):
    """Posterior of pi (uniform prior) given draws of (spec, sens) from their distribution given validation data."""
    if grid is None:
        grid, width = prevalence_grid()
    else:
        grid, width = grid
    fp = 1.0 - np.asarray(spec_draws)
    sens = np.asarray(sens_draws)
    acc = np.full(grid.size, -np.inf)
    for s in range(0, fp.size, chunk):
        ll = field_loglik(y, n, fp[s : s + chunk], sens[s : s + chunk], grid)
        acc = np.logaddexp(acc, special.logsumexp(ll, axis=0))
    logpost = acc + np.log(width)
    mass = np.exp(logpost - logpost.max())
    return Posterior(grid, mass / mass.sum())


def joint_draws(y, n, spec_draws, sens_draws, post: Posterior, rng, k=2000):
    """Draws of (pi, spec) from the joint posterior: pi from its marginal, then spec given pi."""
    idx = rng.choice(post.grid.size, size=k, p=post.mass)
    pis = post.grid[idx]
    fp = 1.0 - np.asarray(spec_draws)
    sens = np.asarray(sens_draws)
    out = np.empty(k)
    for i, pi in enumerate(pis):
        p = np.clip(fp + pi * (sens - fp), 1e-15, 1 - 1e-15)
        ll = y * np.log(p) + (n - y) * np.log1p(-p)
        w = np.exp(ll - ll.max())
        out[i] = spec_draws[rng.choice(fp.size, p=w / w.sum())]
    return pis, out


# ----------------------------------------------------------------------------------------------- pooled model


def beta_draws(y, n, rng, k):
    """Draws of a rate given y successes in n with a uniform prior."""
    return rng.beta(y + 1.0, n - y + 1.0, k)


def pooled_posterior(spec_yn, sens_yn, rng, k=40_000, y=FIELD_POS, n=FIELD_N):
    spec = beta_draws(*spec_yn, rng, k)
    sens = beta_draws(*sens_yn, rng, k)
    return posterior_pi(y, n, spec, sens), spec, sens


# ----------------------------------------------------------------------------------------------- hierarchical model

GH_X, GH_W = hermgauss(60)


def study_marginal_loglik(mu, sigma, y, n):
    """log of the integral of Binomial(y | n, logistic(z)) over z ~ Normal(mu, sigma), by Gauss-Hermite."""
    z = mu[..., None] + np.sqrt(2.0) * sigma[..., None] * GH_X
    ll = y * -np.logaddexp(0.0, -z) + (n - y) * -np.logaddexp(0.0, z)
    return special.logsumexp(ll + np.log(GH_W / np.sqrt(np.pi)), axis=-1)


@dataclass(frozen=True)
class HyperGrid:
    mu: np.ndarray
    sigma: np.ndarray
    mass: np.ndarray  # (len(mu), len(sigma)), sums to 1


def hyper_posterior(studies, tau, mu_range, sigma_max, m_mu=241, m_sigma=181, mu_prior=(4.0, 2.0)):
    """Posterior of (mu, sigma) on the logit scale: mu ~ Normal(4, 2), sigma ~ half-Normal(0, tau)."""
    mu = np.linspace(*mu_range, m_mu)
    sigma = np.linspace(0.0, sigma_max, m_sigma)
    mm, ss = np.meshgrid(mu, sigma, indexing="ij")
    lp = stats.norm.logpdf(mm, *mu_prior) + stats.halfnorm.logpdf(ss, scale=tau)
    for y, n in studies:
        lp = lp + study_marginal_loglik(mm, ss, y, n)
    w = np.exp(lp - lp.max())
    return HyperGrid(mu, sigma, w / w.sum())


def hyper_edge_mass(h: HyperGrid):
    return float(h.mass[0].sum() + h.mass[-1].sum() + h.mass[:, -1].sum())


def predictive_draws(h: HyperGrid, rng, k):
    """Draws of the rate in a new study: (mu, sigma) from the grid posterior, then logistic(Normal(mu, sigma))."""
    idx = rng.choice(h.mass.size, size=k, p=h.mass.ravel())
    i, j = np.unravel_index(idx, h.mass.shape)
    dmu, dsig = h.mu[1] - h.mu[0], h.sigma[1] - h.sigma[0]
    mu = h.mu[i] + dmu * (rng.random(k) - 0.5)
    sig = np.abs(h.sigma[j] + dsig * (rng.random(k) - 0.5))
    return special.expit(mu + sig * rng.standard_normal(k)), sig


def hierarchical_posterior(tau_spec, tau_sens, rng, k=200_000, y=FIELD_POS, n=FIELD_N):
    hs = hyper_posterior(SPEC_STUDIES, tau_spec, (0.0, 12.0), 4.0 * tau_spec + 1.0)
    hd = hyper_posterior(SENS_STUDIES, tau_sens, (-6.0, 8.0), 4.0 * tau_sens + 1.0)
    spec, sig_s = predictive_draws(hs, rng, k)
    sens, sig_d = predictive_draws(hd, rng, k)
    return posterior_pi(y, n, spec, sens), spec, sens, hs, hd


# ----------------------------------------------------------------------------------------------- checks


def pooled_ppc(rng, k=20_000):
    """Posterior predictive check of the pooled-specificity assumption across the 13 studies.

    Statistic: sum over studies of (false positives - expected)^2 / expected, with the expectation from the
    replicate's own pooled rate. Also the chance, under pooling, of no false positive in the 1,102-sample study.
    """
    y, n = pooled(SPEC_STUDIES)
    fp_rate = rng.beta(n - y + 1.0, y + 1.0, k)
    sizes = np.array([s[1] for s in SPEC_STUDIES])
    fp_obs = sizes - np.array([s[0] for s in SPEC_STUDIES])

    def stat(fp):
        rate = fp.sum(axis=-1, keepdims=True) / sizes.sum()
        e = np.maximum(sizes * rate, 1e-9)
        return np.sum((fp - e) ** 2 / e, axis=-1)

    rep = rng.binomial(sizes[None, :], fp_rate[:, None])
    t_obs, t_rep = stat(fp_obs[None, :])[0], stat(rep)
    big = int(np.argmax(sizes))
    return {
        "t_obs": float(t_obs),
        "p_value": float(np.mean(t_rep >= t_obs)),
        "p_zero_in_largest": float(np.mean(rep[:, big] == 0)),
        "largest_n": int(sizes[big]),
    }


def calibration(rng, sims=300, k=3000, n_spec=401, n_sens=122, n_field=FIELD_N):
    """Posterior quantile of the true prevalence (Cook, Gelman and Rubin 2006) for the pooled engine.

    Parameters are drawn from the model's own priors (uniform on pi, spec, sens), data simulated, and the
    posterior computed by the same code used everywhere else.
    """
    grid = prevalence_grid(n_fine=801, n_coarse=301)
    u = np.empty(sims)
    for s in range(sims):
        pi, spec, sens = rng.random(3)
        ys, yd = rng.binomial(n_spec, spec), rng.binomial(n_sens, sens)
        y = rng.binomial(n_field, pi * sens + (1 - pi) * (1 - spec))
        post = posterior_pi(y, n_field, beta_draws(ys, n_spec, rng, k), beta_draws(yd, n_sens, rng, k), grid)
        u[s] = post.cdf(pi)
    return u


# ----------------------------------------------------------------------------------------------- decision 2


def design_widths(n_spec, n_sens, n_field, pi, spec, sens, rng, reps=200, kq=(64, 8), grid=None, chunk=20):
    """Simulate surveys with a given split of tests and return, per simulated survey, the posterior's
    5% and 95% quantiles of the prevalence (pooled model, same-lab validation, uniform priors).

    (spec, sens) are integrated with stratified quantiles of their Beta posteriors (kq[0] x kq[1] points).
    """
    if grid is None:
        grid = prevalence_grid(fine_to=0.12, n_fine=481, n_coarse=2)
    g, width = grid
    ys = rng.binomial(n_spec, spec, reps)
    yd = rng.binomial(n_sens, sens, reps)
    yf = rng.binomial(n_field, pi * sens + (1 - pi) * (1 - spec), reps)
    us = (np.arange(kq[0]) + 0.5) / kq[0]
    ud = (np.arange(kq[1]) + 0.5) / kq[1]
    q05, q95 = np.empty(reps), np.empty(reps)
    for s in range(0, reps, chunk):
        sl = slice(s, min(s + chunk, reps))
        sp = stats.beta.ppf(us[None, :], ys[sl, None] + 1.0, n_spec - ys[sl, None] + 1.0)  # (r, kq0)
        se = stats.beta.ppf(ud[None, :], yd[sl, None] + 1.0, n_sens - yd[sl, None] + 1.0)  # (r, kq1)
        fp = (1 - sp)[:, :, None]
        sens_ = se[:, None, :]
        p = fp[..., None] + g[None, None, None, :] * (sens_ - fp)[..., None]  # (r, kq0, kq1, nodes)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        y = yf[sl, None, None, None]
        ll = y * np.log(p) + (n_field - y) * np.log1p(-p)
        lp = special.logsumexp(ll, axis=(1, 2)) + np.log(width)[None, :]
        m = np.exp(lp - lp.max(axis=1, keepdims=True))
        c = np.cumsum(m, axis=1)
        c /= c[:, -1:]
        for r in range(c.shape[0]):
            q05[s + r], q95[s + r] = np.interp([0.05, 0.95], c[r], g)
    return q05, q95


def other_lab_floor(sigma, mean_spec, sens_yn, rng, k=200_000, y=FIELD_POS, n=FIELD_N):
    """Prevalence posterior when the survey lab's specificity is only known to come from a population of labs
    with mean logit(mean_spec) and between-lab sd ``sigma`` (as if other labs' validation were unlimited)."""
    spec = special.expit(special.logit(mean_spec) + sigma * rng.standard_normal(k))
    sens = beta_draws(*sens_yn, rng, k)
    return posterior_pi(y, n, spec, sens)


def same_lab(n_neg, true_spec, sens_yn, rng, k=200_000, y=FIELD_POS, n=FIELD_N):
    """Prevalence posterior after ``n_neg`` known negatives tested in the survey's own lab, at the expected count."""
    ys = n_neg * true_spec
    spec = rng.beta(ys + 1.0, n_neg - ys + 1.0, k)
    sens = beta_draws(*sens_yn, rng, k)
    return posterior_pi(y, n, spec, sens)
