"""Methods for ranking groups from means of unequal sample sizes.

Model, for group i with n_i ratings and mean y_i:

    y_i | theta_i ~ N(theta_i, sigma^2 / n_i)          sampling noise of a mean of n_i ratings
    theta_i       ~ N(x_i' beta, tau^2)                   true group means around a regression line

sigma is the spread of single ratings within a group, tau the spread of the true group means. Both are
unknown. The regression coefficients beta get a flat prior and are integrated out in closed form, so
the posterior of (sigma, tau) is computed exactly on a grid, with uniform priors on both. Draws of
beta and theta then follow from normal conditionals.
"""

import numpy as np
from scipy import stats

SIGMA_GRID = np.round(np.arange(0.50, 4.0001, 0.01), 4)
TAU_GRID = np.round(np.arange(0.005, 3.0001, 0.01), 4)


def design(n_groups, covariate=None):
    """Intercept-only design, or intercept and one centred covariate."""
    if covariate is None:
        return np.ones((n_groups, 1))
    x = np.asarray(covariate, dtype=float)
    return np.column_stack([np.ones(n_groups), x - x.mean()])


def log_evidence(y, n, X, sigma, tau):
    """log p(y | sigma, tau) with beta integrated out under a flat prior, up to a constant.

    sigma and tau broadcast against each other (e.g. a grid); y, n are length N; X is N x p.
    """
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    s = np.asarray(sigma, float)[..., None]
    t = np.asarray(tau, float)[..., None]
    v = t**2 + s**2 / n  # (..., N)
    w = 1.0 / v
    A = np.einsum("...n,np,nq->...pq", w, X, X)
    rhs = np.einsum("...n,np,n->...p", w, X, y)
    bhat = np.linalg.solve(A, rhs[..., None])[..., 0]
    resid = y - np.einsum("np,...p->...n", X, bhat)
    _, logdet = np.linalg.slogdet(A)
    return -0.5 * np.log(v).sum(-1) - 0.5 * logdet - 0.5 * (w * resid**2).sum(-1)


def grid_posterior(y, n, X, sigma_grid=SIGMA_GRID, tau_grid=TAU_GRID):
    """Posterior probabilities of (sigma, tau) on the grid (rows sigma, columns tau)."""
    S, T = np.meshgrid(sigma_grid, tau_grid, indexing="ij")
    le = log_evidence(y, n, X, S, T)
    p = np.exp(le - le.max())
    return p / p.sum()


def posterior_draws(y, n, X, draws, rng, sigma_grid=SIGMA_GRID, tau_grid=TAU_GRID):
    """Joint posterior draws of sigma, tau, beta and the true group means theta."""
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    post = grid_posterior(y, n, X, sigma_grid, tau_grid)
    flat = rng.choice(post.size, size=draws, p=post.ravel())
    i, j = np.unravel_index(flat, post.shape)
    ds = sigma_grid[1] - sigma_grid[0]
    dt = tau_grid[1] - tau_grid[0]
    sigma = sigma_grid[i] + rng.uniform(-ds / 2, ds / 2, draws)
    tau = np.clip(tau_grid[j] + rng.uniform(-dt / 2, dt / 2, draws), 1e-6, None)
    v = tau[:, None] ** 2 + sigma[:, None] ** 2 / n
    w = 1.0 / v
    A = np.einsum("dn,np,nq->dpq", w, X, X)
    rhs = np.einsum("dn,np,n->dp", w, X, y)
    Ainv = np.linalg.inv(A)
    bhat = np.einsum("dpq,dq->dp", Ainv, rhs)
    L = np.linalg.cholesky(Ainv)
    beta = bhat + np.einsum("dpq,dq->dp", L, rng.standard_normal(bhat.shape))
    prior_mean = beta @ X.T  # (draws, N)
    prec = 1.0 / tau[:, None] ** 2 + n / sigma[:, None] ** 2
    mean = (prior_mean / tau[:, None] ** 2 + n * y / sigma[:, None] ** 2) / prec
    theta = mean + rng.standard_normal(mean.shape) / np.sqrt(prec)
    return {"sigma": sigma, "tau": tau, "beta": beta, "theta": theta, "grid": post}


def shrinkage_weight(n, sigma, tau):
    """Weight on a group's own mean in the posterior mean: tau^2 / (tau^2 + sigma^2 / n)."""
    return tau**2 / (tau**2 + sigma**2 / np.asarray(n, float))


def ranks_desc(values):
    """Rank 1 = largest, along the last axis."""
    order = np.argsort(-np.asarray(values), axis=-1, kind="stable")
    return np.argsort(order, axis=-1, kind="stable") + 1


def interval(x, level=0.90, axis=0):
    """Equal-tailed interval."""
    a = (1 - level) / 2
    return np.quantile(x, a, axis=axis), np.quantile(x, 1 - a, axis=axis)


def rank_interval(rank_draws, level=0.90):
    """Equal-tailed interval of integer ranks (lower quantile rounded down, upper rounded up)."""
    lo, hi = interval(rank_draws, level, axis=0)
    return np.floor(lo).astype(int), np.ceil(hi).astype(int)


def funnel_limits(centre, sigma, n, z):
    """Control limits centre +/- z sigma / sqrt(n) for a mean of n observations."""
    half = z * sigma / np.sqrt(np.asarray(n, float))
    return centre - half, centre + half


def extreme_positions(means, k=3):
    """Indices of the k largest and the k smallest values."""
    order = np.argsort(-np.asarray(means), kind="stable")
    return order[:k], order[-k:]


def thin_at_extremes(means, thin, k=3):
    """How many of the k top and k bottom positions belong to thin groups (last axis = groups)."""
    means = np.asarray(means)
    thin = np.asarray(thin, bool)
    order = np.argsort(-means, axis=-1, kind="stable")
    ends = np.concatenate([order[..., :k], order[..., -k:]], axis=-1)
    return thin[ends].sum(-1)


def simulate_means(n, mu, sigma, tau, rng):
    """Group means from the model; mu, sigma, tau are scalars or arrays of draws (one row per draw)."""
    n = np.asarray(n, float)
    mu = np.atleast_1d(mu)[:, None]
    sigma = np.atleast_1d(sigma)[:, None]
    tau = np.atleast_1d(tau)[:, None]
    size = (max(len(mu), len(sigma), len(tau)), len(n))
    theta = mu + tau * rng.standard_normal(size)
    return theta + sigma / np.sqrt(n) * rng.standard_normal(size), theta


def pearson(x, y):
    """Pearson correlation along the last axis."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean(-1, keepdims=True)
    yc = y - y.mean(-1, keepdims=True)
    return (xc * yc).sum(-1) / np.sqrt((xc**2).sum(-1) * (yc**2).sum(-1))


def spearman(x, y):
    """Spearman rank correlation of two vectors."""
    return float(stats.spearmanr(x, y).statistic)


# ---------------------------------------------------------------- discrete 0-10 ratings, for the check


def skewed_noise(size, sd, rng, shape=4.0):
    """Mean-zero noise with standard deviation sd and a long left tail (skewness -2/sqrt(shape))."""
    g = rng.gamma(shape, 1.0, size)
    return -sd * (g - shape) / np.sqrt(shape)


def ratings(location, size, sd, rng):
    """Integer ratings 0-10: location plus skewed noise, rounded and clipped."""
    return np.clip(np.rint(location + skewed_noise(size, sd, rng)), 0, 10)


def expected_rating_curve(locations, sd, rng, size=400_000):
    """E[rating | location] on a grid of locations, by Monte Carlo with common random numbers."""
    e = skewed_noise(size, sd, rng)
    return np.array([np.clip(np.rint(loc + e), 0, 10).mean() for loc in locations])
