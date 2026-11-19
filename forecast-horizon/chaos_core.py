"""Models, integrators and measurements for the forecast-horizon article.

Everything that feeds back into a chaotic integration is computed with elementwise operations or exactly
rounded sums (math.fsum), so a run gives the same numbers on any machine.
"""

import math

import numpy as np
from scipy.io import netcdf_file
from scipy.special import erf

# Lorenz (1996): one time unit of the Lorenz-96 model corresponds to about five days of weather.
DAYS_PER_UNIT = 5.0


# ---------------------------------------------------------------- models


def rk4(f, x, dt):
    """One fourth-order Runge-Kutta step."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def lorenz63(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz (1963), three variables; works on the last axis."""
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([sigma * (b - a), a * (rho - c) - b, a * b - beta * c], axis=-1)


def lorenz96(x, forcing=8.0):
    """Lorenz (1996), one scale: N variables on a circle of latitude; works on the last axis."""
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + forcing


class TwoScale:
    """Lorenz (1996) with two scales: K slow variables X, each coupled to J fast variables Y.

    The state is [X_1..X_K, Y_1..Y_KJ]; Y_(k-1)J+1 .. Y_kJ belong to X_k and the Y run round the circle too.
    """

    def __init__(self, k=36, j=10, forcing=10.0, h=1.0, c=10.0, b=10.0):
        self.k, self.j, self.forcing = k, j, forcing
        self.coupling = h * c / b
        self.c, self.cb = c, c * b

    def __call__(self, s):
        k, j = self.k, self.j
        x, y = s[..., :k], s[..., k:]
        blocks = y.reshape(*y.shape[:-1], k, j)
        ysum = blocks[..., 0].copy()
        for i in range(1, j):  # a fixed order of additions
            ysum = ysum + blocks[..., i]
        dx = (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + self.forcing - self.coupling * ysum
        dy = (
            -self.cb * np.roll(y, -1, -1) * (np.roll(y, -2, -1) - np.roll(y, 1, -1))
            - self.c * y
            + self.coupling * np.repeat(x, j, -1)
        )
        return np.concatenate([dx, dy], axis=-1)


def run(f, x, dt, steps):
    """Integrate for a number of steps and return the final state."""
    for _ in range(steps):
        x = rk4(f, x, dt)
    return x


def sample(f, x, dt, steps, every):
    """Integrate and keep every `every`-th state."""
    out = []
    for i in range(1, steps + 1):
        x = rk4(f, x, dt)
        if i % every == 0:
            out.append(x.copy())
    return np.array(out), x


# ---------------------------------------------------------------- measurements


def norm(v):
    """Euclidean norm with an exactly rounded sum."""
    return math.sqrt(math.fsum((np.asarray(v, dtype=float) ** 2).ravel()))


def fixed(value, digits=6):
    """Round a scale that feeds a chaotic run, so the last bits of a library sum cannot change the run."""
    return float(round(value, digits))


def lyapunov(f, x, dt, steps, renorm=10, eps=1e-8, seed=0):
    """Largest Lyapunov exponent by renormalising a nearby trajectory (Benettin's method), per unit time."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(x.shape)
    y = x + d * (eps / norm(d))
    total = 0.0
    for i in range(1, steps + 1):
        x = rk4(f, x, dt)
        y = rk4(f, y, dt)
        if i % renorm == 0:
            dist = norm(y - x)
            total += math.log(dist / eps)
            y = x + (y - x) * (eps / dist)
    return total / (steps * dt)


def rms_rows(v, part=slice(None)):
    """Root-mean-square over the variables in `part`, one value per row."""
    w = v[..., part]
    return np.sqrt((w**2).mean(axis=-1))


def error_curves(f, starts, perturbation, dt, steps, every, part=slice(None)):
    """Error of a perturbed run against the unperturbed one, every `every` steps.

    Returns times (model units) and an array (times, starts) of RMS errors over the variables in `part`.
    """
    a = starts.copy()
    b = starts + perturbation
    out = [rms_rows(a - b, part)]
    for i in range(1, steps + 1):
        a = rk4(f, a, dt)
        b = rk4(f, b, dt)
        if i % every == 0:
            out.append(rms_rows(a - b, part))
    times = np.arange(len(out)) * every * dt
    return times, np.array(out)


def pooled(err):
    """Pool errors over starting states: square root of the mean squared error."""
    return np.sqrt((err**2).mean(axis=-1))


def first_crossing(times, values, level):
    """First time a curve reaches `level`, interpolating linearly between samples; nan if never."""
    above = np.nonzero(values >= level)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(times[0])
    t0, t1, v0, v1 = times[i - 1], times[i], values[i - 1], values[i]
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0))


def horizons(f, starts, scales, epsilons, dt, steps, every, part, level, seed):
    """Forecast horizon (time to reach `level`) for initial errors eps * scales, one eps at a time."""
    rng = np.random.default_rng(seed)
    out, curves = [], []
    for eps in epsilons:
        pert = rng.standard_normal(starts.shape) * scales * eps
        times, err = error_curves(f, starts, pert, dt, steps, every, part)
        p = pooled(err)
        out.append(first_crossing(times, p, level))
        curves.append(p)
    return np.array(out), times, np.array(curves)


def days_per_decade(epsilons, hz):
    """Days gained by each tenfold reduction of the initial error."""
    return -np.diff(hz) * DAYS_PER_UNIT / np.diff(np.log10(epsilons))


# ---------------------------------------------------------------- ensembles


def crps_normal(mu, sd, y):
    """Continuous ranked probability score of a normal forecast distribution."""
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + erf(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def crps_fair(members, y):
    """Fair ensemble CRPS (Ferro 2014): unbiased for the score of the distribution the members come from.

    members: (..., m), y: (...).
    """
    m = members.shape[-1]
    t1 = np.abs(members - y[..., None]).mean(axis=-1)
    srt = np.sort(members, axis=-1)
    i = np.arange(1, m + 1)
    # sum over pairs |x_i - x_j| = 2 * sum_i (2i - m - 1) x_(i)
    pair_sum = 2 * (srt * (2 * i - m - 1)).sum(axis=-1)
    return t1 - pair_sum / (2 * m * (m - 1))


def anomaly_correlation(forecast, truth, clim_mean):
    """Anomaly correlation across variables and cases, uncentred, as weather centres compute it."""
    fa, ta = forecast - clim_mean, truth - clim_mean
    return float((fa * ta).sum() / math.sqrt((fa**2).sum() * (ta**2).sum()))


def ensemble_experiment(f, truths, clim_mean, clim_sd, init_err, members, dt, steps, every, seed):
    """Perfect-model ensemble forecasts.

    Each case has a truth, an analysis (truth plus an error of size init_err * clim_sd), a single forecast
    from the analysis, and `members` forecasts from the analysis plus perturbations of the same size.
    Returns lead times (model units) and, per lead: RMSE of the single forecast and the ensemble mean,
    ensemble spread, anomaly correlations, and the CRPS skill score against climatology.
    """
    rng = np.random.default_rng(seed)
    n, nv = truths.shape
    s = init_err * clim_sd
    truth = truths.copy()
    analysis = truths + rng.standard_normal((n, nv)) * s
    single = analysis.copy()
    ens = analysis[:, None, :] + rng.standard_normal((n, members, nv)) * s
    rows = []
    for i in range(steps + 1):
        if i > 0:
            truth = rk4(f, truth, dt)
            single = rk4(f, single, dt)
            ens = rk4(f, ens, dt)
        if i % every == 0:
            mean = ens.mean(axis=1)
            crps_e = crps_fair(np.moveaxis(ens, 1, -1), truth).mean()
            crps_c = crps_normal(clim_mean, clim_sd, truth).mean()
            rows.append(
                (
                    i * dt,
                    float(np.sqrt(((single - truth) ** 2).mean())),
                    float(np.sqrt(((mean - truth) ** 2).mean())),
                    float(np.sqrt(ens.var(axis=1, ddof=1).mean())),
                    anomaly_correlation(single, truth, clim_mean),
                    anomaly_correlation(mean, truth, clim_mean),
                    float(1 - crps_e / crps_c),
                )
            )
    return np.array(rows), (truth, single, ens)


# ---------------------------------------------------------------- the real atmosphere


def load_air(path):
    """NCEP/NCAR reanalysis near-surface air temperature, 6-hourly, from xarray's tutorial file."""
    f = netcdf_file(path, "r", mmap=False)
    air = f.variables["air"][:].astype(float) * float(f.variables["air"].scale_factor)
    lat = f.variables["lat"][:].astype(float)
    lon = f.variables["lon"][:].astype(float)
    hours = f.variables["time"][:].astype(float)
    f.close()
    return air, lat, lon, hours


def daily_anomalies(air, lat, hours, harmonics=3):
    """Daily means, minus each grid point's seasonal cycle (mean and `harmonics` annual harmonics).

    Returns anomalies scaled by the square root of area weights (cos latitude, summing to one), so that the
    squared Euclidean distance between two days is their area-weighted mean squared difference; and the day
    of the year of each day.
    """
    n = air.shape[0] // 4
    daily = air[: 4 * n].reshape(n, 4, -1).mean(axis=1)
    day = hours[::4][:n] / 24.0
    cols = [np.ones(n)]
    for k in range(1, harmonics + 1):
        w = 2 * np.pi * k * day / 365.25
        cols += [np.cos(w), np.sin(w)]
    design = np.array(cols).T
    coef, *_ = np.linalg.lstsq(design, daily, rcond=None)
    anom = daily - design @ coef
    weights = np.repeat(np.cos(np.deg2rad(lat)), air.shape[2])
    weights = weights / weights.sum()
    return anom * np.sqrt(weights), day % 365.25


def distances(a):
    """Euclidean distances between all pairs of rows."""
    g = a @ a.T
    sq = np.diag(g)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * g, 0))


def analogue_divergence(a, doy, horizon=20, min_sep=30, season=45):
    """For each day, the closest other day of the same season (at least `min_sep` days apart), and how the
    distance between the two sequences develops over the next `horizon` days.

    Returns the curves (pairs, horizon + 1), the eligible-pair mask and the distance matrix.
    """
    n = len(a)
    d = distances(a)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    dd = np.abs(doy[:, None] - doy[None, :])
    dd = np.minimum(dd, 365.25 - dd)
    ok = (sep >= min_sep) & (dd <= season) & (idx[:, None] < n - horizon) & (idx[None, :] < n - horizon)
    masked = np.where(ok, d, np.inf)
    best = masked.argmin(axis=1)
    have = np.isfinite(masked[idx, best])
    rows = idx[have]
    steps = np.arange(horizon + 1)
    curves = d[rows[:, None] + steps[None, :], best[rows][:, None] + steps[None, :]]
    return curves, ok, d


def saturation(a):
    """Expected distance between two independent days: square root of twice the mean squared anomaly."""
    return float(np.sqrt(2 * (a**2).sum(axis=1).mean()))


def leading_components(a, r):
    """Leading r principal components of the rows of a, and the share of variance they hold."""
    centred = a - a.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :r] * s[:r], float((s[:r] ** 2).sum() / (s**2).sum())


def var1_surrogate(p, n, seed, burn=200):
    """A linear stochastic copy of a multivariate series: x_t+1 = x_t A + noise, fitted by least squares
    and simulated for n steps. No chaos, only memory and random forcing."""
    a = np.linalg.lstsq(p[:-1], p[1:], rcond=None)[0]
    resid = p[1:] - p[:-1] @ a
    chol = np.linalg.cholesky(np.cov(resid.T) + 1e-9 * np.eye(p.shape[1]))
    rng = np.random.default_rng(seed)
    z = np.zeros(p.shape[1])
    out = np.empty((n, p.shape[1]))
    for t in range(n + burn):
        z = z @ a + chol @ rng.standard_normal(p.shape[1])
        if t >= burn:
            out[t - burn] = z
    return out


def relative_curve(curves, sat):
    """Pooled analogue distance at each lead, as a share of the saturation distance."""
    return np.sqrt((curves**2).mean(axis=0)) / sat
