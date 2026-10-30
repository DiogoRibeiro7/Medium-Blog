"""Small, transparent estimators for the integrated assessment (standard library only).

Everything here works on plain lists and dictionaries so each step can be
checked by hand: least squares by Gaussian elimination, the two-way within
transformation of a balanced panel, cluster-robust (CR1) standard errors, a wild
cluster bootstrap for few clusters, the Wooldridge test for serial correlation in
first differences, bootstrap intervals for correlations, and Fisher intervals with
an effective sample size for autocorrelated time series. No estimator here
identifies a causal effect; they describe conditional associations.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

Matrix = list[list[float]]
Vector = list[float]
U = TypeVar("U", bound=Hashable)


def solve(a: Matrix, b: Vector) -> Vector:
    """Solve a x = b by Gaussian elimination with partial pivoting (raises if singular)."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix: a regressor is collinear or has no variation")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def inverse(a: Matrix) -> Matrix:
    """Inverse of a small square matrix."""
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def crossprod(x: Sequence[Vector]) -> Matrix:
    """X'X for rows of X."""
    k = len(x[0])
    out = [[0.0] * k for _ in range(k)]
    for row in x:
        for i in range(k):
            ri = row[i]
            if ri:
                for j in range(i, k):
                    out[i][j] += ri * row[j]
    for i in range(k):
        for j in range(i):
            out[i][j] = out[j][i]
    return out


def matvec(a: Matrix, v: Vector) -> Vector:
    """Matrix-vector product."""
    return [sum(ai * vi for ai, vi in zip(row, v, strict=True)) for row in a]


@dataclass
class Fit:
    """A least-squares fit with cluster-robust inference."""

    names: list[str]
    beta: Vector
    se: Vector
    n: int
    clusters: int
    resid: Vector = field(repr=False)
    vcov: Matrix = field(repr=False)
    conventional_se: Vector = field(default_factory=list)

    def ci(self, j: int, level: float = 0.95) -> tuple[float, float]:
        """Normal-approximation interval using t with G-1 degrees of freedom."""
        q = t_quantile(0.5 + level / 2, self.clusters - 1)
        return self.beta[j] - q * self.se[j], self.beta[j] + q * self.se[j]

    def as_dict(self, digits: int = 4) -> dict[str, object]:
        """Coefficients, CR1 standard errors and 95% intervals, rounded."""
        return {
            "n": self.n, "clusters": self.clusters,
            "coefficients": {name: {"estimate": round(self.beta[j], digits), "se_cr1": round(self.se[j], digits),
                                    "ci95_cr1": [round(v, digits) for v in self.ci(j)],
                                    "se_conventional": round(self.conventional_se[j], digits) if self.conventional_se else None}
                             for j, name in enumerate(self.names)},
        }


def ols(x: Sequence[Vector], y: Vector, clusters: Sequence[Hashable], names: list[str]) -> Fit:
    """OLS with CR1 cluster-robust variance: (G/(G-1)) ((N-1)/(N-K)) A [sum_g s_g s_g'] A.

    K counts the regressors in ``x`` only; fixed effects removed beforehand by
    demeaning are nested in the clusters when clusters are the panel units.
    """
    n, k = len(x), len(x[0])
    if n != len(y) or n != len(clusters) or n <= k:
        raise ValueError("Inconsistent or too few observations")
    xtx = crossprod(x)
    a = inverse(xtx)
    xty = [sum(row[i] * yi for row, yi in zip(x, y, strict=True)) for i in range(k)]
    beta = matvec(a, xty)
    resid = [yi - sum(b * xi for b, xi in zip(beta, row, strict=True)) for row, yi in zip(x, y, strict=True)]
    scores: dict[Hashable, Vector] = defaultdict(lambda: [0.0] * k)
    for row, u, g in zip(x, resid, clusters, strict=True):
        s = scores[g]
        for i in range(k):
            s[i] += row[i] * u
    meat = [[sum(s[i] * s[j] for s in scores.values()) for j in range(k)] for i in range(k)]
    g_count = len(scores)
    if g_count < 2:
        raise ValueError("Need at least two clusters")
    adj = (g_count / (g_count - 1)) * ((n - 1) / (n - k))
    am = [[sum(a[i][m] * meat[m][j] for m in range(k)) for j in range(k)] for i in range(k)]
    vcov = [[adj * sum(am[i][m] * a[m][j] for m in range(k)) for j in range(k)] for i in range(k)]
    sigma2 = sum(u * u for u in resid) / (n - k)
    conventional = [math.sqrt(sigma2 * a[j][j]) for j in range(k)]
    return Fit(names, beta, [math.sqrt(max(vcov[j][j], 0.0)) for j in range(k)], n, g_count, resid, vcov, conventional)


def wild_cluster_bootstrap(x: Sequence[Vector], y: Vector, clusters: Sequence[Hashable], j: int,
                           reps: int = 1999, seed: int = 20261029) -> dict[str, float]:
    """Wild cluster bootstrap-t (Rademacher weights) for coefficient ``j``.

    Re-uses X'X: each replication only re-weights the cluster scores, so the
    bootstrap is exact for the linear model and cheap. Returns

    * a percentile-t 95% interval from the unrestricted bootstrap (WCU);
    * the symmetric p-value for beta_j = 0 from the unrestricted bootstrap; and
    * the symmetric p-value from the restricted bootstrap (WCR), which imposes
      beta_j = 0 when generating the bootstrap samples and is the better-behaved
      test with few or unequal clusters (Cameron, Gelbach and Miller 2008;
      MacKinnon and Webb 2017).
    """
    fit = ols(x, y, clusters, [str(i) for i in range(len(x[0]))])
    k = len(x[0])
    a = inverse(crossprod(x))
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for idx, g in enumerate(clusters):
        groups[g].append(idx)
    keys = list(groups)
    s = {g: [sum(x[i][c] * fit.resid[i] for i in groups[g]) for c in range(k)] for g in keys}
    m = {g: crossprod([x[i] for i in groups[g]]) for g in keys}
    n, g_count = len(y), len(keys)
    adj = (g_count / (g_count - 1)) * ((n - 1) / (n - k))
    rng = random.Random(seed)
    t_stats = []
    for _ in range(reps):
        v = {g: (1.0 if rng.random() < 0.5 else -1.0) for g in keys}
        total = [sum(v[g] * s[g][c] for g in keys) for c in range(k)]
        delta = matvec(a, total)                       # beta* - beta_hat
        meat = [[0.0] * k for _ in range(k)]
        for g in keys:
            md = matvec(m[g], delta)
            sg = [v[g] * s[g][c] - md[c] for c in range(k)]  # X_g' u*_g
            for p in range(k):
                for q in range(k):
                    meat[p][q] += sg[p] * sg[q]
        var_j = adj * sum(a[j][p] * meat[p][q] * a[q][j] for p in range(k) for q in range(k))
        if var_j > 0:
            t_stats.append(delta[j] / math.sqrt(var_j))
    t_stats.sort()
    lo_q, hi_q = quantile(t_stats, 0.025), quantile(t_stats, 0.975)
    t_obs = fit.beta[j] / fit.se[j] if fit.se[j] > 0 else float("inf")
    p_value = sum(1 for t in t_stats if abs(t) >= abs(t_obs)) / len(t_stats)

    # Restricted bootstrap: residuals and fitted values of the model without regressor j.
    if k == 1:
        y_null, u_null = [0.0] * n, list(y)
    else:
        xr = [[row[c] for c in range(k) if c != j] for row in x]
        restricted = ols(xr, y, clusters, [str(c) for c in range(k - 1)])
        u_null = restricted.resid
        y_null = [yi - ui for yi, ui in zip(y, u_null, strict=True)]
    b_null = matvec(a, [sum(row[c] * yi for row, yi in zip(x, y_null, strict=True)) for c in range(k)])
    s_null = {g: [sum(x[i][c] * u_null[i] for i in groups[g]) for c in range(k)] for g in keys}
    f_null = {g: [sum(x[i][c] * y_null[i] for i in groups[g]) for c in range(k)] for g in keys}
    rng_r = random.Random(seed + 1)
    exceed, valid = 0, 0
    for _ in range(reps):
        v = {g: (1.0 if rng_r.random() < 0.5 else -1.0) for g in keys}
        total = [sum(v[g] * s_null[g][c] for g in keys) for c in range(k)]
        beta_star = [b + d for b, d in zip(b_null, matvec(a, total), strict=True)]
        meat = [[0.0] * k for _ in range(k)]
        for g in keys:
            mb = matvec(m[g], beta_star)
            sg = [f_null[g][c] + v[g] * s_null[g][c] - mb[c] for c in range(k)]
            for p in range(k):
                for q in range(k):
                    meat[p][q] += sg[p] * sg[q]
        var_j = adj * sum(a[j][p] * meat[p][q] * a[q][j] for p in range(k) for q in range(k))
        if var_j > 0:
            valid += 1
            exceed += abs(beta_star[j] / math.sqrt(var_j)) >= abs(t_obs)
    return {"ci95_low": fit.beta[j] - hi_q * fit.se[j], "ci95_high": fit.beta[j] - lo_q * fit.se[j],
            "p_value": p_value, "p_value_restricted": exceed / valid if valid else float("nan"),
            "reps": float(len(t_stats)), "clusters": float(g_count)}


def two_way_within(values: Mapping[tuple[U, int], float], units: Sequence[U], years: Sequence[int]) -> dict[tuple[U, int], float]:
    """x_it - mean_i - mean_t + grand mean, for a balanced panel (raises if a cell is missing)."""
    for u in units:
        for t in years:
            if (u, t) not in values:
                raise ValueError(f"Unbalanced panel: missing {(u, t)}")
    unit_mean = {u: sum(values[(u, t)] for t in years) / len(years) for u in units}
    year_mean = {t: sum(values[(u, t)] for u in units) / len(units) for t in years}
    grand = sum(unit_mean.values()) / len(units)
    return {(u, t): values[(u, t)] - unit_mean[u] - year_mean[t] + grand for u in units for t in years}


def within_share(values: Mapping[tuple[U, int], float], units: Sequence[U], years: Sequence[int]) -> float:
    """Share of the total variance of a balanced panel variable left after two-way demeaning."""
    data = [values[(u, t)] for u in units for t in years]
    mean = sum(data) / len(data)
    total = sum((v - mean) ** 2 for v in data)
    within = two_way_within(values, units, years)
    return sum(v * v for v in within.values()) / total if total > 0 else 0.0


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence."""
    if not sorted_values:
        raise ValueError("Empty input")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def t_quantile(p: float, df: int) -> float:
    """Quantile of Student's t (Hill's approximation, accurate to ~1e-4 for df >= 3)."""
    if df < 1:
        raise ValueError("df must be positive")
    z = normal_quantile(p)
    if df > 1000:
        return z
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation."""
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("Need two equal-length sequences of at least three values")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("Constant input")
    return sxy / math.sqrt(sxx * syy)


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            out[k] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation."""
    return pearson(ranks(x), ranks(y))


def cluster_bootstrap_corr(x: Sequence[float], y: Sequence[float], clusters: Sequence[Hashable],
                           reps: int = 1999, seed: int = 20261029) -> dict[str, float]:
    """Pearson and Spearman correlations with cluster-bootstrap percentile 95% intervals.

    Resampling whole clusters (e.g. NUTS 3 regions) keeps neighbouring
    municipalities together, so the interval reflects spatial dependence.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, g in enumerate(clusters):
        groups[g].append(i)
    keys = list(groups)
    rng = random.Random(seed)
    boots: dict[str, list[float]] = {"pearson": [], "spearman": []}
    for _ in range(reps):
        idx = [i for _k in keys for i in groups[keys[rng.randrange(len(keys))]]]
        bx, by = [x[i] for i in idx], [y[i] for i in idx]
        try:
            boots["pearson"].append(pearson(bx, by))
            boots["spearman"].append(spearman(bx, by))
        except ValueError:
            continue
    out: dict[str, float] = {"n": float(len(x)), "clusters": float(len(keys))}
    for name, fn in (("pearson", pearson), ("spearman", spearman)):
        b = sorted(boots[name])
        out[name] = fn(x, y)
        out[f"{name}_ci95_low"], out[f"{name}_ci95_high"] = quantile(b, 0.025), quantile(b, 0.975)
    return out


def autocorrelation(x: Sequence[float], lag: int = 1) -> float:
    """Sample autocorrelation at ``lag``."""
    n = len(x)
    m = sum(x) / n
    den = sum((v - m) ** 2 for v in x)
    if den == 0 or n <= lag:
        raise ValueError("Constant or too short series")
    return sum((x[t] - m) * (x[t - lag] - m) for t in range(lag, n)) / den


def effective_n(x: Sequence[float], y: Sequence[float]) -> float:
    """Bartlett-type effective sample size for a correlation of two AR(1)-like series."""
    r1x, r1y = autocorrelation(x), autocorrelation(y)
    return max(3.0, len(x) * (1 - r1x * r1y) / (1 + r1x * r1y))


def fisher_ci(r: float, n: float, level: float = 0.95) -> tuple[float, float]:
    """Fisher-z interval for a correlation with (effective) sample size n."""
    if n <= 3:
        return -1.0, 1.0
    z = math.atanh(max(min(r, 0.999999), -0.999999))
    q = normal_quantile(0.5 + level / 2) / math.sqrt(n - 3)
    return math.tanh(z - q), math.tanh(z + q)


def wooldridge_serial(fd_resid: Mapping[tuple[U, int], float], units: Sequence[U],
                      years: Sequence[int]) -> dict[str, float]:
    """Wooldridge (2002) test: regress e_it on e_i,t-1 of first-difference residuals.

    With serially uncorrelated errors in levels the slope is -0.5; a slope near 0
    points to a random walk in levels (first differences preferred), near -0.5 to
    fixed effects being adequate.
    """
    x, y, cl = [], [], []
    for u in units:
        for t0, t1 in zip(years, years[1:], strict=False):
            if (u, t0) in fd_resid and (u, t1) in fd_resid:
                x.append([fd_resid[(u, t0)]])
                y.append(fd_resid[(u, t1)])
                cl.append(u)
    fit = ols(x, y, cl, ["lag"])
    se = fit.se[0]
    return {"slope": fit.beta[0], "se_cr1": se, "t_vs_minus_half": (fit.beta[0] + 0.5) / se if se else float("inf"),
            "n": float(fit.n)}
