"""Building blocks for storm_clustering.py: data loading, the storm severity index,
the episode catalogue, dispersion statistics, and a Hawkes process with a
covariate-driven background rate.

Everything here is deterministic given the files in data/ and a random seed.
"""

from __future__ import annotations

import gzip
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

DATA = Path(os.environ.get("STORM_DATA", Path(__file__).resolve().parent / "data"))

WINTER_MONTHS = (10, 11, 12, 1, 2, 3)
REF_PERIOD = ("1981-01-01", "2010-12-31")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def season_of(dates: pd.DatetimeIndex) -> np.ndarray:
    """Winter season labelled by the year it starts: Oct 2025 - Mar 2026 is 2025."""
    return np.where(dates.month >= 10, dates.year, dates.year - 1)


def load_era5(data: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily maximum 10 m wind at the ERA5 nodes: (wide frame date x node, node table)."""
    pts = pd.read_csv(data / "era5_points.csv")
    cols = {}
    for _, r in pts.iterrows():
        f = data / "era5" / f"era5_{r.lat:.2f}_{r.lon:.2f}.csv.gz"
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["wind_max_ms"]
        cols[f"{r.lat:.2f}_{r.lon:.2f}"] = s
    wide = pd.DataFrame(cols).sort_index()
    pts["node"] = [f"{a:.2f}_{b:.2f}" for a, b in zip(pts.lat, pts.lon)]
    return wide, pts


def load_stations(data: Path = DATA, min_syn: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily maximum wind at the eight synoptic times, kept only on days with at least
    `min_syn` of the eight reports: (wide frame date x station, station table)."""
    d = pd.read_csv(data / "daily_wind.csv.gz", parse_dates=["date"])
    d = d[d.n_syn_wind >= min_syn]
    wide = d.pivot(index="date", columns="station", values="wind_max_syn_ms").sort_index()
    st = pd.read_csv(data / "stations.csv", dtype={"wmo": str})
    return wide, st


TELE_NAMES = ["NAO", "EA", "WP", "EP_NP", "PNA", "EA_WR", "SCA", "TNH", "POL", "PT"]


def load_teleconnections(data: Path = DATA) -> pd.DataFrame:
    """Monthly CPC teleconnection indices indexed by (year, month). The file is fixed
    width and adjacent fields can touch ("-0.71-99.90"), so numbers are read with a
    regular expression. -99.9 (pattern not a leading mode that month) becomes NaN.
    Falls back to the NAO-only file if the table is not on disk."""
    import re
    path = data / "cpc_tele_index_nh.txt"
    num = re.compile(r"-?\d+\.\d+")
    if path.exists():
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*(\d{4})\s+(\d{1,2})\s(.*)$", line)
            if not m:
                continue
            vals = [float(x) for x in num.findall(m.group(3))]
            if len(vals) < len(TELE_NAMES):
                continue
            rows.append([int(m.group(1)), int(m.group(2))] + vals[:len(TELE_NAMES)])
        if rows:
            df = pd.DataFrame(rows, columns=["year", "month"] + TELE_NAMES)
            df = df.mask(df <= -99.0)
            return df.set_index(["year", "month"])
    df = pd.read_csv(data / "nao_monthly.csv")
    return df.rename(columns={"nao": "NAO"}).set_index(["year", "month"])


def load_nao_daily(data: Path = DATA) -> pd.Series | None:
    path = data / "cpc_nao_daily.txt"
    if not path.exists():
        return None
    txt = path.read_text(errors="replace")
    body = "\n".join(l for l in txt.splitlines() if not l.startswith("#"))
    if "," in body.splitlines()[0]:
        df = pd.read_csv(io.StringIO(body))
        cols = {c.lower(): c for c in df.columns}
        y, m, d = cols.get("year"), cols.get("month"), cols.get("day")
        val = [c for c in df.columns if c.lower() not in ("year", "month", "day")][0]
        idx = pd.to_datetime(dict(year=df[y], month=df[m], day=df[d]))
        return pd.Series(df[val].values, index=idx).replace(-99.9, np.nan).sort_index()
    rows = []
    for line in body.splitlines():
        tok = line.split()
        if len(tok) == 4:
            try:
                rows.append((pd.Timestamp(int(tok[0]), int(tok[1]), int(tok[2])), float(tok[3])))
            except ValueError:
                continue
    return pd.Series(dict(rows)).sort_index()


# ----------------------------------------------------------------------------
# Storm severity index and episodes
# ----------------------------------------------------------------------------

def thresholds(wide: pd.DataFrame, q: float = 0.98, period=REF_PERIOD) -> pd.Series:
    ref = wide.loc[period[0]:period[1]]
    return ref.quantile(q)


def severity_index(wide: pd.DataFrame, v98: pd.Series, min_frac: float = 0.6) -> pd.DataFrame:
    """Daily storm severity index: the mean over locations of the cubed relative excess
    of the daily maximum wind over its local 98th percentile (Klawa and Ulbrich 2003),
    times 1000; and the share of locations above their 98th percentile. Days on which
    fewer than `min_frac` of the locations report are NaN."""
    rel = wide.div(v98, axis=1) - 1.0
    exc = rel.clip(lower=0) ** 3
    avail = wide.notna().mean(axis=1)
    ssi = exc.mean(axis=1) * 1000
    area = (rel > 0).sum(axis=1) / wide.notna().sum(axis=1)
    out = pd.DataFrame({"ssi": ssi, "area": area, "avail": avail})
    out.loc[avail < min_frac, ["ssi", "area"]] = np.nan
    return out


def winter_days(index: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(index.month, WINTER_MONTHS)


@dataclass
class Catalogue:
    episodes: pd.DataFrame        # one row per episode
    days: pd.DataFrame            # the daily index with storm-day flags
    threshold: float
    rate_days: float              # storm days per season
    seasons: np.ndarray           # seasons with enough data (others are dropped)


def episodes(daily: pd.DataFrame, q: float = 0.96, gap: int = 0,
             first_season: int | None = None, last_season: int | None = None,
             min_cov: float = 0.9, threshold: float | None = None) -> Catalogue:
    """Storm days are winter days (Oct-Mar) whose index exceeds the q-quantile of the
    index over all winter days (or a given threshold). Storm days separated by `gap` or
    fewer calm days form one episode, dated by its peak day; its severity is the summed
    index. Seasons in which fewer than `min_cov` of the days have an index are dropped."""
    d = daily.copy()
    d = d[winter_days(d.index)]
    d["season"] = season_of(d.index)
    if first_season is not None:
        d = d[d.season >= first_season]
    if last_season is not None:
        d = d[d.season <= last_season]
    cov = d.groupby("season").ssi.apply(lambda x: x.notna().sum() / season_length(int(x.name)))
    good = cov.index[cov >= min_cov].values
    d = d[d.season.isin(good)]
    u = float(np.nanquantile(d.ssi, q)) if threshold is None else float(threshold)
    d["storm"] = d.ssi > u
    rows = []
    cur = None
    calm = 0
    for t, r in d.iterrows():
        if r.storm:
            if cur is not None and calm <= gap and cur["season"] == r.season:
                cur["days"].append((t, r.ssi, r.area))
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {"season": r.season, "days": [(t, r.ssi, r.area)]}
            calm = 0
        else:
            calm += 1
            if cur is not None and calm > gap:
                rows.append(cur)
                cur = None
    if cur is not None:
        rows.append(cur)
    out = []
    for e in rows:
        ts, ss, aa = zip(*e["days"])
        k = int(np.argmax(ss))
        out.append({"season": e["season"], "start": ts[0], "end": ts[-1], "peak": ts[k],
                    "n_days": len(ts), "peak_ssi": ss[k], "severity": float(np.sum(ss)),
                    "peak_area": aa[k]})
    ep = pd.DataFrame(out)
    seasons = np.unique(d.season)
    rate = d.storm.sum() / len(seasons)
    return Catalogue(ep, d, u, rate, seasons)


def season_bounds(season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(season, 10, 1), pd.Timestamp(season + 1, 3, 31)


def season_length(season: int) -> int:
    a, b = season_bounds(season)
    return (b - a).days + 1


# ----------------------------------------------------------------------------
# Dispersion
# ----------------------------------------------------------------------------

def dispersion(counts: np.ndarray) -> float:
    c = np.asarray(counts, float)
    return c.var(ddof=1) / c.mean() - 1.0


def dispersion_test(counts: np.ndarray) -> float:
    """Two-sided p-value of the Poisson index-of-dispersion test."""
    c = np.asarray(counts, float)
    n = len(c)
    stat = (n - 1) * c.var(ddof=1) / c.mean()
    p_hi = stats.chi2.sf(stat, n - 1)
    return float(min(1.0, 2 * min(p_hi, 1 - p_hi)))


# ----------------------------------------------------------------------------
# A self-exciting model on the daily grid of the catalogue
# ----------------------------------------------------------------------------
#
# The events are episode START days. On each "live" day of a winter an episode starts
# with probability 1 - exp(-lambda_d), where
#
#     lambda_d = Z_s * mu_d + n * (1 - e^-b) * sum_{j < d} e^{-b (d - j - 1)} y_j
#
# mu_d is the background (season and circulation), Z_s an unobserved multiplier for
# winter s with a Gamma(k, 1/k) distribution (the "stormy winter" frailty, optional),
# y_j = 1 if an episode started on day j, and the excitation kernel puts total mass n
# (the branching ratio) on lags of 1, 2, ... days with time scale 1/b. Days on which a
# new episode cannot start, because one is under way or because the catalogue forces a
# calm day after it, are "dead" and drop out of the likelihood. Without the dead time a
# kernel that peaks at lag zero would predict follow-on storms exactly where the
# catalogue cannot contain them, and the fit would understate the excitation.


@dataclass
class DayData:
    seasons: np.ndarray            # (S,) season labels
    X: np.ndarray                  # (D, p) background design, all seasons stacked
    y: np.ndarray                  # (D,) 1 on episode start days
    live: np.ndarray               # (D,) True where an episode could start
    day_offset: np.ndarray         # (S,) first row of each season
    T: np.ndarray                  # (S,) days per season

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def season_of_day(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.seasons)), self.T.astype(int))

    @property
    def n_events(self) -> int:
        return int(self.y.sum())


def background_design(seasons, tele: pd.DataFrame | None, covars: list[str],
                      trend_center: float | None = None, daily_cov: pd.Series | None = None):
    """Per-day design matrix for each season: intercept, annual harmonic, monthly
    teleconnection indices, optionally a daily covariate (aligned by date) and a linear
    trend (per decade from trend_center)."""
    out = {}
    for s in seasons:
        a, b = season_bounds(s)
        days = pd.date_range(a, b, freq="D")
        doy = days.dayofyear.values
        cols = [np.ones(len(days)), np.cos(2 * np.pi * doy / 365.25), np.sin(2 * np.pi * doy / 365.25)]
        key = list(zip(days.year, days.month))
        for c in covars:
            cols.append(np.nan_to_num(tele[c].reindex(key).values))
        if daily_cov is not None:
            cols.append(np.nan_to_num(daily_cov.reindex(days).values))
        if trend_center is not None:
            cols.append(np.full(len(days), (s - trend_center) / 10.0))
        out[s] = np.column_stack(cols)
    return out


def season_arrays(ep: pd.DataFrame, s: int, gap: int):
    """Start indicator and live mask for one season from an episode table."""
    T = season_length(s)
    y = np.zeros(T, dtype=np.int8)
    live = np.ones(T, dtype=bool)
    for _, r in ep[ep.season == s].iterrows():
        i0 = (r.start - pd.Timestamp(s, 10, 1)).days
        i1 = (r.end - pd.Timestamp(s, 10, 1)).days
        y[i0] = 1
        live[i0 + 1:min(T, i1 + gap + 2)] = False
    return y, live


def build_days(ep: pd.DataFrame, seasons, design: dict, gap: int) -> DayData:
    seasons = np.asarray(list(seasons))
    ys, lives = zip(*(season_arrays(ep, int(s), gap) for s in seasons))
    T = np.array([len(v) for v in ys], float)
    off = np.concatenate([[0], np.cumsum(T[:-1])]).astype(int)
    return DayData(seasons, np.vstack([design[s] for s in seasons]), np.concatenate(ys),
                   np.concatenate(lives), off, T)


def from_arrays(seasons, design, ys, lives) -> DayData:
    seasons = np.asarray(list(seasons))
    T = np.array([len(v) for v in ys], float)
    off = np.concatenate([[0], np.cumsum(T[:-1])]).astype(int)
    return DayData(seasons, np.vstack([design[s] for s in seasons]), np.concatenate(ys).astype(np.int8),
                   np.concatenate(lives).astype(bool), off, T)


def _excitation(d: DayData, beta: float) -> np.ndarray:
    """K_d = (1 - e^-b) * sum_{j<d, same season} e^{-b (d - j - 1)} y_j (sums to 1 per event)."""
    from scipy.signal import lfilter
    q = math.exp(-beta)
    out = np.empty(len(d.y))
    for i in range(len(d.seasons)):
        a0, a1 = d.day_offset[i], d.day_offset[i] + int(d.T[i])
        yy = d.y[a0:a1].astype(float)
        # K_d = q K_{d-1} + (1-q) y_{d-1}
        out[a0:a1] = lfilter([0.0, 1.0 - q], [1.0, -q], yy)
    return out


def _unpack(theta, p, hawkes, trend_n, frailty=False):
    gamma = theta[:p]
    i = p
    if hawkes:
        a, beta = theta[i], math.exp(theta[i + 1])
        i += 2
        delta = theta[i] if trend_n else 0.0
        i += 1 if trend_n else 0
    else:
        a, beta, delta = -np.inf, 1.0, 0.0
    shape = math.exp(theta[i]) if frailty else None
    return gamma, a, beta, delta, shape


def branching(a, delta, season, center):
    z = a + delta * (np.asarray(season, float) - center) / 10.0
    return 1.0 / (1.0 + np.exp(-z))


N_QUAD = 24
N_MAX = 0.9


def gamma_quadrature(shape: float, n: int = N_QUAD):
    """Nodes and normalised weights for E[f(Z)], Z ~ Gamma(shape, 1/shape), by generalised
    Gauss-Laguerre quadrature computed with the Golub-Welsch eigenvalue method, so the
    weights stay finite for large shapes (scipy's unnormalised weights overflow)."""
    from scipy.linalg import eigh_tridiagonal
    alpha = shape - 1.0
    i = np.arange(n)
    diag = 2 * i + alpha + 1
    off = np.sqrt(np.arange(1, n) * (np.arange(1, n) + alpha))
    u, V = eigh_tridiagonal(diag, off)
    w = V[0, :] ** 2
    return u / shape, w / w.sum()   # branching ratios above this are treated as implausible (near-explosive)


def _day_ll(lam, y):
    """Bernoulli log-likelihood with complementary log-log link, per day."""
    return np.where(y == 1, np.log(-np.expm1(-lam)), -lam)


def negloglik(theta, d: DayData, hawkes: bool, trend_n: bool = False, center: float = 1988.0,
              frailty: bool = False, _cache: dict | None = None) -> float:
    p = d.p
    gamma, a, beta, delta, shape = _unpack(theta, p, hawkes, trend_n, frailty)
    eta = d.X @ gamma
    if np.any(eta > 5):
        return 1e12
    mu = np.exp(eta)
    exc = np.zeros(len(mu))
    if hawkes:
        nb_s = branching(a, delta, d.seasons, center)
        if np.any(nb_s > N_MAX):
            return 1e12
        exc = np.repeat(nb_s, d.T.astype(int)) * _excitation(d, beta)
    live = d.live
    if not frailty:
        lam = mu + exc
        return float(-np.sum(_day_ll(lam[live], d.y[live])))
    z, w = gamma_quadrature(shape)
    sod = d.season_of_day[live]
    S = len(d.seasons)
    m, e, y = mu[live], exc[live], d.y[live]
    ll = np.empty((S, len(z)))
    for j in range(len(z)):
        ll[:, j] = np.bincount(sod, weights=_day_ll(z[j] * m + e, y), minlength=S)
    with np.errstate(divide="ignore"):
        lw = np.log(w)
    tot = special.logsumexp(ll + lw[None, :], axis=1)
    return float(-tot.sum())


@dataclass
class Fit:
    theta: np.ndarray
    nll: float
    p: int
    hawkes: bool
    trend_n: bool
    center: float
    cov: np.ndarray | None
    frailty: bool = False

    @property
    def k(self) -> int:
        return len(self.theta)

    @property
    def aic(self) -> float:
        return 2 * self.nll + 2 * self.k

    def parts(self):
        return _unpack(self.theta, self.p, self.hawkes, self.trend_n, self.frailty)

    def branching_ratio(self, season: float | None = None) -> float:
        if not self.hawkes:
            return 0.0
        g, a, beta, delta, _ = self.parts()
        return float(branching(a, delta, self.center if season is None else season, self.center))

    def timescale(self) -> float:
        return math.exp(-self.theta[self.p + 1]) if self.hawkes else float("nan")

    def frailty_shape(self) -> float:
        return self.parts()[4] if self.frailty else float("inf")

    def se(self) -> np.ndarray:
        if self.cov is None:
            return np.full(self.k, np.nan)
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))


def fit(d: DayData, hawkes: bool, trend_n: bool = False, center: float = 1988.0,
        hessian: bool = True, quick: bool = False, frailty: bool = False) -> Fit:
    p = d.p
    start = np.zeros(p)
    start[0] = math.log(max(d.n_events, 1) / max(d.live.sum(), 1))
    bounds = [(-20, 20)] * p
    starts = [start]
    if hawkes:
        bounds += [(-9, 2), (math.log(1 / 60.0), math.log(2.0))] + ([(-1.5, 1.5)] if trend_n else [])
        grid = ((-2.0, math.log(1 / 5.0)), (-1.0, math.log(1 / 2.0)), (-2.0, math.log(1 / 15.0)), (-3.0, math.log(1 / 40.0)))
        if quick:
            grid = grid[:2]
        starts = [np.concatenate([start, [a0, b0] + ([0.0] if trend_n else [])]) for a0, b0 in grid]
    if frailty:
        base = fit(d, hawkes, trend_n, center, hessian=False, quick=quick)
        bounds += [(math.log(0.5), math.log(2000.0))]
        starts = [np.concatenate([base.theta, [math.log(k0)]]) for k0 in (10.0, 100.0)]
    best = None
    for s0 in starts:
        r = optimize.minimize(negloglik, s0, args=(d, hawkes, trend_n, center, frailty),
                              method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    # Round the estimates so that last-bit differences between machines (BLAS kernels)
    # cannot change the random draws of the simulations that use them.
    x = np.round(best.x, 6)
    best_fun = negloglik(x, d, hawkes, trend_n, center, frailty)
    cov = None
    if hessian:
        H = numerical_hessian(lambda th: negloglik(th, d, hawkes, trend_n, center, frailty), x)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = None
    return Fit(x, float(best_fun), p, hawkes, trend_n, center, cov, frailty)


def numerical_hessian(f, x, eps=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    fx = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = eps
            ej = np.zeros(k); ej[j] = eps
            if i == j:
                H[i, i] = (f(x + ei) - 2 * fx + f(x - ei)) / eps ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
    return H


def posterior_frailty(f: Fit, d: DayData) -> np.ndarray:
    """Posterior mean of each winter's multiplier Z (ones without a frailty)."""
    if not f.frailty:
        return np.ones(len(d.seasons))
    gamma, a, beta, delta, shape = f.parts()
    mu = np.exp(d.X @ gamma)
    exc = np.zeros(len(mu))
    if f.hawkes:
        exc = np.repeat(branching(a, delta, d.seasons, f.center), d.T.astype(int)) * _excitation(d, beta)
    live = d.live
    z, w = gamma_quadrature(shape)
    sod = d.season_of_day[live]
    S = len(d.seasons)
    ll = np.empty((S, len(z)))
    for j in range(len(z)):
        ll[:, j] = np.bincount(sod, weights=_day_ll(z[j] * mu[live] + exc[live], d.y[live]), minlength=S)
    with np.errstate(divide="ignore"):
        lp = ll + np.log(w)[None, :]
    lp -= special.logsumexp(lp, axis=1, keepdims=True)
    return np.exp(lp) @ z


def intensity_path(f: Fit, d: DayData, season: int, z: float = 1.0):
    """Daily background (times the winter's multiplier) and total rate through one season."""
    gamma, a, beta, delta, _ = f.parts()
    i = int(np.where(d.seasons == season)[0][0])
    a0, a1 = d.day_offset[i], d.day_offset[i] + int(d.T[i])
    bg = z * np.exp(d.X[a0:a1] @ gamma)
    tot = bg.copy()
    if f.hawkes:
        tot = tot + float(branching(a, delta, season, f.center)) * _excitation(d, beta)[a0:a1]
    return bg, tot, d.y[a0:a1], d.live[a0:a1]


def simulate(f: Fit, X: np.ndarray, season: int, R: int, rng: np.random.Generator,
             pool_len: np.ndarray, pool_sev: np.ndarray, gap: int):
    """R replicate winters on the daily grid. Each new episode draws its length and
    severity jointly from the observed catalogue, which sets its dead time.
    Returns (starts, severity, live): (R, T) arrays."""
    gamma, a, beta, delta, shape = f.parts()
    mu = np.exp(X @ gamma)
    T = len(mu)
    Z = rng.gamma(shape, 1.0 / shape, R) if shape is not None else np.ones(R)
    nb = float(branching(a, delta, season, f.center)) if f.hawkes else 0.0
    q = math.exp(-beta) if f.hawkes else 0.0
    K = np.zeros(R)
    dead_until = np.full(R, -1)
    starts = np.zeros((R, T), dtype=bool)
    sev = np.zeros((R, T))
    live = np.zeros((R, T), dtype=bool)
    for dday in range(T):
        lv = dday > dead_until
        live[:, dday] = lv
        lam = Z * mu[dday] + nb * K
        y = lv & (rng.random(R) < -np.expm1(-lam))
        idx = np.flatnonzero(y)
        if len(idx):
            k = rng.integers(0, len(pool_len), len(idx))
            starts[idx, dday] = True
            sev[idx, dday] = pool_sev[k]
            dead_until[idx] = dday + pool_len[k] - 1 + gap + 1
        K = q * K + (1 - q) * y
    return starts, sev, live


def fit_continuous(ep: pd.DataFrame, seasons, design: dict, center_day: str = "peak"):
    """For comparison only: the textbook continuous-time Hawkes process on episode peak
    times (middle of the peak day), with an exponential kernel that starts at lag zero and
    no dead time. Returns (Poisson log-lik, Hawkes log-lik, branching ratio, time scale)."""
    seasons = np.asarray(list(seasons))
    X = np.vstack([design[s] for s in seasons])
    T = np.array([design[s].shape[0] for s in seasons], float)
    off = np.concatenate([[0], np.cumsum(T[:-1])]).astype(int)
    tt, es, er = [], [], []
    for i, s in enumerate(seasons):
        t = np.sort([(x - pd.Timestamp(int(s), 10, 1)).days + 0.5 for x in ep[ep.season == s][center_day]])
        tt.append(np.asarray(t, float))
        es.append(np.full(len(t), i))
        er.append(off[i] + np.asarray(t, float).astype(int))
    t, es, er = np.concatenate(tt), np.concatenate(es).astype(int), np.concatenate(er).astype(int)
    p = X.shape[1]

    def exc_sums(beta):
        A = np.zeros(len(t))
        cut = np.flatnonzero(np.r_[True, es[1:] != es[:-1], True])
        for a0, a1 in zip(cut[:-1], cut[1:]):
            tk = t[a0:a1]
            E = np.exp(beta * (tk - tk[0]))
            A[a0:a1] = np.concatenate([[0.0], np.cumsum(E[:-1])]) * np.exp(-beta * (tk - tk[0]))
        return A

    def nll(th, hawkes):
        mu = np.exp(X @ th[:p])
        lam = mu[er]
        comp = mu.sum()
        if hawkes:
            n, beta = 1 / (1 + math.exp(-th[p])), math.exp(th[p + 1])
            lam = lam + n * beta * exc_sums(beta)
            comp += n * np.sum(1 - np.exp(-beta * (T[es] - t)))
        return comp - np.sum(np.log(lam))

    x0 = np.zeros(p)
    x0[0] = math.log(len(t) / T.sum())
    r0 = optimize.minimize(nll, x0, args=(False,), method="L-BFGS-B")
    best = None
    for a0, b0 in ((-1.0, math.log(1 / 3.0)), (-2.5, math.log(1 / 1.5)), (0.0, math.log(1 / 8.0)), (-1.0, math.log(1 / 20.0))):
        r = optimize.minimize(nll, np.r_[r0.x, a0, b0], args=(True,), method="L-BFGS-B",
                              bounds=[(-20, 20)] * p + [(-9, 4), (math.log(1 / 60.0), math.log(2.0))])
        if best is None or r.fun < best.fun:
            best = r
    x = np.round(best.x, 6)
    return (-float(r0.fun), -float(nll(x, True)), 1 / (1 + math.exp(-x[p])), math.exp(-x[p + 1]))


def window_max(arr: np.ndarray, w: int) -> np.ndarray:
    """Largest sum over any w consecutive days, row by row."""
    c = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(arr, axis=1)], axis=1)
    if arr.shape[1] <= w:
        return c[:, -1]
    return np.max(c[:, w:] - c[:, :-w], axis=1)


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test: (Kendall tau against time, two-sided p-value)."""
    t = np.arange(len(x))
    tau, p = stats.kendalltau(t, x)
    return float(tau), float(p)
