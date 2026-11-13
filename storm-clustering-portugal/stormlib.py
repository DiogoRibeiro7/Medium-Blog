"""Building blocks for storm_clustering.py: data loading, the storm severity index,
the episode catalogue, dispersion statistics, and a Hawkes process with a
covariate-driven background rate.

Everything here is deterministic given the files in data/ and a random seed.
"""

from __future__ import annotations

import gzip
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

DATA = Path(__file__).resolve().parent / "data"

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


def load_teleconnections(data: Path = DATA) -> pd.DataFrame:
    """Monthly CPC teleconnection indices indexed by (year, month). Falls back to the
    NAO-only file if the full table is not on disk. Missing values (-99.9) become NaN."""
    path = data / "cpc_tele_index_nh.txt"
    if path.exists():
        lines = path.read_text(errors="replace").splitlines()
        header = None
        rows = []
        for line in lines:
            tok = line.split()
            if not tok:
                continue
            if header is None and "NAO" in tok and "EA" in tok:
                header = [t for t in tok if t not in ("yyyy", "mm")]
                continue
            if header is not None and len(tok) >= 2 + 5 and tok[0].isdigit() and len(tok[0]) == 4:
                vals = [float(x) for x in tok[2:2 + len(header)]]
                rows.append([int(tok[0]), int(tok[1])] + vals)
        if header and rows:
            df = pd.DataFrame(rows, columns=["year", "month"] + header)
            df = df.replace(-99.9, np.nan).replace(-99.90, np.nan)
            df.columns = [c.replace("/", "_") for c in df.columns]
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


def season_time(t: pd.Timestamp, season: int) -> float:
    """Days since 1 October of the season, at the middle of the day."""
    return (t - pd.Timestamp(season, 10, 1)).days + 0.5


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


def block_counts(ep: pd.DataFrame, seasons: np.ndarray, length: int) -> np.ndarray:
    """Counts of episodes in consecutive non-overlapping blocks of `length` days within
    each season (the incomplete final block of a season is dropped)."""
    out = []
    for s in seasons:
        L = season_length(s)
        nb = L // length
        if nb == 0:
            continue
        t = np.array([season_time(p, s) for p in ep.peak[ep.season == s]])
        c = np.histogram(t, bins=np.arange(0, (nb + 1) * length, length))[0]
        out.extend(c[:nb])
    return np.array(out)


# ----------------------------------------------------------------------------
# Hawkes process with a covariate-driven background (days as the time unit)
# ----------------------------------------------------------------------------

@dataclass
class PPData:
    """Episode times for a set of seasons, stacked for a vectorised likelihood."""
    seasons: np.ndarray           # season labels, in order
    T: np.ndarray                 # season lengths (days)
    X: np.ndarray                 # stacked per-day background design, all seasons
    day_offset: np.ndarray        # first row of each season in X
    times: np.ndarray             # event times (days since 1 Oct), grouped by season
    ev_season: np.ndarray         # position (0..S-1) of each event's season
    ev_row: np.ndarray            # row of X for each event's day

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def n_events(self) -> int:
        return len(self.times)


def background_design(seasons, tele: pd.DataFrame | None, covars: list[str],
                      trend_center: float | None = None, daily_cov: pd.Series | None = None):
    """Per-day design matrix for each season: intercept, annual harmonic, monthly
    teleconnection indices, optionally a daily covariate and a linear trend (per decade)."""
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


def build_data(times_by_season: dict, seasons, design: dict) -> PPData:
    seasons = np.asarray(list(seasons))
    T = np.array([design[s].shape[0] for s in seasons], float)
    X = np.vstack([design[s] for s in seasons])
    off = np.concatenate([[0], np.cumsum(T[:-1])]).astype(int)
    tt, es, er = [], [], []
    for i, s in enumerate(seasons):
        t = np.sort(np.asarray(times_by_season.get(s, []), float))
        tt.append(t)
        es.append(np.full(len(t), i))
        er.append(off[i] + np.minimum(t.astype(int), int(T[i]) - 1))
    return PPData(seasons, T, X, off, np.concatenate(tt) if tt else np.array([]),
                  np.concatenate(es).astype(int), np.concatenate(er).astype(int))


def episode_times(ep: pd.DataFrame) -> dict:
    out = {}
    for s, g in ep.groupby("season"):
        out[int(s)] = [season_time(t, int(s)) for t in g.peak]
    return out


def _excitation_sums(times, ev_season, beta):
    """A_k = sum over earlier events j of the same season of exp(-beta (t_k - t_j)),
    by an exclusive cumulative sum within each season (no cross-season subtraction)."""
    A = np.zeros(len(times))
    if len(times) == 0:
        return A
    cut = np.flatnonzero(np.r_[True, ev_season[1:] != ev_season[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        t = times[a:b]
        E = np.exp(beta * (t - t[0]))
        excl = np.concatenate([[0.0], np.cumsum(E[:-1])])
        A[a:b] = excl * np.exp(-beta * (t - t[0]))
    return A


def _unpack(theta, p, hawkes, trend_n):
    gamma = theta[:p]
    if not hawkes:
        return gamma, -np.inf, 1.0, 0.0
    return gamma, theta[p], math.exp(theta[p + 1]), (theta[p + 2] if trend_n else 0.0)


def branching(a, delta, season, center):
    z = a + delta * (np.asarray(season, float) - center) / 10.0
    return 1.0 / (1.0 + np.exp(-z))


def negloglik(theta, d: PPData, hawkes: bool, trend_n: bool = False, center: float = 1988.0) -> float:
    p = d.p
    gamma, a, beta, delta = _unpack(theta, p, hawkes, trend_n)
    eta = d.X @ gamma
    if np.any(eta > 50):
        return 1e12
    mu = np.exp(eta)
    comp = mu.sum()
    lam = mu[d.ev_row]
    if hawkes:
        nb_s = branching(a, delta, d.seasons, center)
        nb = nb_s[d.ev_season]
        A = _excitation_sums(d.times, d.ev_season, beta)
        lam = lam + nb * beta * A
        comp += np.sum(nb * (1.0 - np.exp(-beta * (d.T[d.ev_season] - d.times))))
    return float(comp - np.sum(np.log(lam)))


@dataclass
class Fit:
    theta: np.ndarray
    nll: float
    p: int
    hawkes: bool
    trend_n: bool
    center: float
    cov: np.ndarray | None

    @property
    def k(self) -> int:
        return len(self.theta)

    @property
    def aic(self) -> float:
        return 2 * self.nll + 2 * self.k

    def branching_ratio(self, season: float | None = None) -> float:
        if not self.hawkes:
            return 0.0
        g, a, beta, delta = _unpack(self.theta, self.p, True, self.trend_n)
        return float(branching(a, delta, self.center if season is None else season, self.center))

    def timescale(self) -> float:
        return math.exp(-self.theta[self.p + 1]) if self.hawkes else float("nan")

    def se(self) -> np.ndarray:
        if self.cov is None:
            return np.full(self.k, np.nan)
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))


def fit(d: PPData, hawkes: bool, trend_n: bool = False, center: float = 1988.0,
        hessian: bool = True, quick: bool = False) -> Fit:
    p = d.p
    start = np.zeros(p)
    start[0] = math.log(max(d.n_events, 1) / d.T.sum())
    bounds = [(-20, 20)] * p
    starts = [start]
    if hawkes:
        bounds += [(-9, 4), (math.log(1 / 60.0), math.log(2.0))] + ([(-6, 6)] if trend_n else [])
        grid = ((-1.0, math.log(1 / 3.0)), (-2.5, math.log(1 / 1.5)), (0.0, math.log(1 / 8.0)), (-1.0, math.log(1 / 20.0)))
        if quick:
            grid = grid[:2]
        starts = [np.concatenate([start, [a0, b0] + ([0.0] if trend_n else [])]) for a0, b0 in grid]
    best = None
    for s0 in starts:
        r = optimize.minimize(negloglik, s0, args=(d, hawkes, trend_n, center),
                              method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    cov = None
    if hessian:
        H = numerical_hessian(lambda th: negloglik(th, d, hawkes, trend_n, center), best.x)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = None
    return Fit(best.x, float(best.fun), p, hawkes, trend_n, center, cov)


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


def rescaled_intervals(f: Fit, d: PPData) -> np.ndarray:
    """Time-rescaled intervals between successive episodes within each season (the first
    measured from 1 October); Exp(1) and independent if the model is right."""
    gamma, a, beta, delta = _unpack(f.theta, f.p, f.hawkes, f.trend_n)
    mu = np.exp(d.X @ gamma)
    out = []
    for i, s in enumerate(d.seasons):
        m = mu[d.day_offset[i]:d.day_offset[i] + int(d.T[i])]
        cm = np.concatenate([[0.0], np.cumsum(m)])
        t = d.times[d.ev_season == i]
        day = np.minimum(np.floor(t).astype(int), len(m) - 1)
        L = cm[day] + m[day] * (t - day)
        if f.hawkes:
            nb = float(branching(a, delta, s, f.center))
            L = L + np.array([nb * np.sum(1.0 - np.exp(-beta * (tk - t[:k]))) for k, tk in enumerate(t)])
        out.extend(np.diff(np.concatenate([[0.0], L])))
    return np.array(out)


def intensity_path(f: Fit, d: PPData, season: int, step: float = 0.1):
    """Background and total conditional intensity through one season."""
    gamma, a, beta, delta = _unpack(f.theta, f.p, f.hawkes, f.trend_n)
    i = int(np.where(d.seasons == season)[0][0])
    m = np.exp(d.X[d.day_offset[i]:d.day_offset[i] + int(d.T[i])] @ gamma)
    grid = np.arange(0, d.T[i], step)
    bg = m[np.minimum(grid.astype(int), len(m) - 1)]
    t = d.times[d.ev_season == i]
    tot = bg.copy()
    if f.hawkes:
        nb = float(branching(a, delta, season, f.center))
        for tk in t:
            after = grid > tk
            tot[after] += nb * beta * np.exp(-beta * (grid[after] - tk))
    return grid, bg, tot


def simulate_season(f: Fit, X: np.ndarray, season: int, rng: np.random.Generator) -> np.ndarray:
    """Event times for one season by the cluster (branching) representation."""
    gamma, a, beta, delta = _unpack(f.theta, f.p, f.hawkes, f.trend_n)
    mu = np.exp(X @ gamma)
    T = len(mu)
    counts = rng.poisson(mu)
    imm = (np.repeat(np.arange(T), counts) + rng.random(counts.sum())).astype(float)
    if not f.hawkes:
        return np.sort(imm)
    nb = float(branching(a, delta, season, f.center))
    events = [imm]
    frontier = imm
    while len(frontier):
        k = rng.poisson(nb, len(frontier))
        if k.sum() == 0:
            break
        kids = np.repeat(frontier, k) + rng.exponential(1.0 / beta, k.sum())
        kids = kids[kids < T]
        events.append(kids)
        frontier = kids
    return np.sort(np.concatenate(events))


def to_day_events(times: np.ndarray) -> np.ndarray:
    """Simulated continuous times on the daily grid the catalogue uses: at most one
    episode peak per day, and episodes on consecutive days merge (as in the catalogue,
    where an episode is a run of storm days)."""
    if len(times) == 0:
        return times
    d = np.unique(np.floor(times).astype(int))
    keep = [d[0]]
    for x in d[1:]:
        if x - keep[-1] > 1:
            keep.append(x)
    return np.array(keep) + 0.5


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test: (Kendall tau against time, two-sided p-value)."""
    t = np.arange(len(x))
    tau, p = stats.kendalltau(t, x)
    return float(tau), float(p)
