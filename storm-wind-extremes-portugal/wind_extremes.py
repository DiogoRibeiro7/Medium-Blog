"""How extreme were storm Kristin's winds, is the tail of Portugal's storm winds getting
heavier, and do the country's wind design values sit where the tail says they should?

Reads ERA5 and station data already committed in ../storm-clustering-portugal/data and
the ERA5 gusts and station-cell series that fetch.py adds to ./data. Prints every number
quoted in the article and draws the figures and equation images.

    python wind_extremes.py > results/output.txt

Set WIND_QUICK=1 for a fast smoke test with few permutations and bootstrap samples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import evtlib as ev  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SIB = ROOT.parent / "storm-clustering-portugal" / "data"
FIGURES = ROOT / "figures"
EQUATIONS = FIGURES / "equations"

SEED = 20260128
FIRST, LAST = 1950, 2025        # seasons, labelled by the October they start in
KRISTIN = pd.Timestamp("2026-01-28")
N_BOOT = 1000                   # season-block bootstrap samples
N_PERM = 999                    # permutations of the season order
N_POWER = 200                   # simulated histories per trend size
N_SEL = 2000                    # simulated records in the selection-bias check
if os.environ.get("WIND_QUICK"):
    N_BOOT, N_PERM, N_POWER, N_SEL = 40, 39, 8, 100

# Eurocode 1 part 1-4, Portuguese National Annex (NP EN 1991-1-4): basic wind velocity,
# 10-minute mean at 10 m over terrain category II with an annual exceedance probability
# of 0.02, and the annex's conversion to other return periods.
VB0 = {"A": 27.0, "B": 30.0}
I_V10 = 1 / np.log(10 / 0.05)          # turbulence intensity at 10 m, terrain II (k_I = 1, c_o = 1)
PEAK = np.sqrt(1 + 7 * I_V10)          # peak velocity / mean velocity at 10 m, terrain II
# Overhead lines: EN 50341-3-17 as reproduced in two Porto theses (10-minute mean 20 and
# 22 m/s, gust factor 1.5, factor 1.1484 below 20 m) and E-REDES D00-C10-001 (34 m/s).
LINE_GUST = {"A": 20 * 1.5 * 1.1484, "B": 22 * 1.5 * 1.1484}
EREDES_GUST = 34.0


def c_prob(T):
    """Ratio of the T-year to the 50-year basic wind velocity in the Portuguese annex."""
    return 0.7 - 0.077 * np.log(-np.log(1 - 1 / np.asarray(T, float)))


# --- chart style (light surface, fixed categorical order) -------------------
SURFACE = "#ffffff"
INK = "#292929"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BAND = "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#dbe9f8", "#a9cbef", "#6fa6e3", "#3b7fd2", "#1d58a8", "#0f3672"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "axes.axisbelow": True,
})
DPI = 160


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def season_label(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


def fmt_T(T: float) -> str:
    if not np.isfinite(T):
        return "inf"
    if T >= 1e5:
        return f">{1e5:.0f}"
    return f"{T:.0f}" if T >= 10 else f"{T:.1f}"


def season_of(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index.month >= 10, index.year, index.year - 1)


WINTER = [10, 11, 12, 1, 2, 3]


# =============================================================================
# 1. Data
# =============================================================================

def load_nodes():
    pts = pd.read_csv(SIB / "era5_points.csv")
    cols = {}
    for _, p in pts.iterrows():
        x = pd.read_csv(SIB / "era5" / f"era5_{p.lat:.2f}_{p.lon:.2f}.csv.gz", parse_dates=["date"])
        cols[(round(p.lat_out, 2), round(p.lon_out, 2))] = x.set_index("date").wind_max_ms
    W = pd.DataFrame(cols)
    W.columns = pd.MultiIndex.from_tuples(W.columns, names=["lat", "lon"])
    G = None
    if (DATA / "era5_gust").exists():
        gcols = {}
        for _, p in pts.iterrows():
            f = DATA / "era5_gust" / f"gust_{p.lat:.2f}_{p.lon:.2f}.csv.gz"
            if f.exists():
                g = pd.read_csv(f, parse_dates=["date"]).set_index("date").gust_max_ms
                gcols[(round(p.lat_out, 2), round(p.lon_out, 2))] = g
        if len(gcols) == len(pts):
            G = pd.DataFrame(gcols)[W.columns]
    return W, G, pts


def season_maxima(D: pd.DataFrame):
    win = D[D.index.month.isin(WINTER)]
    s = season_of(win.index)
    M = win.groupby(s).max().loc[FIRST:LAST]
    A = win.groupby(s).idxmax().loc[FIRST:LAST]
    return M, A


def load_stations():
    d = pd.read_csv(SIB / "daily_wind.csv.gz", parse_dates=["date"])
    st = pd.read_csv(SIB / "stations.csv").set_index("id")
    d = d[d.date.dt.month.isin(WINTER)].copy()
    d["season"] = season_of(pd.DatetimeIndex(d.date))
    return d, st


STATIONS = {  # key: (GHCNh id, Eurocode zone, short description)
    "porto": ("POMU0068545", "B", "Porto airport, 4 km from the sea"),
    "faro": ("POMU0068554", "B", "Faro airport, on the coast"),
    "beja": ("POI0000LPBJ", "A", "Beja air base, inland Alentejo"),
    "braganca": ("POMU0068575", "B", "Braganca, 718 m"),
    "portalegre": ("POM00008571", "A", "Portalegre, 590 m"),
    "lisboa_gc": ("POM00008579", "A", "Lisbon, Gago Coutinho (urban)"),
    "vila_real": ("POMU0068566", "A", "Vila Real, 479 m"),
    "monte_real": ("POI0000LPMR", "A", "Monte Real air base"),
    "montijo": ("POI0000LPMT", "A", "Montijo air base, Tagus estuary"),
}


QC_MIN, QC_RATIO = 20.0, 2.2   # station days at or above 20 m/s and 2.2 times the ERA5 cell are dropped
FLAGGED: dict = {}


def era5_site(key):
    f = DATA / "era5_sites" / f"site_{key}.csv.gz"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["date"]).set_index("date").wind_max_ms


def station_maxima(d, sid, min_syn=6, min_frac=0.9):
    """Season maxima of the synoptic-hour wind (and of all reports) at one station.

    Raw GHCNh passes some isolated spikes on calm, high-pressure days (a 31 m/s report at
    Portalegre at 1016 hPa, for instance). Where the ERA5 series of the station's grid cell
    is on disk, a day is dropped when the station reports at least QC_MIN m/s and more than
    QC_RATIO times the reanalysis maximum for that cell and day."""
    x = d[(d.station == sid) & (d.n_syn_wind >= min_syn)]
    key = next((k for k, v in STATIONS.items() if v[0] == sid), None)
    e = era5_site(key) if key else None
    if e is not None:
        ev_day = e.reindex(pd.DatetimeIndex(x.date)).values
        bad = (x.wind_max_syn_ms.values >= QC_MIN) & (x.wind_max_syn_ms.values > QC_RATIO * ev_day)
        bad |= (x.wind_max_ms.values >= QC_MIN) & (x.wind_max_ms.values > QC_RATIO * ev_day)
        FLAGGED[key] = x[bad][["date", "wind_max_syn_ms", "wind_max_ms", "slp_min_hpa"]].assign(era5=ev_day[bad])
        x = x[~bad]
    g = x.groupby("season")
    n = g.size()
    good = n[n >= min_frac * 182].index
    good = good[(good >= FIRST) & (good <= LAST)]
    return g.wind_max_syn_ms.max().loc[good], g.wind_max_ms.max().loc[good], g.n_wind_obs.median().loc[good]


def load_nao():
    nao = pd.read_csv(SIB / "nao_monthly.csv")
    nao = nao[nao.month.isin(WINTER)].copy()
    nao["season"] = np.where(nao.month >= 10, nao.year, nao.year - 1)
    return nao.groupby("season").nao.mean().loc[FIRST:LAST]


def describe_data(W, M, A, G):
    section("1. Data")
    print(f"ERA5 daily maximum 10 m wind at {W.shape[1]} nodes, {W.index.min().date()} to {W.index.max().date()}")
    print(f"Blocks: October-March seasons {season_label(FIRST)} to {season_label(LAST)} ({len(M)} seasons)")
    hyd = W.groupby(season_of(W.index)).max().loc[FIRST:LAST - 1]
    outside = (hyd.values > M.loc[FIRST:LAST - 1].values + 1e-9).mean()
    print(f"Share of October-September maxima that fall in April-September: {100 * outside:.1f}%")
    K = W.loc[KRISTIN]
    same = np.isclose(M.loc[LAST].values, K.values)
    print(f"Kristin = ERA5 daily maximum on {KRISTIN.date()}; it is the 2025/26 season maximum at "
          f"{same.sum()} of {len(K)} nodes")
    prev = M.loc[:LAST - 1]
    rec = (K.values > prev.max().values)
    print(f"Nodes where Kristin beat every earlier season maximum (1950/51 to 2024/25): {rec.sum()}")
    band = (K.index.get_level_values("lat") < 40.0)
    print(f"  all of them south of 40.0 N: {bool(rec[~band].sum() == 0)}; nodes south of 40.0 N: {band.sum()}")
    if G is not None:
        print(f"ERA5 daily maximum 10 m gust at the same nodes: {G.index.min().date()} to {G.index.max().date()}")
    return K, rec


# =============================================================================
# 2. How extreme was Kristin
# =============================================================================

def kristin_return_periods(M, K, rng):
    section("2. Kristin's return period, node by node")
    X = M.values
    J = X.shape[1]
    rows = []
    for j in range(J):
        f_in = ev.fit_gev(X[:, j])
        f_ex = ev.fit_gev(X[:-1, j])
        rows.append(dict(
            node=j, lat=M.columns[j][0], lon=M.columns[j][1], K=K.values[j],
            mu=f_in.mu, sigma=f_in.sigma, xi=f_in.xi,
            T_in=float(ev.return_period(K.values[j], f_in.mu, f_in.sigma, f_in.xi)),
            T_ex=float(ev.return_period(K.values[j], f_ex.mu, f_ex.sigma, f_ex.xi)),
            z50=float(ev.return_level(50, f_in.mu, f_in.sigma, f_in.xi)),
            rank=int((X[:-1, j] >= K.values[j]).sum()) + 1))
    R = pd.DataFrame(rows)
    pin, pex = ev.fit_pooled_shape(X), ev.fit_pooled_shape(X[:-1])
    R["T_pool_in"] = ev.return_period(K.values, pin["mu"], pin["sigma"], pin["xi"])
    R["T_pool_ex"] = ev.return_period(K.values, pex["mu"], pex["sigma"], pex["xi"])
    R["z50_pool"] = ev.return_level(50, pin["mu"], pin["sigma"], pin["xi"])
    R["cv"] = R.sigma / R.mu
    print(f"Separate fits: shape xi ranges {R.xi.min():+.2f} to {R.xi.max():+.2f}, median {R.xi.median():+.2f}")
    # season-block bootstrap of the pooled fit: resample whole seasons, keep all nodes
    boot_xi, boot_T = [], []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X))
        pb = ev.fit_pooled_shape(X[idx], start=pin["theta"])
        boot_xi.append(pb["xi"])
        boot_T.append(ev.return_period(K.values, pb["mu"], pb["sigma"], pb["xi"]))
    boot_xi, boot_T = np.array(boot_xi), np.minimum(np.array(boot_T), 1e12)
    lo, hi = np.percentile(boot_xi, [2.5, 97.5])
    print(f"Pooled shape, all 76 seasons: xi = {pin['xi']:+.3f} (season-bootstrap 95% interval {lo:+.3f} to {hi:+.3f})")
    print(f"Pooled shape without 2025/26: xi = {pex['xi']:+.3f}")
    print(f"Share of bootstrap samples with xi < 0: {100 * (boot_xi < 0).mean():.0f}%")
    R["T_pool_lo"] = np.percentile(boot_T, 2.5, axis=0)
    hi_T = np.percentile(boot_T, 97.5, axis=0)
    R["T_pool_hi"] = np.where(hi_T >= 1e11, np.inf, hi_T)
    R["share_inf"] = (boot_T >= 1e11).mean(axis=0)
    # the building code's assumption: Gumbel (xi = 0) at every node
    gum = [ev.fit_gev(X[:, j], fixed_xi=0.0) for j in range(J)]
    R["T_gumbel"] = [float(ev.return_period(K.values[j], g.mu, g.sigma, 0.0)) for j, g in enumerate(gum)]
    R["z50_gumbel"] = [float(ev.return_level(50, g.mu, g.sigma, 0.0)) for g in gum]
    ll_g = -sum(g.nll for g in gum)
    print(f"Likelihood ratio, pooled shape against Gumbel at every node: {2 * (-pin['nll'] - ll_g):.1f} "
          f"(1 df if the nodes were independent; they are not)")
    print()
    print("Nodes ranked by Kristin's return period (pooled shape, with 2025/26):")
    print("  lat    lon    Kristin  50-yr   rank  T sep(in)  T sep(ex)  T pool(in) [95% boot] (share inf)  T pool(ex)  T Gumbel")
    for _, r in R.sort_values("T_pool_in", ascending=False).head(16).iterrows():
        print(f"  {r.lat:5.2f} {r.lon:6.2f}  {r.K:6.1f}  {r.z50_pool:6.1f}   {int(r['rank']):3d}   {fmt_T(r.T_in):>8}   {fmt_T(r.T_ex):>8}"
              f"   {fmt_T(r.T_pool_in):>8} [{fmt_T(r.T_pool_lo)} to {fmt_T(r.T_pool_hi)}] ({100 * r.share_inf:.0f}%)"
              f"   {fmt_T(r.T_pool_ex):>8}   {fmt_T(r.T_gumbel):>7}")
    for thr in (10, 50, 100, 500):
        print(f"Nodes with pooled return period of at least {thr} seasons: with 2025/26 {(R.T_pool_in >= thr).sum()}, "
              f"without {(R.T_pool_ex >= thr).sum()}, Gumbel {(R.T_gumbel >= thr).sum()}")
    top = R.sort_values("T_pool_in", ascending=False).iloc[0]
    print(f"Highest node: pooled {fmt_T(top.T_pool_in)} against Gumbel {fmt_T(top.T_gumbel)} "
          f"(ratio {top.T_pool_in / top.T_gumbel:.1f})")
    print(f"Nodes north of 40.0 N with Kristin above their median season maximum: "
          f"{((R.lat >= 40.0) & (R.T_pool_in > 2)).sum()} of {(R.lat >= 40.0).sum()}")
    return R, pin, pex, boot_xi


def profile_intervals(M, K, R, nodes):
    section("2b. Profile-likelihood intervals at the most affected nodes")
    out = {}
    for j in nodes:
        x = M.values[:, j]
        Tin, loin, hiin, cin = ev.profile_return_period(x, K.values[j])
        Tex, loex, hiex, cex = ev.profile_return_period(x[:-1], K.values[j])
        zh, zl, zu = ev.profile_return_level(x, 50)
        out[j] = dict(cin=cin, cex=cex)
        print(f"Node {R.lat[j]:.2f} N {R.lon[j]:.2f} E (Kristin {K.values[j]:.1f} m/s; 50-yr level {zh:.1f}, 95% {zl:.1f} to {zu:.1f}):")
        print(f"   with 2025/26: T = {fmt_T(Tin)} (95% profile interval {fmt_T(loin)} to {fmt_T(hiin)})")
        print(f"   without:      T = {fmt_T(Tex)} (95% profile interval {fmt_T(loex)} to {fmt_T(hiex)})")
    return out


def selection_bias(M, R, pin, rng, j):
    """Simulate 76-season records at node j from the pooled fit, keep those whose last
    season is a record beyond the true 50-year level (the kind of event that triggers
    a study), and compare return-period estimates that include or exclude it."""
    section("2c. What including or excluding the trigger does, by simulation")
    mu, sig, xi = pin["mu"][j], pin["sigma"][j], pin["xi"]
    n = M.shape[0]
    z50 = float(ev.return_level(50, mu, sig, xi))
    kept = 0
    ratios_in, ratios_ex, covered_in, covered_ex = [], [], [], []
    tries = 0
    while kept < N_SEL and tries < 400 * N_SEL:
        tries += 1
        x = stats.genextreme.rvs(c=-xi, loc=mu, scale=sig, size=n, random_state=rng)
        if not (x[-1] > x[:-1].max() and x[-1] > z50):
            continue
        kept += 1
        T_true = float(ev.return_period(x[-1], mu, sig, xi))
        f_in, f_ex = ev.fit_gev(x), ev.fit_gev(x[:-1])
        T_in = float(ev.return_period(x[-1], f_in.mu, f_in.sigma, f_in.xi))
        T_ex = float(ev.return_period(x[-1], f_ex.mu, f_ex.sigma, f_ex.xi))
        ratios_in.append(np.log10(T_in / T_true))
        ratios_ex.append(np.log10(min(T_ex, 1e12) / T_true))
    ri, re_ = np.array(ratios_in), np.array(ratios_ex)
    print(f"Node {R.lat[j]:.2f} N {R.lon[j]:.2f} E, pooled parameters; {kept} simulated records kept out of {tries}")
    print(f"  (last season a record and above the true 50-year level of {z50:.1f} m/s)")
    print(f"Median estimated / true return period: including the trigger {10 ** np.median(ri):.2f}, "
          f"excluding it {10 ** np.median(re_):.2f}")
    print(f"Share of records where the estimate is within a factor 2 of the truth: including {100 * np.mean(np.abs(ri) < np.log10(2)):.0f}%, "
          f"excluding {100 * np.mean(np.abs(re_) < np.log10(2)):.0f}%")
    print(f"Share where the estimate exceeds the truth: including {100 * np.mean(ri > 0):.0f}%, excluding {100 * np.mean(re_ > 0):.0f}%")
    print(f"Share where excluding the trigger puts it beyond the fitted upper end point (infinite T): "
          f"{100 * np.mean(re_ >= np.log10(1e12) - np.log10(1e3)):.1f}%")
    return ri, re_


def peaks_over_threshold(W, M, K, R, rng, q=0.98, r=2):
    """Peaks over a high threshold as a check on the shape: every storm above the node's
    98th percentile of winter days, declustered into storms (runs of 2 calm days)."""
    section("2d. Peaks over a threshold: the shape from every storm, not only the season maximum")
    win = W[W.index.month.isin(WINTER)]
    s_of = season_of(win.index)
    keep = (s_of >= FIRST) & (s_of <= LAST)
    win, s_of = win[keep], s_of[keep]
    Y, peaks_season, lam, th_runs, th_int, u_all = [], [], [], [], [], []
    for j, c in enumerate(W.columns):
        x = win[c].values
        u = np.quantile(x, q)
        pk, n_exc, n_cl = ev.runs_decluster(x, u, r=r)
        Y.append(x[pk] - u)
        peaks_season.append(s_of[pk])
        lam.append(n_cl / len(M))
        th_runs.append(n_cl / n_exc)
        th_int.append(ev.intervals_extremal_index(x, u))
        u_all.append(u)
    lam, u_all = np.array(lam), np.array(u_all)
    f = ev.fit_gpd_pooled(Y)
    print(f"Threshold: each node's {100 * q:.0f}th percentile of October-March days ({np.median(u_all):.1f} m/s median); "
          f"storms separated by {r} days below it")
    print(f"Storms per season above the threshold: median {np.median(lam):.2f} over nodes")
    print(f"Extremal index (days above the threshold per storm, inverted): runs {np.median(th_runs):.2f}, "
          f"intervals estimator {np.nanmedian(th_int):.2f} (medians over nodes)")
    # season-block bootstrap of the pooled shape
    seasons = np.arange(FIRST, LAST + 1)
    bx = []
    for b in range(min(N_BOOT, 400)):
        pick = rng.choice(seasons, len(seasons))
        cnt = pd.Series(pick).value_counts()
        Yb = []
        for j in range(len(Y)):
            w = cnt.reindex(peaks_season[j]).fillna(0).astype(int).values
            Yb.append(np.repeat(Y[j], w))
        if min(len(y) for y in Yb) < 5:
            continue
        bx.append(ev.fit_gpd_pooled(Yb)["xi"])
    lo, hi = np.percentile(bx, [2.5, 97.5])
    print(f"Pooled GPD shape: xi = {f['xi']:+.3f} (season-bootstrap 95% interval {lo:+.3f} to {hi:+.3f})")
    T = 1 / (lam * ev.gpd_sf(np.maximum(K.values - u_all, 0), f["sigma"], f["xi"]))
    T = np.where(K.values > u_all, T, np.nan)
    R["T_pot"] = T
    order = R.sort_values("T_pool_in", ascending=False).index[:6]
    for j in order:
        print(f"   {R.lat[j]:.2f} N {R.lon[j]:.2f} E: return period from peaks over threshold {fmt_T(T[j])}, "
              f"from season maxima {fmt_T(R.T_pool_in[j])}")
    return f


def early_record(M, K, R):
    """ERA5 back to 1940, which brings in the cyclone of 15 February 1941."""
    section("2f. The record back to 1940")
    cols = {}
    for c in M.columns:
        f = DATA / "era5_early" / f"early_{c[0]:.2f}_{c[1]:.2f}.csv.gz"
        if not f.exists():
            # the requested lattice coordinate differs from the returned cell for two nodes
            alt = {(37.25, -8.5): (37.0, -8.5), (38.75, -9.0): (38.5, -9.0)}.get(c)
            f = DATA / "era5_early" / f"early_{alt[0]:.2f}_{alt[1]:.2f}.csv.gz" if alt else f
        if not f.exists():
            print("ERA5 before 1950 not on disk; section skipped")
            return None
        cols[c] = pd.read_csv(f, parse_dates=["date"]).set_index("date").wind_max_ms
    E = pd.DataFrame(cols)
    E.columns = M.columns
    Ew = E[E.index.month.isin(WINTER)]
    se = season_of(Ew.index)
    ME = Ew.groupby(se).max().loc[1940:1949]
    AE = Ew.groupby(se).idxmax().loc[1940:1949]
    print(f"Seasons {season_label(1940)} to {season_label(1949)} added ({len(ME)}); ERA5 before 1950 assimilates far fewer observations")
    c41 = E.loc["1941-02-15"]
    hi41, hik = c41.idxmax(), K.idxmax()
    print(f"15 February 1941: highest node {c41.max():.1f} m/s at {hi41[0]:.2f} N {hi41[1]:.2f} E; "
          f"Kristin's highest {K.max():.1f} m/s at {hik[0]:.2f} N {hik[1]:.2f} E")
    above = (c41.values > K.values)
    print(f"Nodes where 15 February 1941 beat Kristin: {above.sum()} of {len(K)}")
    south = M.columns.get_level_values("lat") < 40.0
    print(f"  south of 40 N: {above[south].sum()} of {south.sum()}; north: {above[~south].sum()} of {(~south).sum()}")
    X86 = np.vstack([ME.values, M.values])
    p86 = ev.fit_pooled_shape(X86)
    T86 = ev.return_period(K.values, p86["mu"], p86["sigma"], p86["xi"])
    print(f"Pooled shape with {X86.shape[0]} seasons: xi = {p86['xi']:+.3f}")
    order = R.sort_values("T_pool_in", ascending=False).index[:6]
    for j in order:
        rank = int((X86[:-1, j] >= K.values[j]).sum()) + 1
        print(f"   {R.lat[j]:.2f} N {R.lon[j]:.2f} E: Kristin rank {rank} of {X86.shape[0]}; return period {fmt_T(T86[j])} "
              f"(76 seasons: {fmt_T(R.T_pool_in[j])}); 15 Feb 1941 here {c41.values[j]:.1f} m/s")
    n41 = (AE.loc[1940] == pd.Timestamp("1941-02-15")).sum()
    print(f"Nodes where 15 February 1941 was the 1940/41 season maximum: {n41}")
    rank41 = [(int((np.delete(X86[:, j], 0) >= c41.values[j]).sum()) + 1) for j in range(len(K))]
    print(f"Nodes where 15 February 1941 is the highest of all {X86.shape[0]} seasons: {sum(r == 1 for r in rank41)}; "
          f"Kristin the highest: {sum(int((X86[:-1, j] >= K.values[j]).sum()) == 0 for j in range(len(K)))}")
    T41 = ev.return_period(c41.values, p86["mu"], p86["sigma"], p86["xi"])
    j41 = int(np.argmax(T41))
    print(f"Return period of 15 February 1941 under the 86-season fit: highest {fmt_T(T41[j41])} at "
          f"{M.columns[j41][0]:.2f} N {M.columns[j41][1]:.2f} E; nodes at 100 or more: {(T41 >= 100).sum()}")
    return dict(c41=c41, ME=ME, p86=p86, T86=T86)


def shape_sensitivity(M, K, R, pin, boot_xi):
    """Kristin's return period at the most affected nodes as a function of a fixed shape,
    with the composite profile of the shape over all nodes."""
    section("2e. How much the answer depends on the shape")
    X = M.values
    grid = np.round(np.arange(-0.25, 0.151, 0.01), 2)
    top = list(R.sort_values("T_pool_in", ascending=False).index[:3])
    prof, curves = [], {j: [] for j in top}
    for xi in grid:
        fits = [ev.fit_gev(X[:, j], fixed_xi=float(xi)) for j in range(X.shape[1])]
        prof.append(-sum(f.nll for f in fits))
        for j in top:
            f = fits[j]
            curves[j].append(float(ev.return_period(K.values[j], f.mu, f.sigma, f.xi)))
    prof = np.array(prof)
    lo, hi = np.percentile(boot_xi, [2.5, 97.5])
    for j in top:
        c = np.array(curves[j])
        at = lambda v: float(np.interp(v, grid, np.log10(np.minimum(c, 1e12))))
        print(f"   {R.lat[j]:.2f} N {R.lon[j]:.2f} E: return period at xi = {lo:+.2f}: {fmt_T(10 ** at(lo))}; "
              f"at xi = {pin['xi']:+.2f}: {fmt_T(10 ** at(pin['xi']))}; at xi = {hi:+.2f}: {fmt_T(10 ** at(hi))}; "
              f"Gumbel (xi = 0): {fmt_T(10 ** at(0.0))}")
    return grid, prof, curves, top


# =============================================================================
# 3. Is the tail getting heavier?
# =============================================================================

def tau_of(seasons):
    return (np.asarray(seasons, float) - (FIRST + LAST) / 2) / 10   # decades from the middle


def ns_fits(X, cov, scale_too=False, starts=None):
    """Per-node GEV with location (and optionally log scale) linear in `cov`.
    Returns relative location effect (per unit of cov, as a fraction of mu0), scale effect
    (change in log sigma per unit), LR statistic against the stationary fit."""
    J = X.shape[1]
    rel, ds, lr = np.empty(J), np.empty(J), np.empty(J)
    for j in range(J):
        x = X[:, j]
        f0 = ev.fit_gev(x) if starts is None else starts[j]
        st = np.r_[f0.mu, 0.0, f0.beta_s[0]] if not scale_too else np.r_[f0.mu, 0.0, f0.beta_s[0], 0.0]
        st = np.r_[st, f0.xi]
        f1 = ev.fit_gev(x, Xmu=cov[:, None], Xsig=cov[:, None] if scale_too else None, start=st, multi_start=False)
        rel[j] = f1.beta_mu[1] / f1.beta_mu[0]
        ds[j] = f1.beta_s[1] if scale_too else 0.0
        lr[j] = 2 * (f0.nll - f1.nll)
    return rel, ds, lr


def trend_analysis(M, nao, rng):
    section("3. Trend in the season maxima (location and scale of the GEV)")
    X = M.values
    seasons = M.index.values
    tau = tau_of(seasons)
    f0 = [ev.fit_gev(X[:, j]) for j in range(X.shape[1])]
    rel, ds, lr = ns_fits(X, tau, scale_too=True, starts=f0)
    rel1, _, lr1 = ns_fits(X, tau, scale_too=False, starts=f0)
    p_node = stats.chi2.sf(lr, 2)
    print(f"Location trend, mean over nodes: {100 * rel1.mean():+.2f}% of the location per decade "
          f"(nodes from {100 * rel1.min():+.2f}% to {100 * rel1.max():+.2f}%)")
    print(f"With the scale also trending: location {100 * rel.mean():+.2f}% per decade, scale {100 * ds.mean():+.2f}% per decade")
    print(f"Nodes with a 2-df likelihood-ratio p < 0.05: {(p_node < 0.05).sum()} of {len(p_node)}")
    # 50-year level at the start and end of the record from the trend-in-location-and-scale fit
    # (computed again below to report the implied change)
    # Field significance by permuting the season order (same permutation for every node)
    null_rel, null_ds, null_cnt = [], [], []
    for b in range(N_PERM):
        perm = rng.permutation(len(seasons))
        r_b, d_b, l_b = ns_fits(X[perm], tau, scale_too=True, starts=f0)
        null_rel.append(r_b.mean())
        null_ds.append(d_b.mean())
        null_cnt.append((stats.chi2.sf(l_b, 2) < 0.05).sum())
    null_rel, null_ds, null_cnt = map(np.array, (null_rel, null_ds, null_cnt))
    p_rel = (1 + np.sum(np.abs(null_rel) >= abs(rel.mean()))) / (N_PERM + 1)
    p_ds = (1 + np.sum(np.abs(null_ds) >= abs(ds.mean()))) / (N_PERM + 1)
    p_cnt = (1 + np.sum(null_cnt >= (p_node < 0.05).sum())) / (N_PERM + 1)
    print(f"Permutation tests ({N_PERM} reorderings of the {len(seasons)} seasons, all nodes together):")
    print(f"  mean location trend: p = {p_rel:.3f}; mean scale trend: p = {p_ds:.3f}; "
          f"count of nodes with p < 0.05: p = {p_cnt:.3f}")
    print(f"  95% of reorderings give a mean location trend within +-{100 * np.percentile(np.abs(null_rel), 95):.2f}% per decade "
          f"and a mean scale trend within +-{100 * np.percentile(np.abs(null_ds), 95):.2f}% per decade")
    print(f"  under reordering, the count of nodes with p < 0.05 has median {np.median(null_cnt):.0f} and 95th percentile "
          f"{np.percentile(null_cnt, 95):.0f}")
    # implied change in the 50-year level, start to end of record, averaged over nodes
    ch = []
    for j in range(X.shape[1]):
        f = ev.fit_gev(X[:, j], Xmu=tau[:, None], Xsig=tau[:, None],
                       start=np.r_[f0[j].mu, 0, f0[j].beta_s[0], 0, f0[j].xi])
        a, b = tau[0], tau[-1]
        z_a = ev.return_level(50, f.beta_mu[0] + f.beta_mu[1] * a, np.exp(f.beta_s[0] + f.beta_s[1] * a), f.xi)
        z_b = ev.return_level(50, f.beta_mu[0] + f.beta_mu[1] * b, np.exp(f.beta_s[0] + f.beta_s[1] * b), f.xi)
        ch.append(z_b / z_a - 1)
    ch = np.array(ch)
    print(f"Implied change of the 50-year level from {season_label(FIRST)} to {season_label(LAST)}: "
          f"mean {100 * ch.mean():+.1f}%, nodes from {100 * ch.min():+.1f}% to {100 * ch.max():+.1f}%")

    # Positive control: the winter NAO, which is known to steer storms towards Iberia
    z = ((nao - nao.mean()) / nao.std()).loc[seasons].values
    reln, _, lrn = ns_fits(X, z, scale_too=False, starts=f0)
    pn = stats.chi2.sf(lrn, 1)
    nul = []
    for b in range(N_PERM):
        perm = rng.permutation(len(seasons))
        r_b, _, _ = ns_fits(X[perm], z, scale_too=False, starts=f0)
        nul.append(r_b.mean())
    nul = np.array(nul)
    p_nao = (1 + np.sum(np.abs(nul) >= abs(reln.mean()))) / (N_PERM + 1)
    print()
    print("Positive control, winter (October-March) NAO as the covariate of the location:")
    print(f"  mean effect {100 * reln.mean():+.2f}% of the location per standard deviation of the NAO "
          f"(nodes {100 * reln.min():+.2f}% to {100 * reln.max():+.2f}%), permutation p = {p_nao:.3f}")
    print(f"  nodes with p < 0.05: {(pn < 0.05).sum()} of {len(pn)}")
    south = M.columns.get_level_values("lat") < 40.0
    print(f"  mean effect south of 40 N: {100 * reln[south].mean():+.2f}%; north: {100 * reln[~south].mean():+.2f}%")
    # trend after the NAO
    both = []
    for j in range(X.shape[1]):
        fa = ev.fit_gev(X[:, j], Xmu=z[:, None], start=np.r_[f0[j].mu, 0, f0[j].beta_s[0], f0[j].xi])
        fb = ev.fit_gev(X[:, j], Xmu=np.column_stack([z, tau]),
                        start=np.r_[fa.beta_mu, 0, fa.beta_s, fa.xi])
        both.append(fb.beta_mu[2] / fb.beta_mu[0])
    both = np.array(both)
    print(f"  location trend with the NAO also in the model: mean {100 * both.mean():+.2f}% per decade")
    # 1979 onwards
    sel = M.index.values >= 1979
    X79, tau79 = X[sel], tau_of(M.index.values[sel])
    f79 = [ev.fit_gev(X79[:, j]) for j in range(X79.shape[1])]
    r79, _, l79 = ns_fits(X79, tau79, scale_too=False, starts=f79)
    nul79 = []
    for b in range(N_PERM):
        perm = rng.permutation(sel.sum())
        r_b, _, _ = ns_fits(X79[perm], tau79, scale_too=False, starts=f79)
        nul79.append(r_b.mean())
    p79 = (1 + np.sum(np.abs(np.array(nul79)) >= abs(r79.mean()))) / (N_PERM + 1)
    print()
    print(f"Satellite era only ({season_label(1979)} to {season_label(LAST)}): mean location trend "
          f"{100 * r79.mean():+.2f}% per decade, permutation p = {p79:.3f}")
    # halves: pooled shape and dispersion
    half = M.index.values < 1988
    ph1, ph2 = ev.fit_pooled_shape(X[half]), ev.fit_pooled_shape(X[~half])
    cv1, cv2 = np.median(ph1["sigma"] / ph1["mu"]), np.median(ph2["sigma"] / ph2["mu"])
    d_xi = []
    for b in range(min(N_BOOT, 300)):
        i1 = rng.choice(np.where(half)[0], half.sum())
        i2 = rng.choice(np.where(~half)[0], (~half).sum())
        d_xi.append(ev.fit_pooled_shape(X[i2], start=ph2["theta"])["xi"] - ev.fit_pooled_shape(X[i1], start=ph1["theta"])["xi"])
    lo, hi = np.percentile(d_xi, [2.5, 97.5])
    print(f"Halves {season_label(FIRST)}-{season_label(1987)} and {season_label(1988)}-{season_label(LAST)}: pooled xi "
          f"{ph1['xi']:+.3f} and {ph2['xi']:+.3f} (difference 95% bootstrap {lo:+.3f} to {hi:+.3f}); "
          f"median sigma/mu {cv1:.3f} and {cv2:.3f}")
    return dict(rel1=rel1, rel=rel, ds=ds, lr=lr, null_rel=null_rel, reln=reln, nul_nao=nul,
                p_rel=p_rel, p_nao=p_nao, f0=f0)


def power_analysis(M, tr, rng):
    section("3b. Power of the field test for a location trend")
    X = M.values
    tau = tau_of(M.index.values)
    f0 = tr["f0"]
    mu0 = np.array([f.mu for f in f0])
    crit = np.percentile(np.abs(tr["null_rel"]), 95)
    sizes = [0.0, 0.5, 1.0, 2.0, 3.0]
    out = []
    for s in sizes:
        hits = 0
        for b in range(N_POWER):
            perm = rng.permutation(len(X))
            Y = X[perm] + (s / 100) * mu0[None, :] * tau[:, None]
            r, _, _ = ns_fits(Y, tau, scale_too=True)
            hits += abs(r.mean()) >= crit
        out.append(hits / N_POWER)
        total = s * (LAST - FIRST) / 10
        print(f"Location trend {s:.1f}% per decade ({total:.0f}% over the record): detected in {100 * out[-1]:.0f}% "
              f"of {N_POWER} simulated histories")
    return sizes, out


def station_trends(d):
    section("3c. Station check (synoptic-hour maxima, raw GHCNh, not homogenised)")
    for key in ("porto", "faro", "beja"):
        sid = STATIONS[key][0]
        sm, _, _ = station_maxima(d, sid)
        tau = tau_of(sm.index.values)
        f0 = ev.fit_gev(sm.values)
        f1 = ev.fit_gev(sm.values, Xmu=tau[:, None], start=np.r_[f0.mu, 0, f0.beta_s[0], f0.xi])
        lr = 2 * (f0.nll - f1.nll)
        print(f"{key:10s} {season_label(int(sm.index[0]))} to {season_label(int(sm.index[-1]))} ({len(sm)} seasons): "
              f"location trend {100 * f1.beta_mu[1] / f1.beta_mu[0]:+.2f}% per decade, LR p = {stats.chi2.sf(lr, 1):.3f}")


# =============================================================================
# 4. Design values against the tail
# =============================================================================

def design_shape(M, R, pin, boot_xi):
    section("4. The shape the code assumes against the shape in the data")
    print(f"Annex conversion v(T)/v(50) = 0.7 - 0.077 ln(-ln(1 - 1/T)): Gumbel with sigma/mu = {0.077 / 0.7:.3f}")
    print(f"  implied ratios: v(10)/v(50) = {c_prob(10):.3f}, v(100)/v(50) = {c_prob(100):.3f}, "
          f"v(500)/v(50) = {c_prob(500):.3f}, v(1000)/v(50) = {c_prob(1000):.3f}")
    cv = pin["sigma"] / pin["mu"]
    print(f"ERA5 nodes, pooled-shape fit: sigma/mu median {np.median(cv):.3f} (nodes {cv.min():.3f} to {cv.max():.3f}); "
          f"xi = {pin['xi']:+.3f}")
    for T in (10, 100, 500, 1000):
        r = ev.return_level(T, pin["mu"], pin["sigma"], pin["xi"]) / ev.return_level(50, pin["mu"], pin["sigma"], pin["xi"])
        g = ev.return_level(T, pin["mu"], pin["sigma"], 0.0) / ev.return_level(50, pin["mu"], pin["sigma"], 0.0)
        print(f"  v({T})/v(50): ERA5 pooled fit median {np.median(r):.3f}; same location and scale with xi = 0: {np.median(g):.3f}")
    # Kristin relative to the node's 50-year level, and what the annex would call that ratio
    ratio = R.K / R.z50_pool
    top = R.assign(ratio=ratio).sort_values("ratio", ascending=False).head(6)
    print("Kristin relative to the node's 50-year level (pooled fit), and the return period the annex's")
    print("formula gives to the same ratio:")
    for _, r in top.iterrows():
        # solve c_prob(T) = ratio
        y = np.exp((0.7 - r.ratio) / 0.077)
        T = 1 / (1 - np.exp(-y))
        print(f"  {r.lat:.2f} N {r.lon:.2f} E: Kristin / v(50) = {r.ratio:.3f}; annex return period {fmt_T(T)}; "
              f"ERA5 pooled return period {fmt_T(r.T_pool_in)}")
    return cv


def design_stations(d, pin, rng):
    section("4b. Station 50-year mean winds against the basic wind velocity")
    # sampling: synoptic-hour maxima miss the peak between synoptic hours
    ratios = []
    for key in ("porto", "faro"):
        sm, am, nobs = station_maxima(d, STATIONS[key][0])
        ok = nobs >= 40
        ratios += list((am[ok] / sm[ok]).values)
    ratios = np.array(ratios)
    r_med = float(np.median(ratios))
    print(f"Season maximum from all reports / from synoptic hours, Porto and Faro seasons with >= 40 reports a day: "
          f"median {r_med:.3f}, quartiles {np.percentile(ratios, 25):.3f} to {np.percentile(ratios, 75):.3f} ({len(ratios)} seasons)")
    # half-hourly reports sample one ten-minute window in three; extrapolating the growth of the
    # maximum in log(reports per day) from 8 to 48 on to 144
    extra = 1 + (r_med - 1) * np.log(144 / 48) / np.log(48 / 8)
    print(f"Extrapolated further factor from 48 to 144 ten-minute windows a day: {extra:.3f}")
    corr = r_med * extra
    rows = []
    for key, (sid, zone, desc) in STATIONS.items():
        sm, am, nobs = station_maxima(d, sid)
        if len(sm) < 19:
            continue
        f = ev.fit_gev(sm.values)
        fx = ev.fit_gev(sm.values, fixed_xi=pin["xi"])
        z50 = float(ev.return_level(50, f.mu, f.sigma, f.xi))
        z50x = float(ev.return_level(50, fx.mu, fx.sigma, fx.xi))
        rows.append((key, zone, len(sm), int(sm.index[0]), int(sm.index[-1]), f.xi, f.sigma / f.mu, z50, z50x,
                     z50x * corr, sm.max(), int(sm.idxmax())))
        print(f"{desc:34s} zone {zone}: {len(sm)} seasons {season_label(int(sm.index[0]))}-{season_label(int(sm.index[-1]))}; "
              f"own xi {f.xi:+.2f}, sigma/mu {f.sigma / f.mu:.2f}; 50-yr synoptic {z50:.1f} m/s "
              f"(regional shape {z50x:.1f}; sampling-corrected {z50x * corr:.1f}); record {sm.max():.1f} in {season_label(int(sm.idxmax()))}; "
              f"basic velocity {VB0[zone]:.0f}")
    S = pd.DataFrame(rows, columns=["key", "zone", "n", "first", "last", "xi", "cv", "z50", "z50x", "z50c", "rec", "rec_season"])
    nflag = sum(len(v) for v in FLAGGED.values())
    print(f"Station days dropped by the ERA5 cross-check (>= {QC_MIN:.0f} m/s and > {QC_RATIO} x the cell): {nflag}")
    for k, v in FLAGGED.items():
        for _, r in v.iterrows():
            print(f"   {k:11s} {r.date.date()}: station {r.wind_max_syn_ms:.1f} (synoptic) / {r.wind_max_ms:.1f} (all), "
                  f"ERA5 cell {r.era5:.1f}, lowest pressure {r.slp_min_hpa:.0f} hPa")
    S["ratio"] = S.z50c / S.zone.map(VB0)
    print(f"Sampling-corrected 50-year level / basic wind velocity: median {S.ratio.median():.2f} "
          f"(stations {S.ratio.min():.2f} to {S.ratio.max():.2f})")
    print(f"Station sigma/mu: median {S.cv.median():.2f} (annex 0.11)")
    return S, corr


def kristin_points(d):
    section("4c. Kristin at the stations: ten-minute means and gusts against design values")
    # GHCNh airports
    for key in ("monte_real", "montijo", "beja", "faro", "porto"):
        sid = STATIONS[key][0]
        x = d[(d.station == sid) & (d.date == KRISTIN)]
        if len(x):
            r = x.iloc[0]
            print(f"{STATIONS[key][2]:34s} 28 Jan: highest synoptic-hour mean {r.wind_max_syn_ms:.1f} m/s, "
                  f"highest report {r.wind_max_ms:.1f} m/s ({int(r.n_wind_obs)} reports); zone {STATIONS[key][1]} basic velocity {VB0[STATIONS[key][1]]:.0f}")
    path = DATA / "kristin_reports.csv"
    if path.exists() and not path.read_text()[:40].startswith("version https://git-lfs"):
        rep = pd.read_csv(path, dtype=str)
        mr = rep[(rep.site == "monte_real") & rep.DATE.str.startswith("2026-01-28")]
        print("Monte Real reports on 28 January (UTC; direction; ten-minute mean m/s; gust m/s):")
        for _, r in mr.iterrows():
            if r.DATE[11:13] <= "09":
                g = r.wind_gust if isinstance(r.wind_gust, str) else ""
                print(f"   {r.DATE[11:16]}  {r.wind_direction if isinstance(r.wind_direction, str) else '':>4}  "
                      f"{r.wind_speed if isinstance(r.wind_speed, str) else '':>5}  {g:>5}")
        w = pd.to_numeric(mr.wind_speed, errors="coerce")
        i = w.idxmax()
        mr_mean = (float(w.max()), mr.DATE[i][11:16])
        print(f"Highest ten-minute mean reported at Monte Real: {mr_mean[0]:.1f} m/s at {mr_mean[1]} UTC "
              f"({100 * (mr_mean[0] / VB0['A'] - 1):+.0f}% against zone A's basic wind velocity)")
    else:
        mr_mean = None
    print()
    print(f"Eurocode peak velocity at 10 m, terrain II: mean x sqrt(1 + 7 I_v) with I_v = {I_V10:.3f}, factor {PEAK:.3f}")
    print(f"  zone A {VB0['A'] * PEAK:.1f} m/s ({3.6 * VB0['A'] * PEAK:.0f} km/h), zone B {VB0['B'] * PEAK:.1f} m/s ({3.6 * VB0['B'] * PEAK:.0f} km/h)")
    print(f"Overhead line design gust below 20 m (EN 50341-3-17): zone A {LINE_GUST['A']:.1f} m/s ({3.6 * LINE_GUST['A']:.0f} km/h), "
          f"zone B {LINE_GUST['B']:.1f} m/s ({3.6 * LINE_GUST['B']:.0f} km/h); E-REDES specification {EREDES_GUST:.0f} m/s")
    gusts = pd.read_csv(DATA / "kristin_gusts.csv")
    print("Gusts reported on 28 January 2026, km/h and m/s, against the design gusts of the zone:")
    n_line = n_code = 0
    for _, g in gusts.iterrows():
        v = g.gust_kmh / 3.6
        code, line = VB0[g.zone] * PEAK, LINE_GUST[g.zone]
        n_line += v > line
        n_code += v > code
        print(f"   {g['name']:32s} {g.gust_kmh:6.1f} km/h {v:5.1f} m/s  zone {g.zone}: {100 * v / code:4.0f}% of the building peak, "
              f"{100 * v / line:4.0f}% of the line gust")
    print(f"Above the line design gust: {n_line} of {len(gusts)}; above the building peak velocity: {n_code} of {len(gusts)}")
    return gusts, mr_mean


def era5_cells(d):
    """ERA5 at the grid cells of the stations, against the anemometers."""
    section("4d. ERA5 at the station cells against the anemometers")
    out = {}
    for key, (sid, zone, desc) in STATIONS.items():
        f = DATA / "era5_sites" / f"site_{key}.csv.gz"
        if not f.exists():
            continue
        e = pd.read_csv(f, parse_dates=["date"]).set_index("date").wind_max_ms
        em, _ = season_maxima(e.to_frame("v"))
        sm, _, _ = station_maxima(d, sid)
        common = sm.index.intersection(em.index)
        if len(common) < 15:
            continue
        a, b = sm.loc[common].values, em.v.loc[common].values
        rho = stats.spearmanr(a, b).statistic
        x = d[(d.station == sid) & (d.n_syn_wind >= 6)].set_index("date").wind_max_syn_ms
        x = x[x.index.month.isin(WINTER)]
        j = pd.concat([x, e], axis=1, join="inner").dropna()
        rho_d = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1]).statistic
        top_st = set(j.iloc[:, 0].nlargest(10).index)
        top_e = set(j.iloc[:, 1].nlargest(30).index)
        hit = len(top_st & top_e)
        fs, fe = ev.fit_gev(a), ev.fit_gev(b)
        kv = e.loc[KRISTIN]
        Tk = float(ev.return_period(kv, *(lambda f: (f.mu, f.sigma, f.xi))(ev.fit_gev(em.v.values))))
        out[key] = dict(ratio=np.median(a / b), rho=rho, rho_d=rho_d, hit=hit, kristin=kv, T=Tk,
                        cv_s=fs.sigma / fs.mu, cv_e=fe.sigma / fe.mu)
        print(f"{desc:34s} {len(common)} seasons: station/ERA5 ratio of season maxima {np.median(a / b):.2f}; "
              f"rank correlation of season maxima {rho:.2f}, of winter days {rho_d:.2f}; "
              f"station's 10 windiest days among ERA5's 30: {hit}; sigma/mu station {fs.sigma / fs.mu:.2f}, ERA5 {fe.sigma / fe.mu:.2f}; "
              f"ERA5 Kristin {kv:.1f} m/s, ERA5 return period {fmt_T(Tk)}")
    if out:
        O = pd.DataFrame(out).T
        print(f"Medians over stations: rank correlation of season maxima {O.rho.median():.2f}, of winter days {O.rho_d.median():.2f}; "
              f"sigma/mu station {O.cv_s.median():.2f}, ERA5 {O.cv_e.median():.2f}")
    # Kristin's day at the cells of the stations whose gusts were reported
    gusts = pd.read_csv(DATA / "kristin_gusts.csv")
    keys = ["monte_real", "degracias_soure", "cabo_carvoeiro", "ansiao", "leiria", "castelo_branco", "foia",
            "tomar_vale_donas", "cabo_da_roca", "santa_cruz"]
    ratios = []
    print("ERA5 on 28 January 2026 at the cells of the stations IPMA reported (daily maximum wind, gust; observed gust):")
    for k, (_, g) in zip(keys, gusts.iterrows()):
        f = DATA / "era5_kristin" / f"kristin_{k}.csv.gz"
        if not f.exists():
            continue
        e = pd.read_csv(f, parse_dates=["date"]).set_index("date").loc[KRISTIN]
        ratios.append(e.gust_max_ms / (g.gust_kmh / 3.6))
        print(f"   {g['name']:32s} ERA5 {e.wind_max_ms:5.1f} / {e.gust_max_ms:5.1f} m/s; observed {g.gust_kmh / 3.6:5.1f} m/s; "
              f"ERA5 gust / observed {ratios[-1]:.2f}")
    if ratios:
        print(f"ERA5 gust as a fraction of the reported gust: median {np.median(ratios):.2f}, range {min(ratios):.2f} to {max(ratios):.2f}")
    return out


def gust_tail(G, KG):
    section("4e. ERA5 gusts")
    MG, _ = season_maxima(G)
    pg = ev.fit_pooled_shape(MG.values)
    T = ev.return_period(KG.values, pg["mu"], pg["sigma"], pg["xi"])
    z50 = ev.return_level(50, pg["mu"], pg["sigma"], pg["xi"])
    print(f"Pooled shape of ERA5 gust maxima: xi = {pg['xi']:+.3f}; sigma/mu median {np.median(pg['sigma'] / pg['mu']):.3f}")
    print(f"ERA5 50-year gust: median over nodes {np.median(z50):.1f} m/s (nodes {z50.min():.1f} to {z50.max():.1f})")
    top = KG.idxmax()
    print(f"ERA5 gust on 28 Jan 2026: highest {KG.max():.1f} m/s at {top[0]:.2f} N {top[1]:.2f} E")
    for thr, lab in ((LINE_GUST["A"] / 1.1484, "line reference gust at 10 m, zone A"),
                     (LINE_GUST["B"] / 1.1484, "line reference gust at 10 m, zone B")):
        print(f"{lab} ({thr:.0f} m/s): ERA5 50-year gust above it at {(z50 > thr).sum()} of {len(z50)} nodes; "
              f"reached at one node or more in {(MG.max(axis=1) >= thr).sum()} of {len(MG)} seasons")
    order = np.argsort(-T)[:6]
    for j in order:
        print(f"   {MG.columns[j][0]:.2f} N {MG.columns[j][1]:.2f} E: Kristin gust {KG.values[j]:.1f} m/s, 50-yr {z50[j]:.1f}, "
              f"return period {fmt_T(T[j])}")
    return MG, pg, T


# =============================================================================
# 5. One storm, many sites
# =============================================================================

def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def network_exposure(M, R, pin):
    section("5. Spatial dependence of the season maxima and exposure of a network")
    U = ev.ranks_to_uniform(M.values)
    coords = list(M.columns)
    J = len(coords)
    dist, theta = [], []
    for i in range(J):
        for j in range(i + 1, J):
            dist.append(haversine_km(coords[i], coords[j]))
            theta.append(ev.madogram_theta(U[:, i], U[:, j]))
    dist, theta = np.array(dist), np.array(theta)
    for a, b in ((0, 60), (60, 120), (120, 200), (200, 300), (300, 600)):
        s = (dist >= a) & (dist < b)
        print(f"Pairs {a}-{b} km apart: {s.sum():3d}, median pairwise extremal coefficient {np.median(theta[s]):.2f}")
    th_all = ev.field_theta(U)
    south = np.array([c[0] < 40.0 for c in coords])
    th_s, th_n = ev.field_theta(U[:, south]), ev.field_theta(U[:, ~south])
    print(f"Whole field of {J} nodes behaves like {th_all:.1f} independent sites for its maximum "
          f"(south of 40 N: {th_s:.1f} of {south.sum()}; north: {th_n:.1f} of {(~south).sum()})")
    for th, lab in ((1, "one site"), (th_s, "south of 40 N"), (th_all, "all nodes")):
        p = 1 - (1 - 1 / 50) ** th
        print(f"  chance a season brings the 50-year wind to at least one node, {lab}: {100 * p:.1f}% (once in {1 / p:.1f} seasons)")
    exc = M.values > R.z50_pool.values[None, :]
    any_exc = exc.any(axis=1)
    print(f"Observed: seasons with at least one node above its own 50-year level (pooled fit): {any_exc.sum()} of {len(M)} "
          f"({', '.join(season_label(int(s)) for s in M.index[any_exc])})")
    cnt = exc.sum(axis=1)
    print(f"Nodes above their 50-year level in 2025/26: {cnt[-1]}; most in any earlier season: {cnt[:-1].max()} "
          f"({season_label(int(M.index[:-1][np.argmax(cnt[:-1])]))})")
    return dist, theta, th_all, cnt


# =============================================================================
# Figures
# =============================================================================

def draw_outline(ax):
    path = DATA / "portugal_outline.csv"
    if not path.exists():
        return
    o = pd.read_csv(path)
    for (c, r), g in o.groupby(["country", "ring"]):
        ax.plot(g.lon, g.lat, color=INK_2 if c == "PRT" else GRID, lw=0.9 if c == "PRT" else 0.8, zorder=1)


def fig_map(R, gusts):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), sharey=True)
    bounds = [1, 2, 5, 10, 50, 100, 500, 1e9]
    labels = ["< 2", "2-5", "5-10", "10-50", "50-100", "100-500", "> 500"]
    cols = [GRID] + SEQ
    for ax, key, title in ((axes[0], "T_pool_ex", "Fitted without 2025/26"), (axes[1], "T_pool_in", "Fitted with 2025/26")):
        draw_outline(ax)
        k = np.digitize(R[key].values, bounds) - 1
        k = np.clip(k, 0, len(labels) - 1)
        for i in range(len(labels)):
            s = k == i
            ax.scatter(R.lon[s], R.lat[s], s=210, marker="s", color=cols[i], edgecolor=SURFACE, lw=2, zorder=3,
                       label=labels[i] if ax is axes[1] else None)
        ax.set_xlim(-9.8, -6.0)
        ax.set_ylim(36.8, 42.3)
        ax.set_aspect(1 / np.cos(np.radians(39.5)))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(False)
        ax.set_xlabel("Longitude")
    for _, g in gusts.iterrows():
        axes[1].plot(g.lon, g.lat, "o", ms=5, mfc=SURFACE, mec=ORANGE, mew=1.6, zorder=4)
    axes[1].plot([], [], "o", ms=5, mfc=SURFACE, mec=ORANGE, mew=1.6, label="Station gust reported by IPMA")
    axes[0].set_ylabel("Latitude")
    fig.suptitle("Return period of Kristin's ERA5 daily maximum wind at each node (one shape for all nodes)",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Return period (seasons)", loc="lower center", ncol=4, fontsize=9,
               title_fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIGURES / "01_kristin_return_periods.png", dpi=DPI)
    plt.close(fig)


def fig_return_level(M, K, R, j, prof):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    x = np.sort(M.values[:, j])
    n = len(x)
    emp_T = 1 / (1 - (np.arange(1, n + 1) - 0.44) / (n + 0.12))   # Gringorten plotting positions
    Tg = np.logspace(0.05, 4, 200)
    for data, color, label in ((M.values[:-1, j], BLUE, "Fit without 2025/26"), (M.values[:, j], ORANGE, "Fit with 2025/26")):
        f = ev.fit_gev(data)
        ax.plot(Tg, ev.return_level(Tg, f.mu, f.sigma, f.xi), color=color, lw=2, label=label)
    ax.plot(emp_T[:-1], x[:-1], "o", ms=4.5, color=INK_2, label="Season maxima 1950/51-2024/25")
    ax.plot(emp_T[-1], K.values[j], "D", ms=8, color=INK, label="Kristin, 28 January 2026", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Return period (seasons)")
    ax.set_ylabel("ERA5 daily maximum wind (m/s)")
    ax.set_title(f"Node {R.lat[j]:.2f} N, {abs(R.lon[j]):.2f} W", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=9, loc="upper left")
    bx = axes[1]
    for key, color, label in (("cex", BLUE, "Without 2025/26"), ("cin", ORANGE, "With 2025/26")):
        c = prof[j][key]
        bx.plot(10 ** c[:, 0], c[:, 1], color=color, lw=2, label=label)
    bx.axhline(stats.chi2.ppf(0.95, 1), color=INK_2, lw=1, ls=(0, (4, 3)))
    bx.text(1.2, stats.chi2.ppf(0.95, 1) + 0.25, "95% cut-off", fontsize=9, color=INK_2)
    bx.set_xscale("log")
    bx.set_ylim(0, 12)
    bx.set_xlabel("Return period of Kristin's value (seasons)")
    bx.set_ylabel("Profile deviance")
    bx.set_title("Profile likelihood of Kristin's return period", loc="left", fontsize=11, color=INK)
    bx.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_return_level_and_profile.png", dpi=DPI)
    plt.close(fig)


def fig_shape(grid, prof, curves, top, R, pin, boot_xi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    lo, hi = np.percentile(boot_xi, [2.5, 97.5])
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0, label="95% season-bootstrap interval")
    ax.plot(grid, prof - prof.max(), color=BLUE, lw=2, label="All 38 nodes treated as independent")
    ax.axvline(0, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(0.005, -2, "Gumbel, as the\nbuilding code assumes", fontsize=9, color=INK_2, va="top")
    ax.set_ylim(-60, 3)
    ax.set_xlabel("Shape parameter xi")
    ax.set_ylabel("Log-likelihood minus its maximum")
    ax.set_title("What the 76 seasons say about the shape", loc="left", fontsize=11, color=INK)
    ax.text(-0.245, -38, "The curve is too narrow because\nneighbouring nodes share storms;\nthe shaded band is the honest range.",
            fontsize=8.5, color=INK_2, va="top")
    ax.legend(fontsize=9, loc="lower left")
    bx = axes[1]
    bx.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    for j, color in zip(top, (ORANGE, BLUE, AQUA)):
        bx.plot(grid, np.minimum(np.array(curves[j]), 1e6), color=color, lw=2,
                label=f"Node {R.lat[j]:.2f} N, {abs(R.lon[j]):.2f} W")
    bx.axvline(0, color=INK_2, lw=1, ls=(0, (4, 3)))
    bx.axvline(pin["xi"], color=INK, lw=1)
    bx.text(pin["xi"] + 0.005, 2.0e5, "fitted", fontsize=9, color=INK)
    bx.set_yscale("log")
    bx.set_ylim(10, 1e6)
    bx.set_xlabel("Shape parameter xi (held fixed)")
    bx.set_ylabel("Kristin's return period (seasons)")
    bx.set_title("Kristin's return period follows the shape", loc="left", fontsize=11, color=INK)
    bx.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_shape_sensitivity.png", dpi=DPI)
    plt.close(fig)


def fig_trend(M, tr, sizes, power):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    bins = np.linspace(-3, 3, 49)
    ax.hist(100 * tr["null_rel"], bins=bins, color=BLUE, alpha=0.35, label="Season order shuffled (time)")
    ax.hist(100 * tr["nul_nao"], bins=bins, color=AQUA, alpha=0.35, label="Season order shuffled (NAO)")
    ax.axvline(100 * tr["rel1"].mean(), color=BLUE, lw=2.2)
    ax.text(100 * tr["rel1"].mean(), ax.get_ylim()[1] * 0.92, "  trend\n  per decade", color=INK, fontsize=9, va="top")
    ax.axvline(100 * tr["reln"].mean(), color=AQUA, lw=2.2)
    ax.text(100 * tr["reln"].mean(), ax.get_ylim()[1] * 0.92, "NAO effect  \nper SD  ", color=INK, fontsize=9, va="top", ha="right")
    ax.set_xlabel("Mean effect on the GEV location over 38 nodes (% of the location)")
    ax.set_ylabel("Reorderings")
    ax.set_title("Observed effects against shuffled seasons", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.75))
    bx = axes[1]
    bx.bar(range(len(sizes)), power, color=[INK_2] + [BLUE] * (len(sizes) - 1), width=0.55)
    for i, p in enumerate(power):
        bx.text(i, p + 0.02, f"{100 * p:.0f}%", ha="center", fontsize=9.5, color=INK)
    bx.axhline(0.8, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    bx.text(-0.45, 0.81, "80% power", fontsize=8.5, color=INK_2, ha="left", va="bottom")
    bx.set_xticks(range(len(sizes)))
    bx.set_xticklabels([f"{s:g}%" for s in sizes])
    bx.set_ylim(0, 1.08)
    bx.set_xlabel("Location trend added (% per decade)")
    bx.set_ylabel("Share detected at the 5% level")
    bx.set_title("Power of the same test", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_trend_test_and_power.png", dpi=DPI)
    plt.close(fig)


def fig_design(S, gusts, mr_mean):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    S = S.sort_values("z50c")
    y = np.arange(len(S))
    ax.hlines(y, S.z50x, S.z50c, color=GRID, lw=3)
    ax.plot(S.z50x, y, "o", color=BLUE, ms=7, label="50-year mean wind, synoptic hours")
    ax.plot(S.z50c, y, "o", color=ORANGE, ms=7, label="Corrected for sampling")
    ax.plot(S.rec, y, "|", color=INK, ms=12, mew=2, label="Highest season maximum")
    for i, (_, r) in enumerate(S.iterrows()):
        ax.plot(VB0[r.zone], i, "s", color=INK_2, ms=7, mfc=SURFACE, mew=1.6, label="Basic wind velocity (code)" if i == 0 else None)
        if r.key == "monte_real" and mr_mean is not None:
            ax.plot(mr_mean[0], i, "*", color=ORANGE, ms=13, mec=INK, mew=0.6, label=f"Kristin, {mr_mean[1]} UTC report")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k.replace('_', ' ').title().replace('Gc', 'GC')} ({z})" for k, z in zip(S.key, S.zone)])
    ax.set_xlabel("Ten-minute mean wind at 10 m (m/s)")
    ax.set_title("Stations against the building code (zone)", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
    ax.set_xlim(10, 37)
    bx = axes[1]
    g = gusts.sort_values("gust_kmh")
    yy = np.arange(len(g))
    v = g.gust_kmh.values / 3.6
    bx.barh(yy, v, color=[ORANGE if z == "A" else YELLOW for z in g.zone], height=0.6)
    for i, (_, r) in enumerate(g.iterrows()):
        bx.plot([LINE_GUST[r.zone]] * 2, [i - 0.38, i + 0.38], color=INK, lw=2)
        bx.plot([VB0[r.zone] * PEAK] * 2, [i - 0.38, i + 0.38], color=INK_2, lw=2, ls=(0, (2, 1.5)))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK, lw=2, label="Line design gust, below 20 m"),
               Line2D([], [], color=INK_2, lw=2, ls=(0, (2, 1.5)), label="Building code peak velocity"),
               Patch(color=ORANGE, label="Station in zone A"), Patch(color=YELLOW, label="Station in zone B")]
    bx.set_yticks(yy)
    bx.set_yticklabels(g["name"])
    bx.set_xlabel("Gust on 28 January 2026 (m/s)")
    bx.set_title("Kristin's reported gusts against design gusts", loc="left", fontsize=11, color=INK)
    bx.legend(handles=handles, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_design_values.png", dpi=DPI)
    plt.close(fig)


def fig_dependence(dist, theta, cnt, M):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(dist, theta, "o", ms=3, color=BLUE, alpha=0.45)
    edges = np.arange(0, 650, 50)
    mid = (edges[:-1] + edges[1:]) / 2
    med = [np.median(theta[(dist >= a) & (dist < b)]) if ((dist >= a) & (dist < b)).sum() > 5 else np.nan
           for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mid, med, color=INK, lw=2, label="Median in 50 km bins")
    ax.axhline(2, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    ax.text(10, 2.02, "independent", fontsize=9, color=INK_2, va="bottom")
    ax.axhline(1, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    ax.text(10, 1.02, "identical", fontsize=9, color=INK_2, va="bottom")
    ax.set_ylim(0.9, 2.15)
    ax.set_xlabel("Distance between nodes (km)")
    ax.set_ylabel("Pairwise extremal coefficient")
    ax.set_title("How far a season maximum is shared", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=9, loc="lower right")
    bx = axes[1]
    s = M.index.values
    bx.bar(s, cnt, color=[ORANGE if v == LAST else BLUE for v in s], width=0.8)
    bx.set_xlabel("Season (labelled by its October)")
    bx.set_ylabel("Nodes above their own 50-year level")
    from matplotlib.ticker import MaxNLocator
    bx.yaxis.set_major_locator(MaxNLocator(integer=True))
    bx.set_title("Footprint of each season's strongest winds", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_dependence_and_footprint.png", dpi=DPI)
    plt.close(fig)


EQUATION_TEX = {
    "eq01_gev": r"$G(x) = \exp\left\{-\left[1 + \xi\,\dfrac{x - \mu}{\sigma}\right]^{-1/\xi}\right\}$",
    "eq02_return_period": r"$T(x) = \dfrac{1}{1 - G(x)}\ \ \mathrm{seasons}, \qquad z_T = \mu + \dfrac{\sigma}{\xi}\left[\left(-\ln(1 - 1/T)\right)^{-\xi} - 1\right]$",
    "eq03_pooled": r"$\ell_{\mathrm{pool}}(\mu_1, \ldots, \mu_{38}, \sigma_1, \ldots, \sigma_{38}, \xi) = \sum_{j=1}^{38} \sum_{s=1}^{76} \ln g(m_{sj};\, \mu_j, \sigma_j, \xi)$",
    "eq04_profile": r"$\left\{\,T : 2\left[\ell(\hat\mu, \hat\sigma, \hat\xi) - \max_{\sigma, \xi}\ \ell\left(\mu_T(\sigma, \xi), \sigma, \xi\right)\right] \leq \chi^2_{1,\,0.95}\,\right\}$",
    "eq05_trend": r"$\mu_j(s) = \mu_{0j}\,(1 + \beta_j\,\tau_s), \qquad \ln\sigma_j(s) = \ln\sigma_{0j} + \gamma_j\,\tau_s, \qquad \tau_s = \dfrac{s - 1987.5}{10}$",
    "eq06_annex": r"$\dfrac{v_b(T)}{v_{b,0}} = 0.7 - 0.077\,\ln\left[-\ln\left(1 - 1/T\right)\right]$",
    "eq07_peak": r"$v_p = v_m\sqrt{1 + 7\,I_v}, \qquad I_v(10\,\mathrm{m}) = \dfrac{1}{\ln(10/0.05)} = 0.189$",
    "eq08_madogram": r"$\nu_{ij} = \frac{1}{2}\,\mathrm{E}\,\left|F_i(M_i) - F_j(M_j)\right|, \qquad \theta_{ij} = \dfrac{1 + 2\nu_{ij}}{1 - 2\nu_{ij}}$",
    "eq09_network": r"$\Pr\left(\mathrm{some\ site\ exceeds\ its\ 50\,yr\ level\ in\ a\ season}\right) = 1 - \left(1 - \frac{1}{50}\right)^{\theta}$",
}


def draw_equations() -> None:
    """Render each displayed equation on a canvas of fixed width (1600 px) so all share one scale."""
    EQUATIONS.mkdir(parents=True, exist_ok=True)
    width_in, dpi = 8.0, 200
    for name, tex in EQUATION_TEX.items():
        fig = plt.figure(figsize=(width_in, 1.0), dpi=dpi)
        fig.patch.set_facecolor(SURFACE)
        with plt.rc_context({"mathtext.fontset": "cm"}):
            size = 19
            text = fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=size, color=INK)
            fig.canvas.draw()
            box = text.get_window_extent()
            if box.width > 0.94 * width_in * dpi:
                size = size * 0.94 * width_in * dpi / box.width
                text.set_fontsize(size)
                fig.canvas.draw()
                box = text.get_window_extent()
        height_in = box.height / dpi + 0.32
        fig.set_size_inches(width_in, height_in)
        fig.savefig(EQUATIONS / f"{name}.png", dpi=dpi, facecolor=SURFACE)
        plt.close(fig)


# =============================================================================

def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    np.set_printoptions(precision=4, suppress=True)
    print("Storm wind extremes in Portugal: printed results")
    print("================================================")
    W, G, pts = load_nodes()
    M, A = season_maxima(W)
    K, rec = describe_data(W, M, A, G)
    d, st = load_stations()
    nao = load_nao()

    R, pin, pex, boot_xi = kristin_return_periods(M, K, rng)
    top = list(R.sort_values("T_pool_in", ascending=False).index[:3])
    prof = profile_intervals(M, K, R, top)
    selection_bias(M, R, pin, rng, top[0])
    peaks_over_threshold(W, M, K, R, rng)
    shape = shape_sensitivity(M, K, R, pin, boot_xi)
    early = early_record(M, K, R)

    tr = trend_analysis(M, nao, rng)
    sizes, power = power_analysis(M, tr, rng)
    station_trends(d)

    design_shape(M, R, pin, boot_xi)
    S, corr = design_stations(d, pin, rng)
    gusts, mr_mean = kristin_points(d)
    era5_cells(d)
    if G is not None:
        gust_tail(G, G.loc[KRISTIN])

    dist, theta, th_all, cnt = network_exposure(M, R, pin)

    fig_map(R, gusts)
    fig_return_level(M, K, R, top[0], prof)
    fig_shape(*shape, R, pin, boot_xi)
    fig_trend(M, tr, sizes, power)
    fig_design(S, gusts, mr_mean)
    fig_dependence(dist, theta, cnt, M)
    draw_equations()
    print()
    print("Figures written to figures/, equation images to figures/equations/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
