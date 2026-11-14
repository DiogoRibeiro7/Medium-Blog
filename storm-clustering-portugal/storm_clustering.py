"""Storm clustering in Portugal: how rare was the 2025/26 storm train, and is clustering
increasing?

Reads the snapshots in data/ (written by fetch.py and fetch_era5.py through GitHub
Actions), prints every number quoted in the article to standard output, and draws the
figures and equation images. No network access.

Sections
  1. Data and the storm severity index (ERA5 nodes and NOAA stations)
  2. The episode catalogue and its validation against named storms and the stations
  3. Dispersion of storm counts, and how much the circulation indices explain
  4. Hawkes point-process models: is there self-excitation beyond the circulation?
  5. How rare was 2025/26?
  6. Has clustering increased? Trends, split periods, moving windows, statistical power
  7. Robustness
  8. Figures and equation images
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import stormlib as sl  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
EQUATIONS = FIGURES / "equations"

SEED = 20260128
Q_MAIN = 0.96            # storm day: index above its 96th percentile over winter days
GAP_MAIN = 0             # storm days separated by one calm day or more are separate episodes
FIRST, LAST = 1950, 2025  # seasons (labelled by the October they start in)
TARGET = 2025            # the 2025/26 winter
N_SIM = 20000            # simulated winters per scenario
N_BOOT = 200             # simulated histories for the self-excitation check
N_POWER = 200            # simulated histories per effect size in the power analysis
if os.environ.get("STORM_QUICK"):  # smoke test only
    N_SIM, N_BOOT, N_POWER = 2000, 10, 5

# --- chart style (light surface, fixed categorical order) -------------------
SURFACE = "#ffffff"
INK = "#292929"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BAND = "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "axes.axisbelow": True,
})
DPI = 160

# Storms named by IPMA (or well documented before naming began) and the UTC day of
# their main impact on mainland Portugal. Sources are in data/known_storms.csv.
KNOWN = pd.read_csv(sl.DATA / "known_storms.csv", parse_dates=["date"])


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def season_label(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


# =============================================================================
# 1. Data and index
# =============================================================================

def build_index():
    section("1. Data and the storm severity index")
    era5, pts = sl.load_era5()
    era5 = era5.loc[f"{FIRST}-10-01":f"{LAST + 1}-03-31"]
    v98 = sl.thresholds(era5)
    daily = sl.severity_index(era5, v98)
    print(f"ERA5 nodes: {era5.shape[1]}, days {era5.index[0].date()} to {era5.index[-1].date()}, "
          f"missing values {era5.isna().sum().sum()}")
    print(f"98th percentile of daily maximum 10 m wind (1981-2010), across nodes: "
          f"median {v98.median():.1f} m/s, range {v98.min():.1f}-{v98.max():.1f} m/s")
    north = pts.lat >= 40.0
    print(f"  nodes north of 40N: {north.sum()}, south: {(~north).sum()}")

    st, meta = sl.load_stations()
    long3 = ["POMU0068545", "POMU0068554", "POI0000LPBJ"]
    names = dict(zip(meta.id, meta.name))
    stw = st[long3]
    sv98 = sl.thresholds(stw)
    sdaily = sl.severity_index(stw, sv98, min_frac=0.99)
    print("Stations used for the independent check (synoptic-hour maxima, >= 6 of 8 reports a day):")
    for c in long3:
        print(f"  {c} {names.get(c, '')}: 98th percentile {sv98[c]:.1f} m/s, "
              f"{stw[c].notna().sum()} valid days")
    synop = st.columns[st.loc["2015":"2022"].notna().mean() > 0.5]
    late = st.loc["2025-10-01":"2026-03-31"].notna().mean()
    print(f"Stations with >50% valid days in 2015-2022: {len(synop)}; "
          f"of those with >90% valid days in Oct 2025-Mar 2026: {(late[synop] > 0.9).sum()}")
    return era5, pts, v98, daily, stw, sdaily


# =============================================================================
# 2. Catalogue and validation
# =============================================================================

def catalogue(daily, q=Q_MAIN, gap=GAP_MAIN, first=FIRST, last=LAST, **kw):
    return sl.episodes(daily, q=q, gap=gap, first_season=first, last_season=last, **kw)


def validate(cat, daily, scat, sdaily):
    section("2. Episode catalogue and validation")
    ep = cat.episodes
    S = len(cat.seasons)
    print(f"Seasons {season_label(cat.seasons[0])} to {season_label(cat.seasons[-1])}: {S}")
    print(f"Storm-day threshold (96th percentile of the winter index): {cat.threshold:.2f}; "
          f"storm days per season {cat.rate_days:.2f}")
    print(f"Episodes: {len(ep)}, per season {len(ep) / S:.2f}; mean length {ep.n_days.mean():.2f} days; "
          f"share lasting one day {pct((ep.n_days == 1).mean())}")
    print("Twenty most severe episodes (severity = summed daily index):")
    top = ep.sort_values("severity", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"  {r.peak.date()}  {r.n_days} d  severity {r.severity:7.1f}  peak area {pct(r.peak_area)}")

    w = daily[sl.winter_days(daily.index)].ssi.dropna()
    print("Named and documented storms (rank of the day's index among all winter days since 1950):")
    hits = 0
    for _, k in KNOWN.iterrows():
        win = daily.loc[k.date - pd.Timedelta(days=1):k.date + pd.Timedelta(days=1), "ssi"]
        best = win.idxmax()
        val = win.max()
        rank_pct = (w < val).mean()
        in_ep = ((ep.start <= best) & (ep.end >= best)).any()
        hits += in_ep
        print(f"  {k['name']:<22} {k.date.date()}  max index {val:7.2f} on {best.date()}, "
              f"above {pct(rank_pct, 1)} of winter days; in an episode: {'yes' if in_ep else 'no'}")
    print(f"Documented storms inside an episode (+/- 1 day): {hits} of {len(KNOWN)}")

    # Cross-check with the three long station records
    both = pd.concat([daily.ssi.rename("era5"), sdaily.ssi.rename("st")], axis=1).dropna()
    both = both[sl.winter_days(both.index)]
    rho = stats.spearmanr(both.era5, both.st).statistic
    print(f"Winter days with both indices: {len(both)}; Spearman correlation {rho:.2f}")
    sep = scat.episodes
    top_st = sep.sort_values("severity", ascending=False).head(50)
    matched = sum(((ep.start - pd.Timedelta(days=1) <= r.peak) & (ep.end + pd.Timedelta(days=1) >= r.peak)).any()
                  for _, r in top_st.iterrows())
    print(f"Of the 50 most severe station episodes ({season_label(scat.seasons[0])} onwards), "
          f"{matched} coincide with an ERA5 episode (+/- 1 day)")
    common = np.intersect1d(cat.seasons, scat.seasons)
    c1 = ep.groupby("season").size().reindex(common, fill_value=0)
    c2 = sep.groupby("season").size().reindex(common, fill_value=0)
    r = stats.pearsonr(c1, c2).statistic
    print(f"Seasonal episode counts, ERA5 vs stations, {len(common)} common seasons: correlation {r:.2f}")
    return both


def describe_target(cat, daily):
    ep = cat.episodes
    cur = ep[ep.season == TARGET]
    print(f"Episodes in {season_label(TARGET)}: {len(cur)} (average {len(ep) / len(cat.seasons):.2f})")
    for _, r in cur.iterrows():
        near = KNOWN[(KNOWN.date >= r.start - pd.Timedelta(days=1)) & (KNOWN.date <= r.end + pd.Timedelta(days=1))]
        tag = ", ".join(near["name"]) if len(near) else ""
        rk = (ep.severity > r.severity).sum() + 1
        print(f"  {r.start.date()} to {r.end.date()}  peak {r.peak.date()}  severity {r.severity:7.1f} "
              f"(rank {rk} of {len(ep)})  {tag}")


# =============================================================================
# 3. Dispersion
# =============================================================================

def poisson_glm(y, X):
    """Poisson regression by iteratively reweighted least squares: (coef, se, deviance, pearson chi2)."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.mean(), 1e-6))
    for _ in range(100):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        W = mu
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta)
    cov = np.linalg.inv((X.T * mu) @ X)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
    chi2 = np.sum((y - mu) ** 2 / mu)
    return beta, np.sqrt(np.diag(cov)), dev, chi2, mu


def monthly_counts(cat, tele):
    ep = cat.episodes
    rows = []
    for s in cat.seasons:
        for m in sl.WINTER_MONTHS:
            y = s if m >= 10 else s + 1
            n = ((ep.peak.dt.year == y) & (ep.peak.dt.month == m)).sum()
            rows.append({"season": s, "year": y, "month": m, "n": n})
    df = pd.DataFrame(rows)
    df = df.join(tele, on=["year", "month"])
    return df


def dispersion_analysis(cat, tele, rng):
    section("3. Dispersion of storm counts and the circulation")
    ep = cat.episodes
    counts = ep.groupby("season").size().reindex(cat.seasons, fill_value=0)
    psi = sl.dispersion(counts.values)
    print(f"Episodes per season: mean {counts.mean():.2f}, variance {counts.var(ddof=1):.2f}, "
          f"dispersion psi = {psi:.2f}, Poisson test p = {sl.dispersion_test(counts.values):.2g}")
    print(f"Most active seasons: " + ", ".join(f"{season_label(s)} ({n})" for s, n in counts.sort_values(ascending=False).head(6).items()))
    print(f"Seasons with no episode: {(counts == 0).sum()}")
    print(f"{season_label(TARGET)} count {counts[TARGET]}: more than in "
          f"{pct((counts < counts[TARGET]).mean())} of seasons, at least as many as in "
          f"{pct((counts <= counts[TARGET]).mean())}")

    mc = monthly_counts(cat, tele)
    month_dummies = pd.get_dummies(mc.month.astype(int), prefix="m", drop_first=True).astype(float)
    base = np.column_stack([np.ones(len(mc)), month_dummies.values])
    y = mc.n.values.astype(float)
    b0, se0, dev0, chi0, mu0 = poisson_glm(y, base)
    df0 = len(y) - base.shape[1]
    print(f"Monthly counts ({len(y)} months): dispersion after month-of-season means only: "
          f"Pearson chi2/df = {chi0 / df0:.2f}")
    cands = ["NAO", "EA", "SCA", "EA_WR", "POL"]
    results = {}
    for c in cands:
        X = np.column_stack([base, mc[c].fillna(0).values])
        b, se, dev, chi, _ = poisson_glm(y, X)
        results[c] = (b[-1], se[-1], dev0 - dev)
        print(f"  + {c:<5}: coefficient {b[-1]:+.3f} (se {se[-1]:.3f}), rate ratio per unit "
              f"{math.exp(b[-1]):.2f}, deviance drop {dev0 - dev:.1f}")
    # forward selection by AIC
    chosen = []
    cur_dev = dev0
    while True:
        best = None
        for c in cands:
            if c in chosen:
                continue
            X = np.column_stack([base] + [mc[k].fillna(0).values for k in chosen + [c]])
            b, se, dev, chi, _ = poisson_glm(y, X)
            if cur_dev - dev > 2.0 and (best is None or dev < best[1]):
                best = (c, dev)
        if best is None:
            break
        chosen.append(best[0])
        cur_dev = best[1]
    X = np.column_stack([base] + [mc[k].fillna(0).values for k in chosen])
    b, se, dev, chi, mu = poisson_glm(y, X)
    dfm = len(y) - X.shape[1]
    print(f"Indices kept by AIC (forward selection): {', '.join(chosen)}")
    for k, c in enumerate(chosen):
        j = base.shape[1] + k
        print(f"  {c}: rate ratio per standard deviation {math.exp(b[j]):.2f} "
              f"(95% CI {math.exp(b[j] - 1.96 * se[j]):.2f}-{math.exp(b[j] + 1.96 * se[j]):.2f})")
    print(f"Monthly dispersion after the indices: Pearson chi2/df = {chi / dfm:.2f} "
          f"(p = {stats.chi2.sf(chi, dfm):.2g} against Poisson)")
    print(f"Share of the extra-Poisson variance of monthly counts removed by the indices: "
          f"{pct(1 - (chi / dfm - 1) / (chi0 / df0 - 1))}")
    # seasonal counts against the seasonal mean of the chosen indices
    seas = mc.groupby("season")[chosen].mean()
    Xs = np.column_stack([np.ones(len(counts))] + [seas.loc[counts.index, c].values for c in chosen])
    bs, ses, devs, chis, mus = poisson_glm(counts.values.astype(float), Xs)
    print(f"Seasonal counts after the seasonal-mean indices: Pearson chi2/df = {chis / (len(counts) - Xs.shape[1]):.2f}")
    print(f"{season_label(TARGET)} season means: " + ", ".join(f"{c} {seas.loc[TARGET, c]:+.2f}" for c in chosen)
          + f"; expected episodes given those {mus[list(counts.index).index(TARGET)]:.1f}, observed {counts[TARGET]}")
    return counts, mc, chosen


# =============================================================================
# 4. The self-exciting model
# =============================================================================

def day_data(cat, tele, covars, trend=False, gap=GAP_MAIN):
    des = sl.background_design(cat.seasons, tele, covars, trend_center=1988.0 if trend else None)
    return sl.build_days(cat.episodes, cat.seasons, des, gap), des


def season_rows(d):
    """Observed start indicators, one (1, T) array per season."""
    return [d.y[d.day_offset[i]:d.day_offset[i] + int(d.T[i])][None, :].astype(bool) for i in range(len(d.seasons))]


def pools(cat):
    return cat.episodes.n_days.values.astype(int), cat.episodes.severity.values


def simulate_history(f, designs, seasons, R, rng, pool, gap=GAP_MAIN, with_live=False):
    """R replicate histories of the given seasons: lists of (R, T) start (and live) arrays."""
    st, lv = [], []
    for s in seasons:
        a, _, c = sl.simulate(f, designs[s], int(s), R, rng, pool[0], pool[1], gap)
        st.append(a)
        lv.append(c)
    return (st, lv) if with_live else st


def lr_p(l0, l1, df=1, boundary=False):
    lr = max(0.0, 2 * (l0 - l1))
    p = stats.chi2.sf(lr, df)
    return lr, (0.5 * p if boundary else p)


def branching_ci(f):
    """Wald interval on the logit scale, mapped back."""
    from scipy.special import expit
    a, se = f.theta[f.p], f.se()[f.p]
    if not np.isfinite(se):
        return float("nan"), float("nan")
    return float(expit(a - 1.96 * se)), float(expit(a + 1.96 * se))


def timescale_ci(f):
    lb, se = f.theta[f.p + 1], f.se()[f.p + 1]
    if not np.isfinite(se) or se > 20:
        return float("nan"), float("nan")
    return math.exp(-(lb + 1.96 * se)), math.exp(-(lb - 1.96 * se))


def describe_fit(name, f):
    extra = ""
    if f.hawkes:
        lo, hi = branching_ci(f)
        tlo, thi = timescale_ci(f)
        extra += (f"; n = {f.branching_ratio():.2f} (95% CI {lo:.2f}-{hi:.2f}), "
                  f"time scale {f.timescale():.1f} d ({tlo:.1f}-{thi:.1f})")
    if f.frailty:
        extra += f"; winter-to-winter SD of the background multiplier {1 / math.sqrt(f.frailty_shape()):.2f}"
    print(f"  {name:<46} log-lik {-f.nll:9.2f}  AIC {f.aic:8.1f}{extra}")


def hawkes_analysis(cat, tele, chosen, rng, cat_daily_for_comparison=None):
    section("4. Storms that bring storms: a self-exciting model on the daily grid")
    d0, des0 = day_data(cat, tele, [])
    d1, des1 = day_data(cat, tele, chosen)
    print(f"Winter days {len(d1.y)}, of which days on which an episode could start {d1.live.sum()} "
          f"(the rest are inside an episode or the calm day after it)")
    m = {
        "P0": sl.fit(d0, False), "P1": sl.fit(d1, False), "P1f": sl.fit(d1, False, frailty=True),
        "H0": sl.fit(d0, True), "H1": sl.fit(d1, True), "H1f": sl.fit(d1, True, frailty=True),
    }
    labels = {"P0": "No excitation: seasonal cycle", "P1": "No excitation: + circulation indices",
              "P1f": "No excitation: + indices + stormy winters", "H0": "Self-exciting: seasonal cycle",
              "H1": "Self-exciting: + indices", "H1f": "Self-exciting: + indices + stormy winters"}
    for k, f in m.items():
        describe_fit(labels[k], f)
    print("  Background rate ratios per standard deviation of each index (self-exciting model with indices):")
    for j, nm in enumerate(chosen, start=3):
        b, se = m["H1"].theta[j], m["H1"].se()[j]
        print(f"    {nm}: {math.exp(b):.2f} (95% CI {math.exp(b - 1.96 * se):.2f}-{math.exp(b + 1.96 * se):.2f})")
    for a_, b_, df, bnd, what in (("P0", "P1", len(chosen), False, "circulation indices"),
                                  ("P1", "P1f", 1, True, "stormy winters beyond the indices"),
                                  ("P0", "H0", 2, True, "self-excitation, seasonal cycle only"),
                                  ("P1", "H1", 2, True, "self-excitation beyond the indices"),
                                  ("P1f", "H1f", 2, True, "self-excitation beyond indices and stormy winters"),
                                  ("H1", "H1f", 1, True, "stormy winters, given self-excitation")):
        lr, pv = lr_p(m[a_].nll, m[b_].nll, df=df, boundary=bnd)
        print(f"  LR, {what}: {lr:.1f}, p {'(chi-bar approx.) ' if bnd else ''}= {pv:.2g}")

    # Can winter-to-winter storminess alone mimic self-excitation? Simulate histories from
    # the model with indices and stormy winters but no excitation, and fit the models
    # without the frailty, which is where such confusion would show up.
    lr_obs = 2 * (m["P1"].nll - m["H1"].nll)
    pool = pools(cat)
    sts, lvs = simulate_history(m["P1f"], des1, cat.seasons, N_BOOT, rng, pool, with_live=True)
    boot, n_boot = [], []
    for r in range(N_BOOT):
        db = sl.from_arrays(cat.seasons, des1, [s[r] for s in sts], [l[r] for l in lvs])
        f0 = sl.fit(db, False, hessian=False)
        f1 = sl.fit(db, True, hessian=False, quick=True)
        boot.append(max(0.0, 2 * (f0.nll - f1.nll)))
        n_boot.append(f1.branching_ratio())
    boot = np.array(boot)
    p_boot = (1 + (boot >= lr_obs).sum()) / (1 + len(boot))
    print(f"  Simulation check ({N_BOOT} histories from the model with indices and stormy winters, no excitation): "
          f"LR of self-excitation {np.median(boot):.1f} median, {np.percentile(boot, 95):.1f} at the 95th percentile, "
          f"{boot.max():.1f} at most, against {lr_obs:.1f} observed; p = {p_boot:.3f}; branching ratio fitted to "
          f"the simulated histories: median {np.median(n_boot):.2f}, 95th percentile {np.percentile(n_boot, 95):.2f}")
    # For comparison: the textbook continuous-time model on peak times, without dead time
    for g in (0, 1):
        cg = cat if g == 0 else catalogue(cat_daily_for_comparison, gap=1)
        desg = sl.background_design(cg.seasons, tele, chosen)
        l0, l1, nc, tc = sl.fit_continuous(cg.episodes, cg.seasons, desg)
        lr_c, p_c = lr_p(-l0, -l1, df=2, boundary=True)
        print(f"  Continuous-time model on peak times, no dead time, episodes {'separated by one calm day' if g == 0 else 'merged across one calm day'}: "
              f"branching ratio {nc:.2f}, time scale {tc:.1f} d, LR {lr_c:.1f}, p = {p_c:.2g}")
    # Sub-monthly circulation: add the daily NAO averaged over the seven days before each day
    nao_d = sl.load_nao_daily()
    if nao_d is not None:
        nao7 = nao_d.rolling(7).mean().shift(1)
        des_d = sl.background_design(cat.seasons, tele, chosen, daily_cov=nao7)
        dd = sl.build_days(cat.episodes, cat.seasons, des_d, GAP_MAIN)
        pdd, hdd = sl.fit(dd, False), sl.fit(dd, True)
        lo, hi = branching_ci(hdd)
        lr_d, p_d = lr_p(pdd.nll, hdd.nll, df=2, boundary=True)
        b7 = pdd.theta[pdd.p - 1]
        print(f"  Adding the daily NAO (mean of the previous 7 days) to the background: rate ratio per unit "
              f"{math.exp(b7):.2f} without excitation; branching ratio {hdd.branching_ratio():.2f} "
              f"(95% CI {lo:.2f}-{hi:.2f}), time scale {hdd.timescale():.1f} d; LR {lr_d:.1f}, p = {p_d:.2g}")
    # Are follow-on episodes weaker or stronger than the rest?
    ep = cat.episodes.sort_values("start").reset_index(drop=True)
    prev_end = ep.groupby("season").end.shift(1)
    quick_follow = (ep.start - prev_end).dt.days <= 5
    a_sev, b_sev = ep.severity[quick_follow], ep.severity[~quick_follow]
    u = stats.mannwhitneyu(a_sev, b_sev)
    print(f"  Episodes starting within 5 days of the end of the previous one: {quick_follow.sum()} of {len(ep)}; "
          f"median severity {a_sev.median():.1f} against {b_sev.median():.1f} for the others "
          f"(Mann-Whitney p = {u.pvalue:.2f}); share in the top quarter {pct((a_sev >= ep.severity.quantile(0.75)).mean())} "
          f"against {pct((b_sev >= ep.severity.quantile(0.75)).mean())}")
    n = m["H1"].branching_ratio()
    print(f"  Branching ratio {n:.2f}: about {pct(n)} of episodes are triggered by an earlier one; "
          f"a background episode starts a family of {1 / (1 - n):.2f} episodes on average")
    tau = m["H1"].timescale()
    for days in (3, 7, 14):
        print(f"    share of the triggered episodes expected within {days} days of their parent: "
              f"{pct(1 - math.exp(-days / tau))}")
    fits = {"poisson": m["P1f"], "hawkes": m["H1"], **m}
    return fits, d1, des1, p_boot


def predictive_checks(cat, fits, d1, des1, rng, max_lag=30, reps=400):
    """Pairs of episode starts by lag, and dispersion of counts by block length: observed,
    and in histories simulated from the no-excitation and self-exciting models."""
    pool = pools(cat)
    obs_rows = season_rows(d1)
    sims = {k: simulate_history(fits[k], des1, cat.seasons, reps, rng, pool) for k in ("poisson", "hawkes")}

    def pairs(rows):
        R = rows[0].shape[0]
        h = np.zeros((R, max_lag + 1))
        for st in rows:
            st = st.astype(np.int32)
            for L in range(1, max_lag + 1):
                h[:, L] += np.sum(st[:, :-L] * st[:, L:], axis=1)
        return h

    obs = pairs(obs_rows)[0]
    sp = {k: pairs(v) for k, v in sims.items()}
    base = sp["poisson"].mean(axis=0)
    print("Pairs of episode starts by separation, observed and expected (no excitation: indices and stormy winters):")
    for lo, hi in ((2, 5), (5, 10), (10, 20), (20, 31)):
        o, e, h = obs[lo:hi].sum(), base[lo:hi].sum(), sp["hawkes"][:, lo:hi].sum(axis=1).mean()
        share = (sp["poisson"][:, lo:hi].sum(axis=1) >= o).mean()
        print(f"  {lo:>2}-{hi - 1:>2} days: observed {o:.0f}; no excitation {e:.1f} (ratio {o / e:.2f}, "
              f"histories with as many {pct(share, 1)}); self-exciting {h:.1f}")

    lengths = [7, 14, 30, 61, 91, 182]

    def disp(rows, L):
        cs = []
        for st in rows:
            nb = st.shape[1] // L
            if nb:
                cs.append(st[:, :nb * L].reshape(st.shape[0], nb, L).sum(axis=2))
        C = np.concatenate(cs, axis=1).astype(float)
        return C.var(axis=1, ddof=1) / C.mean(axis=1) - 1

    dobs = np.array([disp(obs_rows, L)[0] for L in lengths])
    dsim = {k: np.column_stack([disp(v, L) for L in lengths]) for k, v in sims.items()}
    print("Dispersion of episode counts by block length (observed; 90% ranges without and with excitation):")
    for i, L in enumerate(lengths):
        p5, p95 = np.percentile(dsim["poisson"][:, i], [5, 95])
        h5, h95 = np.percentile(dsim["hawkes"][:, i], [5, 95])
        print(f"  {L:>3} days: {dobs[i]:+.2f}   no excitation {p5:+.2f} to {p95:+.2f}   self-exciting {h5:+.2f} to {h95:+.2f}")
    # chance of a follow-on episode
    def follow(rows, k):
        n_ev, n_f = 0, 0
        for st in rows:
            T = st.shape[1]
            for dd in range(T):
                ev = st[:, dd]
                if not ev.any():
                    continue
                nxt = st[:, dd + 1:min(T, dd + 1 + k)].any(axis=1)
                n_ev = n_ev + ev.astype(int)
                n_f = n_f + (ev & nxt).astype(int)
        return n_f / np.maximum(n_ev, 1)

    for k in (5, 10):
        fo = follow(obs_rows, k)[0]
        fp = follow(sims["poisson"], k)
        fh = follow(sims["hawkes"], k)
        print(f"Share of episodes followed by another within {k} days: observed {pct(fo)}; "
              f"no excitation {pct(np.mean(fp))} (90% range {pct(np.percentile(fp, 5))}-{pct(np.percentile(fp, 95))}); "
              f"self-exciting {pct(np.mean(fh))} ({pct(np.percentile(fh, 5))}-{pct(np.percentile(fh, 95))})")
    return obs, sp, base, lengths, dobs, dsim


# =============================================================================
# 5. How rare was 2025/26?
# =============================================================================

def metrics(starts, sev, strong_thr):
    strong = starts & (sev >= strong_thr)
    return pd.DataFrame({
        "N": starts.sum(axis=1), "K21": sl.window_max(starts.astype(float), 21),
        "K30": sl.window_max(starts.astype(float), 30), "S30": sl.window_max(sev, 30),
        "SEV": sev.sum(axis=1), "N_strong": strong.sum(axis=1),
        "K30_strong": sl.window_max(strong.astype(float), 30), "K14_strong": sl.window_max(strong.astype(float), 14),
    })


def observed_matrix(cat, s):
    T = sl.season_length(s)
    st = np.zeros((1, T), bool)
    sv = np.zeros((1, T))
    for _, r in cat.episodes[cat.episodes.season == s].iterrows():
        i = (r.start - pd.Timestamp(s, 10, 1)).days
        st[0, i] = True
        sv[0, i] = r.severity
    return st, sv


LABELS = {"N": "episodes in the winter", "K21": "most episodes in any 21 days",
          "K30": "most episodes in any 30 days", "S30": "largest 30-day severity",
          "SEV": "winter total severity", "N_strong": "strong episodes in the winter",
          "K30_strong": "most strong episodes in any 30 days", "K14_strong": "most strong episodes in any 14 days"}


def rarity(cat, fits, des1, rng):
    section("5. How rare was 2025/26?")
    ep = cat.episodes
    strong_thr = float(np.quantile(ep.severity, 0.75))
    print(f"Strong episode: severity in the top quarter of all episodes (at least {strong_thr:.1f})")
    obs = pd.concat([metrics(*observed_matrix(cat, s), strong_thr).assign(season=s) for s in cat.seasons]).set_index("season")
    tgt = obs.loc[TARGET]
    for k, lab in LABELS.items():
        v = tgt[k]
        rank = (obs[k] > v).sum() + 1
        ties = (obs[k] == v).sum() - 1
        val = f"{v:.0f}" if k not in ("S30", "SEV") else f"{v:.1f}"
        top = ", ".join(f"{season_label(s)} ({x:.0f})" if k not in ("S30", "SEV") else f"{season_label(s)} ({x:.1f})"
                        for s, x in obs[k].sort_values(ascending=False).head(4).items())
        print(f"  {lab}: {val}; rank {rank} of {len(obs)}" + (f" (tied with {ties})" if ties else "")
              + f"; at least as high in {(obs[k] >= v).sum()} winters; highest: {top}")
    pool = pools(cat)
    out = {}
    for model in ("poisson", "hawkes"):
        f = fits[model]
        st, sv, _ = sl.simulate(f, des1[TARGET], TARGET, N_SIM, rng, pool[0], pool[1], GAP_MAIN)
        out[(model, "conditional")] = metrics(st, sv, strong_thr)
        draws = rng.choice(cat.seasons, N_SIM)
        parts = []
        for s in np.unique(draws):
            c = int((draws == s).sum())
            st, sv, _ = sl.simulate(f, des1[s], int(s), c, rng, pool[0], pool[1], GAP_MAIN)
            parts.append(metrics(st, sv, strong_thr))
        out[(model, "any")] = pd.concat(parts, ignore_index=True)
    print(f"Return periods in winters, from {N_SIM:,} simulated winters per scenario "
          "(any: circulation of a random past winter; 2025/26: the circulation of 2025/26):")
    rp = {}
    for k, lab in LABELS.items():
        parts = []
        for model, nm in (("poisson", "no excitation"), ("hawkes", "self-exciting")):
            for cond, cn in (("any", "any"), ("conditional", "2025/26")):
                pr = (out[(model, cond)][k] >= tgt[k]).mean()
                rp[(k, model, cond)] = 1 / pr if pr > 0 else float("inf")
                parts.append(f"{nm}/{cn} {rp[(k, model, cond)]:.0f}")
        print(f"  {lab}: " + ", ".join(parts))
    print("Winters with at least three episodes within w days, and with at least three strong ones "
          "(observed; self-exciting model, any circulation; no-excitation model):")
    hk, pk = out[("hawkes", "any")], out[("poisson", "any")]
    sims_w = {}
    for model in ("poisson", "hawkes"):
        draws = rng.choice(cat.seasons, 4000)
        acc_all, acc_str = [], []
        for s in np.unique(draws):
            c = int((draws == s).sum())
            st, sv, _ = sl.simulate(fits[model], des1[s], int(s), c, rng, pool[0], pool[1], GAP_MAIN)
            acc_all.append(st)
            acc_str.append(st & (sv >= strong_thr))
        sims_w[model] = (acc_all, acc_str)
    for w in (4, 6, 7, 14, 28, 182):
        o_all = np.array([sl.window_max(observed_matrix(cat, s)[0].astype(float), w)[0] >= 3 for s in cat.seasons])
        o_str = np.array([sl.window_max((lambda a: (a[0] & (a[1] >= strong_thr)).astype(float))(observed_matrix(cat, s)), w)[0] >= 3
                          for s in cat.seasons])
        res = []
        for model in ("hawkes", "poisson"):
            a_all = np.concatenate([sl.window_max(x.astype(float), w) for x in sims_w[model][0]]) >= 3
            a_str = np.concatenate([sl.window_max(x.astype(float), w) for x in sims_w[model][1]]) >= 3
            res.append((a_all.mean(), a_str.mean()))
        fmt = lambda pr: f"once in {1 / pr:.1f}" if pr > 0 else "never"
        print(f"  {w:>3} days: any three {o_all.sum()} of {len(o_all)} winters ({fmt(o_all.mean())}); "
              f"self-exciting {fmt(res[0][0])}; no excitation {fmt(res[1][0])} | strong three {o_str.sum()} winters "
              f"({fmt(o_str.mean())}); self-exciting {fmt(res[0][1])}; no excitation {fmt(res[1][1])}; "
              f"2025/26: {'yes' if o_all[list(cat.seasons).index(TARGET)] else 'no'}/{'yes' if o_str[list(cat.seasons).index(TARGET)] else 'no'}")
    return obs, out, rp, strong_thr


# =============================================================================
# 6. Trends
# =============================================================================

def trend_analysis(cat, tele, chosen, label="ERA5"):
    section(f"6. Has clustering increased? ({label}, {season_label(cat.seasons[0])} to {season_label(cat.seasons[-1])})")
    counts = cat.episodes.groupby("season").size().reindex(cat.seasons, fill_value=0)
    years = cat.seasons.astype(float)
    Xs = np.column_stack([np.ones(len(years)), (years - 1988) / 10])
    b, se, dev, chi, mu = poisson_glm(counts.values.astype(float), Xs)
    print(f"Episodes per winter, Poisson trend: {100 * (math.exp(b[1]) - 1):+.1f}% per decade "
          f"(95% CI {100 * (math.exp(b[1] - 1.96 * se[1]) - 1):+.1f}% to {100 * (math.exp(b[1] + 1.96 * se[1]) - 1):+.1f}%)")
    tau, p = sl.mann_kendall(counts.values)
    print(f"Mann-Kendall on episodes per winter: tau {tau:+.3f}, p = {p:.2f}")
    d_t, des_t = day_data(cat, tele, chosen, trend=True)
    f_bg = sl.fit(d_t, True)
    f_both = sl.fit(d_t, True, trend_n=True)
    j = f_bg.p - 1
    print(f"Self-exciting model with a trend in the background: {100 * (math.exp(f_bg.theta[j]) - 1):+.1f}% per decade "
          f"(95% CI {100 * (math.exp(f_bg.theta[j] - 1.96 * f_bg.se()[j]) - 1):+.1f}% to "
          f"{100 * (math.exp(f_bg.theta[j] + 1.96 * f_bg.se()[j]) - 1):+.1f}%); branching ratio {f_bg.branching_ratio():.2f}")
    delta, dse = f_both.theta[f_both.p + 2], f_both.se()[f_both.p + 2]
    lr, pv = lr_p(f_bg.nll, f_both.nll, df=1)
    print(f"Adding a trend in the branching ratio: logit change per decade {delta:+.2f} (se {dse:.2f}); "
          f"LR {lr:.2f}, p = {pv:.2f}; implied branching ratio {season_label(int(cat.seasons[0]))} "
          f"{f_both.branching_ratio(cat.seasons[0]):.2f}, {season_label(int(cat.seasons[-1]))} "
          f"{f_both.branching_ratio(cat.seasons[-1]):.2f}")
    half = int(np.median(cat.seasons))
    halves = {}
    for nm, sel in (("first half", cat.seasons <= half), ("second half", cat.seasons > half)):
        ss = cat.seasons[sel]
        sub = sl.Catalogue(cat.episodes[cat.episodes.season.isin(ss)], cat.days, cat.threshold, cat.rate_days, ss)
        d, _ = day_data(sub, tele, chosen)
        f = sl.fit(d, True)
        lo, hi = branching_ci(f)
        halves[nm] = (f, lo, hi)
        print(f"  {nm} ({season_label(int(ss[0]))} to {season_label(int(ss[-1]))}): branching ratio "
              f"{f.branching_ratio():.2f} (95% CI {lo:.2f}-{hi:.2f}), time scale {f.timescale():.1f} days, "
              f"{len(sub.episodes)} episodes")
    return f_bg, f_both, halves, d_t, des_t


def moving_windows(cat, tele, chosen, width=31):
    out = []
    for i in range(len(cat.seasons) - width + 1):
        ss = cat.seasons[i:i + width]
        sub = sl.Catalogue(cat.episodes[cat.episodes.season.isin(ss)], cat.days, cat.threshold, cat.rate_days, ss)
        d, _ = day_data(sub, tele, chosen)
        f = sl.fit(d, True)
        lo, hi = branching_ci(f)
        c = sub.episodes.groupby("season").size().reindex(ss, fill_value=0)
        out.append({"center": int(ss[width // 2]), "n": f.branching_ratio(), "lo": lo, "hi": hi,
                    "psi": sl.dispersion(c.values), "rate": c.mean()})
    df = pd.DataFrame(out)
    print(f"Moving {width}-winter windows: branching ratio between {df.n.min():.2f} and {df.n.max():.2f}; "
          f"window centred on {df.center.iloc[0]}: {df.n.iloc[0]:.2f}; centred on {df.center.iloc[-1]}: {df.n.iloc[-1]:.2f}")
    print(f"  dispersion of episodes per winter between {df.psi.min():+.2f} and {df.psi.max():+.2f} across windows")
    return df


def power_analysis(cat, f_bg, des_t, rng):
    """How often the branching-trend test (LR, 5% level) detects a trend of a given size
    in a record as long as this one, simulating from the fitted model."""
    section("6b. Statistical power for a change in clustering")
    pool = pools(cat)
    n_mid = f_bg.branching_ratio()
    a_mid = math.log(n_mid / (1 - n_mid))
    span = (cat.seasons[-1] - cat.seasons[0]) / 10.0
    results = []
    for change in (0.0, 0.1, 0.2, 0.3, 0.4):
        lo_n, hi_n = max(0.01, n_mid - change / 2), min(0.85, n_mid + change / 2)
        delta = (math.log(hi_n / (1 - hi_n)) - math.log(lo_n / (1 - lo_n))) / span
        theta = np.concatenate([f_bg.theta[:f_bg.p], [a_mid, f_bg.theta[f_bg.p + 1], delta]])
        truth = sl.Fit(theta, 0.0, f_bg.p, True, True, 1988.0, None)
        sts, lvs = simulate_history(truth, des_t, cat.seasons, N_POWER, rng, pool, with_live=True)
        rej = 0
        for r in range(N_POWER):
            db = sl.from_arrays(cat.seasons, des_t, [s[r] for s in sts], [l[r] for l in lvs])
            f0 = sl.fit(db, True, hessian=False, quick=True)
            f1 = sl.fit(db, True, trend_n=True, hessian=False, quick=True)
            rej += 2 * (f0.nll - f1.nll) > stats.chi2.ppf(0.95, 1)
        results.append((change, lo_n, hi_n, rej / N_POWER))
        print(f"  branching ratio {lo_n:.2f} -> {hi_n:.2f} across the record: detected in {pct(rej / N_POWER)} "
              f"of {N_POWER} simulated records")
    return results


# =============================================================================
# 7. Robustness
# =============================================================================

def robustness(era5, tele, chosen, rng, scat):
    section("7. Robustness")
    v98 = sl.thresholds(era5)
    full = sl.severity_index(era5, v98)
    north = [c for c in era5.columns if float(c.split("_")[0]) >= 40.0]
    south = [c for c in era5.columns if float(c.split("_")[0]) < 40.0]
    variants = [
        ("main: 96th percentile, gap 0, all nodes, 1950-2025", catalogue(full), GAP_MAIN),
        ("threshold at the 94th percentile", catalogue(full, q=0.94), GAP_MAIN),
        ("threshold at the 98th percentile", catalogue(full, q=0.98), GAP_MAIN),
        ("episodes merged across one calm day", catalogue(full, gap=1), 1),
        ("episodes merged across two calm days", catalogue(full, gap=2), 2),
        ("northern nodes only (40N and above)", catalogue(sl.severity_index(era5[north], v98[north])), GAP_MAIN),
        ("southern nodes only (below 40N)", catalogue(sl.severity_index(era5[south], v98[south])), GAP_MAIN),
        ("satellite era only (1979/80 onwards)", catalogue(full, first=1979), GAP_MAIN),
        ("stations Porto, Faro, Beja (1973/74 onwards)", scat, GAP_MAIN),
    ]
    rows = []
    for name, cat, gap in variants:
        rows.append(robust_row(name, cat, tele, chosen, gap, rng))
    print("variant | episodes per winter | psi | n (95% CI) | time scale | p, self-excitation | "
          "branching trend per decade (p) | 2025/26 strong in 30 d | return period, self-exciting")
    for r in rows:
        print("  " + " | ".join(str(x) for x in r))
    return rows


def robust_row(name, cat, tele, chosen, gap, rng):
    counts = cat.episodes.groupby("season").size().reindex(cat.seasons, fill_value=0)
    d1, des1 = day_data(cat, tele, chosen, gap=gap)
    m1 = sl.fit(d1, False, hessian=False)
    m3 = sl.fit(d1, True)
    lr, p = lr_p(m1.nll, m3.nll, df=2, boundary=True)
    lo, hi = branching_ci(m3)
    d_t, _ = day_data(cat, tele, chosen, trend=True, gap=gap)
    fb = sl.fit(d_t, True, hessian=False)
    ft = sl.fit(d_t, True, trend_n=True)
    lr_t, p_t = lr_p(fb.nll, ft.nll, df=1)
    strong_thr = float(np.quantile(cat.episodes.severity, 0.75))
    if TARGET in cat.seasons:
        st, sv = observed_matrix(cat, TARGET)
        k30 = int(sl.window_max((st & (sv >= strong_thr)).astype(float), 30)[0])
        pool = pools(cat)
        draws = rng.choice(cat.seasons, 4000)
        hits = 0
        for s in np.unique(draws):
            c = int((draws == s).sum())
            a, b, _ = sl.simulate(m3, des1[s], int(s), c, rng, pool[0], pool[1], gap)
            hits += (sl.window_max((a & (b >= strong_thr)).astype(float), 30) >= k30).sum()
        rp = f"{4000 / hits:.0f} winters" if hits else "> 4000 winters"
    else:
        k30, rp = "n/a", "n/a"
    return [name, f"{counts.mean():.2f}", f"{sl.dispersion(counts.values):+.2f}",
            f"{m3.branching_ratio():.2f} ({lo:.2f}-{hi:.2f})", f"{m3.timescale():.1f} d", f"{p:.2g}",
            f"{ft.theta[ft.p + 2]:+.2f} ({p_t:.2f})", f"{k30}", rp]


# =============================================================================
# 8. Figures
# =============================================================================

def fig_season(daily, sdaily, cat):
    a, b = pd.Timestamp(TARGET, 10, 1), pd.Timestamp(TARGET + 1, 3, 31)
    d = daily.loc[a:b]
    s = sdaily.loc[a:b]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    ax = axes[0]
    for _, r in cat.episodes[cat.episodes.season == TARGET].iterrows():
        ax.axvspan(r.start - pd.Timedelta(hours=12), r.end + pd.Timedelta(hours=12), color=BAND, lw=0, zorder=0)
    ax.bar(d.index, d.ssi, width=0.8, color=BLUE, label="ERA5, 38 nodes")
    ax.axhline(cat.threshold, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(a + pd.Timedelta(days=2), cat.threshold * 1.15, "storm-day threshold", fontsize=9, color=INK_2, va="bottom")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, max(d.ssi.max(), 1) * 3)
    ax.set_ylabel("Storm severity index")
    ax.set_title(f"Daily storm severity over mainland Portugal, winter {season_label(TARGET)}", loc="left", fontsize=12)
    # label positions: (days to shift the text, height of the text as a fraction of the axis)
    place = {"Ingrid": (-16, 0.42), "Joseph": (-15, 0.62), "Kristin": (-14, 0.88),
             "Leonardo": (12, 0.82), "Marta": (14, 0.64)}
    k = KNOWN[(KNOWN.date >= a) & (KNOWN.date <= b)]
    for _, r in k.iterrows():
        y = max(d.ssi.get(r.date, 0), 0.05)
        dx, fy = place.get(r["name"], (0, 0.9))
        ax.annotate(r["name"], xy=(r.date, y), xytext=(r.date + pd.Timedelta(days=dx), fy),
                    textcoords=("data", "axes fraction"), ha="right" if dx < 0 else "left", va="center",
                    fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6,
                                                            shrinkA=2, shrinkB=1))
    bx = axes[1]
    bx.bar(s.index, s.ssi, width=0.8, color=ORANGE, label="Stations: Porto, Faro, Beja")
    bx.set_yscale("symlog", linthresh=1)
    bx.set_ylabel("Station index")
    bx.set_title("Same index from three airport stations (synoptic hours)", loc="left", fontsize=10, color=INK_2)
    bx.xaxis.set_major_locator(mdates.MonthLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIGURES / "01_winter_2025_26.png", dpi=DPI)
    plt.close(fig)


def fig_counts(counts, mc, chosen):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.3]})
    ax = axes[0]
    x = counts.index.values
    colors = [ORANGE if s == TARGET else BLUE for s in x]
    ax.bar(x, counts.values, width=0.75, color=colors)
    m = counts.mean()
    lo, hi = stats.poisson.ppf([0.05, 0.95], m)
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.axhline(m, color=INK_2, lw=1)
    ax.text(x[0] - 0.5, hi + 0.2, "90% range if counts were Poisson", fontsize=9, color=INK_2, va="bottom")
    ax.set_ylabel("Storm episodes")
    ax.set_title("Storm episodes per winter (October to March), mainland Portugal, ERA5", loc="left", fontsize=12)
    ax.annotate(season_label(TARGET), (TARGET, counts[TARGET]), xytext=(-30, 8), textcoords="offset points",
                fontsize=9, color=INK, ha="right", arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6))
    bx = axes[1]
    nao = mc.groupby("season")["NAO"].mean().reindex(x)
    bx.bar(x, nao.values, width=0.75, color=[AQUA if v >= 0 else YELLOW for v in nao.values])
    bx.axhline(0, color=INK_2, lw=0.8)
    bx.set_ylabel("NAO, Oct-Mar mean")
    bx.set_title("North Atlantic Oscillation (negative favours storms reaching Iberia)", loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_episodes_per_winter.png", dpi=DPI)
    plt.close(fig)


def fig_dispersion(lengths, obs, sims):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(lengths))
    for name, color, off, label in (("poisson", BLUE, -0.12, "No excitation: circulation and stormy winters"),
                                    ("hawkes", ORANGE, 0.12, "Self-exciting: circulation and storm families")):
        lo, med, hi = np.percentile(sims[name], [5, 50, 95], axis=0)
        ax.vlines(x + off, lo, hi, color=color, lw=6, alpha=0.35)
        ax.plot(x + off, med, "o", color=color, ms=6, label=label + " (median, 90% range)")
    ax.plot(x, obs, "D", color=INK, ms=7, label="Observed", zorder=5)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L} days" if L < 182 else "whole winter" for L in lengths])
    ax.set_xlabel("Length of the counting block")
    ax.set_ylabel("Dispersion ψ = variance / mean − 1")
    ax.set_title("Over-dispersion of storm counts at every time scale", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_dispersion_by_block.png", dpi=DPI)
    plt.close(fig)


def fig_pairs(obs, sims, base):
    lags = np.arange(len(obs))
    sel = lags >= 2
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for name, color, label in (("poisson", BLUE, "No excitation: circulation and stormy winters"),
                               ("hawkes", ORANGE, "Self-exciting: circulation and storm families")):
        r = sims[name][:, sel] / base[sel]
        lo, med, hi = np.percentile(r, [5, 50, 95], axis=0)
        ax.fill_between(lags[sel], lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(lags[sel], med, color=color, lw=2, label=f"{label} (median, 90% range)")
    k = 3
    bins = np.arange(2, len(obs), k)
    ob = [obs[b:b + k].sum() / base[b:b + k].sum() for b in bins]
    ax.plot(bins + (k - 1) / 2, ob, "D", color=INK, ms=7, label="Observed (3-day bins)", zorder=5)
    ax.axhline(1, color=INK_2, lw=0.8)
    ax.set_xlabel("Days between the starts of two storm episodes in the same winter")
    ax.set_ylabel("Pairs relative to no excitation")
    ax.set_title("Storm episodes follow each other within days more often than the circulation explains",
                 loc="left", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_pair_excess.png", dpi=DPI)
    plt.close(fig)


def fig_intensity(fits, d):
    f = fits["hawkes"]
    bg, tot, y, live = sl.intensity_path(f, d, TARGET)
    dates = pd.date_range(pd.Timestamp(TARGET, 10, 1), periods=len(bg), freq="D")
    p_bg, p_tot = -np.expm1(-bg), -np.expm1(-tot)
    p_tot = np.where(live, p_tot, np.nan)   # no episode can start on a dead day
    fig, ax = plt.subplots(figsize=(11, 4.6))
    dead = ~live
    for dd in np.flatnonzero(dead):
        ax.axvspan(dates[dd] - pd.Timedelta(hours=12), dates[dd] + pd.Timedelta(hours=12), color=BAND, lw=0, zorder=0)
    ax.fill_between(dates, 0, 100 * p_bg, step="mid", color=BLUE, alpha=0.3, lw=0,
                    label="Background: season and circulation")
    ax.step(dates, 100 * p_tot, where="mid", color=ORANGE, lw=1.6, label="Background plus excitation by earlier episodes")
    ev = dates[y == 1]
    ax.plot(ev, np.full(len(ev), -1.2), "^", color=INK, ms=8, label="Episode starts", clip_on=False)
    ax.set_ylim(-2.5, None)
    ax.set_ylabel("Chance a new episode starts that day (%)")
    ax.set_title(f"Fitted daily chance of a new storm episode, winter {season_label(TARGET)}", loc="left", fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_intensity_2025_26.png", dpi=DPI)
    plt.close(fig)


def fig_rarity(obs, out, keys):
    fig, axes = plt.subplots(1, len(keys), figsize=(11.5, 4.8))
    scen = ((("poisson", "any"), BLUE, "-", "o", "No excitation, any winter"),
            (("hawkes", "any"), ORANGE, "-", "s", "Self-exciting, any winter"),
            (("hawkes", "conditional"), ORANGE, (0, (4, 2)), "^", "Self-exciting, 2025/26 circulation"))
    for ax, key in zip(axes, keys):
        discrete = key not in ("S30", "SEV")
        tv = obs.loc[TARGET, key]
        if discrete:
            ms = np.arange(1, int(max(obs[key].max(), tv)) + 2)
            for (model, cond), color, ls, mk, label in scen:
                v = out[(model, cond)][key].values
                rp = [1 / (v >= m).mean() if (v >= m).mean() > 0 else np.nan for m in ms]
                ax.plot(ms, rp, color=color, ls=ls, lw=1.6, marker=mk, ms=6, label=label)
            orp = [len(obs) / (obs[key] >= m).sum() if (obs[key] >= m).sum() else np.nan for m in ms]
            ax.plot(ms, orp, "D", color=INK, ms=7, mfc=SURFACE, mew=1.5, label="Past winters (empirical)", zorder=5)
            ax.set_xticks(ms)
            ax.set_xlabel("At least this many in a window")
        else:
            for (model, cond), color, ls, mk, label in scen:
                v = np.sort(out[(model, cond)][key].values)
                exc = 1 - np.arange(len(v)) / len(v)
                ax.plot(v, 1 / exc, color=color, ls=ls, lw=1.8, label=label)
            ov = np.sort(obs[key].values)
            oexc = 1 - np.arange(len(ov)) / len(ov)
            ax.plot(ov, 1 / oexc, "D", color=INK, ms=5, mfc=SURFACE, mew=1.2, label="Past winters (empirical)", zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(1, max(ov.max(), tv) * 2)
            ax.set_xlabel("Summed severity index")
        ax.axvline(tv, color=INK, lw=1, ls=(0, (2, 2)))
        ax.text(tv, 1500, f" {season_label(TARGET)}", fontsize=9, color=INK, va="top")
        ax.set_yscale("log")
        ax.set_ylim(1, 3000)
        ax.set_title(LABELS[key][0].upper() + LABELS[key][1:], loc="left", fontsize=11)
    axes[0].set_ylabel("Return period (winters)")
    axes[0].legend(loc="upper left", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_return_periods.png", dpi=DPI)
    plt.close(fig)


def fig_trend(mw, power, span_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.fill_between(mw.center, mw.lo, mw.hi, color=ORANGE, alpha=0.18, lw=0, label="95% interval")
    ax.plot(mw.center, mw.n, color=ORANGE, lw=2, label="Branching ratio, 31-winter window")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Central winter of the window")
    ax.set_ylabel("Branching ratio n")
    ax.set_title("Storm families over time", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9.5)
    bx = axes[1]
    ch = [f"{lo:.2f}→{hi:.2f}" for _, lo, hi, _ in power]
    pw = [p for *_, p in power]
    bx.bar(range(len(pw)), pw, color=[INK_2 if i == 0 else BLUE for i in range(len(pw))], width=0.55)
    for i, p in enumerate(pw):
        bx.text(i, p + 0.02, pct(p), ha="center", fontsize=9.5, color=INK)
    bx.axhline(0.8, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    bx.text(len(pw) - 0.5, 0.81, "80% power", fontsize=8.5, color=INK_2, ha="right", va="bottom")
    bx.set_xticks(range(len(pw)))
    bx.set_xticklabels(ch, fontsize=9.5)
    bx.set_ylim(0, 1)
    bx.set_xlabel(f"True change in n, {span_label}")
    bx.set_ylabel("Chance the test detects it")
    bx.set_title("How large a change the record can detect", loc="left", fontsize=12)
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_trend_and_power.png", dpi=DPI)
    plt.close(fig)


EQUATION_TEX = {
    "eq01_severity": r"$I_d = \dfrac{1000}{N}\ \sum_{i=1}^{N}\ \max\!\left(0,\ \dfrac{v_{i,d}}{v_{i,98}} - 1\right)^{3}$",
    "eq02_dispersion": r"$\psi = \dfrac{\mathrm{Var}(N)}{\mathrm{E}(N)} - 1$",
    "eq03_background": r"$\log \mu_d = \gamma_0 + \gamma_1 \cos\dfrac{2\pi\, t_d}{365.25} + \gamma_2 \sin\dfrac{2\pi\, t_d}{365.25} + \sum_{j} \beta_j\, z_j(d)$",
    "eq04_rate": r"$\lambda_d = Z_s\,\mu_d + n\,(1 - e^{-1/\tau}) \sum_{d' < d} e^{-(d - d' - 1)/\tau}\, y_{d'}$",
    "eq05_probability": r"$\Pr(y_d = 1) = 1 - e^{-\lambda_d}\quad \mathrm{on\ live\ days}$",
    "eq06_loglik": r"$\ell = \sum_{d\ \mathrm{live}} \left[\, y_d \log\!\left(1 - e^{-\lambda_d}\right) - (1 - y_d)\, \lambda_d \,\right]$",
    "eq07_frailty": r"$L_s = \int_0^{\infty} \prod_{d \in s} p_d(z)\ \dfrac{k^k z^{k-1} e^{-kz}}{\Gamma(k)}\, dz,\qquad \mathrm{E}\,Z = 1,\ \ \mathrm{Var}\,Z = 1/k$",
    "eq08_cluster": r"$\mathrm{E}[\,\mathrm{family\ size}\,] = 1 + n + n^2 + \cdots = \dfrac{1}{1-n}$",
    "eq09_return": r"$T(m) = \dfrac{1}{\Pr(M \geq m)}\ \ \mathrm{winters}$",
    "eq10_trend": r"$n(s) = \dfrac{1}{1 + e^{-(a + \delta\,(s - 1988)/10)}}$",
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
            if box.width > 0.94 * width_in * dpi:  # too wide for the canvas: shrink to fit
                size = size * 0.94 * width_in * dpi / box.width
                text.set_fontsize(size)
                fig.canvas.draw()
                box = text.get_window_extent()
        height_in = box.height / dpi + 0.32
        fig.set_size_inches(width_in, height_in)
        fig.savefig(EQUATIONS / f"{name}.png", dpi=dpi, facecolor=SURFACE)
        plt.close(fig)


# =============================================================================

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("Storm clustering in Portugal: printed results")
    print("=============================================")
    era5, pts, v98, daily, stw, sdaily = build_index()
    tele = sl.load_teleconnections()
    cat = catalogue(daily)
    scat = sl.episodes(sdaily, q=Q_MAIN, gap=GAP_MAIN, first_season=1973, last_season=LAST)
    validate(cat, daily, scat, sdaily)
    describe_target(cat, daily)
    counts, mc, chosen = dispersion_analysis(cat, tele, rng)
    fits, d1, des1, p_boot = hawkes_analysis(cat, tele, chosen, rng, cat_daily_for_comparison=daily)
    obs_pairs, sims_pairs, base_pairs, lengths, disp_obs, disp_sims = predictive_checks(cat, fits, d1, des1, rng)
    obs, out, rp, strong_thr = rarity(cat, fits, des1, rng)
    f_bg, f_both, halves, d_t, des_t = trend_analysis(cat, tele, chosen)
    mw = moving_windows(cat, tele, chosen)
    trend_analysis(scat, tele, chosen, label="stations")
    power = power_analysis(cat, f_bg, des_t, rng)
    robustness(era5, tele, chosen, rng, scat)

    fig_season(daily, sdaily, cat)
    fig_counts(counts, mc, chosen)
    fig_dispersion(lengths, disp_obs, disp_sims)
    fig_pairs(obs_pairs, sims_pairs, base_pairs)
    fig_intensity(fits, d1)
    fig_rarity(obs, out, ["K14_strong", "S30"])
    fig_trend(mw, power, f"{season_label(int(cat.seasons[0]))} to {season_label(int(cat.seasons[-1]))}")
    draw_equations()
    print()
    print("Figures written to figures/, equation images to figures/equations/.")


if __name__ == "__main__":
    sys.exit(main())
