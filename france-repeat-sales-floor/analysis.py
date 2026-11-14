"""Is the purchase price a floor in France too? Repeat sales in the DVF files, 2021-2025.

    python analysis.py     # reads data/*.csv (built by build.py), writes results/

The companion analysis of England and Wales (``../uk-repeat-sales-rigidity``) found that about a
quarter of the resales that would have landed just below the seller's purchase price are missing.
This script asks the same question of French repeat sales, with the same estimator, and puts the
two countries side by side on the same years (resales 2021-2025) and the same holding periods
(under five years). It adds:

- a donut version of the estimator, which leaves out the bins within 1% of the old price, because
  in both countries resales at exactly the old price draw mass from both sides of it;
- a scan of the same estimator over cutoffs from -15% to +15%, which asks whether French sellers
  anchor on the price plus the purchase costs (notary fees are about 7-8% of an older dwelling's
  price) instead of the price itself;
- region-years in both countries against the strength of the local market, to separate a weaker
  French floor from a weaker French market;
- a bunching estimator: how much of the mass missing below the old price reappears at it;
- simulations of each estimator under a floor, under rounding to the old price and under
  withdrawal from the market.

The England and Wales aggregates are read from ``../uk-repeat-sales-rigidity/data`` and the OECD
house price indices from ``../housing-losses-inflation/data``; both are committed in this
repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
UK = HERE.parent / "uk-repeat-sales-rigidity" / "data"
OECD = HERE.parent / "housing-losses-inflation" / "data" / "oecd_house_prices.csv"

BIN = 0.005
BW = 0.05                    # bandwidth of the jump estimator, each side of the cutoff
DONUT = 0.01                 # bins within this distance of the cutoff are left out by the donut
PLACEBO_CUTS = (-0.15, -0.10, -0.075, 0.075, 0.10, 0.15)
SCAN_MIN = 0.02             # cutoffs closer to zero than this re-measure the jump at zero
YEARS = list(range(2021, 2026))
UK_HOLDS = ["0.5-2y", "2-5y"]
FR_HOLDS = ["0.5-1y", "1-2y", "2-3y", "3-4y", "4y+"]
FR_HOLDS_2_5 = ["2-3y", "3-4y", "4y+"]
FR_REGIONS = ["Paris", "Inner ring", "Outer Ile-de-France", "Big cities", "Rest of France", "Overseas"]
DIFFS = [-10000, -5000, -2500, -1000, 0, 1000, 2500, 5000, 10000]
BUNCH = dict(lo=-0.25, hi=0.25, zl=-0.10, zh=0.02, deg=7)
SEED = 20261114
OFF_PLAN_KEY = "mutations_Vente en l'état futur d'achèvement"


def pct(x: float, nd: int = 1) -> str:
    return "nan" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}%"


# ----------------------------------------------------------------------------- estimators
def _side_fit(counts: pd.Series, ks: list[int], cut: float) -> tuple[float, float] | None:
    x = (np.asarray(ks) + 0.5) * BIN - cut
    y = counts.reindex(ks).fillna(0.0).to_numpy(float)
    m = y > 0
    if m.sum() < 3:
        return None
    X = np.column_stack([np.ones(m.sum()), x[m]])
    w = y[m]
    XtW = X.T * w
    V = np.linalg.inv(XtW @ X)
    b = V @ XtW @ np.log(y[m])
    return float(b[0]), float(V[0, 0])


def density_jump(counts: pd.Series, cut: float, bw: float = BW, drop_exact: float = 0.0
                 ) -> tuple[float, float]:
    """The estimator of the England and Wales analysis, unchanged: log jump in the density of
    0.5% bins at ``cut`` from weighted least squares of log count on the bin centre on each side
    within ``bw``, Poisson weights. ``drop_exact`` resales at exactly the old price are removed from
    the first bin right of the cut. Returns theta and its standard error."""
    k0 = int(round(cut / BIN))
    nb = int(round(bw / BIN))
    c = counts.astype(float).copy()
    if drop_exact:
        c.loc[k0] = max(c.get(k0, 0.0) - drop_exact, 0.0)
    left = _side_fit(c, list(range(k0 - nb, k0)), cut)
    right = _side_fit(c, list(range(k0, k0 + nb)), cut)
    if left is None or right is None:
        return np.nan, np.nan
    return right[0] - left[0], float(np.sqrt(left[1] + right[1]))


def donut_jump(counts: pd.Series, cut: float, bw: float = BW, inner: float = DONUT,
               mask: tuple[float, float] | None = None) -> tuple[float, float]:
    """The same jump, with the bins within ``inner`` of the cut left out on both sides, so the
    resales at exactly the old price and the dip around them do not enter. ``mask`` also leaves
    out a fixed interval (the neighbourhood of the old price when scanning other cutoffs)."""
    k0 = int(round(cut / BIN))
    nb, ni = int(round(bw / BIN)), int(round(inner / BIN))
    L = list(range(k0 - nb, k0 - ni))
    R = list(range(k0 + ni, k0 + nb))
    if mask is not None:
        m0, m1 = int(round(mask[0] / BIN)), int(round(mask[1] / BIN))
        L = [k for k in L if not m0 <= k < m1]
        R = [k for k in R if not m0 <= k < m1]
    left, right = _side_fit(counts, L, cut), _side_fit(counts, R, cut)
    if left is None or right is None:
        return np.nan, np.nan
    return right[0] - left[0], float(np.sqrt(left[1] + right[1]))


def scan_mask(cut: float) -> tuple[float, float]:
    """For a cutoff away from zero, the bins left out of its windows: everything on the other side
    of zero and the bins within 1% of zero, so no window straddles the jump at the old price."""
    return (-1.0, DONUT) if cut > 0 else (-DONUT, 1.0)


def missing_share(theta: float) -> float:
    """Share of the resales that would have landed just below the cutoff and are missing:
    1 - exp(-theta), with the density just above as the counterfactual."""
    return float(-np.expm1(-theta))


def bunching(counts: pd.Series, lo: float = -0.25, hi: float = 0.25, zl: float = -0.10,
             zh: float = 0.02, deg: int = 7) -> dict:
    """Counterfactual density from a polynomial in the bin centre fitted to the counts in
    [lo, hi) outside the excluded window [zl, zh), with a dummy for every excluded bin. Missing
    mass: counterfactual minus observed in [zl, 0); excess: observed minus counterfactual in
    [0, zh), resales at exactly the old price included. Both as shares of the counterfactual mass
    in [zl, 0)."""
    ks = np.arange(int(round(lo / BIN)), int(round(hi / BIN)))
    y = counts.reindex(ks).fillna(0.0).to_numpy(float)
    x = (ks + 0.5) * BIN
    ex = (x >= zl) & (x < zh)
    X = np.column_stack([(x / 0.1) ** p for p in range(deg + 1)])
    D = np.column_stack([(ks == k).astype(float) for k in ks[ex]])
    b = np.linalg.lstsq(np.column_stack([X, D]), y, rcond=None)[0]
    cf = X @ b[: deg + 1]
    below, above = (x >= zl) & (x < 0), (x >= 0) & (x < zh)
    base = cf[below].sum()
    return {"missing": float((cf[below] - y[below]).sum() / base),
            "excess": float((y[above] - cf[above]).sum() / base), "cf": cf, "x": x, "y": y}


def bunching_se(counts: pd.Series, reps: int = 200, seed: int = SEED, **kw) -> tuple[float, float]:
    """Parametric bootstrap: Poisson redraws of every bin."""
    rng = np.random.default_rng(seed)
    ks = counts.index.to_numpy()
    lam = counts.to_numpy(float)
    m, e = [], []
    for _ in range(reps):
        r = bunching(pd.Series(rng.poisson(lam), index=ks), **kw)
        m.append(r["missing"])
        e.append(r["excess"])
    return float(np.std(m, ddof=1)), float(np.std(e, ddof=1))


def wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with heteroskedasticity-robust (HC1) standard errors."""
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    b = A @ XtW @ y
    e = y - X @ b
    meat = (X.T * (w * e) ** 2) @ X
    n, k = X.shape
    V = A @ meat @ A * n / (n - k)
    return b, np.sqrt(np.diag(V))


# ----------------------------------------------------------------------------- data helpers
def hist(h: pd.DataFrame, kind: str = "nominal", **sel) -> pd.Series:
    d = h[h.kind == kind]
    for k, v in sel.items():
        d = d[d[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    d = d[(d.bin != -999) & (d.bin != 999)]
    return d.groupby("bin").n.sum().astype(float)


def exact(diffs: pd.DataFrame, **sel) -> float:
    d = diffs
    for k, v in sel.items():
        d = d[d[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    return float(d.d0.sum())


def sel_rows(df: pd.DataFrame, **sel) -> pd.DataFrame:
    for k, v in sel.items():
        df = df[df[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    return df


def jumps(h: pd.DataFrame, diffs: pd.DataFrame, **sel) -> dict:
    c = hist(h, **sel)
    ex = exact(diffs, **sel)
    out = {"n": int(c.sum()), "exact": ex}
    out["theta"], out["se"] = density_jump(c, 0.0)
    out["theta_ex"], out["se_ex"] = density_jump(c, 0.0, drop_exact=ex)
    out["theta_donut"], out["se_donut"] = donut_jump(c, 0.0)
    for cut in PLACEBO_CUTS:
        out[f"placebo_{cut:+.3f}"], out[f"placebo_se_{cut:+.3f}"] = donut_jump(c, cut)
    return out


def annual_inflation(cpi: pd.DataFrame) -> pd.Series:
    c = cpi.assign(year=pd.to_datetime(cpi.month).dt.year).groupby("year").cpi.mean()
    return (c / c.shift(1) - 1).rename("inflation")


def bmn_index(qq: pd.DataFrame) -> pd.Series:
    """Repeat-sales index (log, first quarter 0) by least squares on quarter pairs."""
    g = qq.groupby(["q0", "q1"])[["n", "sum_lr"]].sum().reset_index()
    qs = sorted(set(g.q0) | set(g.q1))
    ix = {q: i for i, q in enumerate(qs)}
    T = len(qs)
    A = np.zeros((T, T))
    b = np.zeros(T)
    for r in g.itertuples():
        i, j = ix[r.q0], ix[r.q1]
        A[i, i] += r.n
        A[j, j] += r.n
        A[i, j] -= r.n
        A[j, i] -= r.n
        b[j] += r.sum_lr
        b[i] -= r.sum_lr
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    return pd.Series(np.linalg.solve(A, b), index=qs)


def qlabel(q: int) -> str:
    return f"{2014 + q // 4}Q{q % 4 + 1}"


# ----------------------------------------------------------------------------- simulation
def simulate(q: float = 0.0, r: float = 0.0, withdraw: float = 0.0, mu: float = 0.08,
             sd: float = 0.15, n: int = 600_000, seed: int = SEED) -> tuple[pd.Series, float]:
    """Log price changes N(mu, sd). A share ``q`` of the would-be losses of up to 10% sell at
    exactly the old price (a floor that converts); a share ``withdraw`` of them do not sell at all
    (a floor that removes them); a share ``r`` of the changes within 1% of zero, on either side,
    are rounded to exactly zero. Returns the histogram and the number of exact resales."""
    rng = np.random.default_rng(seed)
    x = rng.normal(mu, sd, n)
    small_loss = (x < 0) & (x >= -0.10)
    u = rng.random(n)
    anchor = small_loss & (u < q)
    gone = small_loss & (u >= q) & (u < q + withdraw)
    near = (np.abs(x) < 0.01) & (rng.random(n) < r)
    x = np.where(anchor | near, 0.0, x)[~gone]
    counts = pd.Series(np.floor(x / BIN).astype(int)).value_counts().sort_index().astype(float)
    return counts, float((x == 0.0).sum())


# ----------------------------------------------------------------------------- main
def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    S: list[str] = []
    line = S.append

    h = pd.read_csv(DATA / "pairs_hist.csv")
    cells = pd.read_csv(DATA / "pairs_cells.csv")
    cells["same_struct"] = cells.same_struct.astype("boolean").fillna(False).astype(bool)
    diffs = pd.read_csv(DATA / "pairs_diffs.csv")
    dep_hist = pd.read_csv(DATA / "dep_hist.csv", dtype={"dep": str})
    dep_cells = pd.read_csv(DATA / "dep_cells.csv", dtype={"dep": str})
    qq = pd.read_csv(DATA / "pairs_qq.csv")
    sales = pd.read_csv(DATA / "sales_monthly.csv", parse_dates=["month"])
    price = pd.read_csv(DATA / "price_m2_quarterly.csv")
    cpi = pd.read_csv(DATA / "fr_cpi_monthly.csv")
    blog = dict(l.split("\t") for l in (DATA / "build_log.txt").read_text().splitlines() if "\t" in l)
    uh = pd.read_csv(UK / "pairs_hist.csv")
    uc = pd.read_csv(UK / "pairs_cells.csv")
    ud = pd.read_csv(UK / "pairs_diffs.csv")
    ucpi = pd.read_csv(UK / "uk_cpi_monthly.csv")
    oecd = pd.read_csv(OECD)
    infl = annual_inflation(cpi)
    uinfl = annual_inflation(ucpi)
    main_cells = cells[(cells.first_sale == "sale") & (cells.second_sale == "sale") & cells.same_struct]
    uk_cells = uc[uc.first_sale_new == "N"]
    UKSEL = {"y": YEARS, "hold": UK_HOLDS}

    line("REPEAT SALES IN FRANCE (DVF, 2021-2025): every number used in the article")
    line("=" * 76)

    # ---- [0] data -------------------------------------------------------------------------
    line("")
    line("[0] Data and matching")
    line(f"DVF rows {int(blog['rows']):,}; deeds {int(blog['mutations']):,}, dated {blog['first_date']} to "
         f"{blog['last_date']}; ordinary sales {int(blog['mutations_Vente']):,}, off-plan "
         f"{int(blog.get(OFF_PLAN_KEY, 0)):,}, "
         f"auctions {int(blog.get('mutations_Adjudication', 0)):,}")
    line(f"Deeds transferring exactly one dwelling (one house or one flat, any annexes, no shop, one "
         f"disposition, one price): {int(blog['single_dwelling_mutations']):,}; with a property key "
         f"{int(blog['keyed_sales']):,} (houses {int(blog['keyed_houses']):,}, flats "
         f"{int(blog['keyed_flats']):,}); dropped as same-day duplicates {int(blog['same_day_duplicates_dropped']):,}")
    raw, kept = int(blog["raw_pairs"]), int(blog["pairs_after_price_and_time_filters"])
    changed = int(blog["pairs_structure_changed"])
    line(f"Consecutive sales of the same key: {raw:,}; after the time and price filters {kept:,}; of "
         f"which rooms, surface or land changed {changed:,} ({pct(changed / kept)})")
    others = {k[6:]: int(v) for k, v in blog.items() if k.startswith("pairs_") and "_then_" in k}
    line("With rooms, surface and land unchanged, by nature of the two sales: " +
         ", ".join(f"{k} {v:,}" for k, v in sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))))
    n_main = int(main_cells.n.sum())
    line(f"Main sample: {n_main:,} repeat sales, resales {blog['first_resale_month'][:7]} to "
         f"{blog['last_resale_month'][:7]}; houses {int(main_cells[main_cells.ptype == 'house'].n.sum()):,}, "
         f"flats {int(main_cells[main_cells.ptype == 'flat'].n.sum()):,}")
    line("By region: " + ", ".join(f"{r} {int(main_cells[main_cells.region == r].n.sum()):,}" for r in FR_REGIONS))
    line("By holding period: " + ", ".join(f"{k} {int(main_cells[main_cells.hold == k].n.sum()):,}" for k in FR_HOLDS))
    line("By year of resale: " + ", ".join(f"{y} {int(main_cells[main_cells.y == y].n.sum()):,}" for y in YEARS))
    sc = cells[(cells.first_sale == "sale") & (cells.second_sale == "sale")]
    med_same = np.average(sc[sc.same_struct].median_lr, weights=sc[sc.same_struct].n)
    med_chg = np.average(sc[~sc.same_struct].median_lr, weights=sc[~sc.same_struct].n)
    line(f"Median log change (cell medians, weighted): unchanged structure {med_same:.3f}, changed "
         f"{med_chg:.3f}")
    uk_main = sel_rows(uk_cells, **UKSEL)
    line(f"England and Wales, same resale years and holds under five years: {int(uk_main.n.sum()):,} "
         f"repeat sales")

    # ---- [1] market -----------------------------------------------------------------------
    line("")
    line("[1] The market: OECD nominal house price index (2015 = 100), price per square metre, volume")
    hp = oecd[(oecd.measure == "HPI") & oecd.iso.isin(["FRA", "GBR"])].pivot(index="quarter",
                                                                             columns="iso", values="value")
    hp = hp.loc["2019-Q1":]
    for iso, lab in [("FRA", "France"), ("GBR", "United Kingdom")]:
        s = hp[iso].dropna()
        s2 = s.loc["2020-Q1":]
        pk = s2.idxmax()
        after = s2.loc[pk:]
        tr = after.idxmin()
        line(f"  {lab}: 2021-Q1 {s.get('2021-Q1', np.nan):.1f}, peak {pk} {s[pk]:.1f}, low after it {tr} "
             f"{s[tr]:.1f} ({pct(s[tr] / s[pk] - 1)}), last {s.index[-1]} {s.iloc[-1]:.1f} "
             f"({pct(s.iloc[-1] / s[pk] - 1)} from peak)")
    for iso, lab in [("FRA", "France"), ("GBR", "United Kingdom")]:
        line(f"  {lab}: 2021-Q1 to 2025-Q4 {pct(hp.loc['2025-Q4', iso] / hp.loc['2021-Q1', iso] - 1)}")
    hp.to_csv(RESULTS / "oecd_hpi.csv", float_format="%.6g")
    for reg in ["France", "Paris", "Inner ring", "Outer Ile-de-France", "Big cities", "Rest of France"]:
        for pt in ["flat", "house"]:
            p = price[(price.region == reg) & (price.ptype == pt)].set_index("q").median_eur_m2
            if len(p) < 8:
                continue
            p4 = p.rolling(4).mean().dropna()
            pk = p4.idxmax()
            tr = p4.loc[pk:].idxmin()
            line(f"  median price per m2, {reg}, {pt}s (4-quarter mean): peak {qlabel(pk)} "
                 f"EUR {p4[pk]:,.0f}, low after it {qlabel(tr)} EUR {p4[tr]:,.0f} ({pct(p4[tr] / p4[pk] - 1)}), "
                 f"last {qlabel(p4.index[-1])} EUR {p4.iloc[-1]:,.0f}")
    vol = sales[sales.nature == "sale"].groupby("month").sales.sum().sort_index()
    roll = vol.rolling(12).sum().dropna()
    pk = roll.idxmax()
    tr = roll.loc[pk:].idxmin()
    line(f"  single-dwelling ordinary sales, 12-month total: peak to {pk:%Y-%m} {roll[pk]:,.0f}; low "
         f"to {tr:%Y-%m} {roll[tr]:,.0f} ({pct(roll[tr] / roll[pk] - 1)}); to {roll.index[-1]:%Y-%m} "
         f"{roll.iloc[-1]:,.0f}")
    roll.to_frame("sales_12m").to_csv(RESULTS / "volume_12m.csv", float_format="%.6g")
    idx = bmn_index(qq)
    line(f"  repeat-sales index from the pairs (not used for the market path; see text): 2021Q1 = 100, "
         f"2022Q4 {100 * np.exp(idx.get(35, np.nan)):.1f}, 2024Q4 {100 * np.exp(idx.get(43, np.nan)):.1f}, "
         f"2025Q4 {100 * np.exp(idx.get(47, np.nan)):.1f}")
    big = h[(h.kind == "nominal")]
    share_big = big[big.bin >= 60].n.sum() / big.n.sum()
    ub = sel_rows(uh[uh.kind == "nominal"], **UKSEL)
    line(f"  share of repeat sales at least 30% (log) above the purchase price: France {pct(share_big)}, "
         f"England and Wales same years and holds {pct(ub[ub.bin >= 60].n.sum() / ub.n.sum())}")
    line(f"  French CPI inflation by year: " + ", ".join(f"{y} {pct(infl.get(y))}" for y in YEARS) +
         "; UK: " + ", ".join(f"{y} {pct(uinfl.get(y))}" for y in YEARS))

    # ---- [2] losses ----------------------------------------------------------------------
    line("")
    line("[2] Resales at a loss, by year of resale")
    rows = []
    for y in YEARS:
        f = main_cells[main_cells.y == y]
        u = sel_rows(uk_cells, y=[y], hold=UK_HOLDS)
        r = {"y": y, "n": int(f.n.sum()), "p_nominal": f.nominal_loss.sum() / f.n.sum(),
             "p_real": f.real_loss.sum() / f.n.sum(), "p_exact": f.exact_old_price.sum() / f.n.sum(),
             "hidden_share": f.hidden_loss.sum() / f.real_loss.sum(), "inflation": infl.get(y),
             "uk_n": int(u.n.sum()), "uk_p_nominal": u.nominal_loss.sum() / u.n.sum(),
             "uk_p_real": u.real_loss.sum() / u.n.sum(), "uk_hidden_share": u.hidden_loss.sum() / u.real_loss.sum(),
             "uk_p_exact": u.exact_old_price.sum() / u.n.sum()}
        rows.append(r)
        line(f"  {y}: France n={r['n']:,} nominal loss {pct(r['p_nominal'])}, real loss {pct(r['p_real'])}, "
             f"hidden/real {pct(r['hidden_share'])}, exact {pct(r['p_exact'], 2)} | England and Wales "
             f"n={r['uk_n']:,} nominal {pct(r['uk_p_nominal'])}, real {pct(r['uk_p_real'])}, hidden/real "
             f"{pct(r['uk_hidden_share'])}, exact {pct(r['uk_p_exact'], 2)}")
    pd.DataFrame(rows).to_csv(RESULTS / "losses_by_year.csv", index=False, float_format="%.6g")
    for reg in FR_REGIONS:
        f = main_cells[(main_cells.region == reg) & main_cells.y.isin([2024, 2025])]
        line(f"  2024-2025, {reg}: nominal loss {pct(f.nominal_loss.sum() / f.n.sum())}, real loss "
             f"{pct(f.real_loss.sum() / f.n.sum())} [n={int(f.n.sum()):,}]")
    for pt in ["house", "flat"]:
        f = main_cells[(main_cells.ptype == pt) & main_cells.y.isin([2024, 2025])]
        line(f"  2024-2025, {pt}s: nominal loss {pct(f.nominal_loss.sum() / f.n.sum())}, real loss "
             f"{pct(f.real_loss.sum() / f.n.sum())}")
    tot = main_cells[["n", "nominal_loss", "real_loss", "hidden_loss"]].sum()
    utot = uk_main[["n", "nominal_loss", "real_loss", "hidden_loss"]].sum()
    line(f"  all years: France nominal {pct(tot.nominal_loss / tot.n)}, real {pct(tot.real_loss / tot.n)}, "
         f"hidden/real {pct(tot.hidden_loss / tot.real_loss)}; England and Wales nominal "
         f"{pct(utot.nominal_loss / utot.n)}, real {pct(utot.real_loss / utot.n)}, hidden/real "
         f"{pct(utot.hidden_loss / utot.real_loss)}")

    # ---- [3] exact old price ---------------------------------------------------------------
    line("")
    line("[3] Resales at exactly the old price, against resales at exact round differences (EUR / GBP)")

    def diff_line(d: pd.DataFrame) -> dict:
        s = d[[f"d{k}" for k in DIFFS]].sum()
        n = d.n.sum()
        return {"n": int(n), "share_exact": s["d0"] / n,
                "x1000": s["d0"] / s[["d-1000", "d1000"]].mean(),
                "x5000": s["d0"] / s[["d-5000", "d5000"]].mean(),
                "x10000": s["d0"] / s[["d-10000", "d10000"]].mean(),
                "ba1000": s["d-1000"] / s["d1000"], "ba5000": s["d-5000"] / s["d5000"],
                "ba10000": s["d-10000"] / s["d10000"]}

    xr = []
    for lab, d in [("France", diffs), ("England and Wales", sel_rows(ud, **UKSEL))]:
        r = diff_line(d)
        r["sample"] = lab
        xr.append(r)
        line(f"  {lab}: exact {pct(r['share_exact'], 2)} of {r['n']:,}; exact vs mean of -/+1,000 "
             f"{r['x1000']:.2f}x, -/+5,000 {r['x5000']:.2f}x, -/+10,000 {r['x10000']:.2f}x; below/above at "
             f"1,000 {r['ba1000']:.2f}, 5,000 {r['ba5000']:.2f}, 10,000 {r['ba10000']:.2f}")
    for hold in FR_HOLDS:
        r = diff_line(diffs[diffs.hold == hold])
        line(f"  France, holding {hold}: exact {pct(r['share_exact'], 2)}, vs -/+5,000 {r['x5000']:.2f}x")
    for hold in UK_HOLDS:
        r = diff_line(sel_rows(ud, y=YEARS, hold=[hold]))
        line(f"  England and Wales, holding {hold}: exact {pct(r['share_exact'], 2)}, vs -/+5,000 {r['x5000']:.2f}x")
    for pt in ["house", "flat"]:
        r = diff_line(diffs[diffs.ptype == pt])
        line(f"  France, {pt}s: exact {pct(r['share_exact'], 2)}, vs -/+5,000 {r['x5000']:.2f}x")
    rk = diffs[["n", "round_k0", "round_k1"]].sum()
    line(f"  French prices that are whole thousands of euros: purchase {pct(rk.round_k0 / rk.n)}, resale "
         f"{pct(rk.round_k1 / rk.n)}")
    pd.DataFrame(xr).to_csv(RESULTS / "exact_price.csv", index=False, float_format="%.6g")

    # ---- [4] the jump at zero --------------------------------------------------------------
    line("")
    line(f"[4] Jump in density at a nominal change of zero (0.5% bins, bandwidth {BW:.0%} each side)")
    J = []

    def report(lab: str, r: dict, indent: str = "  ") -> None:
        line(f"{indent}{lab}: theta {r['theta']:.3f} ({r['se']:.3f}); without exact resales "
             f"{r['theta_ex']:.3f} ({r['se_ex']:.3f}) = missing {pct(missing_share(r['theta_ex']))}; donut "
             f"{r['theta_donut']:.3f} ({r['se_donut']:.3f}) = missing {pct(missing_share(r['theta_donut']))}"
             f"  [n={r['n']:,}]")

    fr_all = jumps(h, diffs)
    uk_all = jumps(uh, ud, **UKSEL)
    uk_long = jumps(uh, ud, hold=["0.5-2y", "2-5y", "5-10y"])
    report("France, all", fr_all)
    report("England and Wales, 2021-2025, holds under 5 years", uk_all)
    report("England and Wales, 1995-2026, holds under 10 years (the earlier article)", uk_long)
    d_ex = fr_all["theta_ex"] - uk_all["theta_ex"]
    d_dn = fr_all["theta_donut"] - uk_all["theta_donut"]
    line(f"  France minus England and Wales: without exact {d_ex:+.3f} (se "
         f"{np.hypot(fr_all['se_ex'], uk_all['se_ex']):.3f}); donut {d_dn:+.3f} (se "
         f"{np.hypot(fr_all['se_donut'], uk_all['se_donut']):.3f})")
    for lab, r in [("France", fr_all), ("England and Wales", uk_all)]:
        line(f"  placebo cutoffs (donut), {lab}: " + ", ".join(
            f"{100 * c:+.1f}% {r[f'placebo_{c:+.3f}']:.3f} ({r[f'placebo_se_{c:+.3f}']:.3f})" for c in PLACEBO_CUTS))
    fc = hist(h)
    uc_ = hist(uh, **UKSEL)
    sens = []
    for bw in (0.025, 0.05, 0.075, 0.10):
        for inner in (0.0, 0.01, 0.015, 0.02):
            if bw - inner < 0.02:          # keep at least four bins on each side
                continue
            if inner == 0.0:
                tf, sf = density_jump(fc, 0.0, bw, drop_exact=fr_all["exact"])
                tu, su = density_jump(uc_, 0.0, bw, drop_exact=uk_all["exact"])
            else:
                tf, sf = donut_jump(fc, 0.0, bw, inner)
                tu, su = donut_jump(uc_, 0.0, bw, inner)
            sens.append({"bw": bw, "inner": inner, "fr": tf, "fr_se": sf, "uk": tu, "uk_se": su})
    SE = pd.DataFrame(sens)
    SE.to_csv(RESULTS / "jump_sensitivity.csv", index=False, float_format="%.6g")
    line("  sensitivity (bandwidth / donut; 0 = exact resales removed instead): " + "; ".join(
        f"{r.bw:.1%}/{r.inner:.1%} FR {pct(missing_share(r.fr))} UK {pct(missing_share(r.uk))}"
        for r in SE.itertuples()))
    line(f"  range of the French missing share over these {len(SE)} variants: "
         f"{pct(missing_share(SE.fr.min()))} to {pct(missing_share(SE.fr.max()))}; England and Wales "
         f"{pct(missing_share(SE.uk.min()))} to {pct(missing_share(SE.uk.max()))}")
    far = hist(h, "real_far_excl_exact")
    tr_, sr_ = density_jump(far, 0.0)
    trd, srd = donut_jump(far, 0.0)
    line(f"  real-zero placebo (pairs with cumulative CPI inflation of at least 10%, exact resales out, "
         f"n={int(far.sum()):,}): theta {tr_:.3f} ({sr_:.3f}); donut {trd:.3f} ({srd:.3f})")
    for pt in ["house", "flat"]:
        r = jumps(h, diffs, ptype=[pt])
        r.update({"group": f"type: {pt}"})
        J.append(r)
        report(f"France, {pt}s", r)
    for reg in FR_REGIONS:
        r = jumps(h, diffs, region=[reg])
        r.update({"group": f"region: {reg}"})
        J.append(r)
        report(f"France, {reg}", r)
    idf = jumps(h, diffs, region=["Paris", "Inner ring", "Outer Ile-de-France"])
    report("France, Ile-de-France (three groups pooled)", idf)
    for hold in FR_HOLDS:
        r = jumps(h, diffs, hold=[hold])
        r.update({"group": f"hold: {hold}"})
        J.append(r)
        report(f"France, holding {hold}", r)
    for hold in UK_HOLDS:
        r = jumps(uh, ud, y=YEARS, hold=[hold])
        report(f"England and Wales 2021-2025, holding {hold}", r)
    for y in YEARS:
        r = jumps(h, diffs, y=[y])
        u = jumps(uh, ud, y=[y], hold=UK_HOLDS)
        r.update({"group": f"year: {y}", "uk_theta_ex": u["theta_ex"], "uk_se_ex": u["se_ex"],
                  "uk_theta_donut": u["theta_donut"], "uk_se_donut": u["se_donut"]})
        J.append(r)
        line(f"  {y}: France without exact {r['theta_ex']:.3f} ({r['se_ex']:.3f}) = missing "
             f"{pct(missing_share(r['theta_ex']))}, donut {pct(missing_share(r['theta_donut']))} [n={r['n']:,}] | "
             f"England and Wales {pct(missing_share(u['theta_ex']))}, donut {pct(missing_share(u['theta_donut']))}")
    r = jumps(h, diffs, y=[2021, 2022])
    line(f"  France 2021-2022 pooled: missing {pct(missing_share(r['theta_ex']))} ({r['se_ex']:.3f}); "
         f"2023-2025 pooled: {pct(missing_share(jumps(h, diffs, y=[2023, 2024, 2025])['theta_ex']))}")
    pd.DataFrame(J).to_csv(RESULTS / "jumps_by_group.csv", index=False, float_format="%.6g")

    # ---- [5] scan over cutoffs ------------------------------------------------------------
    line("")
    line(f"[5] The same jump at other cutoffs (donut estimator, bandwidth {BW:.0%} each side; no window "
         f"reaches within 1% of zero or crosses it)")
    scan = []
    for k in range(-30, 31):
        cut = round(k * BIN, 4)
        for lab, c in [("France", fc), ("England and Wales", uc_)]:
            if k == 0:
                t, s = donut_jump(c, 0.0)
            else:
                t, s = donut_jump(c, cut, inner=0.0, mask=scan_mask(cut))
            scan.append({"sample": lab, "cut": cut, "theta": t, "se": s})
    SC = pd.DataFrame(scan)
    SC.to_csv(RESULTS / "jump_scan.csv", index=False, float_format="%.6g")
    for lab in ["France", "England and Wales"]:
        s = SC[(SC["sample"] == lab)].dropna()
        z0 = float(s[s.cut == 0].theta.iloc[0])
        off = s[s.cut.abs() >= SCAN_MIN - 1e-9]
        line(f"  {lab}: at zero {z0:.3f} (missing {pct(missing_share(z0))}); at the {len(off)} cutoffs at least "
             f"{SCAN_MIN:.0%} from zero: sd {off.theta.std():.3f}, largest {off.theta.max():+.3f} at "
             f"{100 * off.cut[off.theta.idxmax()]:+.1f}%, smallest {off.theta.min():+.3f}; cutoffs with a jump at "
             f"least as large as at zero: {int((off.theta >= z0).sum())} of {len(off)}")
        for c in (0.05, 0.07, 0.075, 0.08):
            row = off[np.isclose(off.cut, c)]
            if len(row):
                line(f"    at {100 * c:+.1f}%: {row.theta.iloc[0]:+.3f} ({row.se.iloc[0]:.3f})")
    line("  round price ratios fall at: " + ", ".join(f"x{r:.2f} = {100 * np.log(r):+.2f}%"
                                                    for r in (0.90, 0.95, 1.05, 1.08, 1.10, 1.15)))

    # ---- [6] market state -----------------------------------------------------------------
    line("")
    line("[6] Is France different, or is its market? Region-years, holds under 5 years")
    rows = []
    for (y, reg), _ in h[h.kind == "nominal"].groupby(["y", "region"]):
        if y not in (2023, 2024, 2025):
            continue
        r = jumps(h, diffs, y=[y], region=[reg])
        cc = main_cells[(main_cells.y == y) & (main_cells.region == reg) & main_cells.hold.isin(FR_HOLDS_2_5)]
        cl = main_cells[(main_cells.y == y) & (main_cells.region == reg)]
        rows.append({"country": "France", "y": y, "region": reg, "theta_ex": r["theta_ex"], "se_ex": r["se_ex"],
                     "median_lr_2_5y": np.average(cc.median_lr, weights=cc.n) if cc.n.sum() >= 50 else np.nan,
                     "p_nominal": cl.nominal_loss.sum() / cl.n.sum(), "n": r["n"]})
    for (y, reg), _ in uh[(uh.kind == "nominal") & uh.y.between(1996, 2025)].groupby(["y", "region"]):
        r = jumps(uh, ud, y=[y], region=[reg], hold=UK_HOLDS)
        cc = uk_cells[(uk_cells.y == y) & (uk_cells.region == reg) & (uk_cells.hold == "2-5y")]
        cl = sel_rows(uk_cells, y=[y], region=[reg], hold=UK_HOLDS)
        rows.append({"country": "England and Wales", "y": y, "region": reg, "theta_ex": r["theta_ex"],
                     "se_ex": r["se_ex"],
                     "median_lr_2_5y": np.average(cc.median_lr, weights=cc.n) if cc.n.sum() >= 50 else np.nan,
                     "p_nominal": cl.nominal_loss.sum() / cl.n.sum(), "n": r["n"]})
    RY = pd.DataFrame(rows)
    RY.to_csv(RESULTS / "region_years.csv", index=False, float_format="%.6g")
    RY = RY.dropna(subset=["theta_ex", "se_ex", "median_lr_2_5y"])
    RY = RY[np.isfinite(RY.se_ex) & (RY.se_ex > 0)]
    fr_ry = RY[RY.country == "France"]
    uk_ry = RY[RY.country == "England and Wales"]
    line(f"  {len(uk_ry)} England and Wales region-years (1996-2025), {len(fr_ry)} French region-years "
         f"(2023-2025; earlier years have no holds of two years or more)")
    for v, lab in [("median_lr_2_5y", "median log change of 2-5 year holds"),
                   ("p_nominal", "share of resales at a nominal loss")]:
        for name, d in [("England and Wales", uk_ry), ("France", fr_ry)]:
            b, se = wls(d.theta_ex.to_numpy(), np.column_stack([np.ones(len(d)), d[v]]), 1 / d.se_ex.to_numpy() ** 2)
            line(f"  {name}: theta_ex = {b[0]:.3f} ({se[0]:.3f}) {b[1]:+.3f} ({se[1]:.3f}) x {lab}")
        X = np.column_stack([np.ones(len(RY)), RY[v], (RY.country == "France").astype(float)])
        b, se = wls(RY.theta_ex.to_numpy(), X, 1 / RY.se_ex.to_numpy() ** 2)
        line(f"  pooled: France shift {b[2]:+.3f} (se {se[2]:.3f}) at the same {lab}; slope {b[1]:+.3f} ({se[1]:.3f})")
        bu, _ = wls(uk_ry.theta_ex.to_numpy(), np.column_stack([np.ones(len(uk_ry)), uk_ry[v]]),
                    1 / uk_ry.se_ex.to_numpy() ** 2)
        pred = bu[0] + bu[1] * fr_ry[v]
        w = 1 / fr_ry.se_ex ** 2
        line(f"    French region-years: mean of {lab} {np.average(fr_ry[v], weights=w):.3f}; predicted from "
             f"the England and Wales line {np.average(pred, weights=w):.3f} (missing "
             f"{pct(missing_share(np.average(pred, weights=w)))}), observed {np.average(fr_ry.theta_ex, weights=w):.3f} "
             f"(missing {pct(missing_share(np.average(fr_ry.theta_ex, weights=w)))})")
    q_uk = uk_ry.assign(band=pd.cut(uk_ry.p_nominal, [0, 0.05, 0.10, 0.15, 0.25, 1.0]))
    for band, d in q_uk.groupby("band", observed=True):
        line(f"  England and Wales region-years with nominal-loss share {band}: {len(d)}, weighted theta_ex "
             f"{np.average(d.theta_ex, weights=1 / d.se_ex ** 2):.3f}")
    line(f"  French region-years: nominal-loss share {fr_ry.p_nominal.min():.3f} to {fr_ry.p_nominal.max():.3f}")
    # departements within France
    g = (dep_cells[dep_cells.y.isin([2023, 2024, 2025])]
         .assign(w_med=lambda d: d.median_lr_1_3y.fillna(0) * d.n_1_3y)
         .groupby(["dep", "y"])[["n", "exact_old_price", "n_1_3y", "w_med", "nominal_loss"]].sum().reset_index())
    g = g[g.n_1_3y >= 100].assign(med=lambda d: d.w_med / d.n_1_3y)
    g["quartile"] = g.groupby("y").med.transform(lambda s: pd.qcut(s, 4, labels=False))
    dq = []
    for qt, d in g.groupby("quartile"):
        keys = set(zip(d.dep, d.y))
        hh = dep_hist[[k in keys for k in zip(dep_hist.dep, dep_hist.y)]]
        c = hh[(hh.bin != -999) & (hh.bin != 999)].groupby("bin").n.sum().astype(float)
        t, s = density_jump(c, 0.0, drop_exact=float(d.exact_old_price.sum()))
        td, sd_ = donut_jump(c, 0.0)
        dq.append({"quartile": int(qt) + 1, "dep_years": len(d), "median_lr_1_3y": float(d.med.median()),
                   "p_nominal": d.nominal_loss.sum() / d.n.sum(), "theta_ex": t, "se_ex": s,
                   "theta_donut": td, "se_donut": sd_, "n": int(c.sum())})
        line(f"  departement-years, quartile {int(qt) + 1} of the median change of 1-3 year holds "
             f"(within year): {len(d)} cells, median {d.med.median():.3f}, nominal-loss share "
             f"{pct(d.nominal_loss.sum() / d.n.sum())}; missing {pct(missing_share(t))} ({s:.3f}), donut "
             f"{pct(missing_share(td))} ({sd_:.3f})")
    pd.DataFrame(dq).to_csv(RESULTS / "departement_quartiles.csv", index=False, float_format="%.6g")

    # ---- [7] bunching ---------------------------------------------------------------------
    line("")
    line(f"[7] Bunching: counterfactual polynomial (degree {BUNCH['deg']}) over {BUNCH['lo']:+.0%} to "
         f"{BUNCH['hi']:+.0%}, excluded window {BUNCH['zl']:+.0%} to {BUNCH['zh']:+.0%}")
    brows = []
    nz = []
    obs = []
    for lab, c, ex in [("France", fc, fr_all["exact"]), ("England and Wales", uc_, uk_all["exact"])]:
        b = bunching(c, **BUNCH)
        sm, se_ = bunching_se(c, **BUNCH)
        obs.append({"sample": lab, "missing": b["missing"], "excess": b["excess"], "se_missing": sm, "se_excess": se_})
        tot = c.sum()
        for x, yv, cf in zip(b["x"], b["y"], b["cf"]):
            k = int(round(x / BIN - 0.5))
            nz.append({"sample": lab, "bin": k, "x": x, "share": yv / tot, "cf_share": cf / tot,
                       "exact_share": (ex / tot) if k == 0 else 0.0})
        line(f"  {lab}: missing below the old price {pct(b['missing'])} (se {pct(sm)}) of the counterfactual "
             f"mass in [-10%, 0); excess in [0, +2%) {pct(b['excess'])} (se {pct(se_)}); excess / missing "
             f"{b['excess'] / b['missing']:.2f}")
        for deg in (5, 7, 9):
            for zl in (-0.05, -0.10, -0.15):
                for zh in (0.02, 0.03):
                    bb = bunching(c, BUNCH["lo"], BUNCH["hi"], zl, zh, deg)
                    brows.append({"sample": lab, "deg": deg, "zl": zl, "zh": zh, **{k: bb[k] for k in ("missing", "excess")}})
    pd.DataFrame(nz).to_csv(RESULTS / "near_zero.csv", index=False, float_format="%.6g")
    pd.DataFrame(obs).to_csv(RESULTS / "bunching_observed.csv", index=False, float_format="%.6g")
    BR = pd.DataFrame(brows)
    BR.to_csv(RESULTS / "bunching_grid.csv", index=False, float_format="%.6g")
    for lab in ["France", "England and Wales"]:
        d = BR[BR["sample"] == lab]
        ok = d[d.missing > 0.05]
        ratio = ok.excess / ok.missing
        line(f"  {lab}, over {len(d)} variants: missing {pct(d.missing.min())} to {pct(d.missing.max())}, "
             f"excess {pct(d.excess.min())} to {pct(d.excess.max())}; in the {len(ok)} variants with missing "
             f"mass above 5%, excess / missing {ratio.min():.2f} to {ratio.max():.2f} (median {ratio.median():.2f})")

    # ---- [8] simulations ------------------------------------------------------------------
    line("")
    line("[8] Simulations: 600,000 log changes N(+8%, 15%); floors act on would-be losses of up to 10%")
    sims = [("no floor, no rounding", {}), ("30% sell at exactly the old price", {"q": 0.3}),
            ("30% withdraw", {"withdraw": 0.3}), ("rounding only: half within 1% go to zero", {"r": 0.5}),
            ("30% at the old price + rounding", {"q": 0.3, "r": 0.5}),
            ("10% at the old price + 20% withdraw + rounding", {"q": 0.1, "withdraw": 0.2, "r": 0.5})]
    srows = []
    for lab, kw in sims:
        c, ex = simulate(**kw)
        t_ex, _ = density_jump(c, 0.0, drop_exact=ex)
        t_dn, _ = donut_jump(c, 0.0)
        b = bunching(c, **BUNCH)
        srows.append({"scenario": lab, "missing_181": missing_share(t_ex), "missing_donut": missing_share(t_dn),
                      "bunch_missing": b["missing"], "bunch_excess": b["excess"]})
        line(f"  {lab}: estimator of the earlier article {pct(missing_share(t_ex))}, donut "
             f"{pct(missing_share(t_dn))}, bunching missing {pct(b['missing'])} / excess {pct(b['excess'])}")
    pd.DataFrame(srows).to_csv(RESULTS / "simulations.csv", index=False, float_format="%.6g")

    # ---- [9] other pairs -----------------------------------------------------------------
    line("")
    line("[9] Pairs kept out of the main sample")
    for (f0, f1), lab in [(("auction", "sale"), "bought at auction, resold by ordinary sale"),
                          (("off-plan", "sale"), "bought off plan, resold"),
                          (("sale", "auction"), "resold at auction")]:
        d = cells[(cells.first_sale == f0) & (cells.second_sale == f1) & cells.same_struct]
        if d.n.sum():
            line(f"  {lab}: {int(d.n.sum()):,}; median log change {np.average(d.median_lr, weights=d.n):.3f}; "
                 f"nominal loss {pct(d.nominal_loss.sum() / d.n.sum())}; median hold "
                 f"{np.average(d.median_hold_y, weights=d.n):.2f} years")
    d = sc[~sc.same_struct]
    line(f"  ordinary sales with rooms, surface or land changed: {int(d.n.sum()):,}; median log change "
         f"{np.average(d.median_lr, weights=d.n):.3f}; nominal loss {pct(d.nominal_loss.sum() / d.n.sum())}")

    (RESULTS / "summary.txt").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
