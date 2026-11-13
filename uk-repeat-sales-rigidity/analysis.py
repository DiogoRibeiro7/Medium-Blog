"""Do sellers refuse nominal losses? Repeat sales in England and Wales, 1995-2026.

    python analysis.py     # reads data/*.csv (built by build.py), writes results/

The test is about the nominal purchase price as a floor. If sellers resist selling below what
they paid, the distribution of the log price change between two sales of the same home should
show (a) more resales at exactly the old price than at nearby round differences, and (b) a jump
in density at a nominal change of zero: fewer resales just below zero than just above. The same
estimator applied where nothing should happen (real zero, other nominal cutoffs) is the placebo.
The conjecture under test is that the jump is largest when inflation is low but positive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

BIN = 0.005
BW = 0.05                      # bandwidth of the density-jump estimator, each side of the cutoff
NEAR_HOLDS = ["0.5-2y", "2-5y", "5-10y"]
PLACEBO_HOLDS = ["5-10y", "10-20y"]   # holds long enough that real zero sits well above nominal zero
PLACEBO_CUTS = (-0.15, -0.10, -0.075, 0.075, 0.10, 0.15)   # windows stay clear of [-2.5%, +2.5%]
DIFFS = [-10000, -5000, -2500, -1000, 0, 1000, 2500, 5000, 10000]
PERIODS = [(1996, 2007, "1996-2007"), (2008, 2012, "2008-2012"), (2013, 2021, "2013-2021"),
           (2022, 2023, "2022-2023"), (2024, 2026, "2024-2026")]
SEED = 20261113


def pct(x: float, nd: int = 1) -> str:
    return "nan" if pd.isna(x) else f"{100 * x:.{nd}f}%"


# ----------------------------------------------------------------------------- estimator
def density_jump(counts: pd.Series, cut: float, bw: float = BW, drop_exact: float = 0.0
                 ) -> tuple[float, float, int, int]:
    """Log jump in density at ``cut`` from a histogram of 0.5% bins.

    ``counts`` is indexed by integer bin k (interval [k*BIN, (k+1)*BIN)). Weighted least squares
    of log count on the bin centre is fitted separately on each side, within ``bw`` of the cut,
    with Poisson weights; theta = intercept(right) - intercept(left) at the cut. ``drop_exact``
    removes that many observations from the first bin right of the cut (resales at exactly the
    old price), so the jump can be measured with and without them. Returns theta, its standard
    error and the counts used on each side.
    """
    k0 = int(round(cut / BIN))
    nb = int(round(bw / BIN))
    left = counts.reindex(range(k0 - nb, k0)).fillna(0.0).astype(float)
    right = counts.reindex(range(k0, k0 + nb)).fillna(0.0).astype(float)
    right.iloc[0] = max(right.iloc[0] - drop_exact, 0.0)
    out = []
    for side in (left, right):
        x = (side.index.to_numpy() + 0.5) * BIN - cut
        y = side.to_numpy()
        m = y > 0
        if m.sum() < 3:
            return np.nan, np.nan, int(left.sum()), int(right.sum())
        X = np.column_stack([np.ones(m.sum()), x[m]])
        w = y[m]
        XtW = X.T * w
        V = np.linalg.inv(XtW @ X)
        b = V @ XtW @ np.log(y[m])
        out.append((b[0], V[0, 0]))
    (aL, vL), (aR, vR) = out
    return aR - aL, float(np.sqrt(vL + vR)), int(left.sum()), int(right.sum())


def missing_share(theta_ex: float) -> float:
    """Share of the resales that would have landed just below the cutoff and are missing there,
    when the density just above (without exact resales) is the counterfactual: 1 - exp(-theta)."""
    return float(-np.expm1(-theta_ex))


def hist_series(h: pd.DataFrame, kind: str, **sel) -> pd.Series:
    d = h[h.kind == kind]
    for k, v in sel.items():
        d = d[d[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    d = d[(d.bin != -999) & (d.bin != 999)]
    return d.groupby("bin").n.sum()


def annual_inflation(cpi: pd.DataFrame) -> pd.Series:
    c = cpi.assign(year=pd.to_datetime(cpi.month).dt.year).groupby("year").cpi.mean()
    return (c / c.shift(1) - 1).rename("inflation")


def period_of(y: int) -> str | None:
    for lo, hi, lab in PERIODS:
        if lo <= y <= hi:
            return lab
    return None


# ----------------------------------------------------------------------------- main
def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    S: list[str] = []
    line = S.append

    h = pd.read_csv(DATA / "pairs_hist.csv")
    cells = pd.read_csv(DATA / "pairs_cells.csv")
    diffs = pd.read_csv(DATA / "pairs_diffs.csv")
    monthly = pd.read_csv(DATA / "pairs_monthly.csv", parse_dates=["month"])
    sales = pd.read_csv(DATA / "sales_monthly.csv", parse_dates=["month"])
    cpi = pd.read_csv(DATA / "uk_cpi_monthly.csv")
    blog = dict(l.split("\t") for l in (DATA / "build_log.txt").read_text().splitlines() if "\t" in l)
    infl = annual_inflation(cpi)
    full_years = sorted(y for y in cells.y.unique()
                        if monthly[monthly.month.dt.year == y].month.dt.month.nunique() == 12)
    last_year = int(cells.y.max())

    line("REPEAT SALES IN ENGLAND AND WALES: every number used in the article")
    line("=" * 72)
    line("")
    line("[0] Data")
    main_cells = cells[cells.first_sale_new == "N"]
    line(f"Standard (category A) sales of houses and flats with a usable address: {int(blog['standard_sales']):,}")
    line(f"Consecutive sales of the same address: {int(blog['raw_pairs']):,}; kept after filters: "
         f"{int(blog['clean_pairs']):,}; of which first sale a new build: "
         f"{int(cells[cells.first_sale_new == 'Y'].n.sum()):,} (set aside)")
    line(f"Main sample: {int(main_cells.n.sum()):,} repeat sales, resales {blog['first_resale_month'][:7]} "
         f"to {blog['last_resale_month'][:7]}; last complete year {max(full_years)}")
    by_reg = main_cells.groupby("region").n.sum()
    line("By region: " + ", ".join(f"{k} {v:,}" for k, v in by_reg.sort_values(ascending=False).items()))
    by_hold = main_cells.groupby("hold").n.sum()
    line("By holding period: " + ", ".join(f"{k} {by_hold.get(k, 0):,}" for k in
                                             ["0.5-2y", "2-5y", "5-10y", "10-20y", "20y+"]))

    # ---- [1] losses by resale year ----------------------------------------------------
    yr = main_cells.groupby("y")[["n", "nominal_loss", "real_loss", "hidden_loss",
                                  "exact_old_price"]].sum()
    yr["p_nominal"] = yr.nominal_loss / yr.n
    yr["p_real"] = yr.real_loss / yr.n
    yr["hidden_share"] = yr.hidden_loss / yr.real_loss
    yr["p_exact"] = yr.exact_old_price / yr.n
    yr = yr.join(infl, how="left")
    lon = main_cells[main_cells.region == "London"].groupby("y")[["n", "nominal_loss", "real_loss",
                                                                  "hidden_loss"]].sum()
    yr["london_p_nominal"] = lon.nominal_loss / lon.n
    yr["london_p_real"] = lon.real_loss / lon.n
    yr["london_hidden_share"] = lon.hidden_loss / lon.real_loss
    yr.to_csv(RESULTS / "losses_by_year.csv", float_format="%.6g")
    line("")
    line("[1] Resales at a loss, by year of resale (England and Wales; London)")
    for y, r in yr.iterrows():
        line(f"  {y}{'*' if y not in full_years else ' '} n={int(r.n):7,d}  CPI {pct(r.inflation):>6s}  "
             f"nominal loss {pct(r.p_nominal):>6s}  real loss {pct(r.p_real):>6s}  hidden/real "
             f"{pct(r.hidden_share):>6s}  exact old price {pct(r.p_exact, 2):>6s}  | London nominal "
             f"{pct(r.london_p_nominal):>6s} real {pct(r.london_p_real):>6s} hidden "
             f"{pct(r.london_hidden_share):>6s}")
    line("  (* incomplete year)")
    per = main_cells.assign(period=main_cells.y.map(period_of)).dropna(subset=["period"])
    pr = per.groupby("period")[["n", "nominal_loss", "real_loss", "hidden_loss", "exact_old_price"]].sum()
    for p_, r in pr.iterrows():
        yrs = [y for lo, hi, lab in PERIODS if lab == p_ for y in range(lo, hi + 1)]
        line(f"  {p_}: n={int(r.n):,}, nominal loss {pct(r.nominal_loss / r.n)}, real loss "
             f"{pct(r.real_loss / r.n)}, hidden/real {pct(r.hidden_loss / r.real_loss)}, exact old "
             f"price {pct(r.exact_old_price / r.n, 2)}; mean CPI inflation "
             f"{pct(infl.reindex(yrs).mean())}")

    # ---- [2] exact old price and its placebos --------------------------------------------
    line("")
    line("[2] Resales at exactly the old price, against resales at exact round differences")
    dsum = diffs[diffs.columns[diffs.columns.str.startswith("d")]].sum()
    tot = diffs.n.sum()
    line("  all years, share of resales at old price + k GBP: " + ", ".join(
        f"{c[1:]}: {pct(v / tot, 3)}" for c, v in dsum.items()))
    ratio = dsum["d0"] / dsum[["d-5000", "d5000"]].mean()
    line(f"  exact old price vs mean of -5,000 and +5,000: {ratio:.2f}x; vs -1,000 and +1,000: "
         f"{dsum['d0'] / dsum[['d-1000', 'd1000']].mean():.2f}x; vs -10,000 and +10,000: "
         f"{dsum['d0'] / dsum[['d-10000', 'd10000']].mean():.2f}x")
    line(f"  below vs above at the same distance: -1,000/+1,000 {dsum['d-1000'] / dsum['d1000']:.2f}; "
         f"-2,500/+2,500 {dsum['d-2500'] / dsum['d2500']:.2f}; -5,000/+5,000 "
         f"{dsum['d-5000'] / dsum['d5000']:.2f}; -10,000/+10,000 {dsum['d-10000'] / dsum['d10000']:.2f}")
    dp = diffs.assign(period=diffs.y.map(period_of)).dropna(subset=["period"]).groupby("period").sum(
        numeric_only=True)
    drows = []
    for p_, r in dp.iterrows():
        rr = {"period": p_, "n": int(r.n), "share_exact": r.d0 / r.n,
              "exact_vs_pm5000": r.d0 / ((r["d-5000"] + r["d5000"]) / 2),
              "below_above_5000": r["d-5000"] / r["d5000"],
              "below_above_1000": r["d-1000"] / r["d1000"]}
        drows.append(rr)
        line(f"  {p_}: exact {pct(rr['share_exact'], 2)}, exact / mean(+-5,000) {rr['exact_vs_pm5000']:.2f}x, "
             f"-5,000/+5,000 {rr['below_above_5000']:.2f}, -1,000/+1,000 {rr['below_above_1000']:.2f}")
    pd.DataFrame(drows).to_csv(RESULTS / "exact_price_by_period.csv", index=False, float_format="%.6g")
    dh = diffs[diffs.hold.isin(NEAR_HOLDS)].groupby("hold").sum(numeric_only=True)
    for k in NEAR_HOLDS:
        r = dh.loc[k]
        line(f"  holding {k}: exact {pct(r.d0 / r.n, 2)}, exact / mean(+-5,000) "
             f"{r.d0 / ((r['d-5000'] + r['d5000']) / 2):.2f}x")

    # ---- [3] density jump at nominal zero ----------------------------------------------
    line("")
    line(f"[3] Jump in density at a nominal change of zero (log, bandwidth {BW:.0%} each side; "
         f"holdings under 10 years)")
    exact_by = {}
    for (y, reg, hold), r in diffs.set_index(["y", "region", "hold"]).iterrows():
        exact_by[(y, reg, hold)] = r.d0

    def exact_for(years, regions=None, holds=NEAR_HOLDS):
        d = diffs[diffs.y.isin(years) & diffs.hold.isin(holds)]
        if regions is not None:
            d = d[d.region.isin(regions)]
        return float(d.d0.sum())

    def jumps(years, regions=None, holds=NEAR_HOLDS):
        sel = {"y": list(years), "hold": holds}
        if regions is not None:
            sel["region"] = regions
        hn = hist_series(h, "nominal", **sel)
        sel_p = dict(sel, hold=PLACEBO_HOLDS)
        hr = hist_series(h, "real_excl_exact", **sel_p)
        ex = exact_for(years, regions, holds)
        out = {"n": int(hn.sum())}
        out["theta"], out["se"], out["nL"], out["nR"] = density_jump(hn, 0.0)
        out["theta_ex"], out["se_ex"], _, _ = density_jump(hn, 0.0, drop_exact=ex)
        out["theta_real"], out["se_real"], _, _ = density_jump(hr, 0.0)
        for c in PLACEBO_CUTS:
            out[f"theta_{c:+.2f}"], out[f"se_{c:+.2f}"], _, _ = density_jump(hn, c)
        return out

    J = []
    for y in sorted(h.y.unique()):
        r = jumps([y])
        r.update({"y": y, "inflation": infl.get(y, np.nan), "full_year": y in full_years})
        J.append(r)
    JY = pd.DataFrame(J).set_index("y")
    JY.to_csv(RESULTS / "density_jump_by_year.csv", float_format="%.6g")
    allr = jumps(sorted(h.y.unique()))
    line(f"  all years: theta {allr['theta']:.3f} (se {allr['se']:.3f}) -> density {pct(np.expm1(allr['theta']))} "
         f"higher just above zero; without exact old-price resales {allr['theta_ex']:.3f} "
         f"(se {allr['se_ex']:.3f}), i.e. {pct(missing_share(allr['theta_ex']))} of the resales that would "
         f"have landed just below the old price are missing; real zero (placebo) {allr['theta_real']:.3f} "
         f"(se {allr['se_real']:.3f})")
    line("  nominal placebo cutoffs: " + ", ".join(
        f"{100 * c:+.1f}%: {allr[f'theta_{c:+.2f}']:.3f} (se {allr[f'se_{c:+.2f}']:.3f})" for c in PLACEBO_CUTS))
    hn_all = hist_series(h, "nominal", y=sorted(h.y.unique()), hold=NEAR_HOLDS)
    ex_all = exact_for(sorted(h.y.unique()))
    sens = []
    for bw in (0.025, 0.05, 0.075, 0.10):
        th, se, _, _ = density_jump(hn_all, 0.0, bw=bw, drop_exact=ex_all)
        sens.append(f"{100 * bw:.1f}%: {th:.3f} (se {se:.3f}) = missing {pct(missing_share(th))}")
    line("  bandwidth sensitivity (without exact): " + "; ".join(sens))
    for p_lo, p_hi, p_ in PERIODS:
        r = jumps(range(p_lo, p_hi + 1))
        line(f"  {p_}: theta {r['theta']:.3f} ({r['se']:.3f}), without exact {r['theta_ex']:.3f} "
             f"({r['se_ex']:.3f}) = missing {pct(missing_share(r['theta_ex']))}, real-zero placebo "
             f"{r['theta_real']:.3f} ({r['se_real']:.3f}), "
             f"nominal placebos " + ", ".join(f"{100 * c:+.1f}% {r[f'theta_{c:+.2f}']:.3f}" for c in PLACEBO_CUTS)
             + f"  [n={r['n']:,}]")
    for reg in ["London", "South", "North and Midlands", "Wales"]:
        r = jumps(sorted(h.y.unique()), [reg])
        line(f"  {reg}: theta {r['theta']:.3f} ({r['se']:.3f}), without exact {r['theta_ex']:.3f} "
             f"= missing {pct(missing_share(r['theta_ex']))}, "
             f"real-zero placebo {r['theta_real']:.3f}  [n={r['n']:,}]")
    for hold in NEAR_HOLDS:
        r = jumps(sorted(h.y.unique()), None, [hold])
        line(f"  holding {hold}: theta {r['theta']:.3f} ({r['se']:.3f}), without exact "
             f"{r['theta_ex']:.3f} = missing {pct(missing_share(r['theta_ex']))}, real-zero placebo "
             f"{r['theta_real']:.3f}  [n={r['n']:,}]")

    # ---- [4] does the jump depend on inflation? ----------------------------------------
    line("")
    line("[4] Jump at nominal zero against CPI inflation in the year of resale (complete years)")
    jy = JY[JY.full_year & JY.theta.notna() & JY.inflation.notna()]
    rho = stats.spearmanr(jy.inflation, jy.theta)
    rho_ex = stats.spearmanr(jy.inflation, jy.theta_ex)
    rho_r = stats.spearmanr(jy.inflation, jy.theta_real, nan_policy="omit")
    line(f"  {len(jy)} years. Spearman(theta, inflation) {rho.statistic:.2f} (p {rho.pvalue:.3f}); "
         f"without exact resales {rho_ex.statistic:.2f} (p {rho_ex.pvalue:.3f}); real-zero placebo "
         f"{rho_r.statistic:.2f} (p {rho_r.pvalue:.3f}; holdings of 5-20 years, {int(jy.theta_real.notna().sum())} years)")
    for col, se_col, lab in [("theta", "se", "with exact"), ("theta_ex", "se_ex", "without exact")]:
        w = 1 / jy[se_col] ** 2
        X = np.column_stack([np.ones(len(jy)), jy.inflation])
        XtW = X.T * w.to_numpy()
        A = np.linalg.inv(XtW @ X)
        b = A @ XtW @ jy[col].to_numpy()
        resid = jy[col].to_numpy() - X @ b
        meat = (X.T * (w.to_numpy() * resid) ** 2) @ X
        Vb = A @ meat @ A * len(jy) / (len(jy) - 2)
        line(f"  weighted fit ({lab}): theta = {b[0]:.3f} {b[1]:+.2f} x inflation (robust se of slope "
             f"{np.sqrt(Vb[1, 1]):.2f}); at 2% inflation {b[0] + 0.02 * b[1]:.3f}, at 8% "
             f"{b[0] + 0.08 * b[1]:.3f}")
    lowpos = jy[(jy.inflation > 0) & (jy.inflation <= 0.025)]
    high = jy[jy.inflation > 0.04]
    line(f"  years with inflation in (0, 2.5%]: {len(lowpos)}, median missing share "
         f"{pct(missing_share(lowpos.theta_ex.median()))}; above 4%: {len(high)} "
         f"({', '.join(str(i) for i in high.index)}), median missing share "
         f"{pct(missing_share(high.theta_ex.median()))}")
    for y, r in jy.iterrows():
        line(f"    {y}: CPI {pct(r.inflation)}, theta {r.theta:.3f}, without exact {r.theta_ex:.3f} "
             f"(se {r.se_ex:.3f}) = missing {pct(missing_share(r.theta_ex))}, real-zero placebo "
             f"{r.theta_real:.3f}, n {int(r.n):,}")

    # ---- [4b] region-years: inflation and market conditions together ----------------------
    line("")
    line("[4b] Region-years: missing share against inflation and the local market (WLS, region fixed effects, "
         "SE clustered by year)")
    rows = []
    for (y, reg), _ in h[h.y.isin(full_years)].groupby(["y", "region"]):
        r = jumps([y], [reg])
        c = main_cells[(main_cells.y == y) & (main_cells.region == reg) & main_cells.hold.isin(["2-5y"])]
        rows.append({"y": y, "region": reg, "theta_ex": r["theta_ex"], "se_ex": r["se_ex"],
                     "inflation": infl.get(y, np.nan),
                     "median_lr_2_5y": np.average(c.median_lr, weights=c.n) if c.n.sum() else np.nan,
                     "n": r["n"]})
    RY = pd.DataFrame(rows).dropna()
    RY = RY[np.isfinite(RY.se_ex) & (RY.se_ex > 0)]
    RY.to_csv(RESULTS / "density_jump_region_year.csv", index=False, float_format="%.6g")
    regs = sorted(RY.region.unique())
    Xcols = [np.ones(len(RY)), RY.inflation.to_numpy(), RY.median_lr_2_5y.to_numpy()]
    Xcols += [(RY.region == g).to_numpy(float) for g in regs[1:]]
    X = np.column_stack(Xcols)
    wv = (1 / RY.se_ex ** 2).to_numpy()
    yv = RY.theta_ex.to_numpy()
    A_ = np.linalg.inv((X.T * wv) @ X)
    bb = A_ @ (X.T * wv) @ yv
    e = yv - X @ bb
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in RY.y.unique():
        mk = (RY.y == g).to_numpy()
        sc = (X[mk].T * wv[mk]) @ e[mk]
        meat += np.outer(sc, sc)
    G = RY.y.nunique()
    Vb = A_ @ meat @ A_ * G / (G - 1)
    se_b = np.sqrt(np.diag(Vb))
    line(f"  {len(RY)} region-years, {G} years. theta_ex = a {bb[1]:+.2f} (se {se_b[1]:.2f}) x inflation "
         f"{bb[2]:+.2f} (se {se_b[2]:.2f}) x median nominal change of 2-5y holdings + region effects")
    line(f"  a 5 pp rise in inflation -> missing share changes by about "
         f"{pct(missing_share(0.30 + 0.05 * bb[1]) - missing_share(0.30))} at a base of {pct(missing_share(0.30))}; "
         f"a 10 pp lower median nominal change -> {pct(missing_share(0.30 - 0.10 * bb[2]) - missing_share(0.30))}")

    # ---- [5] volume -----------------------------------------------------------------------
    line("")
    line("[5] Transaction volume (all standard sales, 12-month totals)")
    sm = sales.groupby("month").sales.sum().sort_index()
    roll = sm.rolling(12).sum().dropna()
    for a, b_ in [("2007-06", "2009-06"), ("2022-06", "2023-12"), ("2022-06", "2025-06")]:
        va = roll.loc[a:a].iloc[0] if len(roll.loc[a:a]) else np.nan
        vb = roll.loc[b_:b_].iloc[0] if len(roll.loc[b_:b_]) else np.nan
        line(f"  12 months to {a}: {va:,.0f}; to {b_}: {vb:,.0f}; change {pct(vb / va - 1)}")
    (pd.DataFrame({"sales_12m": roll})).to_csv(RESULTS / "volume_12m.csv", float_format="%.6g")

    # ---- [6] does the estimator recover a known floor? -------------------------------------
    line("")
    line("[6] Simulation check: normal log changes, a share q of would-be nominal losses sold at exactly "
         "the old price (800,000 draws each)")
    for mu, sd in [(-0.05, 0.08), (0.0, 0.05), (0.02, 0.15), (0.10, 0.10)]:
        res = []
        for q in (0.0, 0.3):
            rng = np.random.default_rng(SEED)
            x = rng.normal(mu, sd, 800_000)
            anch = (x < 0) & (rng.random(x.size) < q)
            x = np.where(anch, 0.0, x)
            cnt = pd.Series(np.floor(x / BIN).astype(int)).value_counts().sort_index()
            th = density_jump(cnt, 0.0, drop_exact=float(anch.sum()))[0]
            res.append(missing_share(th))
        line(f"  mean {mu:+.2f}, sd {sd:.2f}: estimated missing share {pct(res[0])} when q = 0, "
             f"{pct(res[1])} when q = 30%")

    (RESULTS / "summary.txt").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
