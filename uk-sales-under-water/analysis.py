"""Lock-in below the purchase price and the fall in sales: every number used in the article.

    python analysis.py            # writes results/*.csv and results/summary.txt

Reads the aggregates in ``data/`` written by ``build.py``.

[1] Resale rate by year for the stock of owners who bought since 1995, and the share of that
    stock below its purchase price according to the area index (m < 0).
[2] The lock-in curve: a linear probability model of reselling within the year, on 2% bands of
    m (owners with m in [0, 2%) are the reference), with postcode area x year fixed effects
    (absorbing everything common to an area in a year: its market, its credit conditions) and
    whole-years-since-purchase x flat and new-build x holding-band effects. Weighted by stock;
    standard errors clustered by postcode area.
[3] Counterfactual: the resale rate each year if owners below their purchase price had sold at
    the rate of owners just above it (the m < 0 band effects set to zero).
[4] Districts within the same postcode area: the change in the log resale rate over a fall in
    prices against the share of each district's stock below its purchase price, with area fixed
    effects, controls for the age, type and new-build composition of the stock, and the change
    over the two previous years; and the same regression for the earlier change as a placebo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RES = HERE / "data", HERE / "results"
M_BIN = 0.02
M_LO, M_HI = -15, 50           # 2% bins kept separate: [-30%, +100%); outside pooled
REF_BIN = 0                     # [0, 2%)
PERIODS = [("all", 2000, 2025), ("2001-2007", 2001, 2007), ("2008-2012", 2008, 2012),
           ("2013-2021", 2013, 2021), ("2022-2025", 2022, 2025)]
MIN_BIN_STOCK = 5000
EPISODES = [("2007-2009", 2007, 2009, 2005), ("2007-2010", 2007, 2010, 2005),
            ("2022-2024", 2022, 2024, 2020)]
M_BAND_ORDER = ["below_m10", "m10_0", "0_10", "10_25", "25_50", "above_50"]
DUR_BAND_EDGES = [(0, 2), (2, 5), (5, 10), (10, 99)]


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------- estimation
def within(df: pd.DataFrame, cols: list[str], group: str, w: str) -> pd.DataFrame:
    """Weighted demeaning of ``cols`` within ``group``."""
    _, inv = np.unique(df[group].to_numpy(), return_inverse=True)
    X = demean(df[cols].to_numpy(float), inv, df[w].to_numpy(float))
    return pd.DataFrame(X, columns=cols, index=df.index)


def demean(X: np.ndarray, inv: np.ndarray, wt: np.ndarray) -> np.ndarray:
    sw = np.bincount(inv, weights=wt)
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = X[:, j] - (np.bincount(inv, weights=wt * X[:, j]) / sw)[inv]
    return out


def wls_cluster(y: np.ndarray, X: np.ndarray, w: np.ndarray, cl: np.ndarray):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    xtx = Xw.T @ Xw
    beta = np.linalg.lstsq(xtx, Xw.T @ yw, rcond=None)[0]
    u = y - X @ beta
    _, inv = np.unique(cl, return_inverse=True)
    G = inv.max() + 1
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, inv, X * (w * u)[:, None])
    bread = np.linalg.pinv(xtx)
    V = G / (G - 1) * bread @ (sc.T @ sc) @ bread
    return beta, np.sqrt(np.diag(V)), V


def lock_in_curve(hc: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    d = hc[(hc.year >= a) & (hc.year <= b)].copy()
    d["mb"] = d.m_bin.clip(M_LO - 1, M_HI)
    # bins with too little stock in this period are pooled with their neighbour towards zero
    tot = d.groupby("mb").stock.sum()
    thin = sorted(int(x) for x in tot[tot < MIN_BIN_STOCK].index)
    for x in thin:
        d.loc[d.mb == x, "mb"] = x + 1 if x < 0 else x - 1
    d["y"] = d.resales / d.stock
    d["ay"] = d.area + "|" + d.year.astype(str)
    d["dcell"] = (d.dur_year.astype(str) + "|" + d.flat.astype(str) + "|" + d.new.astype(str))
    bins = sorted(int(x) for x in d.mb.unique() if x != REF_BIN)
    dcells = sorted(d.dcell.unique())
    ref_d = "3|False|False" if "3|False|False" in dcells else dcells[0]
    dcells = [c_ for c_ in dcells if c_ != ref_d]
    mb = d.mb.to_numpy()
    dc = d.dcell.to_numpy()
    X = np.zeros((len(d), len(bins) + len(dcells)), dtype=float)
    for j, x in enumerate(bins):
        X[:, j] = mb == x
    for j, c_ in enumerate(dcells):
        X[:, len(bins) + j] = dc == c_
    _, inv = np.unique(d.ay.to_numpy(), return_inverse=True)
    wt = d.stock.to_numpy(float)
    Xd = demean(X, inv, wt)
    yd = demean(d[["y"]].to_numpy(float), inv, wt)[:, 0]
    del X
    beta, se, _ = wls_cluster(yd, Xd, wt, d.area.to_numpy())
    cols = [f"m_{x}" for x in bins]
    res = []
    stock_by = d.groupby("mb").stock.sum()
    resales_by = d.groupby("mb").resales.sum()
    base = resales_by.get(REF_BIN, np.nan) / stock_by.get(REF_BIN, np.nan)
    for x in [REF_BIN] + bins:
        if x == REF_BIN:
            bb, ss = 0.0, 0.0
        else:
            j = cols.index(f"m_{x}")
            bb, ss = beta[j], se[j]
        res.append({"m_bin": x, "m_lo": x * M_BIN, "effect": bb, "se": ss,
                    "stock": int(stock_by.get(x, 0)), "resales": int(resales_by.get(x, 0)),
                    "raw_rate": resales_by.get(x, 0) / max(stock_by.get(x, 1), 1),
                    "ref_rate": base})
    out = pd.DataFrame(res).sort_values("m_bin")
    out.attrs["n_cells"] = len(d)
    out.attrs["pooled"] = thin
    return out


def counterfactual(hc: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Resale rate by year with and without the effect of being below the purchase price."""
    eff = curve.set_index("m_bin").effect
    d = hc.copy()
    d["mb"] = d.m_bin.clip(M_LO - 1, M_HI)
    d["eff"] = d.mb.map(eff)
    # pooled bins take the effect of the bin they were pooled into
    for x in curve.attrs.get("pooled", []):
        tgt = x + 1 if x < 0 else x - 1
        while tgt not in eff.index:
            tgt = tgt + 1 if tgt < 0 else tgt - 1
        d.loc[d.mb == x, "eff"] = eff[tgt]
    d["eff"] = d.eff.fillna(0.0)
    d["under"] = d.m_bin < 0
    # wider counterfactual: everyone with m below +10% sells like owners with m in [+10%, +20%)
    plateau = eff[(eff.index >= 5) & (eff.index < 10)]
    plateau = float(plateau.mean()) if len(plateau) else 0.0
    d["low"] = d.m_bin < 5
    rows = []
    for y, g in d.groupby("year"):
        st = g.stock.sum()
        lock = (g.stock * np.where(g.under, -g.eff, 0.0)).sum() / st
        wide = (g.stock * np.where(g.low, plateau - g.eff, 0.0)).sum() / st
        rows.append({"year": y, "stock": int(st), "resales": int(g.resales.sum()),
                     "rate": g.resales.sum() / st, "share_under": g.stock[g.under].sum() / st,
                     "share_under_10": g.stock[g.m_bin < -5].sum() / st,
                     "share_below_plus10": g.stock[g.low].sum() / st,
                     "lock_in_drag": lock, "position_drag": wide})
    out = pd.DataFrame(rows)
    out["rate_without_lock_in"] = out.rate + out.lock_in_drag
    out["rate_without_position"] = out.rate + out.position_drag
    out.attrs["plateau"] = plateau
    return out


def district_test(dy: pd.DataFrame, y0: int, y1: int, ypre: int) -> dict:
    """Change in the log resale rate y0 -> y1 against the share under water at the start of y1."""
    a = dy[dy.year == y0].set_index("district")
    b = dy[dy.year == y1].set_index("district")
    p = dy[dy.year == ypre].set_index("district")
    idx = a.index.intersection(b.index).intersection(p.index)
    a, b, p = a.loc[idx], b.loc[idx], p.loc[idx]
    keep = (a.resales >= 20) & (b.resales >= 20) & (p.resales >= 20) & (b.stock >= 500)
    a, b, p = a[keep], b[keep], p[keep]
    d = pd.DataFrame(index=a.index)
    d["dy"] = np.log(b.resales / b.stock) - np.log(a.resales / a.stock)
    d["pre"] = np.log(a.resales / a.stock) - np.log(p.resales / p.stock)
    d["under"] = (b.stock_below_m10 + b.stock_m10_0) / b.stock
    d["under10"] = b.stock_below_m10 / b.stock
    d["d0_2"] = b.stock_d0_2 / b.stock
    d["d2_5"] = b.stock_d2_5 / b.stock
    d["d5_10"] = b.stock_d5_10 / b.stock
    d["flat"] = b.stock_flat / b.stock
    d["new"] = b.stock_new / b.stock
    d["area"] = b.area
    d["w"] = b.stock.astype(float)
    out = {"episode": f"{y0}-{y1}", "n_districts": len(d), "n_areas": d.area.nunique(),
           "mean_under": float(np.average(d.under, weights=d.w)),
           "p10_under": float(d.under.quantile(0.1)), "p90_under": float(d.under.quantile(0.9)),
           "mean_dy": float(np.average(d.dy, weights=d.w))}
    ctrl = ["d0_2", "d2_5", "d5_10", "flat", "new"]
    specs = {"under": ["under"], "under_controls": ["under"] + ctrl,
             "under_controls_pre": ["under"] + ctrl + ["pre"]}
    for name, xs in specs.items():
        W = within(d, ["dy"] + xs, "area", "w")
        beta, se, _ = wls_cluster(W.dy.to_numpy(), W[xs].to_numpy(), d.w.to_numpy(), d.area.to_numpy())
        out[f"{name}_b"], out[f"{name}_se"] = float(beta[0]), float(se[0])
    # placebo: does the later under-water share predict the earlier change?
    W = within(d, ["pre", "under"] + ctrl, "area", "w")
    beta, se, _ = wls_cluster(W.pre.to_numpy(), W[["under"] + ctrl].to_numpy(), d.w.to_numpy(),
                              d.area.to_numpy())
    out["placebo_pre_b"], out["placebo_pre_se"] = float(beta[0]), float(se[0])
    # deeper losses
    W = within(d, ["dy", "under10", "under"] + ctrl + ["pre"], "area", "w")
    xs = ["under10", "under"] + ctrl + ["pre"]
    beta, se, _ = wls_cluster(W.dy.to_numpy(), W[xs].to_numpy(), d.w.to_numpy(), d.area.to_numpy())
    out["deep_b"], out["deep_se"] = float(beta[0]), float(se[0])
    out["shallow_b"], out["shallow_se"] = float(beta[1]), float(se[1])
    out["mean_under10"] = float(np.average(d.under10, weights=d.w))
    out["p10_under10"], out["p90_under10"] = float(d.under10.quantile(0.1)), float(d.under10.quantile(0.9))
    W = within(d, ["pre", "under10", "under"] + ctrl, "area", "w")
    xs = ["under10", "under"] + ctrl
    beta, se, _ = wls_cluster(W.pre.to_numpy(), W[xs].to_numpy(), d.w.to_numpy(), d.area.to_numpy())
    out["placebo_deep_b"], out["placebo_deep_se"] = float(beta[0]), float(se[0])
    # binned relation for the figure: residualise on area and controls, 10 bins of the share
    W = within(d, ["dy", "under"] + ctrl + ["pre"], "area", "w")
    Z = W[ctrl + ["pre"]].to_numpy()
    sw = np.sqrt(d.w.to_numpy())
    def resid(v):
        g = np.linalg.lstsq(Z * sw[:, None], v * sw, rcond=None)[0]
        return v - Z @ g
    ry, rx = resid(W.dy.to_numpy()), resid(W.under.to_numpy())
    q = pd.qcut(rx, 10, labels=False, duplicates="drop")
    binned = pd.DataFrame({"q": q, "x": rx + np.average(d.under, weights=d.w),
                           "y": ry + np.average(d.dy, weights=d.w), "w": d.w})
    bins = binned.groupby("q").apply(lambda g: pd.Series({"x": np.average(g.x, weights=g.w),
                                                           "y": np.average(g.y, weights=g.w),
                                                           "n": len(g)}), include_groups=False)
    bins = bins.reset_index().assign(episode=out["episode"])
    return out, bins


def main() -> int:
    RES.mkdir(exist_ok=True)
    L: list[str] = []
    w = L.append
    log = dict(line.split("\t") for line in (DATA / "build_log.txt").read_text().splitlines())
    hc = pd.read_csv(DATA / "hazard_cells.csv.gz")
    dy = pd.read_csv(DATA / "district_year.csv.gz")
    nat = pd.read_csv(DATA / "national_monthly.csv")
    w("OWNERS UNDER WATER AND THE FALL IN SALES: every number used in the article")
    w("=" * 76)
    w("")
    w("[0] Data")
    w(f"Standard sales {int(log['standard_sales']):,}; ownership spells {int(log['spells']):,}; "
      f"postcode areas with an index {log['areas_in_index']}; districts {int(log['districts']):,}")
    w(f"Owner-years {int(log['owner_years']):,} and resales {int(log['resales']):,}, years {log['years']}; "
      f"hazard cells {len(hc):,}")

    # [1] national series
    cf_all = None
    yr = hc.groupby("year").agg(stock=("stock", "sum"), resales=("resales", "sum")).reset_index()
    sales = nat.groupby("year").sales.sum()
    w("")
    w("[1] By year: stock (owners who bought since 1995), resale rate, share below the purchase price, "
      "all standard sales")
    under = hc.assign(u=hc.m_bin < 0, u10=hc.m_bin < -5).groupby("year").apply(
        lambda g: pd.Series({"under": g.stock[g.u].sum() / g.stock.sum(),
                             "under10": g.stock[g.u10].sum() / g.stock.sum()}), include_groups=False)
    for r in yr.itertuples():
        w(f"  {r.year}: stock {r.stock:>11,}  rate {pct(r.resales / r.stock, 2)}  under water "
          f"{pct(under.loc[r.year, 'under'])} (below -10%: {pct(under.loc[r.year, 'under10'])})  "
          f"sales {int(sales.get(r.year, 0)):,}")
    rate = (yr.set_index("year").resales / yr.set_index("year").stock)
    for a, b in [(2007, 2009), (2007, 2008), (2022, 2023), (2022, 2024), (2021, 2023)]:
        w(f"  change {a}->{b}: resale rate {100 * (rate[b] / rate[a] - 1):+.1f}%, standard sales "
          f"{100 * (sales[b] / sales[a] - 1):+.1f}%")

    # [2] lock-in curves
    curves = []
    w("")
    w("[2] Lock-in curve: effect on the annual resale probability (percentage points) against owners with "
      "m in [0, 2%), area x year and holding-year x type x new-build effects removed; se clustered by area")
    for name, a, b in PERIODS:
        c = lock_in_curve(hc, a, b)
        c.insert(0, "period", name)
        curves.append(c)
        if name == "all":
            cf_all = c
        ref = c[c.m_bin == REF_BIN].iloc[0]
        w(f"  {name}: {c.attrs['n_cells']:,} cells; reference rate {pct(ref.ref_rate, 2)}; pooled thin bins "
          f"{c.attrs['pooled'][:6]}{'...' if len(c.attrs['pooled']) > 6 else ''}")
        show = c[c.m_bin.isin([-16, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 5, 10, 15, 25, 50])]
        w("    " + "; ".join(f"[{100 * r.m_lo:+.0f}%] {100 * r.effect:+.2f} ({100 * r.se:.2f})"
                             for r in show.itertuples()))
    cdf = pd.concat(curves)
    cdf.to_csv(RES / "lock_in_curve.csv", index=False, float_format="%.6g")

    # [3] counterfactual
    cf = counterfactual(hc, cf_all)
    cf.to_csv(RES / "counterfactual.csv", index=False, float_format="%.6g")
    w("")
    w("[3] Resale rate with and without the lock-in below the purchase price (all-years curve); "
      f"plateau effect of m in [+10%, +20%): {100 * cf.attrs['plateau']:+.2f} pp")
    for r in cf.itertuples():
        w(f"  {r.year}: rate {pct(r.rate, 2)}  under water {pct(r.share_under)}  drag {100 * r.lock_in_drag:.2f} pp"
          f"  without lock-in {pct(r.rate_without_lock_in, 2)}  below +10% {pct(r.share_below_plus10)} "
          f"wide drag {100 * r.position_drag:.2f} pp")
    c = cf.set_index("year")
    for a, b in [(2007, 2009), (2007, 2008), (2022, 2023), (2022, 2024)]:
        fall = c.rate[a] - c.rate[b]
        fall_cf = c.rate_without_lock_in[a] - c.rate_without_lock_in[b]
        fall_w = c.rate_without_position[a] - c.rate_without_position[b]
        w(f"  {a}->{b}: fall {100 * fall:.2f} pp ({100 * fall / c.rate[a]:.1f}%), without lock-in "
          f"{100 * fall_cf:.2f} pp; share of the fall due to lock-in {pct((fall - fall_cf) / fall)}; "
          f"with everyone below +10% selling like +10-20%: {pct((fall - fall_w) / fall)}")
    for name in ["2008-2012", "2022-2025"]:
        cp = cdf[cdf.period == name].copy()
        cp.attrs["pooled"] = []
        cfp = counterfactual(hc, cp).set_index("year")
        a, b = (2007, 2009) if name == "2008-2012" else (2022, 2024)
        fall = cfp.rate[a] - cfp.rate[b]
        fall_cf = cfp.rate_without_lock_in[a] - cfp.rate_without_lock_in[b]
        w(f"  with the {name} curve: {a}->{b} share of the fall due to lock-in {pct((fall - fall_cf) / fall)}; "
          f"drag in {b} {100 * cfp.lock_in_drag[b]:.2f} pp")

    # [4] districts
    w("")
    w("[4] Districts within the same postcode area: change in log resale rate vs share of stock under water")
    res4, bins4 = [], []
    for name, y0, y1, ypre in EPISODES:
        o, bns = district_test(dy, y0, y1, ypre)
        res4.append(o)
        bins4.append(bns)
        w(f"  {o['episode']}: {o['n_districts']:,} districts in {o['n_areas']} areas; under water mean "
          f"{pct(o['mean_under'])} (p10 {pct(o['p10_under'])}, p90 {pct(o['p90_under'])}); mean change "
          f"{o['mean_dy']:+.3f}")
        w(f"    slope: alone {o['under_b']:+.3f} ({o['under_se']:.3f}); with controls {o['under_controls_b']:+.3f} "
          f"({o['under_controls_se']:.3f}); plus pre-trend {o['under_controls_pre_b']:+.3f} "
          f"({o['under_controls_pre_se']:.3f}); placebo on pre-trend {o['placebo_pre_b']:+.3f} "
          f"({o['placebo_pre_se']:.3f})")
        w(f"    deep (below -10%, mean {pct(o['mean_under10'])}, p10 {pct(o['p10_under10'])}, p90 "
          f"{pct(o['p90_under10'])}) {o['deep_b']:+.3f} ({o['deep_se']:.3f}) and any under water "
          f"{o['shallow_b']:+.3f} ({o['shallow_se']:.3f}); placebo deep on pre-trend "
          f"{o['placebo_deep_b']:+.3f} ({o['placebo_deep_se']:.3f})")
    pd.DataFrame(res4).to_csv(RES / "district_tests.csv", index=False, float_format="%.6g")
    pd.concat(bins4).to_csv(RES / "district_binned.csv", index=False, float_format="%.6g")

    # [5] monthly: when did sales fall, and for whom?
    mp_path = DATA / "monthly_position.csv"
    if mp_path.exists():
        mp = pd.read_csv(mp_path)
        mp["date"] = pd.to_datetime(dict(year=mp.year, month=mp.month, day=1))
        tot = mp.groupby("date").agg(stock=("stock", "sum"), resales=("resales", "sum"))
        und = mp[mp.m_band.isin(["below_m10", "m10_0"])].groupby("date").stock.sum()
        und10 = mp[mp.m_band == "below_m10"].groupby("date").stock.sum()
        series = tot.assign(rate=tot.resales / tot.stock, share_under=und.reindex(tot.index).fillna(0) / tot.stock,
                            share_under10=und10.reindex(tot.index).fillna(0) / tot.stock)
        # resale rate by band of m within the 2-5 year holding band (holds composition roughly fixed)
        sub = mp[mp.dur_band == "d2_5"]
        byb = sub.groupby(["date", "m_band"]).agg(stock=("stock", "sum"), resales=("resales", "sum"))
        byb = (byb.resales / byb.stock).unstack()
        out5 = series.join(byb.add_prefix("rate_2_5y_")).reset_index()
        out5.to_csv(RES / "monthly_series.csv", index=False, float_format="%.6g")
        w("")
        w("[5] Monthly: resale rate of the stock and the share of the stock below its purchase price "
          "(m at the end of the previous month)")
        sr = series
        def avg(a, b, col="rate"):
            return float(sr.loc[a:b, col].mean())
        for lab, a, b in [("2007 Q3", "2007-07", "2007-09"), ("2007 Q4", "2007-10", "2007-12"),
                          ("2008 Q1", "2008-01", "2008-03"), ("2008 Q2", "2008-04", "2008-06"),
                          ("2008 Q3", "2008-07", "2008-09"), ("2008 Q4", "2008-10", "2008-12"),
                          ("2009 Q1", "2009-01", "2009-03"), ("2009 Q2", "2009-04", "2009-06"),
                          ("2009 H2", "2009-07", "2009-12"), ("2012", "2012-01", "2012-12"),
                          ("2014", "2014-01", "2014-12"), ("2022 H1", "2022-01", "2022-06"),
                          ("2022 H2", "2022-07", "2022-12"), ("2023 H1", "2023-01", "2023-06"),
                          ("2023 H2", "2023-07", "2023-12"), ("2024", "2024-01", "2024-12")]:
            w(f"  {lab}: monthly resale rate {100 * avg(a, b):.3f}%  under water {pct(avg(a, b, 'share_under'))} "
              f"(below -10%: {pct(avg(a, b, 'share_under10'))})")
        w("  fall in the monthly resale rate, owners held 2-5 years, by band of m: 2007 average to 2009 "
          "average, and 2022 to 2023")
        for col in ["rate_2_5y_" + b for b in M_BAND_ORDER]:
            if col not in out5:
                continue
            x = out5.set_index("date")[col]
            a07, a09 = x.loc["2007-01":"2007-12"].mean(), x.loc["2009-01":"2009-12"].mean()
            a22, a23 = x.loc["2022-01":"2022-12"].mean(), x.loc["2023-01":"2023-12"].mean()
            st09 = sub[(sub.m_band == col[10:]) & (sub.year == 2009)].stock.mean()
            st23 = sub[(sub.m_band == col[10:]) & (sub.year == 2023)].stock.mean()
            def fmt(a, b, st):
                return (f"{100 * a:.3f}% -> {100 * b:.3f}% ({100 * (b / a - 1):+.0f}%, avg stock {st:,.0f})"
                        if np.isfinite(a) and np.isfinite(b) and a > 0 else "n/a")
            w(f"    {col[10:]:>9}: 2007->2009 {fmt(a07, a09, st09)}; 2022->2023 {fmt(a22, a23, st23)}")
    text = "\n".join(L) + "\n"
    (RES / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
