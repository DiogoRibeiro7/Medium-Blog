"""Nominal, real and total-return losses on housing: how much of the loss hides in inflation.

    python analysis.py      # writes results/*.csv and results/summary.txt

Three lenses on the same house:

- N, the nominal price index (what a seller's asking price shows);
- R, the real price index (nominal price deflated by consumer prices);
- T, the real total return (nominal capital gain plus net rental return, deflated).

A *hidden* loss is a real loss while the nominal index did not fall. The script measures how
often each lens shows a loss, how much of every large real drawdown came through inflation
rather than through a nominal price fall, how that depends on inflation, and what rents do to
the picture. Data: Jorda-Schularick-Taylor R6 (annual, 1870-2020) and the OECD analytical house
price indicators (quarterly, 1970-2026). See README.md for definitions and caveats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

SEED = 20261113
N_BOOT = 2000
THRESHOLD = 0.10                      # a drawdown episode is a real fall of at least 10%
THRESHOLDS_SENS = (0.05, 0.20)
HORIZONS_Y = (1, 5, 10, 20)           # holding periods, years (JST)
HORIZONS_Q = (4, 20, 40)              # holding periods, quarters (OECD)
INFL_BINS = [-np.inf, 0.0, 0.02, 0.05, 0.10, np.inf]
INFL_LABELS = ["<=0%", "0-2%", "2-5%", "5-10%", ">10%"]
ERAS = [(1871, 1913, "1871-1913"), (1914, 1945, "1914-1945"), (1946, 1972, "1946-1972"),
        (1973, 1991, "1973-1991"), (1992, 2020, "1992-2020")]
YIELD_HAIRCUTS = (0.0, 0.25, 0.50)    # proportional cuts to JST's net rental return
HYPER = 0.50                          # episodes with inflation above 50% a year are set aside in
                                      # log-weighted sums and in the speed regression
OECD_MEMBERS = sorted(
    "AUS AUT BEL CAN CHE CHL COL CZE DEU DNK ESP EST FIN FRA GBR GRC HUN IRL ISL ISR ITA JPN "
    "KOR LTU LUX LVA MEX NLD NOR NZL POL PRT SVK SVN SWE TUR USA".split()
)
JST_RETURNS_BENCH = 2020              # last JST year; OECD extends prices and yields after it
LAST_FULL_YEAR = 2025


# ----------------------------------------------------------------------------- helpers
def era_of(year: int) -> str | None:
    for lo, hi, lab in ERAS:
        if lo <= year <= hi:
            return lab
    return None


def infl_bin(x: pd.Series) -> pd.Series:
    return pd.cut(x, INFL_BINS, labels=INFL_LABELS, right=True)


def forward_sum(s: pd.Series, h: int) -> pd.Series:
    """Sum of s over (t, t+h], indexed by t. s must be on a gap-free integer index;
    any missing value inside the window gives NaN."""
    return s.rolling(h, min_periods=h).sum().shift(-h)


def segments(s: pd.Series) -> list[pd.Series]:
    """Split a series on an integer index into runs of consecutive non-missing values."""
    s = s.dropna()
    if s.empty:
        return []
    idx = s.index.to_numpy()
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return [s.iloc[a:b] for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(s)])]


def drawdowns(level: pd.Series, threshold: float) -> list[dict]:
    """Peak-to-trough episodes of a positive series on a consecutive index.

    An episode starts at a running peak, lasts while the series is below that peak and ends
    when the peak is regained (recovery) or the data end. Kept if the fall from peak to the
    lowest point is at least ``threshold``.
    """
    v = level.to_numpy(float)
    t = level.index.to_numpy()
    out: list[dict] = []
    n = len(v)
    peak = 0
    k = 1
    while k < n:
        if v[k] >= v[peak]:
            peak = k
            k += 1
            continue
        j, trough = k, k
        while j < n and v[j] < v[peak]:
            if v[j] < v[trough]:
                trough = j
            j += 1
        depth = 1.0 - v[trough] / v[peak]
        if depth >= threshold:
            out.append({"peak": t[peak], "trough": t[trough],
                        "recovery": t[j] if j < n else None, "depth": depth,
                        "i_peak": peak, "i_trough": trough})
        if j >= n:
            break
        peak, k = j, j + 1
    return out


def max_drawdown(x: np.ndarray) -> float:
    """Largest fall from a running maximum, as a fraction."""
    run = np.maximum.accumulate(x)
    return float(np.max(1.0 - x / run))


def decompose(real_log: float, nominal_log: float) -> tuple[float, float]:
    """Split a real log loss L = -real_log > 0 into a visible part (nominal fall, capped at L)
    and a hidden part (the rest, which inflation produced)."""
    L = -real_log
    visible = min(L, max(0.0, -nominal_log))
    return visible, L - visible


def cluster_bootstrap(counts: np.ndarray, num: int, den: int, rng: np.random.Generator,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% percentile interval for sum(counts[:, num]) / sum(counts[:, den]) when countries
    (rows) are resampled with replacement."""
    g = counts.shape[0]
    if g < 2:
        return (np.nan, np.nan)
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot)
    tot = w @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        r = tot[:, num] / tot[:, den]
    r = r[np.isfinite(r)]
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with CR1 cluster-robust standard errors."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((k, k))
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    for gi in range(G):
        m = codes == gi
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V))


# ----------------------------------------------------------------------------- data
def load_jst() -> pd.DataFrame:
    j = pd.read_csv(DATA / "jst_housing.csv")
    frames = []
    for iso, g in j.groupby("iso", sort=True):
        g = g.set_index("year").reindex(range(g.year.min(), g.year.max() + 1))
        g["iso"], g["country"] = iso, g["country"].ffill().bfill()
        g["dlc"] = np.log(g.cpi).diff()
        g["dlp"] = np.log(g.hpnom).diff()                       # KSS national price index
        g["dlcg"] = np.log1p(g.housing_capgain)                  # price series behind JST returns
        g["dltr"] = np.log1p(g.housing_tr)                       # nominal total return
        g["rr"] = g.housing_rent_rtn
        frames.append(g.reset_index(names="year"))
    return pd.concat(frames, ignore_index=True)


def load_oecd() -> pd.DataFrame:
    o = pd.read_csv(DATA / "oecd_house_prices.csv")
    o = o[o.iso.isin(OECD_MEMBERS)]
    w = o.pivot_table(index=["iso", "quarter"], columns="measure", values="value").reset_index()
    w["year"] = w.quarter.str[:4].astype(int)
    w["q"] = w.quarter.str[-1].astype(int)
    w["t"] = w.year * 4 + (w.q - 1)
    frames = []
    for iso, g in w.groupby("iso", sort=True):
        g = g.set_index("t").reindex(range(g.t.min(), g.t.max() + 1))
        g["iso"] = iso
        g["defl"] = g.HPI / g.RHP                                # private consumption deflator
        g["dlp"] = np.log(g.HPI).diff()
        g["dlc"] = np.log(g.defl).diff()
        frames.append(g.reset_index(names="t"))
    out = pd.concat(frames, ignore_index=True)
    out["label"] = (out.t // 4).astype(int).astype(str) + "-Q" + (out.t % 4 + 1).astype(int).astype(str)
    return out


def oecd_annual() -> pd.DataFrame:
    o = pd.read_csv(DATA / "oecd_house_prices.csv")
    o["year"] = o.quarter.str[:4].astype(int)
    full = o.groupby(["iso", "measure", "year"]).value.agg(["mean", "size"]).reset_index()
    full = full[full["size"] == 4]                               # complete years only
    w = full.pivot_table(index=["iso", "year"], columns="measure", values="mean").reset_index()
    w["defl"] = w.HPI / w.RHP
    return w


# ----------------------------------------------------------------------------- holding periods
def windows_annual(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, g in p.groupby("iso", sort=True):
        g = g.set_index("year")
        for h in HORIZONS_Y:
            d = pd.DataFrame({
                "iso": iso, "start": g.index, "h": h,
                "lN": forward_sum(g.dlp, h), "lC": forward_sum(g.dlc, h),
                "lNcg": forward_sum(g.dlcg, h), "lTnom": forward_sum(g.dltr, h),
            })
            rows.append(d)
    w = pd.concat(rows, ignore_index=True)
    w["lR"] = w.lN - w.lC
    w["lRcg"] = w.lNcg - w.lC
    w["lT"] = w.lTnom - w.lC
    w["infl"] = np.expm1(w.lC / w.h)
    w["era"] = w.start.map(lambda y: era_of(int(y) + 1))
    w["ibin"] = infl_bin(w.infl)
    return w


def windows_quarterly(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, g in q.groupby("iso", sort=True):
        g = g.set_index("t")
        for h in HORIZONS_Q:
            rows.append(pd.DataFrame({"iso": iso, "start": g.index, "h": h,
                                      "lN": forward_sum(g.dlp, h), "lC": forward_sum(g.dlc, h)}))
    w = pd.concat(rows, ignore_index=True)
    w["lR"] = w.lN - w.lC
    w["infl"] = np.expm1(w.lC / (w.h / 4))
    w["ibin"] = infl_bin(w.infl)
    w["decade"] = ((w.start // 4) // 10 * 10).astype(int).astype(str) + "s"
    return w


def loss_table(w: pd.DataFrame, n_col: str, r_col: str, t_col: str | None, sample: str,
               rng: np.random.Generator, groupings: dict[str, str | None]) -> pd.DataFrame:
    need = [n_col, r_col] + ([t_col] if t_col else [])
    w = w.dropna(subset=need + ["infl"])
    out = []
    for h, wh in w.groupby("h"):
        for gname, gcol in groupings.items():
            parts = [("all", wh)] if gcol is None else [
                (str(k), v) for k, v in wh.groupby(gcol, observed=True)]
            for gval, d in parts:
                nl = (d[n_col] < 0).astype(int)
                rl = (d[r_col] < 0).astype(int)
                hid = ((d[r_col] < 0) & (d[n_col] >= 0)).astype(int)
                tl = (d[t_col] < 0).astype(int) if t_col else pd.Series(0, index=d.index)
                c = pd.DataFrame({"iso": d.iso, "one": 1, "nl": nl, "rl": rl, "hid": hid, "tl": tl}
                                 ).groupby("iso").sum()
                arr = c[["one", "nl", "rl", "hid", "tl"]].to_numpy(float)
                row = {"sample": sample, "h": h, "group": gname, "value": gval,
                       "windows": int(arr[:, 0].sum()), "countries": int(len(c)),
                       "p_nominal_loss": arr[:, 1].sum() / arr[:, 0].sum(),
                       "p_real_loss": arr[:, 2].sum() / arr[:, 0].sum(),
                       "p_hidden_loss": arr[:, 3].sum() / arr[:, 0].sum(),
                       "hidden_share_of_real_losses": (arr[:, 3].sum() / arr[:, 2].sum()
                                                       if arr[:, 2].sum() else np.nan)}
                row["p_real_loss_lo"], row["p_real_loss_hi"] = cluster_bootstrap(arr, 2, 0, rng)
                row["hidden_share_lo"], row["hidden_share_hi"] = cluster_bootstrap(arr, 3, 2, rng)
                if t_col:
                    row["p_total_return_loss"] = arr[:, 4].sum() / arr[:, 0].sum()
                    row["p_tr_loss_lo"], row["p_tr_loss_hi"] = cluster_bootstrap(arr, 4, 0, rng)
                out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- episodes
def price_episodes(panel: pd.DataFrame, time_col: str, lp: str, lc: str, threshold: float,
                   periods_per_year: int, label_col: str | None = None,
                   extra: dict | None = None) -> pd.DataFrame:
    """Real price drawdowns with their nominal/inflation decomposition."""
    rows = []
    for iso, g in panel.groupby("iso", sort=True):
        g = g.set_index(time_col)
        base = pd.DataFrame({"lp": g[lp], "lc": g[lc]})
        for seg in segments(base.lp.where(base.lc.notna())):
            idx = seg.index
            lN = base.lp.loc[idx].copy()
            lC = base.lc.loc[idx].copy()
            lN.iloc[0], lC.iloc[0] = 0.0, 0.0            # levels start at 1 on the segment
            N = np.exp(lN.cumsum())
            C = np.exp(lC.cumsum())
            R = N / C
            for e in drawdowns(R, threshold):
                a, b = e["i_peak"], e["i_trough"]
                real_log = float(np.log(R.iloc[b] / R.iloc[a]))
                nom_log = float(np.log(N.iloc[b] / N.iloc[a]))
                infl_log = float(np.log(C.iloc[b] / C.iloc[a]))
                vis, hid = decompose(real_log, nom_log)
                dur = (b - a) / periods_per_year
                row = {"iso": iso, "peak": e["peak"], "trough": e["trough"],
                       "recovery": e["recovery"], "ongoing": e["recovery"] is None,
                       "real_fall": e["depth"], "real_log": real_log, "nominal_log": nom_log,
                       "inflation_log": infl_log, "visible": vis, "hidden": hid,
                       "visible_share": vis / -real_log, "hidden_share": hid / -real_log,
                       "nominal_fall_peak_to_trough": max(0.0, 1 - N.iloc[b] / N.iloc[a]),
                       "max_nominal_drawdown_inside": max_drawdown(N.iloc[a:b + 1].to_numpy()),
                       "years_to_trough": dur,
                       "infl_annual": np.expm1(infl_log / dur) if dur > 0 else np.nan,
                       "real_fall_per_year": -real_log / dur if dur > 0 else np.nan}
                if label_col:
                    row["peak_label"] = g.loc[e["peak"], label_col]
                    row["trough_label"] = g.loc[e["trough"], label_col]
                    row["recovery_label"] = (g.loc[e["recovery"], label_col]
                                             if e["recovery"] is not None else "not yet")
                if extra:
                    for k, f in extra.items():
                        row[k] = f(g, e["peak"], e["trough"])
                rows.append(row)
    return pd.DataFrame(rows)


def tr_over(g: pd.DataFrame, a: int, b: int) -> float:
    """Real total-return log change from year a to b (returns set); NaN if incomplete."""
    sl = g.loc[a + 1:b]
    if len(sl) != b - a or sl[["dltr", "dlc"]].isna().any().any():
        return np.nan
    return float((sl.dltr - sl.dlc).sum())


def episode_summary(E: pd.DataFrame, by: str | None) -> pd.DataFrame:
    parts = [("all", E)] if by is None else list(E.groupby(by, observed=True))
    rows = []
    for k, d in parts:
        L = -d.real_log
        nh = d[d.infl_annual.fillna(0) <= HYPER]
        rows.append({"group": by or "all", "value": str(k), "episodes": len(d),
                     "countries": d.iso.nunique(),
                     "median_real_fall": d.real_fall.median(),
                     "median_hidden_share": d.hidden_share.median(),
                     "mostly_hidden": int((d.hidden_share > 0.5).sum()),
                     "fully_hidden": int((d.nominal_log >= 0).sum()),
                     "no_nominal_fall_inside": int((d.max_nominal_drawdown_inside < 0.01).sum()),
                     "hidden_share_of_log_losses": d.hidden.sum() / L.sum(),
                     "hidden_share_of_log_losses_ex_hyper": nh.hidden.sum() / (-nh.real_log).sum(),
                     "hyper_episodes": int(len(d) - len(nh)),
                     "median_years_to_trough": d.years_to_trough.median(),
                     "median_infl_annual": d.infl_annual.median()})
    return pd.DataFrame(rows)


def spearman_cluster_ci(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    members = [np.where(codes == g)[0] for g in range(G)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(x[idx])) > 2:
            out.append(stats.spearmanr(x[idx], y[idx]).statistic)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def real_tr_episodes(p: pd.DataFrame, threshold: float, haircut: float = 0.0) -> pd.DataFrame:
    rows = []
    for iso, g in p.groupby("iso", sort=True):
        g = g.set_index("year")
        rr = g.rr * (1 - haircut)
        dltr = np.log1p(np.expm1(g.dlcg) + rr)
        base = pd.DataFrame({"ltr": dltr, "lcp": g.dlc, "lcg": g.dlcg})
        for seg in segments(base.ltr.where(base.lcp.notna() & base.lcg.notna())):
            idx = seg.index
            d = base.loc[idx].copy()
            d.iloc[0] = 0.0
            W = np.exp((d.ltr - d.lcp).cumsum())
            for e in drawdowns(W, threshold):
                a, b = e["i_peak"], e["i_trough"]
                sl = d.iloc[a + 1:b + 1]
                rows.append({"iso": iso, "peak": e["peak"], "trough": e["trough"],
                             "recovery": e["recovery"], "real_tr_fall": e["depth"],
                             "price_log": sl.lcg.sum(), "rent_log": (sl.ltr - sl.lcg).sum(),
                             "inflation_log": sl.lcp.sum(),
                             "years_to_trough": b - a,
                             "infl_annual": np.expm1(sl.lcp.sum() / (b - a))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- extensions
def extend_returns_after_2020(p: pd.DataFrame, oa: pd.DataFrame) -> pd.DataFrame:
    """JST returns countries, 2021-2025, from OECD annual averages: price growth from HPI,
    inflation from the consumption deflator, net rental yield rolled forward from JST's 2020
    yield with the OECD price-to-rent ratio (the rent-price approach JST use)."""
    rows = []
    jst20 = p[p.year == JST_RETURNS_BENCH].set_index("iso")
    for iso, g in oa.groupby("iso"):
        if iso not in jst20.index or pd.isna(jst20.loc[iso, "housing_rent_yd"]):
            continue
        g = g.set_index("year").sort_index()
        if JST_RETURNS_BENCH not in g.index or LAST_FULL_YEAR not in g.index:
            continue
        y20 = jst20.loc[iso, "housing_rent_yd"]
        pr = g.HPI_RPI.ffill()                     # price-to-rent carried forward if not yet published
        pr20 = pr.loc[JST_RETURNS_BENCH]
        for y in range(JST_RETURNS_BENCH + 1, LAST_FULL_YEAR + 1):
            gn = g.loc[y, "HPI"] / g.loc[y - 1, "HPI"] - 1
            pi = g.loc[y, "defl"] / g.loc[y - 1, "defl"] - 1
            yd = y20 * pr20 / pr.loc[y]
            tr = gn + yd * (1 + gn)
            rows.append({"iso": iso, "year": y, "g_nominal": gn, "inflation": pi,
                         "net_yield": yd, "total_return": tr,
                         "price_to_rent_carried": bool(pd.isna(g.loc[y, "HPI_RPI"])),
                         "real_price": (1 + gn) / (1 + pi) - 1,
                         "real_total_return": (1 + tr) / (1 + pi) - 1})
    return pd.DataFrame(rows)


def portugal_series(p: pd.DataFrame, oa: pd.DataFrame) -> pd.DataFrame:
    """Portugal 1948-2025, annual. Prices: JST returns series to 1988, national index (OECD,
    identical to JST hpnom) from 1989. Consumer prices: JST CPI to 2020, OECD consumption
    deflator after. Net rental return, two variants: (J) JST's published rent return to 2020,
    rolled forward with the OECD price-to-rent ratio; (O) from 1989 the rent-price approach
    applied to the OECD rent and price indices, anchored to JST's 2015 yield."""
    j = p[p.iso == "PRT"].set_index("year")
    o = oa[oa.iso == "PRT"].set_index("year").sort_index()
    years = range(1948, LAST_FULL_YEAR + 1)
    rows = []
    y15 = j.loc[2015, "housing_rent_yd"]
    for y in years:
        if y <= 1988:
            gn = j.loc[y, "housing_capgain"]
            src = "JST returns price series"
        else:
            gn = o.loc[y, "HPI"] / o.loc[y - 1, "HPI"] - 1
            src = "national index (OECD = JST hpnom)"
        if y <= JST_RETURNS_BENCH:
            pi = j.loc[y, "cpi"] / j.loc[y - 1, "cpi"] - 1
            rr_j = j.loc[y, "housing_rent_rtn"]
        else:
            pi = o.loc[y, "defl"] / o.loc[y - 1, "defl"] - 1
            yd = j.loc[JST_RETURNS_BENCH, "housing_rent_yd"] * (
                o.loc[JST_RETURNS_BENCH, "HPI_RPI"] / o.loc[y, "HPI_RPI"])
            rr_j = yd * (1 + gn)
        if y <= 1988:
            rr_o = j.loc[y, "housing_rent_rtn"]
        else:
            yd_o = y15 * (o.loc[y, "RPI"] / o.loc[2015, "RPI"]) / (o.loc[y, "HPI"] / o.loc[2015, "HPI"])
            rr_o = yd_o * (1 + gn)
        rows.append({"year": y, "price_source": src, "g_nominal": gn, "inflation": pi,
                     "rent_return_J": rr_j, "rent_return_O": rr_o})
    d = pd.DataFrame(rows).set_index("year")
    d["N"] = (1 + d.g_nominal).cumprod()
    d["C"] = (1 + d.inflation).cumprod()
    d["R"] = d.N / d.C
    d["T_J"] = ((1 + d.g_nominal + d.rent_return_J) / (1 + d.inflation)).cumprod()
    d["T_O"] = ((1 + d.g_nominal + d.rent_return_O) / (1 + d.inflation)).cumprod()
    for c in ["N", "C", "R", "T_J", "T_O"]:
        d[c] = d[c] / d.loc[1948, c] if d.loc[1948, c] else d[c]
    return d.reset_index()


def window_change(d: pd.DataFrame, a: int, b: int) -> dict:
    s = d.set_index("year")
    f = {c: s.loc[b, c] / s.loc[a, c] - 1 for c in ["N", "C", "R", "T_J", "T_O"]}
    return {"from": a, "to": b, "nominal_price": f["N"], "consumer_prices": f["C"],
            "real_price": f["R"], "real_total_return_J": f["T_J"],
            "real_total_return_O": f["T_O"]}


# ----------------------------------------------------------------------------- main
def pct(x: float, nd: int = 1) -> str:
    return "nan" if pd.isna(x) else f"{100 * x:.{nd}f}%"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    S: list[str] = []

    def line(s: str = "") -> None:
        S.append(s)

    p = load_jst()
    q = load_oecd()
    oa = oecd_annual()

    # ---- data notes ------------------------------------------------------------
    price_c = sorted(p.dropna(subset=["dlp", "dlc"]).iso.unique())
    ret_c = sorted(p.dropna(subset=["dltr", "dlc", "dlcg"]).iso.unique())
    both = p.dropna(subset=["dlp", "dlcg"])
    gap = (both.dlp - both.dlcg).abs()
    gap_c = both.assign(gap=gap).groupby("iso").gap.agg(lambda s: (s > 0.02).sum())
    line("HOUSING LOSSES AND INFLATION: every number used in the article")
    line("=" * 72)
    line("")
    line("[0] Data")
    line(f"JST R6 price set (KSS national nominal index + CPI): {len(price_c)} countries, "
         f"{int(p.dropna(subset=['dlp', 'dlc']).shape[0])} country-years of annual changes, "
         f"{int(p.dropna(subset=['dlp']).year.min())}-{int(p.year.max())}")
    line(f"JST R6 returns set (returns price series + net rent + CPI): {len(ret_c)} countries, "
         f"{int(p.dropna(subset=['dltr', 'dlc', 'dlcg']).shape[0])} country-years")
    line("Country-years where the two JST nominal price series differ by more than 2 pp: "
         + ", ".join(f"{k} {v}" for k, v in gap_c[gap_c > 0].items()))
    qc = q.dropna(subset=["dlp", "dlc"])
    line(f"OECD quarterly set: {qc.iso.nunique()} OECD members, {len(qc)} country-quarters, "
         f"{q.dropna(subset=['HPI']).label.min()} to {q.dropna(subset=['HPI']).label.max()}")
    pd.DataFrame({"iso": gap_c.index, "years_gap_gt_2pp": gap_c.values}).to_csv(
        RESULTS / "jst_price_series_gap.csv", index=False, float_format="%.10g")

    # ---- [1] loss years -----------------------------------------------------------
    a = p.dropna(subset=["dlp", "dlc"]).copy()
    a["era"] = a.year.map(era_of)
    a["infl"] = np.expm1(a.dlc)
    a["ibin"] = infl_bin(a.infl)
    a["nl"] = a.dlp < 0
    a["rl"] = (a.dlp - a.dlc) < 0
    a["hid"] = a.rl & ~a.nl
    yrs = []
    for gname, gcol in [("all", None), ("era", "era"), ("inflation", "ibin")]:
        parts = [("all", a)] if gcol is None else list(a.groupby(gcol, observed=True))
        for k, d in parts:
            yrs.append({"group": gname, "value": str(k), "years": len(d),
                        "median_inflation": d.infl.median(),
                        "p_nominal_fall": d.nl.mean(), "p_real_fall": d.rl.mean(),
                        "hidden_share_of_real_falls": d.hid.sum() / d.rl.sum()})
    Y = pd.DataFrame(yrs)
    Y.to_csv(RESULTS / "loss_years_jst.csv", index=False, float_format="%.10g")
    line("")
    line("[1] Single years, JST price set")
    r0 = Y[Y.group == "all"].iloc[0]
    line(f"Nominal price fell in {pct(r0.p_nominal_fall)} of country-years; real price fell in "
         f"{pct(r0.p_real_fall)}; {pct(r0.hidden_share_of_real_falls)} of real falls had no nominal fall")
    for _, r in Y[Y.group != "all"].iterrows():
        line(f"  {r.group:9s} {r.value:10s} n={r.years:4d}  infl(med) {pct(r.median_inflation)}  "
             f"nominal fall {pct(r.p_nominal_fall)}  real fall {pct(r.p_real_fall)}  "
             f"hidden {pct(r.hidden_share_of_real_falls)}")

    # ---- [2] holding periods ------------------------------------------------------
    wa = windows_annual(p)
    groups = {"all": None, "era": "era", "inflation": "ibin"}
    H1 = loss_table(wa, "lN", "lR", None, "JST price set", rng, groups)
    H2 = loss_table(wa, "lNcg", "lRcg", "lT", "JST returns set", rng, groups)
    wq = windows_quarterly(q)
    H3 = loss_table(wq, "lN", "lR", None, "OECD quarterly", rng,
                    {"all": None, "decade": "decade", "inflation": "ibin"})
    H = pd.concat([H1, H2, H3], ignore_index=True)
    H.to_csv(RESULTS / "holding_periods.csv", index=False, float_format="%.10g")
    line("")
    line("[2] Holding periods: share of purchases showing a loss after h years")
    line("    (95% intervals: countries resampled with replacement)")
    for _, r in H[H.group == "all"].iterrows():
        hh = f"{r.h}y" if r["sample"] != "OECD quarterly" else f"{r.h // 4}y"
        s = (f"  {r['sample']:16s} {hh:>4s} n={r.windows:5d}  nominal {pct(r.p_nominal_loss)}  "
             f"real {pct(r.p_real_loss)} [{pct(r.p_real_loss_lo)}, {pct(r.p_real_loss_hi)}]  "
             f"hidden/real {pct(r.hidden_share_of_real_losses)} "
             f"[{pct(r.hidden_share_lo)}, {pct(r.hidden_share_hi)}]")
        if r["sample"] == "JST returns set":
            s += (f"  total return {pct(r.p_total_return_loss)} "
                  f"[{pct(r.p_tr_loss_lo)}, {pct(r.p_tr_loss_hi)}]")
        line(s)
    line("  10-year holdings by average inflation over the holding period:")
    for samp, hz in [("JST price set", 10), ("JST returns set", 10), ("OECD quarterly", 40)]:
        for _, r in H[(H["sample"] == samp) & (H.h == hz) & (H.group == "inflation")].iterrows():
            s = (f"    {samp:16s} {r.value:6s} n={r.windows:4d}  nominal {pct(r.p_nominal_loss)}  "
                 f"real {pct(r.p_real_loss)}  hidden/real {pct(r.hidden_share_of_real_losses)}")
            if samp == "JST returns set":
                s += f"  total return {pct(r.p_total_return_loss)}"
            line(s)
    line("  10-year holdings by era of purchase (JST):")
    for samp in ["JST price set", "JST returns set"]:
        for _, r in H[(H["sample"] == samp) & (H.h == 10) & (H.group == "era")].iterrows():
            s = (f"    {samp:16s} {r.value:10s} n={r.windows:4d}  nominal {pct(r.p_nominal_loss)}  "
                 f"real {pct(r.p_real_loss)}  hidden/real {pct(r.hidden_share_of_real_losses)}")
            if samp == "JST returns set":
                s += f"  total return {pct(r.p_total_return_loss)}"
            line(s)

    # ---- [3] episodes ---------------------------------------------------------------
    p_ret = p.copy()
    EJ = price_episodes(p, "year", "dlp", "dlc", THRESHOLD, 1,
                        extra={"real_tr_log_same_window": tr_over})
    EJ["era"] = EJ.peak.map(lambda y: era_of(int(y) + 1) or era_of(int(y)))
    EJcg = price_episodes(p_ret, "year", "dlcg", "dlc", THRESHOLD, 1,
                          extra={"real_tr_log_same_window": tr_over})
    EJcg["era"] = EJcg.peak.map(lambda y: era_of(int(y) + 1) or era_of(int(y)))
    EQ = price_episodes(q, "t", "dlp", "dlc", THRESHOLD, 4, label_col="label")
    EQ["decade"] = ((EQ.peak // 4) // 10 * 10).astype(int).astype(str) + "s"
    EJ.to_csv(RESULTS / "episodes_jst.csv", index=False, float_format="%.10g")
    EJcg.to_csv(RESULTS / "episodes_jst_returns_series.csv", index=False, float_format="%.10g")
    EQ.to_csv(RESULTS / "episodes_oecd.csv", index=False, float_format="%.10g")
    ES = pd.concat([
        episode_summary(EJ, None).assign(sample="JST price set"),
        episode_summary(EJ, "era").assign(sample="JST price set"),
        episode_summary(EJcg, None).assign(sample="JST returns set"),
        episode_summary(EQ, None).assign(sample="OECD quarterly"),
        episode_summary(EQ, "decade").assign(sample="OECD quarterly"),
    ], ignore_index=True)
    ES.to_csv(RESULTS / "episode_summary.csv", index=False, float_format="%.10g")
    line("")
    line(f"[3] Real price drawdowns of at least {THRESHOLD:.0%} (peak to trough)")
    for _, r in ES.iterrows():
        line(f"  {r['sample']:16s} {r.value:10s} episodes {r.episodes:3d} ({r.countries} countries)  "
             f"median fall {pct(r.median_real_fall)}  median hidden share {pct(r.median_hidden_share)}  "
             f"mostly hidden {r.mostly_hidden}  nominal up at trough {r.fully_hidden}  "
             f"no nominal fall inside {r.no_nominal_fall_inside}  log-weighted hidden share "
             f"{pct(r.hidden_share_of_log_losses)} ({pct(r.hidden_share_of_log_losses_ex_hyper)} without "
             f"{r.hyper_episodes} hyperinflation episodes)  median years to trough "
             f"{r.median_years_to_trough:.2f}  median inflation {pct(r.median_infl_annual)}")
    sens = []
    for thr in (THRESHOLD,) + THRESHOLDS_SENS:
        for nm, E_ in [("JST price set", price_episodes(p, "year", "dlp", "dlc", thr, 1)),
                       ("OECD quarterly", price_episodes(q, "t", "dlp", "dlc", thr, 4))]:
            sens.append({"threshold": thr, "sample": nm, "episodes": len(E_),
                         "median_hidden_share": E_.hidden_share.median(),
                         "mostly_hidden": int((E_.hidden_share > 0.5).sum()),
                         "spearman_hidden_vs_inflation": stats.spearmanr(
                             E_.infl_annual, E_.hidden_share, nan_policy="omit").statistic})
    SENS = pd.DataFrame(sens)
    SENS.to_csv(RESULTS / "episode_threshold_sensitivity.csv", index=False, float_format="%.10g")
    line("  Threshold sensitivity:")
    for _, r in SENS.iterrows():
        line(f"    {r['sample']:16s} >= {r.threshold:.0%}: {r.episodes:3d} episodes, median hidden share "
             f"{pct(r.median_hidden_share)}, mostly hidden {r.mostly_hidden}, Spearman(hidden share, "
             f"inflation) {r.spearman_hidden_vs_inflation:.2f}")

    # ---- [4] inflation, visibility and speed -------------------------------------------
    line("")
    line("[4] Inflation during the fall, visibility and speed of the real correction")
    reg_rows = []
    for nm, E_ in [("JST price set", EJ), ("OECD quarterly", EQ)]:
        d = E_[E_.years_to_trough > 0].copy()
        d["infl_log_annual"] = d.inflation_log / d.years_to_trough
        rho = stats.spearmanr(d.infl_annual, d.hidden_share)
        lo, hi = spearman_cluster_ci(d.infl_annual.to_numpy(), d.hidden_share.to_numpy(),
                                     d.iso.to_numpy(), rng)
        line(f"  {nm}: Spearman(hidden share, annual inflation during the fall) = "
             f"{rho.statistic:.2f} [{lo:.2f}, {hi:.2f}] (countries resampled; n = {len(d)} episodes, "
             f"{d.iso.nunique()} countries)")
        base = d[np.expm1(d.infl_log_annual) <= HYPER]
        for spec, dd in [("all but hyperinflation", base), ("completed only", base[~base.ongoing]),
                         ("all episodes", d)]:
            X = np.column_stack([np.ones(len(dd)), np.log(-dd.real_log), dd.infl_log_annual])
            b, se = ols_cluster(np.log(dd.years_to_trough.to_numpy()), X, dd.iso.to_numpy())
            reg_rows.append({"sample": nm, "spec": spec, "n": len(dd), "countries": dd.iso.nunique(),
                             "b_log_depth": b[1], "se_log_depth": se[1],
                             "b_inflation": b[2], "se_inflation": se[2]})
            line(f"    log(years to trough) = a + {b[1]:.2f} (se {se[1]:.2f}) log(real log fall) "
                 f"{b[2]:+.2f} (se {se[2]:.2f}) x annual log inflation   [{spec}, n={len(dd)}, "
                 f"{dd.iso.nunique()} countries, SE clustered by country]; +5 pp inflation -> "
                 f"{pct(np.expm1(0.05 * b[2]))} time to trough")
        bins = infl_bin(d.infl_annual)
        for k, dd in d.groupby(bins, observed=True):
            line(f"    inflation {k:6s}: {len(dd):3d} episodes, median hidden share "
                 f"{pct(dd.hidden_share.median())}, median real fall per year "
                 f"{pct(dd.real_fall_per_year.median())}, median years to trough "
                 f"{dd.years_to_trough.median():.2f}")
    pd.DataFrame(reg_rows).to_csv(RESULTS / "speed_regression.csv", index=False, float_format="%.10g")

    # ---- [5] two busts: 2007-type vs 2021-type --------------------------------------
    line("")
    line("[5] Two busts in the OECD data: real peaks 2005-2009 versus real peaks 2020 or later")
    t05, t09, t20 = 2005 * 4, 2009 * 4 + 3, 2020 * 4
    B1 = EQ[(EQ.peak >= t05) & (EQ.peak <= t09)].copy()
    B2 = EQ[EQ.peak >= t20].copy()
    for nm, B in [("peak 2005-2009", B1), ("peak 2020+", B2)]:
        line(f"  {nm}: {len(B)} episodes >= {THRESHOLD:.0%}, median real fall {pct(B.real_fall.median())}, "
             f"median hidden share {pct(B.hidden_share.median())}, mostly hidden "
             f"{int((B.hidden_share > 0.5).sum())}, "
             f"median inflation {pct(B.infl_annual.median())}, nominal up at trough "
             f"{int((B.nominal_log >= 0).sum())}, still below the real peak {int(B.ongoing.sum())}")
    pair = B1.merge(B2, on="iso", suffixes=("_0709", "_2020"))
    line(f"  countries with a >= {THRESHOLD:.0%} fall in both: {len(pair)} "
         f"({', '.join(pair.iso)}); hidden share higher in the 2020s fall in "
         f"{int((pair.hidden_share_2020 > pair.hidden_share_0709).sum())} of them")
    # every OECD member: change from real peak after 2020 to trough (any size)
    post = []
    for iso, g in q.groupby("iso"):
        g = g.set_index("t").dropna(subset=["RHP", "HPI"])
        g = g[g.index >= t20]
        if len(g) < 4:
            continue
        a_ = g.RHP.idxmax()
        tail = g.loc[a_:]
        b_ = tail.RHP.idxmin()
        real_log = float(np.log(g.RHP[b_] / g.RHP[a_]))
        nom_log = float(np.log(g.HPI[b_] / g.HPI[a_]))
        vis, hid = decompose(real_log, nom_log) if real_log < 0 else (0.0, 0.0)
        post.append({"iso": iso, "real_peak": g.label[a_], "trough": g.label[b_],
                     "real_fall": 1 - g.RHP[b_] / g.RHP[a_],
                     "nominal_change": g.HPI[b_] / g.HPI[a_] - 1,
                     "deflator_change": g.defl[b_] / g.defl[a_] - 1,
                     "hidden_share": hid / -real_log if real_log < 0 else np.nan,
                     "real_now_vs_peak": g.RHP.iloc[-1] / g.RHP[a_] - 1,
                     "nominal_now_vs_real_peak": g.HPI.iloc[-1] / g.HPI[a_] - 1,
                     "latest": g.label.iloc[-1]})
    POST = pd.DataFrame(post).sort_values("real_fall", ascending=False)
    POST.to_csv(RESULTS / "post2020_oecd.csv", index=False, float_format="%.10g")
    line(f"  All {len(POST)} OECD members since 2020-Q1: real fall from post-2020 peak >= 10% in "
         f"{int((POST.real_fall >= 0.10).sum())}, >= 5% in {int((POST.real_fall >= 0.05).sum())}; "
         f"nominal index lower at the real trough than at the real peak in "
         f"{int((POST.nominal_change < 0).sum())}")
    for _, r in POST.head(20).iterrows():
        line(f"    {r.iso} real peak {r.real_peak} trough {r.trough}: real {pct(-r.real_fall)}, "
             f"nominal {pct(r.nominal_change)}, deflator {pct(r.deflator_change)}, hidden "
             f"{pct(r.hidden_share)}; latest ({r.latest}) real {pct(r.real_now_vs_peak)} vs peak")

    # ---- [6] rents: real total return lens ---------------------------------------
    line("")
    line("[6] Rents: the real total return lens (JST returns set)")
    m = p.dropna(subset=["dltr", "dlc", "dlcg"])
    line(f"  Mean annual real capital gain {pct(np.expm1(m.dlcg - m.dlc).mean())}, mean annual net "
         f"rental return {pct(m.rr.mean())}, mean annual real total return "
         f"{pct(np.expm1(m.dltr - m.dlc).mean())} ({len(m)} country-years)")
    hair = []
    for hc in YIELD_HAIRCUTS:
        w2 = wa.copy()
        if hc:
            # rebuild total return with a proportional cut to the rent return
            parts = []
            for iso, g in p.groupby("iso"):
                g = g.set_index("year")
                dl = np.log1p(np.expm1(g.dlcg) + g.rr * (1 - hc))
                for h in HORIZONS_Y:
                    parts.append(pd.DataFrame({"iso": iso, "start": g.index, "h": h,
                                               "lTnom_h": forward_sum(dl, h)}))
            add = pd.concat(parts, ignore_index=True)
            w2 = w2.merge(add, on=["iso", "start", "h"], how="left")
            w2["lT"] = w2.lTnom_h - w2.lC
        d = w2.dropna(subset=["lNcg", "lRcg", "lT", "infl"])
        mm = p.dropna(subset=["dltr", "dlc", "dlcg"])
        mean_rt = np.expm1(np.log1p(np.expm1(mm.dlcg) + mm.rr * (1 - hc)) - mm.dlc).mean()
        for h in HORIZONS_Y:
            dh = d[d.h == h]
            hair.append({"haircut": hc, "h": h, "windows": len(dh),
                         "mean_annual_real_tr": mean_rt,
                         "p_real_price_loss": (dh.lRcg < 0).mean(),
                         "p_real_tr_loss": (dh.lT < 0).mean(),
                         "p_real_tr_loss_infl_gt10": (dh[dh.infl > 0.10].lT < 0).mean(),
                         "p_real_tr_loss_infl_le5": (dh[dh.infl <= 0.05].lT < 0).mean(),
                         "share_of_tr_losses_with_infl_gt5": ((dh.lT < 0) & (dh.infl > 0.05)).sum()
                         / max(1, (dh.lT < 0).sum())})
    HC = pd.DataFrame(hair)
    HC.to_csv(RESULTS / "rent_haircut.csv", index=False, float_format="%.10g")
    for _, r in HC[HC.h.isin([5, 10])].iterrows():
        line(f"  rent return cut {r.haircut:.0%}, {int(r.h):2d}y: mean real TR {pct(r.mean_annual_real_tr)}; "
             f"P(real price loss) {pct(r.p_real_price_loss)}; P(real TR loss) {pct(r.p_real_tr_loss)}; "
             f"if inflation > 10% {pct(r.p_real_tr_loss_infl_gt10)}; if <= 5% "
             f"{pct(r.p_real_tr_loss_infl_le5)}; TR losses with inflation > 5% "
             f"{pct(r.share_of_tr_losses_with_infl_gt5)}")
    TE = real_tr_episodes(p, THRESHOLD)
    TE["era"] = TE.peak.map(lambda y: era_of(int(y) + 1) or era_of(int(y)))
    TE.to_csv(RESULTS / "episodes_real_total_return.csv", index=False, float_format="%.10g")
    allyrs = p.dropna(subset=["dltr", "dlc", "dlcg"])
    line(f"  Real total-return drawdowns >= {THRESHOLD:.0%}: {len(TE)} in {TE.iso.nunique()} countries "
         f"(vs {len(EJcg)} real price drawdowns on the same price series)")
    line(f"    nominal price higher at the trough than at the peak in {int((TE.price_log >= 0).sum())} "
         f"of {len(TE)}; inflation above 5% a year during the fall in {int((TE.infl_annual > 0.05).sum())}; "
         f"median inflation during the fall {pct(TE.infl_annual.median())} (all country-years: "
         f"{pct(np.expm1(allyrs.dlc).median())}); peak in 1914-1945: {int((TE.era == '1914-1945').sum())}")
    line(f"    median rent offset over the fall {pct(np.expm1(TE.rent_log).median())} of the peak value")
    for k, d in TE.groupby("era", observed=True):
        line(f"    {k}: {len(d)} episodes, median fall {pct(d.real_tr_fall.median())}, median inflation "
             f"{pct(d.infl_annual.median())}")
    for hc in YIELD_HAIRCUTS[1:]:
        TEh = real_tr_episodes(p, THRESHOLD, hc)
        line(f"    with rent return cut {hc:.0%}: {len(TEh)} real total-return drawdowns >= {THRESHOLD:.0%}")
    covered = EJcg.real_tr_log_same_window.notna()
    still = (EJcg.real_tr_log_same_window < 0) & covered
    line(f"  Of {int(covered.sum())} real price drawdowns with complete returns data, real total return "
         f"over the same peak-to-trough window was still negative in {int(still.sum())}; "
         f"median real TR over the window {pct(np.expm1(EJcg.loc[covered, 'real_tr_log_same_window']).median())}")

    # ---- [7] after 2020 with rents ------------------------------------------------------
    X20 = extend_returns_after_2020(p, oa)
    X20.to_csv(RESULTS / "returns_2021_2025.csv", index=False, float_format="%.10g")
    line("")
    line(f"[7] 2021-{LAST_FULL_YEAR}, JST returns countries extended with OECD annual averages")
    agg = []
    for iso, d in X20.groupby("iso"):
        rp = np.prod(1 + d.real_price) - 1
        rt = np.prod(1 + d.real_total_return) - 1
        npc = np.prod(1 + d.g_nominal) - 1
        lvl_p = np.r_[1, np.cumprod(1 + d.real_price.to_numpy())]
        lvl_t = np.r_[1, np.cumprod(1 + d.real_total_return.to_numpy())]
        agg.append({"iso": iso, "nominal_price": npc, "real_price": rp, "real_total_return": rt,
                    "worst_real_price_dd": max_drawdown(lvl_p), "worst_real_tr_dd": max_drawdown(lvl_t),
                    "mean_net_yield": d.net_yield.mean()})
    AGG = pd.DataFrame(agg)
    AGG.to_csv(RESULTS / "returns_2021_2025_summary.csv", index=False, float_format="%.10g")
    carried = X20[X20.price_to_rent_carried].groupby("iso").year.apply(list)
    line("  price-to-rent ratio not yet published, last value carried forward: "
         + ("; ".join(f"{k} {v}" for k, v in carried.items()) if len(carried) else "none"))
    line(f"  {len(AGG)} countries. Cumulative 2020->{LAST_FULL_YEAR}: nominal price up in "
         f"{int((AGG.nominal_price > 0).sum())}; real price down in {int((AGG.real_price < 0).sum())}; "
         f"real total return down in {int((AGG.real_total_return < 0).sum())}")
    line(f"  Annual real price drawdown inside 2020-{LAST_FULL_YEAR} >= 10% in "
         f"{int((AGG.worst_real_price_dd >= 0.10).sum())}; annual real total-return drawdown >= 10% in "
         f"{int((AGG.worst_real_tr_dd >= 0.10).sum())}")
    for _, r in AGG.sort_values("worst_real_price_dd", ascending=False).iterrows():
        line(f"    {r.iso}: nominal {pct(r.nominal_price)}, real {pct(r.real_price)}, real TR "
             f"{pct(r.real_total_return)}; worst real price drawdown {pct(r.worst_real_price_dd)}, "
             f"worst real TR drawdown {pct(r.worst_real_tr_dd)}; mean net yield {pct(r.mean_net_yield)}")

    # ---- [8] Portugal ---------------------------------------------------------------
    PT = portugal_series(p, oa)
    PT.to_csv(RESULTS / "portugal_annual.csv", index=False, float_format="%.10g")
    line("")
    line(f"[8] Portugal, 1948-{LAST_FULL_YEAR} (annual)")
    gp = p[(p.iso == "PRT") & p.year.between(1989, 2013)].dropna(subset=["dlp", "dlcg"])
    line(f"  Two JST price series for Portugal, 1989-2013: national index (hpnom = OECD) "
         f"{pct(np.expm1(gp.dlp.sum()))} cumulative, JST returns series "
         f"{pct(np.expm1(gp.dlcg.sum()))}; they coincide from 2014. The national index is used from 1989.")
    for a_, b_ in [(1948, 1973), (1973, 1985), (1973, 1990), (1990, 2007), (2007, 2013),
                   (2013, 2020), (2020, LAST_FULL_YEAR), (1948, LAST_FULL_YEAR)]:
        c = window_change(PT, a_, b_)
        line(f"  {a_}-{b_}: nominal {pct(c['nominal_price'])}, consumer prices {pct(c['consumer_prices'])}, "
             f"real price {pct(c['real_price'])}, real total return {pct(c['real_total_return_J'])} "
             f"(rents J) / {pct(c['real_total_return_O'])} (rents O)")
    pt = PT.set_index("year")
    pt_lvl = pt[["N", "C"]].copy()
    pt_panel = pd.DataFrame({"iso": "PRT", "year": pt.index,
                             "dlp": np.log(pt.N).diff().to_numpy(),
                             "dlc": np.log(pt.C).diff().to_numpy()})
    EP = price_episodes(pt_panel, "year", "dlp", "dlc", THRESHOLD, 1)
    EP.to_csv(RESULTS / "portugal_episodes.csv", index=False, float_format="%.10g")
    for _, r in EP.iterrows():
        rec = "not yet" if r.recovery is None or pd.isna(r.recovery) else int(r.recovery)
        line(f"  real drawdown {int(r.peak)}-{int(r.trough)} (recovered {rec}): real {pct(-r.real_fall)}, "
             f"nominal peak-to-trough {pct(np.expm1(r.nominal_log))}, max nominal fall inside "
             f"{pct(r.max_nominal_drawdown_inside)}, hidden share {pct(r.hidden_share)}, inflation "
             f"{pct(r.infl_annual)} a year")
    for nm, Tcol in [("J", "T_J"), ("O", "T_O")]:
        for e in drawdowns(pt[Tcol], THRESHOLD):
            line(f"  real total-return drawdown (rents {nm}) {e['peak']}-{e['trough']}: {pct(-e['depth'])}, "
                 f"recovered {e['recovery'] if e['recovery'] is not None else 'not yet'}")
    w_pt = pt_panel.set_index("year")
    for h in (5, 10):
        lN = forward_sum(w_pt.dlp, h)
        lR = lN - forward_sum(w_pt.dlc, h)
        ok = lN.notna() & lR.notna()
        line(f"  {h}-year holdings from 1948 on: nominal loss {pct((lN[ok] < 0).mean())}, real loss "
             f"{pct((lR[ok] < 0).mean())}, hidden {pct(((lR[ok] < 0) & (lN[ok] >= 0)).mean())} "
             f"(n = {int(ok.sum())})")
    pq = POST[POST.iso == "PRT"]
    if len(pq):
        r = pq.iloc[0]
        if r.real_fall < 0.005:
            line(f"  Portugal after 2020 (OECD quarterly): no real fall; the highest real level since "
                 f"2020-Q1 is the latest quarter ({r.latest})")
        else:
            line(f"  Portugal after 2020 (OECD quarterly): largest real fall from a post-2020 peak "
                 f"{pct(r.real_fall)} ({r.real_peak} to {r.trough}); latest real level vs that peak "
                 f"{pct(r.real_now_vs_peak)}")
    eq_pt = EQ[EQ.iso == "PRT"]
    for _, r in eq_pt.iterrows():
        line(f"  OECD quarterly episode {r.peak_label} to {r.trough_label} (recovered "
             f"{r.recovery_label}): real {pct(-r.real_fall)}, nominal {pct(np.expm1(r.nominal_log))}, "
             f"max nominal fall inside {pct(r.max_nominal_drawdown_inside)}, hidden {pct(r.hidden_share)}")

    # ---- [9] gains: how much of nominal appreciation is inflation -------------------------
    line("")
    line("[9] Gains: share of cumulative nominal appreciation that was consumer-price inflation")
    gains = []
    for iso, g in p.dropna(subset=["dlp", "dlc"]).groupby("iso"):
        gains.append({"iso": iso, "from": int(g.year.min()) - 1, "to": int(g.year.max()),
                      "nominal_log": g.dlp.sum(), "inflation_log": g.dlc.sum()})
    G = pd.DataFrame(gains)
    G["inflation_share"] = G.inflation_log / G.nominal_log
    G["real_annual"] = np.expm1((G.nominal_log - G.inflation_log) / (G["to"] - G["from"]))
    G.to_csv(RESULTS / "gains_inflation_share.csv", index=False, float_format="%.10g")
    line(f"  JST price set, whole sample per country: median inflation share {pct(G.inflation_share.median())} "
         f"(range {pct(G.inflation_share.min())} to {pct(G.inflation_share.max())}); median real price "
         f"growth {pct(G.real_annual.median(), 2)} a year")
    ptN, ptC = np.log(pt.N.iloc[-1] / pt.N.iloc[0]), np.log(pt.C.iloc[-1] / pt.C.iloc[0])
    line(f"  Portugal 1948-{LAST_FULL_YEAR}: inflation share {pct(ptC / ptN)}; real price growth "
         f"{pct(np.expm1((ptN - ptC) / (LAST_FULL_YEAR - 1948)), 2)} a year")

    (RESULTS / "summary.txt").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
