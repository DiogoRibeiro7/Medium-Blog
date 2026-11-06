"""Bust episodes in EU-27 industry employment, 1995-2025, and what followed them.

    python episodes.py

Writes ``results/episodes.csv`` (one row per episode) and ``results/episodes_summary.txt``.

Definitions
-----------
Series: employment (persons) by member state and A*64 industry, Eurostat nama_10_a64_e,
1995-2025, market industries only (public administration, education, health, households
and extraterritorial bodies excluded), leaf codes only so that industries do not overlap.

Episode: starting from the running maximum of log employment, a fall of at least ``THETA``
(log points) below it. The peak is the year of that maximum; the trough is the lowest year
between the peak and the first later year in which employment exceeds the peak again (or
the end of the series). Depth is log(peak / trough). A series can have several episodes
only if it regains its previous peak in between.

Outcomes, measured from the trough:
- years until half of the lost log employment is regained (right-censored at the last year);
- the log rebound after five years, and the fraction of the loss it represents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stages import stage, transition

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = (
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
)
LEAF = (
    "A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
    "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_C32", "C33", "D", "E36", "E37-E39", "F", "G45",
    "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60", "J61", "J62_J63", "K64", "K65",
    "K66", "L", "M69_M70", "M71", "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82", "R90-R92", "R93",
    "S94", "S95", "S96",
)
SECTOR = {
    "A": "agriculture", "B": "industry", "C": "manufacturing", "D": "industry", "E": "industry", "F": "construction",
    "G": "trade", "H": "transport", "I": "accommodation", "J": "information", "K": "finance", "L": "real estate",
    "M": "business services", "N": "business services", "R": "other services", "S": "other services",
}
# goods are tradable; construction, utilities and services are not (sensitivity: the
# services most traded across borders, listed below, counted as tradable)
TRADABLE_SERVICES = ("H50", "H51", "J58", "J59_J60", "J61", "J62_J63", "K64", "K65", "K66", "M69_M70", "M71", "M72", "M73", "M74_M75")
THETA = 0.20
FIRST, LAST = 1995, 2025
MIN_SIZE = 2.0  # thousand persons at the peak
MIN_SIZE_GVA = 50.0  # million euro (2020 prices) at the peak


def panel(name: str, **filters) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{name}.csv", low_memory=False)
    for k, v in filters.items():
        d = d[d[k] == v]
    return d.pivot_table(index="time", columns=["geo", "nace_r2"], values="value").loc[FIRST:LAST]


def find_episodes(x: pd.Series, theta: float = THETA) -> list[dict]:
    """Episodes in one log-employment series (index = year)."""
    x = x.dropna()
    if len(x) < 15:
        return []
    years = x.index.to_numpy()
    v = x.to_numpy()
    out = []
    i = 0
    n = len(v)
    while i < n:
        peak_i = i
        j = i
        # walk forward, tracking the running maximum, until a fall of theta below it
        hit = None
        while j < n:
            if v[j] > v[peak_i]:
                peak_i = j
            if v[peak_i] - v[j] >= theta:
                hit = j
                break
            j += 1
        if hit is None:
            break
        # end of the episode: first year after the peak in which the peak is exceeded
        k = hit
        while k < n and v[k] <= v[peak_i]:
            k += 1
        seg = slice(peak_i, k)
        trough_i = peak_i + int(np.argmin(v[seg]))
        out.append({"peak_year": int(years[peak_i]), "trough_year": int(years[trough_i]), "peak_log": v[peak_i], "trough_log": v[trough_i], "regained_year": int(years[k]) if k < n else None})
        i = k if k < n else n
    return out


def window_growth(series: pd.Series, a: int, b: int) -> float:
    """Annualised log growth between years a and b (nan if either is missing)."""
    if a not in series.index or b not in series.index or b <= a:
        return np.nan
    va, vb = series.loc[a], series.loc[b]
    if not (np.isfinite(va) and np.isfinite(vb)) or va <= 0 or vb <= 0:
        return np.nan
    return float(np.log(vb / va) / (b - a))


def relative(p: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    """Own log level minus the cumulated leave-one-out mean log change of the other columns.

    The result keeps each country's own year-to-year movements net of what the same sector
    did on average elsewhere in the same year, so a common trend or a shock that hits every
    country alike does not register as a bust.
    """
    dx = p.diff()
    n = dx.notna().sum(axis=1)
    tot = dx.sum(axis=1, min_count=1)
    rel = pd.DataFrame(index=p.index, columns=p.columns, dtype=float)
    for c in p.columns:
        valid = p[c].dropna().index
        if len(valid) == 0:
            continue
        k = n - dx[c].notna().astype(int)
        others = ((tot - dx[c].fillna(0)) / k).where(k >= min_peers)
        cum = others.reindex(valid).iloc[1:].cumsum()
        rel[c] = p[c].reindex(valid) - pd.concat([pd.Series([0.0], index=[valid[0]]), cum])
    return rel


def build(theta: float = THETA, measure: str = "emp", version: str = "absolute") -> pd.DataFrame:
    """One row per episode. ``measure`` is ``emp`` (persons) or ``gva`` (volume);
    ``version`` is ``absolute`` or ``relative`` (to the other member states)."""
    emp = panel("employment_a64", unit="THS_PER", na_item="EMP_DC")
    gva = panel("gva_a64_volume")
    gdp = pd.read_csv(DATA / "gdp_volume.csv").pivot_table(index="time", columns="geo", values="value").loc[FIRST:LAST]
    hpi = pd.read_csv(DATA / "house_prices.csv").pivot_table(index="time", columns="geo", values="value")
    level = emp if measure == "emp" else gva
    logs = np.log(level.where(level > 0))
    if version == "relative":
        parts = []
        for ind in LEAF:
            cols = [(g, ind) for g in EU27 if (g, ind) in logs.columns]
            if cols:
                parts.append(relative(logs[cols]))
        track = pd.concat(parts, axis=1)
    else:
        track = logs
    min_size = MIN_SIZE if measure == "emp" else MIN_SIZE_GVA
    rows = []
    for geo in EU27:
        for ind in LEAF:
            if (geo, ind) not in track.columns or (geo, ind) not in level.columns:
                continue
            raw = level[(geo, ind)]
            if raw.dropna().empty or (raw.dropna() <= 0).any():
                continue
            x = track[(geo, ind)].dropna()
            if x.empty:
                continue
            last = int(x.index.max())
            first = int(x.index.min())
            eu_emp = emp[("EU27_2020", ind)] if ("EU27_2020", ind) in emp.columns else pd.Series(dtype=float)
            eu_gva = gva[("EU27_2020", ind)] if ("EU27_2020", ind) in gva.columns else pd.Series(dtype=float)
            own_gva = gva[(geo, ind)] if (geo, ind) in gva.columns else pd.Series(dtype=float)
            for ep in find_episodes(x, theta):
                pk, tr = ep["peak_year"], ep["trough_year"]
                depth = ep["peak_log"] - ep["trough_log"]
                if tr >= last or raw.get(pk, np.nan) < min_size:
                    continue
                target = ep["trough_log"] + 0.5 * depth
                after = x.loc[tr + 1 :].dropna()
                hit = after[after >= target]
                t_half = int(hit.index[0] - tr) if len(hit) else int(last - tr)
                event = int(len(hit) > 0)
                g5 = float(x.loc[tr + 5] - ep["trough_log"]) if (tr + 5) in x.index and np.isfinite(x.loc[tr + 5]) else np.nan
                w_end = min(tr + 5, last)
                rows.append(
                    {
                        "geo": geo,
                        "industry": ind,
                        "sector": SECTOR[ind[0]],
                        "measure": measure,
                        "version": version,
                        "peak_year": pk,
                        "trough_year": tr,
                        "peak_level": float(raw.get(pk, np.nan)),
                        "depth": depth,
                        "loss_share": 1 - np.exp(-depth),
                        "fall_years": tr - pk,
                        "fall_speed": depth / max(tr - pk, 1),
                        "left_censored_peak": pk == first,
                        "t_half": t_half,
                        "half_recovered": event,
                        "regained_peak": ep["regained_year"] is not None,
                        "t_full": int(ep["regained_year"] - tr) if ep["regained_year"] is not None else int(last - tr),
                        "full_recovered": int(ep["regained_year"] is not None),
                        "follow_up": int(last - tr),
                        "tradable": int(ind[0] in "ABC"),
                        "tradable_wide": int(ind[0] in "ABC" or ind in TRADABLE_SERVICES),
                        "stage": stage(geo, tr),
                        "stage_at_peak": stage(geo, pk),
                        "transition": transition(geo, tr),
                        "g5": g5,
                        "r5": g5 / depth if np.isfinite(g5) else np.nan,
                        "eu_emp_trend": window_growth(eu_emp, int(eu_emp.dropna().index.min()), int(eu_emp.dropna().index.max())) if eu_emp.notna().any() else np.nan,
                        "eu_emp_growth_after": window_growth(eu_emp, tr, w_end),
                        "eu_gva_growth_after": window_growth(eu_gva, tr, w_end),
                        "own_gva_growth_after": window_growth(own_gva, tr, w_end),
                        "gdp_growth_after": window_growth(gdp[geo], tr, w_end) if geo in gdp.columns else np.nan,
                        "hpi_growth_after": window_growth(hpi[geo], tr, w_end) if geo in hpi.columns else np.nan,
                        "covid": tr in (2020, 2021) and pk >= 2018,
                        "great_recession": 2008 <= tr <= 2016 and pk >= 2004,
                    }
                )
    ep = pd.DataFrame(rows)
    ep["main"] = ~ep.left_censored_peak & ~ep.transition
    return ep


def main() -> None:
    RES.mkdir(exist_ok=True)
    ep = build()
    ep.to_csv(RES / "episodes.csv", index=False)
    allv = pd.concat([build(THETA, m, v) for m in ("emp", "gva") for v in ("absolute", "relative")], ignore_index=True)
    allv.to_csv(RES / "episodes_all.csv", index=False)
    L = []
    say = L.append
    say(f"episodes (fall of at least {THETA} log points, peak at least {MIN_SIZE} thousand persons): {len(ep)}")
    say(f"series covered: {ep[['geo', 'industry']].drop_duplicates().shape[0]}; member states {ep.geo.nunique()}; industries {ep.industry.nunique()}")
    say(f"half-recovered by the last year: {ep.half_recovered.mean():.3f}; regained the peak: {ep.regained_peak.mean():.3f}")
    say(f"depth: median {ep.depth.median():.3f} log points (loss share {ep.loss_share.median():.3f}); 90th percentile {ep.depth.quantile(0.9):.3f}")
    say(f"pandemic episodes (peak 2018+, trough 2020-21): {int(ep.covid.sum())}; great-recession episodes (peak 2004+, trough 2008-16): {int(ep.great_recession.sum())}")
    for (m, v), g in allv[allv.main].groupby(["measure", "version"]):
        say(f"main sample {m}/{v}: {len(g)} episodes; regained the peak {g.full_recovered.mean():.3f} (tradable {g[g.tradable == 1].full_recovered.mean():.3f}, non-tradable {g[g.tradable == 0].full_recovered.mean():.3f})")
    by = ep.groupby("sector").agg(n=("depth", "size"), depth=("depth", "median"), half=("half_recovered", "mean"))
    for s, r in by.sort_values("n", ascending=False).iterrows():
        say(f"  {s}: {int(r.n)} episodes, median depth {r.depth:.3f}, half-recovered {r.half:.3f}")
    text = "\n".join(L) + "\n"
    (RES / "episodes_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
