"""Bust episodes in long sectoral series (AMECO, 1960-2025), coded by integration stage.

    python long_episodes.py

Writes ``results/long_episodes.csv`` and ``results/long_episodes_summary.txt``.

Series
------
AMECO (via DBnomics, snapshot in ``data/ameco_long.csv``), five sectors per country:
agriculture, industry excluding construction, manufacturing, construction and services.
Two capacity measures:

- ``gva``: gross value added at 2020 prices in the country's own currency unit;
- ``emp``: employment, persons (national accounts).

Germany is West Germany up to 1991, chained at 1991 to unified Germany (growth rates are
kept, the level jump of reunification is not).

Two versions of each series:

- ``absolute``: the country's own log level;
- ``relative``: the country's log level minus the cumulated mean log change of the same
  sector in the other countries of the pool (leave-one-out, equal weights). This removes
  sector trends common to all advanced economies (the long fall of farm employment, the
  global path of manufacturing), so a bust here is a loss of the country's place relative
  to its peers.

Episodes use the same rule as ``episodes.py``: a fall of at least ``THETA`` log points from
the running maximum; the episode ends at the first later year above the peak.

Integration stage (at the trough year, when recovery starts)
-------------------------------------------------------------
- ``pre-accession``: before the country joined the EEC/EU (for the founding six, before the
  customs union was complete in July 1968);
- ``customs union``: member, before the single market (1993; 1994 for AT, FI, SE via the EEA);
- ``single market``: from 1993 (or accession, if later) until the country adopted the euro;
- ``euro``: from euro adoption;
- ``outside``: countries that are not EU members (US, Japan, Canada, Australia, New Zealand,
  Switzerland, Norway, Iceland) and the UK after it left.

Central and eastern European episodes with troughs before accession are flagged as
``transition`` and kept out of the main estimates, as are episodes whose peak is the first
year of the series (``left_censored_peak``): their true peak may lie before the data start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from episodes import find_episodes, relative
from stages import stage, transition

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

THETA = 0.15
LAST_YEAR = 2025

# AMECO country code -> (ISO2, own currency suffix in the GVA unit)
COUNTRIES = {
    "aut": ("AT", "ats"), "bel": ("BE", "bef"), "bgr": ("BG", "bgn"), "cyp": ("CY", "cyp"), "cze": ("CZ", "czk"),
    "deu": ("DE", "dem"), "dnk": ("DK", "dkk"), "esp": ("ES", "esp"), "est": ("EE", "eek"), "fin": ("FI", "fim"),
    "fra": ("FR", "frf"), "grc": ("EL", "grd"), "hrv": ("HR", "hrk"), "hun": ("HU", "huf"), "irl": ("IE", "iep"),
    "ita": ("IT", "itl"), "ltu": ("LT", "ltl"), "lux": ("LU", "luf"), "lva": ("LV", "lvl"), "mlt": ("MT", "mtl"),
    "nld": ("NL", "nlg"), "pol": ("PL", "pln"), "prt": ("PT", "pte"), "rom": ("RO", "ron"), "svk": ("SK", "skk"),
    "svn": ("SI", "sit"), "swe": ("SE", "sek"), "gbr": ("UK", "gbp"),
    "usa": ("US", "usd"), "jpn": ("JP", "jpy"), "can": ("CA", "cad"), "aus": ("AU", "aud"), "nzl": ("NZ", "nzd"),
    "che": ("CH", "chf"), "nor": ("NO", "nok"), "isl": ("IS", "isk"),
}
SECTORS = {"1": "agriculture", "2": "industry", "M": "manufacturing", "4": "construction", "5": "services"}
TRADABLE = {"agriculture": 1, "industry": 1, "manufacturing": 1, "construction": 0, "services": 0}


def load_ameco() -> pd.DataFrame:
    """Long table: geo, measure, sector, year, value."""
    a = pd.read_csv(DATA / "ameco_long.csv", low_memory=False)
    dims = a.dimensions.map(json.loads)
    a["ameco_geo"] = dims.map(lambda d: d.get("geo"))
    a["unit"] = dims.map(lambda d: d.get("unit"))
    a["year"] = pd.to_numeric(a.period.astype(str).str[:4], errors="coerce")
    a["value"] = pd.to_numeric(a.value, errors="coerce")
    a = a[a.dataset.str[3:].isin(SECTORS) & a.dataset.str[:3].isin(["NET", "OVG"])]
    a["sector"] = a.dataset.str[3:].map(SECTORS)
    a["measure"] = np.where(a.dataset.str.startswith("NET"), "emp", "gva")
    rows = []
    for code, (iso, cur) in list(COUNTRIES.items()) + [("d-w", ("DE-W", "dem"))]:
        s = a[a.ameco_geo == code]
        keep = (s.measure == "emp") & (s.unit == "1000-persons")
        keep |= (s.measure == "gva") & (s.unit.str.endswith("-" + cur) | (s.unit == "national-currency"))
        s = s[keep & s.value.notna() & (s.value > 0)]
        s = s.groupby(["measure", "sector", "year"], as_index=False).value.mean()
        s["geo"] = iso
        rows.append(s)
    d = pd.concat(rows, ignore_index=True)
    return chain_germany(d)


def chain_germany(d: pd.DataFrame) -> pd.DataFrame:
    """West Germany to 1991, unified Germany after, chained on 1991 log levels."""
    out = [d[~d.geo.isin(["DE", "DE-W"])]]
    for (m, s), g in d[d.geo.isin(["DE", "DE-W"])].groupby(["measure", "sector"]):
        uni = g[g.geo == "DE"].set_index("year").value
        west = g[g.geo == "DE-W"].set_index("year").value
        if uni.empty:
            continue
        if 1991 in uni.index and 1991 in west.index:
            shift = np.log(uni.loc[1991]) - np.log(west.loc[1991])
            early = np.exp(np.log(west[west.index < 1991]) + shift)
            series = pd.concat([early, uni])
        else:
            series = uni
        out.append(pd.DataFrame({"measure": m, "sector": s, "year": series.index, "value": series.values, "geo": "DE"}))
    return pd.concat(out, ignore_index=True)


def log_panels(d: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """(measure, sector) -> year x geo panel of log levels."""
    out = {}
    for (m, s), g in d.groupby(["measure", "sector"]):
        p = g.pivot_table(index="year", columns="geo", values="value").sort_index()
        p = p.loc[:LAST_YEAR]
        out[(m, s)] = np.log(p)
    return out


def episodes_for(x: pd.Series, geo: str, measure: str, sector: str, version: str, theta: float) -> list[dict]:
    x = x.dropna()
    if len(x) < 15:
        return []
    first, last = int(x.index.min()), int(x.index.max())
    rows = []
    for ep in find_episodes(x, theta):
        pk, tr = ep["peak_year"], ep["trough_year"]
        if tr >= last:
            continue
        depth = ep["peak_log"] - ep["trough_log"]
        after = x.loc[tr + 1 :]
        half = after[after >= ep["trough_log"] + 0.5 * depth]
        full = after[after > ep["peak_log"]]
        st = stage(geo, tr)
        rows.append(
            {
                "geo": geo,
                "measure": measure,
                "sector": sector,
                "version": version,
                "tradable": TRADABLE[sector],
                "series_first": first,
                "series_last": last,
                "peak_year": pk,
                "trough_year": tr,
                "depth": depth,
                "fall_years": tr - pk,
                "left_censored_peak": pk == first,
                "stage": st,
                "stage_at_peak": stage(geo, pk),
                "transition": transition(geo, tr),
                "t_half": int(half.index[0] - tr) if len(half) else int(last - tr),
                "half_recovered": int(len(half) > 0),
                "t_full": int(full.index[0] - tr) if len(full) else int(last - tr),
                "full_recovered": int(len(full) > 0),
                "follow_up": last - tr,
            }
        )
    return rows


def build(theta: float = THETA) -> pd.DataFrame:
    d = load_ameco()
    rows = []
    for (m, s), p in log_panels(d).items():
        r = relative(p)
        for geo in p.columns:
            rows += episodes_for(p[geo], geo, m, s, "absolute", theta)
            rows += episodes_for(r[geo], geo, m, s, "relative", theta)
    ep = pd.DataFrame(rows)
    ep["main"] = ~ep.left_censored_peak & ~ep.transition
    return ep


def main() -> None:
    RES.mkdir(exist_ok=True)
    ep = build()
    ep.to_csv(RES / "long_episodes.csv", index=False)
    L = []
    say = L.append
    say(f"long-series episodes (fall of at least {THETA} log points): {len(ep)}; main sample (not left-censored, not transition): {int(ep.main.sum())}")
    m = ep[ep.main]
    for (meas, ver), g in m.groupby(["measure", "version"]):
        say(f"\n{meas} / {ver}: {len(g)} episodes")
        tab = g.groupby(["stage", "tradable"]).agg(n=("depth", "size"), full=("full_recovered", "mean"), half=("half_recovered", "mean"), follow=("follow_up", "median"))
        for (st, tr), r in tab.iterrows():
            say(f"  {st:14s} {'tradable' if tr else 'non-tradable':13s} n={int(r.n):3d}  regained peak {r.full:.2f}  half {r.half:.2f}  median follow-up {r.follow:.0f} y")
    text = "\n".join(L) + "\n"
    (RES / "long_episodes_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
