"""Building blocks: deflators, PPPs, the price drift and its ICT counterfactuals.

The drift of a country (or group) against the United States in year t, relative to the reference year 2020, is

    ln D_t = ln(P_t / P_US,t) - ln(PPP_t)          (all indices 2020 = 1, PPP in national currency per USD)

the change in relative prices recorded by national deflators minus the change recorded by the PPPs. It equals
the ratio of the current-PPP level to the constant-PPP level, which is how the first part of the series met it.
For a group, the sums are taken first: D_G = (sum of nominal values at current PPPs) / (sum of volumes at 2020
PPPs), relative to the same ratio for the United States.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
BASE = 2020
LAST = 2024   # last year used; 2025 values exist for some economies but are preliminary and incomplete

EU = ["AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
      "LTU", "LUX", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
# the group of the first part of the series: EU27 without Cyprus and Malta, without Estonia (no 1995 data there)
EU_GROUP = [c for c in EU if c != "EST"]
NON_EU_EUROPE = ["GBR", "CHE", "NOR", "ISL", "TUR", "SRB", "MKD", "ALB"]
NON_EUROPE_OECD = ["JPN", "KOR", "CAN", "AUS", "NZL", "ISR", "MEX", "CHL", "COL", "CRI"]

AGGREGATES = {  # name: (sector, transaction, PPP code in Table 4)
    "GDP": ("S1", "B1GQ", "PPP_B1GQ"),
    "AIC": ("S1", "P41", "PPP_P41"),
    "HFCE": ("S14", "P3", "PPP_P31S14"),
}
ICT_CONSUMPTION = ["CP08", "CP091"]          # communication; audio-visual, photographic and IT equipment
ICT_INDUSTRIES = ["C26", "J"]                # computer, electronic and optical products; information and communication
SECTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]


def _pivot(df: pd.DataFrame, **match) -> pd.DataFrame:
    x = df[df.TIME_PERIOD <= LAST]
    for k, v in match.items():
        x = x[x[k] == v]
    return x.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="first")


def load() -> dict:
    t1 = pd.read_csv(DATA / "oecd_t1.csv", low_memory=False)
    t4 = pd.read_csv(DATA / "oecd_t4.csv", low_memory=False)
    t5 = pd.read_csv(DATA / "oecd_t5.csv", low_memory=False)
    t6 = pd.read_csv(DATA / "oecd_t6.csv", low_memory=False)
    gf = pd.read_csv(DATA / "oecd_gfcf.csv", low_memory=False)
    out: dict = {"V": {}, "LR": {}, "PPP": {}}
    for name, (sector, tr, ppp) in AGGREGATES.items():
        x = t1[(t1.SECTOR == sector) & (t1.TRANSACTION == tr)]
        out["V"][name] = _pivot(x, PRICE_BASE="V")
        out["LR"][name] = _pivot(x, PRICE_BASE="LR")
        out["PPP"][name] = _pivot(t4, TRANSACTION=ppp)
    out["EXC"] = _pivot(t4, TRANSACTION="EXC_A")
    out["coicop_V"] = {k: _pivot(t5, EXPENDITURE=k, PRICE_BASE="V") for k in ["_T", *ICT_CONSUMPTION]}
    out["coicop_LR"] = {k: _pivot(t5, EXPENDITURE=k, PRICE_BASE="LR") for k in ["_T", *ICT_CONSUMPTION]}
    acts = ["_T", *SECTIONS, "C26", "J58", "J59_60", "J61", "J62_63"]
    out["va_V"] = {a: _pivot(t6, ACTIVITY=a, PRICE_BASE="V") for a in acts}
    out["va_LR"] = {a: _pivot(t6, ACTIVITY=a, PRICE_BASE="LR") for a in acts}
    out["gf_V"] = {a: _pivot(gf, INSTR_ASSET=a, PRICE_BASE="V") for a in gf.INSTR_ASSET.unique()}
    out["gf_LR"] = {a: _pivot(gf, INSTR_ASSET=a, PRICE_BASE="LR") for a in gf.INSTR_ASSET.unique()}
    return out


def deflator(v: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    """Implicit deflator, BASE = 1."""
    p = v / lr
    return p / p.loc[BASE]


def group_drift(d: dict, agg: str, codes: list[str], ref: str = "USA", adj: pd.DataFrame | None = None) -> pd.Series:
    """ln D for a group of economies against `ref`, with an optional cumulated log adjustment to the deflators.

    `adj` holds, per economy and year, the log amount by which the deflator is changed (zero in BASE). A higher
    deflator means a lower volume, so the constant-PPP history moves and the current-PPP history does not.
    """
    v, lr, ppp = d["V"][agg], d["LR"][agg], d["PPP"][agg]
    vol = lr / lr.loc[BASE]
    cur = v / ppp                                    # USD at current PPPs
    con = (v.loc[BASE] / ppp.loc[BASE]) * vol        # USD at 2020 PPPs
    if adj is not None:
        con = con * np.exp(-adj.reindex(index=con.index, columns=con.columns).fillna(0.0))
    g = cur[codes].sum(axis=1, min_count=len(codes)) / con[codes].sum(axis=1, min_count=len(codes))
    r = cur[ref] / con[ref]
    return np.log(g / r)


def group_parts(d: dict, agg: str, codes: list[str], ref: str = "USA") -> pd.DataFrame:
    """Split ln D into the relative deflator and the PPP index (exact for the sum-based group)."""
    v, lr, ppp = d["V"][agg], d["LR"][agg], d["PPP"][agg]
    vol = lr / lr.loc[BASE]
    fixed = v[codes].div(ppp.loc[BASE, codes], axis=1).sum(axis=1)       # nominal at 2020 PPPs
    current = (v[codes] / ppp[codes]).sum(axis=1)                        # nominal at current PPPs
    volume = (vol[codes] * (v.loc[BASE, codes] / ppp.loc[BASE, codes])).sum(axis=1)
    p_group = fixed / volume                                             # group deflator, 2020 = 1
    ppp_index = fixed / current                                          # group PPP, 2020 = 1
    p_ref = deflator(v[ref], lr[ref])
    rel = np.log(p_group / p_ref)
    return pd.DataFrame({"relative_deflator": rel, "ppp_index": np.log(ppp_index), "drift": rel - np.log(ppp_index)})


def country_drift(d: dict, agg: str, ref: str = "USA") -> pd.DataFrame:
    p = deflator(d["V"][agg], d["LR"][agg])
    ppp = d["PPP"][agg] / d["PPP"][agg].loc[BASE]
    ppp_ref = ppp[ref]
    return np.log(p.div(p[ref], axis=0) / ppp.div(ppp_ref, axis=0))


def cumulate(step: pd.DataFrame) -> pd.DataFrame:
    """Turn annual log adjustments (indexed by the later year) into levels with the BASE year at zero."""
    step = step.fillna(0.0).sort_index()
    level = step.cumsum()
    return level - level.loc[BASE]


def torn_share(v_part: pd.DataFrame, v_tot: pd.DataFrame) -> pd.DataFrame:
    s = v_part / v_tot
    return (s + s.shift(1)) / 2


def dlog(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x).diff()


def consumption_ict(d: dict, codes: list[str]) -> dict:
    """Annual consumer-price pieces for the ICT categories, from Table 5 (domestic concept)."""
    vT, lT = d["coicop_V"]["_T"], d["coicop_LR"]["_T"]
    pT = dlog(vT / lT)
    pieces = {}
    for k in ICT_CONSUMPTION:
        v, lr = d["coicop_V"][k], d["coicop_LR"][k]
        pieces[k] = {"share": torn_share(v, vT), "rel": dlog(v / lr) - pT}   # price change relative to all consumption
    # reference path for Europe: consumption-weighted mean of the relative price changes of the EU members reporting
    eu_rel = {}
    for k in ICT_CONSUMPTION:
        rel = pieces[k]["rel"].reindex(columns=codes)
        # weights: consumption converted to US dollars at the household-consumption PPP of each year
        usd = vT.reindex(columns=codes) / d["PPP"]["HFCE"].reindex(index=vT.index, columns=codes)
        w = usd.where(rel.notna())
        eu_rel[k] = (rel * w).sum(axis=1, min_count=1) / w.sum(axis=1, min_count=1)
    return {"pieces": pieces, "eu_rel": eu_rel, "total_infl": pT, "usd": usd}


def consumption_adjustments(d: dict, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Annual log adjustments to household-consumption deflators under three ICT counterfactuals."""
    ci = consumption_ict(d, codes)
    pieces, eu_rel = ci["pieces"], ci["eu_rel"]
    cols = sorted(set(codes) | {"USA"})
    idx = d["V"]["HFCE"].index
    us_like_eu = pd.DataFrame(0.0, index=idx, columns=cols)
    eu_like_us = pd.DataFrame(0.0, index=idx, columns=cols)
    excluded = pd.DataFrame(0.0, index=idx, columns=cols)
    s_tot = pd.DataFrame(0.0, index=idx, columns=cols)
    for k in ICT_CONSUMPTION:
        s = pieces[k]["share"].reindex(index=idx, columns=cols)
        rel = pieces[k]["rel"].reindex(index=idx, columns=cols)
        # United States priced like Europe: replace its relative ICT price change by the European one
        us_like_eu["USA"] += (s["USA"] * (eu_rel[k].reindex(idx) - rel["USA"])).fillna(0.0)
        # each European economy priced like the United States
        for c in codes:
            eu_like_us[c] += (s[c] * (rel["USA"] - rel[c])).fillna(0.0)
        # ICT categories left out of the deflator: P_x moves by -sum(s*rel)/(1-sum(s))
        excluded += (-(s * rel)).fillna(0.0)
        s_tot += s.fillna(0.0)
    excluded = excluded / (1 - s_tot)
    return {"us_like_eu": us_like_eu, "eu_like_us": eu_like_us, "excluded": excluded}


def production_adjustments(d: dict, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Annual log adjustments to GDP deflators from the value added of the ICT-producing industries."""
    vT, lT = d["va_V"]["_T"], d["va_LR"]["_T"]
    pT = dlog(vT / lT)
    cols = sorted(set(codes) | {"USA"})
    idx = vT.index
    rel, share = {}, {}
    for a in ICT_INDUSTRIES:   # C26 lies inside section C and J is a section of its own, so the two are disjoint
        v, lr = d["va_V"][a], d["va_LR"][a]
        rel[a] = (dlog(v / lr) - pT).reindex(index=idx, columns=cols)
        share[a] = torn_share(v, vT).reindex(index=idx, columns=cols)
    eu_rel = {}
    usd = vT.reindex(columns=codes) / d["PPP"]["GDP"].reindex(index=idx, columns=codes)   # GVA at GDP PPPs
    for a in ICT_INDUSTRIES:
        w = usd.where(rel[a][codes].notna())
        eu_rel[a] = (rel[a][codes] * w).sum(axis=1, min_count=1) / w.sum(axis=1, min_count=1)
    excluded = pd.DataFrame(0.0, index=idx, columns=cols)
    s_tot = pd.DataFrame(0.0, index=idx, columns=cols)
    us_like_eu = pd.DataFrame(0.0, index=idx, columns=cols)
    eu_like_us = pd.DataFrame(0.0, index=idx, columns=cols)
    for a in ICT_INDUSTRIES:
        s, r = share[a], rel[a]
        excluded += (-(s * r)).fillna(0.0)
        s_tot += s.fillna(0.0)
        us_like_eu["USA"] += (s["USA"] * (eu_rel[a] - r["USA"])).fillna(0.0)
        for c in codes:
            eu_like_us[c] += (s[c] * (r["USA"] - r[c])).fillna(0.0)
    excluded = excluded / (1 - s_tot)
    return {"us_like_eu": us_like_eu, "eu_like_us": eu_like_us, "excluded": excluded,
            "rel": rel, "share": share, "eu_rel": eu_rel}


def relative_price_change(v: pd.DataFrame, lr: pd.DataFrame, vt: pd.DataFrame, lrt: pd.DataFrame,
                          ppp: pd.DataFrame, codes: list[str], a: int, b: int) -> dict:
    """Cumulated change a->b of an item's price relative to its aggregate, for the US and for Europe.

    Europe's annual change is the mean over the members reporting the item, weighted by the size of the
    aggregate in US dollars at PPPs; shares are averaged over the same years with the same weights.
    """
    rel = (dlog(v / lr) - dlog(vt / lrt)).loc[a + 1:b]
    usd = (vt / ppp.reindex(index=vt.index, columns=vt.columns)).loc[a + 1:b]
    cols = [c for c in codes if c in rel.columns]
    w = usd.reindex(columns=cols).where(rel[cols].notna())
    eu = (rel[cols] * w).sum(axis=1, min_count=1) / w.sum(axis=1, min_count=1)
    share = (v / vt).loc[a + 1:b]
    s_eu = float((share.reindex(columns=cols) * w).sum().sum() / w.where(share.reindex(columns=cols).notna()).sum().sum())
    reporting = [c for c in cols if rel[c].notna().all()]
    return {"us": float(rel["USA"].sum()), "eu": float(eu.sum()), "share_us": float(share["USA"].mean()),
            "share_eu": s_eu, "n": len(reporting), "per_country": rel.sum(min_count=len(rel))}
