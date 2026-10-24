"""Further analyses for the household-spending article.

1. Engel curves in Working-Leser form, fitted within each budget-survey wave
   across the five income fifths:  w_i(q) = a_i + b_i * log c(q), where w_i is
   the cash budget share of item i and c(q) cash consumption per adult
   equivalent. The budget elasticity at the mean share is e_i = 1 + b_i / mean(w_i).
   Five points per wave: the slopes are descriptive, not estimates with a
   sampling distribution.
2. Inflation by income fifth as an annual chained index from 1996, with weights
   updated at each survey wave, and two fixed-weight alternatives to show how
   much the result depends on the weights.
3. The discretionary part of the budget: cash shares of the non-core divisions
   by income fifth.
4. Price and volume: whether movements in the aggregate basic-needs share came
   from relative prices or relative quantities. Prices are HICP component
   indices aggregated with Tornqvist weights from national-accounts spending;
   volume is the residual. The split with the national accounts' own
   chain-linked volumes is also written, with the comparison of the implicit
   deflators against the HICP that explains why it is not used.
5. Other household splits: type of household, age of the reference person and
   degree of urbanisation.
6. Portugal against Spain, Italy and Greece.
7. HICP weights: what the headline index weights more than resident budgets do,
   and each division's contribution to the 2021-2023 difference.
8. Euro amounts per month: the budget survey's PPS means converted with INE's
   published mean expenditure per household, against the minimum wage.

Run after ``spending.py``::

    python deeper.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import spending as sp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

ITEMS = {
    "food": ["CP01"], "alcohol_tobacco": ["CP02"], "clothing": ["CP03"],
    "rent_actual": ["CP041"], "energy": ["CP045"], "water_repairs": ["CP043", "CP044"],
    "housing_cash": ["CP041", "CP043", "CP044", "CP045"], "furnishings": ["CP05"], "health": ["CP06"],
    "vehicles": ["CP071"], "transport_running": ["CP072", "CP073"], "transport": ["CP07"],
    "communications": ["CP08"], "recreation": ["CP09"], "education": ["CP10"],
    "restaurants_hotels": ["CP11"], "miscellaneous": ["CP12"],
}
CASH_WAVES = [1988, 1994, 1999, 2005, 2010, 2015]
DISCRETIONARY = ["alcohol_tobacco", "clothing", "furnishings", "vehicles", "communications",
                 "recreation", "education", "restaurants_hotels", "miscellaneous"]


def cash_shares() -> pd.DataFrame:
    """Cash budget shares by fifth and wave (imputed rent removed), plus cash consumption per adult eq."""
    q = sp.budget_survey_quintiles() / 1000
    e = pd.read_csv(DATA / "hbs_exp_t133.csv")
    ae = e[e["unit"] == "PPS_AE"].pivot_table(index="quant_inc", columns="year", values="value")
    rows = []
    for (quint, year), r in q.iterrows():
        if year not in CASH_WAVES:
            continue
        cash = 1 - r["CP042"]
        rec = {"quintile": quint, "year": year,
               "cash_consumption_pps_ae": ae.loc[quint, year] * cash}
        for name, codes in ITEMS.items():
            rec[name] = r[codes].sum() / cash
        rows.append(rec)
    return pd.DataFrame(rows).set_index(["quintile", "year"]).sort_index()


def working_leser(cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in CASH_WAVES:
        d = cs.xs(year, level="year").loc[sp.QUINTILES]
        x = np.log(d["cash_consumption_pps_ae"].to_numpy())
        for item in ITEMS:
            y = d[item].to_numpy()
            b, a = np.polyfit(x, y, 1)
            fitted = a + b * x
            ss_res = float(((y - fitted) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            rows.append({"year": year, "item": item, "slope": b, "mean_share": y.mean(),
                         "elasticity": 1 + b / y.mean() if y.mean() > 0 else np.nan,
                         "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan})
    return pd.DataFrame(rows)


def wave_weights(wave: int, quint: str) -> pd.Series:
    q = sp.budget_survey_quintiles() / 1000
    t = sp.budget_survey_total() / 1000
    r = t.loc[wave] if quint == "TOTAL" else q.loc[(quint, wave)]
    w = r[sp.DIVISIONS].copy()
    w["CP04"] = w["CP04"] - r["CP042"]
    return w / w.sum()


SEGMENTS = [(1996, 2005, 1999), (2005, 2010, 2005), (2010, 2015, 2010), (2015, sp.HICP_END, 2015)]


def chained_index(quint: str, segments=SEGMENTS) -> pd.Series:
    """Annual index, 1996 = 100, weights switching at each segment start (Laspeyres within segments)."""
    p = sp.hicp()[sp.DIVISIONS]
    level, out = 100.0, {}
    for start, end, wave in segments:
        w = wave_weights(wave, quint)
        for y in range(start, end + 1):
            out[y] = level * float((w * p.loc[y] / p.loc[start]).sum())
        level = out[end]
    return pd.Series(out)


def fixed_index(quint: str, wave: int, start: int = 1996, end: int = sp.HICP_END) -> pd.Series:
    p = sp.hicp()[sp.DIVISIONS]
    w = wave_weights(wave, quint)
    return pd.Series({y: 100 * float((w * p.loc[y] / p.loc[start]).sum()) for y in range(start, end + 1)})


def main() -> None:
    lines: list[str] = []
    w = lines.append

    cs = cash_shares()
    cs.round(5).to_csv(RESULTS / "cash_shares_by_quintile.csv")
    wl = working_leser(cs)
    wl.round(5).to_csv(RESULTS / "working_leser.csv", index=False)
    w("Working-Leser Engel slopes across income fifths within each wave (cash shares, log cash consumption per adult equivalent)")
    w("  item: slope b (percentage points of budget share per unit of log consumption) / budget elasticity at mean share / R2")
    for item in ["food", "housing_cash", "rent_actual", "energy", "water_repairs", "health", "transport_running",
                 "vehicles", "restaurants_hotels", "recreation", "education", "clothing", "communications"]:
        d = wl[wl["item"] == item].set_index("year")
        w(f"  {item}: " + "; ".join(
            f"{y} b {100 * r['slope']:+.1f} e {r['elasticity']:.2f} R2 {r['r2']:.2f}" for y, r in d.iterrows()))
    w("")

    w("Discretionary items, cash share by fifth (QU1 / QU5)")
    for item in DISCRETIONARY + ["transport_running", "health"]:
        w(f"  {item}: " + "; ".join(
            f"{y} {100 * cs.loc[('QU1', y), item]:.1f}/{100 * cs.loc[('QU5', y), item]:.1f}" for y in CASH_WAVES))
    disc = cs[DISCRETIONARY].sum(axis=1)
    w("  all discretionary items listed: " + "; ".join(
        f"{y} {100 * disc[('QU1', y)]:.1f}/{100 * disc[('QU5', y)]:.1f}" for y in CASH_WAVES))
    w("")

    idx = pd.DataFrame({q: chained_index(q) for q in sp.QUINTILES + ["TOTAL"]})
    idx["HICP"] = 100 * sp.hicp()["CP00"].loc[1996:sp.HICP_END] / sp.hicp().loc[1996, "CP00"]
    idx.round(3).to_csv(RESULTS / "quintile_price_index_chained.csv")
    w("Chained price index by fifth, 1996 = 100 (weights 1999 wave to 2005, 2005 to 2010, 2010 to 2015, 2015 after)")
    for y in (2000, 2005, 2010, 2015, 2020, 2021, 2022, 2023, 2024, 2025):
        w(f"  {y}: " + ", ".join(f"{c} {idx.loc[y, c]:.1f}" for c in idx.columns))
    gap = idx["QU1"] - idx["QU5"]
    w("  QU1 minus QU5, index points: " + ", ".join(f"{y} {gap[y]:+.1f}" for y in (2000, 2005, 2010, 2015, 2020, 2025)))
    w(f"  cumulative 1996-2025: QU1 +{idx.loc[2025, 'QU1'] - 100:.1f}%, QU5 +{idx.loc[2025, 'QU5'] - 100:.1f}%, "
      f"HICP +{idx.loc[2025, 'HICP'] - 100:.1f}%")
    w("  sensitivity to the weights (QU1 minus QU5 in 2025, cumulative from 1996, percentage points):")
    sens = {"chained": idx.loc[2025, "QU1"] - idx.loc[2025, "QU5"]}
    for wave in (1999, 2005, 2010, 2015):
        a, b = fixed_index("QU1", wave), fixed_index("QU5", wave)
        sens[f"fixed {wave}"] = a[2025] - b[2025]
    w("    " + ", ".join(f"{k} {v:+.1f}" for k, v in sens.items()))
    pd.Series(sens).round(6).to_csv(RESULTS / "quintile_inflation_weight_sensitivity.csv", header=["qu1_minus_qu5_2025"])
    w("")

    price_volume(w)
    household_splits(w)
    peers(w)
    hicp_weights(w)
    euro_levels(w)

    (RESULTS / "deeper_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------ price/volume ---

CASH_COMPONENTS = ["CP01", "CP02", "CP03", "CP041", "CP043", "CP044", "CP045", "CP05", "CP06",
                   "CP07", "CP08", "CP09", "CP10", "CP11", "CP12"]


def tornqvist_volume(cp: pd.DataFrame, vol: pd.DataFrame, comps: list[str]) -> pd.Series:
    """Chained Tornqvist volume index of the aggregate of ``comps`` (first year = 1)."""
    shares = cp[comps].div(cp[comps].sum(axis=1), axis=0)
    dlog = np.log(vol[comps]).diff()
    avg = 0.5 * (shares + shares.shift(1))
    growth = (avg * dlog).sum(axis=1)
    growth.iloc[0] = 0.0
    return np.exp(growth.cumsum())


def price_volume(w) -> None:
    cp = pd.read_csv(DATA / "nama_10_co3_p3.csv").pivot_table(index="year", columns="coicop", values="value")
    vol = pd.read_csv(DATA / "nama_10_co3_p3_clv20.csv").pivot_table(index="year", columns="coicop", values="value")
    cp, vol = cp.loc[1995:sp.NA_END], vol.loc[1995:sp.NA_END]
    v_core = tornqvist_volume(cp, vol, sp.CORE)
    v_all = tornqvist_volume(cp, vol, CASH_COMPONENTS)
    nom_core, nom_all = cp[sp.CORE].sum(axis=1), cp[CASH_COMPONENTS].sum(axis=1)
    p_core, p_all = nom_core / v_core, nom_all / v_all
    share = nom_core / nom_all
    out = pd.DataFrame({"share": share, "rel_price": p_core / p_all, "rel_volume": v_core / v_all})
    out.round(6).to_csv(RESULTS / "core_share_price_volume.csv")
    check = (out["rel_price"] * out["rel_volume"] / out["share"]).std()
    w("Core basic-needs share of cash consumption (national accounts): change split into relative price and relative volume")
    w("  (Tornqvist chained volumes from CLV20 components; log points; total = price + volume)")
    for a, b in ((1995, 2008), (2008, 2013), (2013, 2019), (2019, 2020), (2020, 2022), (1995, 2022)):
        tot = 100 * np.log(out.loc[b, "share"] / out.loc[a, "share"])
        pr = 100 * np.log(out.loc[b, "rel_price"] / out.loc[a, "rel_price"])
        vo = 100 * np.log(out.loc[b, "rel_volume"] / out.loc[a, "rel_volume"])
        w(f"  {a}-{b}: share {100 * out.loc[a, 'share']:.1f}% -> {100 * out.loc[b, 'share']:.1f}%, "
          f"total {tot:+.1f} = price {pr:+.1f} + volume {vo:+.1f}")
    # The residual is at machine precision and its exact value depends on the CPU, so only its size is reported.
    w(f"  identity check (std of price x volume / share): {'below 1e-12' if check < 1e-12 else f'{check:.2e}'}")
    w("  volume growth 2008-2013: core " + f"{100 * (v_core[2013] / v_core[2008] - 1):+.1f}%, "
      f"all cash consumption {100 * (v_all[2013] / v_all[2008] - 1):+.1f}%; 2019-2020: core "
      f"{100 * (v_core[2020] / v_core[2019] - 1):+.1f}%, all {100 * (v_all[2020] / v_all[2019] - 1):+.1f}%")
    # The national-accounts implicit deflators against the HICP.
    h = sp.hicp()
    w("  implicit national-accounts deflator vs HICP, 1996 relative to 2020:")
    for c in ("CP01", "CP041", "CP045"):
        na_defl = cp.loc[1996, c] / vol.loc[1996, c] / (cp.loc[2020, c] / vol.loc[2020, c])
        w(f"    {c}: national accounts {na_defl:.3f}, HICP {h.loc[1996, c] / h.loc[2020, c]:.3f}; "
          f"national-accounts volume 1996-2020 {100 * (vol.loc[2020, c] / vol.loc[1996, c] - 1):+.1f}%")
    na_tot = cp.loc[1996, "TOTAL"] / vol.loc[1996, "TOTAL"] / (cp.loc[2020, "TOTAL"] / vol.loc[2020, "TOTAL"])
    w(f"    total: national accounts {na_tot:.3f}, HICP {h.loc[1996, 'CP00'] / h.loc[2020, 'CP00']:.3f}")

    # Same split with HICP prices: Tornqvist price index of each aggregate, volume as the residual.
    def tornqvist_price(comps: list[str]) -> pd.Series:
        c = cp.loc[1996:sp.NA_END, comps]
        shares = c.div(c.sum(axis=1), axis=0)
        dlog = np.log(h.loc[1996:sp.NA_END, comps]).diff()
        g = (0.5 * (shares + shares.shift(1)) * dlog).sum(axis=1)
        g.iloc[0] = 0.0
        return np.exp(g.cumsum())

    pc, pa = tornqvist_price(sp.CORE), tornqvist_price(CASH_COMPONENTS)
    sh = share.loc[1996:sp.NA_END]
    rel_p = pc / pa
    rel_v = sh / sh.iloc[0] / rel_p
    hv = pd.DataFrame({"share": sh, "rel_price_hicp": rel_p, "rel_volume_residual": rel_v})
    hv.round(6).to_csv(RESULTS / "core_share_price_volume_hicp.csv")
    w("  with HICP prices (Tornqvist over components, current-price weights) and volume as the residual:")
    for a, b in ((1996, 2008), (2008, 2013), (2013, 2019), (2019, 2020), (2020, 2022), (1996, 2022)):
        tot = 100 * np.log(sh[b] / sh[a])
        pr = 100 * np.log(rel_p[b] / rel_p[a])
        w(f"    {a}-{b}: share {100 * sh[a]:.1f}% -> {100 * sh[b]:.1f}%, total {tot:+.1f} = price {pr:+.1f} "
          f"+ volume {tot - pr:+.1f}")
    w("")


# -------------------------------------------------------- household splits ---

SPLITS = {
    "hbs_str_t224": ("hhcomp", {"A1": "single adult", "A1_DCH": "single adult with children",
                                "A2": "two adults", "A2_DCH": "two adults with children",
                                "A_GE3": "three or more adults", "A_GE3_DCH": "three or more adults with children"}),
    "hbs_str_t225": ("age", {"Y_LT30": "under 30", "Y30-44": "30 to 44", "Y45-59": "45 to 59",
                             "Y_GE60": "60 and over"}),
    "hbs_str_t226": ("deg_urb", {"DEG1": "cities", "DEG2": "towns and suburbs", "DEG3": "rural areas"}),
}


def split_shares(table: str) -> pd.DataFrame:
    dim, labels = SPLITS[table]
    d = pd.read_csv(DATA / f"{table}.csv")
    d = d[d[dim].isin(labels)]
    piv = d.pivot_table(index=[dim, "year"], columns="coicop", values="value") / 1000
    rows = []
    for (g, year), r in piv.iterrows():
        rec = {"group": labels[g], "year": year, "housing_incl_imputed": r.get("CP04", np.nan),
               "food_total_basis": r.get("CP01", np.nan)}
        if pd.notna(r.get("CP042")):
            cash = 1 - r["CP042"]
            rec.update({"core_ex_imputed": r[sp.CORE].sum() / cash, "food": r["CP01"] / cash,
                        "rent_actual": r["CP041"] / cash, "energy": r["CP045"] / cash,
                        "health": r["CP06"] / cash, "imputed_rent": r["CP042"]})
        rows.append(rec)
    return pd.DataFrame(rows).set_index(["group", "year"]).sort_index()


def household_splits(w) -> None:
    for table, (dim, labels) in SPLITS.items():
        s = split_shares(table)
        s.round(5).to_csv(RESULTS / f"split_{dim}.csv")
        w(f"Budget survey by {dim}: core basic needs, cash share (1988 / 1999 / 2015), and 2015 rent, energy, food, health")
        for g in labels.values():
            if g not in s.index.get_level_values(0):
                continue
            c = s.loc[g, "core_ex_imputed"]
            r15 = s.loc[(g, 2015)]
            w(f"  {g}: core " + " / ".join(f"{100 * c.get(y, np.nan):.1f}" for y in (1988, 1999, 2015))
              + f"; 2015 rent {100 * r15['rent_actual']:.1f}, energy {100 * r15['energy']:.1f}, "
              f"food {100 * r15['food']:.1f}, health {100 * r15['health']:.1f}, imputed rent "
              f"{100 * r15['imputed_rent']:.1f} of total; housing incl. imputed 2020 (2022/23) "
              f"{100 * s.loc[(g, 2020), 'housing_incl_imputed']:.1f}")
        w("")


# ------------------------------------------------------------------- peers ---

PEER_GEOS = ("PT", "ES", "IT", "EL")


def sp_waves() -> list[int]:
    return [1988, 1994, 1999, 2005, 2010, 2015, 2020]


def peers(w) -> None:
    pt = pd.read_csv(DATA / "hbs_str_t223.csv")
    pe = pd.read_csv(DATA / "hbs_str_t223_peers.csv")
    d = pd.concat([pt, pe], ignore_index=True)
    d = d[d["quant_inc"].isin(sp.QUINTILES)]
    piv = d.pivot_table(index=["geo", "quant_inc", "year"], columns="coicop", values="value") / 1000
    rows = []
    for (geo, q, year), r in piv.iterrows():
        if pd.isna(r.get("CP042")):
            continue
        cash = 1 - r["CP042"]
        rows.append({"geo": geo, "quintile": q, "year": year, "core_ex_imputed": r[sp.CORE].sum() / cash,
                     "food": r["CP01"] / cash, "rent_actual": r["CP041"] / cash, "energy": r["CP045"] / cash,
                     "imputed_rent": r["CP042"]})
    out = pd.DataFrame(rows).set_index(["geo", "quintile", "year"]).sort_index()
    out.round(5).to_csv(RESULTS / "peers_by_quintile.csv")
    w("Core basic needs, cash share, poorest and richest fifth, by country, every wave with components")
    for geo in PEER_GEOS:
        parts = []
        for y in sp_waves():
            try:
                a, b = out.loc[(geo, "QU1", y), "core_ex_imputed"], out.loc[(geo, "QU5", y), "core_ex_imputed"]
                parts.append(f"{y} {100 * a:.1f}/{100 * b:.1f} (ratio {a / b:.2f})")
            except KeyError:
                continue
        w(f"  {geo}: " + "; ".join(parts))
    for y in (2015, 2020):
        w(f"Actual rent, energy and food, cash share, poorest / richest fifth, {y}")
        for geo in PEER_GEOS:
            try:
                r1, r5 = out.loc[(geo, "QU1", y)], out.loc[(geo, "QU5", y)]
            except KeyError:
                w(f"  {geo}: n/a")
                continue
            w(f"  {geo}: rent {100 * r1['rent_actual']:.1f} / {100 * r5['rent_actual']:.1f}; energy "
              f"{100 * r1['energy']:.1f} / {100 * r5['energy']:.1f}; food {100 * r1['food']:.1f} / {100 * r5['food']:.1f}")
    w("")


# ------------------------------------------------------------------ euros ---

def minimum_wage() -> pd.Series:
    """National minimum wage, annual average of the daily rate in force, EUR per month (14 payments a year)."""
    c = pd.read_csv(DATA / "minimum_wage_changes.csv", parse_dates=["effective"])
    c["eur"] = np.where(c["currency"] == "PTE", c["value"] / 200.482, c["value"])
    days = pd.date_range(c["effective"].min(), f"{c['effective'].dt.year.max()}-12-31", freq="D")
    rate = c.set_index("effective")["eur"].reindex(days).ffill()
    return rate.groupby(rate.index.year).mean()


def euro_levels(w) -> None:
    """Approximate euro amounts per month for the waves whose INE euro total is known.

    The budget survey tables give PPS. INE publishes the mean expenditure per household in euros;
    the ratio of the two, applied to every fifth, converts the fifths' PPS amounts into euros.
    """
    ine = pd.read_csv(DATA / "ine_idef_totals.csv").set_index("wave")["mean_expenditure_eur_per_household"]
    e = pd.read_csv(DATA / "hbs_exp_t133.csv")
    pps = e.pivot_table(index=["unit", "quant_inc"], columns="year", values="value")
    q = sp.budget_survey_quintiles() / 1000
    mw = minimum_wage()
    rows = []
    w("Euro amounts per month (INE mean expenditure per household / Eurostat PPS mean, applied to each fifth)")
    fieldwork = pd.read_csv(DATA / "ine_idef_totals.csv").set_index("wave")["fieldwork"]
    mw_wave = {wave: float(np.mean([mw[int(y)] for y in fieldwork[wave].split("/")])) for wave in ine.index}
    pd.DataFrame({"fieldwork": fieldwork, "monthly": pd.Series(mw_wave),
                  "monthly_over_12": pd.Series(mw_wave) * 14 / 12}).rename_axis("wave").round(2).to_csv(
        RESULTS / "minimum_wage_fieldwork.csv")
    for wave, eur in ine.items():
        factor = eur / pps.loc[("PPS_HH", "TOTAL"), wave]
        w(f"  {wave} wave (fieldwork {fieldwork[wave]}): factor {factor:.3f} EUR per PPS; minimum wage averaged over the "
          f"fieldwork years {mw_wave[wave]:.1f} a month, {mw_wave[wave] * 14 / 12:.0f} spread over twelve months")
        for quint in sp.QUINTILES:
            r = q.loc[(quint, wave)]
            hh = factor * pps.loc[("PPS_HH", quint), wave] / 12
            ae = factor * pps.loc[("PPS_AE", quint), wave] / 12
            rec = {"wave": wave, "quintile": quint, "factor": factor, "consumption_hh": hh, "consumption_ae": ae,
                   "food_hh": hh * r["CP01"]}
            if pd.notna(r.get("CP042")):
                cash = 1 - r["CP042"]
                core = r[sp.CORE].sum()
                housing = r[["CP041", "CP043", "CP044", "CP045"]].sum()
                rec.update({"cash_hh": hh * cash, "core_hh": hh * core, "left_hh": hh * (cash - core),
                            "core_ae": ae * core, "left_ae": ae * (cash - core),
                            "food_ae": ae * r["CP01"], "housing_cash_ae": ae * housing})
            rows.append(rec)
    out = pd.DataFrame(rows).set_index(["wave", "quintile"])
    out.round({**{c: 2 for c in out.columns}, "factor": 6}).to_csv(RESULTS / "euro_levels.csv")
    for wave in ine.index:
        for quint in ("QU1", "QU3", "QU5"):
            r = out.loc[(wave, quint)]
            line = f"    {wave} {quint}: consumption {r['consumption_hh']:.0f} per household, {r['consumption_ae']:.0f} per adult eq.; food {r['food_hh']:.0f}"
            if pd.notna(r.get("core_hh")):
                line += (f"; cash {r['cash_hh']:.0f}, core {r['core_hh']:.0f}, left after core {r['left_hh']:.0f} per household"
                         f" ({r['core_ae']:.0f} and {r['left_ae']:.0f} per adult eq.)")
            w(line)
    w("  per adult equivalent, 2015/16, food / cash housing (rent, energy, water, repairs) a month: " + ", ".join(
        f"{qq} {out.loc[(2015, qq), 'food_ae']:.0f} / {out.loc[(2015, qq), 'housing_cash_ae']:.0f}" for qq in sp.QUINTILES))
    ppp = pd.read_csv(DATA / "prc_ppp_ind_a01.csv")
    ppp = ppp[ppp["na_item"] == "PPP_EU27_2020"].set_index("year")["value"]
    w("  check: Eurostat PPP for actual individual consumption, EUR per PPS (EU27_2020): " + ", ".join(
        f"{y} {ppp[y]:.3f}" for y in (2010, 2011, 2015, 2016, 2020, 2022, 2023) if y in ppp.index))
    # The 2015 core basket of the poorest fifth at 2025 prices, quantities held fixed.
    h = sp.hicp()
    r = q.loc[("QU1", 2015)]
    wk = r[sp.CORE] / r[sp.CORE].sum()
    rel = h.loc[sp.HICP_END, sp.CORE] / h.loc[2015, sp.CORE]
    up = float((wk * rel).sum())
    core15 = out.loc[(2015, "QU1"), "core_hh"]
    w(f"  poorest fifth's 2015 core basket at {sp.HICP_END} prices (fixed quantities): {core15:.0f} -> {core15 * up:.0f} "
      f"a month per household (+{100 * (up - 1):.1f}%); minimum wage {mw[2015]:.0f} -> {mw[sp.HICP_END]:.0f} "
      f"(+{100 * (mw[sp.HICP_END] / mw[2015] - 1):.1f}%); core basket as % of the minimum wage spread over twelve months: "
      f"{100 * core15 / (mw_wave[2015] * 14 / 12):.1f}% (2015/16) -> {100 * core15 * up / (mw[sp.HICP_END] * 14 / 12):.1f}%")
    for wave in (2010, 2015):
        r = out.loc[(wave, "QU1")]
        w(f"  {wave} QU1 core per household as % of the minimum wage over twelve months: "
          f"{100 * r['core_hh'] / (mw_wave[wave] * 14 / 12):.1f}%; all consumption per household: "
          f"{100 * r['consumption_hh'] / (mw_wave[wave] * 14 / 12):.1f}%")
    r = out.loc[(2020, "QU1")]
    w(f"  2020 (2022/23) QU1 consumption per household as % of the minimum wage over twelve months: "
      f"{100 * r['consumption_hh'] / (mw_wave[2020] * 14 / 12):.1f}%; food {r['food_hh']:.0f} a month")
    w("")


# ------------------------------------------------------------ HICP weights ---

def hicp_weights(w) -> None:
    wt = pd.read_csv(DATA / "prc_hicp_inw.csv").pivot_table(index="year", columns="coicop", values="value") / 1000
    t = sp.budget_survey_total() / 1000
    cash_total = t.loc[2015, sp.DIVISIONS].copy()
    cash_total["CP04"] -= t.loc[2015, "CP042"]
    cash_total = cash_total / cash_total.sum()
    w("HICP division weights against budget-survey cash shares (2015 wave)")
    for c in ("CP01", "CP04", "CP07", "CP11", "CP06", "CP09"):
        w1 = wave_weights(2015, "QU1")[c]
        w5 = wave_weights(2015, "QU5")[c]
        w(f"  {c}: HICP weight 2015 {100 * wt.loc[2015, c]:.1f}%, 2022 {100 * wt.loc[2022, c]:.1f}%, "
          f"2023 {100 * wt.loc[2023, c]:.1f}%; survey cash share all households {100 * cash_total[c]:.1f}%, "
          f"QU1 {100 * w1:.1f}%, QU5 {100 * w5:.1f}%")
    p = sp.hicp()
    hicp_w = wt.loc[2023, sp.DIVISIONS] / wt.loc[2023, sp.DIVISIONS].sum()
    rel = p[sp.DIVISIONS].loc[2023] / p[sp.DIVISIONS].loc[2021]
    resident = cash_total
    swap = hicp_w.copy()
    swap["CP11"] = resident["CP11"]
    swap = swap / swap.sum()
    w(f"  2021-2023 basket with HICP 2023 weights: {100 * ((hicp_w * rel).sum() - 1):.1f}%; "
      f"same with restaurants and hotels at the survey cash share: {100 * ((swap * rel).sum() - 1):.1f}%; "
      f"survey cash weights: {100 * ((resident * rel).sum() - 1):.1f}%; published HICP: "
      f"{100 * (p.loc[2023, 'CP00'] / p.loc[2021, 'CP00'] - 1):.1f}%")
    # Difference between the two baskets, division by division. Both weight vectors sum to one, so
    # sum((w_hicp - w_survey) * rel) = sum((w_hicp - w_survey) * (rel - r_survey)), with r_survey the survey
    # basket's relative price: a division adds to the gap when the HICP gives it more weight and its price
    # rose faster than the survey basket, or less weight and its price rose more slowly.
    r_survey = float((resident * rel).sum())
    contrib = (hicp_w - resident) * (rel - r_survey)
    pd.DataFrame({
        "hicp_weight_2015": wt.loc[2015, sp.DIVISIONS] / wt.loc[2015, sp.DIVISIONS].sum(),
        "hicp_weight_2023": hicp_w, "survey_cash_all_2015": resident,
        "survey_cash_qu1_2015": wave_weights(2015, "QU1"), "survey_cash_qu5_2015": wave_weights(2015, "QU5"),
        "price_change_2021_2023": rel - 1, "contribution_to_gap": contrib,
    }).rename_axis("coicop").round(6).to_csv(RESULTS / "hicp_weights_vs_survey.csv")
    w("  HICP 2023 weights minus survey cash weights, contribution to the 2021-2023 difference (points): "
      + ", ".join(f"{c} {100 * v:+.2f} (weight {100 * hicp_w[c]:.1f} vs {100 * resident[c]:.1f}, price "
                  f"{100 * (rel[c] - 1):+.1f}%)" for c, v in contrib.sort_values(ascending=False).items() if abs(v) >= 0.0005))
    w(f"  total {100 * contrib.sum():+.2f}")
    w("")


if __name__ == "__main__":
    main()
