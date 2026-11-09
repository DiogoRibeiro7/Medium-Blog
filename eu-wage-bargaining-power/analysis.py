"""Empirical side: where each lever shows up in EU wage data, and Portugal as a special case.

    python analysis.py

Reads the snapshot in data/ (fetch.py) and the model results in results/ (experiments.py),
writes CSV tables and results/analysis_summary.txt with every number used in the article.

1. The reserve price. Structure of Earnings Survey (SES) waves 2006-2022, full-time employees,
   monthly earnings in October: the minimum wage relative to the 10th percentile and to the
   median, the low-pay share, and the within-country elasticity of P10/median with respect
   to the minimum wage/median (country and wave fixed effects, bootstrap over countries).
2. Who sits at the table. SES employees and earnings by type of collective pay agreement,
   and OECD/AIAS bargaining coverage and union density.
3. Portugal. The minimum wage against compensation per employee, the Kaitz index under four
   wage concepts, the adjusted wage share, and the model evaluated at Portugal's floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE", "DK": "DNK", "EE": "EST",
    "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LV": "LVA",
    "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK",
    "SI": "SVN", "ES": "ESP", "SE": "SWE",
}
EU27 = list(ISO3)
FOURTEEN_PAYMENTS = {"EL", "ES", "PT"}       # Eurostat converts these to a 12-month basis (x 14 / 12)
SES_WAVES = (2006, 2010, 2014, 2018, 2022)
MULTI = ("NAT", "IND", "INDREG")              # multi-employer agreements
SINGLE = ("ENT", "UNIT")                      # single-employer agreements


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", low_memory=False)


def minimum_wage_as_paid() -> pd.DataFrame:
    """Monthly minimum wage in euro as paid (14-payment countries converted back), by semester."""
    mw = read("minimum_wage_eur")
    mw = mw[mw.geo.isin(EU27)].copy()
    mw["year"] = mw.time.str[:4].astype(int)
    mw["semester"] = mw.time.str[-2:]
    mw["mw_12m"] = mw.value
    mw["mw_paid"] = np.where(mw.geo.isin(FOURTEEN_PAYMENTS), mw.value * 12 / 14, mw.value)
    return mw[["geo", "year", "semester", "mw_12m", "mw_paid"]]


def ses_floor_panel() -> pd.DataFrame:
    mw = minimum_wage_as_paid()
    mw_oct = mw[mw.semester == "S2"].set_index(["geo", "year"])["mw_paid"]
    rows = []
    for y in SES_WAVES:
        s = read(f"ses_deciles_{y}")
        qcol = "quantile" if "quantile" in s.columns else "quant_inc"
        s = s[(s.worktime == "FT") & s.geo.isin(EU27)]
        eur = s[s.unit == "EUR"].pivot_table(index="geo", columns=qcol, values="value")
        pc = s[(s.unit == "PC") & (s[qcol] == "S2")].set_index("geo")["value"]
        for geo, r in eur.iterrows():
            if not {"P10", "MED", "P90"} <= set(r.dropna().index):
                continue
            m = mw_oct.get((geo, y), np.nan)
            rows.append({"geo": geo, "wave": y, "p10": r.P10, "median": r.MED, "p90": r.P90,
                         "low_pay_share": pc.get(geo, np.nan), "mw_paid": m})
    p = pd.DataFrame(rows)
    p["kaitz"] = p.mw_paid / p["median"]
    p["mw_over_p10"] = p.mw_paid / p.p10
    p["p10_over_median"] = p.p10 / p["median"]
    p["p90_over_median"] = p.p90 / p["median"]
    return p


def fe_elasticity(p: pd.DataFrame, y: str, x: str, n_boot: int = 2000, seed: int = 1) -> dict:
    """Slope of log y on log x with country and wave fixed effects; bootstrap over countries."""
    d = p.dropna(subset=[y, x]).copy()
    d["ly"], d["lx"] = np.log(d[y]), np.log(d[x])
    d = d[d.groupby("geo").geo.transform("size") >= 2]

    def slope(df: pd.DataFrame) -> float:
        """OLS slope with country and wave dummies (exact two-way fixed effects)."""
        X = np.column_stack([df.lx.values,
                             pd.get_dummies(df.geo).values.astype(float),
                             pd.get_dummies(df.wave).values[:, 1:].astype(float)])
        coef, *_ = np.linalg.lstsq(X, df.ly.values, rcond=None)
        return float(coef[0])

    b = slope(d)
    rng = np.random.default_rng(seed)
    geos = d.geo.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(geos, size=geos.size, replace=True)
        parts = [d[d.geo == g].assign(geo=f"{g}_{k}") for k, g in enumerate(pick)]
        bd = pd.concat(parts)
        if bd.lx.std() > 0:
            boots.append(slope(bd))
    lo, hi = np.percentile(boots, [5, 95])
    return {"slope": b, "ci90_lo": lo, "ci90_hi": hi, "n_obs": len(d), "n_countries": int(geos.size)}


def agreements() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for y in (2010, 2014, 2018, 2022):
        s = read(f"ses_agreement_employees_{y}")
        s = s[(s.nace_r2 == "B-S_X_O") & (s.sizeclas == "GE10") & s.geo.isin(EU27)]
        if "sex" in s.columns:
            s = s[s.sex == "T"]
        t = s.pivot_table(index="geo", columns="cpayagr", values="value")
        for geo, r in t.iterrows():
            tot = r.get("TOTAL", np.nan)
            if not np.isfinite(tot) or tot <= 0:
                continue
            get = lambda keys: np.nansum([r.get(k, np.nan) for k in keys])  # noqa: E731
            rows.append({"geo": geo, "wave": y, "multi_employer": get(MULTI) / tot,
                         "single_employer": get(SINGLE) / tot, "other": get(("OTH",)) / tot,
                         "none": r.get("NONE", np.nan) / tot})
    shares = pd.DataFrame(rows)
    prem = []
    for y in (2018, 2022):
        e = read(f"ses_agreement_earnings_{y}")
        e = e[(e.nace_r2 == "B-S_X_O") & (e.sizeclas == "GE10") & (e.indic_se == "ERN") & (e.unit == "EUR")
              & e.geo.isin(EU27)]
        if "sex" in e.columns:
            e = e[e.sex == "T"]
        t = e.pivot_table(index="geo", columns="cpayagr", values="value")
        for geo, r in t.iterrows():
            none = r.get("NONE", np.nan)
            prem.append({"geo": geo, "wave": y, "mean_earn_none": none,
                         "ratio_multi_to_none": _mean_of(r, MULTI) / none,
                         "ratio_single_to_none": _mean_of(r, SINGLE) / none})
    return shares, pd.DataFrame(prem)


def _mean_of(r: pd.Series, keys: tuple[str, ...]) -> float:
    v = [r.get(k, np.nan) for k in keys]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def oecd_series(name: str) -> pd.DataFrame:
    d = read(name)
    inv = {v: k for k, v in ISO3.items()}
    d = d[d.REF_AREA.isin(inv)].copy()
    d["geo"] = d.REF_AREA.map(inv)
    return d.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})[["geo", "year", "value"] +
                                                                          [c for c in d.columns if c in ("AGGREGATION_OPERATION",)]]


def national_accounts() -> pd.DataFrame:
    na = read("national_accounts_cp").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    emp = read("employment_persons").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    vol = read("gdp_volume").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    x = na.join(emp).join(vol.rename(columns={"B1GQ": "B1GQ_vol"}))
    x["comp_per_employee_month"] = x.D1 / x.SAL_DC * 1e3 / 12        # million euro / thousand persons -> euro
    x["wage_share_adjusted"] = (x.D1 / x.SAL_DC) / (x.B1GQ / x.EMP_DC)
    x["real_gdp_per_worker"] = x.B1GQ_vol / x.EMP_DC
    return x.reset_index().rename(columns={"time": "year"})


def model_at_floor(share: float) -> dict:
    """Interpolate the model's floor grid at a given share of workers at the floor."""
    g = pd.read_csv(RESULTS / "model_floor_grid.csv")
    out = {}
    for v, gv in g[g.variant != "base"].groupby("variant"):
        gv = gv.sort_values("share_at_floor_base")
        x = gv.share_at_floor_base.values
        out[v] = {k: float(np.interp(share, x, gv[k].values))
                  for k in ("m", "kaitz_base", "employment_base", "d_p10", "d_p25", "d_p50", "d_p75", "d_p90", "d_mean_wage")}
    return out


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    L: list[str] = ["# Empirical results (analysis.py)", ""]

    # 1. reserve price
    p = ses_floor_panel()
    p.to_csv(RESULTS / "ses_floor_panel.csv", index=False)
    mwc = p.dropna(subset=["mw_paid"])
    L.append("## 1. Minimum wage against the SES earnings distribution (full-time, October, monthly)")
    for y in SES_WAVES:
        w = mwc[mwc.wave == y]
        L.append(f"- {y}: {len(w)} countries with a statutory minimum wage; median Kaitz {w.kaitz.median():.3f}; "
                 f"median MW/P10 {w.mw_over_p10.median():.3f}; median P10/median {w.p10_over_median.median():.3f}")
    w22 = mwc[mwc.wave == 2022].sort_values("kaitz", ascending=False)
    L.append("- 2022 by country (Kaitz, MW/P10, P10/median, P90/median, low-pay share %): " + "; ".join(
        f"{r.geo} {r.kaitz:.2f}/{r.mw_over_p10:.2f}/{r.p10_over_median:.2f}/{r.p90_over_median:.2f}/{r.low_pay_share:.1f}"
        for r in w22.itertuples()))
    rho = w22[["kaitz", "low_pay_share"]].corr(method="spearman").iloc[0, 1]
    rho2 = w22[["kaitz", "p10_over_median"]].corr(method="spearman").iloc[0, 1]
    rho3 = w22[["kaitz", "p90_over_median"]].corr(method="spearman").iloc[0, 1]
    L.append(f"- 2022 cross-country Spearman: Kaitz vs low-pay share {rho:.2f}; Kaitz vs P10/median {rho2:.2f}; "
             f"Kaitz vs P90/median {rho3:.2f} (n={len(w22)})")
    e1 = fe_elasticity(mwc, "p10_over_median", "kaitz")
    e2 = fe_elasticity(mwc, "p90_over_median", "kaitz")
    e3 = fe_elasticity(mwc, "p10", "mw_paid")
    L.append(f"- Within-country elasticity of P10/median w.r.t. MW/median (country and wave FE): {e1['slope']:.3f} "
             f"[90% CI {e1['ci90_lo']:.3f}, {e1['ci90_hi']:.3f}], {e1['n_obs']} obs, {e1['n_countries']} countries")
    L.append(f"- Same for P90/median (placebo, top of the distribution): {e2['slope']:.3f} "
             f"[{e2['ci90_lo']:.3f}, {e2['ci90_hi']:.3f}]")
    L.append(f"- Elasticity of the P10 level w.r.t. the MW level (country and wave FE): {e3['slope']:.3f} "
             f"[{e3['ci90_lo']:.3f}, {e3['ci90_hi']:.3f}]")
    e4 = fe_elasticity(mwc[mwc.wave < 2022], "p10_over_median", "kaitz")
    e5 = fe_elasticity(mwc[mwc.geo != "PT"], "p10_over_median", "kaitz")
    e6 = fe_elasticity(mwc, "low_pay_share", "kaitz")
    L.append(f"- Robustness, waves 2006-2018 only (all at enterprises with 10+ employees): {e4['slope']:.3f} "
             f"[{e4['ci90_lo']:.3f}, {e4['ci90_hi']:.3f}], {e4['n_obs']} obs; without Portugal: {e5['slope']:.3f} "
             f"[{e5['ci90_lo']:.3f}, {e5['ci90_hi']:.3f}]")
    L.append(f"- Elasticity of the low-pay share (below two-thirds of the median) w.r.t. MW/median: {e6['slope']:.3f} "
             f"[{e6['ci90_lo']:.3f}, {e6['ci90_hi']:.3f}]")
    pd.DataFrame([dict(outcome="p10_over_median", **e1), dict(outcome="p90_over_median", **e2),
                  dict(outcome="p10_level_on_mw_level", **e3), dict(outcome="p10_over_median_2006_2018", **e4),
                  dict(outcome="p10_over_median_without_PT", **e5), dict(outcome="low_pay_share", **e6)]
                 ).to_csv(RESULTS / "floor_elasticities.csv", index=False)
    pt = p[p.geo == "PT"].sort_values("wave")
    L.append("- Portugal by wave (MW as paid, P10, median, Kaitz, MW/P10, P10/median, low-pay share %): " + "; ".join(
        f"{r.wave}: {r.mw_paid:.0f}/{r.p10:.0f}/{r.median:.0f}/{r.kaitz:.3f}/{r.mw_over_p10:.3f}/{r.p10_over_median:.3f}/{r.low_pay_share:.1f}"
        for r in pt.itertuples()))
    for geo in ("PT", "DE", "ES", "EL", "FR"):
        g = p[p.geo == geo].sort_values("wave")
        L.append(f"- {geo} by wave (Kaitz, MW/P10, P10/median, P90/median): " + "; ".join(
            f"{r.wave}: {r.kaitz:.3f}/{r.mw_over_p10:.3f}/{r.p10_over_median:.3f}/{r.p90_over_median:.3f}"
            for r in g.itertuples() if np.isfinite(r.kaitz)))
    a22 = p[p.wave == 2022]
    has = a22.mw_paid.notna()
    L.append(f"- 2022 means, countries with a statutory minimum wage (n={int(has.sum())}): P10/median "
             f"{a22[has].p10_over_median.mean():.3f}, P90/median {a22[has].p90_over_median.mean():.3f}; without "
             f"(n={int((~has).sum())}): P10/median {a22[~has].p10_over_median.mean():.3f}, P90/median "
             f"{a22[~has].p90_over_median.mean():.3f}")
    nc = a22[~has & (a22.geo != "CY")]
    L.append(f"- 2022 means without a statutory minimum wage, excluding Cyprus (minimum wage from 2023), "
             f"{', '.join(nc.geo)}: P10/median {nc.p10_over_median.mean():.3f}, P90/median {nc.p90_over_median.mean():.3f}")
    d_pt = pt.set_index("wave")
    L.append(f"- Portugal 2006-2022: log change of P10 {np.log(d_pt.p10[2022] / d_pt.p10[2006]):.3f}, of the median "
             f"{np.log(d_pt['median'][2022] / d_pt['median'][2006]):.3f}, of P90 {np.log(d_pt.p90[2022] / d_pt.p90[2006]):.3f}, "
             f"of the minimum wage {np.log(d_pt.mw_paid[2022] / d_pt.mw_paid[2006]):.3f}")
    no_mw = p[p.mw_paid.isna() & (p.wave == 2022)]
    L.append("- 2022, countries without a statutory minimum wage (P10/median, P90/median): " + "; ".join(
        f"{r.geo} {r.p10_over_median:.2f}/{r.p90_over_median:.2f}" for r in no_mw.itertuples()))
    L.append("")

    # 2. who sits at the table
    shares, prem = agreements()
    shares.to_csv(RESULTS / "ses_agreement_shares.csv", index=False)
    prem.to_csv(RESULTS / "ses_agreement_earnings_ratio.csv", index=False)
    cbc = oecd_series("oecd_bargaining_coverage")
    tud = oecd_series("oecd_union_density")
    cbc.to_csv(RESULTS / "oecd_coverage.csv", index=False)
    tud.to_csv(RESULTS / "oecd_union_density.csv", index=False)
    L.append("## 2. Collective pay agreements (SES, enterprises with 10+ employees, B-S excluding O)")
    s22 = shares[shares.wave == 2022].sort_values("none")
    L.append("- 2022 shares (multi-employer / single-employer / other / none, %): " + "; ".join(
        f"{r.geo} {100 * r.multi_employer:.0f}/{100 * r.single_employer:.0f}/{100 * r.other:.0f}/{100 * r.none:.0f}"
        for r in s22.itertuples()))
    j = s22.merge(mwc[mwc.wave == 2022][["geo", "p10_over_median", "p90_over_median"]], on="geo", how="left")
    jj = s22.merge(p[p.wave == 2022][["geo", "p10_over_median", "p90_over_median"]], on="geo", how="left").dropna()
    r1 = jj[["none", "p90_over_median"]].corr(method="spearman").iloc[0, 1]
    r2 = jj[["none", "p10_over_median"]].corr(method="spearman").iloc[0, 1]
    L.append(f"- 2022 Spearman across countries: share with no agreement vs P90/median {r1:.2f}; vs P10/median {r2:.2f} "
             f"(n={len(jj)}, all countries with SES deciles)")
    del j
    p22 = prem[prem.wave == 2022].sort_values("ratio_multi_to_none")
    L.append("- 2022 mean monthly earnings relative to no agreement (multi-employer, single-employer): " + "; ".join(
        f"{r.geo} {r.ratio_multi_to_none:.2f}/{r.ratio_single_to_none:.2f}" for r in p22.itertuples()
        if np.isfinite(r.ratio_multi_to_none) or np.isfinite(r.ratio_single_to_none)))
    lat_cbc = cbc.sort_values("year").groupby("geo").tail(1)
    L.append("- OECD/AIAS coverage, latest year (%): " + "; ".join(
        f"{r.geo} {r.value:.0f} ({r.year})" for r in lat_cbc.sort_values("value").itertuples()))
    cmp_ = s22.merge(lat_cbc[["geo", "value"]], on="geo").dropna(subset=["none", "value"])
    rc = np.corrcoef(1 - cmp_.none, cmp_.value / 100)[0, 1]
    L.append(f"- Correlation of SES covered share (1 - none) with OECD coverage across countries: {rc:.2f} (n={len(cmp_)})")
    for geo in ("PT", "EL", "ES", "DE", "FR"):
        c = cbc[cbc.geo == geo].set_index("year").value
        t = tud[tud.geo == geo].set_index("year").value
        yrs = [yy for yy in (2000, 2008, 2011, 2014, 2017, 2020, 2023) if yy in c.index]
        L.append(f"- {geo} coverage: " + ", ".join(f"{yy} {c[yy]:.1f}" for yy in yrs) +
                 " | union density: " + ", ".join(f"{yy} {t[yy]:.1f}" for yy in yrs if yy in t.index))
    pts = shares[shares.geo == "PT"].sort_values("wave")
    L.append("- Portugal SES shares by wave (multi / single / other / none, %): " + "; ".join(
        f"{r.wave}: {100 * r.multi_employer:.0f}/{100 * r.single_employer:.0f}/{100 * r.other:.0f}/{100 * r.none:.0f}"
        for r in pts.itertuples()))
    ppt = prem[prem.geo == "PT"]
    L.append("- Portugal earnings ratios (multi/none, single/none): " + "; ".join(
        f"{r.wave}: {r.ratio_multi_to_none:.2f}/{r.ratio_single_to_none:.2f}" for r in ppt.itertuples()))
    L.append("")

    # 3. Portugal
    L.append("## 3. Portugal")
    mw = minimum_wage_as_paid()
    jan = mw[(mw.semester == "S1")].set_index(["geo", "year"]).mw_paid
    na = national_accounts()
    na.to_csv(RESULTS / "national_accounts_derived.csv", index=False)
    cpe = na.set_index(["geo", "year"]).comp_per_employee_month
    for a, b in ((2014, 2025), (2006, 2025), (2014, 2019), (2019, 2025)):
        g_mw = jan[("PT", b)] / jan[("PT", a)] - 1
        g_cpe = cpe[("PT", b)] / cpe[("PT", a)] - 1
        L.append(f"- {a}-{b}: minimum wage (January, as paid) {jan[('PT', a)]:.1f} -> {jan[('PT', b)]:.1f} ({100 * g_mw:+.1f}%); "
                 f"compensation per employee (national accounts, monthly, 12-month basis) {cpe[('PT', a)]:.0f} -> "
                 f"{cpe[('PT', b)]:.0f} ({100 * g_cpe:+.1f}%)")
    L.append(f"- Minimum wage January 2026 as paid: {jan[('PT', 2026)]:.1f} (Eurostat 12-month equivalent "
             f"{mw[(mw.geo == 'PT') & (mw.year == 2026) & (mw.semester == 'S1')].mw_12m.iloc[0]:.0f})")
    ratio = read("minimum_wage_ratio")
    kz = ratio[(ratio.geo == "PT") & (ratio.nace_r2 == "B-S") & (ratio.indic_se == "MMW_MED_ME_PP")].set_index("time").value
    om = oecd_series("oecd_min_to_average")
    okz = om[(om.geo == "PT") & (om.AGGREGATION_OPERATION == "MEDIAN")].set_index("year").value
    L.append("- Kaitz, Eurostat earn_mw_avgr2 (12-month MW / median, B-S, %): " + ", ".join(
        f"{yy} {kz[yy]:.1f}" for yy in (2009, 2014, 2019, 2022, 2025, 2026) if yy in kz.index))
    L.append("- Kaitz, OECD (MW / median earnings of full-time workers, %): " + ", ".join(
        f"{yy} {okz[yy]:.1f}" for yy in (2000, 2006, 2010, 2014, 2018, 2022, 2025) if yy in okz.index))
    last = om[om.AGGREGATION_OPERATION == "MEDIAN"].sort_values("year").groupby("geo").tail(1).sort_values("value")
    L.append("- OECD Kaitz, latest year, EU countries with data (%): " + "; ".join(
        f"{r.geo} {r.value:.1f} ({r.year})" for r in last.itertuples()))
    ws = na.set_index(["geo", "year"]).wage_share_adjusted
    L.append("- Adjusted wage share (compensation per employee / GDP per person employed): PT " + ", ".join(
        f"{yy} {ws[('PT', yy)]:.3f}" for yy in (1995, 2001, 2007, 2013, 2016, 2019, 2022, 2025)) +
        " | EU27 " + ", ".join(f"{yy} {ws[('EU27_2020', yy)]:.3f}" for yy in (1995, 2001, 2007, 2013, 2016, 2019, 2022, 2025)))
    pts_ws = ws.xs("PT", level="geo").loc[1995:2025]
    L.append(f"- PT wage share minimum {pts_ws.min():.3f} in {pts_ws.idxmin()}, maximum {pts_ws.max():.3f} in {pts_ws.idxmax()}")
    for share in (0.18, 0.21, 0.24):
        mf = model_at_floor(share)
        for v, r in mf.items():
            L.append(f"- Model at {100 * share:.0f}% of workers on the floor (m={r['m']:.3f}, Kaitz {r['kaitz_base']:.3f}, "
                     f"employment {100 * r['employment_base']:.1f}%), {v}: dP10 {r['d_p10']:+.2f}%, dP25 {r['d_p25']:+.2f}%, "
                     f"dP50 {r['d_p50']:+.2f}%, dP90 {r['d_p90']:+.2f}%, dmean {r['d_mean_wage']:+.2f}%")
    (RESULTS / "analysis_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
