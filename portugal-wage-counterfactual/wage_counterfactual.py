"""Counterfactual wages for Portugal: the base-year wage moved by inflation plus productivity.

Each year the wage is raised by consumer-price inflation plus real
labour-productivity growth,

    W_t = W_{t-1} * (1 + pi_t + g_t)            additive rule (headline)
    W_t = W_{t-1} * (1 + pi_t) * (1 + g_t)      multiplicative rule

and the result in 2024 is compared with what wages actually reached.

The wage series that reaches back to 1975 is national-accounts compensation
per employee (D1 / employees): AMECO for 1975-1995, chain-linked at 1995 to the
Eurostat snapshot in ``../portugal-labour-share/data`` for 1995-2024. The
counterfactual-to-actual ratio is then applied to the 2024 Quadros de Pessoal
mean and median monthly earnings. No Portuguese median wage exists for 1975,
so the median result assumes the median/mean ratio of the base year equals
that of 2024.

Run from this folder::

    python wage_counterfactual.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SNAPSHOT = HERE.parent / "portugal-labour-share" / "data"
RESULTS = HERE / "results"

START, END, LINK = 1975, 2024, 1995
PRODUCTIVITY = {"person": "g_person", "hour": "g_hour"}


def load_ameco() -> pd.DataFrame:
    return pd.read_csv(DATA / "ameco_prt.csv", index_col="year")


def load_eurostat() -> pd.DataFrame:
    gdp = pd.read_csv(SNAPSHOT / "nama_10_gdp.csv")
    gdp = gdp[gdp["geo"] == "PT"].set_index("year")[["B1GQ", "D1"]]
    emp = pd.read_csv(SNAPSHOT / "nama_10_a10_e_PT.csv").set_index("year")
    extra = pd.read_csv(DATA / "eurostat_productivity_d11.csv", index_col="year")
    e = gdp.join(emp[["EMP_DC_THS_PER", "SAL_DC_THS_PER", "EMP_DC_THS_HW", "SAL_DC_THS_HW"]]).join(extra)
    e["comp_per_emp"] = e["D1"] / e["SAL_DC_THS_PER"]            # thousand EUR
    e["wages_per_emp"] = e["d11_meur"] / e["SAL_DC_THS_PER"]     # thousand EUR
    e["gdp_per_worker"] = e["B1GQ"] / e["EMP_DC_THS_PER"]        # thousand EUR
    e["adj_wage_share"] = 100 * e["comp_per_emp"] / e["gdp_per_worker"]
    e["wage_share"] = 100 * e["D1"] / e["B1GQ"]
    return e


def load_qp() -> pd.DataFrame:
    return pd.read_csv(DATA / "qp_earnings.csv", index_col="year")


def growth(levels: pd.Series) -> pd.Series:
    return levels / levels.shift(1) - 1


def linked(ameco_part: pd.Series, euro_part: pd.Series) -> pd.Series:
    """AMECO levels to LINK rescaled onto Eurostat, Eurostat levels after LINK."""
    factor = euro_part.loc[LINK] / ameco_part.loc[LINK]
    return pd.concat([ameco_part.loc[START:LINK] * factor, euro_part.loc[LINK + 1:END]])


def build_inputs() -> pd.DataFrame:
    """Annual inflation, productivity growth and linked actual series, START..END."""
    ameco, euro = load_ameco(), load_eurostat()
    cpi = pd.read_csv(DATA / "cpi_inflation.csv", index_col="year")["cpi_inflation_pct"] / 100

    t = pd.DataFrame(index=pd.RangeIndex(START, END + 1, name="year"))
    t["infl"] = cpi.reindex(t.index)
    t["g_person"] = pd.concat([
        growth(ameco["gdp_real_per_emp"]).loc[START:LINK],
        growth(euro["rlpr_per_i15"]).loc[LINK + 1:END],
    ])
    t["g_hour"] = pd.concat([
        growth(ameco["gdp_real"] / ameco["hours"]).loc[START:LINK],
        growth(euro["rlpr_hw_i15"]).loc[LINK + 1:END],
    ])
    t["actual"] = 1000 * linked(ameco["comp_per_emp"], euro["comp_per_emp"])   # EUR per year
    t["adj_wage_share"] = linked(ameco["adj_wage_share"], euro["adj_wage_share"])
    nominal_per_worker = linked(
        ameco["gdp_real_per_emp"] * ameco["gdp_deflator"], euro["gdp_per_worker"])
    real_per_worker = (1 + t["g_person"].where(t.index > START, 0)).cumprod()
    t["deflator_index"] = nominal_per_worker / real_per_worker
    t["cpi_index"] = (1 + t["infl"].where(t.index > START, 0)).cumprod()
    if t[["infl", "g_person", "g_hour", "actual"]].loc[START + 1:].isna().any().any():
        raise ValueError("missing input values between START and END")
    return t


FLOORS = (None, "nominal", "components")


def counterfactual(t: pd.DataFrame, base: int, productivity: str = "person",
                   rule: str = "additive", start_level: float | None = None,
                   floor: str | None = None) -> pd.Series:
    """Wage path from ``base`` to END, indexed to inflation plus productivity growth.

    ``floor="nominal"``: the wage never falls below the previous year's (no nominal
    cuts; a year with inflation plus productivity growth below zero leaves it flat).
    ``floor="components"``: neither deflation nor a fall in productivity is passed
    on (inflation and productivity growth each floored at zero).
    """
    if floor not in FLOORS:
        raise ValueError(floor)
    g = t.loc[base + 1:END, PRODUCTIVITY[productivity]]
    pi = t.loc[base + 1:END, "infl"]
    if floor == "components":
        g = g.clip(lower=0)
        pi = pi.clip(lower=0)
    if rule == "additive":
        step = 1 + pi + g
    elif rule == "multiplicative":
        step = (1 + pi) * (1 + g)
    else:
        raise ValueError(rule)
    if floor == "nominal":
        step = step.clip(lower=1)
    level = t.loc[base, "actual"] if start_level is None else start_level
    path = pd.concat([pd.Series([1.0], index=[base]), step.cumprod()])
    return level * path


def ratio(t: pd.DataFrame, base: int, **kw) -> float:
    """Counterfactual over actual in END."""
    return counterfactual(t, base, **kw).loc[END] / t.loc[END, "actual"]


def decomposition(t: pd.DataFrame, base: int) -> dict[str, float]:
    """Multiplicative rule: CF/actual = (CPI / GDP deflator) x (AWS_base / AWS_END)."""
    cpi = t.loc[END, "cpi_index"] / t.loc[base, "cpi_index"]
    dfl = t.loc[END, "deflator_index"] / t.loc[base, "deflator_index"]
    return {
        "cf_over_actual": ratio(t, base, rule="multiplicative"),
        "cpi_over_deflator": cpi / dfl,
        "wage_share_ratio": t.loc[base, "adj_wage_share"] / t.loc[END, "adj_wage_share"],
    }


PTE_PER_EUR = 200.482


def load_minimum_wage() -> pd.Series:
    """National minimum wage, annual average of the daily rate in force, EUR per month.

    Built from DGERT's dates of change (``data/minimum_wage_changes.csv``) so that it
    matches annual-average consumer prices. PORDATA's one-value-per-year series
    (``data/minimum_wage.csv``) uses the rate at the end of the year for some years
    and at the start for others, and is kept only for comparison.
    """
    c = pd.read_csv(DATA / "minimum_wage_changes.csv", parse_dates=["effective"])
    c["eur"] = np.where(c["currency"] == "PTE", c["value"] / PTE_PER_EUR, c["value"])
    days = pd.date_range(c["effective"].min(), f"{c['effective'].dt.year.max()}-12-31", freq="D")
    rate = c.set_index("effective")["eur"].reindex(days).ffill()
    annual = rate.groupby(rate.index.year).mean()
    return annual.loc[annual.index > c["effective"].min().year]   # first year is partial


def comp_per_employee_hour() -> pd.Series:
    """Compensation per hour worked by employees, EUR, Eurostat (from LINK)."""
    e = load_eurostat().loc[LINK:END]
    return 1000 * e["D1"] / e["SAL_DC_THS_HW"]


def extensions(t: pd.DataFrame, euro: pd.DataFrame, qp: pd.DataFrame,
               sens: pd.DataFrame, w) -> None:
    """Minimum wage, per-hour pay and a summary that does not hinge on one base year."""
    qp_mean, qp_median = qp.loc[END, "qp_mean_eur"], qp.loc[END, "qp_median_eur"]

    # 1. National minimum wage, same rule from every base year.
    smn = load_minimum_wage()
    rows = []
    for b in range(START, END):
        rows.append({
            "base": b, "smn_base": smn[b],
            "cf": counterfactual(t, b, start_level=smn[b]).loc[END],
            "cf_no_cut": counterfactual(t, b, start_level=smn[b], floor="nominal").loc[END],
            "cf_hour": counterfactual(t, b, start_level=smn[b], productivity="hour").loc[END],
        })
    m = pd.DataFrame(rows).set_index("base")
    m.round(2).to_csv(RESULTS / "minimum_wage_by_base_year.csv")
    pordata = pd.read_csv(DATA / "minimum_wage.csv", index_col="year")["smn_eur_month"]
    w(f"National minimum wage (annual average of the daily rate): actual {END} {smn[END]:,.0f}, 2026 {smn[2026]:,.0f}")
    diff = (100 * (smn.loc[START:END] / pordata.loc[START:END] - 1)).round(1)
    w("  vs PORDATA year values, years differing by more than 0.1%: " + ", ".join(
        f"{y} {smn[y]:.2f} vs {pordata[y]:.1f} ({d:+.1f}%)" for y, d in diff.items() if abs(d) > 0.1))
    w(f"  base {START} ({smn[START]:.1f}): CF {m.loc[START, 'cf']:,.0f}  ratio {m.loc[START, 'cf'] / smn[END]:.3f}; "
      f"no cut {m.loc[START, 'cf_no_cut']:,.0f}; per hour {m.loc[START, 'cf_hour']:,.0f}")
    for col, label in (("cf", "rule"), ("cf_no_cut", "no cut"), ("cf_hour", "per hour")):
        up = m.index[m[col] > smn[END]]
        w(f"  {label}: base years above actual: {', '.join(str(b) for b in up)}")
    late = m.loc[m.index >= 1984, "cf"]
    w(f"  rule, base years 1984-{END - 1}: CF {late.min():,.0f} (base {late.idxmin()}) to "
      f"{late.max():,.0f} (base {late.idxmax()})")
    real = smn.loc[START:END] / t["cpi_index"]
    real = 100 * real / real.loc[START]
    w("  real minimum wage, " + str(START) + " = 100: " + ", ".join(
        f"{y} {real.loc[y]:.1f}" for y in (1977, 1980, 1985, 1990, 2000, 2010, 2014, 2016, 2019, 2020, END)))
    back = real.loc[START + 1:]
    w(f"  first year back at or above the {START} real level: {back[back >= 100].index.min()}; "
      f"low point {back.min():.1f} in {back.idxmin()}; first year at or above 90: {back[back >= 90].index.min()}")
    w(f"  minimum wage / QP mean: " + ", ".join(
        f"{y} {smn[y] / qp.loc[y, 'qp_mean_eur']:.3f}" for y in (1985, 1995, 2005, 2014, 2019, END)))
    w(f"  minimum wage {2016}-{END}: +{100 * (smn[END] / smn[2016] - 1):.1f}%")
    w("")

    # 2. Pay per hour against productivity per hour, Eurostat only.
    ch = comp_per_employee_hour()
    rows = []
    for b in range(LINK, END):
        rows.append({
            "base": b,
            "per_hour": counterfactual(t, b, productivity="hour", start_level=ch[b]).loc[END] / ch[END],
            "per_hour_no_cut": counterfactual(t, b, productivity="hour", start_level=ch[b],
                                              floor="nominal").loc[END] / ch[END],
        })
    h = pd.DataFrame(rows).set_index("base")
    h["per_person"] = sens.set_index("base")["cf_over_actual"].loc[LINK:END - 1]
    h["per_person_no_cut"] = sens.set_index("base")["cf_over_actual_no_cut"].loc[LINK:END - 1]
    h.round(4).to_csv(RESULTS / "per_hour_by_base_year.csv")
    w(f"Compensation per employee hour vs productivity per hour, base years {LINK}-{END - 1}")
    w(f"  compensation per employee hour: {LINK} {ch[LINK]:.2f}, {END} {ch[END]:.2f} EUR")
    for col in ("per_hour", "per_hour_no_cut", "per_person", "per_person_no_cut"):
        up = h.index[h[col] > 1]
        w(f"  {col:17s} median {h[col].median():.3f}, min {h[col].min():.3f}, max {h[col].max():.3f}; "
          f"above actual: {', '.join(str(b) for b in up) or 'none'}")
    hours = euro["SAL_DC_THS_HW"] / euro["SAL_DC_THS_PER"]
    w("  hours per employee: " + ", ".join(f"{y} {hours[y]:,.0f}" for y in (1995, 2010, 2019, 2020, 2021, END)))
    hstep = (1 + t["infl"] + t["g_hour"]).loc[LINK + 1:END]
    w("  per-hour years with inflation + productivity < 0: " + ", ".join(
        f"{y} ({100 * (v - 1):+.2f}%)" for y, v in hstep[hstep < 1].items()))
    w("")

    # 3. Summaries that do not hinge on one base year.
    w("Across base years and against a reference-period wage share")
    for col, label in (("cf_over_actual", "rule"), ("cf_over_actual_no_cut", "no cut")):
        qs = sens[col].quantile([0.25, 0.5, 0.75])
        w(f"  {label}: median over all {len(sens)} base years {qs[0.5]:.3f} (QP mean {qp_mean * qs[0.5]:,.0f}), "
          f"interquartile range {qs[0.25]:.3f}-{qs[0.75]:.3f} ({qp_mean * qs[0.25]:,.0f}-{qp_mean * qs[0.75]:,.0f})")
    aws = t["adj_wage_share"]
    for lo in (START, 1985, LINK):
        ref = aws.loc[lo:END].mean()
        r = ref / aws[END]
        w(f"  adjusted wage share {lo}-{END} average {ref:.2f}% vs {END} {aws[END]:.2f}%: ratio {r:.3f}; "
          f"QP mean {qp_mean * r:,.0f}, median {qp_median * r:,.0f}")
    w("")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t = build_inputs()
    euro, qp, ameco = load_eurostat(), load_qp(), load_ameco()
    qp_mean, qp_median = qp.loc[END, "qp_mean_eur"], qp.loc[END, "qp_median_eur"]
    a_end = t.loc[END, "actual"]
    lines: list[str] = []
    w = lines.append

    w(f"Portugal: base-year wage indexed to CPI inflation plus productivity, {START}-{END}")
    w("")
    w("Actual levels")
    w(f"  compensation per employee {END}: {a_end:,.0f} EUR/yr ({a_end / 14:,.0f}/month over 14)")
    wages = 1000 * euro.loc[END, "wages_per_emp"]
    w(f"  wages and salaries per employee {END}: {wages:,.0f} EUR/yr ({wages / 14:,.0f}/month over 14)")
    w(f"  D11/D1: {LINK} {euro.loc[LINK, 'd11_meur'] / euro.loc[LINK, 'D1']:.4f}, "
      f"{END} {euro.loc[END, 'd11_meur'] / euro.loc[END, 'D1']:.4f}")
    w(f"  Quadros de Pessoal {END}: mean {qp_mean:,.1f}, median {qp_median:,.2f}, "
      f"median/mean {qp_median / qp_mean:.4f}")
    w(f"  compensation per employee {START} (linked): {t.loc[START, 'actual']:,.1f} EUR/yr "
      f"= {t.loc[START, 'actual'] * 200.482:,.0f} escudos/yr")
    w("")

    w(f"Base {START}, counterfactual in {END}")
    for prod in PRODUCTIVITY:
        for rule in ("additive", "multiplicative"):
            r = ratio(t, START, productivity=prod, rule=rule)
            w(f"  productivity per {prod:6s} {rule:14s} CF/actual {r:.3f}  "
              f"compensation {a_end * r:>8,.0f}  QP mean {qp_mean * r:>6,.0f}  "
              f"QP median {qp_median * r:>6,.0f}")
    w("")

    w(f"Cumulative factors {START}-{END}")
    prod_f = (1 + t.loc[START + 1:END, "g_person"]).prod()
    cpi_f = t.loc[END, "cpi_index"] / t.loc[START, "cpi_index"]
    act_f = a_end / t.loc[START, "actual"]
    w(f"  real productivity per person employed x{prod_f:.3f}")
    w(f"  CPI x{cpi_f:.1f}; nominal compensation per employee x{act_f:.1f}; "
      f"real compensation per employee x{act_f / cpi_f:.3f}")
    real = t["actual"] / t["cpi_index"]
    real = 100 * real / real.loc[START]
    w("  real compensation per employee, " + str(START) + " = 100: " + ", ".join(
        f"{y} {real.loc[y]:.1f}" for y in (1977, 1980, 1985, 1990, 1995, 2000, 2010, 2015, END)))
    gdp75 = ameco.loc[START, "gdp_real"] * ameco.loc[START, "gdp_deflator"] / 100
    w(f"  wage share {START}: adjusted {t.loc[START, 'adj_wage_share']:.1f}%, unadjusted "
      f"{100 * ameco.loc[START, 'comp_total'] / gdp75:.1f}%; self-employed "
      f"{100 * (1 - ameco.loc[START, 'employees'] / ameco.loc[START, 'employment']):.1f}% of employment")
    w(f"  wage share {END}: adjusted {t.loc[END, 'adj_wage_share']:.1f}%, unadjusted "
      f"{euro.loc[END, 'wage_share']:.1f}%")
    w("")

    w("Decomposition (multiplicative rule): CF/actual = CPI/deflator x wage-share ratio")
    for base in (1975, 1980, 1985, 1995, 2000, 2010):
        d = decomposition(t, base)
        w(f"  base {base}: {d['cf_over_actual']:.3f} = {d['cpi_over_deflator']:.3f} x "
          f"{d['wage_share_ratio']:.3f}  (adjusted wage share "
          f"{t.loc[base, 'adj_wage_share']:.1f}% -> {t.loc[END, 'adj_wage_share']:.1f}%)")
    w("")

    w(f"Quadros de Pessoal mean only (salary series, no splicing), base 1985")
    for prod in PRODUCTIVITY:
        cf = counterfactual(t, 1985, productivity=prod, start_level=qp.loc[1985, "qp_mean_eur"])
        w(f"  productivity per {prod:6s} CF {cf.loc[END]:,.0f}  actual {qp_mean:,.1f}  "
          f"CF/actual {cf.loc[END] / qp_mean:.3f}")
    w(f"Quadros de Pessoal median only, base 2021")
    cf = counterfactual(t, 2021, start_level=qp.loc[2021, "qp_median_eur"])
    w(f"  productivity per person  CF {cf.loc[END]:,.0f}  actual {qp_median:,.2f}  "
      f"CF/actual {cf.loc[END] / qp_median:.3f}")
    w("")

    w("CPI check, World Bank vs AMECO national CPI")
    zc = ameco["cpi"]
    for y in (1976, 1977, 1980, 1984):
        w(f"  {y}: World Bank {100 * t.loc[y, 'infl']:.1f}%, AMECO {100 * (zc[y] / zc[y - 1] - 1):.1f}%")
    alt = t.copy()
    alt.loc[START + 1:LINK, "infl"] = growth(zc).loc[START + 1:LINK]
    w(f"  base {START} CF/actual with AMECO CPI to {LINK}: {ratio(alt, START):.3f}")
    w("")

    rows = []
    for base in range(START, END):
        r = ratio(t, base)
        rows.append({"base": base, "cf_over_actual": r, "qp_mean_cf": qp_mean * r,
                     "qp_median_cf": qp_median * r,
                     "cf_over_actual_mult": ratio(t, base, rule="multiplicative"),
                     "cf_over_actual_hour": ratio(t, base, productivity="hour"),
                     "cf_over_actual_no_cut": ratio(t, base, floor="nominal"),
                     "cf_over_actual_no_fall_components": ratio(t, base, floor="components"),
                     "cf_over_actual_no_cut_hour": ratio(t, base, productivity="hour",
                                                         floor="nominal")})
    sens = pd.DataFrame(rows)
    sens.round(4).to_csv(RESULTS / "base_year_sensitivity.csv", index=False)
    w("Every base year (productivity per person, additive): CF/actual, QP mean and median in " + str(END))
    for r in rows:
        w(f"  {r['base']}  {r['cf_over_actual']:.3f}  {r['qp_mean_cf']:>6,.0f}  {r['qp_median_cf']:>6,.0f}")
    above = sens.loc[sens["cf_over_actual"] > 1, "base"]
    w(f"  base years with counterfactual above actual: {', '.join(str(b) for b in above)}")
    w(f"  from {above.max() + 1} on: CF/actual between {sens.loc[sens['base'] > above.max(), 'cf_over_actual'].min():.3f} "
      f"and {sens.loc[sens['base'] > above.max(), 'cf_over_actual'].max():.3f}, median "
      f"{sens.loc[sens['base'] > above.max(), 'cf_over_actual'].median():.3f}")
    for col, label in (("cf_over_actual_mult", "multiplicative"), ("cf_over_actual_hour", "per hour")):
        w(f"  {label}: base years above actual: "
          f"{', '.join(str(b) for b in sens.loc[sens[col] > 1, 'base'])}")
    w("")

    # Scenario: wages cannot fall from one year to the next.
    step = 1 + t["infl"] + t["g_person"]
    falls = step.loc[START + 1:END][step.loc[START + 1:END] < 1]
    w("Scenario: no year-on-year fall (additive rule, productivity per person)")
    w("  years where inflation + productivity growth < 0: " + ", ".join(
        f"{y} ({100 * (v - 1):+.2f}%)" for y, v in falls.items()))
    gfall = t["g_person"].loc[START + 1:END][t["g_person"].loc[START + 1:END] < 0]
    w("  years where productivity growth < 0: " + ", ".join(
        f"{y} ({100 * v:+.2f}%)" for y, v in gfall.items()))
    w(f"  ratchet from no nominal cut, base {START}: {100 * np.log(1 / falls).sum():.1f} log points; "
      + ", ".join(f"{y} {100 * np.log(1 / v):.2f}" for y, v in falls.items()))
    w(f"  2020 productivity growth: per person {100 * t.loc[2020, 'g_person']:+.2f}%, "
      f"per hour {100 * t.loc[2020, 'g_hour']:+.2f}%; 2021: per person "
      f"{100 * t.loc[2021, 'g_person']:+.2f}%, per hour {100 * t.loc[2021, 'g_hour']:+.2f}%")
    for floor, label in (("nominal", "no nominal cut"), ("components", "no deflation or productivity fall")):
        r = ratio(t, START, floor=floor)
        w(f"  {label:30s} base {START}: CF/actual {r:.3f}  compensation {a_end * r:>8,.0f}  "
          f"QP mean {qp_mean * r:>6,.0f}  QP median {qp_median * r:>6,.0f}")
        rm = ratio(t, START, floor=floor, rule="multiplicative")
        w(f"  {label:30s} base {START}, multiplicative: CF/actual {rm:.3f}  QP mean {qp_mean * rm:>6,.0f}")
    for col, label in (("cf_over_actual_no_cut", "no nominal cut"),
                       ("cf_over_actual_no_fall_components", "no deflation or productivity fall"),
                       ("cf_over_actual_no_cut_hour", "no nominal cut, per hour")):
        up = sens.loc[sens[col] > 1, "base"]
        w(f"  {label}: base years above actual ({len(up)} of {len(sens)}): "
          f"{', '.join(str(b) for b in up)}")
        late = sens.loc[sens["base"] >= 1983, col]
        w(f"    base years 1983-{END - 1}: CF/actual {late.min():.3f} to {late.max():.3f}, median {late.median():.3f}; "
          f"QP mean {qp_mean * late.min():,.0f} to {qp_mean * late.max():,.0f}")
    for b in (1985, 1995, 2000, 2008, 2010, 2013, 2015, 2019, 2021):
        r0 = sens.loc[sens["base"] == b].iloc[0]
        w(f"  base {b}: rule {r0['cf_over_actual']:.3f} ({qp_mean * r0['cf_over_actual']:,.0f}), "
          f"no cut {r0['cf_over_actual_no_cut']:.3f} ({qp_mean * r0['cf_over_actual_no_cut']:,.0f}), "
          f"components floored {r0['cf_over_actual_no_fall_components']:.3f} "
          f"({qp_mean * r0['cf_over_actual_no_fall_components']:,.0f})")
    gap = (sens["cf_over_actual_no_fall_components"] / sens["cf_over_actual_no_cut"] - 1).abs().max()
    w(f"  largest difference between the two floors over all base years: {100 * gap:.3f}%")
    w("  years with deflation: " + ", ".join(
        f"{y} ({100 * v:+.2f}%)" for y, v in t["infl"].loc[START + 1:END].items() if v < 0))
    cf = counterfactual(t, 1985, start_level=qp.loc[1985, "qp_mean_eur"], floor="nominal")
    w(f"  Quadros de Pessoal mean only, base 1985, no cut: CF {cf.loc[END]:,.0f}  "
      f"CF/actual {cf.loc[END] / qp_mean:.3f}")
    # Floor every year except 2020: how much of the scenario is that one year?
    no_2020 = []
    for b in range(START, END):
        st = (1 + t["infl"] + t["g_person"]).loc[b + 1:END]
        st = st.where((st >= 1) | (st.index == 2020), 1.0)
        no_2020.append(t.loc[b, "actual"] * st.prod() / a_end)
    no_2020 = pd.Series(no_2020, index=range(START, END))
    w(f"  no cut except in 2020: base years above actual: "
      f"{', '.join(str(b) for b in no_2020[no_2020 > 1].index)}")
    w("")

    dec = pd.DataFrame([{"base": b, **decomposition(t, b)} for b in range(START, END)])
    dec["log_points_total"] = 100 * np.log(dec["cf_over_actual"])
    dec["log_points_price_wedge"] = 100 * np.log(dec["cpi_over_deflator"])
    dec["log_points_wage_share"] = 100 * np.log(dec["wage_share_ratio"])
    dec.round(4).to_csv(RESULTS / "decomposition_by_base_year.csv", index=False)
    w("Decomposition by base year, log points (multiplicative rule): total = price wedge + wage share")
    for b in (1975, 1977, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020):
        r = dec.loc[dec["base"] == b].iloc[0]
        w(f"  {b}: {r['log_points_total']:+6.1f} = {r['log_points_price_wedge']:+6.1f} "
          f"{r['log_points_wage_share']:+6.1f}")
    w("")

    extensions(t, euro, qp, sens, w)

    path = pd.DataFrame({
        "actual_eur_month": t["actual"] / 14,
        "counterfactual_eur_month": counterfactual(t, START) / 14,
        "counterfactual_no_cut_eur_month": counterfactual(t, START, floor="nominal") / 14,
        "adj_wage_share": t["adj_wage_share"],
    })
    path.index.name = "year"
    path.round(3).to_csv(RESULTS / "path_from_1975.csv")
    t.round(8).to_csv(RESULTS / "inputs_linked.csv")
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
