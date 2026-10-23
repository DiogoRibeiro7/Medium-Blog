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
    e = gdp.join(emp[["EMP_DC_THS_PER", "SAL_DC_THS_PER"]]).join(extra)
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


def counterfactual(t: pd.DataFrame, base: int, productivity: str = "person",
                   rule: str = "additive", start_level: float | None = None) -> pd.Series:
    """Wage path from ``base`` to END, indexed to inflation plus productivity growth."""
    g = t.loc[base + 1:END, PRODUCTIVITY[productivity]]
    pi = t.loc[base + 1:END, "infl"]
    if rule == "additive":
        step = 1 + pi + g
    elif rule == "multiplicative":
        step = (1 + pi) * (1 + g)
    else:
        raise ValueError(rule)
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
                     "cf_over_actual_hour": ratio(t, base, productivity="hour")})
    sens = pd.DataFrame(rows)
    sens.round(4).to_csv(RESULTS / "base_year_sensitivity.csv", index=False)
    w("Every base year (productivity per person, additive): CF/actual, QP mean and median in " + str(END))
    for r in rows:
        w(f"  {r['base']}  {r['cf_over_actual']:.3f}  {r['qp_mean_cf']:>6,.0f}  {r['qp_median_cf']:>6,.0f}")
    first = sens.loc[sens["cf_over_actual"] < 1, "base"].min()
    w(f"  first base year with counterfactual below actual: {first}")

    path = pd.DataFrame({
        "actual_eur_month": t["actual"] / 14,
        "counterfactual_eur_month": counterfactual(t, START) / 14,
        "adj_wage_share": t["adj_wage_share"],
    })
    path.index.name = "year"
    path.round(3).to_csv(RESULTS / "path_from_1975.csv")
    t.round(8).to_csv(RESULTS / "inputs_linked.csv")
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
