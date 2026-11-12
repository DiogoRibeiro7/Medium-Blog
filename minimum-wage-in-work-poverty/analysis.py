"""Minimum wages, the middle of the wage distribution and in-work poverty in Portugal and the South.

    python analysis.py

Reads the snapshot in ``data/`` and writes ``results/*.csv`` and every number used in the
article to ``results/analysis_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"

NAMES = {"PT": "Portugal", "ES": "Spain", "EL": "Greece", "IT": "Italy", "FR": "France", "DE": "Germany",
         "BE": "Belgium", "NL": "Netherlands", "IE": "Ireland", "LU": "Luxembourg", "SI": "Slovenia",
         "MT": "Malta", "CY": "Cyprus", "SK": "Slovakia", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
         "HR": "Croatia", "EU27_2020": "EU27"}
# Minimum wage paid 14 times a year; Eurostat reports a 12-month equivalent (x 14/12)
FOURTEEN = {"PT", "ES", "EL"}
# Euro area members with a statutory minimum wage over the whole 2006-2022 SES window
EURO_2006 = ["PT", "ES", "EL", "FR", "BE", "NL", "IE", "LU", "SI"]
# Euro area members with a statutory minimum wage in 2024 (TaxBEN incomes and thresholds both in euro)
EURO_2024 = ["PT", "ES", "EL", "FR", "DE", "BE", "NL", "IE", "LU", "SI", "MT", "CY", "SK", "EE", "LV", "LT", "HR"]
SOUTH = ["PT", "ES", "EL", "IT"]
ISO3 = {"PRT": "PT", "ESP": "ES", "GRC": "EL", "ITA": "IT", "FRA": "FR", "DEU": "DE", "BEL": "BE", "NLD": "NL",
        "IRL": "IE", "LUX": "LU", "SVN": "SI", "MLT": "MT", "CYP": "CY", "SVK": "SK", "EST": "EE", "LVA": "LV",
        "LTU": "LT", "HRV": "HR"}
HOUSEHOLDS = {("S_C0", "_Z"): "single", ("S_C2", "_Z"): "single_parent",
              ("C_C0", "NOEARN_UNEMP_WO_CONBEN"): "couple_one_earner",
              ("C_C2", "NOEARN_UNEMP_WO_CONBEN"): "couple_2ch_one_earner",
              ("C_C2", "MINW"): "couple_2ch_two_earners"}
HH_LABEL = {"single": "single adult", "single_parent": "single parent, two children",
            "couple_one_earner": "couple, one earner, no children",
            "couple_2ch_one_earner": "couple, one earner, two children",
            "couple_2ch_two_earners": "couple, two earners, two children"}

LINES: list[str] = []


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", low_memory=False)


# ------------------------------------------------------------------------------ inputs
MW = read("minimum_wage")
MW["year"] = MW.time.str[:4].astype(int)
MW["half"] = MW.time.str[-2:]
MW_EUR = MW[MW.currency == "EUR"]
MW_JUL = MW_EUR[MW_EUR.half == "S2"].pivot_table(index="year", columns="geo", values="value")       # from 1 July
MW_YEAR = MW_EUR.pivot_table(index="year", columns="geo", values="value", aggfunc="mean")           # mean of S1, S2
HICP = read("hicp").pivot_table(index="time", columns="geo", values="value")
FTE = read("fte_salary")
FTE = FTE[FTE.unit == "EUR"].pivot_table(index="time", columns="geo", values="value")

SES = pd.concat([read(f"ses_{y}").rename(columns={"quant_inc": "quantile"}) for y in (2006, 2010, 2014, 2018, 2022)])
SES = SES[(SES.unit == "EUR") & (SES.worktime == "FT")].pivot_table(index=["geo", "time"], columns="quantile",
                                                                       values="value")

THR = read("poverty_thresholds")
THR = THR[(THR.statinfo == "MED_EI") & (THR.rskpovth == "B_60") & (THR.unit == "EUR")].pivot_table(
    index=["geo", "time"], columns="hhcomp", values="value")

IMW = read("oecd_imw")
IMW = IMW[(IMW.MEASURE == "NFI") & (IMW.SOC_ASS_BENEFIT == "YES") & (IMW.HOUSE_BENEFIT == "NO")]
IMW = IMW.assign(geo=IMW.REF_AREA.map(ISO3),
                 hh=[HOUSEHOLDS.get((a, b)) for a, b in zip(IMW.HOUSEHOLD_TYPE, IMW.INCOME_PART)]).dropna(
    subset=["geo", "hh"])
NFI = IMW.pivot_table(index=["geo", "TIME_PERIOD"], columns="hh", values="OBS_VALUE")
GFE = read("oecd_imw")
GFE = GFE[(GFE.MEASURE == "GFE") & (GFE.HOUSEHOLD_TYPE == "S_C0") & (GFE.SOC_ASS_BENEFIT == "YES")]
GFE = GFE.assign(geo=GFE.REF_AREA.map(ISO3)).dropna(subset=["geo"]).pivot_table(
    index="TIME_PERIOD", columns="geo", values="OBS_VALUE")

HOURS = read("oecd_hours_poverty")
HOURS = HOURS[(HOURS.INCOME_CURR == "MINW") & (HOURS.HOUSE_BENEFIT == "NO") & (HOURS.TEMP_INTOWORK_BENEFIT == "NO")]
HOURS = HOURS.assign(geo=HOURS.REF_AREA.map(ISO3)).dropna(subset=["geo"]).pivot_table(
    index=["geo", "HOUSEHOLD_TYPE"], columns="TIME_PERIOD", values="OBS_VALUE")


def iw(name: str, col: str, extra: dict | None = None) -> pd.DataFrame:
    d = read(name)
    for k, v in (extra or {}).items():
        d = d[d[k] == v]
    return d.pivot_table(index=["geo", col], columns="time", values="value")


IW_STATUS = iw("iw_status", "wstatus")
IW_HH = iw("iw_household", "hhcomp")
IW_CONTRACT = iw("iw_contract", "wstatus")
IW_TIME = iw("iw_worktime", "worktime")
IW_INT = iw("iw_intensity", "workint", {"hhcomp": "TOTAL"})
AROP = read("arop").pivot_table(index="time", columns="geo", values="value")


def poverty_ratio(geo: str, year: int) -> dict | None:
    """Net family income at the minimum wage in income year `year` divided by the 60% threshold of the
    EU-SILC survey of `year + 1` (whose income reference period is `year`)."""
    if (geo, year) not in NFI.index or (geo, year + 1) not in THR.index:
        return None
    n, t = NFI.loc[(geo, year)], THR.loc[(geo, year + 1)]
    lines = {"single": t["A1"], "single_parent": 1.6 * t["A1"], "couple_one_earner": t["A2"],
             "couple_2ch_one_earner": t["A2_CH2_LT14"], "couple_2ch_two_earners": t["A2_CH2_LT14"]}
    out = {"geo": geo, "year": year}
    for hh, line in lines.items():
        out[hh] = n.get(hh, np.nan) / line
        out[f"{hh}_net"] = n.get(hh, np.nan)
        out[f"{hh}_line"] = line
    return out


def main() -> None:
    RES.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ checks
    section("Checks")
    r21 = (THR["A2_CH2_LT14"] / THR["A1"]).dropna()
    r15 = (THR["A2"] / THR["A1"]).dropna()
    say("Poverty thresholds: two adults with two children / one adult (min; max)", f"{r21.min():.3f}; {r21.max():.3f}")
    say("Poverty thresholds: two adults / one adult (min; max)", f"{r15.min():.3f}; {r15.max():.3f}")
    chk = []
    for g in ["PT", "ES", "EL", "FR", "BE", "NL"]:
        for y in range(2010, 2025):
            if y in GFE.index and g in GFE.columns and y in MW_YEAR.index and not np.isnan(GFE.loc[y, g]):
                chk.append(GFE.loc[y, g] / (12 * MW_YEAR.loc[y, g]))
    chk = np.array(chk)
    say("OECD gross annual minimum-wage earnings / (12 x Eurostat monthly minimum, mean of the two halves), "
        "PT ES EL FR BE NL 2010-2024 (median; min; max)", f"{np.median(chk):.3f}; {chk.min():.3f}; {chk.max():.3f}")

    # ------------------------------------------------------------------ the floor and the middle, SES
    section("The floor and the middle in real terms, SES 2006-2022 (full-time monthly earnings, October)")
    rows = []
    for g in EURO_2006:
        m06, m22 = MW_JUL.loc[2006, g], MW_JUL.loc[2022, g]
        adj = 12 / 14 if g in FOURTEEN else 1.0
        defl = np.log(HICP.loc[2022, g] / HICP.loc[2006, g])
        s06, s22 = SES.loc[(g, 2006)], SES.loc[(g, 2022)]
        rows.append({"geo": g, "name": NAMES[g],
                     "floor": np.log(m22 / m06) - defl,
                     "p10": np.log(s22["P10"] / s06["P10"]) - defl,
                     "median": np.log(s22["MED"] / s06["MED"]) - defl,
                     "p90": np.log(s22["P90"] / s06["P90"]) - defl,
                     "prices": defl,
                     "kaitz_2006": adj * m06 / s06["MED"], "kaitz_2022": adj * m22 / s22["MED"],
                     "median_to_floor_2006": s06["MED"] / (adj * m06), "median_to_floor_2022": s22["MED"] / (adj * m22)})
    real = pd.DataFrame(rows).sort_values("median")
    real.to_csv(RES / "real_growth_ses_2006_2022.csv", index=False, float_format="%.4f")
    for _, r in real.iterrows():
        say(f"{r.geo} 2006-2022 real log change (x100): floor; P10; median; P90; HICP log change",
            f"{100 * r.floor:.1f}; {100 * r.p10:.1f}; {100 * r['median']:.1f}; {100 * r.p90:.1f}; {100 * r.prices:.1f}")
        say(f"{r.geo} minimum as paid / SES median: 2006; 2022 | median / minimum: 2006; 2022",
            f"{r.kaitz_2006:.3f}; {r.kaitz_2022:.3f} | {r.median_to_floor_2006:.2f}; {r.median_to_floor_2022:.2f}")
    med18 = {g: np.log(SES.loc[(g, 2018), "MED"] / SES.loc[(g, 2006), "MED"])
             - np.log(HICP.loc[2018, g] / HICP.loc[2006, g]) for g in EURO_2006}
    say("Real median log change 2006-2018, same survey coverage (x100): " + "; ".join(EURO_2006),
        "; ".join(f"{round(100 * med18[g], 1) + 0.0:.1f}" for g in EURO_2006))
    say("Countries (of 9) where the real floor rose more than the real median, 2006-2022",
        f"{int((real.floor > real['median']).sum())}")
    say("Real median log change 2006-2022, median across the 9 countries (x100)", f"{100 * real['median'].median():.1f}")
    pt = real.set_index("geo").loc["PT"]
    say("Portugal 2006-2022: real floor and real median as % changes",
        f"{100 * (np.exp(pt.floor) - 1):.1f}%; {100 * (np.exp(pt['median']) - 1):.1f}%")
    say("Portugal SES full-time monthly earnings, EUR: median 2006; 2022 | P10 2006; 2022 | minimum as paid 2006; 2022",
        f"{SES.loc[('PT', 2006), 'MED']:.0f}; {SES.loc[('PT', 2022), 'MED']:.0f} | {SES.loc[('PT', 2006), 'P10']:.0f}; "
        f"{SES.loc[('PT', 2022), 'P10']:.0f} | {MW_JUL.loc[2006, 'PT'] * 12 / 14:.0f}; {MW_JUL.loc[2022, 'PT'] * 12 / 14:.0f}")
    waves = []
    for y in (2006, 2010, 2014, 2018, 2022):
        s = SES.loc[("PT", y)]
        d = HICP.loc[y, "PT"] / HICP.loc[2006, "PT"]
        waves.append(f"{y} {s['MED'] / d:.0f}")
    say("Portugal SES median at 2006 prices, EUR, by wave", "; ".join(waves))

    # ------------------------------------------------------------------ Portugal to 2025, national accounts
    section("Portugal to 2025: minimum wage and average full-time salary in real terms (national accounts)")
    idx = []
    for y in range(2006, 2026):
        p = HICP.loc[y, "PT"] / HICP.loc[2006, "PT"]
        idx.append({"year": y,
                    "minimum_real": MW_YEAR.loc[y, "PT"] / MW_YEAR.loc[2006, "PT"] / p * 100,
                    "fte_salary_real": FTE.loc[y, "PT"] / FTE.loc[2006, "PT"] / p * 100,
                    "minimum_to_fte_salary": 12 * MW_YEAR.loc[y, "PT"] / FTE.loc[y, "PT"]})
    idx = pd.DataFrame(idx).set_index("year")
    # poverty line by income year: the survey of year+1 refers to the income of year
    line = pd.Series({y: THR.loc[("PT", y + 1), "A1"] for y in range(2005, 2025) if ("PT", y + 1) in THR.index})
    idx["poverty_line_real"] = [line.get(y, np.nan) / line.get(2006, np.nan) / (HICP.loc[y, "PT"] / HICP.loc[2006, "PT"]) * 100
                                for y in idx.index]
    idx.to_csv(RES / "portugal_real_indices.csv", float_format="%.4f")
    for y in (2014, 2022, 2025):
        say(f"Portugal real index {y} (2006=100): minimum wage; average full-time salary",
            f"{idx.loc[y, 'minimum_real']:.1f}; {idx.loc[y, 'fte_salary_real']:.1f}")
    say("Portugal real index 2024 (2006=100): minimum wage; average full-time salary; poverty line for one adult (income year)",
        f"{idx.loc[2024, 'minimum_real']:.1f}; {idx.loc[2024, 'fte_salary_real']:.1f}; {idx.loc[2024, 'poverty_line_real']:.1f}")
    say("Portugal annual minimum wage / average full-time salary: 2006; 2014; 2022; 2025",
        "; ".join(f"{idx.loc[y, 'minimum_to_fte_salary']:.3f}" for y in (2006, 2014, 2022, 2025)))
    say("Portugal average full-time salary, EUR: 2006; 2014; 2022; 2025",
        "; ".join(f"{FTE.loc[y, 'PT']:,.0f}" for y in (2006, 2014, 2022, 2025)))
    say("Portugal monthly minimum wage as paid (14 payments), EUR: 2006; 2014; 2022; 2025; 2026",
        "; ".join(f"{MW_EUR[(MW_EUR.geo == 'PT') & (MW_EUR.time == f'{y}-S1')].value.iloc[0] * 12 / 14:.0f}"
                  for y in (2006, 2014, 2022, 2025, 2026)))
    g14 = np.log(MW_YEAR.loc[2024, "PT"] / MW_YEAR.loc[2014, "PT"])
    gl = np.log(line[2024] / line[2014])
    gn = np.log(NFI.loc[("PT", 2024), "single"] / NFI.loc[("PT", 2014), "single"])
    say("Portugal 2014-2024 log changes (x100), nominal: minimum wage; net income of a single minimum-wage earner; "
        "poverty line for one adult", f"{100 * g14:.1f}; {100 * gn:.1f}; {100 * gl:.1f}")

    # ------------------------------------------------------------------ in-work poverty
    section("In-work at-risk-of-poverty rate, employed persons aged 18-64 (%)")
    emp = IW_STATUS.xs("EMP", level="wstatus")
    for g in ["PT", "ES", "EL", "IT", "FR", "DE", "EU27_2020"]:
        vals = "; ".join(f"{y} {emp.loc[g, y]:.1f}" for y in (2005, 2010, 2015, 2019, 2022, 2025)
                         if y in emp.columns and not np.isnan(emp.loc[g, y]))
        say(f"{g} by year", vals)
    pts = emp.loc["PT", 2006:2023]
    say("Portugal 2006-2023: min; max; mean", f"{pts.min():.1f}; {pts.max():.1f}; {pts.mean():.1f}")
    say("Portugal: years 2005-2025 with the rate below 9.5%", ", ".join(str(y) for y in emp.columns
                                                                   if y >= 2005 and emp.loc['PT', y] < 9.5))
    sp = IW_HH.xs("A1_DCH", level="hhcomp")
    say("In-work poverty, single adults with children, Portugal by year",
        "; ".join(f"{y} {sp.loc['PT', y]:.1f}" for y in (2010, 2015, 2019, 2022, 2025)))
    say("Whole-population at-risk-of-poverty rate, Portugal: 2015; 2019; 2025",
        f"{AROP.loc[2015, 'PT']:.1f}; {AROP.loc[2019, 'PT']:.1f}; {AROP.loc[2025, 'PT']:.1f}")
    for code, lab in [("SAL", "employees"), ("NSAL", "self-employed")]:
        s = IW_STATUS.xs(code, level="wstatus")
        say(f"In-work poverty, {lab}: PT 2015; PT 2025; EU27 2025",
            f"{s.loc['PT', 2015]:.1f}; {s.loc['PT', 2025]:.1f}; {s.loc['EU27_2020', 2025]:.1f}")
    groups = [
        ("employees", IW_STATUS, "SAL"), ("self-employed", IW_STATUS, "NSAL"),
        ("permanent contract", IW_CONTRACT, "SAL_PERM"), ("temporary contract", IW_CONTRACT, "SAL_TEMP"),
        ("full-time", IW_TIME, "FT"), ("part-time", IW_TIME, "PT"),
        ("single adult", IW_HH, "A1"), ("single adult with children", IW_HH, "A1_DCH"),
        ("two or more adults, no children", IW_HH, "A_GE2_NDCH"), ("two or more adults with children", IW_HH, "A_GE2_DCH"),
        ("very high work intensity", IW_INT, "WI085-1"), ("high work intensity", IW_INT, "WI055-085"),
        ("medium work intensity", IW_INT, "WI045-055"), ("low work intensity", IW_INT, "WI02-045"),
    ]
    grp = []
    for lab, tab, code in groups:
        t = tab.xs(code, level=1)
        row = {"group": lab}
        for g in ["PT", "ES", "EL", "IT", "EU27_2020"]:
            row[g] = t.loc[g, 2025] if g in t.index and 2025 in t.columns else np.nan
        grp.append(row)
        say(f"In-work poverty 2025, {lab}: PT; ES; EL; IT; EU27",
            "; ".join("n/a" if np.isnan(row[g]) else f"{row[g]:.1f}" for g in ["PT", "ES", "EL", "IT", "EU27_2020"]))
    pd.DataFrame(grp).to_csv(RES / "in_work_poverty_groups_2025.csv", index=False, float_format="%.1f")
    emp.loc[["PT", "ES", "EL", "IT", "FR", "DE", "EU27_2020"]].T.to_csv(RES / "in_work_poverty_rates.csv", float_format="%.1f")

    # ------------------------------------------------------------------ net income at the minimum vs the poverty line
    section("Net family income at a full-time minimum wage / 60% at-risk-of-poverty threshold (TaxBEN, EU-SILC)")
    rows = [r for g in EURO_2024 for y in range(2001, 2025) if (r := poverty_ratio(g, y))]
    pr = pd.DataFrame(rows)
    pr.to_csv(RES / "minimum_wage_vs_poverty_line.csv", index=False, float_format="%.4f")
    ptr = pr[pr.geo == "PT"].set_index("year")
    hh_order = ["single", "single_parent", "couple_one_earner", "couple_2ch_one_earner", "couple_2ch_two_earners"]
    for y in (2005, 2010, 2014, 2019, 2022, 2024):
        say(f"Portugal income year {y}: " + "; ".join(HH_LABEL[h] for h in hh_order),
            "; ".join("n/a" if np.isnan(ptr.loc[y, h]) else f"{ptr.loc[y, h]:.2f}" for h in hh_order))
    say("Portugal income year 2024, net income EUR: " + "; ".join(HH_LABEL[h] for h in hh_order),
        "; ".join(f"{ptr.loc[2024, h + '_net']:,.0f}" for h in hh_order))
    say("Portugal income year 2024, poverty line EUR: " + "; ".join(HH_LABEL[h] for h in hh_order),
        "; ".join(f"{ptr.loc[2024, h + '_line']:,.0f}" for h in hh_order))
    gap = ptr.loc[2024, "couple_2ch_one_earner_line"] - ptr.loc[2024, "couple_2ch_one_earner_net"]
    say("Portugal 2024: shortfall of a one-earner couple with two children below the line, EUR per year; per month (12)",
        f"{gap:,.0f}; {gap / 12:,.0f}")
    gap_sp = ptr.loc[2024, "single_parent_line"] - ptr.loc[2024, "single_parent_net"]
    say("Portugal 2024: shortfall of a single parent with two children, EUR per year; per month (12)",
        f"{gap_sp:,.0f}; {gap_sp / 12:,.0f}")
    gross = GFE.loc[2024, "PT"]
    say("Portugal 2024: gross annual minimum wage (TaxBEN), EUR; shortfalls as % of it: one-earner couple with two "
        "children; single parent", f"{gross:,.0f}; {100 * gap / gross:.1f}%; {100 * gap_sp / gross:.1f}%")
    for g in ["ES", "EL", "FR", "DE"]:
        s = pr[pr.geo == g].set_index("year")
        yrs = [y for y in (2010, 2014, 2019, 2024) if y in s.index]
        say(f"{g} by income year, single; single parent; one-earner couple with two children",
            " | ".join(f"{y}: {s.loc[y, 'single']:.2f}; {s.loc[y, 'single_parent']:.2f}; {s.loc[y, 'couple_2ch_one_earner']:.2f}"
                       for y in yrs))
    x24 = pr[pr.year == 2024].set_index("geo")
    x24.to_csv(RES / "minimum_wage_vs_poverty_line_2024.csv", float_format="%.4f")
    n = len(x24)
    for h in hh_order:
        below = x24[x24[h] < 1].index.tolist()
        say(f"2024, {n} euro-area countries with a minimum wage: countries below the line, {HH_LABEL[h]}",
            f"{len(below)} ({', '.join(sorted(below))})")
    say("2024: one-earner couple with two children, highest ratio (country, value)",
        f"{x24.couple_2ch_one_earner.idxmax()}, {x24.couple_2ch_one_earner.max():.2f}")
    say("2024: single adult, median ratio across the countries", f"{x24.single.median():.2f}")
    say("2024: one-earner couple with two children, median ratio across the countries",
        f"{x24.couple_2ch_one_earner.median():.2f}")
    say("2024: Portugal's rank (1 = highest) among the countries: single; one-earner couple with two children",
        f"{int(x24.single.rank(ascending=False)['PT'])}; {int(x24.couple_2ch_one_earner.rank(ascending=False)['PT'])} of {n}")

    # ------------------------------------------------------------------ OECD hours to escape poverty (50% line)
    section("OECD: weekly hours at the minimum wage needed to escape poverty (50% of median, minimum-income claimants)")
    for g in ["PT", "ES", "EL", "FR", "DE"]:
        parts = []
        for hh, lab in [("S_C0", "single"), ("S_C2", "single parent")]:
            if (g, hh) in HOURS.index:
                s = HOURS.loc[(g, hh)]
                yrs = [y for y in (2005, 2014, 2025) if y in s.index and not np.isnan(s[y])]
                parts.append(f"{lab} " + ", ".join(f"{y} {s[y]:.0f}" for y in yrs))
        say(f"{g}", " | ".join(parts))

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
