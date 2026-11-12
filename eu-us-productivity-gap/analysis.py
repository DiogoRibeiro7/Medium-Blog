"""Decompose the EU-US productivity gap from the OECD snapshot in ``data/``.

    python analysis.py

Writes tidy tables to ``results/*.csv`` and every number used in the article to
``results/analysis_summary.txt``. All aggregates are computed from country sums
(GDP, hours, employment, population, gross value added, labour compensation), so
group figures are hours-weighted or population-weighted by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
RES = Path(__file__).parent / "results"
YEAR = 2024  # last year with complete coverage of the 25 EU members in the database

GROUPS: dict[str, list[str]] = {
    "North-West": ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "NLD", "SWE"],
    "Ireland and Luxembourg": ["IRL", "LUX"],
    "South": ["ESP", "GRC", "ITA", "PRT"],
    "Centre-East": ["BGR", "HRV", "CZE", "EST", "HUN", "LVA", "LTU", "POL", "ROU", "SVK", "SVN"],
}
EU25 = [c for members in GROUPS.values() for c in members]  # EU27 minus Cyprus and Malta (not in the database)
GROUP_OF = {c: g for g, members in GROUPS.items() for c in members}

PERIODS = [(1998, 2007), (2008, 2019), (2020, 2023)]  # US growth accounting starts in 1998; ITA and ESP end in 2023
GA_COUNTRIES = ["USA", "DEU", "FRA", "NLD", "AUT", "ITA", "ESP", "SVK", "SVN", "LTU", "EST"]

LINES: list[str] = []


def say(label: str, value) -> None:
    if isinstance(value, float):
        value = f"{value:.4f}"
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


# ---------------------------------------------------------------------------- loading
PDB = pd.read_csv(DATA / "oecd_pdb.csv", low_memory=False)
GNI = pd.read_csv(DATA / "oecd_gni.csv", low_memory=False)


def series(measure: str, unit: str, price: str, transform: str = "N", asset: str = "_Z") -> pd.DataFrame:
    x = PDB[(PDB.MEASURE == measure) & (PDB.UNIT_MEASURE == unit) & (PDB.PRICE_BASE == price)
            & (PDB.TRANSFORMATION == transform) & (PDB.ASSET_CODE == asset)]
    if x.empty:
        raise KeyError((measure, unit, price, transform, asset))
    return x.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="first")


GDP_CUR = series("GDP", "USD_PPP", "V")          # million USD, current prices and current PPPs
GDP_XDC = series("GDP", "XDC", "V")              # million national currency
GDP_CON = series("GDP", "USD_PPP", "LR")         # million USD, chain-linked volumes, 2020 PPPs
GVA_PPP = series("GVA", "USD_PPP", "V")          # million USD, current PPPs
GVA_XDC = series("GVA", "XDC", "V")              # million national currency
LCTOT = series("LCTOT", "XDC", "V")              # million national currency, employees + self-employed imputation
HOURS = series("HRSTO", "H", "_Z")               # million hours
EMP = series("EMP", "PS", "_Z")                  # thousand persons
POP = series("POP", "PS", "_Z")                  # thousand persons
GDPHRS_CUR = series("GDPHRS", "USD_PPP_H", "V")  # published, for checks and the EU27 headline
GDPHRS_CON = series("GDPHRS", "USD_PPP_H", "LR")
GDPPOP_CUR = series("GDPPOP", "USD_PPP_PS", "V")
LP_GROWTH = series("GVAHRS", "XDC_H", "LR", "GY")
MFP_CON = series("MFP_PCCONLP", "XDC_INP", "LR", "GY", "_T")
K_CON = {a: series("KSERHRS_PCCONLP", "XDC_H", "LR", "GY", a) for a in ("_T", "ICT", "NICT")}

INC = pd.read_csv(DATA / "oecd_income.csv", low_memory=False)
inc = INC[INC.ACTIVITY.isin(["_T", "_Z"])].pivot_table(index="TIME_PERIOD", columns=["TRANSACTION", "REF_AREA"],
                                                        values="OBS_VALUE", aggfunc="first")
D2X3 = inc["D2X3"]      # taxes less subsidies on production and imports, million national currency
D1 = inc["D1"]          # compensation of employees
B1GQ_INC = inc["B1GQ"]

gni = GNI.pivot_table(index=["TIME_PERIOD"], columns=["TRANSACTION", "REF_AREA"], values="OBS_VALUE", aggfunc="first")
GNI_RATIO_RAW = gni["B5G"] / gni["B1GQ"]
NNI_RATIO = (gni["B5N"] / gni["B1GQ"]).reindex(GDP_CUR.index).ffill()    # same carry-forward as GNI
CFC_RATIO = (gni["P51C"] / gni["B1GQ"]).reindex(GDP_CUR.index).ffill()
# carry the last observed ratios forward (Bulgaria stops in 2022); record which cells were filled
GNI_RATIO = GNI_RATIO_RAW.reindex(GDP_CUR.index).ffill()
GNI_FILLED = GNI_RATIO_RAW.reindex(GDP_CUR.index).isna() & GNI_RATIO.notna()


def needs(year: int, codes: list[str], *frames: pd.DataFrame) -> list[str]:
    """Countries with every input present in a year."""
    ok = []
    for c in codes:
        if all(c in f.columns and year in f.index and pd.notna(f.loc[year, c]) for f in frames):
            ok.append(c)
    return ok


def agg(year: int, codes: list[str]) -> dict[str, float]:
    """Sum-based aggregate for a list of countries in one year."""
    c = needs(year, codes, GDP_CUR, HOURS, EMP, POP)
    gdp = GDP_CUR.loc[year, c].sum()
    hrs = HOURS.loc[year, c].sum()
    out = {
        "n": len(c),
        "gdp_hour": gdp / hrs,
        "gdp_capita": gdp * 1e6 / (POP.loc[year, c].sum() * 1e3),
        "hours_worker": hrs * 1e6 / (EMP.loc[year, c].sum() * 1e3),
        "emp_pop": EMP.loc[year, c].sum() / POP.loc[year, c].sum(),
        "hours": hrs,
    }
    cc = needs(year, c, GDP_CON)
    out["gdp_hour_const"] = GDP_CON.loc[year, cc].sum() / HOURS.loc[year, cc].sum() if len(cc) == len(c) else np.nan
    cg = needs(year, c, GNI_RATIO)
    out["gni_hour"] = (GDP_CUR.loc[year, cg] * GNI_RATIO.loc[year, cg]).sum() / HOURS.loc[year, cg].sum() if len(cg) == len(c) else np.nan
    cn = needs(year, c, NNI_RATIO, CFC_RATIO)
    if len(cn) == len(c):
        out["nni_hour"] = (GDP_CUR.loc[year, cn] * NNI_RATIO.loc[year, cn]).sum() / hrs
        out["cfc_hour"] = (GDP_CUR.loc[year, cn] * CFC_RATIO.loc[year, cn]).sum() / hrs
    else:
        out["nni_hour"] = out["cfc_hour"] = np.nan
    cl = needs(year, c, GDP_XDC, LCTOT, D2X3)
    if len(cl) == len(c):
        ppp = GDP_CUR.loc[year, cl] / GDP_XDC.loc[year, cl]          # USD PPP per unit of national currency
        lc = (LCTOT.loc[year, cl] * ppp).sum()
        tax = (D2X3.loc[year, cl] * ppp).sum()
        out.update(lc_hour=lc / hrs, tax_hour=tax / hrs, rest_hour=(gdp - lc - tax) / hrs,
                   labour_share=lc / gdp, tax_share=tax / gdp)
    else:
        out.update(lc_hour=np.nan, tax_hour=np.nan, rest_hour=np.nan, labour_share=np.nan, tax_share=np.nan)
    return out


def main() -> None:
    RES.mkdir(exist_ok=True)
    say("Reference year", YEAR)
    say("Countries in the EU aggregate (EU27 minus Cyprus and Malta)", len(EU25))

    # ------------------------------------------------------------------ data checks
    section("Checks")
    ratio = (GDP_CUR.loc[YEAR] / HOURS.loc[YEAR] / GDPHRS_CUR.loc[YEAR]).dropna()
    say("Max abs deviation of GDP/hours from published GDP per hour, 2024", float((ratio - 1).abs().max()))
    filled = [c for c in EU25 + ["USA"] if c in GNI_FILLED.columns and GNI_FILLED.loc[YEAR, c]]
    say("GNI/GDP ratio carried forward from last observed year (2024)", ", ".join(filled) or "none")
    us = agg(YEAR, ["USA"])
    eu = agg(YEAR, EU25)
    say("EU25 countries with complete inputs, 2024", eu["n"])

    # ------------------------------------------------------------------ levels by country
    section("Country levels, 2024, current PPPs (US = 100)")
    rows = []
    for c in EU25 + ["USA"]:
        a = agg(YEAR, [c])
        rows.append({
            "country": c, "group": GROUP_OF.get(c, "United States"),
            "gdp_hour": a["gdp_hour"], "gdp_hour_rel": 100 * a["gdp_hour"] / us["gdp_hour"],
            "gni_hour": a["gni_hour"], "gni_hour_rel": 100 * a["gni_hour"] / us["gni_hour"],
            "gni_gdp_ratio": GNI_RATIO.loc[YEAR, c] if c in GNI_RATIO.columns else np.nan,
            "gdp_hour_const_rel": 100 * a["gdp_hour_const"] / us["gdp_hour_const"],
            "gdp_capita_rel": 100 * a["gdp_capita"] / us["gdp_capita"],
            "hours_worker": a["hours_worker"], "emp_pop": a["emp_pop"],
            # log-point decomposition of the per-capita gap: productivity + hours per worker + employment rate
            "lp_productivity": 100 * np.log(a["gdp_hour"] / us["gdp_hour"]),
            "lp_hours": 100 * np.log(a["hours_worker"] / us["hours_worker"]),
            "lp_employment": 100 * np.log(a["emp_pop"] / us["emp_pop"]),
            "lp_total": 100 * np.log(a["gdp_capita"] / us["gdp_capita"]),
            "labour_share": a["labour_share"], "tax_share": a["tax_share"],
            "lc_hour_rel": 100 * a["lc_hour"] / us["lc_hour"],
            "rest_hour_rel": 100 * a["rest_hour"] / us["rest_hour"],
        })
    countries = pd.DataFrame(rows).sort_values("gdp_hour_rel", ascending=False)
    countries.to_csv(RES / "country_levels_2024.csv", index=False, float_format="%.4f")
    for r in countries.itertuples():
        say(f"{r.country} GDP/hour rel US; GNI/hour rel US; GNI/GDP; per capita rel US; hours/worker",
            f"{r.gdp_hour_rel:.1f}; {r.gni_hour_rel:.1f}; {r.gni_gdp_ratio:.3f}; {r.gdp_capita_rel:.1f}; {r.hours_worker:.0f}")
    check = (countries.lp_productivity + countries.lp_hours + countries.lp_employment - countries.lp_total).abs().max()
    say("Max abs error of the per-capita log decomposition (log points)", float(check))

    # ------------------------------------------------------------------ groups
    section("Groups, 2024")
    grows = []
    for name, codes in [("EU25", EU25), *GROUPS.items(), ("EU25 without Ireland and Luxembourg",
                                                          [c for c in EU25 if c not in ("IRL", "LUX")]), ("United States", ["USA"])]:
        a = agg(YEAR, codes)
        a = {"group": name, **a}
        for k in ("gdp_hour", "gni_hour", "nni_hour", "gdp_hour_const", "gdp_capita", "lc_hour", "tax_hour", "rest_hour"):
            a[k + "_rel"] = 100 * a[k] / us[k]
        a["lp_productivity"] = 100 * np.log(a["gdp_hour"] / us["gdp_hour"])
        a["lp_hours"] = 100 * np.log(a["hours_worker"] / us["hours_worker"])
        a["lp_employment"] = 100 * np.log(a["emp_pop"] / us["emp_pop"])
        a["lp_total"] = 100 * np.log(a["gdp_capita"] / us["gdp_capita"])
        grows.append(a)
    groups = pd.DataFrame(grows)
    groups.to_csv(RES / "group_levels_2024.csv", index=False, float_format="%.4f")
    for r in groups.itertuples():
        say(f"{r.group}: GDP/hour (USD PPP)", r.gdp_hour)
        say(f"{r.group}: GDP/hour rel US; GNI/hour rel US; const-PPP GDP/hour rel US; GDP per capita rel US",
            f"{r.gdp_hour_rel:.1f}; {r.gni_hour_rel:.1f}; {r.gdp_hour_const_rel:.1f}; {r.gdp_capita_rel:.1f}")
        say(f"{r.group}: net national income per hour rel US; depreciation (CFC) per hour, USD PPP",
            f"{r.nni_hour_rel:.1f}; {r.cfc_hour:.1f}")
        say(f"{r.group}: hours per worker; employment/population",
            f"{r.hours_worker:.0f}; {r.emp_pop:.3f}")
        say(f"{r.group}: per-capita gap in log points = productivity + hours + employment",
            f"{r.lp_total:.1f} = {r.lp_productivity:.1f} + {r.lp_hours:.1f} + {r.lp_employment:.1f}")
        say(f"{r.group}: labour share of GDP; labour comp/hour rel US; capital and other income/hour rel US",
            f"{r.labour_share:.3f}; {r.lc_hour_rel:.1f}; {r.rest_hour_rel:.1f}")

    # ------------------------------------------------------------------ the headline under different definitions
    section("EU relative to US under different definitions, 2024 (US = 100)")
    oecd_eu = "EU27_2020"
    defs = [
        ("GDP per capita, current PPPs, OECD EU27 aggregate", 100 * GDPPOP_CUR.loc[YEAR, oecd_eu] / GDPPOP_CUR.loc[YEAR, "USA"]),
        ("GDP per hour, constant 2020 PPPs, OECD EU27 aggregate", 100 * GDPHRS_CON.loc[YEAR, oecd_eu] / GDPHRS_CON.loc[YEAR, "USA"]),
        ("GDP per hour, current PPPs, OECD EU27 aggregate", 100 * GDPHRS_CUR.loc[YEAR, oecd_eu] / GDPHRS_CUR.loc[YEAR, "USA"]),
        ("GDP per hour, current PPPs, sum of 25 members", float(groups.set_index("group").loc["EU25", "gdp_hour_rel"])),
        ("GNI per hour, current PPPs, sum of 25 members", float(groups.set_index("group").loc["EU25", "gni_hour_rel"])),
        ("Net national income per hour, current PPPs, sum of 25 members", float(groups.set_index("group").loc["EU25", "nni_hour_rel"])),
        ("GDP per capita, current PPPs, North-West", float(groups.set_index("group").loc["North-West", "gdp_capita_rel"])),
        ("GDP per hour, current PPPs, North-West", float(groups.set_index("group").loc["North-West", "gdp_hour_rel"])),
        ("GNI per hour, current PPPs, North-West", float(groups.set_index("group").loc["North-West", "gni_hour_rel"])),
        ("GDP per hour, current PPPs, South", float(groups.set_index("group").loc["South", "gdp_hour_rel"])),
        ("GDP per hour, current PPPs, Centre-East", float(groups.set_index("group").loc["Centre-East", "gdp_hour_rel"])),
    ]
    pd.DataFrame(defs, columns=["definition", "eu_relative_to_us"]).to_csv(RES / "headline_definitions_2024.csv", index=False, float_format="%.4f")
    for k, v in defs:
        say(k, f"{v:.1f}")
    vals = [v for _, v in defs]
    say("Range of the ten headline numbers", f"{min(vals):.1f} to {max(vals):.1f}")

    # ------------------------------------------------------------------ composition of the EU-US per-hour gap
    section("Contribution of each group to the EU25-US gap in GDP per hour (current PPPs)")
    comp_rows = []
    for y in (1995, 2000, 2007, 2019, YEAR):
        us_y = agg(y, ["USA"])["gdp_hour"]
        eu_y = agg(y, EU25)
        gap = us_y - eu_y["gdp_hour"]
        for name, codes in GROUPS.items():
            g = agg(y, codes)
            share = g["hours"] / eu_y["hours"]
            comp_rows.append({"year": y, "group": name, "hours_share": share,
                              "gdp_hour_rel": 100 * g["gdp_hour"] / us_y,
                              "contribution": share * (us_y - g["gdp_hour"]) / gap,
                              "eu_gap_pct_of_us": 100 * gap / us_y, "n_eu": eu_y["n"]})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(RES / "gap_composition.csv", index=False, float_format="%.4f")
    for r in comp.itertuples():
        say(f"{r.year} {r.group}: hours share; GDP/hour rel US; share of EU gap",
            f"{r.hours_share:.3f}; {r.gdp_hour_rel:.1f}; {r.contribution:.3f}")
    for y in comp.year.unique():
        say(f"{y} EU gap as % of US GDP/hour (members in sum)",
            f"{comp[comp.year == y].eu_gap_pct_of_us.iloc[0]:.1f} ({comp[comp.year == y].n_eu.iloc[0]})")

    # ------------------------------------------------------------------ time paths
    section("Time paths, GDP per hour relative to US")
    years = range(1995, YEAR + 1)
    stable = {name: [c for c in codes if all(c in needs(y, codes, GDP_CUR, GDP_CON, HOURS, EMP, POP) for y in years)]
              for name, codes in [("EU25", EU25), *GROUPS.items()]}
    say("Members present in every year 1995-2024 (time paths use these)",
        "; ".join(f"{k}: {len(v)}" for k, v in stable.items()))
    say("Dropped from the time paths", ", ".join(sorted(set(EU25) - set(stable["EU25"]))))
    tp = []
    for y in range(1995, YEAR + 1):
        us_y = agg(y, ["USA"])
        row = {"year": y}
        for name, codes in stable.items():
            g = agg(y, codes)
            row[f"{name}|const"] = 100 * g["gdp_hour_const"] / us_y["gdp_hour_const"]
            row[f"{name}|cur"] = 100 * g["gdp_hour"] / us_y["gdp_hour"]
            row[f"{name}|n"] = g["n"]
        row["OECD EU27|const"] = 100 * GDPHRS_CON.loc[y, oecd_eu] / GDPHRS_CON.loc[y, "USA"]
        row["OECD EU27|cur"] = 100 * GDPHRS_CUR.loc[y, oecd_eu] / GDPHRS_CUR.loc[y, "USA"]
        tp.append(row)
    paths = pd.DataFrame(tp)
    paths.to_csv(RES / "time_paths.csv", index=False, float_format="%.4f")
    for y in (1995, 2000, 2007, 2019, YEAR):
        r = paths.set_index("year").loc[y]
        say(f"{y} constant 2020 PPPs: EU25, North-West, South, Centre-East (US=100)",
            f"{r['EU25|const']:.1f}, {r['North-West|const']:.1f}, {r['South|const']:.1f}, {r['Centre-East|const']:.1f}")
        say(f"{y} current PPPs: EU25, North-West, South, Centre-East (US=100)",
            f"{r['EU25|cur']:.1f}, {r['North-West|cur']:.1f}, {r['South|cur']:.1f}, {r['Centre-East|cur']:.1f}")
        say(f"{y} OECD EU27 aggregate, constant vs current PPPs", f"{r['OECD EU27|const']:.1f} vs {r['OECD EU27|cur']:.1f}")
    pt = paths.set_index("year")
    for y in (1995, 2000, 2005, 2010, 2015, 2019, 2020, YEAR):
        say(f"{y} EU25 current-PPP level / constant-PPP level (relative price drift)",
            f"{pt.loc[y, 'EU25|cur'] / pt.loc[y, 'EU25|const']:.3f}")
    for name in ("EU25", "North-West", "South", "Centre-East"):
        say(f"{name}: change 1995-{YEAR} in level rel US, constant vs current PPPs (points)",
            f"{pt.loc[YEAR, name + '|const'] - pt.loc[1995, name + '|const']:.1f} vs {pt.loc[YEAR, name + '|cur'] - pt.loc[1995, name + '|cur']:.1f}")
    diffs = {g: (pt[f"{g}|const"] - pt[f"{g}|cur"]).abs() for g in ("EU25", "North-West", "South", "Centre-East")}
    worst = max(diffs, key=lambda g: diffs[g].max())
    say("Largest gap between constant- and current-PPP levels, any group and year (points; group, year)",
        f"{diffs[worst].max():.1f}; {worst}, {int(diffs[worst].idxmax())}")
    say(f"OECD EU27: change 1995-{YEAR}, constant vs current PPPs (points)",
        f"{pt.loc[YEAR, 'OECD EU27|const'] - pt.loc[1995, 'OECD EU27|const']:.1f} vs {pt.loc[YEAR, 'OECD EU27|cur'] - pt.loc[1995, 'OECD EU27|cur']:.1f}")
    # growth of GDP per hour in constant PPPs, 1995-2024 and 2019-2024, annualised
    for a, b in ((1995, YEAR), (1995, 2007), (2007, 2019), (2019, YEAR)):
        us_g = (agg(b, ["USA"])["gdp_hour_const"] / agg(a, ["USA"])["gdp_hour_const"]) ** (1 / (b - a)) - 1
        codes_a = needs(a, EU25, GDP_CON, HOURS)
        codes = [c for c in codes_a if c in needs(b, EU25, GDP_CON, HOURS)]
        eu_a = GDP_CON.loc[a, codes].sum() / HOURS.loc[a, codes].sum()
        eu_b = GDP_CON.loc[b, codes].sum() / HOURS.loc[b, codes].sum()
        eu_g = (eu_b / eu_a) ** (1 / (b - a)) - 1
        nw = [c for c in GROUPS["North-West"] if c in codes]
        nw_g = ((GDP_CON.loc[b, nw].sum() / HOURS.loc[b, nw].sum()) / (GDP_CON.loc[a, nw].sum() / HOURS.loc[a, nw].sum())) ** (1 / (b - a)) - 1
        say(f"Annual growth of GDP per hour (constant PPPs) {a}-{b}: US, EU (same members), North-West",
            f"{100 * us_g:.2f}%, {100 * eu_g:.2f}%, {100 * nw_g:.2f}% ({len(codes)} members)")

    # ------------------------------------------------------------------ growth accounting
    section("Growth accounting: average annual contributions to labour productivity growth (pp)")
    ga_rows = []
    for c in GA_COUNTRIES:
        for a, b in PERIODS:
            yrs = list(range(a, b + 1))
            def avg(frame: pd.DataFrame) -> float:
                if c not in frame.columns:
                    return np.nan
                v = frame.loc[frame.index.isin(yrs), c]
                return float(v.mean()) if v.notna().sum() == len(yrs) else np.nan
            row = {"country": c, "period": f"{a}-{b}", "lp_growth": avg(LP_GROWTH), "mfp": avg(MFP_CON),
                   "capital_deepening": avg(K_CON["_T"]), "ict_capital": avg(K_CON["ICT"]), "nonict_capital": avg(K_CON["NICT"])}
            ga_rows.append(row)
    ga = pd.DataFrame(ga_rows)
    ga["residual"] = ga.lp_growth - ga.mfp - ga.capital_deepening
    ga.to_csv(RES / "growth_accounting.csv", index=False, float_format="%.4f")
    say("Largest |measured LP growth - (capital deepening + MFP)| by country (pp a year)",
        "; ".join(f"{c} {v:.2f}" for c, v in ga.dropna(subset=["residual"]).groupby("country").residual.apply(lambda x: x.abs().max()).items()))
    for r in ga.dropna(subset=["lp_growth", "mfp", "capital_deepening"]).itertuples():
        say(f"{r.country} {r.period}: LP growth = capital deepening (ICT, non-ICT) + MFP",
            f"{r.lp_growth:.2f} = {r.capital_deepening:.2f} ({r.ict_capital:.2f}, {r.nonict_capital:.2f}) + {r.mfp:.2f}")
    # cumulative US advantage over each country, split into capital deepening and MFP
    cum_rows = []
    for c in GA_COUNTRIES[1:]:
        sub = ga[ga.country.isin(["USA", c])].pivot(index="period", columns="country")[["lp_growth", "capital_deepening", "mfp"]]
        if sub.isna().any().any():
            continue
        tot_lp = sum((sub.loc[p, ("lp_growth", "USA")] - sub.loc[p, ("lp_growth", c)]) * (int(p[5:]) - int(p[:4]) + 1) for p in sub.index)
        tot_k = sum((sub.loc[p, ("capital_deepening", "USA")] - sub.loc[p, ("capital_deepening", c)]) * (int(p[5:]) - int(p[:4]) + 1) for p in sub.index)
        tot_m = sum((sub.loc[p, ("mfp", "USA")] - sub.loc[p, ("mfp", c)]) * (int(p[5:]) - int(p[:4]) + 1) for p in sub.index)
        tot_ga = tot_k + tot_m  # labour productivity growth as decomposed by growth accounting
        cum_rows.append({"country": c, "lp_measured": tot_lp, "lp": tot_ga, "capital_deepening": tot_k, "mfp": tot_m})
        say(f"US minus {c}, cumulative {PERIODS[0][0]}-{PERIODS[-1][1]} (pp): growth-accounting LP = capital deepening + MFP; MFP share; measured LP",
            f"{tot_ga:.1f} = {tot_k:.1f} + {tot_m:.1f}; {tot_m / tot_ga:.2f}; {tot_lp:.1f}")
    pd.DataFrame(cum_rows).to_csv(RES / "growth_gap_cumulative.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------------------ income side of GDP per hour
    section("GDP per hour split into labour compensation, net taxes on production and the rest, 2024")
    gva_basis = (GDP_XDC / GVA_XDC).loc[YEAR]
    say("GDP/GVA in the productivity database: US; EU25 range",
        f"{gva_basis['USA']:.3f}; {gva_basis[EU25].min():.3f} to {gva_basis[EU25].max():.3f}")
    gva_hrs = series("GVAHRS", "USD_PPP_H", "V").loc[YEAR]
    gva_eu = GVA_PPP.loc[YEAR, EU25].sum() / HOURS.loc[YEAR, EU25].sum()
    gva_nw = GVA_PPP.loc[YEAR, GROUPS["North-West"]].sum() / HOURS.loc[YEAR, GROUPS["North-West"]].sum()
    say("Published GVA per hour rel US (US GVA at market prices, EU at basic prices): EU25; North-West",
        f"{100 * gva_eu / gva_hrs['USA']:.1f}; {100 * gva_nw / gva_hrs['USA']:.1f}")
    stat_disc = (B1GQ_INC.loc[YEAR] - D1.loc[YEAR] - inc["B2A3G"].loc[YEAR] - D2X3.loc[YEAR]) / B1GQ_INC.loc[YEAR]
    say("US statistical discrepancy (GDP minus income components) as share of GDP", float(stat_disc["USA"]))
    say("Self-employed imputation, LCTOT/D1: US; EU25 median",
        f"{(LCTOT / D1).loc[YEAR, 'USA']:.3f}; {(LCTOT / D1).loc[YEAR, EU25].median():.3f}")
    inc_rows = []
    for name, codes in [("United States", ["USA"]), ("EU25", EU25), *GROUPS.items()]:
        g = agg(YEAR, codes)
        inc_rows.append({"group": name, "gdp_hour": g["gdp_hour"], "lc_hour": g["lc_hour"], "tax_hour": g["tax_hour"],
                         "rest_hour": g["rest_hour"], "cfc_hour": g["cfc_hour"], "net_rest_hour": g["rest_hour"] - g["cfc_hour"], "labour_share": g["labour_share"], "tax_share": g["tax_share"]})
    incs = pd.DataFrame(inc_rows)
    usr = incs.iloc[0]
    for k in ("gdp_hour", "lc_hour", "tax_hour", "rest_hour", "cfc_hour", "net_rest_hour"):
        incs[f"gap_{k}"] = usr[k] - incs[k]
    incs["share_gap_rest"] = incs.gap_rest_hour / incs.gap_gdp_hour
    incs.to_csv(RES / "income_split_2024.csv", index=False, float_format="%.4f")
    for r in incs.itertuples():
        say(f"{r.group}: rest/hour = depreciation + net remainder (USD PPP)",
            f"{r.rest_hour:.1f} = {r.cfc_hour:.1f} + {r.net_rest_hour:.1f}")
        say(f"{r.group}: GDP/hour = labour comp + net production taxes + rest (USD PPP); labour share; tax share",
            f"{r.gdp_hour:.1f} = {r.lc_hour:.1f} + {r.tax_hour:.1f} + {r.rest_hour:.1f}; {r.labour_share:.3f}; {r.tax_share:.3f}")
        if r.group != "United States":
            say(f"{r.group}: gap to US in GDP/hour = labour comp + taxes + rest",
                f"{r.gap_gdp_hour:.1f} = {r.gap_lc_hour:.1f} + {r.gap_tax_hour:.1f} + {r.gap_rest_hour:.1f}")
    for c in ["DEU", "FRA", "NLD", "BEL", "DNK", "AUT", "SWE", "ITA", "ESP", "PRT", "GRC", "POL", "IRL", "LUX"]:
        g = agg(YEAR, [c])
        say(f"{c}: labour share; labour comp/hour; net taxes/hour; rest/hour (USD PPP)",
            f"{g['labour_share']:.3f}; {g['lc_hour']:.1f}; {g['tax_hour']:.1f}; {g['rest_hour']:.1f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
