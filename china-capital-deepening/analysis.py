"""Capital deepening in Chinese manufacturing: a test of the Part 2 conjecture.

    python analysis.py

Conjecture (Part 2 of "Europe's Low-Wage Path, Measured"): the growth of value added per worker in Chinese
manufacturing after 2013 came with faster growth of capital per worker, so that value added per unit of capital
fell while value added per worker rose. Reads ``data/`` and writes ``results/*.csv`` and every number used in
the article to ``results/analysis_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import cip

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"

ECON = ["CHN", "USA", "JPN", "KOR", "DEU", "FRA", "ITA", "ESP", "NLD", "POL", "CZE", "MEX", "IND", "IDN"]
EUROPE_M = ["DEU", "FRA", "ITA", "ESP", "NLD", "AUT", "BEL", "CZE", "POL", "PRT", "SWE"]
MANUF = EUROPE_M + ["JPN"]
NAMES = {"CHN": "China", "USA": "United States", "JPN": "Japan", "KOR": "Korea", "DEU": "Germany", "FRA": "France",
         "ITA": "Italy", "ESP": "Spain", "NLD": "Netherlands", "POL": "Poland", "CZE": "Czechia", "MEX": "Mexico",
         "IND": "India", "IDN": "Indonesia", "AUT": "Austria", "BEL": "Belgium", "PRT": "Portugal", "SWE": "Sweden"}
PERIODS = [(1995, 2007), (2007, 2013), (2013, 2019), (2019, 2024)]

LINES: list[str] = []


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


def f2(x: float) -> str:
    return f"{round(float(x), 2) + 0.0:.2f}"


def growth(s: pd.Series, a: int, b: int) -> float:
    """Average annual log growth, percent."""
    return 100 * (np.log(s.loc[b]) - np.log(s.loc[a])) / (b - a)


# ---------------------------------------------------------------------------------------------- loaders
def ltb() -> pd.DataFrame:
    d = pd.read_csv(DATA / "oecd_ltb.csv")
    return d.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="MEASURE", values="OBS_VALUE")


def table1() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(DATA / "oecd_t1.csv")
    v = d[d.PRICE_BASE == "V"].pivot_table(index="TIME_PERIOD", columns=["TRANSACTION", "REF_AREA"],
                                           values="OBS_VALUE")
    lr = d[(d.PRICE_BASE == "LR") & (d.TRANSACTION == "B1GQ")].pivot_table(index="TIME_PERIOD",
                                                                           columns="REF_AREA", values="OBS_VALUE")
    return v, lr


def oecd_manufacturing() -> dict[str, pd.DataFrame]:
    k = pd.read_csv(DATA / "oecd_t9a.csv")
    y = pd.read_csv(DATA / "oecd_t6.csv")
    e = pd.read_csv(DATA / "oecd_t7.csv")
    out = {
        "K": k[k.PRICE_BASE == "L"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE"),
        "Y": y[y.PRICE_BASE == "L"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE"),
        "L": e[(e.TRANSACTION == "EMP") & (e.UNIT_MEASURE == "PS")].pivot_table(index="TIME_PERIOD",
                                                                                columns="REF_AREA", values="OBS_VALUE"),
        "H": e[(e.TRANSACTION == "EMP") & (e.UNIT_MEASURE == "H")].pivot_table(index="TIME_PERIOD",
                                                                               columns="REF_AREA", values="OBS_VALUE"),
        "KV": k[k.PRICE_BASE == "V"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE"),
        "YV": y[y.PRICE_BASE == "V"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE"),
    }
    return out


def nbs() -> pd.DataFrame:
    return pd.read_csv(DATA / "nbs_published.csv")


# ---------------------------------------------------------------------------------------------- sections
def economy_wide() -> pd.DataFrame:
    section("Whole economy: OECD productive capital stock, GDP volume and potential employment (EO long-term baseline)")
    p = ltb()
    rows = []
    for c in ECON:
        x = p.loc[c]
        yl, kl, yk = x.GDPV / x.ETPT, x.KTPV_AV / x.ETPT, x.GDPV / x.KTPV_AV
        r = {"country": c}
        for a, b in PERIODS:
            if a in x.index and b in x.index and not np.isnan(kl.loc[a]):
                r[f"yl_{a}_{b}"], r[f"kl_{a}_{b}"], r[f"yk_{a}_{b}"] = growth(yl, a, b), growth(kl, a, b), growth(yk, a, b)
        for y in (1995, 2007, 2013, 2019, 2024):
            r[f"ky_{y}"] = 1 / yk.get(y, np.nan)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("country")
    t.to_csv(RES / "economy_wide.csv", float_format="%.4f")
    (p.loc["CHN"].GDPV / p.loc["CHN"].KTPV_AV).rename("china_total_oecd").rename_axis("year").to_csv(
        RES / "oecd_china_output_per_capital.csv", float_format="%.6f")
    for c in ECON:
        r = t.loc[c]
        say(f"{c} capital-output ratio 1995; 2007; 2013; 2019; 2024",
            "; ".join(f2(r[f"ky_{y}"]) if not np.isnan(r[f"ky_{y}"]) else "n/a" for y in (1995, 2007, 2013, 2019, 2024)))
    for a, b in PERIODS:
        say(f"{a}-{b} average annual growth, percent: output per worker; capital per worker; output per unit of capital",
            " | ".join(f"{c} {f2(t.loc[c, f'yl_{a}_{b}'])}, {f2(t.loc[c, f'kl_{a}_{b}'])}, {f2(t.loc[c, f'yk_{a}_{b}'])}"
                       for c in ECON if f"yl_{a}_{b}" in t.columns and not np.isnan(t.loc[c, f"yl_{a}_{b}"])))
    return t


def icor() -> pd.DataFrame:
    section("Investment rate and incremental capital-output ratio (OECD national accounts, GFCF / GDP over real GDP growth)")
    v, lr = table1()
    inv = 100 * v["P51G"] / v["B1GQ"]
    gr = 100 * lr.pct_change(fill_method=None)
    windows = [(1995, 2007), (2008, 2013), (2014, 2019), (2020, 2023)]
    rows = []
    for c in ECON:
        r = {"country": c}
        for a, b in windows:
            i, g = inv[c].loc[a:b].mean(), gr[c].loc[a:b].mean()
            r[f"inv_{a}_{b}"], r[f"growth_{a}_{b}"], r[f"icor_{a}_{b}"] = i, g, i / g if g > 0.5 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index("country")
    t.to_csv(RES / "icor.csv", float_format="%.4f")
    for a, b in windows:
        say(f"{a}-{b}: investment rate (% of GDP); real GDP growth (%); ICOR (blank when growth < 0.5%)",
            " | ".join(f"{c} {t.loc[c, f'inv_{a}_{b}']:.1f}; {t.loc[c, f'growth_{a}_{b}']:.2f}; "
                       f"{'' if np.isnan(t.loc[c, f'icor_{a}_{b}']) else f'{t.loc[c, f"icor_{a}_{b}"]:.1f}'}"
                       for c in ["CHN", "USA", "JPN", "KOR", "DEU", "FRA", "ITA", "ESP", "POL", "IND", "IDN", "MEX"]))
    say("China investment rate, % of GDP: 2007; 2013; 2019; 2023; 2024",
        "; ".join(f"{inv['CHN'].loc[y]:.1f}" for y in (2007, 2013, 2019, 2023, 2024)))
    return t


def europe_manufacturing() -> pd.DataFrame:
    section("Manufacturing in OECD countries: net fixed assets, value added (volumes), persons and hours (OECD Tables 9A, 6, 7)")
    m = oecd_manufacturing()
    K, Y, L, H = m["K"], m["Y"], m["L"], m["H"]
    windows = [(1995, 2007), (2007, 2013), (2013, 2017), (2013, 2019), (2019, 2023)]
    rows = []
    for c in MANUF:
        r = {"country": c}
        for a, b in windows:
            ok = all(c in s.columns and a in s.index and b in s.index and not np.isnan(s.loc[a, c]) and
                     not np.isnan(s.loc[b, c]) for s in (K, Y, L))
            if ok:
                r[f"yl_{a}_{b}"] = growth(Y[c] / L[c], a, b)
                r[f"kl_{a}_{b}"] = growth(K[c] / L[c], a, b)
                r[f"yk_{a}_{b}"] = growth(Y[c] / K[c], a, b)
        for y in (2013, 2017, 2019, 2023):
            if c in m["KV"].columns and y in m["KV"].index:
                r[f"ky_nominal_{y}"] = m["KV"].loc[y, c] / m["YV"].loc[y, c]
        rows.append(r)
    t = pd.DataFrame(rows).set_index("country")
    t.to_csv(RES / "oecd_manufacturing.csv", float_format="%.4f")
    yk = (Y[EUROPE_M] / K[EUROPE_M]).loc[1995:]
    idx = yk / yk.loc[2007]
    pd.DataFrame({"europe_manufacturing_median": idx.median(axis=1), "countries": idx.notna().sum(axis=1)}).rename_axis(
        "year").to_csv(RES / "europe_manufacturing_output_per_capital.csv", float_format="%.6f")
    for a, b in windows:
        cols = [f"yl_{a}_{b}", f"kl_{a}_{b}", f"yk_{a}_{b}"]
        if cols[0] not in t.columns:
            continue
        sub = t[cols].dropna()
        say(f"Manufacturing {a}-{b}, average annual growth, percent: output per worker; capital per worker; output per unit of capital",
            " | ".join(f"{c} {f2(sub.loc[c, cols[0]])}, {f2(sub.loc[c, cols[1]])}, {f2(sub.loc[c, cols[2]])}" for c in sub.index))
        europe = [c for c in sub.index if c in EUROPE_M]
        say(f"Manufacturing {a}-{b}: European countries with falling output per unit of capital; median growth of output per unit of capital",
            f"{sum(sub.loc[europe, cols[2]] < 0)} of {len(europe)} "
            f"({', '.join(c for c in europe if sub.loc[c, cols[2]] < 0) or 'none'}); {f2(sub.loc[europe, cols[2]].median())}")
    say("Manufacturing net fixed assets / value added at current prices, 2017: " + "; ".join(t.index),
        "; ".join(f2(t.loc[c, "ky_nominal_2017"]) if not np.isnan(t.loc[c].get("ky_nominal_2017", np.nan)) else "n/a"
                  for c in t.index))
    return t


def nbs_after_cip() -> pd.DataFrame:
    section("After the CIP data end: NBS efficiency indicators of industrial enterprises above designated size, 2017-2025")
    d = nbs()
    rows = []
    for ind in ("revenue_per_100_yuan_assets", "revenue_per_capita"):
        x = d[d.indicator == ind].set_index("year")
        idx = {2017: 100.0}
        for y in sorted(x.index):
            level, chg = float(x.loc[y, "value"]), float(x.loc[y, "change_on_comparable_basis"])
            idx[y] = idx[y - 1] * level / (level - chg)
        for y, v in idx.items():
            rows.append({"indicator": ind, "year": y, "index_2017_100": v})
    t = pd.DataFrame(rows).pivot_table(index="year", columns="indicator", values="index_2017_100")
    t["assets_per_employee"] = 100 * t.revenue_per_capita / t.revenue_per_100_yuan_assets
    t.to_csv(RES / "nbs_chained.csv", float_format="%.3f")
    say("Chained on NBS comparable-basis changes, 2017 = 100, 2025: revenue per 100 yuan of assets; revenue per employee; "
        "assets per employee", f"{t.loc[2025, 'revenue_per_100_yuan_assets']:.1f}; {t.loc[2025, 'revenue_per_capita']:.1f}; "
        f"{t.loc[2025, 'assets_per_employee']:.1f}")
    g = {c: 100 * np.log(t.loc[2025, c] / 100) / 8 for c in t.columns}
    say("Average annual log growth 2017-2025, percent: revenue per 100 yuan of assets; revenue per employee; assets per employee",
        f"{f2(g['revenue_per_100_yuan_assets'])}; {f2(g['revenue_per_capita'])}; {f2(g['assets_per_employee'])}")
    say("Years in which revenue per 100 yuan of assets fell on a comparable basis",
        ", ".join(str(y) for y, ch in d[d.indicator == "revenue_per_100_yuan_assets"][["year", "change_on_comparable_basis"]]
                  .itertuples(index=False) if float(ch) < 0))
    pr = d[d.indicator == "profit_rate_of_revenue"].set_index("year").value
    say("Profit rate of revenue, percent: " + "; ".join(str(y) for y in pr.index), "; ".join(f"{v:.2f}" for v in pr))
    return t


def main() -> None:
    RES.mkdir(exist_ok=True)
    have_cip = cip.available()
    if have_cip:
        cip.analyse(say, section, RES)
    economy_wide()
    icor()
    europe_manufacturing()
    nbs_after_cip()
    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
