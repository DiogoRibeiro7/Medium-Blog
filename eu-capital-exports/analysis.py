"""Saving, investment and cross-border capital flows of the EU27: what "EUR 300 billion a year" measures.

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
YEARS = list(range(2015, 2026))          # the BoP financial account for the EU27 is complete from 2015
SECTOR_NAMES = {"S14_S15": "households", "S11": "non-financial corporations", "S12": "financial corporations",
                "S13": "general government", "S1": "total economy"}

LINES: list[str] = []


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", low_memory=False)


def bn(x: float) -> str:
    return f"{x / 1e3:,.0f}"


# ------------------------------------------------------------------------------ inputs
BOP = read("bop_flows")
BOP["year"] = BOP.time.str[:4].astype(int)
_q = BOP.groupby(["partner", "bop_item", "stk_flow", "year"]).value.agg(["sum", "count"]).reset_index()
BOPA = _q[_q["count"] == 4].set_index(["partner", "bop_item", "stk_flow", "year"])["sum"]   # EUR million, full years

PS = read("bop_portfolio_sectors")
PS["year"] = PS.time.str[:4].astype(int)
_q = PS.groupby(["partner", "bop_item", "stk_flow", "sector10", "year"]).value.agg(["sum", "count"]).reset_index()
PSA = _q[_q["count"] == 4].set_index(["partner", "bop_item", "stk_flow", "sector10", "year"])["sum"]

IIP = read("iip")
IIP = IIP[IIP.partner == "EXT_EU27_2020"].pivot_table(index="time", columns=["bop_item", "stk_flow"], values="value")

SA = read("sector_accounts")
GDP = read("gdp")
GDP = GDP[GDP.na_item == "B1GQ"].pivot_table(index="time", columns="geo", values="value")
HH = read("household_assets").pivot_table(index=["geo", "time"], columns="na_item", values="value")
OECD_SI = read("oecd_saving_investment").pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="TRANSACTION",
                                                     values="OBS_VALUE")
OECD_HH = read("oecd_household_assets").pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="INSTR_ASSET",
                                                     values="OBS_VALUE")


def bop(item: str, flow: str, year: int, partner: str = "EXT_EU27_2020") -> float:
    return float(BOPA.get((partner, item, flow, year), np.nan))


def sa(geo: str, sector: str, item: str) -> pd.Series:
    """Sector-account series (EUR million). Balancing items appear on both sides; take one."""
    d = SA[(SA.geo == geo) & (SA.sector == sector) & (SA.na_item == item)]
    side = "RECV" if item in ("B8G", "B8N", "B9", "B6G") else "PAID"
    d = d[d.direct == side] if (d.direct == side).any() else d
    return d.groupby("time").value.first()


def main() -> None:
    RES.mkdir(exist_ok=True)
    g = GDP["EU27_2020"]

    # ------------------------------------------------------------------ the claim and the checks
    section("Checks")
    ca_sa = (sa("EU27_2020", "S1", "B9") / 1e3).loc[2024]
    cka = bop("CKA", "BAL", 2024) / 1e3
    say("EU27 2024 net lending to the rest of the world, EUR bn: sector accounts (B9, all partners); BoP current+capital "
        "account, extra-EU", f"{ca_sa:,.0f}; {cka:,.0f}")
    gaps = [abs(bop("CA", "BAL", y) + bop("KA", "BAL", y) + bop("EO", "NET", y) - bop("FA", "NET", y)) for y in YEARS]
    say("BoP identity, max |current + capital account + errors and omissions - net financial account|, 2015-2025, EUR million",
        f"{max(gaps):,.0f}")
    gaps = [abs(bop("FA__P__F", "NET", y) - (bop("FA__P__F", "ASS", y) - bop("FA__P__F", "LIAB", y))) for y in YEARS]
    say("Portfolio investment: max |net - (assets - liabilities)|, 2015-2025, EUR million", f"{max(gaps):,.0f}")
    s_eu = (sa("EU27_2020", "S1", "B8G") / g * 100)
    oecd_eu = OECD_SI.loc["EU27_2020"]
    s_oecd = (oecd_eu.B8N + oecd_eu.P51C) / oecd_eu.B1GQ * 100
    say("EU27 gross saving, % of GDP, 2024: Eurostat sector accounts; OECD Table 2", f"{s_eu.loc[2024]:.1f}; {s_oecd.loc[2024]:.1f}")

    # ------------------------------------------------------------------ saving and investment
    section("Saving and investment, % of GDP")
    rows = []
    for y in range(1999, 2026):
        if y not in g.index or np.isnan(g.loc[y]):
            continue
        S, I = sa("EU27_2020", "S1", "B8G").get(y, np.nan), sa("EU27_2020", "S1", "P5G").get(y, np.nan)
        C, NL = sa("EU27_2020", "S1", "P51C").get(y, np.nan), sa("EU27_2020", "S1", "B9").get(y, np.nan)
        rows.append({"area": "EU27", "year": y, "saving": S / g[y] * 100, "investment": I / g[y] * 100,
                     "depreciation": C / g[y] * 100, "net_investment": (I - C) / g[y] * 100, "net_lending": NL / g[y] * 100})
    for a in ["USA", "EA20", "JPN", "GBR"]:
        o = OECD_SI.loc[a]
        for y, r in o.iterrows():
            rows.append({"area": a, "year": int(y), "saving": (r.B8N + r.P51C) / r.B1GQ * 100, "investment": r.P5 / r.B1GQ * 100,
                         "depreciation": r.P51C / r.B1GQ * 100, "net_investment": (r.P5 - r.P51C) / r.B1GQ * 100,
                         "net_lending": r.B9 / r.B1GQ * 100})
    si = pd.DataFrame(rows).dropna(subset=["saving"])
    si.to_csv(RES / "saving_investment.csv", index=False, float_format="%.3f")
    S = si.set_index(["area", "year"])
    for a, yrs in [("EU27", (2007, 2013, 2019, 2024, 2025)), ("USA", (2007, 2013, 2019, 2024)), ("EA20", (2024, 2025)),
                   ("JPN", (2024,)), ("GBR", (2024,))]:
        for y in yrs:
            r = S.loc[(a, y)]
            say(f"{a} {y}: gross saving; gross capital formation; depreciation; net investment; net lending (% of GDP)",
                f"{r.saving:.1f}; {r.investment:.1f}; {r.depreciation:.1f}; {r.net_investment:.1f}; {r.net_lending:.1f}")
    for a in ["EU27", "USA"]:
        w = S.loc[a].loc[2015:2024]
        say(f"{a} 2015-2024 averages: gross saving; gross capital formation; net lending (% of GDP)",
            f"{w.saving.mean():.1f}; {w.investment.mean():.1f}; {w.net_lending.mean():.1f}")
    eu24 = S.loc[("EU27", 2024)]
    say("EU27 2024 in EUR bn: GDP; gross saving; gross capital formation; depreciation; net lending",
        f"{bn(g[2024])}; {bn(sa('EU27_2020', 'S1', 'B8G')[2024])}; {bn(sa('EU27_2020', 'S1', 'P5G')[2024])}; "
        f"{bn(sa('EU27_2020', 'S1', 'P51C')[2024])}; {bn(sa('EU27_2020', 'S1', 'B9')[2024])}")
    say("EU27 2024: investment rate if the net lending had been invested at home (% of GDP)",
        f"{eu24.investment + eu24.net_lending:.1f}")
    say("Draghi's additional investment (4.4-4.7% of 2023 GDP) vs EU27 net lending 2024 (% of GDP): net lending covers",
        f"{eu24.net_lending / 4.7 * 100:.0f}% to {eu24.net_lending / 4.4 * 100:.0f}%")
    years_surplus = [y for y in range(2010, 2026) if S.loc[("EU27", y)].net_lending > 0]
    say("EU27 years with positive net lending, 2010-2025", f"{len(years_surplus)} of 16")

    # ------------------------------------------------------------------ net lending by sector
    section("EU27 net lending by institutional sector (% of GDP; EUR bn)")
    nl = {}
    for sec in ["S14_S15", "S11", "S12", "S13", "S1"]:
        nl[sec] = sa("EU27_2020", sec, "B9") / g * 100
    nl = pd.DataFrame(nl).dropna(how="all")
    nl.to_csv(RES / "sector_net_lending.csv", float_format="%.3f")
    for y in (2007, 2013, 2019, 2022, 2024, 2025):
        say(f"{y}: households; non-financial corporations; financial corporations; government; total (% of GDP)",
            "; ".join(f"{nl.loc[y, s]:.1f}" for s in ["S14_S15", "S11", "S12", "S13", "S1"]))
    say("2024 in EUR bn: households; non-financial corporations; financial corporations; government; total",
        "; ".join(bn(sa("EU27_2020", s, "B9")[2024]) for s in ["S14_S15", "S11", "S12", "S13", "S1"]))
    w = nl.loc[2015:2024]
    say("2015-2024 average (% of GDP): households; non-financial corporations; financial corporations; government; total",
        "; ".join(f"{w[s].mean():.1f}" for s in ["S14_S15", "S11", "S12", "S13", "S1"]))
    nfc_pos = [y for y in range(2010, 2026) if nl.loc[y, "S11"] > 0]
    say("Years 2010-2025 in which non-financial corporations were net lenders", f"{len(nfc_pos)} of 16")
    say("EU27 household gross saving, EUR bn: 2022; 2024", f"{bn(sa('EU27_2020', 'S14_S15', 'B8G')[2022])}; "
        f"{bn(sa('EU27_2020', 'S14_S15', 'B8G')[2024])}")

    # ------------------------------------------------------------------ candidate measures of "savings sent abroad"
    section("Candidate measures of EU savings sent abroad, EUR bn (extra-EU unless stated)")
    meas = {
        "current account balance": lambda y: bop("CA", "BAL", y),
        "net lending (current + capital account)": lambda y: bop("CKA", "BAL", y),
        "net financial account": lambda y: bop("FA", "NET", y),
        "net portfolio investment": lambda y: bop("FA__P__F", "NET", y),
        "gross purchases of non-EU securities": lambda y: bop("FA__P__F", "ASS", y),
        "gross purchases of US securities": lambda y: bop("FA__P__F", "ASS", y, "US"),
        "of which by non-bank financial corporations": lambda y: float(PSA.get(("EXT_EU27_2020", "FA__P__F", "ASS", "S12M", y), np.nan)),
        "of which by non-financial corporations and households": lambda y: float(PSA.get(("EXT_EU27_2020", "FA__P__F", "ASS", "S1V", y), np.nan)),
        "non-EU purchases of EU securities": lambda y: bop("FA__P__F", "LIAB", y),
        "household net lending (all destinations)": lambda y: float(sa("EU27_2020", "S14_S15", "B9").get(y, np.nan)),
    }
    tab = pd.DataFrame({k: [f(y) / 1e3 for y in YEARS] for k, f in meas.items()}, index=YEARS)
    tab.to_csv(RES / "candidate_measures.csv", float_format="%.1f")
    for k in meas:
        s = tab[k]
        say(f"{k}: 2023; 2024; 2025; mean 2015-2025; mean 2015-2025 excluding 2022",
            f"{s[2023]:,.0f}; {s[2024]:,.0f}; {s[2025]:,.0f}; {s.mean():,.0f}; {s.drop(2022).mean():,.0f}")
    say("Share of gross portfolio purchases outside the EU that were US securities: 2023; 2024; 2025; 2015-2025 total",
        f"{tab.loc[2023, 'gross purchases of US securities'] / tab.loc[2023, 'gross purchases of non-EU securities']:.2f}; "
        f"{tab.loc[2024, 'gross purchases of US securities'] / tab.loc[2024, 'gross purchases of non-EU securities']:.2f}; "
        f"{tab.loc[2025, 'gross purchases of US securities'] / tab.loc[2025, 'gross purchases of non-EU securities']:.2f}; "
        f"{tab['gross purchases of US securities'].sum() / tab['gross purchases of non-EU securities'].sum():.2f}")
    sv = tab.loc[2017:2025]
    say("Share of gross purchases of non-EU securities by: non-bank financial corporations; non-financial corporations and "
        "households, 2017-2025 totals",
        f"{sv['of which by non-bank financial corporations'].sum() / sv['gross purchases of non-EU securities'].sum():.2f}; "
        f"{sv['of which by non-financial corporations and households'].sum() / sv['gross purchases of non-EU securities'].sum():.2f}")
    say("Years 2015-2025 with net portfolio outflows (positive net portfolio investment)",
        ", ".join(str(y) for y in YEARS if tab.loc[y, "net portfolio investment"] > 0))
    say("Years 2015-2025 with the current account surplus between 250 and 350 bn",
        ", ".join(str(y) for y in YEARS if 250 <= tab.loc[y, "current account balance"] <= 350))

    section("Net financial account by functional category, EUR bn (extra-EU)")
    cats = {"direct investment": "FA__D__F", "portfolio investment": "FA__P__F", "financial derivatives": "FA__F__F7",
            "other investment": "FA__O__F", "reserve assets": None}
    comp = pd.DataFrame({k: [bop(v, "NET", y) / 1e3 for y in YEARS] for k, v in cats.items() if k != "reserve assets"},
                        index=YEARS)
    comp["total"] = [bop("FA", "NET", y) / 1e3 for y in YEARS]
    # reserve assets are not published for the EU27 in bop_eu6_q; they are the residual of the total
    comp["reserve assets"] = comp["total"] - comp[["direct investment", "portfolio investment", "financial derivatives",
                                                    "other investment"]].sum(axis=1)
    comp.to_csv(RES / "financial_account_composition.csv", float_format="%.1f")
    for y in (2023, 2024, 2025):
        say(f"{y}: " + "; ".join(list(cats) + ["total"]) + " (reserve assets as residual)",
            "; ".join(f"{comp.loc[y, c]:,.0f}" for c in list(cats) + ["total"]))
    say("2015-2025 cumulative: " + "; ".join(list(cats) + ["total"]) + " (reserve assets as residual)",
        "; ".join(f"{comp[c].sum():,.0f}" for c in list(cats) + ["total"]))
    eo = sum(bop("EO", "NET", y) for y in YEARS) / 1e3
    say("Net errors and omissions as a share of the cumulative current account, 2015-2025",
        f"{eo / tab['current account balance'].sum():.2f}")
    cum = comp[["direct investment", "portfolio investment", "financial derivatives", "other investment"]].sum()
    say("2015-2025 cumulative shares of the net financial account: direct investment; portfolio; derivatives; other investment",
        "; ".join(f"{v / comp['total'].sum():.2f}" for v in cum))
    say("Portfolio 2015-2025 cumulative by instrument, net: equity and fund shares; debt securities",
        f"{sum(bop('FA__P__F5', 'NET', y) for y in YEARS) / 1e3:,.0f}; {sum(bop('FA__P__F3', 'NET', y) for y in YEARS) / 1e3:,.0f}")
    say("2015-2025 cumulative, EUR bn: current account; capital account; net errors and omissions; net financial account",
        f"{tab['current account balance'].sum():,.0f}; {sum(bop('KA', 'BAL', y) for y in YEARS) / 1e3:,.0f}; "
        f"{sum(bop('EO', 'NET', y) for y in YEARS) / 1e3:,.0f}; {comp['total'].sum():,.0f}")

    section("With the United States, EUR bn")
    for y in (2023, 2024, 2025):
        say(f"{y}: current account; goods and services; primary income; net direct investment; net other investment; "
            "gross purchases of US securities",
            f"{bop('CA', 'BAL', y, 'US') / 1e3:,.0f}; {bop('GS', 'BAL', y, 'US') / 1e3:,.0f}; {bop('IN1', 'BAL', y, 'US') / 1e3:,.0f}; "
            f"{bop('FA__D__F', 'NET', y, 'US') / 1e3:,.0f}; {bop('FA__O__F', 'NET', y, 'US') / 1e3:,.0f}; "
            f"{bop('FA__P__F', 'ASS', y, 'US') / 1e3:,.0f}")
    say("Gross purchases of US securities 2024: equity and fund shares; debt securities",
        f"{bop('FA__P__F5', 'ASS', 2024, 'US') / 1e3:,.0f}; {bop('FA__P__F3', 'ASS', 2024, 'US') / 1e3:,.0f}")
    say("Portfolio liabilities to US residents (bilateral)", "not compiled in bop_eu6_q (empty)")

    # ------------------------------------------------------------------ stocks
    section("International investment position against non-EU residents, EUR bn (excluding reserve assets)")
    items = {"direct investment": "FA__D__F", "portfolio equity": "FA__P__F5", "portfolio debt": "FA__P__F3",
             "derivatives": "FA__F__F7", "other investment": "FA__O__F"}
    pos = pd.DataFrame({k: IIP[(v, "N_LE")] / 1e3 for k, v in items.items()})
    pos["total excl. reserves"] = pos.sum(axis=1, min_count=5)
    pos.to_csv(RES / "net_position.csv", float_format="%.1f")
    for y in (2013, 2015, 2019, 2022, 2024, 2025):
        if y in pos.index:
            say(f"{y} net position: " + "; ".join(pos.columns), "; ".join(f"{pos.loc[y, c]:,.0f}" for c in pos.columns))
    say("2025 portfolio: assets; liabilities", f"{IIP.loc[2025, ('FA__P__F', 'A_LE')] / 1e3:,.0f}; "
        f"{IIP.loc[2025, ('FA__P__F', 'L_LE')] / 1e3:,.0f}")
    say("2025 portfolio equity: assets; liabilities", f"{IIP.loc[2025, ('FA__P__F5', 'A_LE')] / 1e3:,.0f}; "
        f"{IIP.loc[2025, ('FA__P__F5', 'L_LE')] / 1e3:,.0f}")
    say("2025 total net position excluding reserves, % of GDP", f"{pos.loc[2025, 'total excl. reserves'] * 1e3 / g[2025] * 100:.1f}")
    recon = []
    flow_code = {"direct investment": "FA__D__F", "portfolio equity": "FA__P__F5", "portfolio debt": "FA__P__F3",
                 "derivatives": "FA__F__F7", "other investment": "FA__O__F"}
    for k, v in flow_code.items():
        dpos = pos.loc[2025, k] - pos.loc[2015, k]
        flows = sum(bop(v, "NET", y) for y in range(2016, 2026)) / 1e3
        recon.append({"item": k, "change_in_position": dpos, "net_flows": flows, "other_changes": dpos - flows})
    recon = pd.DataFrame(recon)
    recon.loc[len(recon)] = ["total excl. reserves", recon.change_in_position.sum(), recon.net_flows.sum(), recon.other_changes.sum()]
    recon.to_csv(RES / "stock_flow_2015_2025.csv", index=False, float_format="%.1f")
    for _, r in recon.iterrows():
        say(f"2015-2025 {r['item']}: change in net position; cumulative net flows 2016-2025; other changes (valuation etc.)",
            f"{r.change_in_position:,.0f}; {r.net_flows:,.0f}; {r.other_changes:,.0f}")

    # ------------------------------------------------------------------ household balance sheets
    section("Financial assets of households")
    shares = []
    eu = HH.loc["EU27_2020"]
    us = OECD_HH.loc["USA"]
    for lab, frame, area, unit in [("EU27", eu, "EU27", "EUR bn"), ("US", us, "US", "USD bn")]:
        for y in (2005, 2013, 2019, 2022, 2024, 2025):
            if y not in frame.index:
                continue
            r = frame.loc[y]
            row = {"area": area, "year": y, "total": r["F"] / 1e3,
                   "deposits": r["F2"] / r["F"], "debt securities": r["F3"] / r["F"], "listed and other equity": r["F51"] / r["F"],
                   "investment fund shares": r["F52"] / r["F"], "insurance and pensions": r["F6"] / r["F"]}
            shares.append(row)
            say(f"{area} households {y}: total ({unit}); deposits; debt securities; equity; fund shares; insurance and pensions",
                f"{row['total']:,.0f}; {row['deposits']:.3f}; {row['debt securities']:.3f}; {row['listed and other equity']:.3f}; "
                f"{row['investment fund shares']:.3f}; {row['insurance and pensions']:.3f}")
    pd.DataFrame(shares).to_csv(RES / "household_assets.csv", index=False, float_format="%.4f")
    de_eurostat = HH.loc[("DE", 2024), "F"]
    de_oecd = OECD_HH.loc[("DEU", 2024), "F"]
    say("Cross-check, German households' financial assets 2024, EUR million: Eurostat; OECD", f"{de_eurostat:,.0f}; {de_oecd:,.0f}")
    for geo in ["DE", "FR", "IT", "ES", "NL", "PT"]:
        r = HH.loc[(geo, 2025)] if (geo, 2025) in HH.index else HH.loc[(geo, 2024)]
        say(f"{geo} households, latest year: deposits share; equity and fund shares share",
            f"{r['F2'] / r['F']:.3f}; {(r['F51'] + r['F52']) / r['F']:.3f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
