"""Industry decomposition of the gap in operating surplus per hour between North-West Europe and the US.

    python analysis.py

Tests the second conjecture of Part 1 of the series: that the North-West's lower net operating
surplus per hour is concentrated in information and communication, finance and professional services,
and is small or reversed in manufacturing. Reads ``data/`` and writes ``results/*.csv`` and every
number used in the article to ``results/analysis_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"

NW = ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "NLD", "SWE"]
SECTIONS = ["A", "B", "C", "D_E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
NAMES = {"A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D_E": "Utilities", "F": "Construction",
         "G": "Trade", "H": "Transport", "I": "Accommodation and food", "J": "Information and communication",
         "K": "Finance and insurance", "L": "Real estate", "M": "Professional, scientific, technical",
         "N": "Administrative and support", "O": "Public administration", "P": "Education", "Q": "Health and social work",
         "R": "Arts and recreation", "S": "Other services", "T": "Households as employers"}
SUBS = {"J": ["J58T60", "J61", "J62_63"], "K": ["K64", "K65", "K66"], "M": ["M69T71", "M72", "M73T75"],
        "C": ["C21", "C26", "C29"]}
SUB_NAMES = {"J58T60": "Publishing and broadcasting", "J61": "Telecommunications", "J62_63": "IT and information services",
             "K64": "Financial services", "K65": "Insurance and pensions", "K66": "Auxiliary financial services",
             "M69T71": "Legal, accounting, head offices, consulting, engineering", "M72": "Scientific R&D",
             "M73T75": "Advertising, other professional", "C21": "Pharmaceuticals", "C26": "Computers and electronics",
             "C29": "Motor vehicles"}
GROUP_JKMN = ["J", "K", "M", "N"]
YEARS = [2019, 2021]

LINES: list[str] = []


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


STAN = pd.read_csv(DATA / "oecd_stan.csv", low_memory=False)
STAN["key"] = STAN.MEASURE + "_" + STAN.UNIT_MEASURE
P = STAN.pivot_table(index=["REF_AREA", "TIME_PERIOD", "ACTIVITY"], columns="key", values="OBS_VALUE")
PPP = pd.read_csv(DATA / "oecd_ppp.csv")
PPP = PPP[PPP.TRANSACTION == "PPP_B1GQ"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")


def col(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)


def accounts(c: str, y: int, acts: list[str], gross: bool = False) -> pd.DataFrame:
    """Income-side accounts by activity, converted to US dollars at GDP PPP (millions; hours in millions).

    LC  = compensation of employees + imputed labour income of the self-employed (employee pay per hour)
    CFC = consumption of fixed capital
    NR  = net operating surplus and mixed income minus the imputation (gross variant: before CFC)
    FCVA = value added at factor cost = compensation of employees + gross operating surplus and mixed income
    """
    X = P.loc[(c, y)].reindex(acts)
    sal_h, sal_ps = col(X, "SAL_H"), col(X, "SAL_PS")
    if c == "USA":
        self_h = col(X, "SELF_PS") * sal_h / sal_ps
    else:
        self_h = col(X, "EMP_H") - sal_h
    self_h = self_h.clip(lower=0)
    d1, gos = col(X, "D1_XDC"), col(X, "B2A3G_XDC")
    cfc = col(X, "P51C_XDC")
    nos = col(X, "B2A3N_XDC").fillna(gos - cfc)
    # households as employers (T) and any industry without a recorded surplus: no operating surplus
    nos = nos.where(gos.notna() | d1.isna(), 0.0)
    gos = gos.where(gos.notna() | d1.isna(), 0.0)
    imp = (d1 / sal_h * self_h).fillna(0)
    out = pd.DataFrame({"LC": d1 + imp, "CFC": cfc, "NR": (gos if gross else nos) - imp, "FCVA": d1 + gos,
                        "H": sal_h + self_h.fillna(0), "IMP": imp, "D1": d1, "GOS": gos})
    money = ["LC", "CFC", "NR", "FCVA", "IMP", "D1", "GOS"]
    out[money] = out[money] / PPP.loc[y, c]
    return out


def group(y: int, acts: list[str], gross: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    us = accounts("USA", y, acts, gross)
    nw = sum(accounts(c, y, acts, gross).fillna(0) for c in NW)
    return nw, us


def total_hours(c: str, y: int) -> float:
    return float(P.loc[(c, y, "_T"), "EMP_H"])


def main() -> None:
    RES.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ checks
    section("Checks")
    for y in YEARS:
        nw, us = group(y, SECTIONS + ["_T"])
        for lab, fr in [("NW", nw), ("US", us)]:
            say(f"{y} {lab}: sum of 19 sections / total economy for compensation; gross operating surplus; hours",
                f"{fr.loc[SECTIONS, 'D1'].sum() / fr.loc['_T', 'D1']:.3f}; {fr.loc[SECTIONS, 'GOS'].sum() / fr.loc['_T', 'GOS']:.3f}; "
                f"{fr.loc[SECTIONS, 'H'].sum() / fr.loc['_T', 'H']:.3f}")
    dev = []
    for c in NW + ["USA"]:
        for y in YEARS:
            X = P.loc[(c, y)].reindex(SECTIONS)
            if {"B2A3N_XDC", "B2A3G_XDC", "P51C_XDC"} <= set(X.columns):
                r = (X["B2A3G_XDC"] - X["P51C_XDC"] - X["B2A3N_XDC"]).abs() / X["B2A3G_XDC"].abs()
                dev.append(r.max())
    say("Max |gross surplus - depreciation - net surplus| / gross surplus, any section, country, year", f"{np.nanmax(dev):.4f}")
    say("Total hours worked, millions, 2021: NW (8 countries); US", f"{sum(total_hours(c, 2021) for c in NW):,.0f}; "
        f"{total_hours('USA', 2021):,.0f}")
    say("GDP PPP 2021, national currency per USD: " + "; ".join(NW), "; ".join(f"{PPP.loc[2021, c]:.3f}" for c in NW))

    # ------------------------------------------------------------------ per-hour accounts
    section("Income side of an hour of factor-cost value added, USD at GDP PPP (sum of 19 sections / total hours)")
    contrib_tables = {}
    for y in YEARS:
        nw, us = group(y, SECTIONS)
        hn, hu = sum(total_hours(c, y) for c in NW), total_hours("USA", y)
        for k in ["FCVA", "LC", "CFC", "NR"]:
            say(f"{y} {k} per hour: NW; US; NW/US", f"{nw[k].sum() / hn:.1f}; {us[k].sum() / hu:.1f}; "
                f"{nw[k].sum() / hn / (us[k].sum() / hu):.2f}")
        gap = nw["NR"] / hn - us["NR"] / hu
        tot = gap.sum()
        say(f"{y} gap in net remainder per hour (NW - US), USD", f"{tot:.2f}")
        hs_n, hs_u = nw["H"] / nw["H"].sum(), us["H"] / us["H"].sum()
        r_n, r_u = nw["NR"] / nw["H"], us["NR"] / us["H"]
        scale_n, scale_u = nw["H"].sum() / hn, us["H"].sum() / hu
        structure = ((hs_n * scale_n - hs_u * scale_u) * (r_n + r_u) / 2).fillna(0)
        margin = ((r_n - r_u) * (hs_n * scale_n + hs_u * scale_u) / 2)
        margin = margin.fillna(gap - structure)  # sections with no hours in one of the two economies
        t = pd.DataFrame({"name": [NAMES[s] for s in SECTIONS], "contribution": gap, "share_of_gap": gap / tot,
                          "structure": structure, "margin": margin,
                          "nr_per_hour_in_industry_nw": r_n, "nr_per_hour_in_industry_us": r_u,
                          "hours_share_nw": hs_n, "hours_share_us": hs_u,
                          "profit_share_nw": nw["NR"] / nw["FCVA"], "profit_share_us": us["NR"] / us["FCVA"],
                          "va_share_nw": nw["FCVA"] / nw["FCVA"].sum(), "va_share_us": us["FCVA"] / us["FCVA"].sum()})
        t.to_csv(RES / f"contributions_{y}.csv", float_format="%.4f")
        contrib_tables[y] = t
        order = t.sort_values("contribution").index
        say(f"{y} contributions to the gap, USD per hour (share of gap), largest first",
            "; ".join(f"{s} {t.loc[s, 'contribution']:.2f} ({t.loc[s, 'share_of_gap']:.2f})" for s in order[:10]))
        say(f"{y} sections with a positive contribution (NW higher)",
            ", ".join(f"{s} {t.loc[s, 'contribution']:.2f}" for s in SECTIONS if t.loc[s, "contribution"] > 0) or "none")
        say(f"{y} share of the gap: J+K+M+N; J+K+M; real estate L; manufacturing C; all other sections",
            f"{t.loc[GROUP_JKMN, 'share_of_gap'].sum():.2f}; {t.loc[['J', 'K', 'M'], 'share_of_gap'].sum():.2f}; "
            f"{t.loc['L', 'share_of_gap']:.2f}; {t.loc['C', 'share_of_gap']:.2f}; "
            f"{1 - t.loc[GROUP_JKMN + ['L', 'C'], 'share_of_gap'].sum():.2f}")
        say(f"{y} structure and margin parts of the gap (sum over sections), USD per hour",
            f"{t.structure.sum():.2f}; {t.margin.sum():.2f}")
        for s in ["L", "K", "M", "J", "C", "N", "I", "G"]:
            say(f"{y} {s}: contribution = structure + margin (USD/h); hours share NW, US; NR per hour in industry NW, US",
                f"{t.loc[s, 'contribution']:.2f} = {t.loc[s, 'structure']:.2f} + {t.loc[s, 'margin']:.2f}; "
                f"{t.loc[s, 'hours_share_nw']:.3f}, {t.loc[s, 'hours_share_us']:.3f}; "
                f"{t.loc[s, 'nr_per_hour_in_industry_nw']:.1f}, {t.loc[s, 'nr_per_hour_in_industry_us']:.1f}")

    # ------------------------------------------------------------------ currency-free profit shares
    section("Net remainder as a share of factor-cost value added (no currency conversion needed)")
    t = contrib_tables[2021]
    nw, us = group(2021, SECTIONS)
    say("2021 aggregate profit share (19 sections): NW; US", f"{nw['NR'].sum() / nw['FCVA'].sum():.3f}; "
        f"{us['NR'].sum() / us['FCVA'].sum():.3f}")
    for s in ["C", "G", "I", "J", "K", "L", "M", "N", "F", "H", "D_E"]:
        say(f"2021 {s} profit share: NW; US | value-added share: NW; US",
            f"{t.loc[s, 'profit_share_nw']:.3f}; {t.loc[s, 'profit_share_us']:.3f} | {t.loc[s, 'va_share_nw']:.3f}; "
            f"{t.loc[s, 'va_share_us']:.3f}")
    w_bar = (t.va_share_nw + t.va_share_us) / 2
    s_bar = (t.profit_share_nw + t.profit_share_us) / 2
    within = (w_bar * (t.profit_share_nw - t.profit_share_us)).sum()
    between = (s_bar * (t.va_share_nw - t.va_share_us)).sum()
    say("2021 aggregate profit-share gap NW - US = within industries + between industries",
        f"{within + between:.3f} = {within:.3f} + {between:.3f}")
    lower = [s for s in SECTIONS if s != "T" and t.loc[s, "profit_share_nw"] < t.loc[s, "profit_share_us"]]
    say("2021 market sections (excluding O, P, Q, T) where the NW profit share is lower than the US",
        f"{len([s for s in lower if s not in ('O', 'P', 'Q')])} of 15 ({', '.join(s for s in lower if s not in ('O', 'P', 'Q'))})")

    # ------------------------------------------------------------------ owner-occupied housing
    section("Real estate: owner-occupied housing (L68A), gross operating surplus, 2021")
    rows = []
    for c in NW + ["USA"]:
        X = P.loc[(c, 2021)]
        l_gos = X.loc["L", "B2A3G_XDC"]
        a = X.loc["L68A"] if "L68A" in X.index else None
        a_gos = np.nan
        basis = ""
        if a is not None:
            if not np.isnan(a.get("B2A3G_XDC", np.nan)):
                a_gos, basis = a["B2A3G_XDC"], "B2A3G"
            elif not np.isnan(a.get("B1G_XDC", np.nan)):
                a_gos, basis = a["B1G_XDC"], "B1G"
        rows.append({"country": c, "l_gos": l_gos / PPP.loc[2021, c], "l68a_gos": a_gos / PPP.loc[2021, c], "basis": basis,
                     "hours": total_hours(c, 2021)})
    oo = pd.DataFrame(rows).set_index("country")
    oo.to_csv(RES / "owner_occupied_housing_2021.csv", float_format="%.2f")
    for c in NW + ["USA"]:
        say(f"{c}: owner-occupied housing share of real estate gross operating surplus (basis)",
            f"{oo.loc[c, 'l68a_gos'] / oo.loc[c, 'l_gos']:.2f} ({oo.loc[c, 'basis']})")
    hn = oo.loc[NW, "hours"].sum()
    nw_oo, us_oo = oo.loc[NW, "l68a_gos"].sum() / hn, oo.loc["USA", "l68a_gos"] / oo.loc["USA", "hours"]
    nw_l, us_l = oo.loc[NW, "l_gos"].sum() / hn, oo.loc["USA", "l_gos"] / oo.loc["USA", "hours"]
    say("Real estate gross operating surplus per hour of the whole economy, USD: NW; US; gap",
        f"{nw_l:.2f}; {us_l:.2f}; {nw_l - us_l:.2f}")
    say("  of which owner-occupied housing: NW; US; gap; share of the real estate gap",
        f"{nw_oo:.2f}; {us_oo:.2f}; {nw_oo - us_oo:.2f}; {(nw_oo - us_oo) / (nw_l - us_l):.2f}")

    # ------------------------------------------------------------------ sub-industries
    section("Sub-industries: net operating surplus and mixed income per hour of the whole economy, USD, 2021 "
            "(no self-employment imputation)")
    subs = [x for v in SUBS.values() for x in v]
    nw_s, us_s = group(2021, subs + list(SUBS))
    hn, hu = sum(total_hours(c, 2021) for c in NW), total_hours("USA", 2021)
    rows = []
    for parent, kids in SUBS.items():
        nos_n = (nw_s["NR"] + nw_s["IMP"]) / hn
        nos_u = (us_s["NR"] + us_s["IMP"]) / hu
        for k in kids + [parent]:
            rows.append({"parent": parent, "industry": k, "nw": nos_n[k], "us": nos_u[k], "gap": nos_n[k] - nos_u[k]})
        covered = sum(nos_n[k] - nos_u[k] for k in kids)
        say(f"{parent}: gap in net operating surplus per hour (NW - US); sum of the three sub-industries",
            f"{nos_n[parent] - nos_u[parent]:.2f}; {covered:.2f}")
        for k in kids:
            say(f"  {k} {SUB_NAMES[k]}: NW; US; gap", f"{nos_n[k]:.2f}; {nos_u[k]:.2f}; {nos_n[k] - nos_u[k]:.2f}")
    pd.DataFrame(rows).to_csv(RES / "subindustries_2021.csv", index=False, float_format="%.4f")
    missing = {c: [k for k in subs if (c, 2021, k) not in P.index or np.isnan(P.loc[(c, 2021, k)].get("B2A3G_XDC", np.nan))]
               for c in NW}
    say("Sub-industries missing gross operating surplus in 2021, by country",
        "; ".join(f"{c}: {', '.join(v) if v else 'none'}" for c, v in missing.items()))

    # ------------------------------------------------------------------ countries
    section("Each North-West country against the US, 2021, USD per hour")
    rows = []
    hu = total_hours("USA", 2021)
    us = accounts("USA", 2021, SECTIONS)
    for c in NW:
        a = accounts(c, 2021, SECTIONS).fillna(0)
        h = total_hours(c, 2021)
        g = a["NR"] / h - us["NR"] / hu
        row = {"country": c, "nr_per_hour": a["NR"].sum() / h, "lc_per_hour": a["LC"].sum() / h, "gap": g.sum(),
               "L": g["L"], "K": g["K"], "J_M_N": g[["J", "M", "N"]].sum(), "C": g["C"],
               "other": g.sum() - g[["L", "K", "J", "M", "N", "C"]].sum()}
        rows.append(row)
    ct = pd.DataFrame(rows).set_index("country")
    ct.to_csv(RES / "countries_2021.csv", float_format="%.6f")
    say("US 2021: net remainder per hour; labour income per hour", f"{us['NR'].sum() / hu:.1f}; {us['LC'].sum() / hu:.1f}")
    for c in NW:
        r = ct.loc[c]
        say(f"{c}: NR per hour; LC per hour; gap; contributions L; K; J+M+N; C; other",
            f"{r.nr_per_hour:.1f}; {r.lc_per_hour:.1f}; {r.gap:.2f}; {r.L:.2f}; {r.K:.2f}; {r.J_M_N:.2f}; {r.C:.2f}; {r.other:.2f}")
    say("Countries where manufacturing contributes positively (NW country higher than US)",
        ", ".join(c for c in NW if ct.loc[c, "C"] > 0) or "none")

    # ------------------------------------------------------------------ gross basis
    section("Gross basis (before depreciation): the same decomposition with consumption of fixed capital left in")
    say("Latest year with US income components by industry in STAN 2025", "2021 (total economy only for 2022-2024)")
    for y in YEARS:
        nw, us = group(y, SECTIONS, gross=True)
        hn, hu = sum(total_hours(c, y) for c in NW), total_hours("USA", y)
        g = nw["NR"] / hn - us["NR"] / hu
        say(f"{y} gross remainder per hour: NW; US; gap", f"{nw['NR'].sum() / hn:.1f}; {us['NR'].sum() / hu:.1f}; {g.sum():.2f}")
        say(f"{y} shares of the gross gap: J+K+M+N; real estate L; manufacturing C",
            f"{g[GROUP_JKMN].sum() / g.sum():.2f}; {g['L'] / g.sum():.2f}; {g['C'] / g.sum():.2f}")
        pd.DataFrame({"gross_contribution": g}).to_csv(RES / f"gross_contributions_{y}.csv", float_format="%.4f")
        cf = nw["CFC"] / hn - us["CFC"] / hu
        say(f"{y} depreciation per hour, gap NW - US: total; real estate L; manufacturing C",
            f"{cf.sum():.2f}; {cf['L']:.2f}; {cf['C']:.2f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
