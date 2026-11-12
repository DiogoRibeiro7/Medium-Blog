"""Manufacturing labour costs in China against Mexico, Vietnam, the EU and the US, 1995-2022.

    python analysis.py

Reads the OECD snapshot in ``data/`` and the hand-kept IFR figures, writes tidy tables to
``results/*.csv`` and every number used in the article to ``results/analysis_summary.txt``.

Definitions (manufacturing, ISIC C, OECD Trade in Employment 2025 edition):
  compensation per person engaged = compensation of employees (USD, market exchange rates)
                                    / persons engaged (employees and self-employed)
  value added per person engaged  = published VALU_EMPN (USD)
  labour share                    = compensation of employees / value added
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"
LAST = 2022

NAMES = {
    "CHN": "China", "VNM": "Vietnam", "IND": "India", "IDN": "Indonesia", "THA": "Thailand", "MYS": "Malaysia",
    "PHL": "Philippines", "PAK": "Pakistan", "MMR": "Myanmar", "BGD": "Bangladesh", "KHM": "Cambodia",
    "TWN": "Taiwan", "KOR": "Korea", "JPN": "Japan", "MEX": "Mexico", "BRA": "Brazil", "TUR": "Türkiye",
    "MAR": "Morocco", "USA": "United States", "DEU": "Germany", "FRA": "France", "ITA": "Italy", "ESP": "Spain",
    "PRT": "Portugal", "POL": "Poland", "CZE": "Czechia", "SVK": "Slovakia", "HUN": "Hungary", "ROU": "Romania",
    "BGR": "Bulgaria", "GRC": "Greece", "W": "World",
}
LADDER = ["IND", "PHL", "VNM", "IDN", "MEX", "THA", "MYS", "BRA", "TUR", "ROU", "BGR", "CHN", "POL", "HUN",
          "CZE", "SVK", "PRT", "GRC", "ESP", "KOR", "ITA", "FRA", "JPN", "DEU", "USA"]
COMPARE = ["VNM", "IND", "IDN", "THA", "MEX", "TUR", "ROU", "BGR", "POL", "HUN", "CZE", "PRT", "ESP", "KOR",
           "DEU", "USA"]
INDUSTRIES = ["C10T12", "C13T15", "C16", "C17_18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
              "C27", "C28", "C29", "C30", "C31T33"]
LOW_WAGE_TEXTILE = ["VNM", "BGD", "KHM", "IND", "IDN", "PAK", "MMR"]

LINES: list[str] = []


def say(label: str, value) -> None:
    if isinstance(value, float):
        value = f"{value:.4f}"
    LINES.append(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")


TIM = pd.read_csv(DATA / "oecd_tim.csv", low_memory=False)
TIVA = pd.read_csv(DATA / "oecd_tiva.csv", low_memory=False)
FX = pd.read_csv(DATA / "oecd_fx.csv", low_memory=False)
IFR = pd.read_csv(DATA / "ifr_published.csv")
NBS = pd.read_csv(DATA / "nbs_published.csv").set_index(["series", "year"]).value_yuan


def tim(measure: str, unit: str, activity: str = "C") -> pd.DataFrame:
    x = TIM[(TIM.MEASURE == measure) & (TIM.UNIT_MEASURE == unit) & (TIM.ACTIVITY == activity)]
    return x.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="first")


EMP = {a: tim("EMPN", "PS", a) for a in INDUSTRIES + ["C"]}          # thousand persons engaged
LAB = {a: tim("LABR", "USD", a) for a in INDUSTRIES + ["C"]}         # million USD
VAPP = {a: tim("VALU_EMPN", "USD_PS", a) for a in INDUSTRIES + ["C"]}  # USD per person engaged
LSHARE = tim("LABR", "PT_VA", "C") / 100.0

COMP = LAB["C"] * 1e6 / (EMP["C"] * 1e3)  # USD per person engaged per year
VA_C = TIVA[TIVA.ACTIVITY == "C"].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")


def fx(transaction: str, area: str = "CHN") -> pd.Series:
    x = FX[(FX.TRANSACTION == transaction) & (FX.REF_AREA == area)]
    return x.set_index("TIME_PERIOD").OBS_VALUE.sort_index()


def main() -> None:
    RES.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- checks
    section("Checks")
    emp_sum = sum(EMP[a].loc[LAST, "CHN"] for a in INDUSTRIES)
    say("China 2022: sum of 17 industries' employment / manufacturing employment", float(emp_sum / EMP["C"].loc[LAST, "CHN"]))
    va_tim = VAPP["C"].loc[LAST, "CHN"] * EMP["C"].loc[LAST, "CHN"] / 1e3
    say("China 2022: value added from employment x VA per person / TiVA value added", float(va_tim / VA_C.loc[LAST, "CHN"]))
    ls_check = (LAB["C"] / (VAPP["C"] * EMP["C"] / 1e3) - LSHARE).abs().stack().max()
    say("Max abs difference between computed and published labour share (any country, year)", float(ls_check))

    # ---------------------------------------------------------------- wage ladder 2022
    section(f"Manufacturing ladder, {LAST} (USD at market exchange rates, per person engaged per year)")
    rows = []
    for c in LADDER:
        rows.append({"country": c, "name": NAMES[c], "compensation": COMP.loc[LAST, c], "value_added": VAPP["C"].loc[LAST, c],
                     "labour_share": LSHARE.loc[LAST, c], "employment_thousands": EMP["C"].loc[LAST, c],
                     "china_comp_ratio": COMP.loc[LAST, "CHN"] / COMP.loc[LAST, c]})
    ladder = pd.DataFrame(rows).sort_values("compensation")
    ladder.to_csv(RES / "ladder_2022.csv", index=False, float_format="%.4f")
    for r in ladder.itertuples():
        say(f"{r.country}: compensation per person; value added per person; labour share; China/country compensation",
            f"{r.compensation:,.0f}; {r.value_added:,.0f}; {r.labour_share:.3f}; {r.china_comp_ratio:.2f}")

    # ---------------------------------------------------------------- China relative to others over time
    section("China's compensation per person relative to other economies")
    rel = pd.DataFrame({c: COMP["CHN"] / COMP[c] for c in COMPARE})
    rel.index.name = "year"
    rel.to_csv(RES / "china_relative_compensation.csv", float_format="%.4f")
    for c in COMPARE:
        s = rel[c].dropna()
        above = s[s >= 1.0]
        cross = None
        for y in s.index:
            if (s.loc[y:] >= 1.0).all():
                cross = int(y)
                break
        say(f"China/{c} compensation per person: 1995; 2005; 2015; 2022; first year from which China stays above",
            f"{s.loc[1995]:.2f}; {s.loc[2005]:.2f}; {s.loc[2015]:.2f}; {s.loc[2022]:.2f}; {cross if cross else 'never'}")
        del above

    # ---------------------------------------------------------------- yuan and exchange rate
    section("China: dollar compensation growth split into yuan growth and exchange rate")
    exc = fx("EXC_A")      # yuan per US dollar, annual average
    ppp = fx("PPP_B1GQ")   # yuan per international dollar (GDP PPP)
    comp_usd = COMP["CHN"]
    comp_cny = comp_usd * exc.reindex(comp_usd.index)
    comp_ppp = comp_cny / ppp.reindex(comp_usd.index)
    tab = pd.DataFrame({"comp_usd": comp_usd, "yuan_per_usd": exc.reindex(comp_usd.index), "comp_cny": comp_cny,
                        "comp_ppp_usd": comp_ppp})
    tab.to_csv(RES / "china_yuan_decomposition.csv", float_format="%.4f")
    for a, b in ((1995, 2005), (2005, 2015), (2015, 2022), (1995, 2022)):
        g_usd = np.log(comp_usd.loc[b] / comp_usd.loc[a])
        g_cny = np.log(comp_cny.loc[b] / comp_cny.loc[a])
        g_fx = -np.log(exc.loc[b] / exc.loc[a])
        say(f"China {a}-{b}: log growth of USD compensation = yuan growth + exchange rate (log points); annual USD growth",
            f"{100 * g_usd:.1f} = {100 * g_cny:.1f} + {100 * g_fx:.1f}; {100 * (np.exp(g_usd / (b - a)) - 1):.1f}%")
    for y in (1995, 2005, 2015, 2022):
        say(f"China {y}: compensation per person USD; yuan; yuan per USD; at GDP PPP (international dollars)",
            f"{comp_usd.loc[y]:,.0f}; {comp_cny.loc[y]:,.0f}; {exc.loc[y]:.3f}; {comp_ppp.loc[y]:,.0f}")
    shares = []
    for a, b in ((1995, 2005), (2005, 2015), (2015, 2022), (1995, 2022)):
        shares.append(-np.log(exc.loc[b] / exc.loc[a]) / np.log(comp_usd.loc[b] / comp_usd.loc[a]))
    say("Exchange-rate share of China's log growth in USD compensation: 1995-2005; 2005-2015; 2015-2022; 1995-2022",
        "; ".join(f"{v:.3f}" for v in shares))
    mxn = fx("EXC_A", "MEX")
    m_usd = COMP["MEX"]
    g_usd = np.log(m_usd.loc[2022] / m_usd.loc[2015])
    g_fx = -np.log(mxn.loc[2022] / mxn.loc[2015])
    say("Mexico 2015-2022: log growth of USD compensation = peso growth + exchange rate (log points); pesos per USD 2015; 2022",
        f"{100 * g_usd:.1f} = {100 * (g_usd - g_fx):.1f} + {100 * g_fx:.1f}; {mxn.loc[2015]:.2f}; {mxn.loc[2022]:.2f}")
    say("Yuan per USD, 2023; 2024; 2025", f"{exc.loc[2023]:.3f}; {exc.loc[2024]:.3f}; {exc.loc[2025]:.3f}")

    # ---------------------------------------------------------------- China's jobs and output
    section("China manufacturing: employment, value added and value added per person")
    e = EMP["C"]["CHN"]
    peak = int(e.idxmax())
    say("China manufacturing employment: 1995; peak year and value; 2018; 2022 (millions)",
        f"{e.loc[1995] / 1e3:.1f}; {peak} {e.loc[peak] / 1e3:.1f}; {e.loc[2018] / 1e3:.1f}; {e.loc[2022] / 1e3:.1f}")
    say(f"Change in employment {peak}-2022 (%)", f"{100 * (e.loc[2022] / e.loc[peak] - 1):.1f}")
    va = VA_C["CHN"]
    say(f"China manufacturing value added, USD bn: 1995; {peak}; 2022", f"{va.loc[1995] / 1e3:,.0f}; {va.loc[peak] / 1e3:,.0f}; {va.loc[2022] / 1e3:,.0f}")
    say(f"Change in value added (USD) {peak}-2022 (%)", f"{100 * (va.loc[2022] / va.loc[peak] - 1):.1f}")
    v = VAPP["C"]["CHN"]
    say(f"China value added per person, USD: 1995; {peak}; 2022", f"{v.loc[1995]:,.0f}; {v.loc[peak]:,.0f}; {v.loc[2022]:,.0f}")
    say("China labour share of manufacturing value added: 1995; 2005; 2015; 2022",
        f"{LSHARE.loc[1995, 'CHN']:.3f}; {LSHARE.loc[2005, 'CHN']:.3f}; {LSHARE.loc[2015, 'CHN']:.3f}; {LSHARE.loc[2022, 'CHN']:.3f}")
    pd.DataFrame({"employment_thousands": e, "value_added_usd_m": va, "va_per_person_usd": v,
                  "compensation_per_person_usd": comp_usd, "labour_share": LSHARE["CHN"]}).to_csv(
        RES / "china_manufacturing.csv", float_format="%.4f")

    # ---------------------------------------------------------------- shift-share of value added per person
    section("Shift-share of value added per person engaged (USD), manufacturing industries")
    ss_rows = []
    for c in ["CHN", "VNM", "MEX", "POL", "PRT", "DEU", "USA"]:
        for a, b in ((1995, 2022), (2005, 2022)):
            emp_a = pd.Series({j: EMP[j].loc[a, c] for j in INDUSTRIES})
            emp_b = pd.Series({j: EMP[j].loc[b, c] for j in INDUSTRIES})
            p_a = pd.Series({j: VAPP[j].loc[a, c] for j in INDUSTRIES})
            p_b = pd.Series({j: VAPP[j].loc[b, c] for j in INDUSTRIES})
            s_a, s_b = emp_a / emp_a.sum(), emp_b / emp_b.sum()
            ok = p_a.notna() & p_b.notna()
            within = float((s_a[ok] * (p_b[ok] - p_a[ok])).sum())
            between = float(((s_b[ok] - s_a[ok]) * p_a[ok]).sum())
            inter = float(((s_b[ok] - s_a[ok]) * (p_b[ok] - p_a[ok])).sum())
            total = float((s_b[ok] * p_b[ok]).sum() - (s_a[ok] * p_a[ok]).sum())
            ss_rows.append({"country": c, "period": f"{a}-{b}", "total": total, "within": within, "between": between,
                            "interaction": inter, "between_share": between / total,
                            "textile_share_a": s_a["C13T15"], "textile_share_b": s_b["C13T15"],
                            "electronics_share_a": s_a["C26"], "electronics_share_b": s_b["C26"],
                            "vehicles_share_a": s_a["C29"], "vehicles_share_b": s_b["C29"]})
    ss = pd.DataFrame(ss_rows)
    ss.to_csv(RES / "shift_share.csv", index=False, float_format="%.4f")
    for r in ss.itertuples():
        say(f"{r.country} {r.period}: change in VA per person = within + between + interaction (USD); between share",
            f"{r.total:,.0f} = {r.within:,.0f} + {r.between:,.0f} + {r.interaction:,.0f}; {r.between_share:.3f}")
        say(f"{r.country} {r.period}: employment shares textiles; electronics; motor vehicles (start -> end)",
            f"{r.textile_share_a:.3f} -> {r.textile_share_b:.3f}; {r.electronics_share_a:.3f} -> {r.electronics_share_b:.3f}; "
            f"{r.vehicles_share_a:.3f} -> {r.vehicles_share_b:.3f}")

    # ---------------------------------------------------------------- world shares by industry
    section("Shares of world value added (USD at market exchange rates)")
    sh_rows = []
    for act, label in (("C", "manufacturing"), ("C13T15", "textiles, apparel and leather"),
                       ("C26", "computers and electronics"), ("C29", "motor vehicles")):
        p = TIVA[TIVA.ACTIVITY == act].pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        for y in p.index:
            sh_rows.append({"industry": label, "year": int(y), "china": p.loc[y, "CHN"] / p.loc[y, "W"],
                            "low_wage_asia": p.loc[y, LOW_WAGE_TEXTILE].sum() / p.loc[y, "W"],
                            "eu27": p.loc[y, "EU27_2020"] / p.loc[y, "W"], "usa": p.loc[y, "USA"] / p.loc[y, "W"]})
        for y in (1995, 2005, 2015, 2022):
            say(f"{label} {y}: China; Vietnam+Bangladesh+Cambodia+India+Indonesia+Pakistan+Myanmar; EU27; US (share of world VA)",
                f"{p.loc[y, 'CHN'] / p.loc[y, 'W']:.3f}; {p.loc[y, LOW_WAGE_TEXTILE].sum() / p.loc[y, 'W']:.3f}; "
                f"{p.loc[y, 'EU27_2020'] / p.loc[y, 'W']:.3f}; {p.loc[y, 'USA'] / p.loc[y, 'W']:.3f}")
        peak_y = int(p["CHN"].div(p["W"]).idxmax())
        say(f"{label}: year of China's highest world share and the share", f"{peak_y}; {p.loc[peak_y, 'CHN'] / p.loc[peak_y, 'W']:.3f}")
    pd.DataFrame(sh_rows).to_csv(RES / "world_shares.csv", index=False, float_format="%.4f")
    # textile employment
    te = EMP["C13T15"]
    say("Textiles, apparel and leather employment, thousands: China 1995; peak year and value; 2022",
        f"{te.loc[1995, 'CHN']:,.0f}; {int(te['CHN'].idxmax())} {te['CHN'].max():,.0f}; {te.loc[2022, 'CHN']:,.0f}")
    for c in ["VNM", "IND", "IDN", "PAK"]:
        say(f"Textiles employment, thousands, {c}: 1995; 2022", f"{te.loc[1995, c]:,.0f}; {te.loc[2022, c]:,.0f}")

    # ---------------------------------------------------------------- labour shares
    section("Labour share of manufacturing value added (compensation of employees / value added)")
    for c in ["CHN", "VNM", "MEX", "IND", "POL", "ROU", "PRT", "DEU", "USA", "KOR"]:
        say(f"{c}: 1995; 2005; 2015; 2022", f"{LSHARE.loc[1995, c]:.3f}; {LSHARE.loc[2005, c]:.3f}; {LSHARE.loc[2015, c]:.3f}; {LSHARE.loc[2022, c]:.3f}")

    # ---------------------------------------------------------------- growth 2015-2022 and the wage-cut arithmetic
    section("Growth 2015-2022 (USD, market exchange rates) and the arithmetic of matching China's labour share")
    gr = []
    for c in ["CHN", "VNM", "MEX", "IND", "ROU", "POL", "PRT", "ESP", "ITA", "DEU", "USA"]:
        g_comp = (COMP.loc[2022, c] / COMP.loc[2015, c]) ** (1 / 7) - 1
        g_va = (VAPP["C"].loc[2022, c] / VAPP["C"].loc[2015, c]) ** (1 / 7) - 1
        cut = 1 - LSHARE.loc[LAST, "CHN"] / LSHARE.loc[LAST, c]
        gr.append({"country": c, "comp_growth": g_comp, "va_growth": g_va, "cut_to_china_labour_share": cut})
        say(f"{c} 2015-2022 annual growth: compensation per person; VA per person; wage cut to reach China's 2022 labour share",
            f"{100 * g_comp:.1f}%; {100 * g_va:.1f}%; {100 * cut:.1f}%")
    gr = pd.DataFrame(gr).set_index("country")
    gr.to_csv(RES / "growth_2015_2022.csv", float_format="%.4f")
    for c in ["PRT", "POL", "ESP"]:
        gap = np.log(COMP.loc[LAST, c] / COMP.loc[LAST, "CHN"])
        diff = np.log(1 + gr.loc["CHN", "comp_growth"]) - np.log(1 + gr.loc[c, "comp_growth"])
        say(f"Years until China's compensation per person equals {c}'s if 2015-2022 growth rates continued (arithmetic, not a forecast)",
            f"{gap / diff:.1f}" if diff > 0 else "never")

    # ---------------------------------------------------------------- IFR robots
    section("Published IFR robot statistics (not reproduced; see data/ifr_sources.csv)")
    dens = IFR[IFR.metric == "robot_density"]
    for c in ["CHN", "KOR", "JPN", "DEU", "USA", "MEX", "WORLD"]:
        s = dens[dens.country == c].sort_values("year")
        say(f"{c} robot density by year", "; ".join(f"{int(y)} {int(v)}" for y, v in zip(s.year, s.value)))
    stock = float(IFR[(IFR.country == "CHN") & (IFR.metric == "operational_stock")].value.iloc[0])
    d_old = float(IFR[(IFR.country == "CHN") & (IFR.metric == "robot_density") & (IFR.year == 2024)].value.iloc[0])
    d_new = float(IFR[(IFR.country == "CHN") & (IFR.metric == "robot_density_revised")].value.iloc[0])
    say("China 2024 implied manufacturing employees (millions): stock/density before; after the revision",
        f"{stock / d_old * 1e4 / 1e6:.1f}; {stock / d_new * 1e4 / 1e6:.1f}")
    say("OECD persons engaged in China's manufacturing, 2022 (millions)", f"{EMP['C'].loc[2022, 'CHN'] / 1e3:.1f}")
    oecd = EMP["C"].loc[2022, "CHN"] * 1e3
    say("IFR implied employees / OECD persons engaged 2022: before; after the revision",
        f"{stock / d_old * 1e4 / oecd:.2f}; {stock / d_new * 1e4 / oecd:.2f}")
    inst = IFR[IFR.metric == "installations"].set_index("country").value
    say("China's share of world robot installations, 2024", f"{inst['CHN'] / inst['WORLD']:.3f}")
    say("China installations / US installations, 2024", f"{inst['CHN'] / inst['USA']:.1f}")
    cd = dens[dens.country == "CHN"].set_index("year").value
    say("China robot density growth 2015-2023 (multiple; annual %)", f"{cd[2023] / cd[2015]:.1f}; {100 * ((cd[2023] / cd[2015]) ** (1 / 8) - 1):.1f}%")

    yoy = float(IFR[(IFR.country == "CHN") & (IFR.metric == "robot_density_revised_yoy_growth")].value.iloc[0])
    say("China robot density on the revised basis, 2023 (implied by the stated 17% increase); 2024",
        f"{d_new / (1 + yoy):.0f}; {d_new:.0f}")

    # ---------------------------------------------------------------- NBS wages after 2022
    section("NBS average annual wages after the OECD data end (published, yuan; USD at the OECD average exchange rate)")
    exc = fx("EXC_A")
    for s in ["urban_nonprivate_manufacturing", "urban_private_manufacturing", "urban_nonprivate_all", "urban_private_all"]:
        v24, v25 = NBS[(s, 2024)], NBS[(s, 2025)]
        say(f"{s}: 2024 yuan; 2025 yuan; nominal growth; 2025 USD", f"{v24:,.0f}; {v25:,.0f}; {100 * (v25 / v24 - 1):.1f}%; "
            f"{v25 / exc.loc[2025]:,.0f}")
    v = NBS[("production_workers_above_designated_size", 2025)]
    say("Production, manufacturing and related personnel, enterprises above designated size, 2025: yuan; USD",
        f"{v:,.0f}; {v / exc.loc[2025]:,.0f}")
    say("Urban non-private average wage, all sectors, growth 2022-2025 (yuan, %; annual %)",
        f"{100 * (NBS[('urban_nonprivate_all', 2025)] / NBS[('urban_nonprivate_all', 2022)] - 1):.1f}; "
        f"{100 * ((NBS[('urban_nonprivate_all', 2025)] / NBS[('urban_nonprivate_all', 2022)]) ** (1 / 3) - 1):.1f}")
    say("Yuan per USD change 2022-2025 (%)", f"{100 * (exc.loc[2025] / exc.loc[2022] - 1):.1f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).strip() + "\n", encoding="utf-8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
