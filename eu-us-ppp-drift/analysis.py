"""Test the conjecture that the EU-US drift between price bases comes from how ICT is deflated.

    python analysis.py

Writes results/analysis_summary.txt (every number cited in the article) and the CSV files behind the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import drift as dr

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
LINES: list[str] = []
PERIODS = [(1995, 2005), (2005, 2019), (2019, 2024)]
MAIN = (2005, 2019)


def say(label: str, value) -> None:
    LINES.append(f"{label}: {value}")
    print(f"{label}: {value}")


def section(title: str) -> None:
    LINES.append("")
    LINES.append(f"== {title} ==")
    print(f"\n== {title} ==")


def pts(x: float) -> str:
    return f"{100 * x:.1f}"


def change(s: pd.Series, a: int, b: int) -> float:
    return float(s.loc[b] - s.loc[a])


def main() -> None:
    d = dr.load()
    G = dr.EU_GROUP

    # ------------------------------------------------------------------ the drift
    section("The drift, EU group against the United States (log points x 100; 2020 = 0)")
    say("EU group (members)", f"{len(G)}: " + " ".join(G))
    paths = {}
    for agg in dr.AGGREGATES:
        paths[agg] = dr.group_drift(d, agg, G)
    paths = pd.DataFrame(paths)
    paths.index.name = "year"
    out = paths.copy()
    out["HFCE_Japan_vs_US"] = dr.country_drift(d, "HFCE")["JPN"]
    out.mul(100).to_csv(RES / "drift_paths.csv", float_format="%.3f")
    for y in (1995, 2005, 2010, 2015, 2019, 2024):
        say(f"{y} ln drift GDP, AIC, HFCE", ", ".join(pts(paths.loc[y, a]) for a in dr.AGGREGATES))
    for y in (1995, 2005, 2010, 2015, 2019, 2020, 2024):
        say(f"{y} GDP current-PPP level / constant-PPP level", f"{np.exp(paths.loc[y, 'GDP']):.3f}")
    for a, b in PERIODS:
        say(f"Change in drift {a}-{b}: GDP, AIC, HFCE (points)",
            ", ".join(pts(change(paths[g], a, b)) for g in dr.AGGREGATES))
    gdp, hf = change(paths["GDP"], *MAIN), change(paths["HFCE"], *MAIN)
    say("HFCE drift as share of GDP drift, 2005-2019", f"{hf / gdp:.2f}")
    aic = change(paths["AIC"], *MAIN)
    say("AIC drift as share of GDP drift, 2005-2019", f"{aic / gdp:.2f}")
    for agg in dr.AGGREGATES:
        parts = dr.group_parts(d, agg, G)
        rd, pp = change(parts["relative_deflator"], *MAIN), change(parts["ppp_index"], *MAIN)
        say(f"{agg} 2005-2019: change in EU/US relative deflator; in EU PPP (national currency per USD); drift",
            f"{pts(rd)}, {pts(pp)}, {pts(rd - pp)}")
    pg, pa_ = dr.group_parts(d, "GDP", G), dr.group_parts(d, "AIC", G)
    say("GDP minus AIC, 2005-2019: drift; relative deflator; PPP (points)",
        f"{pts(change(pg.drift, *MAIN) - change(pa_.drift, *MAIN))}; "
        f"{pts(change(pg.relative_deflator, *MAIN) - change(pa_.relative_deflator, *MAIN))}; "
        f"{pts(change(pg.ppp_index, *MAIN) - change(pa_.ppp_index, *MAIN))}")
    steps = paths.diff().mul(100)
    steps.to_csv(RES / "drift_annual_steps.csv", float_format="%.3f")
    sub = steps.loc[2006:2019]
    say("Years 2006-2019 in which the GDP drift rose / fell", f"{int((sub.GDP > 0).sum())} / {int((sub.GDP < 0).sum())}")
    top = sub.GDP.sort_values(ascending=False).head(4)
    say("Largest annual GDP-drift steps 2006-2019 (year: points)", "; ".join(f"{y}: {v:.1f}" for y, v in top.items()))
    say("Sum of the four largest steps as share of the 2005-2019 GDP drift", f"{top.sum() / (100 * gdp):.2f}")
    top = sub.HFCE.sort_values(ascending=False).head(4)
    say("Largest annual HFCE-drift steps 2006-2019 (year: points)", "; ".join(f"{y}: {v:.1f}" for y, v in top.items()))

    # ------------------------------------------------------------------ against Japan, and Japan against the US
    section("A second reference economy")
    for agg in dr.AGGREGATES:
        jp = dr.group_drift(d, agg, G, ref="JPN")
        us_jp = dr.country_drift(d, agg)["JPN"]
        say(f"{agg} drift 2005-2019: EU group vs Japan; Japan vs US (points)",
            f"{pts(change(jp, *MAIN))}; {pts(change(us_jp, *MAIN))}")

    # ------------------------------------------------------------------ cross-country
    section("Drift 2005-2019 against the United States, every economy with data (points)")
    rows = []
    for agg in dr.AGGREGATES:
        cd = dr.country_drift(d, agg)
        if MAIN[0] in cd.index and MAIN[1] in cd.index:
            ch = (cd.loc[MAIN[1]] - cd.loc[MAIN[0]]) * 100
            for c, v in ch.dropna().items():
                rows.append({"economy": c, "measure": agg, "drift_2005_2019": v})
        if 2024 in cd.index:
            ch = (cd.loc[2024] - cd.loc[2019]) * 100
            for c, v in ch.dropna().items():
                rows.append({"economy": c, "measure": agg, "drift_2019_2024": v})
    cc = pd.DataFrame(rows).groupby(["economy", "measure"]).first().reset_index()

    def region(c: str) -> str:
        if c in dr.EU:
            return "EU"
        if c in dr.NON_EU_EUROPE:
            return "Other Europe"
        if c in dr.NON_EUROPE_OECD:
            return "Non-European OECD"
        return "Other"
    cc["region"] = cc.economy.map(region)
    cc = cc[cc.region != "Other"]
    cc.to_csv(RES / "country_drift.csv", index=False, float_format="%.3f")
    for agg in dr.AGGREGATES:
        x = cc[(cc.measure == agg)].dropna(subset=["drift_2005_2019"])
        for reg in ("EU", "Other Europe", "Non-European OECD"):
            y = x[x.region == reg].drift_2005_2019
            say(f"{agg} {reg}: n, median, min, max",
                f"{len(y)}, {y.median():.1f}, {y.min():.1f} ({x.loc[y.idxmin(), 'economy']}), "
                f"{y.max():.1f} ({x.loc[y.idxmax(), 'economy']})")
    h = cc[cc.measure == "HFCE"].set_index("economy").drift_2005_2019
    say("HFCE drift 2005-2019, selected (DEU, FRA, ITA, ESP, NLD, SWE, POL, GBR, CHE, NOR, JPN, KOR, CAN, AUS, MEX)",
        ", ".join(f"{c} {h[c]:.1f}" for c in ["DEU", "FRA", "ITA", "ESP", "NLD", "SWE", "POL", "GBR", "CHE", "NOR",
                                              "JPN", "KOR", "CAN", "AUS", "MEX"] if c in h))
    eu_h = h.reindex(dr.EU).dropna()
    say("EU members with HFCE drift above 5 points / at or below 5", f"{int((eu_h > 5).sum())} / {int((eu_h <= 5).sum())}")
    say("EU members with HFCE drift below zero", ", ".join(f"{c} {v:.1f}" for c, v in eu_h[eu_h < 0].items()))
    for agg in ("HFCE", "GDP"):
        x = cc[(cc.measure == agg)].dropna(subset=["drift_2005_2019"])
        tot = ((x.drift_2005_2019 - x.drift_2005_2019.mean()) ** 2).sum()
        within = x.groupby("region").drift_2005_2019.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
        say(f"{agg}: share of cross-economy variance of the 2005-2019 drift explained by region (R2); "
            f"within-region SD (EU, Other Europe, Non-European OECD)",
            f"{1 - within / tot:.2f}; " + ", ".join(f"{x[x.region == r].drift_2005_2019.std():.1f}"
                                                  for r in ("EU", "Other Europe", "Non-European OECD")))
    ne = h.reindex(dr.NON_EUROPE_OECD).dropna()
    say("Non-European OECD economies with HFCE drift above 5 points", ", ".join(f"{c} {v:.1f}" for c, v in ne[ne > 5].items()) or "none")

    # ------------------------------------------------------------------ consumer side ICT
    section("ICT in household consumption (Table 5, domestic concept)")
    ci = dr.consumption_ict(d, G)
    have = [c for c in G if all(c in d["coicop_LR"][k].columns and pd.notna(d["coicop_LR"][k][c].get(2019))
                                for k in dr.ICT_CONSUMPTION)]
    hf = d["V"]["HFCE"]
    cover = (hf.loc[2019, have] / d["PPP"]["HFCE"].loc[2019, have]).sum() / (hf.loc[2019, G] / d["PPP"]["HFCE"].loc[2019, G]).sum()
    say("EU members reporting both ICT categories; share of group consumption at PPPs (2019)",
        f"{len(have)}; {cover:.2f}")
    say("Not reporting", ", ".join(sorted(set(G) - set(have))))
    rel_rows = []
    for k in dr.ICT_CONSUMPTION:
        p = ci["pieces"][k]
        s = p["share"].loc[MAIN[0] + 1:MAIN[1]]
        per = p["rel"].loc[MAIN[0] + 1:MAIN[1]].sum() * 100
        faster = [c for c in have if per[c] < per["USA"]]
        say(f"{k}: EU members whose relative price fell faster than in the US", f"{len(faster)} of {len(have)}")
        for c in ["USA", *have]:
            rel_rows.append({"economy": c, "category": k, "rel_price_2005_2019": per[c], "share": float(s[c].mean())})
    pd.DataFrame(rel_rows).to_csv(RES / "ict_consumer_prices.csv", index=False, float_format="%.4f")
    adj = dr.consumption_adjustments(d, G)
    cf = {"actual": paths["HFCE"]}
    for name, step in adj.items():
        cf[name] = dr.group_drift(d, "HFCE", G, adj=dr.cumulate(step))
    cf = pd.DataFrame(cf)
    cf.mul(100).to_csv(RES / "hfce_counterfactuals.csv", float_format="%.3f")
    base = change(cf["actual"], *MAIN)
    for name in adj:
        v = change(cf[name], *MAIN)
        say(f"HFCE drift 2005-2019 with ICT {name} (points; change vs actual)", f"{pts(v)} ({pts(v - base)})")
    end = 2022
    for name in ["actual", *adj]:
        say(f"HFCE drift 2019-{end}, {name} (points)", pts(change(cf[name], 2019, end)))

    # ------------------------------------------------------------------ production side ICT
    section("ICT-producing industries in value added (Table 6)")
    pa = dr.production_adjustments(d, G)
    vaT = d["va_V"]["_T"]
    missing = [c for c in G if c not in pa["rel"]["C26"].columns or pa["rel"]["C26"][c].loc[MAIN[0] + 1:MAIN[1]].isna().all()]
    say("EU members without C26 volumes (no C26 price in the ICT counterfactuals)", ", ".join(missing) or "none")
    gaps = []
    a, b = MAIN
    for k, label in (("CP08", "Communication (consumption)"), ("CP091", "Audio-visual and IT equipment (consumption)")):
        r = dr.relative_price_change(d["coicop_V"][k], d["coicop_LR"][k], d["coicop_V"]["_T"], d["coicop_LR"]["_T"],
                                     d["PPP"]["HFCE"], G, a, b)
        gaps.append({"item": label, "code": k, **{x: r[x] for x in ("us", "eu", "share_us", "share_eu", "n")}})
    for k, label in (("C26", "Computer, electronic and optical products (value added)"),
                     ("J", "Information and communication (value added)"),
                     ("J58", "Publishing, incl. software (value added)"),
                     ("J59_60", "Film, video and broadcasting (value added)"),
                     ("J61", "Telecommunications (value added)"),
                     ("J62_63", "IT and information services (value added)")):
        r = dr.relative_price_change(d["va_V"][k], d["va_LR"][k], vaT, d["va_LR"]["_T"], d["PPP"]["GDP"], G, a, b)
        gaps.append({"item": label, "code": k, **{x: r[x] for x in ("us", "eu", "share_us", "share_eu", "n")}})
    for k, label in (("N1132G", "ICT equipment (investment)"), ("N1173G", "Software and databases (investment)")):
        r = dr.relative_price_change(d["gf_V"][k], d["gf_LR"][k], d["V"]["GDP"], d["LR"]["GDP"], d["PPP"]["GDP"], G, a, b)
        gaps.append({"item": label, "code": k, **{x: r[x] for x in ("us", "eu", "share_us", "share_eu", "n")}})
        pc = r["per_country"].reindex(G).dropna()
        say(f"{k}: EU members whose relative price fell faster than in the US", f"{int((pc < r['us']).sum())} of {len(pc)}")
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(RES / "ict_price_gaps.csv", index=False, float_format="%.5f")
    for _, g in gaps.iterrows():
        say(f"{g.code} 2005-2019: price relative to its aggregate, US vs Europe (points); share US vs Europe (%); "
            f"EU members reporting", f"{100 * g.us:.0f} vs {100 * g.eu:.0f}; {100 * g.share_us:.1f} vs "
            f"{100 * g.share_eu:.1f}; {int(g.n)}")
    cfp = {"actual": paths["GDP"]}
    for name in ("excluded", "us_like_eu", "eu_like_us"):
        cfp[name] = dr.group_drift(d, "GDP", G, adj=dr.cumulate(pa[name]))
    cfp = pd.DataFrame(cfp)
    cfp.mul(100).to_csv(RES / "gdp_counterfactuals.csv", float_format="%.3f")
    base_g = change(cfp["actual"], *MAIN)
    for name in ("excluded", "us_like_eu", "eu_like_us"):
        v = change(cfp[name], *MAIN)
        say(f"GDP drift 2005-2019 with ICT industries {name} (points; change vs actual; share of drift removed)",
            f"{pts(v)} ({pts(v - base_g)}; {(base_g - v) / base_g:.2f})")
    for name in ("actual", "excluded"):
        say(f"GDP drift 1997-2005 / 2019-2024, {name} (points)",
            f"{pts(change(cfp[name], 1997, 2005))} / {pts(change(cfp[name], 2019, 2024))}")
    eff_early = change(cfp["actual"], 1997, 2005) - change(cfp["excluded"], 1997, 2005)
    eff_main = change(cfp["actual"], *MAIN) - change(cfp["excluded"], *MAIN)
    say("Drift attributable to ICT industries (exclusion variant): 1997-2005; 2005-2019 (points; points a year)",
        f"{pts(eff_early)} ({100 * eff_early / 8:.2f}); {pts(eff_main)} ({100 * eff_main / 14:.2f})")
    # both channels together, generous: production exclusion on GDP plus the consumer exclusion's change on HFCE
    say("Largest share of the 2005-2019 GDP drift attributable to ICT industries in any variant",
        f"{max((base_g - change(cfp[n], *MAIN)) / base_g for n in ('excluded', 'us_like_eu', 'eu_like_us')):.2f}")

    # ------------------------------------------------------------------ investment
    section("ICT investment (GFCF by asset)")
    ub = 0.0
    for _, g in gaps[gaps.code.isin(["N1132G", "N1173G"])].iterrows():
        ub += g.share_us * (g.eu - g.us)   # as if all ICT investment were produced at home, US priced like Europe
    say("Upper bound on the GDP drift from US-European differences in ICT investment prices, 2005-2019 (points)",
        f"{100 * ub:.1f}")
    say("ICT industries excluded plus the investment upper bound, as share of the 2005-2019 GDP drift",
        f"{(eff_main + ub) / base_g:.2f} ({pts(eff_main + ub)} of {pts(base_g)})")
    # the same adjustments read as differences in volume growth
    v, lr, ppp = d["V"]["GDP"], d["LR"]["GDP"], d["PPP"]["GDP"]
    con = (v.loc[dr.BASE] / ppp.loc[dr.BASE]) * (lr / lr.loc[dr.BASE])
    n = MAIN[1] - MAIN[0]
    g_eu = np.log(con[G].sum(axis=1).loc[MAIN[1]] / con[G].sum(axis=1).loc[MAIN[0]]) / n
    g_us = np.log(con["USA"].loc[MAIN[1]] / con["USA"].loc[MAIN[0]]) / n
    say("GDP volume growth 2005-2019, US vs EU group (% a year); gap (points a year)",
        f"{100 * g_us:.2f} vs {100 * g_eu:.2f}; {100 * (g_us - g_eu):.2f}")
    for name in ("excluded", "us_like_eu", "eu_like_us"):
        eff = (base_g - change(cfp[name], *MAIN)) / n
        say(f"ICT industries {name}: change in the US-EU gap in GDP volume growth (points a year)", f"{100 * eff:.2f}")
    say("Investment upper bound as a change in the growth gap (points a year)", f"{100 * ub / n:.2f}")
    for name in ("eu_like_us", "us_like_eu"):
        eff = (base_g - change(cfp[name], *MAIN)) / n
        say(f"Harmonised ({name}) plus investment upper bound: change in the growth gap (points a year; share of gap)",
            f"{100 * (eff + ub / n):.2f}; {(eff + ub / n) / (g_us - g_eu):.2f}")
        say(f"Harmonised ({name}) plus investment upper bound: share of the 2005-2019 GDP drift",
            f"{(n * eff + ub) / base_g:.2f}")

    # ------------------------------------------------------------------ cross-section inside the EU
    section("Inside the EU: does the drift line up with ICT? (2005-2019, members with data)")
    icp = pd.read_csv(RES / "ict_consumer_prices.csv")
    gap = icp.pivot_table(index="economy", columns="category", values="rel_price_2005_2019")
    shr = icp.pivot_table(index="economy", columns="category", values="share")
    # how much each country's own ICT consumer prices would move its deflator relative to the US path
    ict_effect = ((gap.sub(gap.loc["USA"], axis=1)) * shr).sum(axis=1)
    hd = cc[cc.measure == "HFCE"].set_index("economy").drift_2005_2019
    x = pd.DataFrame({"hfce_drift": hd, "ict_consumer_effect": ict_effect}).reindex(G).dropna()
    r = x.corr().iloc[0, 1]
    say("Correlation across EU members: HFCE drift vs own ICT consumer-price effect relative to US (n, r)",
        f"{len(x)}, {r:.2f}")
    say("Range of the ICT consumer-price effect across EU members (points)",
        f"{x.ict_consumer_effect.min():.1f} to {x.ict_consumer_effect.max():.1f}")
    s_ict = sum((d["va_V"][a] / vaT).loc[2005:2019].mean() for a in dr.ICT_INDUSTRIES) * 100
    gd = cc[cc.measure == "GDP"].set_index("economy").drift_2005_2019
    y = pd.DataFrame({"gdp_drift": gd, "ict_share": s_ict}).reindex(G).dropna()
    say("Correlation across EU members: GDP drift vs ICT-industry share of GVA (n, r)",
        f"{len(y)}, {y.corr().iloc[0, 1]:.2f}")
    say("ICT-industry share of GVA 2005-2019: US; EU min (economy); EU max (economy) (%)",
        f"{s_ict['USA']:.1f}; {y.ict_share.min():.1f} ({y.ict_share.idxmin()}); {y.ict_share.max():.1f} ({y.ict_share.idxmax()})")
    for c in ("FIN", "SWE", "IRL", "HUN"):
        if c in y.index:
            say(f"{c}: ICT share of GVA (%), GDP drift 2005-2019 (points)", f"{y.loc[c, 'ict_share']:.1f}, {y.loc[c, 'gdp_drift']:.1f}")
    pd.concat([x, y], axis=1).to_csv(RES / "eu_cross_section.csv", float_format="%.3f")

    # ------------------------------------------------------------------ cross-check of the PPPs
    section("Cross-check: Eurostat and OECD PPPs for the United States against the EU27")
    es = pd.read_csv(dr.DATA / "eurostat_ppp.csv")
    es = es[es.na_item == "PPP_EU27_2020"]
    for cat, code in (("GDP", "PPP_B1GQ"), ("E011", "PPP_P31S14"), ("A01", "PPP_P41")):
        e = es[(es.geo == "US") & (es.ppp_cat == cat)].set_index("time").value
        o = 1 / d["PPP"][{"GDP": "GDP", "E011": "HFCE", "A01": "AIC"}[cat]]["EU"]
        say(f"{cat}: change 2005-2019 in the US PPP against the EU27, Eurostat vs OECD (log points)",
            f"{100 * np.log(e.loc[2019] / e.loc[2005]):.1f} vs {100 * np.log(o.loc[2019] / o.loc[2005]):.1f}")

    (RES / "analysis_summary.txt").write_text("\n".join(LINES).lstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
