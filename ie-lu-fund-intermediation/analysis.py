"""Test the conjecture that Irish and Luxembourg funds buy US securities for investors outside the EU.

    python analysis.py

Writes ``results/analysis_summary.txt`` (every number cited in the article) and the CSV tables the
figures read. All amounts are EUR billion unless stated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import hubs as H

OUT = Path("results")
W = list(range(H.FIRST, H.LAST + 1))
SHOW_YEARS = [2015, 2020, 2023, 2024, 2025]


def bn(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x / 1000:,.0f}"


def pct(x: float, nd: int = 0) -> str:
    return "nan" if pd.isna(x) else f"{100 * x:.{nd}f}%"


def series_line(name: str, s: pd.Series, fmt=bn) -> str:
    return f"{name}: " + ", ".join(f"{k} {fmt(v)}" for k, v in s.items())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = H.load("data")
    L: list[str] = ["# Fund intermediation through Ireland and Luxembourg: analysis summary",
                    "# EUR billion unless stated; flows are net purchases; window 2015-2025", ""]

    # ------------------------------------------------------------------ coverage
    L.append("## Coverage")
    miss_pos = {h: [g for g in H.holdings_of_hub(d, h).columns if pd.isna(H.position(d, g, h, H.F5, "A_LE").loc[2024])]
                for h in H.HUBS}
    L.append(f"Members whose 2024 holdings of IE equity and fund shares are unpublished: {', '.join(miss_pos['IE'])}")
    L.append(f"Members whose 2024 holdings of LU equity and fund shares are unpublished: {', '.join(miss_pos['LU'])}")
    de_last = H.flow(d, "DE", "IE", H.F5, "ASS").dropna().index.max()
    L.append(f"Last quarter of German purchases of IE securities by partner: {de_last}")
    fc = H.filled_cells(d)
    L.append(f"Member-years (positions) and member-quarters (flows) with non-EU holdings filled, 2015 on: "
             f"{fc['pos']} and {fc['flow']}")
    usq = H.us_purchases(d)
    usa = usq.apply(H.annual).loc[W]
    eu_us = H.annual(H.eu_flow(d, H.US, H.F, "ASS")).loc[W]
    diff = usa.sum(axis=1) - eu_us
    L.append(f"Sum of members' purchases of US securities minus EU27 aggregate, 2015-2025 total: {bn(diff.sum())}; "
             f"largest absolute annual difference: {bn(diff.abs().max())} ({diff.abs().idxmax()})")
    L.append("")

    # ------------------------------------------------------------------ who buys US securities
    L.append("## Who buys US securities (flows)")
    hub_us = usa[list(H.HUBS)].sum(axis=1)
    L.append(series_line("EU27 purchases of US securities", eu_us))
    L.append(series_line("Ireland", usa["IE"]))
    L.append(series_line("Luxembourg", usa["LU"]))
    L.append(series_line("Hub share of EU27", hub_us / eu_us, pct))
    tot = {"eu": eu_us.sum(), "hub": hub_us.sum(), "ie": usa.IE.sum(), "lu": usa.LU.sum()}
    L.append(f"2015-2025 totals: EU27 {bn(tot['eu'])}, IE {bn(tot['ie'])}, LU {bn(tot['lu'])}, hubs {bn(tot['hub'])}; "
             f"hub share {pct(tot['hub'] / tot['eu'])} (IE {pct(tot['ie'] / tot['eu'])}, LU {pct(tot['lu'] / tot['eu'])})")
    for item, label in ((H.F5, "equity and fund shares"), (H.F3, "debt securities")):
        u = H.us_purchases(d, item).apply(H.annual).loc[W]
        e = H.annual(H.eu_flow(d, H.US, item, "ASS")).loc[W]
        L.append(f"US {label}, 2015-2025: EU27 {bn(e.sum())}, hubs {bn(u[list(H.HUBS)].sum(axis=1).sum())} "
                 f"({pct(u[list(H.HUBS)].sum(axis=1).sum() / e.sum())})")
    hub_ext = H.annual(H.flow(d, "IE", H.EXT, H.F, "ASS") + H.flow(d, "LU", H.EXT, H.F, "ASS")).loc[W]
    eu_ext = H.annual(H.eu_flow(d, H.EXT, H.F, "ASS")).loc[W]
    L.append(f"Purchases of all non-EU securities, 2015-2025: EU27 {bn(eu_ext.sum())}, hubs {bn(hub_ext.sum())} "
             f"({pct(hub_ext.sum() / eu_ext.sum())})")
    L.append(f"Years in which the hubs bought more than half of the EU27's US securities: "
             f"{int(((hub_us / eu_us) > 0.5).sum())} of {len(W)} (both negative in 2022)")
    L.append("")

    L.append("## Who holds US securities (end of year)")
    usp = H.us_holdings(d).loc[H.years()]
    tot_p = usp.sum(axis=1)
    hub_p = usp[list(H.HUBS)].sum(axis=1)
    L.append(series_line("EU holdings of US securities (sum of members)", tot_p.loc[SHOW_YEARS]))
    L.append(series_line("Hub share", (hub_p / tot_p).loc[[2014] + SHOW_YEARS], pct))
    top = usp.loc[2024].sort_values(ascending=False)
    L.append("Largest holders at end-2024: " + ", ".join(f"{g} {bn(v)}" for g, v in top.head(8).items()))
    L.append(f"Hub share at end-2024 of EU holdings of US equity and fund shares: "
             f"{pct(H.us_holdings(d, H.F5).loc[2024, list(H.HUBS)].sum() / H.us_holdings(d, H.F5).loc[2024].sum())}; "
             f"of US debt securities: "
             f"{pct(H.us_holdings(d, H.F3).loc[2024, list(H.HUBS)].sum() / H.us_holdings(d, H.F3).loc[2024].sum())}")
    pd.DataFrame({"eu27_flow": eu_us, **{f"{g}_flow": usa[g] for g in H.HUBS},
                  "other_flow": usa.drop(columns=list(H.HUBS)).sum(axis=1)}).to_csv(OUT / "us_purchases_annual.csv")
    usp.loc[2024].sort_values(ascending=False).rename("us_holdings_2024").to_csv(OUT / "us_holdings_2024.csv")
    L.append("")

    # ------------------------------------------------------------------ who holds the hubs' securities
    L.append("## Who holds the hubs' equity and fund shares (end of year)")
    hp = {h: H.hub_positions(d, h) for h in H.HUBS}
    cp = H.combined_positions(d)
    for h in H.HUBS:
        p = hp[h]
        l52 = H.position(d, h, H.WRL, H.F52, "L_LE")
        l51 = H.position(d, h, H.WRL, H.F51, "L_LE")
        allf52 = H.fund_shares_outstanding(d, h)
        dom = allf52 - H._get(d.fin, (h, "S2", "ASS", "F52"), H.years())
        L.append(f"{h} end-2024: held by non-residents {bn(p.nonresident[2024])} (fund shares {bn(l52[2024])}, "
                 f"company shares {bn(l51[2024])}); fund shares held by anyone {bn(allf52[2024])}, "
                 f"of which by residents of {h} {bn(dom[2024])} ({pct(dom[2024] / allf52[2024])})")
        L.append(f"{h} EU holders end-2024: published {bn(p.eu_reported[2024])}, imputed {bn(p.eu_imputed[2024])}; "
                 f"outside the EU {bn(p.non_eu[2024])} ({pct(p.non_eu[2024] / p.nonresident[2024])} of non-resident "
                 f"holdings; bounds {pct(p.non_eu_low[2024] / p.nonresident[2024])}-"
                 f"{pct(p.non_eu_high[2024] / p.nonresident[2024])})")
        L.append(series_line(f"{h} share in the intra-EU holdings of members that publish it",
                             p.reporter_share.loc[[2014, 2019, 2024, 2025]], lambda x: pct(x, 1)))
        L.append(series_line(f"{h} share of company shares in members' holdings of {h} equity",
                             H.company_share_ratio(d, h).loc[[2014, 2024]], lambda x: pct(x, 1)))
    L.append(series_line("Both hubs, held outside the EU", cp.non_eu.loc[[2014] + SHOW_YEARS]))
    L.append(series_line("Both hubs, share held outside the EU (of non-resident holdings)",
                         (cp.non_eu / cp.nonresident).loc[[2014] + SHOW_YEARS], pct))
    L.append(series_line("  lower bound", (cp.non_eu_low / cp.nonresident).loc[[2014] + SHOW_YEARS], pct))
    L.append(series_line("  upper bound", (cp.non_eu_high / cp.nonresident).loc[[2014] + SHOW_YEARS], pct))
    eu_l5 = H.eu_position(d, H.EXT, H.F5, "L_LE")
    eu_l = H.eu_position(d, H.EXT, H.F, "L_LE")
    sh = (cp.non_eu / eu_l5).loc[H.years()]
    L.append(f"EU27 equity and fund shares held outside the EU, end-2024: {bn(eu_l5[2024])}; all portfolio "
             f"liabilities {bn(eu_l[2024])}")
    L.append(f"Hub equity and fund shares held outside the EU as a share of all EU27 equity and fund shares held "
             f"outside the EU: end-2024 {pct(sh[2024])}, range 2014-2025 {pct(sh.min())}-{pct(sh.max())}; "
             f"bounds end-2024 {pct(cp.non_eu_low[2024] / eu_l5[2024])}-{pct(cp.non_eu_high[2024] / eu_l5[2024])}")
    other25 = eu_l5[2024] - cp.non_eu[2024]
    L.append(f"Equity and fund shares of the other 25 members held outside the EU, end-2024 (EU27 minus hubs): "
             f"{bn(other25)}; hubs / other 25 = {cp.non_eu[2024] / other25:.2f}")
    omega = pd.DataFrame({h: H.non_eu_ownership(d, h) for h in H.HUBS})
    omega_hi = pd.DataFrame({h: H.non_eu_ownership(d, h, "high") for h in H.HUBS})
    omega_lo = H.ownership_low_joint(d, usp)
    for h in H.HUBS:
        L.append(series_line(f"Non-EU ownership of {h} fund shares (central)", omega[h].loc[[2014] + SHOW_YEARS], pct))
        lo_h = H.non_eu_ownership(d, h, "low")
        L.append(f"  {h} end-2024 bounds (this hub alone): {pct(lo_h[2024])}-{pct(omega_hi.loc[2024, h])}")
    non_ea_2020 = ["BG", "CZ", "DK", "HR", "HU", "PL", "RO", "SE"]
    for h in H.HUBS:
        held = H.holdings_of_hub(d, h)
        non_ea = held[[g for g in non_ea_2020 if g in held.columns]].sum(axis=1)
        r51 = H.company_share_ratio(d, h)
        allf52 = H.fund_shares_outstanding(d, h)
        dom = allf52 - H._get(d.fin, (h, "S2", "ASS", "F52"), H.years())
        ea = (hp[h].eu - non_ea) * (1 - r51)
        L.append(f"{h} end-2020 fund shares held by: other euro-area members {pct(ea[2020] / allf52[2020])}, "
                 f"non-euro EU members {pct(non_ea[2020] * (1 - r51[2020]) / allf52[2020])}, residents of {h} "
                 f"{pct(dom[2020] / allf52[2020])}, outside the EU {pct(omega.loc[2020, h])}")
    l51 = H.position(d, "IE", H.WRL, H.F51, "L_LE")
    eu51 = H.company_share_ratio(d, "IE") * hp["IE"].eu
    L.append(f"IE company shares held outside the EU, end-2024: {bn(l51[2024] - eu51[2024])} of {bn(l51[2024])} "
             f"({pct((l51[2024] - eu51[2024]) / l51[2024])})")
    pos_tab = pd.concat({h: hp[h] for h in H.HUBS}, axis=1)
    pos_tab.columns = [f"{a}_{b}" for a, b in pos_tab.columns]
    pos_tab.join(omega.add_prefix("omega_")).join(omega_lo.add_prefix("omega_low_")).join(
        omega_hi.add_prefix("omega_high_")).to_csv(OUT / "hub_positions.csv")
    L.append("")

    # ------------------------------------------------------------------ flows into the hubs
    L.append("## Who buys the hubs' equity and fund shares (flows)")
    cf_q = H.combined_flows(d)
    cf = cf_q.apply(H.annual).loc[W]
    hf = {h: H.hub_flows(d, h).apply(H.annual).loc[W] for h in H.HUBS}
    L.append(series_line("Non-residents, both hubs", cf.nonresident))
    L.append(series_line("Other EU members", cf.eu))
    L.append(series_line("Outside the EU (central)", cf.non_eu))
    L.append(series_line("  lower bound", cf.non_eu_low))
    L.append(series_line("  upper bound", cf.non_eu_high))
    s = cf.sum()
    L.append(f"2015-2025 totals: non-residents {bn(s.nonresident)}, EU {bn(s.eu)} (published {bn(s.eu_reported)}, "
             f"imputed {bn(s.eu_imputed)}), outside the EU {bn(s.non_eu)} (bounds {bn(s.non_eu_low)}-"
             f"{bn(s.non_eu_high)}); outside-EU share {pct(s.non_eu / s.nonresident)}")
    for h in H.HUBS:
        t = hf[h].sum()
        L.append(f"  {h}: non-residents {bn(t.nonresident)}, outside the EU {bn(t.non_eu)} "
                 f"({pct(t.non_eu / t.nonresident)})")
    eu_in = H.annual(H.eu_flow(d, H.EXT, H.F, "LIAB")).loc[W]
    eu_in5 = H.annual(H.eu_flow(d, H.EXT, H.F5, "LIAB")).loc[W]
    eu_in3 = H.annual(H.eu_flow(d, H.EXT, H.F3, "LIAB")).loc[W]
    L.append(series_line("Non-EU purchases of EU securities (EU27 portfolio liabilities)", eu_in))
    L.append(series_line("  of which equity and fund shares", eu_in5))
    L.append(series_line("Hub securities as a share of non-EU purchases of EU securities", cf.non_eu / eu_in, pct))
    gap = (eu_in - eu_in5 - eu_in3)
    L.append(f"EU27 liabilities: total minus equity minus debt, 2015-2025: {bn(gap.sum())} "
             f"(all in {int(gap.abs().idxmax())}: {bn(gap.loc[gap.abs().idxmax()])})")
    L.append(f"2015-2025: non-EU purchases of EU securities {bn(eu_in.sum())} (equity and fund shares "
             f"{bn(eu_in5.sum())}, debt {bn(eu_in3.sum())}); of hub equity and fund shares {bn(s.non_eu)} "
             f"({pct(s.non_eu / eu_in.sum())}; bounds {pct(s.non_eu_low / eu_in.sum())}-{pct(s.non_eu_high / eu_in.sum())})")
    rest = eu_in5 - cf.non_eu
    L.append(f"Implied non-EU net purchases of equity of the other 25 members, 2015-2025: {bn(rest.sum())} "
             f"(bounds {bn((eu_in5 - cf.non_eu_high).sum())} to {bn((eu_in5 - cf.non_eu_low).sum())}); "
             f"negative in {int((rest < 0).sum())} of {len(W)} years")
    L.append(f"Hubs' purchases of non-EU securities vs non-EU purchases of hub equity and fund shares, 2015-2025: "
             f"{bn(hub_ext.sum())} vs {bn(s.non_eu)} (ratio {hub_ext.sum() / s.non_eu:.2f})")
    hub_intra = H.annual(H.flow(d, "IE", H.EU, H.F, "ASS") + H.flow(d, "LU", H.EU, H.F, "ASS")).loc[W]
    hub_liab = H.annual(H.flow(d, "IE", H.WRL, H.F, "LIAB") + H.flow(d, "LU", H.WRL, H.F, "LIAB")).loc[W]
    L.append(f"Hubs' purchases of securities of other EU members vs EU purchases of hub equity and fund shares, "
             f"2015-2025: {bn(hub_intra.sum())} vs {bn(s.eu)}; hubs' purchases of all foreign securities "
             f"{bn(hub_ext.sum() + hub_intra.sum())}, non-resident purchases of all hub securities {bn(hub_liab.sum())}")
    pd.DataFrame({"eu_inflow": eu_in, "eu_inflow_equity": eu_in5, "hub_non_eu": cf.non_eu,
                  "hub_non_eu_low": cf.non_eu_low, "hub_non_eu_high": cf.non_eu_high, "hub_eu": cf.eu,
                  "hub_nonresident": cf.nonresident, "hub_ext_purchases": hub_ext}).to_csv(OUT / "hub_flows_annual.csv")
    L.append("")

    # ------------------------------------------------------------------ attribution
    L.append("## Attribution of the EU's purchases of US securities")
    usa_all = usq.apply(H.annual)
    att = H.attribution(usa_all, H.mid_year(omega)).loc[W]
    att_lo = H.attribution(usa_all, H.mid_year(H.ownership_low_joint(d, usa_all.reindex(H.years())))).loc[W]
    att_hi = H.attribution(usa_all, H.mid_year(omega_hi)).loc[W]
    other = usa.drop(columns=list(H.HUBS)).sum(axis=1)
    hub_eu = hub_us - att
    L.append(series_line("Bought by other members directly", other))
    L.append(series_line("Bought by the hubs for EU owners", hub_eu))
    L.append(series_line("Bought by the hubs for non-EU owners (central)", att))
    L.append(series_line("  lower bound", att_lo))
    L.append(series_line("  upper bound", att_hi))
    L.append(series_line("Non-EU-owned share of EU27 purchases", att / eu_us, pct))
    L.append(f"2015-2025: for non-EU owners {bn(att.sum())} of {bn(eu_us.sum())} = {pct(att.sum() / eu_us.sum())} "
             f"(bounds {pct(att_lo.sum() / eu_us.sum())}-{pct(att_hi.sum() / eu_us.sum())}); for EU owners via the hubs "
             f"{bn(hub_eu.sum())} ({pct(hub_eu.sum() / eu_us.sum())}); other members directly {bn(other.sum())} "
             f"({pct(other.sum() / eu_us.sum())}); EU-owned total {bn(eu_us.sum() - att.sum())} "
             f"({pct(1 - att.sum() / eu_us.sum())})")
    for y in (2023, 2024, 2025):
        L.append(f"  {y}: EU27 {bn(eu_us[y])}, for non-EU owners {bn(att[y])} ({pct(att[y] / eu_us[y])}), "
                 f"EU-owned {bn(eu_us[y] - att[y])} (bounds {bn(eu_us[y] - att_hi[y])}-{bn(eu_us[y] - att_lo[y])})")
    att_p = H.attribution(usp, omega)
    att_p_lo = H.attribution(usp, omega_lo)
    att_p_hi = H.attribution(usp, omega_hi)
    L.append(series_line("Non-EU-owned share of EU holdings of US securities (central)",
                         (att_p / tot_p).loc[[2014] + SHOW_YEARS], pct))
    rng = (att_p / tot_p).loc[H.years()]
    L.append(f"  range 2014-2025: {pct(rng.min())}-{pct(rng.max())}")
    L.append(f"  end-2024: {bn(att_p[2024])} of {bn(tot_p[2024])}; bounds {pct(att_p_lo[2024] / tot_p[2024])}-"
             f"{pct(att_p_hi[2024] / tot_p[2024])}")
    pd.DataFrame({"eu27": eu_us, "other_members": other, "hubs_eu_owned": hub_eu, "hubs_non_eu": att,
                  "hubs_non_eu_low": att_lo, "hubs_non_eu_high": att_hi}).to_csv(OUT / "attribution_annual.csv")
    L.append("")

    # ------------------------------------------------------------------ co-movement
    L.append("## Quarterly co-movement, 2015Q1-2025Q4 (44 quarters, EUR billion per quarter)")
    q = pd.DataFrame({
        "hub_us": usq[list(H.HUBS)].sum(axis=1), "other_us": usq.drop(columns=list(H.HUBS)).sum(axis=1),
        "hub_ext": H.flow(d, "IE", H.EXT, H.F, "ASS") + H.flow(d, "LU", H.EXT, H.F, "ASS"),
        "non_eu_in": cf_q.non_eu, "eu_in": cf_q.eu,
    }) / 1000
    L.append(f"Quarters used: {len(q.dropna())}; correlation between non-EU and EU purchases of hub securities "
             f"{q.non_eu_in.corr(q.eu_in):.2f}")
    for xc, yc in (("non_eu_in", "hub_us"), ("eu_in", "hub_us"), ("non_eu_in", "other_us")):
        L.append(f"Single-variable slope of {yc} on {xc} (Figure 5): {np.polyfit(q[xc], q[yc], 1)[0]:.2f}")
    for y in ("hub_us", "hub_ext", "other_us"):
        r1, r2 = q[y].corr(q.non_eu_in), q[y].corr(q.eu_in)
        b, se, r2_ = H.ols_hac(q[y].values, q[["non_eu_in", "eu_in"]].values)
        L.append(f"{y}: corr with non-EU purchases of hub securities {r1:.2f}, with EU purchases {r2:.2f}; "
                 f"OLS slopes {b[1]:.2f} (HAC se {se[1]:.2f}, t {b[1] / se[1]:.1f}) and {b[2]:.2f} "
                 f"(se {se[2]:.2f}, t {b[2] / se[2]:.1f}); R2 {r2_:.2f}")
    q.to_csv(OUT / "quarterly_flows.csv")
    L.append("")

    # ------------------------------------------------------------------ sensitivity
    L.append("## Sensitivity: share of EU27 purchases of US securities 2015-2025 made for non-EU owners")
    variants: dict[str, pd.DataFrame] = {
        "central": H.mid_year(omega),
        "no correction for company shares": H.mid_year(pd.DataFrame(
            {h: H.non_eu_ownership(d, h, correct_companies=False) for h in H.HUBS})),
        "owners outside the hub only": H.mid_year(pd.DataFrame(
            {h: H.non_eu_ownership(d, h, denominator="nonresident") for h in H.HUBS})),
        "ownership at start of year": omega.shift(1),
        "lower bound": H.mid_year(H.ownership_low_joint(d, usa_all.reindex(H.years()))),
        "upper bound": H.mid_year(omega_hi),
    }
    flow_share = {h: hf[h].non_eu.sum() / hf[h].nonresident.sum() for h in H.HUBS}
    variants["ownership of net purchases 2015-2025"] = pd.DataFrame(
        {h: pd.Series(flow_share[h], index=H.years()) for h in H.HUBS})
    rows = []
    for name, om in variants.items():
        a = H.attribution(usa_all, om).loc[W]
        rows.append({"variant": name, "share": a.sum() / eu_us.sum(), "amount": a.sum() / 1000})
        L.append(f"{name}: {pct(a.sum() / eu_us.sum(), 1)} ({bn(a.sum())})")
    rows.append({"variant": "holdings end-2024 (central)", "share": att_p[2024] / tot_p[2024],
                 "low": att_p_lo[2024] / tot_p[2024], "high": att_p_hi[2024] / tot_p[2024],
                 "amount": att_p[2024] / 1000})
    L.append(f"holdings end-2024: {pct(att_p[2024] / tot_p[2024], 1)}")
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    L.append(f"Share of hub fund-share purchases by non-residents made outside the EU, 2015-2025: "
             + ", ".join(f"{h} {pct(v)}" for h, v in flow_share.items()))
    L.append("")

    # ------------------------------------------------------------------ verdict
    L.append("## Criteria")
    crit = [
        ("hubs make at least a third of EU27 purchases of US securities", tot["hub"] / tot["eu"] >= 1 / 3),
        ("non-EU-owned share at least a fifth (central)", att.sum() / eu_us.sum() >= 0.2),
        ("non-EU-owned share at least a tenth (lower bound)", att_lo.sum() / eu_us.sum() >= 0.1),
        ("non-EU purchases of hub equity and fund shares cover the attributed US purchases", s.non_eu >= att.sum()),
        ("hub US purchases rise with non-EU purchases of hub securities (HAC t > 2)",
         (lambda b, se: b[1] / se[1] > 2)(*H.ols_hac(q.hub_us.values, q[["non_eu_in", "eu_in"]].values)[:2])),
        ("non-EU purchases of hub securities are at least a third of the EU27 gross inflow", s.non_eu / eu_in.sum() >= 1 / 3),
    ]
    for text, ok in crit:
        L.append(f"{'PASS' if ok else 'FAIL'}: {text}")

    (OUT / "analysis_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
