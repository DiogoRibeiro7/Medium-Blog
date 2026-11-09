"""How much did Germany invest, and what happened to its public capital stock?

    python investment.py

Net public investment is gross fixed capital formation (P.51G) minus consumption of fixed
capital (P.51C) of general government, as a share of GDP. A robustness measure adds investment
grants paid (D.92) to gross investment, because some public infrastructure (rail, for example) is
financed through grants to entities outside general government.

1. Averages by period (1995-2008, before the debt brake; 2011-2019, with the brake in force and
   before the pandemic; 2020-2024) for Germany, five low-debt peers (Netherlands, Austria,
   Finland, Denmark, Sweden) and France, Italy and Belgium as context; the change between the
   first two periods, for Germany against each peer (a difference-in-differences with the peer
   average and with each peer in turn).
2. The net public capital stock (net fixed assets of general government at current prices) as a
   share of GDP, in total and by asset (non-residential buildings, other structures, machinery,
   intellectual property).
3. Cumulative shortfalls: the net investment Germany would have made at the peers' average share
   of GDP, and the capital it would hold at the peers' average stock ratio or at its own 1995 ratio.
4. Net investment by German subsector (federal, Länder, municipalities, social security).
5. Gross investment by function (COFOG), 2011-2019, Germany against the peers.

Writes ``results/investment_*.csv`` and ``results/investment_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import NAMES, PEERS, PERIODS, RESULTS, SUBSECTORS, cofog, fixed_assets, gdp, gov, peer_mean, share, write

SHOW = ("DE",) + PEERS + ("FR", "IT", "BE")
ASSETS = {"N11N": "all fixed assets", "N1121N": "non-residential buildings", "N1122N": "other structures (roads, bridges, networks)", "N11MN": "machinery and equipment", "N117N": "intellectual property"}
FUNCTIONS = {
    "GF01": "general public services", "GF02": "defence", "GF03": "public order and safety", "GF04": "economic affairs",
    "GF0405": "of which transport", "GF05": "environmental protection", "GF06": "housing and community amenities",
    "GF07": "health", "GF08": "recreation, culture and religion", "GF09": "education", "GF10": "social protection",
}


def run() -> str:
    Y = gdp()
    net = share(gov("P51G") - gov("P51C"), Y)
    gross = share(gov("P51G"), Y)
    grants = share(gov("D92PAY"), Y)
    broad = gross + grants
    net_broad = net + grants
    lines = ["Shares of GDP (%), general government; peers = " + ", ".join(NAMES[p] for p in PEERS) + " (unweighted mean)"]
    rows = []
    for label, (a, b) in PERIODS.items():
        for name, frame in (("net", net), ("gross", gross), ("grants", grants), ("gross_plus_grants", broad), ("net_plus_grants", net_broad)):
            m = frame.loc[a:b, list(SHOW)].mean()
            for c, v in m.items():
                rows.append(dict(period=label, measure=name, geo=c, value=float(v)))
    P = pd.DataFrame(rows)
    P.to_csv(RESULTS / "investment_periods.csv", index=False)
    for measure in ("net", "gross", "grants", "net_plus_grants"):
        lines.append("")
        lines.append(f"{measure.replace('_', ' ')} investment, period averages:")
        for label in PERIODS:
            f = P[(P.period == label) & (P.measure == measure)].set_index("geo")["value"]
            lines.append(f"  {label}: " + ", ".join(f"{c} {f[c]:+.2f}" for c in SHOW) + f"; peer mean {f[list(PEERS)].mean():+.2f}; Germany minus peers {f['DE'] - f[list(PEERS)].mean():+.2f}")

    lines.append("")
    d = net["DE"].loc[1995:2025]
    lines.append(f"Germany, net investment: 2024 {d[2024]:+.2f}, 2025 {d[2025]:+.2f} (highest in 1995-2025: {d.idxmax()}, {d.max():+.2f}); peer mean 2024 {peer_mean(net).loc[2024]:+.2f}, 2025 {peer_mean(net).loc[2025]:+.2f}")
    lines.append(f"Germany, gross investment: 2019 {gross.loc[2019, 'DE']:.2f}, 2024 {gross.loc[2024, 'DE']:.2f}, 2025 {gross.loc[2025, 'DE']:.2f}; peer mean 2019 {peer_mean(gross).loc[2019]:.2f}, 2024 {peer_mean(gross).loc[2024]:.2f}, 2025 {peer_mean(gross).loc[2025]:.2f}")

    # difference-in-differences on net investment, before the brake vs brake in force
    pre, post = PERIODS["before the brake, 1995-2008"], PERIODS["brake in force, 2011-2019"]
    ch = net.loc[post[0] : post[1]].mean() - net.loc[pre[0] : pre[1]].mean()
    lines.append("")
    lines.append(f"Change in average net investment, {post[0]}-{post[1]} minus {pre[0]}-{pre[1]} (pp of GDP): " + ", ".join(f"{c} {ch[c]:+.2f}" for c in ("DE",) + PEERS))
    did = ch["DE"] - ch[list(PEERS)].mean()
    each = {p: ch["DE"] - ch[p] for p in PEERS}
    lines.append(f"  Germany minus peer mean: {did:+.2f}; Germany minus each peer: " + ", ".join(f"{p} {v:+.2f}" for p, v in each.items()))
    ranks = ch[["DE"] + list(PEERS)].rank()
    lines.append(f"  Germany's change ranks {int(ranks['DE'])} of {len(ranks)} (1 = largest fall)")

    # capital stock
    lines.append("")
    stock_rows = []
    for code, label in ASSETS.items():
        k = share(fixed_assets(code), Y)
        last = int(k[["DE"] + list(PEERS)].dropna().index.max())
        for c in ("DE",) + PEERS + ("FR", "BE"):
            for yr in (1995, 2008, 2019, last):
                if yr in k.index and c in k and pd.notna(k.loc[yr, c]):
                    stock_rows.append(dict(asset=code, geo=c, year=yr, value=float(k.loc[yr, c])))
        f = k.loc[[1995, 2008, 2019, last]]
        lines.append(f"Net public capital stock, {label}, % of GDP, 1995 / 2008 / 2019 / {last}:")
        for c in ("DE",) + PEERS:
            lines.append(f"  {c}: " + " / ".join(f"{f.loc[yr, c]:.1f}" for yr in f.index) + f"; change 1995-{last} {f.loc[last, c] - f.loc[1995, c]:+.1f}")
        pm = peer_mean(k)
        lines.append(f"  peer mean: " + " / ".join(f"{pm.loc[yr]:.1f}" for yr in f.index) + f"; Germany minus peer mean in {last}: {k.loc[last, 'DE'] - pm.loc[last]:+.1f}")
    pd.DataFrame(stock_rows).to_csv(RESULTS / "investment_stock.csv", index=False)

    # cumulative shortfalls, EUR bn
    lines.append("")
    gap = (peer_mean(net) - net["DE"]) / 100 * Y["DE"] / 1000
    for a, b in ((1995, 2024), (1995, 2008), (2011, 2019), (2020, 2024)):
        lines.append(f"Net investment Germany would have added at the peers' average share of GDP, {a}-{b}: EUR {gap.loc[a:b].sum():.0f} bn (current prices)")
    k = share(fixed_assets("N11N"), Y)
    last = int(k[["DE"] + list(PEERS)].dropna().index.max())
    y_last = Y.loc[last, "DE"] / 1000
    lines.append(f"Capital-stock gaps in {last} (GDP EUR {y_last:.0f} bn): to the peers' average ratio EUR {(peer_mean(k).loc[last] - k.loc[last, 'DE']) / 100 * y_last:.0f} bn; to Germany's own 1995 ratio EUR {(k.loc[1995, 'DE'] - k.loc[last, 'DE']) / 100 * y_last:.0f} bn; to its 2008 ratio EUR {(k.loc[2008, 'DE'] - k.loc[last, 'DE']) / 100 * y_last:.0f} bn")
    for code in ("N1121N", "N1122N"):
        kk = share(fixed_assets(code), Y)
        lines.append(f"  {ASSETS[code]}: Germany {kk.loc[1995, 'DE']:.1f}% of GDP in 1995, {kk.loc[last, 'DE']:.1f}% in {last}; the difference at {last} GDP is EUR {(kk.loc[1995, 'DE'] - kk.loc[last, 'DE']) / 100 * y_last:.0f} bn")

    # subsectors
    lines.append("")
    sub = {s: share(gov("P51G", s) - gov("P51C", s), Y)["DE"] for s in SUBSECTORS}
    S = pd.DataFrame(sub)
    S.to_csv(RESULTS / "investment_subsectors.csv")
    lines.append("Germany, net investment by subsector, % of GDP, period averages:")
    for label, (a, b) in PERIODS.items():
        lines.append(f"  {label}: " + ", ".join(f"{SUBSECTORS[s]} {S.loc[a:b, s].mean():+.2f}" for s in SUBSECTORS))
    lines.append(f"  2025: " + ", ".join(f"{SUBSECTORS[s]} {S.loc[2025, s]:+.2f}" for s in SUBSECTORS))
    mun = (gov("P51G", "S1313") - gov("P51C", "S1313"))["DE"] / 1000
    neg = mun[mun < 0]
    lines.append(f"Municipal net investment negative in {len(neg)} of {mun.loc[1995:2025].notna().sum()} years 1995-2025 ({min(neg.index)}-{max(neg.index)}); cumulative 2003-2019 EUR {mun.loc[2003:2019].sum():.1f} bn; cumulative 1995-2025 EUR {mun.loc[1995:2025].sum():.1f} bn")

    # functions
    lines.append("")
    rows = []
    for code, label in FUNCTIONS.items():
        f = share(cofog(code), Y).loc[2011:2019]
        de, pm = f["DE"].mean(), peer_mean(f).mean()
        rows.append(dict(code=code, function=label, germany=de, peers=pm, gap=de - pm, gap_bn=(de - pm) / 100 * Y.loc[2011:2019, "DE"].mean() / 1000))
    F = pd.DataFrame(rows)
    F.to_csv(RESULTS / "investment_functions.csv", index=False)
    lines.append("Gross investment by function, % of GDP, average 2011-2019: Germany / peer mean / gap (EUR bn a year at average 2011-2019 GDP)")
    for r in F.itertuples():
        lines.append(f"  {r.function}: {r.germany:.2f} / {r.peers:.2f} / {r.gap:+.2f} ({r.gap_bn:+.1f})")
    tot = share(cofog("TOTAL"), Y).loc[2011:2019]
    lines.append(f"  total: {tot['DE'].mean():.2f} / {peer_mean(tot).mean():.2f} / {tot['DE'].mean() - peer_mean(tot).mean():+.2f}")
    return write("investment_summary.txt", lines)


if __name__ == "__main__":
    print(run())
