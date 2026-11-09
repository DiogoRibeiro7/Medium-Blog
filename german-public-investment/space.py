"""Was the debt brake binding, and what did Germany do with its fiscal room?

    python space.py

1. Balances. General government and subsector net lending (B.9) as a share of GDP, 2005-2025,
   against the debt brake's limits: a federal structural deficit of 0.35% of GDP from 2016 (after
   a transition from 2011) and a structural balance for the Länder from 2020. The brake applies to
   structural balances in budget accounting; national-accounts headline balances are used here,
   with a stated allowance for the cycle. The "unused room" is the federal headline balance minus
   the -0.35% limit, summed over 2014-2019.
2. The debt-stabilising primary balance, pb* = d(t-1) (i - g) / (100 + g), with the implicit
   interest rate i and nominal growth g; the excess of the actual primary balance over pb*,
   summed in percent of GDP and in euros.
3. A counterfactual debt path. Germany adds, each year from 2011 to 2019, the difference between
   the peers' average net investment and its own (as a share of GDP), financed by debt at the
   implicit rate; the extra debt compounds as d(t) = d(t-1) (1 + i/100) / (1 + g/100) + x(t).
   No growth dividend from the extra investment is assumed. Variants: the peer with the highest
   net investment (Sweden), and from 1996 (the first year with an implicit rate and growth).
4. Debt and investment across the peers: 2011-2019 averages of the debt ratio and net investment.

Writes ``results/space_*.csv`` and ``results/space_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import NAMES, PEERS, RESULTS, SUBSECTORS, debt, gdp, gov, peer_mean, share, write

FEDERAL_LIMIT = -0.35
CYCLE_ALLOWANCE = 0.5  # pp of GDP: a positive output gap of about 1% times a budget semi-elasticity of about 0.5


def counterfactual(extra: pd.Series, d: pd.Series, i: pd.Series, g: pd.Series, start: int, end: int) -> pd.Series:
    """Extra debt ratio from adding ``extra`` (% of GDP) each year from ``start`` to ``end``, carried to the last year."""
    gap, out = 0.0, {}
    for y in range(start, int(d.index.max()) + 1):
        x = float(extra.get(y, 0.0)) if y <= end else 0.0
        gap = gap * (1 + i[y] / 100) / (1 + g[y] / 100) + max(x, 0.0)
        out[y] = gap
    return pd.Series(out)


def run() -> str:
    Y = gdp()
    lines = []
    bal = pd.DataFrame({s: share(gov("B9", s), Y)["DE"] for s in ("S13",) + tuple(SUBSECTORS)})
    bal["primary"] = share(gov("B9") + gov("D41PAY"), Y)["DE"]
    D = debt()
    d = share(D, Y)
    avg = (D + D.shift()) / 2
    i = 100 * gov("D41PAY") / avg
    g = 100 * Y.pct_change()
    bal["debt"] = d["DE"]
    bal["implicit"] = i["DE"]
    bal["growth"] = g["DE"]
    bal["pb_star"] = d["DE"].shift() * (i["DE"] - g["DE"]) / (100 + g["DE"])
    bal["excess"] = bal["primary"] - bal["pb_star"]
    bal.loc[2005:2025].to_csv(RESULTS / "space_balances.csv")

    lines.append("Germany, net lending as % of GDP (headline, national accounts): general government / federal / Länder / municipalities; primary balance; debt")
    for y in range(2009, 2026):
        r = bal.loc[y]
        lines.append(f"  {y}: {r.S13:+.2f} / {r.S1311:+.2f} / {r.S1312:+.2f} / {r.S1313:+.2f}; primary {r.primary:+.2f}; debt {r.debt:.1f}")
    room = (bal.loc[2014:2019, "S1311"] - FEDERAL_LIMIT)
    y_bn = Y.loc[2014:2019, "DE"] / 1000
    lines.append("")
    lines.append("Federal headline balance above the -0.35% limit, 2014-2019: " + ", ".join(f"{y} {v:+.2f}" for y, v in room.items()) + f"; sum {room.sum():.2f} pp of GDP, EUR {(room * y_bn / 100).sum():.0f} bn")

    lines.append(f"  with a cyclical allowance of {CYCLE_ALLOWANCE} pp a year taken off the general-government surplus, general government still ran a surplus in " + ", ".join(str(y) for y in range(2014, 2020) if bal.loc[y, 'S13'] - CYCLE_ALLOWANCE > 0) + f" (headline surpluses {bal.loc[2014:2019, 'S13'].min():+.2f} to {bal.loc[2014:2019, 'S13'].max():+.2f})")
    lines.append(f"  general-government surplus summed 2014-2019: {bal.loc[2014:2019, 'S13'].sum():.2f} pp of GDP, EUR {(bal.loc[2014:2019, 'S13'] * y_bn / 100).sum():.0f} bn")
    lines.append("  Länder headline balance 2014-2019: " + ", ".join(f"{y} {bal.loc[y, 'S1312']:+.2f}" for y in range(2014, 2020)) + "; municipalities: " + ", ".join(f"{y} {bal.loc[y, 'S1313']:+.2f}" for y in range(2014, 2020)))
    lines.append(f"  after the brake returned: general government 2023 {bal.loc[2023, 'S13']:+.2f}, 2024 {bal.loc[2024, 'S13']:+.2f}, 2025 {bal.loc[2025, 'S13']:+.2f}; federal 2023 {bal.loc[2023, 'S1311']:+.2f}, 2024 {bal.loc[2024, 'S1311']:+.2f}, 2025 {bal.loc[2025, 'S1311']:+.2f}")

    lines.append("")
    ex = bal.loc[2011:2019]
    lines.append(f"Implicit interest rate minus nominal growth, Germany, 2011-2019 average: {(ex.implicit - ex.growth).mean():+.2f} pp (negative in {int((ex.implicit < ex.growth).sum())} of 9 years)")
    lines.append(f"Primary balance above the debt-stabilising level, 2011-2019: average {ex.excess.mean():+.2f} pp a year, sum {ex.excess.sum():.1f} pp of GDP, EUR {(ex.excess * Y.loc[2011:2019, 'DE'] / 100000).sum():.0f} bn; debt ratio {bal.loc[2010, 'debt']:.1f}% (2010) to {bal.loc[2019, 'debt']:.1f}% (2019)")

    # counterfactual debt
    net = share(gov("P51G") - gov("P51C"), Y)
    lines.append("")
    rows = []
    for label, extra, start, end in (
        ("peer mean, 2011-2019", peer_mean(net) - net["DE"], 2011, 2019),
        ("Sweden, 2011-2019", net["SE"] - net["DE"], 2011, 2019),
        ("peer mean, 1996-2019", peer_mean(net) - net["DE"], 1996, 2019),
    ):
        gap = counterfactual(extra, d["DE"], i["DE"], g["DE"], start, end)
        added = extra.loc[start:end].clip(lower=0)
        added_bn = (added * Y.loc[start:end, "DE"] / 100000).sum()
        lines.append(f"Counterfactual: extra net investment at {label} (sum {added.sum():.1f} pp of GDP, EUR {added_bn:.0f} bn): debt ratio 2019 {d.loc[2019, 'DE'] + gap[2019]:.1f}% instead of {d.loc[2019, 'DE']:.1f}%; 2025 {d.loc[2025, 'DE'] + gap[2025]:.1f}% instead of {d.loc[2025, 'DE']:.1f}%")
        for y, v in gap.items():
            rows.append(dict(variant=label, year=y, extra_debt=v, debt_actual=float(d.loc[y, "DE"]), debt_cf=float(d.loc[y, "DE"] + v)))
    pd.DataFrame(rows).to_csv(RESULTS / "space_counterfactual.csv", index=False)
    lines.append("  no growth dividend assumed; interest at Germany's implicit rate on the extra debt")

    # debt against investment across the peers
    lines.append("")
    lines.append("Debt ratio and net investment, 2011-2019 averages (% of GDP):")
    for c in ("DE",) + PEERS:
        lines.append(f"  {NAMES[c]}: debt {d.loc[2011:2019, c].mean():.1f}, net investment {net.loc[2011:2019, c].mean():+.2f}, overall balance {share(gov('B9'), Y).loc[2011:2019, c].mean():+.2f}")
    lines.append("  debt ratio 2025: " + ", ".join(f"{c} {d.loc[2025, c]:.1f}" for c in ("DE",) + PEERS))
    return write("space_summary.txt", lines)


if __name__ == "__main__":
    print(run())
