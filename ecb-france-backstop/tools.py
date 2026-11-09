"""What the ECB's past announcements did to spreads, in monthly averages.

    python tools.py

For each announcement the change in the monthly-average 10-year spread to Germany is measured
from the month before to two months after. Monthly averages blur the day of the announcement and
mix in other news, so these are descriptive; the event-study evidence on daily data (Altavilla,
Giannone and Lenza, 2016) is cited in the article.

Writes ``results/tools_events.csv`` and ``results/tools_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import NAMES, RESULTS, monthly_spreads, write

EVENTS = {
    "SMP (10 May 2010)": "2010-05",
    "Longer-term refinancing, three years (8 Dec 2011)": "2011-12",
    "'Whatever it takes' and OMT (26 Jul - 6 Sep 2012)": "2012-08",
    "PSPP announced (22 Jan 2015)": "2015-01",
    "PEPP (18 Mar 2020)": "2020-03",
    "Emergency meeting and TPI (15 Jun - 21 Jul 2022)": "2022-07",
}
SHOW = ("IT", "ES", "PT", "IE", "BE", "FR")


def run() -> str:
    s = monthly_spreads()
    rows = []
    for name, m in EVENTS.items():
        t = pd.Period(m, "M")
        before, after = s.loc[t - 1], s.loc[t + 2]
        for c in SHOW:
            rows.append(dict(event=name, month=m, country=c, before=before[c], after=after[c], change=after[c] - before[c]))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "tools_events.csv", index=False, float_format="%.4f")
    lines = ["Change in the monthly-average 10-year spread to Germany, month before to two months after (pp):"]
    for name, f in d.groupby("event", sort=False):
        f = f.set_index("country")
        lines.append(f"  {name}: " + ", ".join(f"{NAMES[c]} {f.loc[c, 'before']:.2f} -> {f.loc[c, 'after']:.2f} ({f.loc[c, 'change']:+.2f})" for c in SHOW))
    peak = s.loc["2010-01":"2012-12"].max()
    when = s.loc["2010-01":"2012-12"].idxmax()
    lines.append("Crisis peaks of the monthly average, 2010-2012: " + ", ".join(f"{NAMES[c]} {peak[c]:.2f} ({when[c]})" for c in SHOW))
    lines.append(f"France, latest month {s.index.max()}: {s['FR'].iloc[-1]:.2f}; highest monthly average since 2013: {s.loc['2013-01':, 'FR'].max():.2f} ({s.loc['2013-01':, 'FR'].idxmax()})")
    return write("tools_summary.txt", lines)


if __name__ == "__main__":
    print(run())
