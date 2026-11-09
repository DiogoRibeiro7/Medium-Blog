"""The months of the European decisions, and what Portugal's spread did in them.

    python events.py

Uses the monthly decomposition written by ``contagion.py`` (slopes from November 2009 to July
2012 without February-April 2012): Portugal's spread change in each month split into a common
part (the rest of the euro area), a Greece-specific part, an Ireland-specific part and Portugal's
own part. For each event month it also reports the raw changes for Greece, Ireland, Spain and
Italy, and Portugal's change against its 2009-11 to 2012-07 monthly distribution.

The data are monthly averages, so an event's effect is spread over two months when it falls late
in a month, and months with several events cannot be separated. Those months are marked.

Writes ``results/events.csv`` and ``results/events_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import NAMES, RESULTS, monthly_spreads, write

# month, short label, what happened (dates from the official statements), confounded?
EVENTS = [
    ("2009-12", "Greek downgrades", "Fitch and S&P cut Greece (8 and 16 Dec 2009); first Greek widening after the October deficit revision", False),
    ("2010-04", "Greece asks for help", "Eurostat revises Greece's 2009 deficit to 13.6% (22 Apr); Greece requests assistance (23 Apr); S&P cuts Greece to junk and Portugal two notches (27 Apr)", True),
    ("2010-05", "Greek loan, EFSF, SMP", "Greek programme (2 May); EFSM/EFSF agreed and the ECB starts buying bonds (9-10 May)", True),
    ("2010-10", "Deauville", "Franco-German agreement at Deauville (18 Oct; markets reacted on the 19th) that a permanent mechanism will involve private creditors; endorsed by the European Council (28-29 Oct)", False),
    ("2010-11", "Ireland's request", "Ireland requests assistance (21 Nov); programme agreed (28 Nov); finance ministers' statement that private involvement applies only after mid-2013 (12 Nov)", True),
    ("2011-03", "PEC IV rejected", "Parliament rejects the fourth austerity package (23 Mar); the government resigns; rating cuts; European Council agrees the ESM (24-25 Mar)", True),
    ("2011-04", "Request and ECB hike", "Portugal requests assistance (6-7 Apr); the ECB raises its main rate to 1.25% (7 Apr)", True),
    ("2011-05", "Programme agreed", "Memorandum agreed (3 May) and approved by the Eurogroup (16-17 May)", False),
    ("2011-07", "Junk, hike, summit", "Moody's cuts Portugal to junk, citing the risk that private creditors would have to participate before further official lending (5 Jul); ECB raises its rate to 1.50% (7 Jul); euro summit lowers EFSF rates and adopts Greek private-sector involvement (21 Jul)", True),
    ("2011-11", "ECB cut", "ECB cuts to 1.25% (3 Nov); Greek referendum episode", True),
    ("2011-12", "LTRO, cut", "ECB cuts to 1.00% and announces three-year loans (8 Dec); first allotment (21 Dec)", True),
    ("2012-01", "Peak", "S&P cuts nine euro sovereigns, Portugal to junk (13 Jan); Portugal's spread at its monthly peak", False),
    ("2012-03", "Greek bond exchange", "Greek private-sector exchange completed (9 Mar); the Greek yield series switches to the new bonds", True),
    ("2012-07", "Whatever it takes", "Draghi's London speech (26 Jul)", False),
    ("2012-08", "OMT announced", "ECB announces it will consider outright purchases (2 Aug)", False),
    ("2012-09", "OMT details", "OMT technical features published (6 Sep)", False),
]


def run() -> str:
    m = pd.read_csv(RESULTS / "contagion_monthly_pt.csv", dtype={"month": str})
    s = monthly_spreads()
    ch = s.diff()
    ch.index = ch.index.strftime("%Y-%m")
    lvl = s.copy()
    lvl.index = lvl.index.strftime("%Y-%m")
    crisis = m[m["in_window"]]
    sd, mean = crisis["pt"].std(), crisis["pt"].mean()
    rows = []
    for month, label, what, conf in EVENTS:
        r = m[m["month"] == month]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(dict(month=month, label=label, confounded=conf, what=what, pt_level=lvl.loc[month, "PT"],
                         pt=r.pt, common=r.common, greek=r.greek, irish=r.irish, own=r.own,
                         z=(r.pt - mean) / sd, rank=int((crisis["pt"] >= r.pt).sum()) if r.in_window else None,
                         el=ch.loc[month, "EL"], ie=ch.loc[month, "IE"], es=ch.loc[month, "ES"], it=ch.loc[month, "IT"]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "events.csv", index=False, float_format="%.4f")

    lines = [f"Portugal's monthly spread change, November 2009 - July 2012: mean {mean:+.2f} pp, standard deviation {sd:.2f}; {len(crisis)} months", "",
             "Event months: Portugal's change = common + Greek + Irish + own (pp); z-score and rank among the crisis months; raw changes elsewhere"]
    for r in df.itertuples():
        rank = f"rank {int(r.rank)} of {len(crisis)}" if pd.notna(r.rank) else "outside the window"
        lines.append(f"  {r.month} {r.label}{' (several events)' if r.confounded else ''}: level {r.pt_level:.2f}; change {r.pt:+.2f} = {r.common:+.2f} + {r.greek:+.2f} + {r.irish:+.2f} + {r.own:+.2f}; "
                     f"z {r.z:+.1f}, {rank}; Greece {r.el:+.2f}, Ireland {r.ie:+.2f}, Spain {r.es:+.2f}, Italy {r.it:+.2f}")
    lines += ["", "The five largest monthly increases in Portugal's spread, November 2009 - July 2012:"]
    for r in crisis.nlargest(5, "pt").itertuples():
        ev = [e[1] for e in EVENTS if e[0] == r.month]
        lines.append(f"  {r.month}: {r.pt:+.2f} (common {r.common:+.2f}, Greek {r.greek:+.2f}, Irish {r.irish:+.2f}, own {r.own:+.2f}){'; ' + ev[0] if ev else ''}")
    lines += ["", "Deauville at monthly resolution: October and November 2010 together"]
    two = m[m["month"].isin(["2010-10", "2010-11"])]
    lines.append(f"  Portugal {two.pt.sum():+.2f} = common {two.common.sum():+.2f} + Greek {two.greek.sum():+.2f} + Irish {two.irish.sum():+.2f} + own {two.own.sum():+.2f}; "
                 f"Greece {ch.loc[['2010-10', '2010-11'], 'EL'].sum():+.2f}, Ireland {ch.loc[['2010-10', '2010-11'], 'IE'].sum():+.2f}")
    lines += ["", "The ECB's 2011 hikes: Portugal's spread over the hike months and the month after"]
    for a, b in (("2011-04", "2011-05"), ("2011-07", "2011-08")):
        w = m[m["month"].isin([a, b])]
        lines.append(f"  {a} to {b}: Portugal {w.pt.sum():+.2f} = common {w.common.sum():+.2f} + Greek {w.greek.sum():+.2f} + Irish {w.irish.sum():+.2f} + own {w.own.sum():+.2f}")
    lines += ["", "OMT, July to September 2012 (outside the estimation window):"]
    w = m[m["month"].isin(["2012-07", "2012-08", "2012-09"])]
    lines.append(f"  Portugal {w.pt.sum():+.2f} = common {w.common.sum():+.2f} + Greek {w.greek.sum():+.2f} + Irish {w.irish.sum():+.2f} + own {w.own.sum():+.2f}; "
                 + ", ".join(f"{NAMES[g]} {ch.loc[['2012-07', '2012-08', '2012-09'], g].sum():+.2f}" for g in ("EL", "IE", "ES", "IT")))
    return write("events_summary.txt", lines)


if __name__ == "__main__":
    print(run())
