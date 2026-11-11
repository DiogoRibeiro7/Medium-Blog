"""Every number quoted in Part 4: replay the pipeline over the RKI files and measure it.

Run: python practice_numbers.py > output.txt   (about four minutes)
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P

FIRST_FILE, LAST_FILE = dt.date(2021, 4, 2), dt.date(2023, 12, 31)
FIRST_RUN, LAST_RUN = dt.date(2021, 6, 1), dt.date(2023, 8, 31)
LATE = {dt.date(2022, 3, 15): 2}   # one file reaches the pipeline two days late
INTERVAL_FROM = pd.Timestamp("2021-08-01")


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


tmp = Path(tempfile.mkdtemp())
incoming = P.write_incoming(tmp / "incoming", FIRST_FILE, LAST_FILE, LATE)
t0 = time.time()
pipe = P.replay(FIRST_RUN, LAST_RUN, incoming)
print(f"replay took {time.time() - t0:.0f} s", file=sys.stderr)
con = pipe.con
rep = pipe.report()
rep["run_date"] = pd.to_datetime(rep.run_date)
rep["as_of_publication"] = pd.to_datetime(rep.as_of_publication)
final = P.final_national()

section("P0. The replay")
print(f"code version {P.CODE_VERSION}")
print(f"daily runs: {len(rep)}, {FIRST_RUN} to {LAST_RUN}, each at {P.RUN_TIME}")
nf, nr = con.execute("select count(*), sum(n_rows) from landing.files").fetchone()
print(f"files landed: {nf}; landing rows: {nr:,}; delay-profile rows: "
      f"{con.execute('select count(*) from core.delay_profile').fetchone()[0]:,}; report rows: {len(rep)}")
print(f"files written for {FIRST_FILE} to {LAST_FILE}: {len(incoming)}, of which {nf} had arrived by the last run; "
      "the file of 2022-03-15 arrives two days late")
stale = rep[rep.stale]
print("stale runs (latest publication older than the run date): "
      + ", ".join(f"{d.date()} (used {p.date()})" for d, p in zip(stale.run_date, stale.as_of_publication)))

section("P1. The report against the final counts (file of 10 November 2026)")
rows = []
for _, r in rep.iterrows():
    dates = pd.date_range(r.as_of_publication - pd.Timedelta(days=7), r.as_of_publication - pd.Timedelta(days=1))
    rows.append(final.reindex(dates).sum())
rep["final_7d"] = rows
e_pub = rep.published_7d / rep.final_7d - 1
e_now = rep.nowcast_7d / rep.final_7d - 1
print(f"published 7-day count: mean {e_pub.mean() * 100:+.1f}%, mean absolute {pct(e_pub.abs().mean())}, "
      f"more than 5% off on {int((e_pub.abs() > 0.05).sum())} of {len(rep)} runs")
print(f"nowcast: mean {e_now.mean() * 100:+.1f}%, mean absolute {pct(e_now.abs().mean())}, "
      f"more than 5% off on {int((e_now.abs() > 0.05).sum())}, more than 10% off on {int((e_now.abs() > 0.10).sum())}")
wd = rep.run_date.dt.dayofweek
print("mean absolute error by weekday of the run, published / nowcast: " + ", ".join(
    f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]} {pct(e_pub[wd == k].abs().mean())} / {pct(e_now[wd == k].abs().mean())}"
    for k in range(7)))
for y in (2021, 2022, 2023):
    m = rep.run_date.dt.year == y
    print(f"  {y}: published {pct(e_pub[m].abs().mean())}, nowcast {pct(e_now[m].abs().mean())}")
worst = e_now.abs().idxmax()
print(f"largest nowcast error: {e_now[worst] * 100:+.1f}% on {rep.run_date[worst].date()}")
true_c = rep.published_7d / rep.final_7d
print(f"estimated completeness against actual: mean absolute difference {100 * (rep.completeness - true_c).abs().mean():.1f} points; "
      f"estimated {pct(rep.completeness.mean())} on average, actual {pct(true_c.mean())}")
iv = rep[(rep.run_date >= INTERVAL_FROM) & rep.lower_7d.notna()]
inside = (iv.lower_7d <= iv.final_7d) & (iv.final_7d <= iv.upper_7d)
below = iv.final_7d < iv.lower_7d
print(f"90% interval from the pipeline's own past errors, runs from {INTERVAL_FROM.date()}: {len(iv)} runs, "
      f"contained the final count on {int(inside.sum())} ({pct(inside.mean())}); final below {int(below.sum())}, "
      f"above {int((iv.final_7d > iv.upper_7d).sum())}; median width {pct(((iv.upper_7d - iv.lower_7d) / iv.nowcast_7d).median())} of the nowcast")

section("P2. Week-over-week change against the final counts")
fin_wow = rep.set_index("run_date").final_7d / rep.set_index("run_date").final_7d.shift(7) - 1
fw = fin_wow.reindex(rep.run_date).to_numpy()
ok = ~np.isnan(fw) & rep.wow_published.notna().to_numpy()
for col, label in (("wow_published", "published, each week from its own run"), ("wow_nowcast", "nowcast, each week from its own run")):
    v = rep[col].to_numpy()
    print(f"  {label:40s} same sign as final on {int((np.sign(v[ok]) == np.sign(fw[ok])).sum())} of {int(ok.sum())}; "
          f"mean absolute difference {100 * np.nanmean(np.abs(v[ok] - fw[ok])):.1f} points")
one_file = []
for _, r in rep.iterrows():
    p = r.as_of_publication.date()
    known = con.execute("""
        select reporting_date, cases from core.known_at(?::timestamp) where publication_date = ?""",
                        [P.run_ts(r.run_date.date()), p]).df().set_index("reporting_date")["cases"]
    known.index = pd.to_datetime(known.index)
    this = pd.date_range(r.as_of_publication - pd.Timedelta(days=7), r.as_of_publication - pd.Timedelta(days=1))
    one_file.append(known.reindex(this).fillna(0).sum() / known.reindex(this - pd.Timedelta(days=7)).fillna(0).sum() - 1)
v = np.array(one_file)
print(f"  {'both weeks from the run day file':40s} same sign as final on {int((np.sign(v[ok]) == np.sign(fw[ok])).sum())} of {int(ok.sum())}; "
      f"mean absolute difference {100 * np.nanmean(np.abs(v[ok] - fw[ok])):.1f} points; falling while final rose "
      f"{int(((v[ok] < 0) & (fw[ok] > 0)).sum())}")

section("P3. Reproducing every run")
t0 = time.time()
mismatch = []
for d in rep.run_date:
    d = d.date()
    pr, rr = pipe.reproduce(d)
    ps, rs = pipe.stored(d)
    if not (P.same(pr, ps) and P.same(rr, rs)):
        mismatch.append(d)
print(f"runs recomputed from landing and earlier mart rows: {len(rep)}; identical to what was stored "
      f"(profile and report, every value): {len(rep) - len(mismatch)}; different: {len(mismatch)}", )
print(f"reproduction took {time.time() - t0:.0f} s", file=sys.stderr)

section("P4. The late file")
r15 = rep.set_index("run_date").loc[pd.Timestamp("2022-03-15")]
print(f"run of 2022-03-15: latest publication loaded {r15.as_of_publication.date()}, published 7-day count "
      f"{int(r15.published_7d):,}, nowcast {r15.nowcast_7d:,.0f}, stale {bool(r15.stale)}")
by_pub = con.execute("""
    select sum(cases) from landing.cases
    where publication_date = date '2022-03-15'
      and reporting_date between date '2022-03-08' and date '2022-03-14'""").fetchone()[0]
loaded = con.execute("select loaded_at from landing.files where publication_date = date '2022-03-15'").fetchone()[0]
print(f"the file of 2022-03-15 was loaded at {loaded}; a rebuild that filters by publication date instead of load time "
      f"would report {int(by_pub):,} for that run")

section("P5. Idempotence")
before = pipe.stored(LAST_RUN)
pipe.run(LAST_RUN, incoming)
after = pipe.stored(LAST_RUN)
print(f"rerunning the last day: report identical {P.same(before[1], after[1])}, profile identical {P.same(before[0], after[0])}; "
      f"files landed again: {con.execute('select count(*) from landing.files').fetchone()[0] - nf}")

section("P6. Training rows on both time axes (states, built at the end of the replay)")
tr = pipe.training_rows(P.run_ts(LAST_RUN), dt.date(2021, 5, 1))
tr["run_date"] = pd.to_datetime(tr.run_date)
ratio = tr.x_as_published / tr.x_final
print(f"rows: {len(tr):,} ({tr.run_date.nunique()} run days x {tr.state.nunique()} states), run days {tr.run_date.min().date()} to {tr.run_date.max().date()}")
print("feature as published / feature 28 days later, median by weekday of the run day: " + ", ".join(
    f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]} {pct(v)}" for k, v in ratio.groupby(tr.run_date.dt.dayofweek).median().items()))
print(f"  rows where the feature as published was more than 10% below its 28-day value: {int((ratio < 0.9).sum()):,} ({pct((ratio < 0.9).mean())}); "
      f"more than 25% below: {int((ratio < 0.75).sum()):,}")
by_state = ratio.groupby(tr.state).median().sort_values()
print(f"  lowest median by state: {by_state.iloc[0]:.3f} (state {by_state.index[0]}), highest {by_state.iloc[-1]:.3f} (state {by_state.index[-1]})")

section("P7. Files that added nothing")
nothing = con.execute("""
    with n as (
        select publication_date as p, reporting_date as d, sum(cases) as c
        from landing.cases group by all
    ),
    pairs as (                       -- each file against the file of the day before, on the dates it covers
        select a.p, sum(abs(a.c - coalesce(b.c, 0))) as change
        from n a
        join (select distinct p from n) prev on prev.p = a.p - 1
        left join n b on b.p = a.p - 1 and b.d = a.d
        where a.d >= a.p - 28
        group by a.p
    )
    select p, isodow(p) as weekday, change from pairs order by p""").df()
nothing["p"] = pd.to_datetime(nothing.p)
zero = nothing[nothing.change == 0]
print(f"files compared with the file of the day before: {len(nothing)}; identical on every reporting date they share: {len(zero)}")
print("  by weekday of the file: " + ", ".join(
    f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k - 1]} {int((zero.weekday == k).sum())}" for k in range(1, 8)))
other = zero[~zero.weekday.isin([1, 7])]
print("  identical files dated Tuesday to Saturday: " + ", ".join(str(d.date()) for d in other.p))
err = pd.Series(e_now.to_numpy(), index=rep.as_of_publication)
print("  nowcast error on the runs that used them: " + ", ".join(
    f"{d.date()} {100 * err[d]:+.1f}%" for d in other.p if d in err.index))
rest = err[~err.index.isin(other.p)]
print(f"  mean absolute nowcast error on the other runs: {pct(rest.abs().mean())}")
