"""Every number quoted in Part 5: the Part 4 pipeline replayed with three changes, alone and together
(a 2 x 2 x 2 design), and measured against the final counts.

  content_check  count each date's age from the latest file that brought anything new
  holidays       treat Germany's nationwide public holidays as Sundays
  fallback       last_week instead of level: for a date with too little in hand, the estimate for the
                 same weekday a week earlier instead of the mean of dates 28 to 55 days old

Run: python fixes_numbers.py > output.txt   (about ten minutes on two cores)
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part4"))
import pipeline as P  # noqa: E402

FIRST_FILE, LAST_FILE = dt.date(2021, 4, 2), dt.date(2023, 12, 31)
FIRST_RUN, LAST_RUN = dt.date(2021, 6, 1), dt.date(2023, 8, 31)
LATE = {dt.date(2022, 3, 15): 2}
INTERVAL_FROM = pd.Timestamp("2021-08-01")
# the eight files dated Tuesday to Saturday that were identical to the file before (Part 4, P7),
# and the Sunday and Monday that followed the last two
EMPTY_WEEKDAY_FILES = ["2022-10-04", "2023-04-08", "2023-04-11", "2023-05-02", "2023-05-19", "2023-05-30",
                       "2023-08-25", "2023-08-26"]
FREEZE_WEEKEND = ["2023-08-27", "2023-08-28"]

VARIANTS = list(itertools.product((False, True), (False, True), ("level", "last_week")))
NAMES = {v: ("content check" if v[0] else "") + (" + " if v[0] and v[1] else "") + ("holidays" if v[1] else "")
         for v in VARIANTS}
NAMES = {v: (n or "Part 4") + (", last-week fallback" if v[2] == "last_week" else "") for v, n in NAMES.items()}
BREAKDOWN = ["2022-06-07", "2023-05-02", "2023-07-17", "2023-08-28"]   # runs shown date by date in F9


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_variant(v):
    cc, hol, fb = v
    tmp = Path(tempfile.mkdtemp())
    incoming = P.write_incoming(tmp / "incoming", FIRST_FILE, LAST_FILE, LATE)
    t0 = time.time()
    pipe = P.replay(FIRST_RUN, LAST_RUN, incoming, content_check=cc, holidays=hol, fallback=fb)
    t1 = time.time()
    rep = pipe.report()
    parts = {}
    for d in BREAKDOWN:                 # each of the seven dates of a run, as the report computed them
        run = pd.Timestamp(d).date()
        sql = pipe.report_sql.split("past_runs as (")[0].rstrip().rstrip(",") + "\nselect * from est order by d"
        parts[d] = pipe.con.execute(sql, {"as_of": P.run_ts(run)}).df()
    mismatch = 0
    for d in pd.to_datetime(rep.run_date):
        pr, rr = pipe.reproduce(d.date())
        ps, rs = pipe.stored(d.date())
        mismatch += not (P.same(pr, ps) and P.same(rr, rs))
    return v, pipe.code_version, rep, mismatch, t1 - t0, time.time() - t1, parts


def main() -> None:
    final = P.final_national()
    workers = min(len(VARIANTS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = {r[0]: r[1:] for r in ex.map(run_variant, VARIANTS)}
    for v in VARIANTS:
        print(f"{NAMES[v]}: replay {results[v][3]:.0f} s, reproduction {results[v][4]:.0f} s", file=sys.stderr)
    parts = {v: results[v][5] for v in VARIANTS}

    reps = {}
    for v in VARIANTS:
        rep = results[v][1].copy()
        rep["run_date"] = pd.to_datetime(rep.run_date)
        rep["as_of_publication"] = pd.to_datetime(rep.as_of_publication)
        rep["final_7d"] = [final.reindex(pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))).sum()
                           for p in rep.as_of_publication]
        rep["err"] = rep.nowcast_7d / rep.final_7d - 1
        reps[v] = rep
    base = reps[(False, False, "level")]
    flagged = base.run_date.isin(pd.to_datetime(EMPTY_WEEKDAY_FILES + FREEZE_WEEKEND))

    section("F0. The variants")
    print(f"runs per variant: {len(base)}, {FIRST_RUN} to {LAST_RUN}; files from {FIRST_FILE}; one late file (2022-03-15, two days)")
    print("nationwide public holidays in the calendar, 2021 to 2023: " + ", ".join(
        str(d) for d in P.german_holidays(2021, 2023)))
    for v in VARIANTS:
        _, code, rep, mismatch, _, _, _ = (v,) + results[v]
        print(f"  {NAMES[v]:50s} code version {code}; runs recomputed identically {len(rep) - mismatch} of {len(rep)}")

    section("F1. Nowcast error against the final counts, every variant")
    for v in VARIANTS:
        e = reps[v].err
        print(f"  {NAMES[v]:50s} mean {100 * e.mean():+.1f}%, mean absolute {pct(e.abs().mean())}, "
              f">5% off {int((e.abs() > .05).sum())}, >10% off {int((e.abs() > .10).sum())}, "
              f">20% off {int((e.abs() > .20).sum())}, worst {100 * e[e.abs().idxmax()]:+.1f}% on {reps[v].run_date[e.abs().idxmax()].date()}")

    section("F1b. Error weighted by size: absolute errors in cases, summed, over the final counts, summed")
    for v in VARIANTS:
        r = reps[v]
        w = lambda m: (r.nowcast_7d[m] - r.final_7d[m]).abs().sum() / r.final_7d[m].sum()  # noqa: E731
        print(f"  {NAMES[v]:50s} all runs {pct(w(r.index == r.index))}; " + ", ".join(
            f"{y} {pct(w(r.run_date.dt.year == y))}" for y in (2021, 2022, 2023))
              + f"; absolute error summed over all runs {int((r.nowcast_7d - r.final_7d).abs().sum()):,} cases")
    print(f"final 7-day counts summed over all runs: {int(base.final_7d.sum()):,}; by year " + ", ".join(
        f"{y} {int(base.final_7d[base.run_date.dt.year == y].sum()):,}" for y in (2021, 2022, 2023)))

    section("F2. The ten mornings of Part 4 that used a file with nothing new (eight weekday files, then 27 and 28 August 2023)")
    for v in VARIANTS:
        r = reps[v].set_index("run_date")
        print(f"  {NAMES[v]:50s} " + ", ".join(f"{d[5:]} {100 * r.err[pd.Timestamp(d)]:+.1f}%"
                                                for d in EMPTY_WEEKDAY_FILES + FREEZE_WEEKEND))
    print("mean absolute error on those ten mornings / on the other runs:")
    for v in VARIANTS:
        e = reps[v].err
        print(f"  {NAMES[v]:50s} {pct(e[flagged].abs().mean())} / {pct(e[~flagged].abs().mean())}")

    section("F3. By weekday of the run and by year (mean absolute error)")
    for v in VARIANTS:
        r = reps[v]
        wd = r.run_date.dt.dayofweek
        print(f"  {NAMES[v]:50s} " + ", ".join(f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]} {pct(r.err[wd == k].abs().mean())}"
                                                for k in range(7))
              + "; " + ", ".join(f"{y} {pct(r.err[r.run_date.dt.year == y].abs().mean())}" for y in (2021, 2022, 2023)))

    section("F4. Where the changes help and where they hurt (against Part 4, run by run)")
    for v in VARIANTS[1:]:
        d = reps[v].err.abs() - base.err.abs()
        print(f"  {NAMES[v]:50s} runs better by more than 1 point {int((d < -0.01).sum())}, worse by more than 1 point "
              f"{int((d > 0.01).sum())}, unchanged (within 0.001 points) {int((d.abs() < 1e-5).sum())}; "
              f"largest worsening {100 * d.max():+.1f} points on {reps[v].run_date[d.idxmax()].date()}")

    section("F5. The data cut (content-check variants)")
    for v in VARIANTS:
        if not v[0]:
            continue
        r = reps[v]
        lag = (r.as_of_publication - pd.to_datetime(r.data_cut)).dt.days
        wd = r.as_of_publication.dt.dayofweek
        print(f"  {NAMES[v]:50s} runs whose latest file brought nothing new: {int((lag > 0).sum())} "
              f"(Sun {int(((lag > 0) & (wd == 6)).sum())}, Mon {int(((lag > 0) & (wd == 0)).sum())}, "
              f"other days {int(((lag > 0) & ~wd.isin([0, 6])).sum())}); longest gap {int(lag.max())} days, on {r.run_date[lag.idxmax()].date()}")

    section("F6. Estimated completeness and the 90% interval")
    for v in VARIANTS:
        r = reps[v]
        true_c = r.published_7d / r.final_7d
        iv = r[(r.run_date >= INTERVAL_FROM) & r.lower_7d.notna()]
        inside = (iv.lower_7d <= iv.final_7d) & (iv.final_7d <= iv.upper_7d)
        by_year = ", ".join(f"{y} {pct(inside[iv.run_date.dt.year == y].mean())}" for y in (2021, 2022, 2023))
        r28 = r.set_index("run_date").loc[pd.Timestamp("2023-08-28")]
        print(f"  {NAMES[v]:50s} completeness off by {100 * (r.completeness - true_c).abs().mean():.1f} points on average "
              f"(28 Aug 2023: estimated {pct(r28.completeness)}, actual {pct(r28.published_7d / r28.final_7d)}); "
              f"interval coverage {pct(inside.mean())} of {len(iv)} ({by_year}), median width {pct(((iv.upper_7d - iv.lower_7d) / iv.nowcast_7d).median())}")

    section("F7. Week-over-week change of the nowcast against the final counts")
    for v in VARIANTS:
        r = reps[v].set_index("run_date")
        fin_wow = r.final_7d / r.final_7d.shift(7) - 1
        ok = fin_wow.notna() & r.wow_nowcast.notna()
        same_sign = int((np.sign(r.wow_nowcast[ok]) == np.sign(fin_wow[ok])).sum())
        print(f"  {NAMES[v]:50s} same sign as final on {same_sign} of {int(ok.sum())}; "
              f"mean absolute difference {100 * (r.wow_nowcast[ok] - fin_wow[ok]).abs().mean():.1f} points")

    section("F8. The run of 28 August 2023, with all three changes")
    best = reps[(True, True, "last_week")].set_index("run_date").loc[pd.Timestamp("2023-08-28")]
    print(f"data cut {pd.Timestamp(best.data_cut).date()}; published {int(best.published_7d):,}, nowcast {best.nowcast_7d:,.0f}, "
          f"interval {best.lower_7d:,.0f} to {best.upper_7d:,.0f}, estimated completeness {pct(best.completeness)}, final {int(best.final_7d):,}")

    print_breakdown(parts, final)


def print_breakdown(parts: dict, final: pd.Series) -> None:
    section("F9. Four runs date by date: count in hand, age used, nowcast, final")
    shown = [(False, False, "level"), (True, False, "level"), (True, True, "level"), (True, True, "last_week")]
    for d in BREAKDOWN:
        print(f"run of {d}:")
        for v in shown:
            est = parts[v][d]
            age = est.age_eff if "age_eff" in est else est.age
            cells = ", ".join(f"{pd.Timestamp(x):%a %d %b} {int(k):,} (age {int(a)}) -> {n:,.0f} / {int(final.get(pd.Timestamp(x), 0)):,}"
                              for x, k, a, n in zip(est.d, est.known, age, est.nowcast))
            fin = sum(int(final.get(pd.Timestamp(x), 0)) for x in est.d)
            print(f"  {NAMES[v]:50s} total {est.nowcast.sum():,.0f} / {fin:,} ({100 * (est.nowcast.sum() / fin - 1):+.1f}%): {cells}")


if __name__ == "__main__":
    main()
