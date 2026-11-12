"""Every number quoted in Part 6: the changes of Part 5, frozen, tested on mornings they never saw.

In-sample (the 822 mornings of Parts 4 and 5, 1 June 2021 to 31 August 2023) the script replays the
eight versions of Part 5 and five versions of the graded check sketched in Part 5, one for each
tolerance tau, and picks the tau with the lowest mean absolute error. Out of sample (1 January 2024 to
13 October 2026, the latest morning whose seven dates are 28 days old in the file of 10 November 2026)
it replays all thirteen versions again, recomputes every morning of each, and measures them against the
final counts. The holiday calendar runs to 2027; with that the only change, the code of every Part 5
version is as it was.

Run: python oos_numbers.py > output.txt   (about half an hour on two cores)
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

HOLIDAY_YEARS = (2020, 2027)
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)   # 0.1 added after a first in-sample pass put the best value at 0.25, the edge
IS = dict(first_file=dt.date(2021, 4, 2), last_file=dt.date(2023, 12, 31), first_run=dt.date(2021, 6, 1),
          last_run=dt.date(2023, 8, 31), late={dt.date(2022, 3, 15): 2}, eval_from=pd.Timestamp("2021-06-01"))
OOS = dict(first_file=dt.date(2023, 8, 1), last_file=dt.date(2026, 11, 10), first_run=dt.date(2023, 10, 1),
           last_run=dt.date(2026, 10, 13), late={}, eval_from=pd.Timestamp("2024-01-01"))

P5 = [dict(content_check=c, holidays=h, fallback=f)
      for c, h, f in itertools.product((False, True), (False, True), ("level", "last_week"))]
GRADED = [dict(holidays=True, fallback="last_week", graded=t) for t in TAUS]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BREAKDOWN = ["2024-02-06", "2025-12-27"]   # out-of-sample mornings shown date by date in O10


def name(v: dict) -> str:
    if v.get("graded") is not None:
        return f"graded check (tau {v['graded']}) + holidays + last-week"
    parts = [n for n, on in (("content check", v["content_check"]), ("holidays", v["holidays"])) if on]
    base = " + ".join(parts) or "Part 4"
    return base + (", last-week fallback" if v["fallback"] == "last_week" else "")


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_job(job):
    period, v, reproduce = job
    cfg = IS if period == "in" else OOS
    tmp = Path(tempfile.mkdtemp())
    incoming = P.write_incoming(tmp / "in", cfg["first_file"], cfg["last_file"], cfg["late"])
    t0 = time.time()
    pipe = P.replay(cfg["first_run"], cfg["last_run"], incoming, holiday_years=HOLIDAY_YEARS, **v)
    t1 = time.time()
    rep = pipe.report()
    mismatch = None
    if reproduce:
        mismatch = 0
        for d in pd.to_datetime(rep.run_date):
            pr, rr = pipe.reproduce(d.date())
            ps, rs = pipe.stored(d.date())
            mismatch += not (P.same(pr, ps) and P.same(rr, rs))
    parts = {}
    if period == "out":                         # the mornings shown date by date
        for d in BREAKDOWN:
            run = pd.Timestamp(d).date()
            as_of = P.run_ts(run)
            head = pipe.report_sql.split("past_runs as (")[0].rstrip().rstrip(",")
            est = pipe.con.execute(head + "\nselect * from est order by d", {"as_of": as_of}).df()
            gw = None
            if v.get("graded") is not None:
                gw = pipe.con.execute(head + "\nselect gfile.q, gfile.observed, gfile.expected, gfile.r, gw.w "
                                      "from gfile join gw on gw.q = gfile.q order by gfile.q", {"as_of": as_of}).df()
            pick = pipe.con.execute(pipe.profile_sql.split("grid as (")[0].rstrip().rstrip(",")
                                    + "\nselect d, weekday from pick order by weekday, d", {"as_of": as_of}).df()
            prof = pipe.con.execute("select weekday, age, share from core.delay_profile where computed_at = ?",
                                    [as_of]).df()
            parts[d] = (est, gw, pick, prof)
    extra = None
    if period == "out" and v == P5[0]:          # the source itself, from the landing table
        extra = pipe.con.execute("""
            with n as (select publication_date as p, reporting_date as d, sum(cases) as c from landing.cases group by all),
            pairs as (
                select a.p, sum(abs(a.c - coalesce(b.c, 0))) as change
                from n a join (select distinct p from n) prev on prev.p = a.p - 1
                left join n b on b.p = a.p - 1 and b.d = a.d
                group by a.p)
            select p, isodow(p) as weekday, change from pairs order by p""").df()
    return period, tuple(sorted(v.items())), pipe.code_version, rep, mismatch, t1 - t0, time.time() - t1, extra, parts


def evaluate(rep: pd.DataFrame, final: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    rep = rep.copy()
    rep["run_date"] = pd.to_datetime(rep.run_date)
    rep["as_of_publication"] = pd.to_datetime(rep.as_of_publication)
    rep = rep[rep.run_date >= start].reset_index(drop=True)
    rep["final_7d"] = [final.reindex(pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))).sum()
                       for p in rep.as_of_publication]
    rep["err"] = rep.nowcast_7d / rep.final_7d - 1
    return rep


def summary(r: pd.DataFrame) -> str:
    e = r.err
    w = (r.nowcast_7d - r.final_7d).abs().sum() / r.final_7d.sum()
    i = e.abs().idxmax()
    return (f"mean absolute {pct(e.abs().mean())}, mean {100 * e.mean():+.1f}%, weighted {pct(w)}, "
            f">5% {int((e.abs() > .05).sum())}, >10% {int((e.abs() > .10).sum())}, >20% {int((e.abs() > .20).sum())}, "
            f"worst {100 * e[i]:+.1f}% on {r.run_date[i].date()}")


def main() -> None:
    final = P.final_national()
    jobs = [("in", v, False) for v in P5 + GRADED] + [("out", v, True) for v in P5 + GRADED]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        results = {(r[0], r[1]): r[2:] for r in ex.map(run_job, jobs)}
    key = lambda v: tuple(sorted(v.items()))  # noqa: E731
    for (period, k), (code, rep, mm, t_run, t_rep, _, _) in results.items():
        print(f"{period} {name(dict(k))}: replay {t_run:.0f} s, reproduction {t_rep:.0f} s", file=sys.stderr)
    ev = {(per, key(v)): evaluate(results[(per, key(v))][1], final, (IS if per == "in" else OOS)["eval_from"])
          for per in ("in", "out") for v in P5 + GRADED}

    section("O0. The data")
    s3 = pd.read_csv(P.PART3_DATA / "rki_state.csv.gz", parse_dates=["publication", "date"])
    s6 = pd.read_csv(P.PART6_DATA / "rki_state_2024_2026.csv.gz", parse_dates=["publication", "date"])
    a = s3[s3.publication == "2024-01-01"].sort_values(["state", "date"]).reset_index(drop=True)
    b = s6[s6.publication == "2024-01-01"].sort_values(["state", "date"]).reset_index(drop=True)
    print(f"daily files fetched for Part 6: {s6.publication.nunique()}, {s6.publication.min().date()} to {s6.publication.max().date()}; "
          f"rows {len(s6):,}")
    print(f"the file of 2024-01-01, read by Part 3 and again by Part 6: {len(a)} and {len(b)} rows, identical: {a.equals(b)}")
    m6 = pd.read_csv(P.PART6_DATA / "rki_melde_2024_2026.csv.gz", parse_dates=["publication", "date"])
    nat = m6[(m6.publication - m6.date).dt.days.between(1, 28)].groupby(["publication", "date"]).cases.sum()
    st = s6.groupby(["publication", "date"]).cases.sum()
    j = pd.concat([nat, st], axis=1, keys=["national", "states"]).fillna(0)
    print(f"national counts against the sum of the states, ages 1 to 28: {len(j):,} pairs, differing {int((j.national != j.states).sum())}")
    base = ev[("out", key(P5[0]))]
    print(f"out-of-sample mornings: {len(base)}, {base.run_date.min().date()} to {base.run_date.max().date()}; "
          f"replay from {OOS['first_run']} (files from {OOS['first_file']})")
    for y in (2024, 2025, 2026):
        m = base.run_date.dt.year == y
        print(f"  {y}: {int(m.sum())} mornings, final 7-day count median {int(base.final_7d[m].median()):,}, "
              f"min {int(base.final_7d[m].min()):,}, max {int(base.final_7d[m].max()):,}")
    ins = ev[("in", key(P5[0]))]
    print(f"in-sample mornings: {len(ins)}; final 7-day count median {int(ins.final_7d.median()):,}")
    print("holidays in the calendar, 2024 to 2026: " + ", ".join(str(d) for d in P.german_holidays(2024, 2026)))

    section("O1. Files that added nothing, out of sample")
    extra = results[("out", key(P5[0]))][5]
    extra["p"] = pd.to_datetime(extra.p)
    extra = extra[(extra.p >= OOS["eval_from"]) & (extra.p <= pd.Timestamp(OOS["last_run"]))]
    zero = extra[extra.change == 0]
    print(f"files compared with the file of the day before: {len(extra)}; identical: {len(zero)}")
    print("  by weekday: " + ", ".join(f"{WEEKDAYS[k - 1]} {int((zero.weekday == k).sum())}" for k in range(1, 8)))
    other = zero[~zero.weekday.isin([1, 7])]
    hol = set(P.german_holidays(2023, 2026))
    print(f"  dated Tuesday to Saturday: {len(other)}; of them the day after a nationwide holiday: "
          f"{int(sum((d - pd.Timedelta(days=1)).date() in hol for d in other.p))}")
    print("  " + ", ".join(f"{d.date()}{' (after holiday)' if (d - pd.Timedelta(days=1)).date() in hol else ''}" for d in other.p))
    flagged = base.run_date.isin(other.p)
    after_hol = base.run_date.isin([pd.Timestamp(h) + pd.Timedelta(days=1) for h in hol])
    print(f"out-of-sample mornings using such a file: {int(flagged.sum())}; mornings the day after a holiday: {int(after_hol.sum())}")

    section("O2. Tuning the graded check in sample (mean absolute error and more, 822 mornings)")
    for v in GRADED:
        print(f"  tau {v['graded']}: {summary(ev[('in', key(v))])}")
    best_tau = min(TAUS, key=lambda t: ev[("in", key(GRADED[TAUS.index(t)]))].err.abs().mean())
    print(f"chosen tau (lowest in-sample mean absolute error): {best_tau}")
    for v in (P5[0], P5[7]):
        print(f"  for comparison, {name(v)}: {summary(ev[('in', key(v))])}")

    section("O3. Out of sample, every version")
    for v in P5 + GRADED:
        code, rep, mm, *_ = results[("out", key(v))]
        print(f"  {name(v):48s} code {code}; recomputed identically {len(rep) - mm} of {len(rep)}")
        print(f"  {'':48s} {summary(ev[('out', key(v))])}")

    section("O4. Ranking: in sample against out of sample (by mean absolute error)")
    rows = []
    for v in P5 + GRADED:
        rows.append((name(v), ev[("in", key(v))].err.abs().mean(), ev[("out", key(v))].err.abs().mean()))
    df = pd.DataFrame(rows, columns=["version", "in", "out"])
    df["rank_in"] = df["in"].rank().astype(int)
    df["rank_out"] = df["out"].rank().astype(int)
    for _, r in df.sort_values("rank_out").iterrows():
        print(f"  out {r.rank_out:2d} (in {r.rank_in:2d})  {r.version:48s} in {pct(r['in'])}, out {pct(r['out'])}")
    # Spearman's correlation is Pearson's on the ranks; computed this way it needs no scipy.
    spearman = df["rank_in"].rank().corr(df["rank_out"].rank())
    print(f"Spearman correlation of the two rankings: {spearman:.2f}")

    section("O5. By year and by weekday, out of sample (mean absolute error)")
    shown = [P5[0], P5[1], P5[7]] + [GRADED[TAUS.index(best_tau)]]
    for v in shown:
        r = ev[("out", key(v))]
        wd = r.run_date.dt.dayofweek
        print(f"  {name(v):48s} " + ", ".join(f"{y} {pct(r.err[r.run_date.dt.year == y].abs().mean())}" for y in (2024, 2025, 2026))
              + "; " + ", ".join(f"{WEEKDAYS[k]} {pct(r.err[wd == k].abs().mean())}" for k in range(7)))

    section("O6. The mornings the changes were made for, out of sample")
    for v in P5 + GRADED:
        r = ev[("out", key(v))]
        print(f"  {name(v):48s} files with nothing new (Tue-Sat) {pct(r.err[flagged].abs().mean())}; "
              f"day after a holiday {pct(r.err[after_hol].abs().mean())}; other mornings {pct(r.err[~flagged & ~after_hol].abs().mean())}")

    section("O7. Run by run against Part 4, out of sample")
    b = ev[("out", key(P5[0]))]
    for v in P5[1:] + GRADED:
        d = ev[("out", key(v))].err.abs() - b.err.abs()
        r = ev[("out", key(v))]
        print(f"  {name(v):48s} better by >1 point {int((d < -0.01).sum())}, worse by >1 point {int((d > 0.01).sum())}; "
              f"largest worsening {100 * d.max():+.1f} points on {r.run_date[d.idxmax()].date()}")

    section("O8. Interval, completeness and week over week, out of sample")
    for v in shown:
        r = ev[("out", key(v))]
        iv = r[r.lower_7d.notna()]
        inside = (iv.lower_7d <= iv.final_7d) & (iv.final_7d <= iv.upper_7d)
        true_c = r.published_7d / r.final_7d
        rr = r.set_index("run_date")
        fin_wow = rr.final_7d / rr.final_7d.shift(7) - 1
        ok = fin_wow.notna() & rr.wow_nowcast.notna()
        same = int((np.sign(rr.wow_nowcast[ok]) == np.sign(fin_wow[ok])).sum())
        print(f"  {name(v):48s} interval coverage {pct(inside.mean())} of {len(iv)}, median width "
              f"{pct(((iv.upper_7d - iv.lower_7d) / iv.nowcast_7d).median())}; completeness off by "
              f"{100 * (r.completeness - true_c).abs().mean():.1f} points; week-over-week sign right {same} of {int(ok.sum())}")

    section("O9. The ten worst out-of-sample mornings of the two best Part 5-style versions")
    for v in (P5[7], GRADED[TAUS.index(best_tau)]):
        r = ev[("out", key(v))]
        top = r.reindex(r.err.abs().sort_values(ascending=False).index[:10])
        print(f"  {name(v)}:")
        for _, row in top.iterrows():
            tags = []
            if row.run_date in set(other.p):
                tags.append("file with nothing new")
            if (row.run_date - pd.Timedelta(days=1)).date() in hol:
                tags.append("after holiday")
            print(f"    {row.run_date.date()} {WEEKDAYS[row.run_date.dayofweek]} {100 * row.err:+.1f}% "
                  f"(nowcast {row.nowcast_7d:,.0f}, final {int(row.final_7d):,}){' - ' + ', '.join(tags) if tags else ''}")


    section("O10. Two out-of-sample mornings date by date")
    for d in BREAKDOWN:
        print(f"run of {d}:")
        for v in (P5[0], P5[7], GRADED[TAUS.index(best_tau)]):
            est, gw, pick, prof = results[("out", key(v))][6][d]
            age = est.age_eff if "age_eff" in est else est.age
            fin = [int(final.get(pd.Timestamp(x), 0)) for x in est.d]
            ms = est.missing_share if "missing_share" in est else pd.Series([0.0] * len(est))
            print(f"  {name(v)}: nowcast {est.nowcast.sum():,.0f}, final {sum(fin):,}")
            for x, wdy, a, k, n, f, m_ in zip(est.d, est.weekday, age, est.known, est.nowcast, fin, ms):
                print(f"    {pd.Timestamp(x):%a %d %b} (type {int(wdy)}, age {int(a)}): in hand {int(k):,}, "
                      f"nowcast {n:,.0f}, final {f:,}" + (f", share taken off {m_:.2f}" if m_ else ""))
            mondays = ", ".join(f"{pd.Timestamp(x).date()}" for x in pick[pick.weekday == 1].d)
            s1 = prof[(prof.weekday == 1) & (prof.age == 1)].share.iloc[0]
            print(f"    profile: Monday-type dates used {mondays}; expected share of a Monday one day old {s1:.3f}")
            if gw is not None:
                last = gw[pd.to_datetime(gw.q) >= pd.Timestamp(d) - pd.Timedelta(days=6)]
                print("    files: " + ", ".join(f"{pd.Timestamp(q):%d %b} added {o:,.0f} of {e:,.0f} expected (w {w:.2f})"
                                               for q, o, e, w in zip(last.q, last.observed, last.expected, last.w)))

    # the two small tables figures.py draws from
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df[["version", "in", "out", "rank_in", "rank_out"]].to_csv(out / "versions.csv", index=False, float_format="%.6f")
    rows = []
    for v in shown:
        r = ev[("out", key(v))]
        m = r.groupby(r.run_date.dt.to_period("M")).agg(mae=("err", lambda x: x.abs().mean()), final_7d=("final_7d", "median"))
        for month, row in m.iterrows():
            rows.append((str(month), name(v), row.mae, row.final_7d))
    pd.DataFrame(rows, columns=["month", "version", "mae", "final_7d_median"]).to_csv(
        out / "oos_monthly.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
