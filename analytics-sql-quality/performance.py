"""Timings for the performance and tooling sections of the article.

Timings depend on the machine, so they are written to results/performance.txt
and are not checked in CI. Each variant runs twice to warm up and then REPS
times; the script reports medians and a bootstrap interval for the ratio of
medians. The 10-million-row table is the 2013 flights repeated 30 times with
the timestamps shifted by whole years: the values are real, the volume is not.

Run from this folder:  python performance.py > results/performance.txt
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np

from sqlq import data

HERE = Path(__file__).resolve().parent
REPS = 25
COPIES = 30
rng = np.random.default_rng(20261109)


def timed(fn, reps: int = REPS) -> list[float]:
    for _ in range(2):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def ratio_ci(slow: list[float], fast: list[float], b: int = 2000) -> tuple[float, float, float]:
    s, f = np.asarray(slow), np.asarray(fast)
    point = np.median(s) / np.median(f)
    boots = [np.median(rng.choice(s, s.size)) / np.median(rng.choice(f, f.size)) for _ in range(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


RECORDS: list[dict] = []


def report(label_slow: str, slow: list[float], label_fast: str, fast: list[float], key: str = "") -> None:
    r, lo, hi = ratio_ci(slow, fast)
    RECORDS.append({"key": key, "slow": label_slow, "fast": label_fast, "ratio": r, "lo": lo, "hi": hi,
                    "median_slow_ms": 1000 * statistics.median(slow), "median_fast_ms": 1000 * statistics.median(fast)})
    print(f"  {label_slow}: median {1000 * statistics.median(slow):.3f} ms")
    print(f"  {label_fast}: median {1000 * statistics.median(fast):.3f} ms")
    print(f"  ratio {r:.2f} (95% bootstrap CI {lo:.2f} to {hi:.2f})")


def main() -> None:
    print(f"machine: {platform.platform()}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}")
    print(f"duckdb {duckdb.__version__}, sqlite {sqlite3.sqlite_version}; {REPS} timed runs after 2 warm-ups")
    tmp = Path(tempfile.mkdtemp())
    con = duckdb.connect()
    con.execute("set TimeZone = 'America/New_York'")
    threads = con.execute("select current_setting('threads')").fetchone()[0]
    print(f"duckdb threads: {threads}")
    data.load_raw(con, typed=True)
    con.execute(f"""
        create table big as
        select
            year + k as year, month, day, dep_time, sched_dep_time, dep_delay, arr_time,
            sched_arr_time, arr_delay, carrier, flight, tailnum, origin, dest, air_time,
            distance, hour, minute, time_hour + to_years(cast(k as integer)) as time_hour,
            k * 1000000 + row_number() over () as flight_id
        from raw_flights, range({COPIES}) as t(k)
        order by time_hour
    """)
    n = con.execute("select count(*) from big").fetchone()[0]
    parquet = tmp / "flights.parquet"
    con.execute(f"copy big to '{parquet}' (format parquet)")
    groups = con.execute(f"select count(distinct row_group_id) from parquet_metadata('{parquet}')").fetchone()[0]
    print(f"scaled table: {n:,} rows; Parquet file {parquet.stat().st_size / 1e6:.0f} MB in {groups} row groups, "
          f"sorted by time_hour")

    print("\n[1] SELECT * versus named columns, fetched into pandas (one carrier, all years)")
    star = timed(lambda: con.execute(f"select * from '{parquet}' where carrier = 'UA'").df(), reps=10)
    cols = timed(lambda: con.execute(f"select carrier, dest, arr_delay from '{parquet}' where carrier = 'UA'").df(),
                 reps=10)
    report("select *", star, "three columns", cols, "star_fetch")
    print("  the same aggregate inside one query, star subquery against named columns:")
    inner_star = timed(lambda: con.execute(
        f"select dest, avg(arr_delay) from (select * from '{parquet}') group by dest").fetchall())
    inner_cols = timed(lambda: con.execute(
        f"select dest, avg(arr_delay) from (select dest, arr_delay from '{parquet}') group by dest").fetchall())
    report("star subquery", inner_star, "named columns", inner_cols, "star_inner")

    print("\n[2] Filtering one local day: function on the column versus a range on it")
    day = "2026-06-01"
    wrapped = timed(lambda: con.execute(
        f"select count(*) from '{parquet}' where cast(time_hour as date) = date '{day}'").fetchall())
    ranged = timed(lambda: con.execute(
        f"select count(*) from '{parquet}' where time_hour >= timestamptz '{day} 00:00:00' "
        f"and time_hour < timestamptz '{day} 00:00:00' + interval 1 day").fetchall())
    same = con.execute(f"select count(*) from '{parquet}' where cast(time_hour as date) = date '{day}'").fetchone()[0]
    same2 = con.execute(f"select count(*) from '{parquet}' where time_hour >= timestamptz '{day} 00:00:00' "
                        f"and time_hour < timestamptz '{day} 00:00:00' + interval 1 day").fetchone()[0]
    print(f"  both return {same:,} flights: {same == same2}")
    report("cast(time_hour as date) = day", wrapped, "time_hour range", ranged, "duck_wrapped")

    print("\n[3] Flights later than their route average: correlated subquery, window, join")
    q_corr = """select count(*) from {t} as f where f.arr_delay > (
        select avg(g.arr_delay) from {t} as g where g.origin = f.origin and g.dest = f.dest)"""
    q_win = """select count(*) from (select arr_delay, avg(arr_delay) over (partition by origin, dest) as route_avg
        from {t}) where arr_delay > route_avg"""
    q_join = """select count(*) from {t} as f join (select origin, dest, avg(arr_delay) as route_avg
        from {t} group by origin, dest) as r using (origin, dest) where f.arr_delay > r.route_avg"""
    answers = {k: con.execute(q.format(t="big")).fetchone()[0] for k, q in
               (("corr", q_corr), ("win", q_win), ("join", q_join))}
    print(f"  DuckDB, {n:,} rows; the three queries agree: {len(set(answers.values())) == 1} ({answers['corr']:,})")
    t_corr = timed(lambda: con.execute(q_corr.format(t="big")).fetchall(), reps=10)
    t_win = timed(lambda: con.execute(q_win.format(t="big")).fetchall(), reps=10)
    t_join = timed(lambda: con.execute(q_join.format(t="big")).fetchall(), reps=10)
    report("correlated subquery", t_corr, "pre-aggregated join", t_join, "duck_corr")
    report("window function", t_win, "pre-aggregated join", t_join, "duck_win")

    lite = sqlite3.connect(":memory:")
    rows = con.execute("select origin, dest, arr_delay, strftime(time_hour, '%Y-%m-%d %H:%M:%S'), month "
                       "from raw_flights").fetchall()  # local New York time as text
    lite.execute("create table flights (origin text, dest text, arr_delay integer, time_hour text, month integer)")
    lite.executemany("insert into flights values (?, ?, ?, ?, ?)", rows)
    lite.execute("create index ix_route on flights (origin, dest)")
    lite.execute("create table jan as select * from flights where month = 1")
    lite.execute("create index ix_jan_route on jan (origin, dest)")
    nj = lite.execute("select count(*) from jan").fetchone()[0]
    a = [lite.execute(q.format(t=t)).fetchone()[0] for q, t in ((q_corr, "jan"), (q_win, "jan"), (q_join, "jan"))]
    print(f"  SQLite, January only ({nj:,} rows), index on (origin, dest); answers agree: {len(set(a)) == 1}")
    s_corr = timed(lambda: lite.execute(q_corr.format(t="jan")).fetchall(), reps=5)
    s_join = timed(lambda: lite.execute(q_join.format(t="jan")).fetchall(), reps=5)
    s_win = timed(lambda: lite.execute(q_win.format(t="jan")).fetchall(), reps=5)
    report("correlated subquery", s_corr, "pre-aggregated join", s_join, "lite_corr")
    report("window function", s_win, "pre-aggregated join", s_join, "lite_win")

    print("\n[4] SQLite: a function on an indexed column")
    lite.execute("create index ix_time on flights (time_hour)")
    q_fn = "select count(*) from flights where date(time_hour) = '2013-06-01'"
    q_rng = "select count(*) from flights where time_hour >= '2013-06-01' and time_hour < '2013-06-02'"
    plan_fn = " ".join(r[-1] for r in lite.execute("explain query plan " + q_fn))
    plan_rng = " ".join(r[-1] for r in lite.execute("explain query plan " + q_rng))
    n_fn, n_rng = lite.execute(q_fn).fetchone()[0], lite.execute(q_rng).fetchone()[0]
    agree = n_fn == n_rng
    print(f"  plan with date(): {plan_fn}")
    print(f"  plan with range : {plan_rng}")
    print(f"  same count: {agree} ({n_fn:,} flights)")
    report("date(time_hour) = day", timed(lambda: lite.execute(q_fn).fetchall()),
           "time_hour range", timed(lambda: lite.execute(q_rng).fetchall()), "lite_wrapped")

    print("\n[5] DISTINCT over a fanned-out join, against the join at the right grain")
    con.execute("create table w as select * from raw_weather")
    q_fan = """select count(*), round(avg(dep_delay), 6) from (select distinct f.flight_id, f.dep_delay
        from big as f join w on f.origin = w.origin and f.month = w.month and f.day = w.day
        where f.year = 2013)"""
    q_ok = """select count(*), round(avg(dep_delay), 6) from (select f.flight_id, f.dep_delay
        from big as f join (select distinct origin, month, day from w) as d
        on f.origin = d.origin and f.month = d.month and f.day = d.day where f.year = 2013)"""
    print(f"  answers agree: {con.execute(q_fan).fetchall() == con.execute(q_ok).fetchall()}")
    report("join by day, then DISTINCT", timed(lambda: con.execute(q_fan).fetchall(), reps=10),
           "join to distinct days", timed(lambda: con.execute(q_ok).fetchall(), reps=10), "distinct_fanout")

    print("\n[6] Linting speed: 350 files (the seven shipped models, 50 copies each)")
    corpus = tmp / "corpus"
    corpus.mkdir()
    for k in range(50):
        for p in (HERE / "project_before" / "models").glob("*.sql"):
            shutil.copy(p, corpus / f"{p.stem}_{k}.sql")
    shutil.copy(HERE / ".sqlfluff", tmp / ".sqlfluff")
    shutil.copy(HERE / ".sqruff", tmp / ".sqruff")
    fluff = timed(lambda: subprocess.run(["sqlfluff", "lint", "corpus", "--processes", "1", "--nofail"],
                                         cwd=tmp, capture_output=True), reps=3)
    ruff = timed(lambda: subprocess.run(["sqruff", "lint", "corpus"], cwd=tmp, capture_output=True), reps=3)
    report("sqlfluff (1 process)", fluff, "sqruff", ruff, "lint")
    shutil.rmtree(tmp)
    (HERE / "results" / "performance.json").write_text(json.dumps(RECORDS, indent=1))


if __name__ == "__main__":
    main()
