"""Prints every number in the article except the timings (see performance.py)
and writes the data behind the figures to results/.

Run from this folder:  python analysis.py > results/output.txt
"""

from __future__ import annotations

import difflib
import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import sqlfluff
import sqlglot

from security import demo_queries
from sqlq import data, diff, mutate, project, testing

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "project_before" / "models"
AFTER = HERE / "project_after" / "models"
TESTS = HERE / "project_after" / "tests"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

GRAIN = {
    "daily_carrier_performance": ["carrier", "flight_date"],
    "route_summary": ["origin", "dest"],
    "plane_age_delays": ["age_band"],
    "weather_delays": ["weather_at_departure"],
    "monthly_long_delays": ["carrier", "flight_month"],
    "departure_slot_delays": ["sched_dep_hour"],
    "idle_aircraft": ["flight_month"],
}
MARTS = list(GRAIN)

# The defect in each shipped model, located by a fragment of its text.
DEFECTS = {
    "text comparison": ("monthly_long_delays", "dep_delay > '60'"),
    "session time zone": ("daily_carrier_performance", "CAST(time_hour AS DATE)"),
    "clock arithmetic": ("departure_slot_delays", "cast(dep_time AS integer) - sched_dep_time"),
    "inner join to incomplete dimension": ("route_summary", "join raw_airports a"),
    "join on a column with another meaning": ("plane_age_delays", "f.year = try_cast(p.year as integer)"),
    "join at the wrong grain": ("weather_delays", "f.day = w.day"),
    "nulls coalesced to zero": ("daily_carrier_performance", "COALESCE(TRY_CAST(arr_delay AS INTEGER), 0) <= 15"),
    "not in over a nullable column": ("idle_aircraft", "not in (select f.tailnum"),
}


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------------
def section_data(con) -> None:
    header("1. The data and the raw layer")
    for t in data.RAW_TABLES:
        print(f"raw_{t}: {one(con, f'select count(*) from raw_{t}'):,} rows")
    for t in ("raw_flights", "raw_planes", "raw_weather"):
        cols = con.execute(f"describe {t}").fetchall()
        texty = []
        for name, ctype, *_ in cols:
            if ctype == "VARCHAR":
                n_na = one(con, f"select count(*) from {t} where {name} = 'NA'")
                if n_na:
                    texty.append(f"{name} ({n_na:,} 'NA')")
        print(f"{t}: columns typed VARCHAR because they contain the text 'NA': {', '.join(texty)}")
    n = one(con, "select count(*) from raw_flights")
    canc = one(con, "select count(*) from raw_flights where dep_time = 'NA'")
    no_arr = one(con, "select count(*) from raw_flights where arr_delay = 'NA'")
    print(f"flights: {n:,}; cancelled (no departure time): {canc:,} ({pct(canc / n)}); "
          f"no arrival delay: {no_arr:,}, of which departed but not arrived: {no_arr - canc:,}")
    print(f"departed flights: {n - canc:,}")
    tail_na = one(con, "select count(*) from raw_flights where tailnum = 'NA'")
    tail_na_canc = one(con, "select count(*) from raw_flights where tailnum = 'NA' and dep_time = 'NA'")
    print(f"flights with tail number 'NA': {tail_na:,} (all cancelled: {tail_na == tail_na_canc})")
    miss_dest = con.execute(
        "select string_agg(dest || '=' || n, ', ' order by dest), sum(n) from ("
        "select dest, count(*) n from raw_flights f where not exists "
        "(select 1 from raw_airports a where a.faa = f.dest) group by dest)"
    ).fetchone()
    print(f"flights to destinations missing from airports: {miss_dest[1]:,} ({miss_dest[0]})")
    unreg = con.execute(
        "select count(*), count(distinct tailnum) from raw_flights f where tailnum <> 'NA' "
        "and not exists (select 1 from raw_planes p where p.tailnum = f.tailnum)"
    ).fetchone()
    print(f"flights by tail numbers absent from the registry: {unreg[0]:,} ({pct(unreg[0] / n)}), "
          f"{unreg[1]:,} distinct tails")
    by_carrier = con.execute(
        "select carrier, count(*) n from raw_flights f where tailnum <> 'NA' and not exists "
        "(select 1 from raw_planes p where p.tailnum = f.tailnum) group by carrier order by n desc limit 3"
    ).fetchall()
    print("  largest carriers among them: " + ", ".join(f"{c} {k:,} ({pct(k / unreg[0])})" for c, k in by_carrier))
    print(f"planes with no manufacture year: {one(con, "select count(*) from raw_planes where year = 'NA'")}; "
          f"built in 2013: {one(con, "select count(*) from raw_planes where year = '2013'")}")
    dup = one(con, "select count(*) from (select origin, year, month, day, hour from raw_weather "
                   "group by all having count(*) > 1)")
    dup_ts = one(con, "select count(*) from (select origin, time_hour from raw_weather group by all having count(*) > 1)")
    print(f"weather: local (origin, year, month, day, hour) keys repeated: {dup}; "
          f"(origin, time_hour) keys repeated: {dup_ts}")


# ---------------------------------------------------------------------------
def lint_project(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.sql")):
        for v in sqlfluff.lint(path.read_text(), config_path=str(HERE / ".sqlfluff")):
            out.append({"file": path.stem, **v})
    return out


def section_lint() -> dict:
    header("2. Linters and formatters")
    print(f"sqlfluff {sqlfluff.__version__}, dialect duckdb, all rules")
    res = {}
    for label, folder in (("before", BEFORE), ("after", AFTER)):
        viol = lint_project(folder)
        res[label] = viol
        by_group = Counter(v["code"][:2] for v in viol)
        files = len(list(folder.glob("*.sql")))
        lines = sum(len(p.read_text().splitlines()) for p in folder.glob("*.sql"))
        print(f"{label}: {len(viol)} violations in {files} files ({lines} lines); by rule group: "
              + ", ".join(f"{g} {k}" for g, k in sorted(by_group.items())))
    codes = Counter(v["code"] for v in res["before"])
    print("before, by rule: " + ", ".join(f"{c} {k}" for c, k in codes.most_common()))
    layout_cap = sum(k for c, k in codes.items() if c[:2] in ("LT", "CP"))
    print(f"before: layout and capitalisation rules account for {layout_cap} of {len(res['before'])} "
          f"({pct(layout_cap / len(res['before']))})")

    print("\nRules that fired on the line holding each defect:")
    for name, (model, fragment) in DEFECTS.items():
        lines = (BEFORE / f"{model}.sql").read_text().splitlines()
        line_no = next(i + 1 for i, line in enumerate(lines) if fragment in line)
        fired = sorted({v["code"] for v in res["before"] if v["file"] == model and v["start_line_no"] == line_no})
        print(f"  {name:40s} {model}.sql:{line_no:<3d} {', '.join(fired) or '(none)'}")

    sqruff = shutil.which("sqruff")
    if sqruff:
        for label, folder in (("before", BEFORE), ("after", AFTER)):
            proc = subprocess.run([sqruff, "lint", "-f", "json", str(folder)], capture_output=True, text=True,
                                  cwd=HERE)
            payload = json.loads(proc.stdout or "{}")
            codes_s = Counter(v["code"] for vs in payload.values() for v in vs)
            print(f"sqruff {label}: {sum(codes_s.values())} violations; "
                  + ", ".join(f"{c} {k}" for c, k in sorted(codes_s.items())))

    # Formatting is a separate change: apply sqlfluff fix to a copy of the
    # shipped project and count the lines it rewrites.
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(BEFORE, tmp / "models")
    changed = 0
    fixed_sql = {}
    for path in sorted((tmp / "models").glob("*.sql")):
        original = path.read_text()
        fixed = sqlfluff.fix(original, config_path=str(HERE / ".sqlfluff"))
        fixed_sql[path.stem] = fixed
        changed += sum(1 for line in difflib.unified_diff(original.splitlines(), fixed.splitlines(), lineterm="", n=0)
                       if line[:1] in "+-" and line[:3] not in ("+++", "---"))
    remaining = sum(len(sqlfluff.lint(s, config_path=str(HERE / ".sqlfluff"))) for s in fixed_sql.values())
    print(f"sqlfluff fix on the shipped project: {changed} diff lines (added + removed); "
          f"violations left after fixing: {remaining}")
    shutil.rmtree(tmp)
    res["fixed_sql"] = fixed_sql
    return res


# ---------------------------------------------------------------------------
def section_bugs(con, built) -> dict:
    header("3. What each defect does to the real data")
    out = {}
    n_dep = one(con, "select count(*) from raw_flights where dep_time <> 'NA'")

    print("\n[text comparison] departures delayed more than 60 minutes")
    true_n = one(con, "select count(*) from raw_flights where try_cast(dep_delay as integer) > 60")
    text_n = one(con, "select count(*) from raw_flights where dep_delay > '60'")
    parts = con.execute("""
        select
            count(*) filter (where dep_delay = 'NA' and dep_delay > '60'),
            count(*) filter (where try_cast(dep_delay as integer) between 7 and 9),
            count(*) filter (where try_cast(dep_delay as integer) between 61 and 99),
            count(*) filter (where try_cast(dep_delay as integer) >= 100),
            count(*) filter (where try_cast(dep_delay as integer) >= 100 and dep_delay > '60')
        from raw_flights""").fetchone()
    print(f"numeric: {true_n:,}; text comparison: {text_n:,} ({text_n / true_n - 1:+.1%})")
    print(f"text comparison counts {parts[0]:,} cancellations ('NA' > '60'), {parts[1]:,} delays of 7-9 minutes, "
          f"{parts[2]:,} delays of 61-99 minutes, and {parts[4]:,} of the {parts[3]:,} delays of 100+ minutes")
    print("engine response to the unquoted comparison dep_delay > 60:", end=" ")
    try:
        con.execute("select count(*) from raw_flights where dep_delay > 60")
        print("ran")
    except duckdb.Error as exc:
        print(type(exc).__name__ + ":", str(exc).splitlines()[0])
    cmp = con.execute("""
        select corr(b.share_over_60_min, a.share_over_60_min), max(abs(b.share_over_60_min - a.share_over_60_min)),
               count(*) filter (where b.departures_over_60_min > a.departures_over_60_min)
        from pb.monthly_long_delays b join pa.monthly_long_delays a using (carrier, flight_month)""").fetchone()
    print(f"carrier-months: correlation of the two shares {cmp[0]:.3f}; largest gap {cmp[1] * 100:.1f} points; "
          f"rows where the text version is higher: {cmp[2]} of {one(con, 'select count(*) from pa.monthly_long_delays')}")
    ex = con.execute("""
        select b.carrier, b.flight_month, b.share_over_60_min, a.share_over_60_min
        from pb.monthly_long_delays b join pa.monthly_long_delays a using (carrier, flight_month)
        order by abs(b.share_over_60_min - a.share_over_60_min) desc limit 1""").fetchone()
    print(f"largest gap: {ex[0]} month {ex[1]}: text {ex[2]:.3f} vs numeric {ex[3]:.3f}")

    print("\n[session time zone] local flight date from CAST(time_hour AS DATE)")
    for tz in ("UTC", "Europe/Lisbon", "America/New_York"):
        con.execute(f"set TimeZone = '{tz}'")
        wrong = one(con, "select count(*) from raw_flights where cast(time_hour as date) <> make_date(year, month, day)")
        dates = con.execute("select min(cast(time_hour as date)), max(cast(time_hour as date)) from raw_flights").fetchone()
        out[f"tz_wrong_{tz}"] = wrong
        print(f"session {tz:17s}: {wrong:,} flights ({pct(wrong / 336776)}) on the wrong date; range {dates[0]} to {dates[1]}")
    con.execute("set TimeZone = 'UTC'")
    tz_rows = {}
    for tz in ("UTC", "Europe/Lisbon", "America/New_York"):
        con.execute(f"set TimeZone = '{tz}'")
        con.execute(f"create or replace table tz_{tz.replace('/', '_')} as "
                    f"select * from pb.daily_carrier_performance limit 0")
        con.execute("set search_path = 'pb,main'")
        con.execute(f"insert into main.tz_{tz.replace('/', '_')} {built['before_sql']['daily_carrier_performance']}")
        con.execute("set search_path = 'main'")
        tz_rows[tz] = one(con, f"select count(*) from tz_{tz.replace('/', '_')}")
    con.execute("set TimeZone = 'UTC'")
    for tz in ("UTC", "Europe/Lisbon"):
        d = diff.data_diff(con, f"tz_{tz.replace('/', '_')}", "tz_America_New_York", GRAIN["daily_carrier_performance"])
        print(f"daily mart built in {tz} vs New York: {d.touched:,} of {d.rows_right:,} carrier-days differ "
              f"(only here {d.only_left}, only in New York {d.only_right}, changed {d.changed})")
    eve = one(con, "select count(*) from raw_flights where hour >= 19")
    print(f"flights scheduled at or after 19:00 local: {eve:,}")

    print("\n[clock arithmetic] dep_time - sched_dep_time on HHMM clock readings")
    hh = con.execute("""
        select count(*), count(*) filter (where cast(dep_time as integer) - sched_dep_time <> cast(dep_delay as integer)),
               count(*) filter (where cast(dep_time as integer) - sched_dep_time - cast(dep_delay as integer) not in (0)
                                and (cast(dep_time as integer) // 100) = (sched_dep_time // 100) + 1),
               count(*) filter (where cast(dep_time as integer) < sched_dep_time - 1200)
        from raw_flights where dep_time <> 'NA'""").fetchone()
    print(f"departures: {hh[0]:,}; HHMM difference disagrees with dep_delay: {hh[1]:,} ({pct(hh[1] / hh[0])}); "
          f"of which crossed into the next hour: {hh[2]:,}; departed after midnight from an evening slot: {hh[3]:,}")
    slots = con.execute("""
        select b.sched_dep_hour, b.avg_dep_delay_min, a.avg_dep_delay_min
        from pb.departure_slot_delays b join pa.departure_slot_delays a using (sched_dep_hour)
        order by abs(b.avg_dep_delay_min - a.avg_dep_delay_min) desc""").fetchall()
    print("largest slot gaps (hour: HHMM vs minutes): "
          + "; ".join(f"{h}: {b:.1f} vs {a:.1f}" for h, b, a in slots[:3]))
    signs = sum(1 for _, b, a in slots if (b < 0) != (a < 0))
    print(f"slots where the HHMM version has the opposite sign: {signs} of {len(slots)}")

    print("\n[inner join to incomplete dimension] route summary")
    r_b = con.execute("select count(*), sum(flights) from pb.route_summary").fetchone()
    r_a = con.execute("select count(*), sum(flights) from pa.route_summary").fetchone()
    print(f"routes: {r_b[0]} vs {r_a[0]}; flights: {r_b[1]:,} vs {r_a[1]:,} (dropped {r_a[1] - r_b[1]:,}, "
          f"{pct((r_a[1] - r_b[1]) / r_a[1])})")
    dropped = con.execute("""select origin, dest, flights from pa.route_summary a
        where not exists (select 1 from pb.route_summary b where b.origin = a.origin and b.dest = a.dest)
        order by flights desc""").fetchall()
    print("dropped routes: " + ", ".join(f"{o}-{d} {n:,}" for o, d, n in dropped))

    print("\n[join on a column with another meaning] plane age")
    pb = con.execute("select string_agg(age_band || ' ' || flights, ', '), sum(flights) from pb.plane_age_delays").fetchone()
    pa = con.execute("select string_agg(age_band || ' ' || flights || ' (' || round(avg_dep_delay_min, 1) || ')', ', ' "
                     "order by age_band), sum(flights) from pa.plane_age_delays").fetchone()
    print(f"shipped: {pb[0]} -> {pb[1]:,} of {n_dep:,} departures ({pct(pb[1] / n_dep)})")
    print(f"fixed  : {pa[0]}")

    print("\n[join at the wrong grain] weather by day instead of by hour")
    wb = con.execute("select sum(flights) from pb.weather_delays").fetchone()[0]
    print(f"rows counted as flights: {wb:,} vs {n_dep:,} departures (x{wb / n_dep:.2f})")
    for bucket in ("dry", "precipitation"):
        b = con.execute(f"select flights, avg_dep_delay_min from pb.weather_delays where weather_at_departure = '{bucket}'").fetchone()
        a = con.execute(f"select flights, avg_dep_delay_min from pa.weather_delays where weather_at_departure = '{bucket}'").fetchone()
        print(f"  {bucket:13s}: shipped {b[0]:,} rows, mean delay {b[1]:.1f} min; fixed {a[0]:,} flights, {a[1]:.1f} min")
    a_no = con.execute("select flights from pa.weather_delays where weather_at_departure = 'no observation'").fetchone()[0]
    print(f"  departures with no weather observation for their hour (fixed model): {a_no:,}")
    gap_b = one(con, "select max(avg_dep_delay_min) - min(avg_dep_delay_min) from pb.weather_delays")
    gap_a = one(con, "select max(avg_dep_delay_min) filter (where weather_at_departure = 'precipitation') "
                     "- max(avg_dep_delay_min) filter (where weather_at_departure = 'dry') from pa.weather_delays")
    print(f"  precipitation minus dry: shipped {gap_b:.1f} min, fixed {gap_a:.1f} min")

    print("\n[nulls coalesced to zero] daily on-time rate (pooled over the year)")
    pooled = {}
    for label, sch in (("shipped", "pb"), ("fixed", "pa")):
        r = con.execute(f"select sum(on_time_arrivals), sum(flights_completed), sum(flights_scheduled), "
                        f"sum(avg_arr_delay_min * flights_scheduled) / sum(flights_scheduled) from {sch}.daily_carrier_performance").fetchone()
        pooled[label] = r
    con.execute("create or replace table daily_coalesce_only as select * from pb.daily_carrier_performance limit 0")
    con.execute("set search_path = 'pb,main'")
    con.execute("insert into main.daily_coalesce_only "
                + built["before_sql"]["daily_carrier_performance"].replace("CAST(time_hour AS DATE)", "make_date(year, month, day)"))
    con.execute("set search_path = 'main'")
    on_b = one(con, "select sum(on_time_arrivals)::double / sum(flights_scheduled) from daily_coalesce_only")
    on_a = one(con, "select sum(on_time_arrivals)::double / sum(flights_completed) from pa.daily_carrier_performance")
    arr_b = one(con, "select avg(coalesce(try_cast(arr_delay as integer), 0)) from raw_flights")
    arr_a = one(con, "select avg(try_cast(arr_delay as integer)) from raw_flights")
    print(f"on-time rate: shipped {pct(on_b)} vs fixed {pct(on_a)}; mean arrival delay: shipped {arr_b:.2f} vs fixed {arr_a:.2f} min")
    d = diff.data_diff(con, "daily_coalesce_only", "pa.daily_carrier_performance", GRAIN["daily_carrier_performance"])
    worst = con.execute("""select b.carrier, b.flight_date, b.on_time_rate, a.on_time_rate, a.flights_scheduled - a.flights_completed
        from daily_coalesce_only b join pa.daily_carrier_performance a using (carrier, flight_date)
        order by b.on_time_rate - a.on_time_rate desc limit 1""").fetchone()
    print(f"carrier-days changed by this defect alone: {d.changed:,} of {d.rows_right:,}; "
          f"largest: {worst[0]} on {worst[1]}: {pct(worst[2])} shipped vs {pct(worst[3])} fixed ({worst[4]} flights not completed)")

    print("\n[not in over a nullable column] idle aircraft after the loader is fixed")
    typed = data.raw_connection("UTC", typed=True)
    errors, ok = [], []
    typed.execute("create schema t")
    typed.execute("set search_path = 't,main'")
    for name in project.build_order(built["before"]):
        try:
            typed.execute(f"create table t.{name} as {built['before'][name].sql}")
            ok.append(name)
        except duckdb.Error as exc:
            errors.append((name, type(exc).__name__))
    print(f"typed loader: {len(errors)} of {len(built['before'])} shipped models fail to build: "
          + ", ".join(f"{n} ({e})" for n, e in errors))
    for name in ok:
        print(f"  builds: {name}: {one(typed, f'select count(*) from t.{name}')} rows "
              f"(was {one(con, f'select count(*) from pb.{name}')})")
    nulls = one(typed, "select count(*) from main.raw_flights where tailnum is null")
    print(f"  NULL tail numbers after the fix: {nulls:,}")
    out["typed_errors"] = errors
    typed.close()

    print("\n[integer division] the same rate expression in two engines")
    lite = sqlite3.connect(":memory:")
    rows = con.execute("select carrier, arr_delay_min from pa.stg_flights").fetchall()
    lite.execute("create table stg_flights (carrier text, arr_delay_min integer)")
    lite.executemany("insert into stg_flights values (?, ?)", rows)
    q = ("select carrier, count(*) filter (where arr_delay_min <= 15) / count(arr_delay_min) as on_time_rate "
         "from stg_flights group by carrier order by carrier")
    duck = dict(con.execute(q.replace("from stg_flights", "from pa.stg_flights")).fetchall())
    lite_raw = dict(lite.execute(q).fetchall())
    trans = sqlglot.transpile(q, read="duckdb", write="sqlite")[0]
    lite_t = dict(lite.execute(trans).fetchall())
    agree = sum(1 for c in duck if math.isclose(duck[c], lite_t[c], rel_tol=1e-12))
    print(f"DuckDB: UA {duck['UA']:.4f}; SQLite, same text: UA {lite_raw['UA']} "
          f"(carriers at zero: {sum(1 for v in lite_raw.values() if v == 0)} of {len(lite_raw)})")
    print(f"sqlglot transpiled to SQLite: {trans}")
    print(f"transpiled query agrees with DuckDB for {agree} of {len(duck)} carriers")
    return out


# ---------------------------------------------------------------------------
def run_suite(con, models, schema, cases, defaults, tag, unit_models=None):
    """All three test levels against models already built in ``schema``."""
    con.execute(f"set search_path = '{schema},pa,main'")
    res = testing.run_schema_tests(con, TESTS / "schema.yml")
    res += testing.run_singular_tests(con, TESTS / "singular")
    con.execute("set search_path = 'main'")
    sel = [c for c in cases if unit_models is None or c["model"] in unit_models]
    res += testing.run_unit_tests(con, models, sel, defaults, tag=tag)
    return res


def output_changed(con, affected) -> bool:
    """Does the mutated build differ from the fixed build on the real data?"""
    for a in sorted(affected):
        try:
            if a in GRAIN:
                if diff.data_diff(con, f"m.{a}", f"pa.{a}", GRAIN[a]).touched > 0:
                    return True
            elif one(con, f"select count(*) from (select * from m.{a} except all select * from pa.{a})") > 0:
                return True
        except duckdb.Error:
            return True
    return False


def section_tests(con, built) -> dict:
    header("4. Tests: which layer catches which defect, and mutation testing")
    cases, defaults = built["cases"], built["defaults"]
    res_after = run_suite(con, built["after"], "pa", cases + built["cases_b"], defaults, "ua")
    warn = [r for r in res_after if not r.passed and r.severity == "warn"]
    err = [r for r in res_after if not r.passed and r.severity == "error"]
    levels = Counter(r.level for r in res_after)
    print(f"test suite: {len(res_after)} tests ({', '.join(f'{k} {v}' for k, v in sorted(levels.items()))}); "
          f"fixed project: {len(err)} errors, {len(warn)} warnings")
    for r in warn:
        print(f"  warning: {r.name}: {r.failing_rows:,} rows")

    # Each shipped model, plus the daily model with one defect at a time.
    daily = built["before_sql"]["daily_carrier_performance"]
    variants = {
        "daily_carrier_performance": daily,
        "daily (time zone only)": daily.replace("COALESCE(TRY_CAST(arr_delay AS INTEGER), 0) <= 15", "TRY_CAST(arr_delay AS INTEGER) <= 15")
        .replace("avg(COALESCE(TRY_CAST(arr_delay AS INTEGER), 0))", "avg(TRY_CAST(arr_delay AS INTEGER))")
        .replace("/ count(*)", "/ nullif(sum(case when arr_delay <> 'NA' then 1 else 0 end), 0)"),
        "daily (coalesce only)": daily.replace("CAST(time_hour AS DATE)", "make_date(year, month, day)"),
    }
    rows = []
    entries = [(label, "daily_carrier_performance", sql) for label, sql in variants.items()]
    entries += [(m, m, built["before_sql"][m]) for m in MARTS if m != "daily_carrier_performance"]
    entries.append(("idle (not in, on staging)", "idle_aircraft", (HERE / "review" / "idle_aircraft_not_in.sql").read_text()))
    for label, model, sql in entries:
        deps = frozenset(project.referenced_tables(sql) & built["after"].keys())
        models = {**built["after"], model: project.Model(model, sql, deps)}
        con.execute("drop schema if exists v cascade")
        con.execute("create schema v")
        con.execute("set search_path = 'v,pa,main'")
        engine = "ok"
        try:
            con.execute(f"create table v.{model} as {sql}")
        except duckdb.Error as exc:
            engine = type(exc).__name__
        con.execute("set search_path = 'v,pa,main'")
        schema_r = [r for r in testing.run_schema_tests(con, TESTS / "schema.yml", {model}) if r.severity == "error"]
        singular = [r for r in testing.run_singular_tests(con, TESTS / "singular") if r.model == model]
        con.execute("set search_path = 'main'")
        unit_r = testing.run_unit_tests(con, models, [c for c in cases if c["model"] == model], defaults, tag="uv")
        caught = {
            "schema": [r.name.split(":", 1)[1] for r in schema_r if not r.passed],
            "reconciliation": [r.name for r in singular if not r.passed],
            "unit": [r.name for r in unit_r if not r.passed],
        }
        rows.append((label, engine, caught))
    print("\nDetection by layer (session time zone UTC; 'x' = at least one failing test):")
    print(f"  {'model':34s} {'build':6s} {'schema':7s} {'recon':6s} {'unit':5s}")
    for label, engine, caught in rows:
        marks = ["x" if caught[k] else "." for k in ("schema", "reconciliation", "unit")]
        print(f"  {label:34s} {engine:6s} {marks[0]:7s} {marks[1]:6s} {marks[2]:5s}")
        for k in ("schema", "reconciliation"):
            if caught[k]:
                print(f"      {k}: {'; '.join(caught[k])}")
    built["detection_rows"] = rows

    # Same check for the time-zone defect under a New York session.
    con.execute("set TimeZone = 'America/New_York'")
    con.execute("drop schema if exists v cascade")
    con.execute("create schema v")
    con.execute("set search_path = 'v,pa,main'")
    con.execute(f"create table v.daily_carrier_performance as {variants['daily (time zone only)']}")
    ny = [r for r in testing.run_schema_tests(con, TESTS / "schema.yml", {"daily_carrier_performance"})
          + testing.run_singular_tests(con, TESTS / "singular") if r.model == "daily_carrier_performance" and not r.passed and r.severity == "error"]
    con.execute("set search_path = 'main'")
    models = {**built["after"], "daily_carrier_performance": project.Model("daily_carrier_performance", variants["daily (time zone only)"], frozenset())}
    ny_unit = [r for r in testing.run_unit_tests(con, models, [c for c in cases if c["model"] == "daily_carrier_performance"], defaults, "un") if not r.passed]
    con.execute("set TimeZone = 'UTC'")
    print(f"time-zone defect in a New York session: {len(ny)} data tests and {len(ny_unit)} unit tests fail")

    # A first version of the long-delay fixture: 7, 61, 100, 150 and one cancellation.
    weak = [{**cases[[c["model"] for c in cases].index("monthly_long_delays")]}]
    weak[0] = {**weak[0], "given": {"raw_flights": [
        {"carrier": "UA", "dep_delay": "7"}, {"carrier": "UA", "dep_delay": "61"},
        {"carrier": "UA", "dep_delay": "100"}, {"carrier": "UA", "dep_delay": "150"},
        {"carrier": "UA", "dep_time": "NA", "dep_delay": "NA", "arr_time": "NA", "arr_delay": "NA", "air_time": "NA"}]},
        "expect": [{"carrier": "UA", "flight_month": 1, "departures": 4, "departures_over_60_min": 3, "share_over_60_min": 0.75}]}
    weak_models = {**built["after"], "monthly_long_delays": project.Model("monthly_long_delays", built["before_sql"]["monthly_long_delays"], frozenset())}
    w = testing.run_unit_tests(con, weak_models, weak, defaults, "uw")[0]
    print(f"fixture (7, 61, 100, 150, one cancellation) against the text comparison: passed={w.passed} "
          f"(two errors cancel: 'NA' > '60' adds one, '100' and '150' < '60' remove two, '7' > '60' adds one)")

    # ------------------------------------------------------------- mutation
    print("\nMutation testing on the fixed project")
    base_outputs = {}
    all_mutants = []
    for name in project.build_order(built["after"]):
        _, ms = mutate.mutants(name, built["after"][name].sql)
        all_mutants.extend(ms)
    # baseline: round-tripped SQL must reproduce the outputs
    rt = {n: project.Model(n, mutate.mutants(n, m.sql)[0], m.deps) for n, m in built["after"].items()}
    project.build(con, rt, "rt", "rt,main")
    con.execute("set search_path = 'main'")
    same = all(diff.data_diff(con, f"rt.{m}", f"pa.{m}", GRAIN[m]).touched == 0 for m in MARTS)
    print(f"round-tripped SQL reproduces every mart: {same}")
    per_model = Counter(m.model for m in all_mutants)
    print(f"mutants: {len(all_mutants)} across {len(per_model)} models; by operator: "
          + ", ".join(f"{op} {k}" for op, k in Counter(m.operator for m in all_mutants).most_common()))

    records = []
    for i, mut in enumerate(all_mutants):
        models = dict(built["after"])
        models[mut.model] = project.Model(mut.model, mut.sql, built["after"][mut.model].deps)
        affected = project.downstream(models, mut.model)
        killed = {"build": False, "schema": False, "reconciliation": False, "unit": False, "boundary": False}
        changes = None
        con.execute("drop schema if exists m cascade")
        try:
            project.build(con, models, "m", "m,pa,main", only=affected)
        except duckdb.Error:
            killed["build"] = True
        con.execute("set search_path = 'm,pa,main'")
        if not killed["build"]:
            sch = testing.run_schema_tests(con, TESTS / "schema.yml", affected)
            killed["schema"] = any(not r.passed for r in sch if r.severity == "error")
            sing = testing.run_singular_tests(con, TESTS / "singular")
            killed["reconciliation"] = any(not r.passed for r in sing)
            con.execute("set search_path = 'main'")
            changes = output_changed(con, affected)
        con.execute("set search_path = 'main'")
        unit = testing.run_unit_tests(con, models, [c for c in cases if c["model"] in affected], defaults, tag="mu")
        killed["unit"] = any(not r.passed for r in unit)
        extra = testing.run_unit_tests(con, models, [c for c in built["cases_b"] if c["model"] in affected],
                                       defaults, tag="mb")
        killed["boundary"] = any(not r.passed for r in extra)
        records.append({"i": i, "model": mut.model, "operator": mut.operator, "detail": mut.detail,
                        "changes_real_output": changes, **killed})
    con.execute("drop schema if exists m cascade")

    n = len(records)
    suites = [
        ("build only", ("build",)),
        ("+ schema tests", ("build", "schema")),
        ("+ reconciliation", ("build", "schema", "reconciliation")),
        ("+ unit tests", ("build", "schema", "reconciliation", "unit")),
        ("+ boundary fixtures", ("build", "schema", "reconciliation", "unit", "boundary")),
    ]
    scores = []
    for label, levels_ in suites:
        k = sum(1 for r in records if any(r[lv] for lv in levels_))
        lo, hi = wilson(k, n)
        scores.append((label, k, n, lo, hi))
        print(f"  {label:18s} killed {k:3d} of {n} = {pct(k / n)} (95% CI {pct(lo)} to {pct(hi)})")
    first = ("build", "schema", "reconciliation", "unit")
    alone = {lv: sum(1 for r in records if r[lv]) for lv in first}
    only = {lv: sum(1 for r in records if r[lv] and not any(r[o] for o in first if o != lv)) for lv in first}
    print("  killed by each level on its own: " + ", ".join(f"{lv} {k}" for lv, k in alone.items()))
    print("  killed by that level and no other (first suite): " + ", ".join(f"{lv} {k}" for lv, k in only.items()))
    without_unit = sum(1 for r in records if r["build"] or r["schema"] or r["reconciliation"])
    for label, levels_ in (("first suite", first), ("with boundary fixtures", (*first, "boundary"))):
        survivors = [r for r in records if not any(r[lv] for lv in levels_)]
        eq = [r for r in survivors if r["changes_real_output"] is False]
        real = [r for r in survivors if r["changes_real_output"]]
        adj = (n - len(survivors)) / (n - len(eq)) if n > len(eq) else math.nan
        print(f"  survivors, {label}: {len(survivors)}; identical output on the real data: {len(eq)}; "
              f"different output on the real data: {len(real)}; score excluding identical-output mutants: {pct(adj)}")
        for r in survivors:
            tag = "same output" if r["changes_real_output"] is False else "output differs"
            print(f"    [{tag}] {r['model']}: {r['operator']}: {r['detail']}")
    op_table = defaultdict(lambda: [0, 0])
    for r in records:
        op_table[r["operator"]][0] += 1
        op_table[r["operator"]][1] += any(r[lv] for lv in first)
    print("  by operator, first suite (killed/total): " + ", ".join(f"{op} {k}/{t}" for op, (t, k) in sorted(op_table.items())))
    unit_only_ops = Counter(r["operator"] for r in records if r["unit"] and not (r["build"] or r["schema"] or r["reconciliation"]))
    print("  operators killed only by unit tests: " + ", ".join(f"{op} {k}" for op, k in unit_only_ops.most_common()))
    (RESULTS / "mutants.json").write_text(json.dumps(records, indent=1))
    return {"scores": scores, "records": records, "without_unit": without_unit}


# ---------------------------------------------------------------------------
def section_review(con, built, lint) -> list:
    header("5. Review: the code diff and the data diff")
    rows = []
    for m in MARTS:
        a = built["before_sql"][m].splitlines()
        b = built["after"][m].sql.splitlines()
        changed = sum(1 for line in difflib.unified_diff(a, b, lineterm="", n=0)
                      if line[:1] in "+-" and line[:3] not in ("+++", "---"))
        d = diff.data_diff(con, f"pb.{m}", f"pa.{m}", GRAIN[m])
        rows.append((m, changed, d))
        print(f"{m:27s} SQL diff lines {changed:3d}; output rows: {d.rows_left:,} -> {d.rows_right:,}; "
              f"only before {d.only_left:,}, only after {d.only_right:,}, changed {d.changed:,}, "
              f"unchanged {d.unchanged:,} ({pct(d.touched_share)} touched)")

    # formatting only: the sqlfluff-fixed copy must produce identical outputs
    fixed = {n: project.Model(n, s, frozenset()) for n, s in lint["fixed_sql"].items()}
    project.build(con, {**{k: v for k, v in built["after"].items() if k.startswith("stg_")}, **fixed}, "fmt", "fmt,main")
    con.execute("set search_path = 'main'")
    touched = sum(diff.data_diff(con, f"fmt.{m}", f"pb.{m}", GRAIN[m]).touched for m in MARTS)
    print(f"after sqlfluff fix (formatting only), output rows that differ from the shipped models: {touched}")

    # the typed-loader pull request: one line in the loader, no model changed
    typed = data.raw_connection("UTC", typed=True)
    typed.execute("set search_path = 'main'")
    typed.execute(f"create table idle_new as {built['before_sql']['idle_aircraft']}")
    print(f"typed-loader PR: idle_aircraft {one(con, 'select count(*) from pb.idle_aircraft')} rows before, "
          f"{one(typed, 'select count(*) from idle_new')} after; "
          f"idle aircraft summed over months before: {one(con, 'select sum(idle_aircraft) from pb.idle_aircraft'):,}")
    typed.close()
    return rows


# ---------------------------------------------------------------------------
def section_security(built) -> list:
    header("6. Security: injection through a carrier filter")
    tmp = Path(tempfile.mkdtemp())
    secret = tmp / "api_key.txt"
    secret.write_text("sk-demo-0000-not-a-real-key")
    db_path = tmp / "marts.duckdb"
    setup = duckdb.connect(str(db_path))
    data.load_raw(setup)
    project.build(setup, built["after"], "main", "main", only=project.upstream(built["after"], "monthly_long_delays"))
    setup.execute("create table app_users as select * from (values ('analyst', 'pbkdf2$1a2b'), "
                  "('admin', 'pbkdf2$9f8e')) as t(username, password_hash)")
    setup.close()
    payloads = {
        "benign": "UA",
        "tautology": "UA' or '1' = '1",
        "union, other table": "XX' union all select username, 0, 0 from app_users --",
        "union, local file": f"XX' union all select content, 0, 0 from read_text('{secret.as_posix()}') --",
        "stacked drop": "UA'; drop table monthly_long_delays; --",
    }
    configs = {
        "default": dict(read_only=False, config={}),
        "read-only": dict(read_only=True, config={}),
        "no external access": dict(read_only=False, config={"enable_external_access": False}),
        "read-only + no external access": dict(read_only=True, config={"enable_external_access": False}),
    }
    outcomes = []
    for cname, cfg in configs.items():
        for fname, fn in demo_queries.VALUE_FUNCTIONS.items():
            for pname, payload in payloads.items():
                work = tmp / f"w_{len(outcomes)}.duckdb"
                shutil.copy(db_path, work)
                con = duckdb.connect(str(work), read_only=cfg["read_only"], config=cfg["config"])
                try:
                    rows = fn(con, payload)
                    leaked_users = any("pbkdf2" in str(r) or r[0] in ("analyst", "admin") for r in rows)
                    leaked_file = any("sk-demo" in str(r) for r in rows)
                    carriers = {r[0] for r in rows}
                    status = "ok"
                except duckdb.Error as exc:
                    rows, leaked_users, leaked_file, carriers = [], False, False, set()
                    status = type(exc).__name__
                try:
                    dropped = one(con, "select count(*) from information_schema.tables "
                                       "where table_name = 'monthly_long_delays'") == 0
                except duckdb.Error:
                    dropped = False
                con.close()
                work.unlink()
                outcomes.append({"config": cname, "function": fname, "payload": pname, "rows": len(rows),
                                 "carriers": len(carriers), "leaked_users": leaked_users,
                                 "leaked_file": leaked_file, "dropped": dropped, "status": status})
    for cname in configs:
        print(f"\nconnection: {cname}")
        for fname in demo_queries.VALUE_FUNCTIONS:
            cells = []
            for pname in payloads:
                o = next(x for x in outcomes if x["config"] == cname and x["function"] == fname and x["payload"] == pname)
                if o["status"] != "ok":
                    cell = o["status"].replace("Exception", "").replace("Error", " error").strip()
                elif o["leaked_file"]:
                    cell = "file read"
                elif o["leaked_users"]:
                    cell = "users leaked"
                elif o["dropped"]:
                    cell = "table dropped"
                else:
                    cell = f"{o['rows']} rows/{o['carriers']} carriers"
                cells.append(f"{pname}: {cell}")
            print(f"  {fname:16s} " + " | ".join(cells))
    bandit = shutil.which("bandit")
    if bandit:
        proc = subprocess.run([bandit, "-q", "-f", "json", "security/demo_queries.py"], capture_output=True,
                              text=True, cwd=HERE)
        found = json.loads(proc.stdout)["results"]
        src = (HERE / "security" / "demo_queries.py").read_text().splitlines()

        def owner(line_no: int) -> str:
            for i in range(line_no - 1, -1, -1):
                if src[i].startswith("def "):
                    return src[i][4:].split("(")[0]
            return "?"

        flagged = Counter(owner(r["line_number"]) for r in found)
        conf = {owner(r["line_number"]): r["issue_confidence"] for r in found}
        print("\nBandit B608 findings by function: "
              + ", ".join(f"{f} ({conf[f].lower()} confidence)" for f in flagged))
        for f in ("by_parameter", "_where", "sorted_allowlisted"):
            print(f"  {f}: {'flagged' if f in flagged else 'not flagged'}")
    shutil.rmtree(tmp)
    (RESULTS / "injection.json").write_text(json.dumps(outcomes, indent=1))
    return outcomes


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"duckdb {duckdb.__version__}, sqlglot {sqlglot.__version__}, sqlfluff {sqlfluff.__version__}")
    con = data.raw_connection("UTC")
    built = {
        "before": project.load_project(BEFORE),
        "after": project.load_project(AFTER),
    }
    built["before_sql"] = {n: m.sql for n, m in built["before"].items()}
    built["after"] = {n: project.Model(n, m.sql, m.deps) for n, m in built["after"].items()}
    staging = {k: v for k, v in built["after"].items() if k.startswith("stg_")}
    project.build(con, built["after"], "pa", "pa,main")
    project.build(con, {**staging, **built["before"]}, "pb", "pb,main")
    con.execute("set search_path = 'main'")
    built["cases"], built["defaults"] = testing.load_unit_cases(TESTS / "unit")
    built["cases_b"], _ = testing.load_unit_cases(TESTS / "unit", "cases_boundaries.yml")

    section_data(con)
    lint = section_lint()
    section_bugs(con, built)
    tests = section_tests(con, built)
    review = section_review(con, built, lint)
    section_security(built)

    figure_data = {
        "detection": [(label, engine, {k: bool(v) for k, v in caught.items()}) for label, engine, caught in built["detection_rows"]],
        "scores": tests["scores"],
        "operators": [(r["operator"], any(r[lv] for lv in ("build", "schema", "reconciliation", "unit")),
                       any(r[lv] for lv in ("build", "schema", "reconciliation", "unit", "boundary")),
                       r["unit"] and not (r["build"] or r["schema"] or r["reconciliation"])) for r in tests["records"]],
        "review": [(m, lines, d.touched_share, d.rows_left, d.rows_right) for m, lines, d in review],
    }
    (RESULTS / "figure_data.json").write_text(json.dumps(figure_data, indent=1, default=str))


if __name__ == "__main__":
    main()
