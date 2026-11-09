"""Load nycflights13 into DuckDB with declared types and build the base table.

Every row of ``base`` is one scheduled departure. Times are kept in two forms:
local New York wall-clock time (naive TIMESTAMP) for reasoning about the
schedule, and UTC instants (TIMESTAMPTZ) for joining to the weather.

The prediction is made two hours before the scheduled departure: ``t_local``.
The label ``disrupted`` is 1 when the flight was cancelled, diverted, or
arrived 15 minutes late or more.
"""

from __future__ import annotations

import importlib.util
import tempfile
import zipfile
from pathlib import Path

import duckdb

LEAD_HOURS = 2


def package_data_dir() -> Path:
    spec = importlib.util.find_spec("nycflights13")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("nycflights13 is not installed")
    return Path(spec.submodule_search_locations[0]) / "data"


def load_raw(con: duckdb.DuckDBPyConnection) -> None:
    src = package_data_dir()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(src / "flights.csv.zip") as archive:
            archive.extract("flights.csv", tmp)
        paths = {"flights": Path(tmp) / "flights.csv"}
        for name in ("planes", "weather", "airports", "airlines"):
            paths[name] = src / f"{name}.csv"
        for name, path in paths.items():
            con.execute(
                f"create or replace table raw_{name} as "
                f"select * from read_csv('{path.as_posix()}', nullstr = 'NA')"
            )


BASE_SQL = f"""
create or replace table base as
with f as (
    select
        row_number() over (order by time_hour, carrier, flight, origin, dest) as flight_id,
        carrier, origin, dest, tailnum, distance,
        make_date(year, month, day) as flight_date,
        month,
        hour as sched_hour_local,
        make_timestamp(year, month, day, hour, minute, 0) as sched_local,
        time_hour as sched_hour_utc,
        dep_delay, arr_delay
    from raw_flights
)
select
    *,
    sched_local - interval {LEAD_HOURS} hour as t_local,
    timezone('America/New_York', sched_local) - interval {LEAD_HOURS} hour as t_utc,
    sched_local + to_minutes(cast(dep_delay as bigint)) as actual_dep_local,
    cast(dep_delay is null or arr_delay is null or arr_delay >= 15 as integer) as disrupted
from f
"""


def connect(time_zone: str = "UTC") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"set TimeZone = '{time_zone}'")
    load_raw(con)
    con.execute(BASE_SQL)
    return con
