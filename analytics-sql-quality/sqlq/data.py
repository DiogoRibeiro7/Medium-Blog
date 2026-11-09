"""Load the nycflights13 tables into DuckDB exactly as a naive loader would.

The CSV files ship inside the ``nycflights13`` package (CC0). They are read
with DuckDB's type sniffing and no options, which is how most raw layers are
built in practice: columns that contain the text ``NA`` come out as VARCHAR.
The package's own ``__init__`` imports ``pkg_resources``, which is gone from
recent Python environments, so the files are located without importing it.
"""

from __future__ import annotations

import importlib.util
import tempfile
import zipfile
from pathlib import Path

import duckdb

RAW_TABLES = ("flights", "planes", "weather", "airports", "airlines")


def package_data_dir() -> Path:
    spec = importlib.util.find_spec("nycflights13")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("nycflights13 is not installed")
    return Path(spec.submodule_search_locations[0]) / "data"


def load_raw(
    con: duckdb.DuckDBPyConnection, schema: str = "main", typed: bool = False
) -> None:
    """Create raw_<table> for every source table in ``schema``.

    ``typed=False`` is the naive loader. ``typed=True`` is the one-line fix a
    reviewer would ask for: declare ``NA`` as the null marker, so numeric
    columns get numeric types and missing values become NULL.
    """
    reader = "read_csv('{path}', nullstr = 'NA')" if typed else "read_csv_auto('{path}')"
    src = package_data_dir()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(src / "flights.csv.zip") as archive:
            archive.extract("flights.csv", tmp)
        paths = {name: src / f"{name}.csv" for name in RAW_TABLES}
        paths["flights"] = Path(tmp) / "flights.csv"
        for name, path in paths.items():
            source = reader.format(path=path.as_posix())
            con.execute(f"create or replace table {schema}.raw_{name} as select * from {source}")


def raw_connection(time_zone: str = "UTC", typed: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"set TimeZone = '{time_zone}'")
    load_raw(con, typed=typed)
    return con
