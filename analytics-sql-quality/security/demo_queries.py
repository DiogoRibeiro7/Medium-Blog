"""Seven ways to put a user-supplied carrier code into a query.

This file is deliberately unsafe: five functions build SQL from strings so the
study can measure what an injected value does to each, and what Bandit (a
static analyser for Python) reports. Do not copy the unsafe functions.
"""

from __future__ import annotations

COLUMNS = "carrier, flight_month, departures"
SORTABLE = {"carrier", "flight_month", "departures", "share_over_60_min"}


def by_fstring(con, carrier):
    return con.execute(
        f"select {COLUMNS} from monthly_long_delays where carrier = '{carrier}'"
    ).fetchall()


def by_concatenation(con, carrier):
    sql = "select " + COLUMNS + " from monthly_long_delays where carrier = '" + carrier + "'"
    return con.execute(sql).fetchall()


def by_format_method(con, carrier):
    sql = "select {} from monthly_long_delays where carrier = '{}'".format(COLUMNS, carrier)
    return con.execute(sql).fetchall()


def _where(column, value):
    return column + " = '" + value + "'"


def by_helper(con, carrier):
    clause = _where("carrier", carrier)
    return con.execute("select carrier, flight_month, departures from monthly_long_delays where " + clause).fetchall()


def by_parameter(con, carrier):
    return con.execute(
        "select carrier, flight_month, departures from monthly_long_delays where carrier = ?",
        [carrier],
    ).fetchall()


def sorted_unsafe(con, column):
    return con.execute(f"select {COLUMNS} from monthly_long_delays order by {column}").fetchall()


def sorted_allowlisted(con, column):
    if column not in SORTABLE:
        raise ValueError(f"cannot sort by {column!r}")
    return con.execute(f"select {COLUMNS} from monthly_long_delays order by {column}").fetchall()


VALUE_FUNCTIONS = {
    "f-string": by_fstring,
    "concatenation": by_concatenation,
    "str.format": by_format_method,
    "helper function": by_helper,
    "bound parameter": by_parameter,
}
