"""Fourteen known defects, each written as a rewrite of one month's rows.

Every fault is a SQL view over the delivered file. Most are *silent*: the rows still look like
taxi trips, every value is of the right type and inside its plausible range, and the file
loads without an error; ten are of that kind. Two are *loud* (``cents``, ``stale_copy``) and two touch the schema
(``airport_fee_int``, ``renamed_column``), as controls. Selections that need randomness use a
hash of the row, so a fault is the same on every run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fault:
    name: str
    kind: str          # silent, loud or schema
    story: str         # what happened upstream, in one line


def _cols(names: list[str]) -> dict[str, str]:
    return {n.lower(): '"' + n + '"' for n in names}


def _replace(names: list[str], **expr: str) -> str:
    """SELECT list that keeps every column in place, replacing the lower-case keys given."""
    out = []
    for n in names:
        k = n.lower()
        out.append(f'{expr[k]} AS "{n}"' if k in expr else f'"{n}"')
    return ", ".join(out)


def build(name: str, names: list[str], month: str) -> str:
    """The SQL of fault ``name`` for a file with columns ``names`` delivered for ``month``."""
    c = _cols(names)
    pu, do = c["tpep_pickup_datetime"], c["tpep_dropoff_datetime"]
    vendor, pay = c["vendorid"], c["payment_type"]
    h = "hash(" + ", ".join(c.values()) + ")"
    if name == "late_days":       # the last days of the month had not arrived when the file was cut
        return f"SELECT * FROM t WHERE day({pu}) < 25 OR {pu} < DATE '{month}-01'"
    if name == "vendor_missing":  # one technology vendor's feed failed for the whole month
        return f"SELECT * FROM t WHERE {vendor} IS DISTINCT FROM 1"
    if name == "duplicate_day":   # one day's batch was loaded twice
        return f"SELECT * FROM t UNION ALL SELECT * FROM t WHERE {pu} >= DATE '{month}-15' AND {pu} < DATE '{month}-16'"
    if name == "km_distance":     # a vendor update reports distance in kilometres
        return f"SELECT {_replace(names, trip_distance=c['trip_distance'] + ' * 1.609344')} FROM t"
    if name == "payment_swap":    # a code-mapping change swaps card (1) and cash (2)
        e = f"CASE WHEN {pay} = 1 THEN 2 WHEN {pay} = 2 THEN 1 ELSE {pay} END"
        return f"SELECT {_replace(names, payment_type=e)} FROM t"
    if name == "utc_shift":       # timestamps written in UTC instead of New York time
        shifted = _replace(names, tpep_pickup_datetime=pu + " + INTERVAL 5 HOUR",
                           tpep_dropoff_datetime=do + " + INTERVAL 5 HOUR")
        return f"SELECT {shifted} FROM t"
    if name == "tips_dropped":    # one vendor's card tips stop being itemised (total unchanged)
        e = f"CASE WHEN {vendor} = 2 AND {pay} = 1 THEN 0 ELSE {c['tip_amount']} END"
        return f"SELECT {_replace(names, tip_amount=e)} FROM t"
    if name == "null_passengers": # one vendor stops sending passenger counts
        e = f"CASE WHEN {vendor} = 2 THEN NULL ELSE {c['passenger_count']} END"
        return f"SELECT {_replace(names, passenger_count=e)} FROM t"
    if name == "zone_shift":      # an off-by-one in one vendor's zone lookup
        e = f"CASE WHEN {vendor} = 1 THEN {c['pulocationid']} + 1 ELSE {c['pulocationid']} END"
        return f"SELECT {_replace(names, pulocationid=e)} FROM t"
    if name == "partial_rows":    # an upload stopped early: a random 15% of rows missing
        return f"SELECT * FROM t WHERE {h} % 100 >= 15"
    if name == "cents":           # fares in cents (loud)
        return f"SELECT {_replace(names, fare_amount=c['fare_amount'] + ' * 100')} FROM t"
    if name == "stale_copy":      # last month's file delivered again under this month's name (loud)
        return "SELECT * FROM prev"
    if name == "airport_fee_int": # schema drift: the fee written as an integer column
        e = f"CAST(round({c['airport_fee']}) AS BIGINT)"
        return f"SELECT {_replace(names, airport_fee=e)} FROM t"
    if name == "renamed_column":  # schema drift: a column renamed upstream
        cols = ", ".join(f'"{n}" AS "tip"' if n.lower() == "tip_amount" else f'"{n}"' for n in names)
        return f"SELECT {cols} FROM t"
    raise KeyError(name)


FAULTS = {
    "late_days": Fault("late_days", "silent", "the last days of the month arrive after the file is cut"),
    "vendor_missing": Fault("vendor_missing", "silent", "one vendor's feed fails for the month"),
    "duplicate_day": Fault("duplicate_day", "silent", "one day's batch is loaded twice"),
    "partial_rows": Fault("partial_rows", "silent", "an upload stops early; 15% of rows are missing at random"),
    "km_distance": Fault("km_distance", "silent", "distance reported in kilometres"),
    "payment_swap": Fault("payment_swap", "silent", "card and cash codes swapped"),
    "utc_shift": Fault("utc_shift", "silent", "timestamps written in UTC"),
    "tips_dropped": Fault("tips_dropped", "silent", "one vendor's card tips no longer itemised"),
    "null_passengers": Fault("null_passengers", "silent", "one vendor stops sending passenger counts"),
    "zone_shift": Fault("zone_shift", "silent", "off-by-one in one vendor's pickup zone"),
    "cents": Fault("cents", "loud", "fares in cents"),
    "stale_copy": Fault("stale_copy", "loud", "last month's file delivered again"),
    "airport_fee_int": Fault("airport_fee_int", "schema", "airport fee written as an integer column"),
    "renamed_column": Fault("renamed_column", "schema", "tip_amount renamed upstream"),
}
