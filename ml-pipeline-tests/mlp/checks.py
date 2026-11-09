"""The test layers, written so that each can be pointed at any pipeline variant.

Each check returns a ``Check``: passed or failed, and a short detail. The
pytest suite in ``tests/`` runs them against the reviewed pipeline; the
analysis script runs them against every defect to fill the detection matrix.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.metrics import roc_auc_score

from . import data, model, serving
from .features import Variant, build_features, feature_sql


@dataclass(frozen=True)
class Check:
    layer: str
    name: str
    passed: bool
    detail: str = ""


# ------------------------------------------------------------ data validation
def feature_schema(n_expected: int) -> pa.DataFrameSchema:
    rng = lambda lo, hi: pa.Check.in_range(lo, hi)
    return pa.DataFrameSchema(
        {
            "flight_id": pa.Column(int, unique=True),
            "disrupted": pa.Column(int, pa.Check.isin([0, 1])),
            "sched_hour": pa.Column(int, rng(0, 23)),
            "day_of_week": pa.Column(int, rng(1, 7)),
            "distance": pa.Column(int, pa.Check.gt(0)),
            "plane_age": pa.Column(float, rng(0, 60), nullable=True),
            "cong_departures": pa.Column(int, pa.Check.ge(0)),
            "precip": pa.Column(float, rng(0, 5), nullable=True),
            "visib": pa.Column(float, rng(0, 10), nullable=True),
            "wind_speed": pa.Column(float, rng(0, 100), nullable=True),
            "route_rate": pa.Column(float, rng(0, 1), nullable=True),
        },
        checks=pa.Check(lambda df: len(df) == n_expected,
                        error=f"one row per scheduled flight ({n_expected:,})"),
        coerce=True,
    )


def check_data_validation(con, frames, v: Variant) -> Check:
    n = con.execute("select count(*) from base").fetchone()[0]
    try:
        feature_schema(n).validate(frames[v], lazy=True)
        return Check("data validation", "feature schema", True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        return Check("data validation", "feature schema", False,
                     "; ".join(sorted(set(cases["check"].astype(str))))[:200])


# ------------------------------------------------------------------ unit tests
RAW_TABLES = ("flights", "planes", "weather", "airports", "airlines")


def _raw_ddl(con) -> dict[str, list[tuple[str, str]]]:
    return {t: [(c, ty) for c, ty, *_ in con.execute(f"describe raw_{t}").fetchall()] for t in RAW_TABLES}


FLIGHT_DEFAULTS = {
    "year": 2013, "month": 1, "day": 15, "dep_time": None, "sched_dep_time": 1200, "dep_delay": 0,
    "arr_time": None, "sched_arr_time": 1500, "arr_delay": 0, "carrier": "AA", "flight": 1,
    "tailnum": "N1", "origin": "JFK", "dest": "MIA", "air_time": 150, "distance": 1089,
    "hour": 12, "minute": 0, "time_hour": None,
}


def flight(**kw) -> dict:
    row = {**FLIGHT_DEFAULTS, **kw}
    if row["time_hour"] is None:  # the scheduled hour as a UTC instant (EST in January)
        local = dt.datetime(row["year"], row["month"], row["day"], row["hour"])
        row["time_hour"] = (local + dt.timedelta(hours=5)).strftime("%Y-%m-%d %H:00:00+00")
    return row


def fixture_connection(ddl, flights: list[dict], planes=(), weather=()) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("set TimeZone = 'UTC'")
    rows = {"flights": flights, "planes": list(planes), "weather": list(weather), "airports": [], "airlines": []}
    for t, cols in ddl.items():
        con.execute(f"create table raw_{t} ({', '.join(f'{c} {ty}' for c, ty in cols)})")
        for r in rows[t]:
            values = [r.get(c) for c, _ in cols]
            con.execute(f"insert into raw_{t} values ({', '.join('?' * len(cols))})", values)
    con.execute(data.BASE_SQL)
    return con


def unit_cases():
    """(name, flights, planes, weather, target flight number, assertion on its feature row)."""
    return [
        ("congestion counts only departures that left before T",
         [flight(flight=1, hour=12, minute=0),                       # target, T = 10:00
          flight(flight=2, hour=7, minute=0, dep_delay=30),          # left 07:30, before window
          flight(flight=3, hour=9, minute=0, dep_delay=0),           # left 09:00, counted
          flight(flight=4, hour=9, minute=30, dep_delay=29),         # left 09:59, counted
          flight(flight=5, hour=10, minute=0, dep_delay=30),         # left 10:30, after T
          flight(flight=6, hour=12, minute=10, dep_delay=45)],       # same hour as target, later
         [], [], 1,
         lambda r: r["cong_departures"] == 2 and abs(r["cong_mean_delay"] - 14.5) < 1e-9),
        ("hour is the local scheduled hour",
         [flight(flight=1, hour=20, minute=30, sched_dep_time=2030)], [], [], 1,
         lambda r: r["sched_hour"] == 20),
        ("a cancelled flight keeps its row and is labelled disrupted",
         [flight(flight=1, dep_delay=None, arr_delay=None, air_time=None)], [], [], 1,
         lambda r: r["disrupted"] == 1),
        ("the route rate ignores the flight's own day and later days",
         [flight(flight=1, day=20, arr_delay=0)]
         + [flight(flight=10 + i, day=20, hour=6, arr_delay=120) for i in range(5)]
         + [flight(flight=20 + i, day=25, hour=6, arr_delay=120) for i in range(5)]
         + [flight(flight=30 + i, day=10, hour=6, dest="BOS", arr_delay=0) for i in range(5)],
         [], [], 1,
         lambda r: abs(r["route_rate"] - 0.0) < 1e-9),
        ("a flight's own outcome does not change its features",
         None, [], [], 1, None),  # handled specially below
    ]


def _row(con, v: Variant, flight_no: int) -> pd.Series | None:
    """The feature row of the fixture flight with this flight number."""
    df = con.execute(feature_sql(v)).df()
    order = con.execute(
        "select row_number() over (order by time_hour, carrier, flight, origin, dest) as fid, flight "
        "from raw_flights").df()
    target = order.loc[order["flight"] == flight_no, "fid"]
    hit = df[df["flight_id"].isin(target)]
    return None if hit.empty else hit.iloc[0]


def check_unit_tests(ddl, v: Variant) -> list[Check]:
    out = []
    for name, flights, planes, weather, target, ok in unit_cases():
        if flights is None:  # outcome invariance: same flight, three different outcomes
            rows = []
            for dep, arr in ((0, 0), (90, 95), (None, None)):
                con = fixture_connection(ddl, [
                    flight(flight=1, hour=14, dep_delay=dep, arr_delay=arr),
                    flight(flight=2, hour=11, dep_delay=10, arr_delay=5),
                    flight(flight=3, day=5, hour=14, arr_delay=30),
                ])
                rows.append(_row(con, v, 1))
            cols = [c for c in v.feature_columns() if c not in ("carrier", "origin", "dest")]
            same = all(r is not None for r in rows) and all(
                np.allclose(pd.to_numeric(rows[0][cols], errors="coerce").to_numpy(dtype=float),
                            pd.to_numeric(r[cols], errors="coerce").to_numpy(dtype=float), equal_nan=True)
                for r in rows[1:])
            out.append(Check("unit tests", name, bool(same)))
            continue
        con = fixture_connection(ddl, flights, planes, weather)
        row = _row(con, v, target)
        out.append(Check("unit tests", name, row is not None and bool(ok(row))))
    return out


# ------------------------------------------------------ future perturbation
def _perturb(con, cutoff_local: dt.datetime, seed: int) -> None:
    """Replace, in schema ``p``, everything that was not known at the cut-off:
    departures that had not happened, arrivals of the cut-off day and the day
    before, and weather stamped later than an hour before the cut-off."""
    rng = np.random.default_rng(seed)
    con.execute("drop schema if exists p cascade"); con.execute("create schema p")
    con.execute("create table p.base as select * from main.base")
    con.execute("create table p.raw_weather as select * from main.raw_weather")
    con.execute("create table p.raw_planes as select * from main.raw_planes")
    c = cutoff_local
    n = con.execute("select count(*) from p.base").fetchall()[0][0]
    noise = pd.DataFrame({"i": np.arange(1, n + 1), "u1": rng.random(n), "u2": rng.random(n)})
    con.register("noise_df", noise)
    con.execute("create temp table noise as select * from noise_df")
    con.unregister("noise_df")
    # a flight not yet gone at the cut-off gets a new departure strictly after it
    con.execute("""
        update p.base as b set
            dep_delay = case
                when x.u1 < 0.1 then null
                else greatest(cast(date_diff('minute', b.sched_local, ?::timestamp) as bigint) + 1, -10)
                     + cast(x.u2 * 300 as bigint) end
        from noise as x
        where x.i = b.flight_id
          and (b.actual_dep_local is null or b.actual_dep_local >= ?::timestamp)
          and b.sched_local >= ?::timestamp - interval 1 day
    """, [c, c, c])
    con.execute("""
        update p.base as b set
            arr_delay = case when b.dep_delay is null or x.u1 < 0.05 then null
                             else b.dep_delay + cast(x.u2 * 120 as bigint) - 60 end
        from noise as x
        where x.i = b.flight_id and b.flight_date >= cast(?::timestamp as date) - interval 1 day
    """, [c])
    con.execute("""
        update p.base set
            actual_dep_local = sched_local + to_minutes(dep_delay),
            disrupted = cast(dep_delay is null or arr_delay is null or arr_delay >= 15 as integer)
    """)
    con.execute("""
        update p.raw_weather set precip = precip + 0.5, visib = visib / 2,
                                 wind_speed = wind_speed + 15, temp = temp - 20
        where time_hour > timezone('America/New_York', ?::timestamp) - interval 1 hour
    """, [c])
    con.execute("drop table noise")


def check_future_perturbation(con, frames, v: Variant, cutoffs: list[dt.datetime]) -> Check:
    base = frames[v].set_index("flight_id")
    cols = [c for c in v.feature_columns() if c not in ("carrier", "origin", "dest")]
    changed, targets = 0, 0
    for i, c in enumerate(cutoffs):
        ids = con.execute(
            "select flight_id from base where t_local > ?::timestamp - interval 20 minute "
            "and t_local <= ?::timestamp", [c, c]).df()["flight_id"]
        _perturb(con, c, seed=i)
        con.execute("set search_path = 'p,main'")
        pert = build_features(con, v).set_index("flight_id")
        con.execute("set search_path = 'main'")
        for fid in ids:
            targets += 1
            if fid not in pert.index or fid not in base.index:
                changed += int((fid in pert.index) != (fid in base.index))
                continue
            a = pd.to_numeric(base.loc[fid, cols], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(pert.loc[fid, cols], errors="coerce").to_numpy(dtype=float)
            changed += int(not np.allclose(a, b, equal_nan=True))
    con.execute("drop schema if exists p cascade")
    return Check("future perturbation", "features unchanged when the future changes", changed == 0,
                 f"{changed} of {targets} flights changed")


# ------------------------------------------------------------- null labels
def check_null_label(con, v: Variant, perms: int = 5, max_iter: int = 100) -> tuple[Check, list[float]]:
    """Permute every label, rebuild the features, train and score on the
    test months. Nothing real is left to learn, so the AUC must be near 0.5."""
    aucs = []
    for k in range(perms):
        con.execute("drop schema if exists p cascade"); con.execute("create schema p")
        labels = con.execute("select flight_id, disrupted from main.base order by flight_id").df()
        labels["disrupted"] = np.random.default_rng(1000 + k).permutation(labels["disrupted"].to_numpy())
        con.register("perm_df", labels)
        con.execute("create table p.base as select * from main.base")
        con.execute("update p.base set disrupted = x.disrupted from perm_df as x "
                    "where x.flight_id = p.base.flight_id")
        con.unregister("perm_df")
        con.execute("set search_path = 'p,main'")
        df = build_features(con, v)
        con.execute("set search_path = 'main'")
        s = model.split(df, v, seed=k)
        cols = v.feature_columns()
        m = model.fit(s.train, cols, seed=k, max_iter=max_iter)
        aucs.append(roc_auc_score(s.test["disrupted"], model.scores(m, s.test, cols)))
    con.execute("drop schema if exists p cascade")
    mean = float(np.mean(aucs))
    return Check("null labels", "AUC with permuted labels near 0.5", mean < 0.51,
                 f"mean {mean:.3f} over {perms} permutations"), aucs


# ------------------------------------------------------------- split audit
def check_split_audit(con, frames, v: Variant) -> list[Check]:
    df = frames[v]
    s = model.split(df, v)
    out = []
    ordered = s.train["flight_date"].max() < s.valid["flight_date"].min() <= s.valid["flight_date"].max() \
        < s.test["flight_date"].min()
    out.append(Check("split audit", "train, validation and test are ordered in time", bool(ordered)))
    n_test = con.execute(
        f"select count(*) from base where month between {model.TEST[0]} and {model.TEST[1]}").fetchone()[0]
    complete = len(s.test) == n_test and not v.random_split
    out.append(Check("split audit", "test set is every scheduled flight of the test months", bool(complete),
                     f"{len(s.test):,} rows, {n_test:,} scheduled"))
    out.append(Check("split audit", "threshold chosen on data disjoint from the test set",
                     not v.threshold_on_test))
    return out


# ------------------------------------------------------------ parity test
def check_parity(con, frames, v: Variant, n: int = 300) -> Check:
    ids = con.execute(f"select flight_id from (select flight_id from base where month = 12) "
                      f"using sample reservoir({n} rows) repeatable (11)").df()["flight_id"]
    batch = frames[v].set_index("flight_id")
    served = serving.serve(con, ids, v).set_index("flight_id")
    cols = [c for c in v.feature_columns() if c not in ("carrier", "origin", "dest")]
    bad = 0
    for fid in ids:
        if fid not in batch.index:
            bad += 1
            continue
        a = pd.to_numeric(batch.loc[fid, cols], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(served.loc[fid, cols], errors="coerce").to_numpy(dtype=float)
        bad += int(not np.allclose(a, b, equal_nan=True, atol=1e-9))
    return Check("train/serve parity", "batch features equal serving features", bad == 0,
                 f"{bad} of {len(ids)} December flights differ")


# ---------------------------------------------------------- metric gate
def check_metric_gate(result: model.Result, floor: float = 0.67) -> Check:
    return Check("metric gate", f"reported test AUC at least {floor}", result.reported.auc >= floor,
                 f"AUC {result.reported.auc:.3f}")


# -------------------------------------------------------- property-based
def check_outcome_property(ddl, v: Variant, max_examples: int = 25) -> Check:
    """Hypothesis version of the outcome-invariance unit test: for any outcome
    of the target flight (cancelled, diverted, early, late), its features are
    the features it has with an on-time outcome."""
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    cols = [c for c in v.feature_columns() if c not in ("carrier", "origin", "dest")]
    others = [flight(flight=2, hour=11, dep_delay=10, arr_delay=5), flight(flight=3, day=5, hour=14, arr_delay=30)]
    reference = _row(fixture_connection(ddl, [flight(flight=1, hour=14)] + others), v, 1)
    found: list[str] = []

    outcome = st.one_of(
        st.just((None, None)),
        st.tuples(st.integers(-30, 600), st.none()),
        st.integers(-30, 600).flatmap(lambda d: st.tuples(st.just(d), st.integers(d - 40, d + 120))),
    )

    @settings(max_examples=max_examples, deadline=None, database=None, derandomize=True,
              suppress_health_check=list(HealthCheck))
    @given(outcome)
    def prop(o):
        dep, arr = o
        row = _row(fixture_connection(ddl, [flight(flight=1, hour=14, dep_delay=dep, arr_delay=arr)] + others), v, 1)
        ok = row is not None and np.allclose(
            pd.to_numeric(reference[cols], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float), equal_nan=True)
        if not ok:
            found.append(f"dep_delay={dep}, arr_delay={arr}")
        assert ok

    try:
        prop()
        return Check("property-based", "features independent of the flight's own outcome", True)
    except AssertionError:
        return Check("property-based", "features independent of the flight's own outcome", False,
                     f"smallest failing outcome: {found[-1]}")
