"""Feature construction for the batch (training) path.

``build_features`` returns one row per scheduled flight with the label and the
features. A ``Variant`` switches on the defects of the shipped pipeline one at
a time; ``Variant()`` with no flags is the reviewed pipeline.

Reviewed features, all known two hours before departure (time T):
  carrier, origin, dest, scheduled local hour, day of week, distance;
  aircraft seats and age from the registry;
  congestion: departures from the same airport that actually left in
  [T - 2h, T), their number and mean departure delay;
  weather: the last observation at the origin stamped at least an hour before T;
  route disruption rate over the 28 days ending two days before the flight,
  shrunk towards the all-route rate of the same window.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import duckdb
import pandas as pd

CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "sched_hour", "day_of_week", "distance", "seats", "plane_age",
    "cong_departures", "cong_mean_delay",
    "precip", "visib", "wind_speed", "temp",
    "route_rate",
]
SHRINK = 20  # pseudo-flights for the route-rate prior


@dataclass(frozen=True)
class Variant:
    """Defects of the shipped pipeline. All False is the reviewed pipeline."""

    leak_dep_delay: bool = False        # the flight's own departure delay as a feature
    unlagged_congestion: bool = False   # congestion over the scheduled hour, future included
    global_encoding: bool = False       # route and tail rates from the whole year's labels
    random_split: bool = False          # shuffled split instead of a split in time
    drop_incomplete: bool = False       # cancelled and diverted flights dropped
    utc_hour: bool = False              # hour feature from the UTC timestamp
    threshold_on_test: bool = False     # decision threshold tuned on the test set

    @property
    def name(self) -> str:
        on = [f.name for f in fields(self) if getattr(self, f.name)]
        return "reviewed" if not on else "+".join(on)

    def feature_columns(self) -> list[str]:
        cols = CATEGORICAL + NUMERIC
        if self.leak_dep_delay:
            cols = cols + ["dep_delay"]
        if self.global_encoding:
            cols = cols + ["tail_rate"]
        return cols


DEFECTS = [
    "leak_dep_delay", "unlagged_congestion", "global_encoding", "random_split",
    "drop_incomplete", "utc_hour",
]  # threshold_on_test is studied separately: its effect depends on the shift
SHIPPED = Variant(**{d: True for d in DEFECTS})
FEATURE_FLAGS = ["leak_dep_delay", "unlagged_congestion", "global_encoding", "drop_incomplete", "utc_hour"]


def feature_key(v: Variant) -> Variant:
    """The variant reduced to the flags that change the feature table."""
    return Variant(**{f: getattr(v, f) for f in FEATURE_FLAGS})


class FeatureCache(dict):
    """Feature tables built on first use, keyed by ``feature_key``."""

    def __init__(self, con):
        super().__init__()
        self.con = con

    def __missing__(self, v: Variant) -> pd.DataFrame:
        key = feature_key(v)
        if key not in self.keys():
            dict.__setitem__(self, key, build_features(self.con, key))
        return dict.__getitem__(self, key)

    def __getitem__(self, v: Variant) -> pd.DataFrame:
        key = feature_key(v)
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self.__missing__(v)


def _congestion_sql(v: Variant) -> str:
    if v.unlagged_congestion:
        # every departure from the airport in the flight's scheduled hour,
        # the flight itself and the later ones included
        return """
        congestion as (
            select
                f.flight_id,
                count(o.dep_delay) as cong_departures,
                avg(o.dep_delay) as cong_mean_delay
            from base as f
            join base as o
                on o.origin = f.origin
                and o.flight_date = f.flight_date
                and o.sched_hour_local = f.sched_hour_local
            group by f.flight_id
        )"""
    # cumulative departures per airport, then two as-of lookups: strictly
    # before T and strictly before T - 2h; the difference is the window
    return """
        departed as (
            select
                origin, actual_dep_local,
                count(*) over w as cum_n,
                sum(dep_delay) over w as cum_delay
            from base
            where actual_dep_local is not null
            window w as (partition by origin order by actual_dep_local
                         range between unbounded preceding and current row)
        ),
        congestion as (
            select
                f.flight_id,
                coalesce(hi.cum_n, 0) - coalesce(lo.cum_n, 0) as cong_departures,
                (coalesce(hi.cum_delay, 0) - coalesce(lo.cum_delay, 0))
                / nullif(coalesce(hi.cum_n, 0) - coalesce(lo.cum_n, 0), 0) as cong_mean_delay
            from base as f
            asof left join departed as hi
                on hi.origin = f.origin and f.t_local > hi.actual_dep_local
            asof left join departed as lo
                on lo.origin = f.origin and f.t_local - interval 2 hour > lo.actual_dep_local
        )"""


def _route_sql(v: Variant) -> str:
    if v.global_encoding:
        return f"""
        route as (
            select origin, dest,
                   (sum(disrupted) + {SHRINK} * (select avg(disrupted) from base))
                   / (count(*) + {SHRINK}) as route_rate
            from base group by origin, dest
        ),
        tail as (
            select tailnum,
                   (sum(disrupted) + {SHRINK} * (select avg(disrupted) from base))
                   / (count(*) + {SHRINK}) as tail_rate
            from base where tailnum is not null group by tailnum
        )"""
    return f"""
        route_day as (
            select origin, dest, flight_date, count(*) as n, sum(disrupted) as k
            from base group by origin, dest, flight_date
        ),
        all_day as (
            select flight_date, count(*) as n, sum(disrupted) as k
            from base group by flight_date
        ),
        all_window as (
            select
                flight_date,
                sum(k) over w / nullif(sum(n) over w, 0) as prior
            from all_day
            window w as (order by flight_date
                         range between interval 29 day preceding and interval 2 day preceding)
        ),
        route_window as (
            select
                origin, dest, flight_date,
                coalesce(sum(n) over w, 0) as n28,
                coalesce(sum(k) over w, 0) as k28
            from route_day
            window w as (partition by origin, dest order by flight_date
                         range between interval 29 day preceding and interval 2 day preceding)
        ),
        route as (
            select
                r.origin, r.dest, r.flight_date,
                (r.k28 + {SHRINK} * a.prior) / (r.n28 + {SHRINK}) as route_rate
            from route_window as r
            join all_window as a using (flight_date)
        )"""


def feature_sql(v: Variant) -> str:
    hour = "hour(f.sched_hour_utc)" if v.utc_hour else "f.sched_hour_local"
    route_join = (
        "left join route as r on r.origin = f.origin and r.dest = f.dest\n"
        "left join tail as tl on tl.tailnum = f.tailnum"
        if v.global_encoding
        else "left join route as r on r.origin = f.origin and r.dest = f.dest "
             "and r.flight_date = f.flight_date"
    )
    extra = ", f.dep_delay" if v.leak_dep_delay else ""
    extra += ", tl.tail_rate" if v.global_encoding else ""
    where = "where f.dep_delay is not null and f.arr_delay is not null" if v.drop_incomplete else ""
    return f"""
    with {_congestion_sql(v)},
    {_route_sql(v)},
    weather as (
        -- one observation (EWR, 12 Feb 2013) reports 1,048 mph; treat it as missing
        select origin, time_hour, precip, visib,
               case when wind_speed > 100 then null else wind_speed end as wind_speed, temp
        from raw_weather
    )
    select
        f.flight_id, f.flight_date, f.month, f.disrupted,
        f.carrier, f.origin, f.dest,
        {hour} as sched_hour,
        isodow(f.flight_date) as day_of_week,
        f.distance,
        p.seats,
        2013 - p.year as plane_age,
        coalesce(c.cong_departures, 0) as cong_departures,
        c.cong_mean_delay,
        w.precip, w.visib, w.wind_speed, w.temp,
        r.route_rate{extra}
    from base as f
    left join raw_planes as p on p.tailnum = f.tailnum
    left join congestion as c on c.flight_id = f.flight_id
    {route_join}
    asof left join weather as w
        on w.origin = f.origin and w.time_hour <= f.t_utc - interval 1 hour
    {where}
    order by f.flight_id
    """


def build_features(con: duckdb.DuckDBPyConnection, v: Variant | None = None) -> pd.DataFrame:
    v = Variant() if v is None else v
    df = con.execute(feature_sql(v)).df()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df
