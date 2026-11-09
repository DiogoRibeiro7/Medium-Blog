"""The serving path: features for one flight, computed the way a production
service would compute them two hours before departure.

This is deliberately separate code from the batch path in ``features.py``.
Each query is bound to the prediction time T and can only see rows whose
information was available at T: departures that had already happened, labels
of days that had fully finished, weather stamped at least an hour before T.
The parity test compares the two paths row by row.
"""

from __future__ import annotations

import math

import pandas as pd

from .features import SHRINK, Variant


def _rate(con, where: str, params: list) -> tuple[float, float]:
    n, k = con.execute(f"select count(*), coalesce(sum(disrupted), 0) from base where {where}", params).fetchone()
    return n, k


def serve_one(con, flight_id: int, v: Variant) -> dict:
    f = con.execute(
        "select flight_id, carrier, origin, dest, tailnum, distance, flight_date, "
        "sched_local, t_local, t_utc from base where flight_id = ?", [flight_id]
    ).fetchone()
    (fid, carrier, origin, dest, tailnum, distance, day, sched_local, t_local, t_utc) = f
    out = {"flight_id": fid, "carrier": carrier, "origin": origin, "dest": dest,
           # production reads the schedule, which is in local time
           "sched_hour": sched_local.hour, "day_of_week": day.isoweekday(), "distance": distance}
    plane = con.execute("select seats, year from raw_planes where tailnum = ?", [tailnum]).fetchone()
    out["seats"] = plane[0] if plane else math.nan
    out["plane_age"] = (2013 - plane[1]) if plane and plane[1] is not None else math.nan
    n, mean = con.execute(
        "select count(*), avg(dep_delay) from base where origin = ? "
        "and actual_dep_local >= ?::timestamp - interval 2 hour and actual_dep_local < ?::timestamp",
        [origin, t_local, t_local]).fetchone()
    out["cong_departures"], out["cong_mean_delay"] = n, (mean if mean is not None else math.nan)
    w = con.execute(
        "select precip, visib, case when wind_speed > 100 then null else wind_speed end, temp "
        "from raw_weather where origin = ? and time_hour <= ?::timestamptz - interval 1 hour "
        "order by time_hour desc limit 1", [origin, t_utc]).fetchone()
    for name, value in zip(("precip", "visib", "wind_speed", "temp"), w or (math.nan,) * 4):
        out[name] = math.nan if value is None else value
    # labels are complete for days that ended at least a day before the flight's date
    if v.global_encoding:
        span = "flight_date <= ?::date - interval 2 day"
        n_all, k_all = _rate(con, span, [day])
        prior = k_all / n_all if n_all else math.nan
        n_r, k_r = _rate(con, f"origin = ? and dest = ? and {span}", [origin, dest, day])
        n_t, k_t = _rate(con, f"tailnum = ? and {span}", [tailnum, day])
        out["route_rate"] = (k_r + SHRINK * prior) / (n_r + SHRINK)
        out["tail_rate"] = (k_t + SHRINK * prior) / (n_t + SHRINK) if tailnum else math.nan
    else:
        span = "flight_date between ?::date - interval 29 day and ?::date - interval 2 day"
        n_all, k_all = _rate(con, span, [day, day])
        prior = k_all / n_all if n_all else math.nan
        n_r, k_r = _rate(con, f"origin = ? and dest = ? and {span}", [origin, dest, day, day])
        out["route_rate"] = (k_r + SHRINK * prior) / (n_r + SHRINK)
    if v.leak_dep_delay:
        out["dep_delay"] = math.nan  # not yet known two hours before departure
    return out


def serve(con, flight_ids, v: Variant | None = None) -> pd.DataFrame:
    v = Variant() if v is None else v
    return pd.DataFrame([serve_one(con, int(i), v) for i in flight_ids])
