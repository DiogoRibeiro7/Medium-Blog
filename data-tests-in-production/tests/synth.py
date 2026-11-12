"""A small synthetic month with the shape of a yellow-taxi file, for the unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def month_file(month: str, path: Path, n: int = 20_000, seed: int = 0, airport_int: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{month}-01")
    days = (start + pd.offsets.MonthBegin(1) - start).days
    hour = rng.choice(24, n, p=np.r_[np.full(6, 0.01), np.full(18, 0.94 / 18)])
    t0 = start + pd.to_timedelta(rng.integers(0, days, n), "D") + pd.to_timedelta(hour, "h") \
        + pd.to_timedelta(rng.integers(0, 3600, n), "s")
    dist = np.round(rng.gamma(1.5, 2.0, n), 2)
    dur = pd.to_timedelta((dist * 180 + rng.integers(120, 600, n)).astype(int), "s")
    vendor = rng.choice([1, 2], n, p=[0.3, 0.7])
    pay = rng.choice([1, 2, 3, 4], n, p=[0.75, 0.2, 0.03, 0.02])
    fare = np.round(3.0 + 2.5 * dist + rng.normal(0, 0.5, n).clip(-0.5, 2), 2)
    tip = np.where(pay == 1, np.round(fare * rng.uniform(0.1, 0.25, n), 2), 0.0)
    extra, mta, imp, cong = np.full(n, 1.0), np.full(n, 0.5), np.full(n, 1.0), np.full(n, 2.5)
    pu = rng.integers(1, 266, n)
    airport = np.where(np.isin(pu, [132, 138]), 1.75, 0.0)
    total = np.round(fare + extra + mta + tip + imp + cong + airport, 2)
    df = pd.DataFrame({
        "VendorID": vendor.astype("int32"), "tpep_pickup_datetime": t0, "tpep_dropoff_datetime": t0 + dur,
        "passenger_count": rng.choice([1, 2, 3, 0], n, p=[0.7, 0.2, 0.08, 0.02]).astype("int64"),
        "trip_distance": dist, "RatecodeID": np.ones(n, "int64"),
        "store_and_fwd_flag": rng.choice(["N", "Y"], n, p=[0.99, 0.01]),
        "PULocationID": pu.astype("int32"), "DOLocationID": rng.integers(1, 266, n).astype("int32"),
        "payment_type": pay.astype("int64"), "fare_amount": fare, "extra": extra, "mta_tax": mta, "tip_amount": tip,
        "tolls_amount": np.zeros(n), "improvement_surcharge": imp, "total_amount": total, "congestion_surcharge": cong,
        "Airport_fee": airport.round().astype("int64") if airport_int else airport,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
