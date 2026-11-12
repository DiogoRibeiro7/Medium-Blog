import math

import numpy as np
import pandas as pd
import pytest

from feed import counts as C
from feed.contracts import CONTRACTS
from feed.profile import profile_file
from feed.rules import judge


def clean_frame(feed: str, n: int = 2000, month: str = "2024-03", seed: int = 1) -> pd.DataFrame:
    """A month that obeys every rule of the feed's contract."""
    rng = np.random.default_rng(seed)
    c = CONTRACTS[feed]
    start = pd.Timestamp(f"{month}-01")
    pick = start + pd.to_timedelta(rng.integers(0, 28 * 86400 - 7200, n), unit="s")
    cols = {}
    for f in c.fields:
        if f.kind == "timestamp":
            lower = f.name.lower()
            if "request" in lower:
                cols[f.name] = pick - pd.Timedelta(minutes=5)
            elif "on_scene" in lower:
                cols[f.name] = pick - pd.Timedelta(minutes=1)
            elif "drop" in lower:
                cols[f.name] = pick + pd.to_timedelta(rng.integers(60, 3600, n), unit="s")
            else:
                cols[f.name] = pick
        elif f.codes:
            vals = np.array(f.codes, dtype=object)
            v = vals[rng.integers(0, len(vals), n)]
            cols[f.name] = v.astype(float) if f.kind == "float" else v
        elif f.zone:
            cols[f.name] = rng.integers(1, 266, n)
        elif f.kind == "integer":
            cols[f.name] = rng.integers(0, 5, n)
        elif f.kind == "float":
            cols[f.name] = np.round(rng.uniform(0, 30, n), 2)
        else:
            cols[f.name] = np.array([f"B{rng.integers(1000, 9999):05d}" for _ in range(n)], dtype=object)
    df = pd.DataFrame(cols)
    if c.total:
        total, parts = c.total
        df[total] = df[list(parts)].sum(axis=1).round(2)
    if feed == "fhv":
        df["SR_Flag"] = pd.array(np.where(rng.random(n) < 0.1, 1, 0), dtype="Int64")
        df.loc[df["SR_Flag"] == 0, "SR_Flag"] = pd.NA
    return df


def run(df, tmp_path, feed, month="2024-03"):
    p = tmp_path / f"{feed}.parquet"
    df.to_parquet(p, index=False)
    return profile_file(p, month, feed)


@pytest.mark.parametrize("feed", sorted(CONTRACTS))
def test_clean_month_passes_every_rule(tmp_path, feed):
    p = run(clean_frame(feed), tmp_path, feed)
    fired = [r.name for r in judge(p) if r.fired]
    assert fired == [], fired


def test_strict_and_tolerant_disagree_on_a_small_violation(tmp_path):
    df = clean_frame("fhvhv")
    df.loc[:9, "tips"] = -1.0  # 0.5% of rows
    rules = {r.name: r for r in judge(run(df, tmp_path, "fhvhv"))}
    assert rules["ranges (strict)"].fired and not rules["ranges (tolerant)"].fired
    df.loc[:59, "tips"] = -1.0  # 3% of rows
    rules = {r.name: r for r in judge(run(df, tmp_path, "fhvhv"))}
    assert rules["ranges (tolerant)"].fired


def test_schema_case_and_type(tmp_path):
    df = clean_frame("yellow").rename(columns={"airport_fee": "Airport_fee"})
    df["passenger_count"] = df["passenger_count"].astype(float)
    rules = {r.name: r for r in judge(run(df, tmp_path, "yellow"))}
    assert rules["schema (strict)"].fired
    assert "found Airport_fee" in rules["schema (strict)"].detail
    assert rules["schema (tolerant)"].fired  # a float where an integer is implied is not tolerated
    df["passenger_count"] = df["passenger_count"].astype("int64")
    rules = {r.name: r for r in judge(run(df, tmp_path, "yellow"))}
    assert rules["schema (strict)"].fired and not rules["schema (tolerant)"].fired


def test_codes_zones_order_window_duplicates_total(tmp_path):
    df = clean_frame("yellow")
    df.loc[:99, "VendorID"] = 5
    df.loc[100:199, "PULocationID"] = 300
    df.loc[200:299, "tpep_dropoff_datetime"] = df.loc[200:299, "tpep_pickup_datetime"] - pd.Timedelta(minutes=1)
    df.loc[300:399, "tpep_pickup_datetime"] = pd.Timestamp("2024-04-02")
    df.loc[300:399, "tpep_dropoff_datetime"] = pd.Timestamp("2024-04-02 01:00")
    df.loc[400:499, "total_amount"] += 2.5
    df = pd.concat([df, df.iloc[500:600]], ignore_index=True)
    p = run(df, tmp_path, "yellow")
    assert p["counts"]["code|VendorID"] == 100 and p["offending"]["VendorID"][0] == ["5", 100]
    assert p["counts"]["zone|PULocationID"] == 100
    assert p["counts"]["order|tpep_pickup_datetime<=tpep_dropoff_datetime"] == 100
    assert p["counts"]["window|outside"] == 100
    assert p["counts"]["total|mismatch"] == 100
    assert p["counts"]["unique|duplicates"] == 100
    fired = {r.name for r in judge(p) if r.fired}
    assert {
        "code lists (tolerant)",
        "ranges (tolerant)",
        "month window (tolerant)",
        "reconciliation (tolerant)",
        "unique rows (tolerant)",
    } <= fired


def test_since_and_missing_required(tmp_path):
    df = clean_frame("fhvhv", month="2024-03").drop(columns=["cbd_congestion_fee"])
    assert not any(r.fired for r in judge(run(df, tmp_path, "fhvhv", "2024-03")))
    df2 = clean_frame("fhvhv", month="2025-03").drop(columns=["cbd_congestion_fee"])
    rules = {r.name: r for r in judge(run(df2, tmp_path, "fhvhv", "2025-03"))}
    assert rules["schema (strict)"].fired and rules["not null (tolerant)"].fired


def test_counts_detectors():
    months = [C.shift("2019-01", i) for i in range(90)]
    rng = np.random.default_rng(3)
    season = {m: 0.1 * math.sin(2 * math.pi * int(m[5:]) / 12) for m in months}
    rows = {m: int(1e6 * math.exp(season[m] + rng.normal(0, 0.03))) for m in months}
    report = {m: rows[m] * (1 + rng.normal(0, 0.001)) for m in months}
    clean = C.scores(rows, report)
    assert min(clean["volume"]) >= C.FIRST_EVAL and len(clean["seasonal"]) > 0
    lost = C.scores(rows, report, 0.05)
    m = sorted(lost["outside count"])[10]
    assert lost["outside count"][m] > C.K and lost["outside count, fixed 1%"][m] > 1
    thr = C.matched_threshold(clean["volume"])
    assert np.mean([v > thr for v in clean["volume"].values()]) <= C.BUDGET


def test_read_report(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text(
        'Year/Month,License Class,Trips Per Day\n2024-02-01,Yellow,"100,000"\n'
        "2024-02-01,FHV - Black Car,10\n2024-02-01,FHV - Livery,20\n2024-02-01,FHV - Lux Limo,1\n"
    )
    r = C.read_report(p)
    assert r["yellow"]["2024-02"] == 100_000 * 29 and r["fhv"]["2024-02"] == 31 * 29
