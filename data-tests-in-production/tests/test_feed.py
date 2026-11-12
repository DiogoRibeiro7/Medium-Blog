import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import month_file  # noqa: E402

from feed.checks import CHANGE, TABLE, kind, run_change, run_table  # noqa: E402
from feed.faults import FAULTS, build  # noqa: E402
from feed.profile import profile, profile_file  # noqa: E402
from feed.run import faulted, months, previous  # noqa: E402

MONTHS = months("2023-01:2024-01")


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    d = tmp_path_factory.mktemp("feed")
    files = {m: month_file(m, d / f"{m}.parquet", seed=i) for i, m in enumerate(MONTHS)}
    profiles = {m: profile_file(files[m], m) for m in MONTHS}
    return files, profiles


def test_month_helpers():
    assert months("2023-11:2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]
    assert previous("2024-01") == "2023-12"
    assert kind("BIGINT") == "integer" and kind("DOUBLE") == "float" and kind("TIMESTAMP_NS") == "timestamp"


def test_profile_counts(history):
    _, prof = history
    p = prof["2023-06"]
    assert p["n_rows"] == 20_000 and p["duplicates"] == 0 and p["recon_mismatch"] == 0
    assert sum(p["by_day"].values()) == 20_000 and len(p["by_day"]) == 30
    assert p["time"]["before_month"] == 0 and p["time"]["after_month"] == 0
    assert abs(sum(p["categories"]["payment_type"].values()) - 20_000) == 0


def test_every_fault_builds_and_changes_the_profile(history):
    files, prof = history
    out = faulted(files["2024-01"], files["2023-12"], "2024-01")
    base = prof["2024-01"]
    assert set(out) == set(FAULTS)
    assert out["late_days"]["n_rows"] < base["n_rows"] * 0.85
    assert out["duplicate_day"]["duplicates"] > 0
    assert out["km_distance"]["numeric"]["trip_distance"]["q"]["0.5"] == pytest.approx(
        base["numeric"]["trip_distance"]["q"]["0.5"] * 1.609344, rel=1e-3)
    assert out["payment_swap"]["categories"]["payment_type"]["1"] == base["categories"]["payment_type"]["2"]
    assert out["utc_shift"]["by_hour"] != base["by_hour"]
    assert out["tips_dropped"]["recon_mismatch"] > 0
    assert out["null_passengers"]["categories"]["passenger_count"]["null"] > 0
    assert out["stale_copy"]["time"]["before_month"] == prof["2023-12"]["n_rows"]
    assert dict(out["airport_fee_int"]["schema"])["Airport_fee"] == "BIGINT"
    assert "tip" in dict(out["renamed_column"]["schema"])


def test_checks_quiet_on_clean_history_and_loud_on_faults(history):
    files, prof = history
    hist = [prof[m] for m in MONTHS[:-1]]
    clean = run_change(prof["2024-01"], hist, k=5.0)
    assert set(clean) == set(CHANGE)
    assert sum(r.fired for r in clean.values()) <= 1
    out = faulted(files["2024-01"], files["2023-12"], "2024-01")
    # zone_shift is invisible here: the synthetic zones are uniform, so shifting them changes no distribution
    for name in ("late_days", "vendor_missing", "km_distance", "payment_swap", "utc_shift", "null_passengers"):
        fired = [c for c, r in run_change(out[name], hist, k=5.0).items() if r.fired]
        assert fired, name
    assert run_change(out["airport_fee_int"], hist, k=5.0)["schema change"].fired
    t = run_table(prof["2024-01"])
    assert set(t) == set(TABLE)
    assert not t["unique rows (strict)"].fired and not t["reconciliation (strict)"].fired
    assert run_table(out["duplicate_day"])["unique rows (strict)"].fired
    assert run_table(out["cents"])["volume floor"].fired  # 20,000 rows are below the floor in any case


def test_build_rejects_unknown_fault():
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT 1 AS VendorID")
    with pytest.raises(KeyError):
        build("nope", ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "payment_type"], "2024-01")
    assert profile  # imported for completeness
