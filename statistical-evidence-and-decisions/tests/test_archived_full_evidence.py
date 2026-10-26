"""Verify the preserved detector evidence summary and interpretation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "2026-10-24-detector-evidence"
    / "summary.json"
)


@pytest.fixture(scope="module")
def snapshot() -> dict[str, object]:
    """Read the committed, machine-readable completed-run evidence."""
    with RESULT_PATH.open(encoding="utf-8") as source:
        result = json.load(source)
    assert isinstance(result, dict)
    return result


def test_execution_and_sample_sizes_are_documented(
    snapshot: dict[str, object],
) -> None:
    """Never report an unexecuted smoke test as a full simulation."""
    run = snapshot["run"]
    design = snapshot["design"]
    assert isinstance(run, dict) and isinstance(design, dict)
    assert run["workflow_run_id"] == 40089779139
    assert run["completed"] is True and run["successful"] is True
    assert design["evaluation_streams_per_regime"] == 100
    assert design["calibration_streams_per_null"] == 199


def test_regret_difference_and_verdict_are_consistent(
    snapshot: dict[str, object],
) -> None:
    """Check candidate deltas against archived per-policy regret averages."""
    averages = snapshot["drift_mean_regret"]
    candidates = snapshot["candidates"]
    design = snapshot["design"]
    assert isinstance(averages, dict)
    assert isinstance(candidates, dict)
    assert isinstance(design, dict)
    for name in ("GLR", "BOCPD"):
        candidate = candidates[name]
        assert isinstance(candidate, dict)
        assert candidate["mean_delta_regret"] == pytest.approx(
            averages[name] - averages["CUSUM"], abs=1e-12
        )
        assert candidate["adjusted_lower"] <= candidate["mean_delta_regret"]
        assert candidate["mean_delta_regret"] <= candidate["adjusted_upper"]
        assert candidate["adjusted_upper"] > -design["meaningful_gain"]
        assert candidate["status"] == "material_gain_not_supported"


def test_null_alarm_rates_and_provenance(
    snapshot: dict[str, object],
) -> None:
    """Preserve false-alarm limitations and source integrity metadata."""
    candidates = snapshot["candidates"]
    design = snapshot["design"]
    hashes = snapshot["hashes_sha256"]
    verification = snapshot["verified"]
    assert all(isinstance(v, dict) for v in candidates.values())
    for candidate in candidates.values():
        assert candidate["independent_null_streams"] == 100
        assert candidate["empirical_false_alarm"] == pytest.approx(
            candidate["null_alarms"] / 100
        )
        assert candidate["simultaneous_upper_false_alarm"] > design[
            "false_alarm_ceiling"
        ]
    assert len(hashes) == 5
    assert all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        for digest in hashes.values()
    )
    # The experiment's original manifest and verdict.csv were checked by SHA-256.
    assert hashes["verdict.csv"] == (
        "9cc0089cb55076b10e313dab0255a728a3090b"
        "8da7610739e66cf1813c876a5d"
    )
    assert verification["checksums_matched"] is True
