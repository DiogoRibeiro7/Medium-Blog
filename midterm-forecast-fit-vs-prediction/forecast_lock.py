"""Content-addressed registration of the October 2026 House forecast.

The model was estimated with information frozen on 2026-10-31. This manifest
is *registered* on 2026-11-01, before election day (2026-11-03). Those two
dates have different meanings and must not be retroactively conflated.

Only input, code and computed-evidence files are fingerprinted. Historical
records must not contain a 2026 outcome. The source commit is an immutable
Git object reference; the manifest is not a third-party notarization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from analysis import ROOT, fit_model, prediction_interval, read_elections, read_scenario

SOURCE_COMMIT = "fe98742efb86756fc30967477098482eb7505783"
FORECAST_AS_OF = "2026-10-31"
REGISTERED_ON = "2026-11-01"
ELECTION_DATE = "2026-11-03"
FROZEN_OLS_FORECAST = -32.32288802630563
FROZEN_OLS_95 = (-48.428386550570906, -16.217389502040344)
FROZEN_BOOTSTRAP_95 = (-41.75170811125731, -14.373974322873948)
FROZEN_FORWARD_RMSE = 8.695800072815182
FROZEN_R_SQUARED = 0.893447684694887
FROZEN_TEST_YEARS = (1982, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022)
FROZEN_CANDIDATES: dict[str, float] = {
    "ols": FROZEN_OLS_FORECAST,
    "student_t": -33.75754882316428,
    "ridge": -31.060791226510172,
    "recency_wls": -30.97597265810168,
}
FINGERPRINT_PATHS: tuple[str, ...] = (
    "analysis.py",
    "model_extensions.py",
    "comparison_uncertainty.py",
    "requirements.txt",
    "model_walkthrough.ipynb",
    "data/historical_midterms.csv",
    "data/gallup_alternative.csv",
    "data/2026_scenario.csv",
    "data/PROVENANCE.md",
    "results/summary.json",
    "results/alternative_models_summary.json",
    "results/paired_uncertainty_summary.json",
)


def normalized_sha256(path: Path) -> str:
    """Hash UTF-8 text with normalized newlines across Git/OS checkouts."""

    content = path.read_bytes().replace(b"\r\n", b"\n")
    # The audited assets are text and must not depend on a local encoding.
    content.decode("utf-8", errors="strict")
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    content: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return content


def _same_float(observed: object, expected: float, label: str) -> None:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"Frozen {label} changed: {observed} != {expected}")


def build_forecast_lock(root: Path = ROOT) -> dict[str, Any]:
    """Validate the ex-ante source snapshot and assemble an immutable record."""

    rows = read_elections(root / "data" / "historical_midterms.csv")
    scenario = read_scenario(root / "data" / "2026_scenario.csv")
    if scenario.year != 2026 or scenario.party != "R":
        raise ValueError("The forecast is defined for the 2026 Republican president's party")
    if scenario.seats != 220 or scenario.party_margin != -8.0:
        raise ValueError("Frozen scenario D+8, Republican seats defended=220 was changed")
    if len(rows) != 18 or rows[0].year != 1950 or rows[-1].year != 2022:
        raise ValueError("Expected exactly 18 historical elections from 1950–2022")
    if 1986 in {row.year for row in rows} or any(row.year >= 2026 for row in rows):
        raise ValueError("No unsourced 1986 observation or 2026 outcome may be included")

    historical = fit_model(rows)
    _same_float(historical.predict(scenario.party_margin, scenario.seats),
                FROZEN_OLS_FORECAST, "OLS point forecast")
    interval = prediction_interval(historical, scenario.party_margin, scenario.seats)
    for index, value in enumerate(FROZEN_OLS_95):
        _same_float(interval[index], value, f"OLS interval endpoint {index}")

    summary = _read_json(root / "results" / "summary.json")
    models = _read_json(root / "results" / "alternative_models_summary.json")
    pairs = _read_json(root / "results" / "paired_uncertainty_summary.json")
    if any(source.get("as_of") != FORECAST_AS_OF for source in (summary, models, pairs)):
        raise ValueError("The three analysis snapshots must share the same as-of date")
    _same_float(summary.get("forecast"), FROZEN_OLS_FORECAST, "summary forecast")
    for i, value in enumerate(FROZEN_OLS_95):
        _same_float(summary["parametric_95_interval"][i], value,
                    f"summary prediction interval {i}")
    for candidate, target in FROZEN_CANDIDATES.items():
        _same_float(models["primary"][candidate]["forecast_2026"], target,
                    f"{candidate} forecast")
    _same_float(summary.get("forward_rmse"), FROZEN_FORWARD_RMSE, "forward RMSE")
    _same_float(summary.get("in_sample_r2"), FROZEN_R_SQUARED, "R-squared")
    for i, value in enumerate(FROZEN_BOOTSTRAP_95):
        _same_float(summary["bootstrap_95_interval"][i], value,
                    f"bootstrap sensitivity endpoint {i}")
    if summary.get("observations") != 18 or models.get("forward_count") != 10:
        raise ValueError("Frozen sample or forward evaluation count changed")
    if tuple(models["forward_elections"]) != FROZEN_TEST_YEARS:
        raise ValueError("Frozen test elections changed")
    if pairs.get("bootstrap_seed") != 20261031 or pairs.get("bootstrap_draws") != 20000:
        raise ValueError("Paired comparison reference configuration changed")

    digest = {name: normalized_sha256(root / name) for name in FINGERPRINT_PATHS}
    return {
        "manifest_version": 1,
        "source_repository": "https://github.com/DiogoRibeiro7/Medium-Blog",
        "source_commit": SOURCE_COMMIT,
        "forecast_information_as_of": FORECAST_AS_OF,
        "registration_date": REGISTERED_ON,
        "general_election_date": ELECTION_DATE,
        "target": {
            "description": "Republican president's party net change in U.S. House seats",
            "outcome_definition": "Change in initial-election House party divisions, aligned with the American Presidency Project / House Historian historical series",
            "unit": "House seats",
            "observed_2026_outcome": None,
        },
        "scenario": {
            "year": scenario.year,
            "president_party": scenario.party,
            "democratic_generic_ballot_lead_points": scenario.dem_margin,
            "president_party_signed_generic_margin_points": scenario.party_margin,
            "president_party_seats_defended": scenario.seats,
            "source": "Rattner, presentation of 2026-10-30",
        },
        "historical_sample": {
            "observed_elections": len(rows),
            "first_year": rows[0].year,
            "last_year": rows[-1].year,
            "unsupported_missing_ballot_year": 1986,
            "chronological_test_years": list(FROZEN_TEST_YEARS),
        },
        "primary_forecast": {
            "method": "OLS on the two frozen predictors, not causally identified",
            "predicted_seat_change": FROZEN_OLS_FORECAST,
            "model_based_95_prediction_interval": list(FROZEN_OLS_95),
            "in_sample_r_squared": FROZEN_R_SQUARED,
            "forward_rmse_house_seats": FROZEN_FORWARD_RMSE,
            "interval_caveat": "Conditional on fixed polling inputs and stable classical regression assumptions; not a guarantee",
        },
        "alternative_conditional_point_forecasts": dict(FROZEN_CANDIDATES),
        "bootstrap_95_sensitivity_interval": list(FROZEN_BOOTSTRAP_95),
        "original_published_reference_forecast": -33,
        "source_sha256_normalized_utf8_lf": digest,
        "scoring_protocol": {
            "score_when": "Only after 2026-11-03 and after an identified final House Historian-aligned outcome is available",
            "signed_error_definition": "predicted_minus_observed_seat_change",
            "absolute_error_definition": "abs(predicted_minus_observed_seat_change)",
            "interval_hit_definition": "lower <= observed_outcome <= upper; conditional OLS interval only",
            "record_official_source_url": True,
            "write_result_separately": True,
            "do_not_refit_on_2026_outcome": True,
        },
        "interpretation": (
            "Ex-ante registration relative to the 2026 election outcome, not "
            "pre-registration of the historically selected specification. "
            "A Git commit documents content but is not independent notarization."
        ),
    }


def verify_lock(manifest: Mapping[str, Any], *, current_root: Path | None = None) -> None:
    """Reject altered anchors; optionally verify local source bytes match."""

    if manifest.get("manifest_version") != 1:
        raise ValueError("Forecast lock version changed")
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Original source commit changed")
    if (manifest.get("forecast_information_as_of") != FORECAST_AS_OF
            or manifest.get("registration_date") != REGISTERED_ON
            or manifest.get("general_election_date") != ELECTION_DATE):
        raise ValueError("An as-of, registration or election date changed")
    target = manifest.get("target", {})
    if not isinstance(target, dict) or target.get("observed_2026_outcome") is not None:
        raise ValueError("The pre-election manifest must never contain an observed outcome")
    scenario = manifest.get("scenario")
    if (
        not isinstance(scenario, dict)
        or scenario.get("year") != 2026
        or scenario.get("president_party") != "R"
        or scenario.get("president_party_seats_defended") != 220
    ):
        raise ValueError("Frozen Republican 2026 scenario changed")
    _same_float(scenario.get("democratic_generic_ballot_lead_points"), 8.0,
                "registered Democratic generic ballot")
    _same_float(scenario.get("president_party_signed_generic_margin_points"), -8.0,
                "registered signed generic ballot")
    sample = manifest.get("historical_sample")
    if (
        not isinstance(sample, dict)
        or sample.get("observed_elections") != 18
        or sample.get("first_year") != 1950
        or sample.get("last_year") != 2022
        or sample.get("unsupported_missing_ballot_year") != 1986
        or sample.get("chronological_test_years") != list(FROZEN_TEST_YEARS)
    ):
        raise ValueError("Historical training/test sample changed")
    primary = manifest.get("primary_forecast", {})
    if not isinstance(primary, dict):
        raise ValueError("Missing frozen OLS forecast")
    _same_float(primary.get("predicted_seat_change"), FROZEN_OLS_FORECAST,
                "locked OLS forecast")
    bounds = primary.get("model_based_95_prediction_interval", [])
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Malformed OLS prediction interval")
    for i, value in enumerate(FROZEN_OLS_95):
        _same_float(bounds[i], value, f"locked interval endpoint {i}")
    _same_float(primary.get("in_sample_r_squared"), FROZEN_R_SQUARED,
                "registered R-squared")
    _same_float(primary.get("forward_rmse_house_seats"), FROZEN_FORWARD_RMSE,
                "registered forward RMSE")
    bootstrap_bounds = manifest.get("bootstrap_95_sensitivity_interval", [])
    if not isinstance(bootstrap_bounds, list) or len(bootstrap_bounds) != 2:
        raise ValueError("Malformed bootstrap sensitivity interval")
    for i, value in enumerate(FROZEN_BOOTSTRAP_95):
        _same_float(bootstrap_bounds[i], value, f"locked bootstrap endpoint {i}")
    if manifest.get("original_published_reference_forecast") != -33:
        raise ValueError("Original published reference changed")
    alternatives = manifest.get("alternative_conditional_point_forecasts", {})
    if not isinstance(alternatives, dict) or alternatives.keys() != FROZEN_CANDIDATES.keys():
        raise ValueError("Candidate forecast roster changed")
    for name, forecast in FROZEN_CANDIDATES.items():
        _same_float(alternatives[name], forecast, f"locked {name} alternative")
    fingerprints = manifest.get("source_sha256_normalized_utf8_lf")
    if not isinstance(fingerprints, dict) or fingerprints.keys() != set(FINGERPRINT_PATHS):
        raise ValueError("Manifest source fingerprint list changed")
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value)
           for value in fingerprints.values()):
        raise ValueError("Malformed SHA-256 fingerprint")
    if current_root is not None:
        for name in FINGERPRINT_PATHS:
            if normalized_sha256(current_root / name) != fingerprints[name]:
                raise ValueError(f"Source differs from frozen snapshot: {name}")


def save_forecast_lock(manifest: Mapping[str, Any], output: Path) -> None:
    """Write the deterministic manifest to a chosen reproducibility directory."""

    verify_lock(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Output root containing results/forecast_lock.json")
    args = parser.parse_args()
    manifest = build_forecast_lock()
    verify_lock(manifest, current_root=ROOT)
    output = args.output / "results" / "forecast_lock.json"
    save_forecast_lock(manifest, output)
    print(f"Registered original OLS 2026 seat change {FROZEN_OLS_FORECAST:+.5f}; "
          f"source commit {SOURCE_COMMIT[:10]}; {len(FINGERPRINT_PATHS)} fingerprints")


if __name__ == "__main__":
    main()
