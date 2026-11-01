"""Score a *later*, source-verified House result against the sealed forecast.

This file is deliberately safe to ship before the November 3 election. Its
CLI refuses to score anything before election day, requires the analyst to
affirm that the cited House Historian-aligned result is final, and writes a
new report separately from the untouched forecast manifest. There is no
2026 observed outcome stored in this repository at registration time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from analysis import ROOT
from forecast_lock import ELECTION_DATE, FROZEN_OLS_FORECAST, verify_lock

ACCEPTED_SOURCE_HOSTS: frozenset[str] = frozenset({
    "history.house.gov", "presidency.ucsb.edu", "www.presidency.ucsb.edu",
})


def _require_official_source_url(value: str) -> str:
    """Allow only the declared authoritative House party-division publishers."""

    if not isinstance(value, str):
        raise TypeError("Source URL must be a string")
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname not in ACCEPTED_SOURCE_HOSTS
            or not parsed.path.strip("/") or parsed.username or parsed.password):
        raise ValueError("Cite an HTTPS House Historian or American Presidency Project page")
    return value


def _hit(bounds: list[float], observed: int) -> bool:
    """Check interval containment without interpreting it as a guarantee."""

    return float(bounds[0]) <= observed <= float(bounds[1])


def score_verified_outcome(
    *,
    observed_change: int,
    final_source_url: str,
    source_retrieved_on: date,
    confirmed_final: bool,
    evaluated_on: date,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Calculate prospective diagnostics from the immutable ex-ante manifest.

    ``observed_change`` is the Republican president-party seat change in the
    final *initial-election* House division series; it is **not** the change in
    House control probability, a generic-ballot margin or election-night
    partial count. The signed error convention matches ``analysis.Holdout``:
    prediction minus observation.
    """

    verify_lock(manifest)
    election_day = date.fromisoformat(ELECTION_DATE)
    if (not isinstance(evaluated_on, date) or isinstance(evaluated_on, datetime)
            or not isinstance(source_retrieved_on, date)
            or isinstance(source_retrieved_on, datetime)):
        raise TypeError("Evaluation and retrieval dates must be date objects")
    if evaluated_on < election_day:
        raise ValueError("Cannot score a future election before 2026-11-03")
    if not election_day <= source_retrieved_on <= evaluated_on:
        raise ValueError("A final source must be retrieved after election day and by evaluation")
    if confirmed_final is not True:
        raise ValueError("Explicit final-result verification is required")
    if isinstance(observed_change, bool) or not isinstance(observed_change, int):
        raise TypeError("The observed president-party House change must be an integer")
    # From the frozen 220 Republican seats defended, at most 220 can be lost;
    # at most the remaining 215 seats can be added to the party's total.
    defended = manifest["scenario"]["president_party_seats_defended"]
    if observed_change < -defended or observed_change > 435 - defended:
        raise ValueError("House seat change is impossible for the frozen defended-seat count")
    official_url = _require_official_source_url(final_source_url)

    baseline = manifest["primary_forecast"]
    forecast = float(baseline["predicted_seat_change"])
    signed_error = forecast - observed_change
    forecasts = manifest["alternative_conditional_point_forecasts"]
    alternative_errors = {
        name: {
            "registered_forecast": float(value),
            "signed_error_predicted_minus_observed": float(value) - observed_change,
            "absolute_error_house_seats": abs(float(value) - observed_change),
        }
        for name, value in forecasts.items()
    }
    lock_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")
    return {
        "report_type": "post_election_evaluation_of_registered_forecast",
        "registration_date": manifest["registration_date"],
        "registered_source_commit": manifest["source_commit"],
        "registered_manifest_canonical_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "election_date": ELECTION_DATE,
        "evaluated_on": evaluated_on.isoformat(),
        "official_source_retrieved_on": source_retrieved_on.isoformat(),
        "official_source_url": official_url,
        "analyst_confirms_result_final": True,
        "observed_president_party_seat_change": observed_change,
        "ols_registered_forecast": forecast,
        "ols_signed_error_predicted_minus_observed": signed_error,
        "ols_absolute_error_house_seats": abs(signed_error),
        "ols_observation_inside_conditional_95_interval": _hit(
            baseline["model_based_95_prediction_interval"], observed_change),
        "observation_inside_pairs_residual_sensitivity_band": _hit(
            manifest["bootstrap_95_sensitivity_interval"], observed_change),
        "candidate_models": alternative_errors,
        "reference_original_rattner_minus_33_absolute_error": abs(-33 - observed_change),
        "warning": (
            "One scored election cannot establish calibrated coverage or select "
            "a universally superior model. Source-series definitions and "
            "historical-model choices remain as registered."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-change", type=int, required=True,
                        help="Final Republican president-party House seat change")
    parser.add_argument("--official-url", type=str, required=True,
                        help="Final House Historian or American Presidency Project HTTPS source")
    parser.add_argument("--source-retrieved-on", type=date.fromisoformat, required=True,
                        help="Date (YYYY-MM-DD) when the final official figure was retrieved")
    parser.add_argument("--confirmed-final", action="store_true", required=True,
                        help="Affirm the cited outcome is final, not an election-night tally")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON report path; existing reports are never overwritten")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    manifest: dict[str, Any] = json.loads(
        (ROOT / "results" / "forecast_lock.json").read_text(encoding="utf-8"))
    result = score_verified_outcome(
        observed_change=args.actual_change,
        final_source_url=args.official_url,
        source_retrieved_on=args.source_retrieved_on,
        confirmed_final=args.confirmed_final,
        evaluated_on=today,
        manifest=manifest,
    )
    destination = args.output.resolve()
    protected = (ROOT / "results" / "forecast_lock.json").resolve()
    if destination == protected:
        raise ValueError("The frozen forecast lock may never be overwritten")
    if destination.suffix.lower() != ".json":
        raise ValueError("Evaluation outputs must be JSON")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Registered OLS {FROZEN_OLS_FORECAST:+.2f}; observed {args.actual_change:+d}; "
          f"signed error {result['ols_signed_error_predicted_minus_observed']:+.2f}")


if __name__ == "__main__":
    main()
