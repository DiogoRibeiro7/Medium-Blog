"""Audit macro support and future-reform donor censoring without effects.

For the same balanced outcome window and five pre-reform years as the
exploratory matched study, compare donor risk sets at horizons 0 through 5.
Only *pre-reform* WDI means enter support/distance calculations. A donor's
future tax cut is used solely to audit the current exclusion rule; donors
with intervening reforms are never silently treated as valid controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, history, load_panel
from causal_tax_cuts.macro_matching import (
    MacroValues,
    audit_macro_matching,
    pre_reform_means,
    read_macro_controls,
)
from causal_tax_cuts.world_bank import CORE_COLUMNS

LAGS = 5
MAX_HORIZON = 5


@dataclass(frozen=True, slots=True)
class ReformSupport:
    """One recorded reform and its pre-censoring and last-horizon donor counts."""

    country: str
    reform_year: int
    status: str
    n_pre_censor_donors: int
    n_horizon_zero_donors: int
    n_horizon_five_donors: int
    n_future_exposed_donors_by_five: int


@dataclass(frozen=True, slots=True)
class CandidateExposure:
    """One pre-reform eligible donor and the first later reform, if any."""

    country: str
    reform_year: int
    donor: str
    first_donor_reform_year_t_through_t5: int | None
    first_censored_horizon: int | None
    fixed_pre5_standardized_distance: float


@dataclass(frozen=True, slots=True)
class HorizonOverlap:
    """Support measured at one reform/horizon with a fixed macro scale."""

    country: str
    reform_year: int
    horizon: int
    n_surviving_donors: int
    n_censored_donors: int
    within_gdp_growth_range: bool | None
    within_inflation_range: bool | None
    within_trade_openness_range: bool | None
    within_all_three_marginal_ranges: bool | None
    nearest_fixed_standardized_distance: float | None
    gdp_growth_outside_range_standardized: float | None
    inflation_outside_range_standardized: float | None
    trade_outside_range_standardized: float | None


def _macro_distances(
    target: np.ndarray, profiles: dict[str, np.ndarray]
) -> tuple[dict[str, float], np.ndarray]:
    """Use one *fixed* pre-censoring scale for all event-time risk sets.

    This is a descriptive Euclidean distance, not the ridge metric used to
    select the published exploratory donor set. Re-standardising after each
    censoring step would confound shrinking support with changing scales.
    """
    if not profiles or target.shape != (len(CORE_COLUMNS),):
        raise ValueError("Expected a treated profile and at least one donor")
    ordered = sorted(profiles)
    matrix = np.vstack([target, *(profiles[name] for name in ordered)])
    scales = np.std(matrix, axis=0, ddof=0)
    # Constant series cannot change distance; its unit scale is harmless.
    scales = np.where(scales < 1e-10, 1.0, scales)
    distance = {
        name: float(np.linalg.norm((target - profiles[name]) / scales))
        for name in ordered
    }
    if not all(math.isfinite(value) for value in distance.values()):
        raise ValueError("Nonfinite pre-period donor distance")
    return distance, scales


def _range_overlap(
    target: np.ndarray, profiles: list[np.ndarray], scales: np.ndarray
) -> tuple[tuple[bool | None, ...], tuple[float | None, ...]]:
    """Necessary univariate donor-range checks, not a positivity test.

    A treated point may lie within each marginal range while still being
    outside the donors' multivariate convex hull. One donor can also make a
    range degenerate. These are descriptive, limited support indicators.
    """
    if not profiles:
        return (None,) * len(CORE_COLUMNS), (None,) * len(CORE_COLUMNS)
    matrix = np.vstack(profiles)
    lower = np.min(matrix, axis=0)
    upper = np.max(matrix, axis=0)
    ranges: list[bool | None] = []
    violations: list[float | None] = []
    for index in range(len(CORE_COLUMNS)):
        value = float(target[index])
        in_range = bool(lower[index] <= value <= upper[index])
        extension = max(0.0, float(lower[index] - value), float(value - upper[index]))
        ranges.append(in_range)
        violations.append(extension / float(scales[index]))
    return tuple(ranges), tuple(violations)


def audit_future_donor_exposure(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
) -> tuple[
    list[ReformSupport],
    list[CandidateExposure],
    list[HorizonOverlap],
    dict[str, object],
]:
    """Separate baseline macro support from donor loss after future tax cuts.

    Keep t-5..t+5 income-share outcomes observed for treated and donors, and
    match the same five-year pre-treatment reform histories throughout. This
    isolates *future reform censoring* across horizons from changes in outcome
    availability, while retaining the full-window selection caveat.
    """
    if not panel or set(panel) != set(controls):
        raise ValueError("WID outcome and WDI controls must have identical keys")
    countries = sorted({name for name, _ in panel})
    events = sorted(
        (row for row in panel.values() if row.treated == 1),
        key=lambda row: (row.year, row.country),
    )
    offsets = range(-LAGS, MAX_HORIZON + 1)
    reform_rows: list[ReformSupport] = []
    donor_rows: list[CandidateExposure] = []
    horizon_rows: list[HorizonOverlap] = []

    def exclude(event: Observation, reason: str) -> None:
        reform_rows.append(ReformSupport(event.country, event.year, reason, 0, 0, 0, 0))

    for event in events:
        country, year = event.country, event.year
        treatment_path = history(panel, country, year, LAGS)
        if treatment_path is None or treatment_path[0] != 0:
            exclude(event, "invalid_treatment_history")
            continue
        if any((country, year + offset) not in panel for offset in offsets):
            exclude(event, "incomplete_treated_outcome_window")
            continue
        treated_profile = pre_reform_means(controls, country, year, LAGS)
        if treated_profile is None:
            exclude(event, "incomplete_treated_pre_macro")
            continue

        # Deliberately do *not* censor donors yet. No future macro data are
        # consulted and the five-year WID outcome window is held constant.
        profiles: dict[str, np.ndarray] = {}
        first_future_reform: dict[str, int | None] = {}
        for candidate in countries:
            if (
                candidate == country
                or history(panel, candidate, year, LAGS) != treatment_path
                or any((candidate, year + offset) not in panel for offset in offsets)
            ):
                continue
            profile = pre_reform_means(controls, candidate, year, LAGS)
            if profile is None:
                continue
            profiles[candidate] = profile
            first_future_reform[candidate] = next(
                (
                    future
                    for future in range(year, year + MAX_HORIZON + 1)
                    if panel[candidate, future].treated == 1
                ),
                None,
            )
        if not profiles:
            exclude(event, "no_pre_reform_macro_donors")
            continue

        distances, scales = _macro_distances(treated_profile, profiles)
        for candidate in sorted(profiles):
            first = first_future_reform[candidate]
            donor_rows.append(
                CandidateExposure(
                    country,
                    year,
                    candidate,
                    first,
                    first - year if first is not None else None,
                    distances[candidate],
                )
            )

        survival: dict[int, list[str]] = {}
        # Horizon -1 is the *entire* pre-censor donor pool. Horizon 0 already
        # excludes simultaneous reforms; horizons 1..5 reveal later losses.
        for horizon in range(-1, MAX_HORIZON + 1):
            donors = [
                name
                for name in sorted(profiles)
                if first_future_reform[name] is None
                or first_future_reform[name] > year + horizon
            ]
            survival[horizon] = donors
            marginal, violation = _range_overlap(
                treated_profile, [profiles[name] for name in donors], scales
            )
            horizon_rows.append(
                HorizonOverlap(
                    country,
                    year,
                    horizon,
                    len(donors),
                    len(profiles) - len(donors),
                    marginal[0],
                    marginal[1],
                    marginal[2],
                    all(marginal) if donors else None,
                    min(distances[name] for name in donors) if donors else None,
                    violation[0],
                    violation[1],
                    violation[2],
                )
            )
        reform_rows.append(
            ReformSupport(
                country,
                year,
                "baseline_donors_available",
                len(profiles),
                len(survival[0]),
                len(survival[MAX_HORIZON]),
                len(profiles) - len(survival[MAX_HORIZON]),
            )
        )

    summary_by_horizon: list[dict[str, object]] = []
    for horizon in range(-1, MAX_HORIZON + 1):
        rows = [row for row in horizon_rows if row.horizon == horizon]
        with_donors = [row for row in rows if row.n_surviving_donors > 0]
        with_ranges = [
            row for row in with_donors if row.within_all_three_marginal_ranges
        ]
        nearest = [
            row.nearest_fixed_standardized_distance
            for row in with_donors
            if row.nearest_fixed_standardized_distance is not None
        ]
        summary_by_horizon.append(
            {
                "horizon": horizon,
                "n_baseline_eligible_reforms": len(rows),
                "n_reforms_with_any_donor": len(with_donors),
                "n_reforms_with_at_least_three_donors": sum(
                    row.n_surviving_donors >= 3 for row in rows
                ),
                "n_donor_reform_pairs_at_risk": sum(
                    row.n_surviving_donors for row in rows
                ),
                "n_donor_reform_pairs_censored_cumulative": sum(
                    row.n_censored_donors for row in rows
                ),
                "n_reforms_with_all_three_marginal_ranges": len(with_ranges),
                "mean_nearest_fixed_standardized_distance": (
                    fmean(nearest) if nearest else None
                ),
                "median_nearest_fixed_standardized_distance": (
                    median(nearest) if nearest else None
                ),
            }
        )
    baseline, h5 = summary_by_horizon[0], summary_by_horizon[-1]
    report: dict[str, object] = {
        "status": "future_donor_exposure_and_marginal_support_audit_only",
        "n_original_reform_events": len(events),
        "n_eligible_reforms_before_future_censoring": baseline[
            "n_baseline_eligible_reforms"
        ],
        "n_reforms_with_donors_after_five_year_censoring": h5[
            "n_reforms_with_any_donor"
        ],
        "event_status_counts": dict(
            sorted(Counter(row.status for row in reform_rows).items())
        ),
        "matching_reference": {"lags": LAGS, "max_horizon": MAX_HORIZON},
        "horizon_minus_one": "pre-censor baseline including donors that reform at t",
        "source_outcome_window": "full t-5..t+5 WID outcome window at every horizon",
        "macro_input_window": "only t-5 through t-1 WDI; no imputation",
        "donor_screening": "identical five-year prior tax-cut path and complete observed WID outcome window",
        "censoring": "exclude donors at horizon h if any tax cut from t through t+h",
        "nearest_distance": "Euclidean distance on three lagged macro means, standardized once using pre-censoring treated+donor pool",
        "support_definition": "treated pre-five mean lies within every donor marginal min-max range, not joint convex hull or positivity",
        "n_pre_censor_donor_event_pairs": len(donor_rows),
        "n_donor_event_pairs_with_future_reform_through_h5": sum(
            row.first_censored_horizon is not None for row in donor_rows
        ),
        "by_horizon": summary_by_horizon,
        "inference": "none; no adjusted treatment effect, test or uncertainty interval",
        "caveats": [
            "A marginal-range screen is necessary but not sufficient for multivariate support; it does not prove positivity.",
            "The entire observed outcome window t-5..t+5 remains required even for h=0; this is another possible future-availability selection.",
            "The no-new-cut counterfactual cannot use donors after they reform, but selecting only non-reforming future donors conditions on post-baseline policy.",
            "Donor-event pairs recur across reforms/countries and are not independent observations.",
            "Political, institutional and capital-openness confounders have not been adjusted.",
        ],
    }
    return reform_rows, donor_rows, horizon_rows, report


def save_support(
    output_dir: Path,
    reforms: list[ReformSupport],
    candidates: list[CandidateExposure],
    horizons: list[HorizonOverlap],
    report: dict[str, object],
    *,
    source_panel: Path,
    provenance: Path | None = None,
    panel: dict[Key, Observation] | None = None,
    controls: dict[Key, MacroValues] | None = None,
) -> None:
    """Persist per-event risk sets and verify the existing five/five matches."""
    if not reforms:
        raise ValueError("Refusing to save an empty support audit")
    data = dict(report)
    data["joined_panel_sha256"] = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    if provenance is not None:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
        if manifest.get("publisher") != "World Bank World Development Indicators (DS2)":
            raise ValueError("Unexpected WDI source manifest")
        series = manifest.get("series")
        if not isinstance(series, list) or len(series) != 4:
            raise ValueError("Missing official WDI series provenance")
        data["source_archive_sha256"] = {
            row["indicator_code"]: row["source_sha256"] for row in series
        }
        data["source_last_updated"] = {
            row["indicator_code"]: row["source_last_updated"] for row in series
        }
    if panel is not None and controls is not None:
        previous, _, previous_audit = audit_macro_matching(
            panel, controls, lags=LAGS, max_horizon=MAX_HORIZON, max_donors=5
        )
        reference = {
            (row.country, row.reform_year)
            for row in previous
            if row.status == "matched"
        }
        this_audit = {
            (row.country, row.reform_year)
            for row in reforms
            if row.n_horizon_five_donors > 0
        }
        if reference != this_audit:
            raise ValueError(
                "Horizon-five support differs from previous matching eligibility"
            )
        data["n_previous_reference_matched_events"] = previous_audit["n_matched_events"]
        data["reference_matched_event_ids_consistent"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, fieldnames in (
        (
            "support_reform_eligibility.csv",
            reforms,
            list(ReformSupport.__dataclass_fields__),
        ),
        (
            "future_donor_exposure.csv",
            candidates,
            list(CandidateExposure.__dataclass_fields__),
        ),
        ("support_by_horizon.csv", horizons, list(HorizonOverlap.__dataclass_fields__)),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output_dir / "donor_support_audit.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Audit the future-reform exclusion rule on the real WID/WDI join."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    controls = read_macro_controls(args.panel, set(panel))
    reforms, donors, horizons, report = audit_future_donor_exposure(panel, controls)
    save_support(
        args.output_dir,
        reforms,
        donors,
        horizons,
        report,
        source_panel=args.panel,
        provenance=args.provenance,
        panel=panel,
        controls=controls,
    )
    print(
        f"Donor support: {report['n_eligible_reforms_before_future_censoring']} "
        f"pre-censor, {report['n_reforms_with_donors_after_five_year_censoring']} "
        "at horizon five"
    )
    print("CAUTION: support/selection diagnostic only; no adjusted causal effect")


if __name__ == "__main__":
    main()
