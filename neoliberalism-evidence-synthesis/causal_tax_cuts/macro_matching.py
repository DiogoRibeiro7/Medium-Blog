"""Audit pre-reform WDI matching support and balance without estimating effects.

The original outcome panel is never restricted to complete post-reform macro
observations. Missing WDI controls are checked only at t-1 through t-L, and
the history-only and macro-complete donor sets are reported separately.
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
from statistics import fmean, median, pstdev
from typing import TypeAlias

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, history, load_panel
from causal_tax_cuts.world_bank import CORE_COLUMNS

MacroValues: TypeAlias = tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class EventEligibility:
    """A reform's attrition reason, donor pools and pre-period outcome balance."""

    country: str
    reform_year: int
    status: str
    history_only_donors: int
    macro_complete_donors: int
    selected_donors: tuple[str, ...]
    mean_selected_distance: float | None
    pretrend_slope_gap_pool: float | None
    pretrend_slope_gap_selected: float | None


@dataclass(frozen=True, slots=True)
class MacroBalance:
    """One event/control's common-pool balance before and after ranking."""

    country: str
    reform_year: int
    control: str
    treated_pre_mean: float
    donor_pool_pre_mean: float
    selected_pre_mean: float
    donor_pool_standard_deviation: float
    standardized_gap_pool: float | None
    standardized_gap_selected: float | None


def read_macro_controls(path: Path, expected_keys: set[Key]) -> dict[Key, MacroValues]:
    """Read a joined real-data CSV; preserve missing macro values as None.

    The WID/author treatment and outcome data are independently loaded with
    `load_panel(..., covariates=())`, so a missing macro value cannot delete
    any original country-year before eligibility is assessed.
    """
    required = {"country", "year", *CORE_COLUMNS}
    controls: dict[Key, MacroValues] = {}
    if not expected_keys:
        raise ValueError("Expected a nonempty outcome panel")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not required.issubset(reader.fieldnames)
        ):
            raise ValueError("Joined panel has missing or duplicate WDI columns")
        for line, row in enumerate(reader, start=2):
            try:
                country = (row["country"] or "").strip().upper()
                key = (country, int(row["year"]))
                values = tuple(
                    None if (row[column] or "").strip() == "" else float(row[column])
                    for column in CORE_COLUMNS
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed WDI control on line {line}") from exc
            if key not in expected_keys or key in controls:
                raise ValueError(f"Unexpected or duplicated WDI row: {key}")
            if any(value is not None and not math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite WDI value on line {line}")
            if values[2] is not None and values[2] < 0:
                raise ValueError(f"Negative trade openness on line {line}")
            controls[key] = values
    if set(controls) != expected_keys:
        raise ValueError("Joined WDI panel has missing country-year rows")
    return controls


def pre_reform_means(
    controls: dict[Key, MacroValues], country: str, year: int, lags: int
) -> np.ndarray | None:
    """Return just three five-year pre-treatment means, or None if incomplete."""
    measurements = [controls.get((country, year - lag)) for lag in range(1, lags + 1)]
    if any(row is None or any(value is None for value in row) for row in measurements):
        return None
    # Every nullable cell has been checked; NumPy does not receive missing data.
    matrix = np.asarray(measurements, dtype=float)
    if matrix.shape != (lags, len(CORE_COLUMNS)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid pre-reform WDI trajectory")
    return np.mean(matrix, axis=0)


def _rank_macro_donors(
    treated: np.ndarray,
    donors: dict[str, np.ndarray],
    max_donors: int,
) -> list[tuple[str, float]]:
    """Use within-event z-scores and a fixed ridge Mahalanobis metric.

    Only three pre-reform means are used. With at most 17 donors, the 25% ridge
    term avoids an unstable covariance inverse and is an explicit design choice.
    """
    if treated.shape != (len(CORE_COLUMNS),) or not donors or max_donors < 1:
        raise ValueError("Invalid treated profile or donor configuration")
    names = sorted(donors)
    matrix = np.vstack([treated, *(donors[name] for name in names)])
    scales = np.std(matrix, axis=0, ddof=0)
    scales = np.where(scales < 1e-10, 1.0, scales)
    standardized = (matrix - np.mean(matrix, axis=0)) / scales
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
    regularized = 0.75 * covariance + 0.25 * np.eye(len(CORE_COLUMNS))
    inverse = np.linalg.pinv(regularized, rcond=1e-10)
    ranked = []
    for index, name in enumerate(names, start=1):
        difference = standardized[0] - standardized[index]
        distance = math.sqrt(max(0.0, float(difference @ inverse @ difference)))
        ranked.append((name, distance))
    return sorted(ranked, key=lambda entry: (entry[1], entry[0]))[:max_donors]


def _outcome_slope(
    panel: dict[Key, Observation], country: str, year: int, lags: int
) -> float:
    """OLS pre-treatment outcome slope measured per calendar year."""
    years = list(range(year - lags, year))
    mean_year = fmean(years)
    mean_outcome = fmean(panel[(country, t)].outcome for t in years)
    numerator = sum(
        (t - mean_year) * (panel[(country, t)].outcome - mean_outcome) for t in years
    )
    return numerator / sum((t - mean_year) ** 2 for t in years)


def audit_macro_matching(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    *,
    lags: int = 5,
    max_horizon: int = 5,
    max_donors: int = 5,
) -> tuple[list[EventEligibility], list[MacroBalance], dict[str, object]]:
    """Audit event attrition and pre-period balance with no treatment effects.

    The baseline uses *all* macro-complete history-eligible donors; only the
    choice of the closest K changes in the comparison. The history-only pool
    is counted separately, so eligibility loss is not disguised as a benefit
    of matching. Donor future-exposure censoring follows the existing study.
    """
    if lags < 3 or max_horizon < 0 or max_donors < 1 or not panel:
        raise ValueError("Invalid lags/horizon/max donors or empty outcome panel")
    if set(panel) != set(controls):
        raise ValueError("Outcome and WDI joined-panel keys differ")
    events = sorted(
        (row for row in panel.values() if row.treated == 1),
        key=lambda row: (row.year, row.country),
    )
    countries = sorted({country for country, _ in panel})
    offsets = range(-lags, max_horizon + 1)
    outcomes: list[EventEligibility] = []
    balance: list[MacroBalance] = []

    def record(
        event: Observation,
        status: str,
        history_donors: int = 0,
        macro_donors: int = 0,
        selected: tuple[str, ...] = (),
        distance: float | None = None,
        slope_before: float | None = None,
        slope_after: float | None = None,
    ) -> None:
        outcomes.append(
            EventEligibility(
                event.country,
                event.year,
                status,
                history_donors,
                macro_donors,
                selected,
                distance,
                slope_before,
                slope_after,
            )
        )

    for event in events:
        country, year = event.country, event.year
        treatment_path = history(panel, country, year, lags)
        if treatment_path is None or treatment_path[0] != 0:
            record(event, "invalid_treatment_history")
            continue
        if any((country, year + offset) not in panel for offset in offsets):
            record(event, "incomplete_treated_outcome_window")
            continue

        history_donors = [
            candidate
            for candidate in countries
            if candidate != country
            and history(panel, candidate, year, lags) == treatment_path
            and all((candidate, year + offset) in panel for offset in offsets)
            and all(
                panel[(candidate, future)].treated == 0
                for future in range(year, year + max_horizon + 1)
            )
        ]
        if not history_donors:
            record(event, "no_history_eligible_donors")
            continue
        treated_profile = pre_reform_means(controls, country, year, lags)
        if treated_profile is None:
            record(event, "incomplete_treated_pre_macro", len(history_donors))
            continue
        donor_profiles: dict[str, np.ndarray] = {}
        for candidate in history_donors:
            profile = pre_reform_means(controls, candidate, year, lags)
            if profile is not None:
                donor_profiles[candidate] = profile
        if not donor_profiles:
            record(event, "no_macro_complete_donors", len(history_donors))
            continue

        ranked = _rank_macro_donors(treated_profile, donor_profiles, max_donors)
        selected = tuple(name for name, _ in ranked)
        pool = sorted(donor_profiles)
        treated_slope = _outcome_slope(panel, country, year, lags)
        slope_pool = treated_slope - fmean(
            _outcome_slope(panel, donor, year, lags) for donor in pool
        )
        slope_selected = treated_slope - fmean(
            _outcome_slope(panel, donor, year, lags) for donor in selected
        )
        record(
            event,
            "matched",
            len(history_donors),
            len(donor_profiles),
            selected,
            fmean(distance for _, distance in ranked),
            slope_pool,
            slope_selected,
        )

        for index, column in enumerate(CORE_COLUMNS):
            values = [float(donor_profiles[name][index]) for name in pool]
            scale = pstdev(values)
            baseline_mean = fmean(values)
            selected_mean = fmean(
                float(donor_profiles[name][index]) for name in selected
            )
            # With no donor variation, a standardized difference is undefined,
            # not zero. Retain both unstandardized means for inspection.
            standardized_before = (
                (float(treated_profile[index]) - baseline_mean) / scale
                if scale > 1e-10
                else None
            )
            standardized_after = (
                (float(treated_profile[index]) - selected_mean) / scale
                if scale > 1e-10
                else None
            )
            balance.append(
                MacroBalance(
                    country,
                    year,
                    column,
                    float(treated_profile[index]),
                    baseline_mean,
                    selected_mean,
                    scale,
                    standardized_before,
                    standardized_after,
                )
            )

    matched = [event for event in outcomes if event.status == "matched"]
    variable_balance: dict[str, dict[str, object]] = {}
    for column in CORE_COLUMNS:
        rows = [
            row
            for row in balance
            if row.control == column
            and row.standardized_gap_pool is not None
            and row.standardized_gap_selected is not None
        ]
        variable_balance[column] = {
            "n_evaluable_events": len(rows),
            "n_undefined_due_to_zero_donor_variance": len(matched) - len(rows),
            "mean_abs_standardized_gap_pool": (
                fmean(abs(row.standardized_gap_pool) for row in rows) if rows else None
            ),
            "mean_abs_standardized_gap_selected": (
                fmean(abs(row.standardized_gap_selected) for row in rows)
                if rows
                else None
            ),
            "n_events_abs_gap_reduced": sum(
                abs(row.standardized_gap_selected) < abs(row.standardized_gap_pool)
                for row in rows
            ),
        }
    audit: dict[str, object] = {
        "status": "pre_reform_support_and_balance_only_no_adjusted_effects",
        "n_original_treated_events": len(events),
        "n_matched_events": len(matched),
        "n_matched_treated_countries": len({event.country for event in matched}),
        "event_status_counts": dict(
            sorted(Counter(event.status for event in outcomes).items())
        ),
        "n_history_only_eligible_events": sum(
            event.history_only_donors > 0 for event in outcomes
        ),
        "n_original_history_donors_across_matched_events": sum(
            event.history_only_donors for event in matched
        ),
        "n_macro_complete_donors_across_matched_events": sum(
            event.macro_complete_donors for event in matched
        ),
        "n_selected_donors_across_matched_events": sum(
            len(event.selected_donors) for event in matched
        ),
        "lags": lags,
        "max_horizon": max_horizon,
        "max_donors": max_donors,
        "pre_reform_controls": list(CORE_COLUMNS),
        "selection_profile": "mean of t-1 through t-L for each core WDI series",
        "ranking": "within-event standardized ridge Mahalanobis; 0.75 covariance + 0.25 identity",
        "baseline": "same complete-macro donor pool, prior to nearest-K selection",
        "donor_exposure_rule": "no major tax cut from t through t+max_horizon",
        "outcome_window": "all years from t-L through t+max_horizon required",
        "variable_balance": variable_balance,
        "pretrend_slope_balance": {
            "mean_abs_gap_pool": (
                fmean(abs(event.pretrend_slope_gap_pool) for event in matched)
                if matched
                else None
            ),
            "mean_abs_gap_selected": (
                fmean(abs(event.pretrend_slope_gap_selected) for event in matched)
                if matched
                else None
            ),
            "median_abs_gap_selected": (
                median(abs(event.pretrend_slope_gap_selected) for event in matched)
                if matched
                else None
            ),
        },
        "inference": "none: no adjusted treatment effects, uncertainty or significance",
        "caveats": [
            "Control availability never restricts outcomes at or after reform t.",
            "Matching only three pre-treatment proxy means does not establish exchangeability.",
            "Donor censoring by future reform exposure may introduce selection bias.",
            "Pre-period balance does not establish parallel trends or causal effects.",
            "Donor pools and reforms overlap across events; SMD means are descriptive.",
        ],
    }
    return outcomes, balance, audit


def save_audit(
    output_dir: Path,
    events: list[EventEligibility],
    balance: list[MacroBalance],
    audit: dict[str, object],
    *,
    source_panel: Path,
    provenance: Path | None = None,
) -> None:
    """Write event-level CSVs and a provenance-aware machine-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "macro_event_eligibility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(EventEligibility.__dataclass_fields__)
        )
        writer.writeheader()
        for event in events:
            row = asdict(event)
            row["selected_donors"] = ";".join(event.selected_donors)
            writer.writerow(row)
    with (output_dir / "macro_pretrend_balance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(MacroBalance.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in balance)
    data = dict(audit)
    data["joined_panel_sha256"] = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    if provenance is not None:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
        if manifest.get("publisher") != "World Bank World Development Indicators (DS2)":
            raise ValueError("Unexpected macro data publisher")
        series = manifest.get("series")
        if not isinstance(series, list) or len(series) != 4:
            raise ValueError("Expected exactly four WDI source series")
        codes = {row.get("indicator_code") for row in series}
        if not {"NY.GDP.PCAP.KD.ZG", "FP.CPI.TOTL.ZG", "NE.TRD.GNFS.ZS"}.issubset(
            codes
        ):
            raise ValueError("WDI source manifest is missing a core series")
        data["source_archive_sha256"] = {
            row["indicator_code"]: row["source_sha256"] for row in series
        }
        data["source_last_updated"] = {
            row["indicator_code"]: row["source_last_updated"] for row in series
        }
    (output_dir / "macro_balance_audit.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Read the audited joined WID/WDI panel and report a balance diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=5)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    macro = read_macro_controls(args.panel, set(panel))
    events, balance, audit = audit_macro_matching(
        panel,
        macro,
        lags=args.lags,
        max_horizon=args.max_horizon,
        max_donors=args.max_donors,
    )
    save_audit(
        args.output_dir,
        events,
        balance,
        audit,
        source_panel=args.panel,
        provenance=args.provenance,
    )
    print(
        f"Pre-reform macro balance: {audit['n_matched_events']} matched / "
        f"{audit['n_original_treated_events']} reform events"
    )
    print("CAUTION: donor selection and balance only; no adjusted causal effect")


if __name__ == "__main__":
    main()
