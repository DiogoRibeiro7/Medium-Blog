"""Sensitivity grid for pre-treatment macro donor support and balance only.

The three-by-three grid varies history length and donor cap around the frozen
five-lag/five-donor reference. It never recalculates treatment effects. A
common reform cohort and a fixed three-year pre-reform evaluation window
separate rule sensitivity from changes in sample composition and time scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.macro_matching import (
    EventEligibility,
    MacroValues,
    audit_macro_matching,
    pre_reform_means,
    read_macro_controls,
)
from causal_tax_cuts.world_bank import CORE_COLUMNS

REFERENCE_LAGS = 5
REFERENCE_DONORS = 5
EVALUATION_LAGS = 3
DEFAULT_LAGS: tuple[int, ...] = (3, 5, 7)
DEFAULT_DONOR_CAPS: tuple[int, ...] = (3, 5, 7)


@dataclass(frozen=True, slots=True)
class SensitivityEvent:
    """One event under one rule, with fixed-window balance diagnostics."""

    lags: int
    max_donors: int
    country: str
    reform_year: int
    status: str
    n_history_donors: int
    n_macro_complete_donors: int
    selected_donors: tuple[str, ...]
    reference_donor_jaccard: float | None
    fixed_pre3_outcome_slope_gap: float | None
    fixed_pre3_gdp_smd: float | None
    fixed_pre3_inflation_smd: float | None
    fixed_pre3_trade_smd: float | None


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    """Coverage and identical-cohort diagnostics for one grid configuration."""

    lags: int
    max_donors: int
    n_matched_events: int
    n_treated_countries: int
    n_history_eligible_events: int
    n_macro_complete_donor_event_pairs: int
    n_selected_donor_event_pairs: int
    n_common_cohort_events: int
    mean_reference_donor_jaccard_common: float | None
    mean_abs_fixed_pre3_outcome_slope_gap_common: float | None
    mean_abs_fixed_pre3_gdp_smd_common: float | None
    mean_abs_fixed_pre3_inflation_smd_common: float | None
    mean_abs_fixed_pre3_trade_smd_common: float | None


def _validate_grid(lag_values: tuple[int, ...], donor_caps: tuple[int, ...]) -> None:
    """Require one transparent, nonempty, unique grid with a fixed reference."""
    if (
        not lag_values
        or not donor_caps
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (*lag_values, *donor_caps)
        )
        or any(value < EVALUATION_LAGS for value in lag_values)
        or any(value < 1 for value in donor_caps)
        or len(set(lag_values)) != len(lag_values)
        or len(set(donor_caps)) != len(donor_caps)
        or REFERENCE_LAGS not in lag_values
        or REFERENCE_DONORS not in donor_caps
    ):
        raise ValueError(
            "Grid must include L=5/K=5, have unique lags>=3 and donor caps>=1"
        )


def _jaccard(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    """Compare donor *sets* rather than their distance ranking positions."""
    union = set(first) | set(second)
    if not union:
        raise ValueError("Jaccard is undefined for two empty donor sets")
    return len(set(first) & set(second)) / len(union)


def _fixed_pre3_diagnostics(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    country: str,
    year: int,
    donors: tuple[str, ...],
) -> tuple[float, float | None, float | None, float | None]:
    """Evaluate chosen donors over a uniform t-3..t-1 window.

    Standardisation uses country-level three-year means for *all* countries
    with observed pre-reform controls, regardless of which matching rule made
    the donor eligible. This denominator remains fixed across the grid for
    the same reform, avoiding specification-dependent SMD scales.
    """
    if not donors:
        raise ValueError("Fixed-window diagnostic requires selected donors")
    profiles = {
        candidate: profile
        for candidate in sorted({key[0] for key in panel})
        if (profile := pre_reform_means(controls, candidate, year, EVALUATION_LAGS))
        is not None
    }
    if country not in profiles or any(donor not in profiles for donor in donors):
        raise ValueError("A matched donor lacks required fixed pre-period WDI data")

    def fixed_slope(name: str) -> float:
        """Three equally spaced annual outcomes give an endpoint OLS slope."""
        return (panel[name, year - 1].outcome - panel[name, year - 3].outcome) / 2.0

    treated_slope = fixed_slope(country)
    slope_gap = treated_slope - fmean(fixed_slope(donor) for donor in donors)
    smds: list[float | None] = []
    for index in range(len(CORE_COLUMNS)):
        scale = pstdev(float(profile[index]) for profile in profiles.values())
        gap = float(profiles[country][index]) - fmean(
            float(profiles[donor][index]) for donor in donors
        )
        smds.append(gap / scale if scale > 1e-10 else None)
    return slope_gap, smds[0], smds[1], smds[2]


def _mean_abs(values: list[float | None]) -> float | None:
    """Leave all-undefined summaries null rather than invent a zero."""
    available = [abs(value) for value in values if value is not None]
    return fmean(available) if available else None


def audit_sensitivity_grid(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    *,
    lag_values: tuple[int, ...] = DEFAULT_LAGS,
    donor_caps: tuple[int, ...] = DEFAULT_DONOR_CAPS,
    max_horizon: int = 5,
) -> tuple[list[SensitivityPoint], list[SensitivityEvent], dict[str, object]]:
    """Audit coverage and pre-reform balance for a declared grid without ATT.

    Each configuration invokes the existing, independently tested macro
    matching audit. Cross-rule comparisons are calculated only on the
    intersection of all matched reform IDs, over a fixed three-year window.
    """
    _validate_grid(lag_values, donor_caps)
    if max_horizon < 0 or not panel or set(panel) != set(controls):
        raise ValueError("Invalid horizon or mismatched WID/WDI panel")

    runs: dict[
        tuple[int, int], tuple[dict[Key, EventEligibility], dict[str, object]]
    ] = {}
    matched_sets: list[set[Key]] = []
    for lags in sorted(lag_values):
        for cap in sorted(donor_caps):
            events, _, audit = audit_macro_matching(
                panel, controls, lags=lags, max_horizon=max_horizon, max_donors=cap
            )
            by_event: dict[Key, EventEligibility] = {
                (event.country, event.reform_year): event for event in events
            }
            if len(by_event) != len(events):
                raise ValueError("Duplicate policy reform events in grid run")
            runs[lags, cap] = (by_event, audit)
            matched_sets.append(
                {key for key, event in by_event.items() if event.status == "matched"}
            )
    common = set.intersection(*matched_sets)
    reference_events, reference_audit = runs[REFERENCE_LAGS, REFERENCE_DONORS]

    detailed: list[SensitivityEvent] = []
    grid: list[SensitivityPoint] = []
    for (lags, cap), (events, audit) in sorted(runs.items()):
        detail_for_config: list[SensitivityEvent] = []
        for (country, year), event in sorted(events.items()):
            reference_event = reference_events[country, year]
            selected = event.selected_donors
            matching_both = (
                event.status == "matched" and reference_event.status == "matched"
            )
            jaccard = (
                _jaccard(selected, reference_event.selected_donors)
                if matching_both
                else None
            )
            fixed = (
                _fixed_pre3_diagnostics(panel, controls, country, year, selected)
                if event.status == "matched"
                else (None, None, None, None)
            )
            detail_for_config.append(
                SensitivityEvent(
                    lags,
                    cap,
                    country,
                    year,
                    event.status,
                    event.history_only_donors,
                    event.macro_complete_donors,
                    selected,
                    jaccard,
                    *fixed,
                )
            )
        detailed.extend(detail_for_config)
        on_common = [
            row for row in detail_for_config if (row.country, row.reform_year) in common
        ]
        if len(on_common) != len(common):
            raise ValueError("Common-cohort re-evaluation lost a reform event")
        jaccards = [row.reference_donor_jaccard for row in on_common]
        grid.append(
            SensitivityPoint(
                lags,
                cap,
                audit["n_matched_events"],
                audit["n_matched_treated_countries"],
                audit["n_history_only_eligible_events"],
                audit["n_macro_complete_donors_across_matched_events"],
                audit["n_selected_donors_across_matched_events"],
                len(common),
                fmean(value for value in jaccards if value is not None)
                if jaccards
                else None,
                _mean_abs([row.fixed_pre3_outcome_slope_gap for row in on_common]),
                _mean_abs([row.fixed_pre3_gdp_smd for row in on_common]),
                _mean_abs([row.fixed_pre3_inflation_smd for row in on_common]),
                _mean_abs([row.fixed_pre3_trade_smd for row in on_common]),
            )
        )

    report: dict[str, object] = {
        "status": "pre_reform_sensitivity_only_no_adjusted_att_or_inference",
        "lag_values": sorted(lag_values),
        "max_donors_values": sorted(donor_caps),
        "n_grid_configurations": len(grid),
        "max_horizon_fixed": max_horizon,
        "n_original_events": reference_audit["n_original_treated_events"],
        "n_common_matched_events": len(common),
        "common_matched_event_ids": [
            {"country": country, "reform_year": year}
            for country, year in sorted(common)
        ],
        "reference_rule": {"lags": REFERENCE_LAGS, "max_donors": REFERENCE_DONORS},
        "reference_matches_prior_audit_by_construction": False,
        "comparison_window": "t-3 through t-1 for all configurations",
        "fixed_smd_scale": "population SD of per-country pre-3 means for all WDI-observed countries at the event date",
        "fixed_outcome_slope_unit": "top-1% income-share percentage points per year",
        "missing_smd_rule": "zero cross-country variance gives undefined, not zero",
        "reference_audit_counts": {
            "n_matched_events": reference_audit["n_matched_events"],
            "n_original_treated_events": reference_audit["n_original_treated_events"],
            "n_selected_donor_event_pairs": reference_audit[
                "n_selected_donors_across_matched_events"
            ],
        },
        "grid": [asdict(row) for row in grid],
        "limitations": [
            "No treatment effect or uncertainty estimate is calculated.",
            "The common cohort is selected by meeting every grid specification; it is not a representative random subsample.",
            "The fixed covariate evaluation window overlaps selection features and is not an external validation set.",
            "Changing lag length changes historic treatment constraints and donor eligibility, not just matching similarity.",
            "Excluding donors that receive a future reform can induce post-baseline selection bias.",
            "Overlapping events, country dependence and unmeasured confounders remain unresolved.",
        ],
    }
    return grid, detailed, report


def _check_reference(
    report: dict[str, object],
    frozen_path: Path,
    joined_sha256: str,
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    max_horizon: int,
) -> None:
    """Validate reference figures only if the live joined-panel hash agrees."""
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("joined_panel_sha256") != joined_sha256:
        report["frozen_reference_check"] = "source_vintage_changed_not_comparable"
        return
    _, _, current = audit_macro_matching(
        panel,
        controls,
        lags=REFERENCE_LAGS,
        max_horizon=max_horizon,
        max_donors=REFERENCE_DONORS,
    )
    for field in (
        "n_original_treated_events",
        "n_matched_events",
        "n_selected_donors_across_matched_events",
    ):
        if current[field] != frozen.get(field):
            raise ValueError(f"Reference rule changed from frozen audit: {field}")
    for field in ("mean_abs_gap_pool", "mean_abs_gap_selected"):
        a = current["pretrend_slope_balance"][field]
        b = frozen["pretrend_slope_balance"][field]
        if (
            a is None
            or b is None
            or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError(f"Reference pretrend changed: {field}")
    report["frozen_reference_check"] = "same_source_and_reference_estimates_reproduced"
    report["reference_matches_prior_audit_by_construction"] = True


def save_sensitivity(
    output_dir: Path,
    grid: list[SensitivityPoint],
    details: list[SensitivityEvent],
    report: dict[str, object],
    *,
    source_panel: Path,
    provenance: Path | None = None,
    frozen_reference: Path | None = None,
    panel: dict[Key, Observation] | None = None,
    controls: dict[Key, MacroValues] | None = None,
) -> None:
    """Save machine-readable grid, event-level donor sets and source hashes."""
    if not grid or not details:
        raise ValueError("Refusing to publish an empty sensitivity grid")
    data = dict(report)
    source_hash = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    data["joined_panel_sha256"] = source_hash
    if provenance is not None:
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if source.get("publisher") != "World Bank World Development Indicators (DS2)":
            raise ValueError("Unexpected WDI provenance source")
        series = source.get("series")
        if not isinstance(series, list) or len(series) != 4:
            raise ValueError("Expected four WDI source ZIP audit entries")
        codes = [row.get("indicator_code") for row in series]
        if len(set(codes)) != 4 or not {
            "NY.GDP.PCAP.KD.ZG",
            "FP.CPI.TOTL.ZG",
            "NE.TRD.GNFS.ZS",
            "SL.UEM.TOTL.ZS",
        }.issubset(codes):
            raise ValueError("WDI source manifest lacks the expected indicators")
        data["source_archive_sha256"] = {
            row["indicator_code"]: row["source_sha256"] for row in series
        }
        data["source_last_updated"] = {
            row["indicator_code"]: row["source_last_updated"] for row in series
        }
    if frozen_reference is not None:
        if panel is None or controls is None:
            raise ValueError("Checking frozen reference requires the loaded panel")
        _check_reference(
            data,
            frozen_reference,
            source_hash,
            panel,
            controls,
            int(data["max_horizon_fixed"]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "macro_sensitivity_grid.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SensitivityPoint.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in grid)
    with (output_dir / "macro_sensitivity_events.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SensitivityEvent.__dataclass_fields__)
        )
        writer.writeheader()
        for row in details:
            values = asdict(row)
            values["selected_donors"] = ";".join(row.selected_donors)
            writer.writerow(values)
    (output_dir / "macro_sensitivity_audit.json").write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run a predeclared sensitivity grid on the existing real-data join."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--frozen-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lags", nargs="+", type=int, default=list(DEFAULT_LAGS))
    parser.add_argument(
        "--donor-caps", nargs="+", type=int, default=list(DEFAULT_DONOR_CAPS)
    )
    parser.add_argument("--max-horizon", type=int, default=5)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    controls = read_macro_controls(args.panel, set(panel))
    grid, detailed, report = audit_sensitivity_grid(
        panel,
        controls,
        lag_values=tuple(args.lags),
        donor_caps=tuple(args.donor_caps),
        max_horizon=args.max_horizon,
    )
    save_sensitivity(
        args.output_dir,
        grid,
        detailed,
        report,
        source_panel=args.panel,
        provenance=args.provenance,
        frozen_reference=args.frozen_reference,
        panel=panel,
        controls=controls,
    )
    print(
        f"Saved {len(grid)} sensitivity rules over {report['n_common_matched_events']} "
        "common reform events"
    )
    print("CAUTION: balance sensitivity only; no causal effects or inference")


if __name__ == "__main__":
    main()
