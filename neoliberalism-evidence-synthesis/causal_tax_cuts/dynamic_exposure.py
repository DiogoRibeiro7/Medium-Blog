"""Audit dynamic tax-cut pathways before defining a longitudinal causal design.

The existing matched comparison retains repeat reforms in treated countries
but removes donors after their own reforms. This module measures that
asymmetry and the availability of lagged annual WDI covariates for both
arms, without fitting a propensity model, censoring weights or an ATT.

All baseline eligibility reuses the frozen, full-window donor-support audit.
No post-baseline macro measurement changes that eligibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.donor_support import (
    LAGS,
    MAX_HORIZON,
    CandidateExposure,
    audit_future_donor_exposure,
)
from causal_tax_cuts.macro_matching import MacroValues, read_macro_controls
from causal_tax_cuts.world_bank import CORE_COLUMNS, INDICATORS

Role = Literal["initiator", "comparator"]


@dataclass(frozen=True, slots=True)
class PolicyPathYear:
    """Observed exposure and covariate availability for one arm/event/year."""

    focal_country: str
    reform_year: int
    role: Role
    country: str
    horizon: int
    major_cut_this_year: int
    cumulative_major_cuts_since_t: int
    deviations_from_declared_strategy: int
    adherent_to_declared_strategy: bool
    prior_year_wdi_complete: bool
    all_predecision_wdi_complete_through_h: bool


@dataclass(frozen=True, slots=True)
class ReformRecurrence:
    """A baseline-eligible initiating reform and repeat-cut exposure."""

    country: str
    reform_year: int
    n_baseline_comparators: int
    first_additional_major_cut_year: int | None
    n_additional_major_cuts_through_h5: int


@dataclass(frozen=True, slots=True)
class PathwayEventHorizon:
    """Sustained-strategy feasibility for one original reform and horizon."""

    country: str
    reform_year: int
    horizon: int
    treated_additional_cuts_through_h: int
    treated_single_cut_strategy_adherent: bool
    treated_all_predecision_wdi_complete: bool
    n_baseline_comparators: int
    n_no_cut_comparators_through_h: int
    n_no_cut_comparators_with_sequential_wdi: int
    n_both_strategy_adherent_pairs: int
    n_both_strategy_adherent_pairs_with_sequential_wdi: int


def _wdi_observed(controls: dict[Key, MacroValues], country: str, year: int) -> bool:
    """Check three annual WDI cells without imputing or leaking future data."""
    values = controls.get((country, year))
    if values is None:
        return False
    if len(values) != len(CORE_COLUMNS):
        raise ValueError(f"Malformed WDI profile at {country}, {year}")
    return all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    )


def _path(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    *,
    focal: str,
    year: int,
    role: Role,
    country: str,
) -> list[PolicyPathYear]:
    """Construct one observed path under an explicitly *hypothetical* rule.

    'initiator': cut at t, then no additional major cut through each h.
    'comparator': no major cut at t or through each h. A comparator cut at
    t is already nonadherent at h=0; it is not a later follow-up event.
    """
    if role not in ("initiator", "comparator"):
        raise ValueError("Unrecognised policy-path role")
    if (role == "initiator") != (country == focal):
        raise ValueError("The initiating country must match the focal country")
    if role == "initiator" and panel[country, year].treated != 1:
        raise ValueError("Focal observation lacks its baseline major cut")
    # Same-year reforming comparators are deliberately in the pre-censor
    # pool, but are nonadherent immediately at horizon zero.

    rows: list[PolicyPathYear] = []
    cuts = 0
    deviations = 0
    sequential_wdi = True
    for h in range(MAX_HORIZON + 1):
        reform = panel[country, year + h].treated
        if reform not in (0, 1):
            raise ValueError("Tax-cut indicator must be binary")
        cuts += reform
        deviations += int(reform == 1 and (role == "comparator" or h > 0))
        # Policy timing is yearly, so only *lagged* WDI is a candidate for
        # conditioning at this decision. Availability is not exchangeability.
        previous_wdi = _wdi_observed(controls, country, year + h - 1)
        sequential_wdi = sequential_wdi and previous_wdi
        rows.append(
            PolicyPathYear(
                focal,
                year,
                role,
                country,
                h,
                reform,
                cuts,
                deviations,
                deviations == 0,
                previous_wdi,
                sequential_wdi,
            )
        )
    return rows


def audit_dynamic_exposure(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
) -> tuple[
    list[PolicyPathYear],
    list[ReformRecurrence],
    list[PathwayEventHorizon],
    dict[str, object],
]:
    """Quantify observed treatment-path asymmetry and sequential WDI coverage.

    The complete t-5..t+5 WID window and pre-five WDI support are inherited
    from donor_support. Both candidate strategies are *descriptions* only:
    no treatment assignment mechanism or causal exchangeability is assumed.
    """
    if not panel or set(panel) != set(controls):
        raise ValueError("WID/WDI country-year keys must match")
    reforms, candidates, prior_horizons, previous = audit_future_donor_exposure(
        panel, controls
    )
    eligible = sorted(
        (row for row in reforms if row.status == "baseline_donors_available"),
        key=lambda row: (row.reform_year, row.country),
    )
    if not eligible:
        raise ValueError("No eligible tax-cut reforms with valid donor histories")

    donors_by_event: dict[Key, list[CandidateExposure]] = defaultdict(list)
    for row in candidates:
        donors_by_event[row.country, row.reform_year].append(row)
    prior_risk_sets = {
        (row.country, row.reform_year, row.horizon): row for row in prior_horizons
    }
    all_paths: list[PolicyPathYear] = []
    reform_recurrences: list[ReformRecurrence] = []
    event_horizons: list[PathwayEventHorizon] = []

    for reform in eligible:
        key = (reform.country, reform.reform_year)
        donor_entries = sorted(donors_by_event[key], key=lambda row: row.donor)
        if len(donor_entries) != reform.n_pre_censor_donors:
            raise ValueError(f"Baseline donor support diverged for {key}")

        treated_path = _path(
            panel,
            controls,
            focal=reform.country,
            year=reform.reform_year,
            role="initiator",
            country=reform.country,
        )
        donor_paths = [
            _path(
                panel,
                controls,
                focal=reform.country,
                year=reform.reform_year,
                role="comparator",
                country=entry.donor,
            )
            for entry in donor_entries
        ]
        all_paths.extend(treated_path)
        for donor_path in donor_paths:
            all_paths.extend(donor_path)
        additional_cut_years = [
            reform.reform_year + row.horizon
            for row in treated_path[1:]
            if row.major_cut_this_year == 1
        ]
        reform_recurrences.append(
            ReformRecurrence(
                reform.country,
                reform.reform_year,
                len(donor_entries),
                additional_cut_years[0] if additional_cut_years else None,
                len(additional_cut_years),
            )
        )

        for horizon in range(MAX_HORIZON + 1):
            treated = treated_path[horizon]
            control_rows = [path[horizon] for path in donor_paths]
            nonreforming = [
                row for row in control_rows if row.adherent_to_declared_strategy
            ]
            with_wdi = [
                row
                for row in nonreforming
                if row.all_predecision_wdi_complete_through_h
            ]
            previously = prior_risk_sets.get(
                (reform.country, reform.reform_year, horizon)
            )
            # The previously frozen donor-support audit must be recovered
            # exactly; no new comparison cohort is constructed here.
            if (
                previously is None
                or previously.n_surviving_donors != len(nonreforming)
                or previously.n_censored_donors != len(control_rows) - len(nonreforming)
            ):
                raise ValueError(
                    f"Future donor risk set disagrees with prior audit: {key}, {horizon}"
                )
            event_horizons.append(
                PathwayEventHorizon(
                    reform.country,
                    reform.reform_year,
                    horizon,
                    treated.deviations_from_declared_strategy,
                    treated.adherent_to_declared_strategy,
                    treated.all_predecision_wdi_complete_through_h,
                    len(donor_entries),
                    len(nonreforming),
                    len(with_wdi),
                    len(nonreforming) if treated.adherent_to_declared_strategy else 0,
                    (
                        len(with_wdi)
                        if (
                            treated.adherent_to_declared_strategy
                            and treated.all_predecision_wdi_complete_through_h
                        )
                        else 0
                    ),
                )
            )

    horizon_report: list[dict[str, int]] = []
    for h in range(MAX_HORIZON + 1):
        rows = [row for row in event_horizons if row.horizon == h]
        horizon_report.append(
            {
                "horizon": h,
                "n_eligible_initiating_reforms": len(rows),
                "n_initiators_with_subsequent_major_cut": sum(
                    row.treated_additional_cuts_through_h > 0 for row in rows
                ),
                "n_initiators_single_cut_adherent": sum(
                    row.treated_single_cut_strategy_adherent for row in rows
                ),
                "n_initiators_with_complete_sequential_lagged_wdi": sum(
                    row.treated_all_predecision_wdi_complete for row in rows
                ),
                "n_no_cut_comparator_pairs": sum(
                    row.n_no_cut_comparators_through_h for row in rows
                ),
                "n_no_cut_comparator_pairs_with_sequential_wdi": sum(
                    row.n_no_cut_comparators_with_sequential_wdi for row in rows
                ),
                "n_reforms_with_both_strategy_arms_adherent": sum(
                    row.treated_single_cut_strategy_adherent
                    and row.n_no_cut_comparators_through_h > 0
                    for row in rows
                ),
                "n_pairs_with_both_strategy_arms_adherent": sum(
                    row.n_both_strategy_adherent_pairs for row in rows
                ),
                "n_pairs_with_both_arms_adherent_and_sequential_wdi": sum(
                    row.n_both_strategy_adherent_pairs_with_sequential_wdi
                    for row in rows
                ),
            }
        )
        prior = next(row for row in previous["by_horizon"] if row["horizon"] == h)
        if (
            horizon_report[-1]["n_no_cut_comparator_pairs"]
            != prior["n_donor_reform_pairs_at_risk"]
        ):
            raise ValueError("Aggregate policy risk sets disagree with donor support")

    report: dict[str, object] = {
        "status": "dynamic_policy_pathway_and_covariate_availability_audit_only",
        "baseline_history_years": LAGS,
        "followup_years": MAX_HORIZON,
        "n_original_observed_reform_events": previous["n_original_reform_events"],
        "n_baseline_eligible_reforms": len(eligible),
        "n_pre_censor_comparator_pairs": len(candidates),
        "original_reform_status_counts": dict(
            sorted(Counter(row.status for row in reforms).items())
        ),
        "by_horizon": horizon_report,
        "strategy_a": "major cut at t followed by no additional coded major cuts t+1..t+h",
        "strategy_b": "no coded major cut in t..t+h",
        "observed_original_design": "repeat treated cuts retained; comparator cuts from t onward excluded",
        "predecision_wdi_availability": "three annual WDI variables at t+h-1, cumulatively complete from t-1 to t+h-1; coverage only",
        "cohort_policy": "fixed WID outcome t-5..t+5; matching WDI only t-5..t-1",
        "reconciled_to_prior_donor_risk_sets": True,
        "causal_estimation": "none; no weights, ATT, model, test or uncertainty interval",
        "identification_barriers": [
            "The original comparator's future-reform censoring is asymmetric with retention of repeated treated reforms.",
            "Both strategies are event-history descriptions; binary 2-SD cuts do not specify a uniform policy dose or other tax changes.",
            "Yearly WDI at t+h-1 may be affected by earlier tax cuts and may not be available before the next policy decision.",
            "Sequential exchangeability, baseline policy assignment, genuine joint support, calendar-time common shocks and country dependence are unresolved.",
            "Effective independent units are countries, not re-used donor-reform episodes.",
            "Requiring observed WID outcomes through t+5 already selects on post-baseline data availability.",
        ],
    }
    return all_paths, reform_recurrences, event_horizons, report


def save_dynamic_exposure(
    output_dir: Path,
    paths: list[PolicyPathYear],
    recurrences: list[ReformRecurrence],
    event_horizons: list[PathwayEventHorizon],
    report: dict[str, object],
    *,
    source_panel: Path,
    provenance: Path | None = None,
    frozen_reference: Path | None = None,
) -> None:
    """Write complete event-time paths with provenance and vintage checks."""
    if not paths or not recurrences or not event_horizons:
        raise ValueError("Refusing to save empty exposure-path evidence")
    data = dict(report)
    data["joined_panel_sha256"] = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    if provenance is not None:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
        if manifest.get("publisher") != "World Bank World Development Indicators (DS2)":
            raise ValueError("Unexpected World Bank source provenance")
        series = manifest.get("series")
        if (
            not isinstance(series, list)
            or {row.get("indicator_code") for row in series}
            != {indicator.code for indicator in INDICATORS}
            or len(series) != len(INDICATORS)
        ):
            raise ValueError("Expected the same four official World Bank series")
        data["source_archive_sha256"] = {
            row["indicator_code"]: row["source_sha256"] for row in series
        }
        data["source_last_updated"] = {
            row["indicator_code"]: row["source_last_updated"] for row in series
        }
    if frozen_reference is not None:
        frozen = json.loads(frozen_reference.read_text(encoding="utf-8"))
        if frozen.get("joined_panel_sha256") == data["joined_panel_sha256"]:
            reference_counts = {
                int(row["horizon"]): int(row["n_donor_reform_pairs_at_risk"])
                for row in frozen["by_horizon"]
            }
            for row in data["by_horizon"]:
                if row["n_no_cut_comparator_pairs"] != reference_counts[row["horizon"]]:
                    raise ValueError("Same-source frozen donor exposure mismatch")
            data["frozen_reference_status"] = "same_source_validated"
        else:
            data["frozen_reference_status"] = "source_vintage_changed"

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, fields in (
        ("dynamic_policy_paths.csv", paths, list(PolicyPathYear.__dataclass_fields__)),
        (
            "dynamic_reform_recurrence.csv",
            recurrences,
            list(ReformRecurrence.__dataclass_fields__),
        ),
        (
            "dynamic_event_horizons.csv",
            event_horizons,
            list(PathwayEventHorizon.__dataclass_fields__),
        ),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output_dir / "dynamic_exposure_audit.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Audit the fixed WID/WDI panel without estimating causal effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--frozen-reference", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    controls = read_macro_controls(args.panel, set(panel))
    paths, recurrences, horizons, report = audit_dynamic_exposure(panel, controls)
    save_dynamic_exposure(
        args.output_dir,
        paths,
        recurrences,
        horizons,
        report,
        source_panel=args.panel,
        provenance=args.provenance,
        frozen_reference=args.frozen_reference,
    )
    last = report["by_horizon"][-1]
    print(
        f"Pathways: {len(recurrences)} baseline reforms, "
        f"{last['n_initiators_with_subsequent_major_cut']} with repeat cuts by h5"
    )
    print("CAUTION: dynamic exposure and WDI coverage audit only; no causal estimate")


if __name__ == "__main__":
    main()
