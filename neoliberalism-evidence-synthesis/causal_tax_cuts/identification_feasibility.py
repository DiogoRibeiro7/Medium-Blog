"""Audit whether documented tax cuts support a policy-specific causal design.

Use the *same* 20 Hope-Limberg tax-cut onsets and official WID/WDI/DPI
country-year observations as the earlier audits. A legal tax-period
reference is not an intervention assignment, and donor convex-hull overlap
is not causal positivity. No ATT, weights, or uncertainty are estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HORIZONS = tuple(range(-1, 6))
LEGAL_STATUS = "provision_period_context_only"
NOT_IDENTIFIED = "not_identified_with_current_evidence"


@dataclass(frozen=True, slots=True)
class ReformGate:
    """Independent necessary diagnostics for one unchanged reform onset."""

    country: str
    reform_year: int
    provision_context_reviewed: bool
    composite_policy_attribution_verified: bool
    composite_implementation_date_verified: bool
    strict_joint_wdi_support_at_h0: bool
    strict_joint_wdi_support_at_h3: bool
    strict_joint_wdi_support_at_h5: bool
    n_surviving_donors_at_h5: int
    additional_major_cut_through_h5: bool
    political_pre5_history_complete: bool
    n_h5_donors_with_complete_political_history: int
    pretrend_slope_gap_selected_pp_per_year: float
    policy_specific_causal_design_supported: bool
    feasibility_status: str


@dataclass(frozen=True, slots=True)
class HorizonGate:
    """One reform/horizon comparison retaining the existing donor restrictions."""

    country: str
    reform_year: int
    horizon: int
    provision_context_reviewed: bool
    strict_joint_wdi_support: bool
    within_all_three_marginal_wdi_ranges: bool
    n_surviving_donors: int
    repeated_initiator_cut_by_horizon: bool | None


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Fail closed on missing required data or duplicate CSV headings."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not required.issubset(reader.fieldnames)
        ):
            raise ValueError(f"Unexpected source schema: {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty source: {path.name}")
    return rows


def normalized_scientific_csv_digest(path: Path, field: str) -> str:
    """Hash a complete derived CSV after rounding one numerical-only field.

    Keep every row and all other columns unchanged. Round Mahalanobis
    distances or simplex reconstruction residuals to 12 decimal places;
    this is a *reproducibility identity*, not an analysis transformation.
    Differences above the rounding threshold must still change the hash.
    """
    records = _read_rows(path, {field})
    normalized: list[dict[str, str]] = []
    for row in records:
        record = dict(row)
        raw = record[field]
        if raw:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite derived diagnostic in {field}")
            record[field] = format(round(value, 12), ".12f")
        normalized.append(record)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bool(value: str) -> bool:
    """Reject implicit truthiness of CSV strings such as 'False'."""
    if value not in ("True", "False"):
        raise ValueError(f"Expected explicit boolean, got {value!r}")
    return value == "True"


def _key(row: dict[str, str]) -> tuple[str, int]:
    """Create validated country-year identity for joining source audits."""
    country = row.get("country")
    year = int(row["reform_year"])
    if not country or len(country) != 3 or not 1965 <= year <= 2015:
        raise ValueError(f"Invalid study reform ID: {country} {year}")
    return country, year


def _unique(
    rows: list[dict[str, str]], *, horizon: bool = False
) -> dict[tuple[str, int] | tuple[str, int, int], dict[str, str]]:
    """Index source rows and reject duplicate event/horizon observations."""
    result: dict[tuple[str, int] | tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = _key(row)
        index = (*key, int(row["horizon"])) if horizon else key
        if index in result:
            raise ValueError(f"Duplicate source event/horizon key: {index}")
        result[index] = row
    return result


def _audit_manifest(path: Path, required_status: str) -> dict[str, Any]:
    """Read a versioned frozen/live source manifest with basic shape checks."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != required_status:
        raise ValueError(f"Unexpected source audit status in {path.name}")
    if not isinstance(data.get("joined_panel_sha256"), str):
        raise TypeError(f"Missing joined-panel provenance in {path.name}")
    return data


def evaluate_identification(
    legal_rows: list[dict[str, str]],
    hull_rows: list[dict[str, str]],
    dynamic_rows: list[dict[str, str]],
    political_rows: list[dict[str, str]],
    macro_rows: list[dict[str, str]],
    *,
    expected_reforms: int = 20,
) -> tuple[list[ReformGate], list[HorizonGate], dict[str, Any]]:
    """Join *descriptive* gates without changing baseline eligibility.

    The WDI hull test examines three lagged macro means. Failure means
    interpolation under that specific donor scheme is unsupported; it
    does not prove that no conceivable causal strategy could work.
    Pretrend slopes are diagnostics, not a formal acceptance threshold.
    """
    original_flags: set[tuple[str, int]] = set()
    eligible: set[tuple[str, int]] = set()
    reviewed: set[tuple[str, int]] = set()
    for row in legal_rows:
        key = row["country"], int(row["author_year"])
        if key in original_flags:
            raise ValueError("Duplicate original 2-SD author flag")
        original_flags.add(key)
        if row["legal_context_status"] not in (LEGAL_STATUS, "unreviewed"):
            raise ValueError("Unknown legal review status")
        if _bool(row["is_onset"]) and _bool(row["baseline_donor_eligible_onset"]):
            eligible.add(key)
        if row["legal_context_status"] == LEGAL_STATUS:
            reviewed.add(key)
        if _bool(row["author_composite_effective_date_verified"]) or _bool(
            row["causal_estimand_identified"]
        ):
            raise ValueError("Prior documentary audit does not identify tax effects")
    if len(eligible) != expected_reforms:
        raise ValueError("Eligible cohort changed from frozen reference")

    hull = _unique(hull_rows, horizon=True)
    dynamic = _unique(dynamic_rows, horizon=True)
    political = _unique(political_rows)
    macro = _unique(macro_rows)
    hull_events = {(country, year) for country, year, _ in hull}
    dyn_events = {(country, year) for country, year, _ in dynamic}
    if hull_events != eligible or dyn_events != eligible:
        raise ValueError("Official WDI/dynamic donor samples differ from legal cohort")
    if set(political) != eligible:
        raise ValueError("Political-history cohort differs from legal cohort")
    if {key for key, row in macro.items() if row["status"] == "matched"} != eligible:
        raise ValueError("Macro-matching cohort differs from legal cohort")
    for key in eligible:
        if {h for c, y, h in hull if (c, y) == key} != set(HORIZONS):
            raise ValueError(f"Missing joint-support horizon for {key}")
        if {h for c, y, h in dynamic if (c, y) == key} != set(HORIZONS[1:]):
            raise ValueError(f"Missing dynamic-exposure horizon for {key}")

    event_rows: list[ReformGate] = []
    horizon_rows: list[HorizonGate] = []
    for country, year in sorted(eligible, key=lambda x: (x[1], x[0])):
        key = country, year
        source = political[key]
        prehistory = _bool(source["treated_all_three_pre5_complete"])
        slope = float(macro[key]["pretrend_slope_gap_selected"])
        if not math.isfinite(slope):
            raise ValueError(f"Nonfinite pretrend slope gap for {key}")
        at_five = dynamic[country, year, 5]
        repeats = int(at_five["treated_additional_cuts_through_h"])
        if repeats < 0:
            raise ValueError(f"Invalid post-reform tax-cut count for {key}")
        complete_donors = int(source["n_horizon_five_donors_all_three_complete"])
        if complete_donors < 0:
            raise ValueError("Invalid political donor completeness count")
        for h in HORIZONS:
            joint = hull[country, year, h]
            survivors = int(joint["n_surviving_donors"])
            if survivors < 0:
                raise ValueError("Negative donor count")
            repeated = None
            if h >= 0:
                followup = dynamic[country, year, h]
                if survivors != int(followup["n_no_cut_comparators_through_h"]):
                    raise ValueError(
                        f"Contradictory dynamic and WDI donor set: {key} h={h}"
                    )
                repeated = int(followup["treated_additional_cuts_through_h"]) > 0
            strict = _bool(joint["strict_joint_hull_support"])
            marginal = _bool(joint["within_all_three_marginal_ranges"])
            if strict and (not marginal or survivors == 0):
                raise ValueError("Geometric joint support without marginal support")
            horizon_rows.append(
                HorizonGate(
                    country,
                    year,
                    h,
                    key in reviewed,
                    strict,
                    marginal,
                    survivors,
                    repeated,
                )
            )
        h5_donors = int(hull[country, year, 5]["n_surviving_donors"])
        if complete_donors > h5_donors:
            raise ValueError("Political-complete donors exceed at-risk donors")
        event_rows.append(
            ReformGate(
                country=country,
                reform_year=year,
                provision_context_reviewed=key in reviewed,
                composite_policy_attribution_verified=False,
                composite_implementation_date_verified=False,
                strict_joint_wdi_support_at_h0=_bool(
                    hull[country, year, 0]["strict_joint_hull_support"]
                ),
                strict_joint_wdi_support_at_h3=_bool(
                    hull[country, year, 3]["strict_joint_hull_support"]
                ),
                strict_joint_wdi_support_at_h5=_bool(
                    hull[country, year, 5]["strict_joint_hull_support"]
                ),
                n_surviving_donors_at_h5=h5_donors,
                additional_major_cut_through_h5=repeats > 0,
                political_pre5_history_complete=prehistory,
                n_h5_donors_with_complete_political_history=complete_donors,
                pretrend_slope_gap_selected_pp_per_year=slope,
                policy_specific_causal_design_supported=False,
                feasibility_status=NOT_IDENTIFIED,
            )
        )

    documented = [r for r in event_rows if r.provision_context_reviewed]
    h5_support = [r for r in event_rows if r.strict_joint_wdi_support_at_h5]
    summary: dict[str, Any] = {
        "status": "policy_specific_identification_feasibility_only",
        "n_original_author_flag_years": len(original_flags),
        "n_original_eligible_reform_onsets": len(event_rows),
        "n_eligible_onsets_with_statute_provision_context": len(documented),
        "n_eligible_onsets_without_statute_provision_context": (
            len(event_rows) - len(documented)
        ),
        "n_documented_eligible_with_h0_strict_joint_wdi_support": sum(
            r.strict_joint_wdi_support_at_h0 for r in documented
        ),
        "n_documented_eligible_with_h3_strict_joint_wdi_support": sum(
            r.strict_joint_wdi_support_at_h3 for r in documented
        ),
        "n_documented_eligible_with_h5_strict_joint_wdi_support": sum(
            r.strict_joint_wdi_support_at_h5 for r in documented
        ),
        "n_eligible_with_h5_strict_joint_wdi_support": len(h5_support),
        "h5_support_without_legal_context": [
            f"{r.country}-{r.reform_year}"
            for r in h5_support
            if not r.provision_context_reviewed
        ],
        "n_documented_eligible_with_full_political_pre5": sum(
            r.political_pre5_history_complete for r in documented
        ),
        "n_documented_eligible_with_repeat_cut_by_h5": sum(
            r.additional_major_cut_through_h5 for r in documented
        ),
        "n_eligible_with_repeat_cut_by_h5": sum(
            r.additional_major_cut_through_h5 for r in event_rows
        ),
        "n_policy_specific_interventions_attributed_to_source_law": 0,
        "n_policy_specific_effective_dates_validated": 0,
        "n_policy_specific_causal_designs_supported": 0,
        "horizons_audited": list(HORIZONS),
        "n_horizon_rows": len(horizon_rows),
        "design_rule": (
            "Policy-specific treatment requires documented source-component attribution, "
            "relevant decision/implementation timing, and a justified counterfactual design. "
            "These are not established here; diagnostic support is assessed separately."
        ),
        "interpretation": (
            "NOT identified with currently assembled evidence. Convex-hull failure "
            "limits this particular interpolation design, not every possible method."
        ),
        "pretrend_definition": (
            "Observed pre-period income-share slope gap for macro-selected donors "
            "under the earlier reference selection; not a proof of parallel trends."
        ),
        "inference": "none; no new counterfactual, adjusted effect, SE, CI or p-value",
        "limitations": [
            "A legal provision's tax-period rule does not attribute the author's composite flag.",
            "The three-variable WDI donor-hull check is not formal positivity or full covariate support.",
            "The matching risk sets condition on donor future tax cuts and full follow-up WID outcomes.",
            "DPI political history completeness does not imply political covariate balance.",
            "Pre-period slope diagnostics cannot establish parallel counterfactual trends.",
            "Events reuse treated and donor countries; pair counts are not independent units.",
        ],
    }
    return event_rows, horizon_rows, summary


def run(
    *,
    legal_file: Path,
    hull_file: Path,
    dynamic_file: Path,
    political_file: Path,
    macro_file: Path,
    legal_audit: Path,
    hull_audit: Path,
    dynamic_audit: Path,
    political_audit: Path,
    macro_audit: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Reconcile source vintages and write a deterministic research-gate audit."""
    manifests = (
        _audit_manifest(
            legal_audit, "legislation_context_coverage_and_nonidentification_only"
        ),
        _audit_manifest(
            hull_audit, "three_wdi_dimension_joint_convex_hull_diagnostic_only"
        ),
        _audit_manifest(
            dynamic_audit,
            "dynamic_policy_pathway_and_covariate_availability_audit_only",
        ),
        _audit_manifest(
            political_audit, "pre_tax_cut_political_institutional_coverage_only"
        ),
        _audit_manifest(
            macro_audit, "pre_reform_support_and_balance_only_no_adjusted_effects"
        ),
    )
    joined = {str(item["joined_panel_sha256"]) for item in manifests}
    if len(joined) != 1:
        raise ValueError("Incompatible WID/WDI source vintages across research gates")
    raw_files = {
        "legal_register": legal_file,
        "joint_wdi_hull": hull_file,
        "dynamic_tax_cut_paths": dynamic_file,
        "political_history_coverage": political_file,
        "macro_matching_pretrends": macro_file,
        "legal_source_audit": legal_audit,
        "joint_wdi_source_audit": hull_audit,
        "dynamic_source_audit": dynamic_audit,
        "political_source_audit": political_audit,
        "macro_source_audit": macro_audit,
    }
    rows = {
        key: _read_rows(path, required)
        for key, path, required in (
            (
                "legal",
                legal_file,
                {
                    "country",
                    "author_year",
                    "is_onset",
                    "baseline_donor_eligible_onset",
                    "legal_context_status",
                    "author_composite_effective_date_verified",
                    "causal_estimand_identified",
                },
            ),
            (
                "hull",
                hull_file,
                {
                    "country",
                    "reform_year",
                    "horizon",
                    "n_surviving_donors",
                    "strict_joint_hull_support",
                    "within_all_three_marginal_ranges",
                },
            ),
            (
                "dynamic",
                dynamic_file,
                {
                    "country",
                    "reform_year",
                    "horizon",
                    "n_no_cut_comparators_through_h",
                    "treated_additional_cuts_through_h",
                },
            ),
            (
                "political",
                political_file,
                {
                    "country",
                    "reform_year",
                    "treated_all_three_pre5_complete",
                    "n_horizon_five_donors_all_three_complete",
                },
            ),
            (
                "macro",
                macro_file,
                {"country", "reform_year", "status", "pretrend_slope_gap_selected"},
            ),
        )
    }
    events, horizons, report = evaluate_identification(
        rows["legal"],
        rows["hull"],
        rows["dynamic"],
        rows["political"],
        rows["macro"],
    )
    if (
        report["n_original_author_flag_years"] != manifests[0]["n_author_flag_years"]
        or report["n_original_eligible_reform_onsets"]
        != manifests[0]["n_baseline_donor_eligible_onsets"]
        or report["n_eligible_onsets_with_statute_provision_context"]
        != manifests[0]["n_reviewed_eligible_onsets"]
        or report["n_eligible_with_h5_strict_joint_wdi_support"]
        != manifests[1]["by_horizon"][-1]["n_with_strict_joint_hull_support"]
        or report["n_eligible_with_repeat_cut_by_h5"]
        != manifests[2]["by_horizon"][-1]["n_initiators_with_subsequent_major_cut"]
    ):
        raise ValueError("Source summary disagrees with earlier frozen study gates")
    output = {
        **report,
        "joined_panel_sha256": joined.pop(),
        "author_source_sha256": manifests[0]["author_source_sha256"],
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(raw_files.items())
        },
        "normalized_input_sha256": {
            "joint_wdi_hull": normalized_scientific_csv_digest(
                hull_file, "witness_max_scaled_residual"
            ),
            "macro_matching_pretrends": normalized_scientific_csv_digest(
                macro_file, "mean_selected_distance"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values, rowtype in (
        ("policy_identification_events.csv", events, ReformGate),
        ("policy_identification_event_horizons.csv", horizons, HorizonGate),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rowtype.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(item) for item in values)
    (output_dir / "policy_identification_audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def verify_frozen_reference(actual: dict[str, Any], frozen: dict[str, Any]) -> str:
    """Distinguish original-source changes from numerical serialization drift.

    A raw-hash difference is tolerated *only* for the two specified
    derived numerical diagnostics, and only when the original joined
    panel, all scientifically used columns and the entire substantive
    feasibility result agree. This is not a claim that all raw files are
    byte-identical.
    """
    if actual.get("joined_panel_sha256") != frozen.get(
        "joined_panel_sha256"
    ) or actual.get("author_source_sha256") != frozen.get("author_source_sha256"):
        return "original_source_vintage_changed"
    first = actual.get("input_sha256")
    second = frozen.get("input_sha256")
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise TypeError("Expected complete source hash dictionaries")
    if first.keys() != second.keys():
        raise ValueError("Source input set changed on identical original panel")
    substantive_actual = {k: v for k, v in actual.items() if k != "input_sha256"}
    substantive_frozen = {k: v for k, v in frozen.items() if k != "input_sha256"}
    if substantive_actual != substantive_frozen:
        raise ValueError(
            "Original panel is unchanged but identification results or "
            "normalized scientific inputs have drifted"
        )
    changed = {key for key in first if first[key] != second[key]}
    if not changed:
        return "exact_raw_input_reproduction"
    allowed = {"joint_wdi_hull", "macro_matching_pretrends"}
    if not changed.issubset(allowed):
        raise ValueError(
            "Unexpected raw input differences on unchanged original data: "
            f"{sorted(changed)}"
        )
    return "same_original_data_and_scientific_inputs_derived_roundoff_only"


def main() -> None:
    """Build the audit from an immutable official-source CI collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "legal",
        "hull",
        "dynamic",
        "political",
        "macro",
        "legal-audit",
        "hull-audit",
        "dynamic-audit",
        "political-audit",
        "macro-audit",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        legal_file=args.legal,
        hull_file=args.hull,
        dynamic_file=args.dynamic,
        political_file=args.political,
        macro_file=args.macro,
        legal_audit=args.legal_audit,
        hull_audit=args.hull_audit,
        dynamic_audit=args.dynamic_audit,
        political_audit=args.political_audit,
        macro_audit=args.macro_audit,
        output_dir=args.output_dir,
    )
    print(
        f"Policy-specific evidence: "
        f"{summary['n_eligible_onsets_with_statute_provision_context']}/"
        f"{summary['n_original_eligible_reform_onsets']} eligible onsets "
        "documented, "
        f"{summary['n_documented_eligible_with_h5_strict_joint_wdi_support']}"
        " with year-five strict joint macro donor support"
    )
    print("CAUTION: no treatment mapping, causal ATT, or identified policy effect")


if __name__ == "__main__":
    main()
