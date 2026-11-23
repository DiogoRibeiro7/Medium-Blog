"""Audit 1986 Finnish and 2008 German tax provisions without attributing flags.

The archived Hope-Limberg annual composite, WID/WDI samples and the existing
20-event feasibility matrix are immutable inputs. Official legislation and
contemporaneous government accounts establish provision-specific details,
not a mapping to the composite tax-cut indicator or a causal treatment effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

CASES = frozenset({("FIN", 1986), ("DEU", 2008)})
REQUIRED_FIELDS = frozenset(
    {
        "country",
        "flag_year",
        "component_id",
        "instrument",
        "change_kind",
        "value_before",
        "value_after",
        "unit",
        "law_title",
        "law_identifier",
        "law_enacted_on",
        "period_start",
        "period_scope",
        "cash_flow_caveat",
        "evidence_class",
        "official_source_url",
        "corroborating_source_url",
        "composite_flag_attribution",
    }
)
ALLOWED_DOMAINS = frozenset(
    {
        "www.finlex.fi",
        "www.edilex.fi",
        "publications.bof.fi",
        "www.vero.fi",
        "dserver.bundestag.de",
        "www.bundesfinanzministerium.de",
    }
)
CHANGE_KINDS = frozenset(
    {
        "rate_reduction",
        "wealth_threshold_increase",
        "advance_payment_transition",
        "capital_gains_base_broadening",
        "trade_tax_assessment_reform",
        "interest_deduction_restriction",
        "later_capital_income_withholding",
    }
)
EVIDENCE_CLASSES = frozenset(
    {
        "statute_index_and_contemporaneous_ministry_report",
        "enacted_amendment_and_contemporaneous_ministry_report",
        "parliamentary_law_dossier_and_federal_finance_ministry",
    }
)


@dataclass(frozen=True, slots=True)
class Component:
    """Source-auditable policy instrument, not a measured study treatment."""

    country: str
    flag_year: int
    component_id: str
    instrument: str
    change_kind: str
    value_before: float | None
    value_after: float | None
    unit: str
    law_title: str
    law_identifier: str
    law_enacted_on: str | None
    period_start: str
    period_scope: str
    cash_flow_caveat: str
    evidence_class: str
    official_source_url: str
    corroborating_source_url: str
    composite_flag_attribution: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """Two existing onsets, with prior matching gates copied verbatim."""

    country: str
    reform_year: int
    documented_component_count: int
    statutory_date_observed_for_every_component: bool
    n_components_with_later_tax_year: int
    has_advance_payment_transition: bool
    has_countervailing_tax_base_or_deduction_change: bool
    original_h5_donors: int
    original_h5_complete_political_donors: int
    original_strict_joint_wdi_support_h5: bool
    original_pretrend_slope_gap_pp_per_year: float
    original_repeat_major_cut_h5: bool
    author_composite_flag_attribution_verified: bool
    policy_specific_effect_identified: bool


def _read_rows(path: Path, required: frozenset[str]) -> list[dict[str, str]]:
    """Validate a nonempty CSV with unique headings and required columns."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not required.issubset(reader.fieldnames)
        ):
            raise ValueError(f"Unexpected CSV schema in {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty evidence source: {path.name}")
    return rows


def _date(value: str) -> date:
    """Require a complete ISO date rather than an inferred annual label."""
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError(f"Noncanonical ISO date: {value}")
    return result


def _url(value: str) -> str:
    """Accept only HTTPS government/parliament/central-bank source hosts."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_DOMAINS
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"Unapproved source URL: {value}")
    return value


def _number(value: str) -> float | None:
    """Keep absent rates absent; reject NaN and infinite placeholders."""
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite rate or threshold in documentary evidence")
    return number


def load_components(
    path: Path, *, expected_cases: frozenset[tuple[str, int]] = CASES
) -> list[Component]:
    """Read law-level evidence without recoding author annual treatment."""
    records: list[Component] = []
    seen: set[tuple[str, int, str]] = set()
    for row in _read_rows(path, REQUIRED_FIELDS):
        try:
            year = int(row["flag_year"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid annual author treatment year") from exc
        country = row["country"]
        key = country, year
        component_key = country, year, row["component_id"]
        if key not in expected_cases or component_key in seen:
            raise ValueError(f"Unexpected or duplicate policy component {component_key}")
        seen.add(component_key)
        if row["change_kind"] not in CHANGE_KINDS:
            raise ValueError(f"Unknown tax-policy component kind: {row['change_kind']}")
        if row["evidence_class"] not in EVIDENCE_CLASSES:
            raise ValueError("Unrecognised or overstated legal source evidence class")
        if row["composite_flag_attribution"] != "unverified":
            raise ValueError("No statute-to-composite-flag attribution is verified")
        start = _date(row["period_start"])
        if start.year not in (year, year + 1):
            raise ValueError("Tax-period start is outside the declared flag context")
        enacted = _date(row["law_enacted_on"]) if row["law_enacted_on"] else None
        if enacted is not None and enacted > start:
            raise ValueError("Enactment date follows the stated first legal tax period")
        before = _number(row["value_before"])
        after = _number(row["value_after"])
        if row["change_kind"] == "rate_reduction" and (
            before is None or after is None or after >= before
        ):
            raise ValueError("Documented rate reduction requires before > after")
        if row["change_kind"] == "wealth_threshold_increase" and (
            before is None or after is None or after <= before
        ):
            raise ValueError("Documented exemption increase requires after > before")
        if row["change_kind"] == "advance_payment_transition" and (
            before is None or after != before or not row["cash_flow_caveat"]
        ):
            raise ValueError("Advance payments must preserve the old rate explicitly")
        if not row["law_title"] or not row["law_identifier"] or not row["instrument"]:
            raise ValueError("Missing legal instrument title or identifier")
        if not row["period_scope"]:
            raise ValueError("Missing legal-tax-year scope")
        records.append(
            Component(
                country=country,
                flag_year=year,
                component_id=row["component_id"],
                instrument=row["instrument"],
                change_kind=row["change_kind"],
                value_before=before,
                value_after=after,
                unit=row["unit"],
                law_title=row["law_title"],
                law_identifier=row["law_identifier"],
                law_enacted_on=enacted.isoformat() if enacted else None,
                period_start=start.isoformat(),
                period_scope=row["period_scope"],
                cash_flow_caveat=row["cash_flow_caveat"],
                evidence_class=row["evidence_class"],
                official_source_url=_url(row["official_source_url"]),
                corroborating_source_url=_url(row["corroborating_source_url"]),
                composite_flag_attribution=row["composite_flag_attribution"],
            )
        )
    if {(item.country, item.flag_year) for item in records} != expected_cases:
        raise ValueError("Both original donor-hull-supported onsets are required")
    return sorted(records, key=lambda c: (c.flag_year, c.country, c.component_id))


def assess_candidates(
    records: list[Component],
    previous_rows: list[dict[str, str]],
    previous: dict[str, Any],
    *,
    expected_cases: frozenset[tuple[str, int]] = CASES,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Retain the prior 20-case design and report new measurement caveats."""
    if (
        previous.get("status") != "policy_specific_identification_feasibility_only"
        or previous.get("n_original_eligible_reform_onsets") != 20
        or previous.get("n_eligible_onsets_with_statute_provision_context") != 5
        or previous.get("n_eligible_with_h5_strict_joint_wdi_support") != 2
        or set(previous.get("h5_support_without_legal_context", []))
        != {"FIN-1986", "DEU-2008"}
    ):
        raise ValueError("Prior identification audit is not the frozen reference")
    by_id: dict[tuple[str, int], dict[str, str]] = {}
    for row in previous_rows:
        key = row["country"], int(row["reform_year"])
        if key in by_id:
            raise ValueError(f"Duplicate prior reform in candidate audit {key}")
        by_id[key] = row
    if len(by_id) != 20 or not expected_cases.issubset(by_id):
        raise ValueError("Original event sample changed or candidates are missing")

    grouped: dict[tuple[str, int], list[Component]] = {
        key: [] for key in expected_cases
    }
    for item in records:
        key = item.country, item.flag_year
        if key not in grouped:
            raise ValueError("New component would alter the original treatment sample")
        grouped[key].append(item)
    candidates: list[Candidate] = []
    for country, year in sorted(expected_cases, key=lambda x: (x[1], x[0])):
        key = country, year
        prior = by_id[key]
        if prior["provision_context_reviewed"] != "False":
            raise ValueError("Case is already covered by the legacy legal register")
        if prior["strict_joint_wdi_support_at_h5"] != "True":
            raise ValueError("Candidate no longer has original five-year WDI support")
        if (
            prior["policy_specific_causal_design_supported"] != "False"
            or prior["feasibility_status"] != "not_identified_with_current_evidence"
        ):
            raise ValueError("Existing effect identification status was changed")
        slope = float(prior["pretrend_slope_gap_selected_pp_per_year"])
        if not math.isfinite(slope):
            raise ValueError("Missing or invalid original income-share slope gap")
        groups = grouped[key]
        if not groups:
            raise ValueError(f"No law-level evidence for original case: {key}")
        candidates.append(
            Candidate(
                country=country,
                reform_year=year,
                documented_component_count=len(groups),
                statutory_date_observed_for_every_component=all(
                    row.law_enacted_on is not None for row in groups
                ),
                n_components_with_later_tax_year=sum(
                    int(row.period_start[:4]) == year + 1 for row in groups
                ),
                has_advance_payment_transition=any(
                    row.change_kind == "advance_payment_transition"
                    for row in groups
                ),
                has_countervailing_tax_base_or_deduction_change=any(
                    row.change_kind
                    in {"capital_gains_base_broadening", "interest_deduction_restriction"}
                    for row in groups
                ),
                original_h5_donors=int(prior["n_surviving_donors_at_h5"]),
                original_h5_complete_political_donors=int(
                    prior["n_h5_donors_with_complete_political_history"]
                ),
                original_strict_joint_wdi_support_h5=True,
                original_pretrend_slope_gap_pp_per_year=slope,
                original_repeat_major_cut_h5=(
                    prior["additional_major_cut_through_h5"] == "True"
                ),
                author_composite_flag_attribution_verified=False,
                policy_specific_effect_identified=False,
            )
        )
    if any(
        row.original_h5_donors < row.original_h5_complete_political_donors
        or row.original_h5_donors <= 0
        for row in candidates
    ):
        raise ValueError("Unchanged donors contradict original political coverage")
    report: dict[str, Any] = {
        "status": "finland_1986_germany_2008_tax_component_evidence_only",
        "n_original_author_flag_years": previous["n_original_author_flag_years"],
        "n_original_eligible_reform_onsets": len(by_id),
        "n_prior_documented_eligible_onsets": 5,
        "n_new_candidate_onsets": len(candidates),
        "n_eligible_onsets_with_statutory_or_contemporaneous_context": (
            5 + len(candidates)
        ),
        "n_component_records": len(records),
        "component_change_kinds": dict(
            sorted(Counter(row.change_kind for row in records).items())
        ),
        "n_candidate_cases_with_original_h5_joint_wdi_support": sum(
            row.original_strict_joint_wdi_support_h5 for row in candidates
        ),
        "n_candidates_with_verified_complete_statute_enactment_dates": sum(
            row.statutory_date_observed_for_every_component for row in candidates
        ),
        "n_candidate_components_with_later_tax_year": sum(
            row.n_components_with_later_tax_year for row in candidates
        ),
        "n_candidates_with_collection_lag": sum(
            row.has_advance_payment_transition for row in candidates
        ),
        "n_candidates_with_countervailing_tax_base_measures": sum(
            row.has_countervailing_tax_base_or_deduction_change
            for row in candidates
        ),
        "n_candidates_with_no_additional_author_flag_through_h5": sum(
            not row.original_repeat_major_cut_h5 for row in candidates
        ),
        "n_composite_flag_to_statute_attributions_verified": 0,
        "n_policy_specific_causal_estimands_identified": 0,
        "interpretation": (
            "Legally documented component changes exist for both previously "
            "unreviewed h5-supported events, but neither has a proven "
            "source-component-to-composite flag mapping, clean policy dose, "
            "or justified causal counterfactual."
        ),
        "inference": "none; no updated ATT, model, weights, CI or p-value",
        "known_limitations": [
            "Finland's exact promulgation date for its annual 1986 rate law is not recorded here.",
            "Finnish 1986 corporate advances retained 43 percent despite the 33 percent statutory rate; refunds occurred later.",
            "Finnish 1986 capital-gains base broadening offsets some tax relief.",
            "Germany's 2008 law had multiple components and a private capital-income stage in 2009.",
            "The German 2008 reform intersects major global macroeconomic shocks.",
            "Pretrend slope differences remain observed and are not parallel-trend tests.",
            "Finite-sample WDI donor-hull support is not causal positivity or exchangeability.",
            "Original donor censoring, full WID future-outcome selection and country dependence are unchanged.",
        ],
    }
    return candidates, report


def save_audit(
    output_dir: Path,
    components: list[Component],
    candidates: list[Candidate],
    report: dict[str, Any],
    *,
    components_file: Path,
    previous_csv: Path,
    previous_json: Path,
) -> dict[str, Any]:
    """Persist deterministic audit and precise source lineage."""
    result = {
        **report,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("official_component_ledger", components_file),
                ("prior_event_gates", previous_csv),
                ("prior_identification_audit", previous_json),
            )
        },
        "joined_panel_sha256": json.loads(
            previous_json.read_text(encoding="utf-8")
        )["joined_panel_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, kind in (
        ("policy_component_evidence.csv", components, Component),
        ("supported_reform_candidate_gates.csv", candidates, Candidate),
    ):
        with (output_dir / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kind.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output_dir / "policy_component_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Reconcile candidate laws against the previously frozen event sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--prior-events", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_components(args.components)
    previous_rows = _read_rows(
        args.prior_events,
        frozenset(
            {
                "country",
                "reform_year",
                "provision_context_reviewed",
                "strict_joint_wdi_support_at_h5",
                "policy_specific_causal_design_supported",
                "feasibility_status",
                "n_surviving_donors_at_h5",
                "n_h5_donors_with_complete_political_history",
                "pretrend_slope_gap_selected_pp_per_year",
                "additional_major_cut_through_h5",
            }
        ),
    )
    prior = json.loads(args.prior_audit.read_text(encoding="utf-8"))
    candidates, report = assess_candidates(rows, previous_rows, prior)
    save_audit(
        args.output_dir,
        rows,
        candidates,
        report,
        components_file=args.components,
        previous_csv=args.prior_events,
        previous_json=args.prior_audit,
    )
    print(
        "Audited two statutory-policy contexts across "
        f"{len(rows)} components; 2/2 original h5 WDI hull support, "
        "0/2 composite-attributions or causal effects"
    )


if __name__ == "__main__":
    main()
