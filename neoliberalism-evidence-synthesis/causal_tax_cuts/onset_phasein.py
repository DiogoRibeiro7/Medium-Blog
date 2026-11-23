"""Reconcile author flag-run onsets, donor eligibility, and legal tax periods.

This is a source-preserving *measurement audit*. It does not recode the
Hope-Limberg treatment, equate annual flags to statutes, or estimate effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_FOLLOW_UP = 5
TIMING_STATUS = "author_flag_onsets_and_provision_periods_only"
ATTRIBUTION = "provision_period_verified_author_flag_attribution_unverified"
APPROVED_HOSTS = frozenset({"www.govinfo.gov", "www.legislation.gov.uk"})


@dataclass(frozen=True, slots=True)
class Flag:
    """One previously verified author-coded flag year."""

    country: str
    year: int
    run_start: int
    run_end: int
    run_length: int
    run_position: int
    wid_observed: bool


@dataclass(frozen=True, slots=True)
class Support:
    """Previous donor-support status, unchanged by the timing audit."""

    country: str
    year: int
    status: str
    donors_h5: int


@dataclass(frozen=True, slots=True)
class Onset:
    """First flagged year of a contiguous run, not necessarily a new law."""

    country: str
    first_flag_year: int
    run_end_year: int
    flags_in_run: int
    wid_observed_at_onset: bool
    previously_reported_support_status: str
    baseline_donor_eligible: bool
    original_donors_at_h5: int
    next_flag_within_five_years: int | None
    next_flag_relation: str
    no_additional_flags_through_five_years: bool


@dataclass(frozen=True, slots=True)
class ProvisionContext:
    """Verified statutory tax-period scope, NOT coded-treatment causation."""

    country: str
    author_year: int
    statute_title: str
    enacted_on: str
    provision_identifier: str
    first_applicable_period_start: str
    last_applicable_period_day: str
    taxable_period_kind: str
    official_source_url: str
    attribution_status: str
    period_starts_before_enactment: bool
    period_start_is_january_first: bool


def _read_csv(path: Path, required: frozenset[str]) -> list[dict[str, str]]:
    """Reject unsupported file shapes, duplicate headers, and empty files."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not required.issubset(headers):
            raise ValueError(f"Unexpected CSV schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _bool(raw: str) -> bool:
    """Accept only explicit Boolean strings from the previous audit."""
    if raw not in {"True", "False"}:
        raise ValueError(f"Unrecognised Boolean {raw!r}")
    return raw == "True"


def load_flags(path: Path) -> list[Flag]:
    """Validate the 30-author-flag register and its consecutive-run labeling."""
    required = frozenset(
        {
            "country",
            "author_year",
            "run_start",
            "run_end",
            "run_year_count",
            "year_position_in_run",
            "wid_outcome_observed",
        }
    )
    flags: list[Flag] = []
    seen: set[tuple[str, int]] = set()
    for row in _read_csv(path, required):
        flag = Flag(
            row["country"],
            int(row["author_year"]),
            int(row["run_start"]),
            int(row["run_end"]),
            int(row["run_year_count"]),
            int(row["year_position_in_run"]),
            _bool(row["wid_outcome_observed"]),
        )
        key = (flag.country, flag.year)
        if key in seen or flag.year != flag.run_start + flag.run_position - 1:
            raise ValueError(f"Duplicate or inconsistent flag year: {key}")
        if (
            flag.run_length != flag.run_end - flag.run_start + 1
            or not 1 <= flag.run_position <= flag.run_length
        ):
            raise ValueError(f"Inconsistent run boundaries: {key}")
        seen.add(key)
        flags.append(flag)
    runs: dict[tuple[str, int], list[Flag]] = {}
    for row in flags:
        runs.setdefault((row.country, row.run_start), []).append(row)
    for (country, first), members in runs.items():
        expected = range(first, first + members[0].run_length)
        if sorted(row.year for row in members) != list(expected):
            raise ValueError(f"Incomplete flagged-year sequence: {country}, {first}")
        if len({(row.run_end, row.run_length) for row in members}) != 1:
            raise ValueError(f"Inconsistent run size: {country}, {first}")
    return sorted(flags, key=lambda row: (row.country, row.year))


def load_support(path: Path) -> dict[tuple[str, int], Support]:
    """Load exactly the previously published event statuses (no re-matching)."""
    required = frozenset(
        {"country", "reform_year", "status", "n_horizon_five_donors"}
    )
    result: dict[tuple[str, int], Support] = {}
    for row in _read_csv(path, required):
        year = int(row["reform_year"])
        key = (row["country"], year)
        if key in result:
            raise ValueError(f"Duplicate eligibility record: {key}")
        donors = int(row["n_horizon_five_donors"])
        if donors < 0 or (
            row["status"] != "baseline_donors_available" and donors != 0
        ):
            raise ValueError(f"Invalid inherited donor status: {key}")
        result[key] = Support(key[0], year, row["status"], donors)
    return result


def load_provisions(
    path: Path, flagged: set[tuple[str, int]]
) -> list[ProvisionContext]:
    """Parse official provision periods; never equate them to composite flags."""
    required = frozenset(
        {
            "country",
            "author_year",
            "statute_title",
            "enacted_on",
            "provision_identifier",
            "first_applicable_period_start",
            "last_applicable_period_day",
            "taxable_period_kind",
            "official_source_url",
            "attribution_status",
        }
    )
    result: list[ProvisionContext] = []
    unique: set[tuple[str, int, str]] = set()
    for row in _read_csv(path, required):
        year = int(row["author_year"])
        country = row["country"]
        if (country, year) not in flagged:
            raise ValueError(f"Uncoded author year in provision ledger: {country}-{year}")
        if row["attribution_status"] != ATTRIBUTION:
            raise ValueError("Provision dates cannot certify author-flag attribution")
        url = urlsplit(row["official_source_url"])
        if url.scheme != "https" or url.hostname not in APPROVED_HOSTS:
            raise ValueError("Provision timing must cite an approved official source")
        enacted = date.fromisoformat(row["enacted_on"])
        period_start = date.fromisoformat(row["first_applicable_period_start"])
        period_end = (
            date.fromisoformat(row["last_applicable_period_day"])
            if row["last_applicable_period_day"]
            else None
        )
        if period_end is not None and period_end < period_start:
            raise ValueError("Statutory tax period ends before it begins")
        if not all(row[field].strip() for field in ("statute_title", "provision_identifier", "taxable_period_kind")):
            raise ValueError("Missing statutory evidence description")
        key = (country, year, row["provision_identifier"])
        if key in unique:
            raise ValueError(f"Duplicate provision context: {key}")
        unique.add(key)
        result.append(
            ProvisionContext(
                country,
                year,
                row["statute_title"],
                enacted.isoformat(),
                row["provision_identifier"],
                period_start.isoformat(),
                period_end.isoformat() if period_end else "",
                row["taxable_period_kind"],
                row["official_source_url"],
                row["attribution_status"],
                period_start < enacted,
                (period_start.month, period_start.day) == (1, 1),
            )
        )
    return sorted(result, key=lambda row: (row.country, row.author_year))


def audit_onsets(
    flags: list[Flag],
    support: dict[tuple[str, int], Support],
    provisions: list[ProvisionContext],
) -> tuple[list[Onset], dict[str, Any]]:
    """Reconcile all starts and repeats with unchanged donor risk sets."""
    flagged = {(row.country, row.year): row for row in flags}
    observed = {key for key, row in flagged.items() if row.wid_observed}
    if set(support) != observed:
        raise ValueError("Inherited donor eligibility does not cover each observed flag")
    starts = [row for row in flags if row.run_position == 1]
    if not starts:
        raise ValueError("No policy-onset records found")
    records: list[Onset] = []
    for row in starts:
        key = (row.country, row.year)
        previous = support.get(key)
        status = previous.status if previous else "missing_wid_outcome"
        if previous is None and row.wid_observed:
            raise ValueError(f"WID-observed onset missing eligibility: {key}")
        following = sorted(
            (other for other in flags if other.country == row.country
             and row.year < other.year <= row.year + MAX_FOLLOW_UP),
            key=lambda item: item.year,
        )
        next_year = following[0].year if following else None
        relation = (
            "same_contiguous_flag_run"
            if following and following[0].run_start == row.year
            else "separate_later_flag_run" if following else "none"
        )
        records.append(
            Onset(
                row.country,
                row.year,
                row.run_end,
                row.run_length,
                row.wid_observed,
                status,
                status == "baseline_donors_available",
                previous.donors_h5 if previous else 0,
                next_year,
                relation,
                next_year is None,
            )
        )
    eligible = [row for row in records if row.baseline_donor_eligible]
    eligible_followup = [
        row for row in eligible if row.next_flag_within_five_years is not None
    ]
    if any(row.first_flag_year != row.run_end_year and row.flags_in_run == 1 for row in records):
        raise ValueError("Inconsistent onset sequence")
    if any((row.country, row.author_year) not in flagged for row in provisions):
        raise ValueError("Unmatched legislative context")
    report = {
        "status": TIMING_STATUS,
        "n_author_flag_years": len(flags),
        "n_author_flag_runs": len(records),
        "n_continuation_flags": len(flags) - len(records),
        "n_flag_years_with_wid": len(observed),
        "n_first_year_flags_with_wid": sum(row.wid_observed_at_onset for row in records),
        "n_missing_outcome_first_year_flags": sum(not row.wid_observed_at_onset for row in records),
        "n_baseline_donor_eligible_onsets": len(eligible),
        "onset_support_status_counts": dict(sorted(Counter(row.previously_reported_support_status for row in records).items())),
        "n_eligible_onsets_with_repeat_by_five_years": len(eligible_followup),
        "n_repeat_next_year_same_run": sum(
            row.next_flag_relation == "same_contiguous_flag_run"
            for row in eligible_followup
        ),
        "n_repeat_new_run_within_five_years": sum(
            row.next_flag_relation == "separate_later_flag_run"
            for row in eligible_followup
        ),
        "n_eligible_onsets_without_repeat_by_five": sum(
            row.no_additional_flags_through_five_years for row in eligible
        ),
        "repeating_eligible_events": [
            {
                "country": row.country,
                "onset_year": row.first_flag_year,
                "next_flag_year": row.next_flag_within_five_years,
                "relation": row.next_flag_relation,
            }
            for row in eligible_followup
        ],
        "n_provision_context_rows": len(provisions),
        "n_flag_years_with_provision_context": len({(p.country, p.author_year) for p in provisions}),
        "n_provision_periods_starting_before_enactment": sum(
            p.period_starts_before_enactment for p in provisions
        ),
        "n_author_flags_with_identified_composite_effective_day": 0,
        "treatment_version": (
            "Original author binary annual 2-SD flag-years retained. Run starts, "
            "repeat flags and statute provision periods are diagnostic only."
        ),
        "legislative_scope": (
            "Statute provisions and their tax-period scopes are documented, "
            "but their attribution to the author's composite threshold is not."
        ),
        "inference": "none; no counterfactual weights, adjusted ATT, p-values or inference",
    }
    return records, report


def save_audit(
    output_dir: Path,
    onsets: list[Onset],
    provisions: list[ProvisionContext],
    report: dict[str, Any],
    *,
    flags_path: Path,
    support_path: Path,
    provision_path: Path,
    timing_audit_path: Path,
    donor_audit_path: Path,
) -> None:
    """Save deterministic source-hashed output with upstream reconciliation."""
    old_timing = json.loads(timing_audit_path.read_text(encoding="utf-8"))
    old_support = json.loads(donor_audit_path.read_text(encoding="utf-8"))
    if old_timing.get("joined_panel_sha256") != old_support.get("joined_panel_sha256"):
        raise ValueError("Prior timing and donor audits use different source panels")
    for current, expected, label in (
        (report["n_author_flag_years"], old_timing["n_author_flag_years"], "flags"),
        (report["n_author_flag_runs"], old_timing["n_descriptive_consecutive_runs"], "runs"),
        (report["n_flag_years_with_wid"], old_timing["n_flag_years_with_joined_wid_outcome"], "WID flags"),
        (report["n_baseline_donor_eligible_onsets"], old_support["n_eligible_reforms_before_future_censoring"], "eligible onsets"),
    ):
        if current != expected:
            raise ValueError(f"Original source and onset audit disagree on {label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, kind in (
        ("tax_cut_onset_eligibility.csv", onsets, Onset),
        ("tax_provision_periods.csv", provisions, ProvisionContext),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kind.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    manifest = {
        **report,
        "author_source_sha256": old_timing["author_source_sha256"],
        "joined_panel_sha256": old_timing["joined_panel_sha256"],
        "source_sha256": {
            "author_flag_years": hashlib.sha256(flags_path.read_bytes()).hexdigest(),
            "support_eligibility": hashlib.sha256(support_path.read_bytes()).hexdigest(),
            "tax_provision_ledger": hashlib.sha256(provision_path.read_bytes()).hexdigest(),
            "timing_audit": hashlib.sha256(timing_audit_path.read_bytes()).hexdigest(),
            "donor_audit": hashlib.sha256(donor_audit_path.read_bytes()).hexdigest(),
        },
        "source_vintage": "unmodified Hope-Limberg policy flags and WID/WDI donor selection",
        "statutory_context_license": "public legal text; cited primary legislation",
    }
    (output_dir / "tax_cut_onset_audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Reproduce the onset and provision-period diagnostic from prior audits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--provisions", required=True, type=Path)
    parser.add_argument("--timing-audit", required=True, type=Path)
    parser.add_argument("--donor-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    flags = load_flags(args.flags)
    support = load_support(args.support)
    periods = load_provisions(
        args.provisions, {(row.country, row.year) for row in flags}
    )
    onsets, report = audit_onsets(flags, support, periods)
    save_audit(
        args.output_dir,
        onsets,
        periods,
        report,
        flags_path=args.flags,
        support_path=args.support,
        provision_path=args.provisions,
        timing_audit_path=args.timing_audit,
        donor_audit_path=args.donor_audit,
    )
    print(
        f"Author flags {report['n_author_flag_years']}, runs {report['n_author_flag_runs']}, "
        f"donor-eligible onsets {report['n_baseline_donor_eligible_onsets']}; "
        f"five-year repeats {report['n_eligible_onsets_with_repeat_by_five_years']}"
    )
    print("No law-to-composite-flag attribution or causal estimates")


if __name__ == "__main__":
    main()
