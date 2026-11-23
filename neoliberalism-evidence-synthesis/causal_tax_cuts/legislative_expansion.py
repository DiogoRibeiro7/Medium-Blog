"""Expand primary-law context without changing the tax-cut study treatment.

Every legal record documents a *specific provision* and its tax-period rule.
It does not certify that the provision caused the author's annual composite
2-SD flag. Original 30 flags, 25 runs and 20 donor-eligible onsets are frozen.
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

ATTRIBUTION = "provision_period_verified_author_flag_attribution_unverified"
ALLOWED_HOSTS = frozenset(
    {
        "www.govinfo.gov",
        "www.legislation.gov.uk",
        "www.bgbl.de",
        "www.bundesfinanzministerium.de",
        "zoek.officielebekendmakingen.nl",
    }
)
EVIDENCE_FIELDS = frozenset(
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


@dataclass(frozen=True, slots=True)
class ProvisionEvidence:
    """Audited statute context without any flag-to-law attribution claim."""

    country: str
    author_year: int
    statute_title: str
    enacted_on: str
    provision_identifier: str
    period_start: str
    period_end: str | None
    tax_period_rule: str
    official_source_url: str
    period_rule_source_url: str
    ledger: str
    period_precedes_enactment: bool


@dataclass(frozen=True, slots=True)
class CoverageFlag:
    """One original author flag with explicit legal-context review status."""

    country: str
    author_year: int
    run_start: int
    is_onset: bool
    baseline_donor_eligible_onset: bool
    legal_context_status: str
    statute_title: str
    enacted_on: str
    provision_period_start: str
    provision_precedes_enactment: bool | None
    author_composite_effective_date_verified: bool
    causal_estimand_identified: bool


def _rows(path: Path, fields: frozenset[str]) -> list[dict[str, str]]:
    """Read a nonempty source and validate the exact required columns."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not fields.issubset(reader.fieldnames):
            raise ValueError(f"Invalid source schema: {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Missing original source rows: {path.name}")
    return rows


def _date(value: str) -> date:
    """Accept a complete ISO date, never an inferred year boundary."""
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"Noncanonical legal ISO date: {value!r}")
    return parsed


def _official(url: str) -> str:
    """Restrict source links to the independently reviewed official hosts."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Nonofficial or ambiguous source URL: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Source URLs must not contain credentials")
    return url


def read_evidence(
    legacy_file: Path, expansion_file: Path
) -> tuple[dict[tuple[str, int], ProvisionEvidence], int, int]:
    """Combine the previous 3 cases and 4 new cases, without mutation."""
    evidence: dict[tuple[str, int], ProvisionEvidence] = {}
    for ledger, path in (
        ("previously_reviewed", legacy_file),
        ("new_review", expansion_file),
    ):
        for row in _rows(path, EVIDENCE_FIELDS):
            key = row["country"], int(row["author_year"])
            if key in evidence:
                raise ValueError(f"Duplicate legal evidence for author flag: {key}")
            if row["attribution_status"] != ATTRIBUTION:
                raise ValueError("The composite-index attribution is not verified")
            enacted = _date(row["enacted_on"])
            start = _date(row["first_applicable_period_start"])
            end = (
                _date(row["last_applicable_period_day"])
                if row["last_applicable_period_day"]
                else None
            )
            if end is not None and end < start:
                raise ValueError("Tax-period end precedes its beginning")
            if enacted.year > key[1]:
                raise ValueError("Legislation postdates the author flag-year")
            if start.year != key[1]:
                raise ValueError(
                    "This audit records a provision-year context, not a re-dated flag"
                )
            if (
                not row["statute_title"].strip()
                or not row["provision_identifier"].strip()
            ):
                raise ValueError("Statute and individual provision must be named")
            if not row["taxable_period_kind"].strip():
                raise ValueError("Explicit tax-period scope is mandatory")
            rules_url = row.get("period_rule_source_url") or row["official_source_url"]
            evidence[key] = ProvisionEvidence(
                country=key[0],
                author_year=key[1],
                statute_title=row["statute_title"],
                enacted_on=enacted.isoformat(),
                provision_identifier=row["provision_identifier"],
                period_start=start.isoformat(),
                period_end=end.isoformat() if end else None,
                tax_period_rule=row["taxable_period_kind"],
                official_source_url=_official(row["official_source_url"]),
                period_rule_source_url=_official(rules_url),
                ledger=ledger,
                period_precedes_enactment=start < enacted,
            )
    counts = Counter(item.ledger for item in evidence.values())
    return evidence, counts["previously_reviewed"], counts["new_review"]


def audit_legislative_coverage(
    flagged_rows: list[dict[str, str]],
    onset_rows: list[dict[str, str]],
    evidence: dict[tuple[str, int], ProvisionEvidence],
    previous: dict[str, Any],
    *,
    legacy_count: int,
    expansion_count: int,
) -> tuple[list[CoverageFlag], dict[str, Any]]:
    """Reconcile context to all original flag-years and unchanged onsets."""
    flags: dict[tuple[str, int], int] = {}
    onset_keys: set[tuple[str, int]] = set()
    for row in flagged_rows:
        key = row["country"], int(row["author_year"])
        first = int(row["run_start"])
        if key in flags or first > key[1]:
            raise ValueError(f"Duplicate or invalid original author flag: {key}")
        flags[key] = first
        if first == key[1]:
            onset_keys.add(key)
    if set(evidence) - set(flags):
        raise ValueError(
            "A legislative record refers to a nonexistent 2-SD author flag"
        )

    onsets: dict[tuple[str, int], bool] = {}
    for row in onset_rows:
        key = row["country"], int(row["first_flag_year"])
        if key in onsets or key not in onset_keys:
            raise ValueError("Original run-onset table and author flag-years differ")
        if row["baseline_donor_eligible"] not in ("True", "False"):
            raise ValueError("Unknown original donor-eligibility classification")
        onsets[key] = row["baseline_donor_eligible"] == "True"
    if set(onsets) != onset_keys:
        raise ValueError("Legislation audit would change the original onset cohort")

    if (
        len(flags) != previous["n_author_flag_years"]
        or len(onsets) != previous["n_author_flag_runs"]
        or sum(onsets.values()) != previous["n_baseline_donor_eligible_onsets"]
        or legacy_count != previous["n_provision_context_rows"]
    ):
        raise ValueError("Live input has drifted from the earlier onset audit")

    register: list[CoverageFlag] = []
    for key in sorted(flags, key=lambda pair: (pair[1], pair[0])):
        record = evidence.get(key)
        is_onset = key in onset_keys
        register.append(
            CoverageFlag(
                country=key[0],
                author_year=key[1],
                run_start=flags[key],
                is_onset=is_onset,
                baseline_donor_eligible_onset=onsets.get(key, False),
                legal_context_status="provision_period_context_only"
                if record
                else "unreviewed",
                statute_title=record.statute_title if record else "",
                enacted_on=record.enacted_on if record else "",
                provision_period_start=record.period_start if record else "",
                provision_precedes_enactment=record.period_precedes_enactment
                if record
                else None,
                author_composite_effective_date_verified=False,
                causal_estimand_identified=False,
            )
        )

    reviewed = [row for row in register if row.legal_context_status != "unreviewed"]
    legal_onsets = [row for row in reviewed if row.is_onset]
    eligible_legal = [row for row in legal_onsets if row.baseline_donor_eligible_onset]
    statute_ids = {
        (row.country, row.statute_title, row.enacted_on) for row in evidence.values()
    }
    report: dict[str, Any] = {
        "status": "legislation_context_coverage_and_nonidentification_only",
        "n_author_flag_years": len(register),
        "n_original_onsets": len(onsets),
        "n_baseline_donor_eligible_onsets": sum(onsets.values()),
        "n_previous_legal_context_flag_years": legacy_count,
        "n_new_legal_context_flag_years": expansion_count,
        "n_total_legal_context_flag_years": len(reviewed),
        "n_unreviewed_flag_years": len(register) - len(reviewed),
        "n_reviewed_onsets": len(legal_onsets),
        "n_unreviewed_onsets": len(onsets) - len(legal_onsets),
        "n_reviewed_eligible_onsets": len(eligible_legal),
        "n_unreviewed_eligible_onsets": sum(onsets.values()) - len(eligible_legal),
        "n_distinct_documented_statutes": len(statute_ids),
        "n_reviewed_period_starts_before_enactment": sum(
            item.period_precedes_enactment for item in evidence.values()
        ),
        "n_author_flag_specific_implementation_dates_verified": 0,
        "n_eligible_onsets_with_policy_attribution_verified": 0,
        "n_eligible_onsets_causal_identified": 0,
        "new_legal_case_ids": [
            f"{item.country}-{item.author_year}"
            for item in sorted(
                evidence.values(), key=lambda x: (x.author_year, x.country)
            )
            if item.ledger == "new_review"
        ],
        "legal_scope": (
            "Statute dates and provision-level applicable tax periods only. "
            "Neither proves a one-to-one mapping to the Hope-Limberg composite threshold."
        ),
        "time_order": "No re-dating or grouping of the original annual treatment flags",
        "inference": "none; no treatment effect, policy-specific causal estimand, p-value or uncertainty interval",
        "limitations": [
            "The original latent composite may respond to several contemporaneous tax changes.",
            "A provision tax-period start is not an implementation day for the composite index.",
            "Country-year flags, recurring donor countries and later reform censoring remain unchanged.",
            "Only two of the twenty original eligible onsets had strict joint WDI donor-hull support at year five.",
            "Source citations were manually reviewed; the CI checks schema and hashes but does not re-research historical legal claims.",
        ],
    }
    return register, report


def run(
    flags_file: Path,
    onsets_file: Path,
    legacy_file: Path,
    expansion_file: Path,
    previous_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a version-stamped, hash-reproducible descriptive coverage audit."""
    previous = json.loads(previous_file.read_text(encoding="utf-8"))
    if previous.get("status") != "author_flag_onsets_and_provision_periods_only":
        raise ValueError("Unexpected previous research audit type")
    hashes = {
        "author_flag_years": hashlib.sha256(flags_file.read_bytes()).hexdigest(),
        "onset_eligibility": hashlib.sha256(onsets_file.read_bytes()).hexdigest(),
        "previous_provision_periods": hashlib.sha256(
            legacy_file.read_bytes()
        ).hexdigest(),
        "new_legal_context": hashlib.sha256(expansion_file.read_bytes()).hexdigest(),
        "previous_onset_audit": hashlib.sha256(previous_file.read_bytes()).hexdigest(),
    }
    if (
        hashes["author_flag_years"] != previous["source_sha256"]["author_flag_years"]
        or hashes["previous_provision_periods"]
        != previous["source_sha256"]["tax_provision_ledger"]
    ):
        raise ValueError(
            "Previous author flags or legal context no longer match their original audit"
        )
    evidence, old_count, new_count = read_evidence(legacy_file, expansion_file)
    flags = _rows(flags_file, frozenset({"country", "author_year", "run_start"}))
    onsets = _rows(
        onsets_file,
        frozenset({"country", "first_flag_year", "baseline_donor_eligible"}),
    )
    register, summary = audit_legislative_coverage(
        flags,
        onsets,
        evidence,
        previous,
        legacy_count=old_count,
        expansion_count=new_count,
    )
    report: dict[str, Any] = {
        **summary,
        "source_sha256": hashes,
        "author_source_sha256": previous["author_source_sha256"],
        "joined_panel_sha256": previous["joined_panel_sha256"],
        "evidence_attribution_status": ATTRIBUTION,
        "statutory_sources": [
            {
                "country": item.country,
                "author_year": item.author_year,
                "statute": item.statute_title,
                "date": item.enacted_on,
                "source": item.official_source_url,
                "effective_period_source": item.period_rule_source_url,
                "period_start": item.period_start,
                "period_end": item.period_end,
                "ledger": item.ledger,
                "tax_period_rule": item.tax_period_rule,
                "attribution": ATTRIBUTION,
            }
            for item in sorted(
                evidence.values(), key=lambda x: (x.author_year, x.country)
            )
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "legislative_review_all_flag_years.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(CoverageFlag.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in register)
    (output_dir / "legislative_coverage_audit.json").write_text(
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Run the research audit using immutable upstream GitHub Actions inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--onsets", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.flags,
        args.onsets,
        args.legacy,
        args.expansion,
        args.previous,
        args.output_dir,
    )
    print(
        f"Documented provision context: {report['n_total_legal_context_flag_years']}/"
        f"{report['n_author_flag_years']} flag-years, "
        f"{report['n_reviewed_eligible_onsets']}/"
        f"{report['n_baseline_donor_eligible_onsets']} eligible onsets"
    )
    print("CAUTION: legal-period context only; zero verified composite-treatment dates")


if __name__ == "__main__":
    main()
