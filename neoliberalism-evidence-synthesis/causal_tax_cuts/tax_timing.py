"""Audit annual major-tax-cut flags and legally documented timing examples.

The Hope–Limberg 2-SD indicator identifies country-year observations,
not verified enactment/implementation dates or independent statute events.
Consecutive indicator years are grouped solely for descriptive screening.

No counterfactual, ATT, inference, or alternative treatment assignment
is constructed. The original source and WID outcome window are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from causal_tax_cuts.prepare import PAPER_COUNTRIES

FIRST_YEAR = 1965
LAST_YEAR = 2015
EXPECTED_FLAGS = 30
EXPECTED_SOURCE_COUNTRY_ROWS = 19 * (LAST_YEAR - FIRST_YEAR + 1)
AUTHOR_PAGE = "https://www.taxrich.uk/data"
ALLOWED_STATUTE_HOSTS = frozenset({"www.govinfo.gov", "www.legislation.gov.uk"})
CONTEXT_ONLY = "statute_identified_flag_attribution_unverified"


@dataclass(frozen=True, slots=True)
class StatuteContext:
    """Verified statute date, not the date assigned to an author's flag."""

    country: str
    author_year: int
    title: str
    enacted_on: str
    primary_source_url: str
    tax_period_text: str
    evidence_scope: str


@dataclass(frozen=True, slots=True)
class FlaggedYear:
    """One author-coded major tax-cut year with a descriptive run label."""

    country: str
    author_year: int
    run_start: int
    run_end: int
    run_year_count: int
    year_position_in_run: int
    wid_outcome_observed: bool
    statute_context_title: str
    statute_enacted_on: str
    statute_context_scope: str


@dataclass(frozen=True, slots=True)
class ConsecutiveRun:
    """Consecutive binary flags; NOT a validated legislative episode."""

    country: str
    start_year: int
    end_year: int
    n_flagged_years: int
    n_flags_with_wid_outcomes: int


def _rows(path: Path, required: frozenset[str]) -> list[dict[str, str]]:
    """Read an original CSV, checking schema and nonempty data."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected CSV columns in {path.name}")
        records = list(reader)
    if not records:
        raise ValueError(f"Empty CSV: {path.name}")
    return records


def load_author_flags(
    path: Path,
    *,
    countries: frozenset[str] = PAPER_COUNTRIES,
    start_year: int = FIRST_YEAR,
    end_year: int = LAST_YEAR,
    expected_flags: int | None = EXPECTED_FLAGS,
) -> dict[tuple[str, int], int]:
    """Validate the complete author panel without inferring missing zeroes.

    The original author's country-year panel includes an extra Korea
    series, excluded explicitly from the study's 18-country sample.
    """
    if start_year > end_year or not countries:
        raise ValueError("Invalid study country-year rectangle")
    panel: dict[tuple[str, int], int] = {}
    for raw in _rows(path, frozenset({"iso3", "year", "taxcut_2SD"})):
        country = raw["iso3"].strip()
        if country not in countries:
            if country != "KOR":
                raise ValueError(f"Unexpected country in author file: {country}")
            continue
        try:
            year = int(raw["year"])
        except (ValueError, TypeError) as exc:
            raise ValueError("Noninteger author policy year") from exc
        if not start_year <= year <= end_year:
            raise ValueError(f"Author year outside study period: {year}")
        value = raw["taxcut_2SD"].strip()
        if value not in {"0", "1"}:
            raise ValueError(f"Nonbinary author tax-cut flag: {value}")
        key = (country, year)
        if key in panel:
            raise ValueError(f"Duplicate author country-year {key}")
        panel[key] = int(value)
    expected_keys = {
        (country, year)
        for country in countries
        for year in range(start_year, end_year + 1)
    }
    if set(panel) != expected_keys:
        raise ValueError(
            f"Unobserved author policy values for {len(expected_keys - set(panel))} "
            "country-years; refusing to infer missing zeroes"
        )
    if expected_flags is not None and sum(panel.values()) != expected_flags:
        raise ValueError(
            f"Expected {expected_flags} author flags, found {sum(panel.values())}"
        )
    return panel


def load_joined_outcomes(
    path: Path, author: dict[tuple[str, int], int]
) -> set[tuple[str, int]]:
    """Verify that the WID-derived join retains the author's treatment flags."""
    observed: set[tuple[str, int]] = set()
    for row in _rows(path, frozenset({"country", "year", "treated"})):
        key = (row["country"], int(row["year"]))
        if key not in author or key in observed:
            raise ValueError(f"Invalid joined outcome country-year: {key}")
        if row["treated"] != str(author[key]):
            raise ValueError(f"Policy flag disagrees between WID join and author: {key}")
        observed.add(key)
    return observed


def load_statute_contexts(
    path: Path, author: dict[tuple[str, int], int]
) -> dict[tuple[str, int], StatuteContext]:
    """Validate manually sourced statutory context without asserting equivalence.

    Only dated public primary-legislation records are permitted.
    Context entries cannot be used as event timing or re-coding evidence.
    """
    fields = frozenset({
        "country", "author_year", "title", "enacted_on",
        "primary_source_url", "tax_period_text", "evidence_scope",
    })
    result: dict[tuple[str, int], StatuteContext] = {}
    for raw in _rows(path, fields):
        country = raw["country"]
        try:
            year = int(raw["author_year"])
            enacted = date.fromisoformat(raw["enacted_on"])
        except ValueError as exc:
            raise ValueError("Invalid statute or author year") from exc
        key = (country, year)
        if author.get(key) != 1:
            raise ValueError(f"Statute context is not an observed author flag: {key}")
        if raw["evidence_scope"] != CONTEXT_ONLY:
            raise ValueError("Statutory context cannot certify a coded flag's cause")
        source_url = raw["primary_source_url"].strip()
        parsed = urlsplit(source_url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_STATUTE_HOSTS:
            raise ValueError(f"Not an approved official statute URL: {source_url}")
        if enacted.year > year:
            raise ValueError("Future statute cannot explain an earlier author flag")
        if key in result:
            raise ValueError(f"Duplicate statutory context {key}")
        if not raw["title"].strip():
            raise ValueError("Missing statute title")
        result[key] = StatuteContext(
            country, year, raw["title"].strip(),
            enacted.isoformat(), source_url, raw["tax_period_text"].strip(),
            raw["evidence_scope"],
        )
    return result


def audit_annual_timing(
    author: dict[tuple[str, int], int],
    observed: set[tuple[str, int]],
    context: dict[tuple[str, int], StatuteContext],
) -> tuple[list[FlaggedYear], list[ConsecutiveRun], dict[str, Any]]:
    """Classify consecutive *flag-year sequences* without merging laws.

    A consecutive sequence is a descriptive property of the author flags.
    Neither it nor a statute reference establishes identical legislation.
    """
    if not author or not observed.issubset(author):
        raise ValueError("Missing author panel or unexpected joined outcome keys")
    if any(value not in (0, 1) for value in author.values()):
        raise ValueError("Nonbinary author event panel")
    flagged = {key for key, value in author.items() if value == 1}
    if not flagged or not set(context).issubset(flagged):
        raise ValueError("Invalid statute context or empty author event pool")
    by_country: dict[str, list[int]] = defaultdict(list)
    for country, year in sorted(flagged):
        by_country[country].append(year)

    runs: list[ConsecutiveRun] = []
    flags: list[FlaggedYear] = []
    for country, years in sorted(by_country.items()):
        grouped: list[list[int]] = []
        for year in sorted(years):
            if grouped and grouped[-1][-1] == year - 1:
                grouped[-1].append(year)
            else:
                grouped.append([year])
        for group in grouped:
            first, last = group[0], group[-1]
            runs.append(
                ConsecutiveRun(
                    country, first, last, len(group),
                    sum((country, year) in observed for year in group),
                )
            )
            for position, year in enumerate(group, start=1):
                source = context.get((country, year))
                flags.append(
                    FlaggedYear(
                        country,
                        year,
                        first,
                        last,
                        len(group),
                        position,
                        (country, year) in observed,
                        source.title if source else "",
                        source.enacted_on if source else "",
                        source.evidence_scope if source else "no_statute_checked",
                    )
                )
    if sum(run.n_flagged_years for run in runs) != len(flags):
        raise ValueError("Run membership does not partition author flag years")

    with_outcome = sum(row.wid_outcome_observed for row in flags)
    report: dict[str, Any] = {
        "status": "author_flag_years_and_legal_timing_context_only",
        "n_author_country_years": len(author),
        "n_author_flag_years": len(flagged),
        "n_distinct_countries_with_flags": len(by_country),
        "n_descriptive_consecutive_runs": len(runs),
        "n_multiyear_runs": sum(run.n_flagged_years > 1 for run in runs),
        "n_flag_years_inside_multiyear_runs": sum(
            run.n_flagged_years for run in runs if run.n_flagged_years > 1
        ),
        "n_flag_years_after_previous_flag_same_country": sum(
            run.n_flagged_years - 1 for run in runs
        ),
        "n_joined_outcome_country_years": len(observed),
        "n_flag_years_with_joined_wid_outcome": with_outcome,
        "n_flag_years_missing_joined_wid_outcome": len(flags) - with_outcome,
        "missing_flag_year_ids": [
            f"{country}-{year}"
            for country, year in sorted(flagged - observed)
        ],
        "n_flags_with_primary_statute_context": len(context),
        "n_primary_statutes_distinct_by_title_and_enactment": len({
            (row.country, row.title, row.enacted_on)
            for row in context.values()
        }),
        "n_flags_with_verified_implementation_date": 0,
        "treatment_semantics": (
            "The binary 2-SD variable denotes author-coded annual changes "
            "in a latent composite of taxes on top incomes, capital and assets; "
            "an author-year flag is not a unique statute or an enactment date."
        ),
        "run_semantics": (
            "Maximal contiguous flagged country-years: not evidence of one "
            "or multiple laws and not an alternative treatment definition."
        ),
        "source_page": AUTHOR_PAGE,
        "outcome_timing": (
            "WID top 1% pre-tax income share is annual; the author policy-year "
            "label alone cannot determine anticipation, exposure within "
            "the outcome year, or the tax-year/calendar-year mismatch."
        ),
        "statute_scope": (
            "Primary sources verify named acts/enactment dates only. The "
            "one-to-one match to author composite flags, exact policy "
            "effective dates, and treatment intensity remain unverified."
        ),
        "inference": "none; no ATT, timing re-alignment, causal design or p-values",
    }
    return flags, runs, report


def save_timing_audit(
    destination: Path,
    flags: list[FlaggedYear],
    runs: list[ConsecutiveRun],
    audit: dict[str, Any],
    *,
    author_path: Path,
    joined_path: Path,
    ledger_path: Path,
    prior_reference: Path | None = None,
) -> None:
    """Write source-hashed reproducible outputs, failing on stale risk sets."""
    if not flags or not runs:
        raise ValueError("Refusing empty annual-policy audit")
    outcome_digest = hashlib.sha256(joined_path.read_bytes()).hexdigest()
    output = {
        **audit,
        "author_source_sha256": hashlib.sha256(author_path.read_bytes()).hexdigest(),
        "joined_panel_sha256": outcome_digest,
        "statute_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    }
    if prior_reference is not None:
        reference = json.loads(prior_reference.read_text(encoding="utf-8"))
        if reference.get("joined_panel_sha256") == outcome_digest:
            if (
                audit["n_flag_years_with_joined_wid_outcome"]
                != reference["n_original_observed_reform_events"]
            ):
                raise ValueError("Same-vintage reform count differs from previous audit")
            output["frozen_reference_status"] = "same_source_validated"
        else:
            output["frozen_reference_status"] = "source_vintage_changed"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, records, kind in (
        ("author_flag_years.csv", flags, FlaggedYear),
        ("author_consecutive_flag_runs.csv", runs, ConsecutiveRun),
    ):
        with (destination / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kind.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(item) for item in records)
    (destination / "author_timing_audit.json").write_text(
        json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Audit official author flags without moving or collapsing their years."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-csv", type=Path, required=True)
    parser.add_argument("--joined-panel", type=Path, required=True)
    parser.add_argument("--statute-ledger", type=Path, required=True)
    parser.add_argument("--frozen-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_author_flags(args.author_csv)
    observed = load_joined_outcomes(args.joined_panel, panel)
    context = load_statute_contexts(args.statute_ledger, panel)
    flags, runs, report = audit_annual_timing(panel, observed, context)
    save_timing_audit(
        args.output_dir, flags, runs, report,
        author_path=args.author_csv, joined_path=args.joined_panel,
        ledger_path=args.statute_ledger, prior_reference=args.frozen_reference,
    )
    print(
        f"{report['n_author_flag_years']} author flag years, "
        f"{report['n_descriptive_consecutive_runs']} contiguous runs; "
        f"{report['n_flag_years_missing_joined_wid_outcome']} WID gap(s)"
    )
    print("WARNING: coded years are not statute enactment or implementation dates")


if __name__ == "__main__":
    main()
