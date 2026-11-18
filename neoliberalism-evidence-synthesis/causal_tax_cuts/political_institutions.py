"""Audit DPI2023 political-institutional coverage before tax-cut reforms.

The Inter-American Development Bank codes executive economic orientation,
checks on executive power and electoral competitiveness at country-year
level. This module preserves categorical ideology and missing-value types,
then checks availability *only at t-5 through t-1* for the existing tax-cut
cohort. It does not use political variables to match or estimate effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.donor_support import audit_future_donor_exposure
from causal_tax_cuts.macro_matching import MacroValues, read_macro_controls
from causal_tax_cuts.prepare import PAPER_COUNTRIES

DPI_CSV_URL = (
    "https://data.iadb.org/files/download/30f45628-8608-4c0c-ad55-bc4c870dd62d"
)
DPI_DATASET_URL = (
    "https://data.iadb.org/dataset/the-database-of-political-institutions-dpi-2023"
)
DPI_CODEBOOK_URL = (
    "https://data.iadb.org/files/download/7d23d94b-dfec-4c84-b50e-174320432521"
)
DPI_DOI = "10.60966/8tx1xs4i"
MAX_SOURCE_BYTES = 25_000_000
SOURCE_START_YEAR = 1975
STUDY_START_YEAR = 1965
STUDY_END_YEAR = 2015
PRE_YEARS = 5
HORIZON = 5
REQUIRED_FIELDS = frozenset({"countryname", "year", "execrlc", "checks", "eiec"})

# Exact names verified in the published CSV; not fuzzy country matching.
# "GDR" is a separate original country and is never mapped to DEU.
DPI_TO_ISO3: dict[str, str] = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Switzerland": "CHE",
    "FRG/Germany": "DEU",
    "Denmark": "DNK",
    "Finland": "FIN",
    "France": "FRA",
    "UK": "GBR",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Japan": "JPN",
    "Netherlands": "NLD",
    "Norway": "NOR",
    "New Zealand": "NZL",
    "Sweden": "SWE",
    "USA": "USA",
}

if set(DPI_TO_ISO3.values()) != PAPER_COUNTRIES:
    raise ValueError("DPI mapping must cover the exact 18-country paper sample")


@dataclass(frozen=True, slots=True)
class PoliticalYear:
    """One DPI country-year, with separate ideology missingness categories."""

    country: str
    year: int
    executive_orientation: str | None
    orientation_status: str
    checks: float | None
    electoral_competitiveness: float | None


@dataclass(frozen=True, slots=True)
class PoliticalReformCoverage:
    """Pre-five-year completeness for one previously eligible reform."""

    country: str
    reform_year: int
    n_treated_pre5_orientation: int
    n_treated_pre5_checks: int
    n_treated_pre5_electoral: int
    treated_all_three_pre5_complete: bool
    n_pre_censor_donors: int
    n_pre_censor_donors_all_three_complete: int
    n_horizon_five_donors: int
    n_horizon_five_donors_all_three_complete: int


@dataclass(frozen=True, slots=True)
class PoliticalDonorCoverage:
    """One previously eligible donor's political-history coverage."""

    country: str
    reform_year: int
    donor: str
    first_censored_horizon: int | None
    n_donor_pre5_orientation: int
    n_donor_pre5_checks: int
    n_donor_pre5_electoral: int
    donor_all_three_pre5_complete: bool


def _parse_orientation(raw: str) -> tuple[str | None, str]:
    """Keep unknown, NA and observed executive party classes distinct."""
    value = raw.strip().lower()
    if value in {"right", "1", "1.0"}:
        return "right", "known"
    if value in {"center", "centre", "2", "2.0"}:
        return "center", "known"
    if value in {"left", "3", "3.0"}:
        return "left", "known"
    if value in {"0", "0.0"}:
        return None, "unknown"
    if value in {"-999", "-999.0", "na", "n/a"}:
        return None, "not_applicable"
    if not value:
        return None, "missing"
    raise ValueError(f"Unknown DPI executive-orientation coding: {raw!r}")


def _parse_numeric(raw: str, *, field: str) -> float | None:
    """Interpret DPI missing sentinels; reject impossible non-finite data."""
    value = raw.strip().lower()
    if value in {"", "na", "n/a", "-999", "-999.0"}:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid DPI {field} value: {raw!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite DPI {field}: {raw!r}")
    if field == "checks" and not 1 <= number <= 50:
        raise ValueError(f"Impossible DPI checks value: {number}")
    if field == "eiec" and not 1 <= number <= 7:
        raise ValueError(f"Impossible DPI eiec value: {number}")
    return number


def parse_dpi(
    path: Path, *, require_all_countries: bool = True
) -> tuple[dict[Key, PoliticalYear], dict[str, object]]:
    """Validate the actual IDB CSV and extract only the paper sample.

    This function does not fill missing years, infer other countries' codes,
    merge an East German series into FRG/Germany, or score political ideology.
    """
    selected: dict[Key, PoliticalYear] = {}
    n_source_rows = 0
    sample_countries: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected DPI2023 CSV fields: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            n_source_rows += 1
            label = row.get("countryname")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"DPI row {line_number} has no country")
            if label not in DPI_TO_ISO3:
                continue
            source_year = row.get("year")
            if not isinstance(source_year, str):
                raise TypeError(f"Missing DPI date at row {line_number}")
            try:
                parsed_date = date.fromisoformat(source_year)
            except ValueError as exc:
                raise ValueError(f"Invalid DPI year {source_year!r}") from exc
            if parsed_date.month != 1 or parsed_date.day != 1:
                raise ValueError(f"DPI date is not a January-1 snapshot: {source_year}")
            year = parsed_date.year
            if not STUDY_START_YEAR <= year <= STUDY_END_YEAR:
                continue
            if year < SOURCE_START_YEAR:
                raise ValueError("Unexpected pre-1975 DPI country-year")
            country = DPI_TO_ISO3[label]
            key = (country, year)
            if key in selected:
                raise ValueError(f"Duplicate DPI country-year for {country} {year}")
            orientation, status = _parse_orientation(row["execrlc"] or "")
            selected[key] = PoliticalYear(
                country,
                year,
                orientation,
                status,
                _parse_numeric(row["checks"] or "", field="checks"),
                _parse_numeric(row["eiec"] or "", field="eiec"),
            )
            sample_countries.add(country)
    if not selected:
        raise ValueError("No paper-sample political country-years in DPI source")
    if require_all_countries and sample_countries != PAPER_COUNTRIES:
        raise ValueError(
            "Missing country mapping in DPI CSV: "
            f"{sorted(PAPER_COUNTRIES - sample_countries)}"
        )
    return selected, {
        "source_rows_read": n_source_rows,
        "selected_country_years_1975_to_2015": len(selected),
        "selected_countries": sorted(sample_countries),
        "observed_fields": sorted(REQUIRED_FIELDS),
        "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_csv_bytes": path.stat().st_size,
    }


def download_dpi(path: Path, *, timeout_seconds: int = 90) -> None:
    """Download the official IDB CSV with size and content-type limits."""
    request = Request(DPI_CSV_URL, headers={"User-Agent": "tax-study-source-audit/1.0"})
    total = 0
    with urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "csv" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Official DPI URL returned non-CSV: {content_type}")
        with path.open("wb") as handle:
            while chunk := response.read(128 * 1024):
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError("Official DPI CSV exceeds the maximum source size")
                handle.write(chunk)
    if total == 0:
        raise ValueError("Downloaded empty DPI CSV")


def _coverage(
    country: str, reform_year: int, years: dict[Key, PoliticalYear]
) -> tuple[int, int, int]:
    """Count *observed* political variables in t-5 through t-1 only."""
    prehistory = [
        years.get((country, year))
        for year in range(reform_year - PRE_YEARS, reform_year)
    ]
    return (
        sum(
            row is not None and row.executive_orientation is not None
            for row in prehistory
        ),
        sum(row is not None and row.checks is not None for row in prehistory),
        sum(
            row is not None and row.electoral_competitiveness is not None
            for row in prehistory
        ),
    )


def audit_political_coverage(
    panel: dict[Key, Observation],
    macro: dict[Key, MacroValues],
    political: dict[Key, PoliticalYear],
) -> tuple[
    list[PoliticalReformCoverage],
    list[PoliticalDonorCoverage],
    dict[str, object],
]:
    """Hold WID/WDI eligibility and donor reform censorship completely fixed."""
    if not panel or set(panel) != set(macro):
        raise ValueError("WID/WDI keys must be identical")
    reforms, candidates, _, prior = audit_future_donor_exposure(panel, macro)
    matched = sorted(
        (row for row in reforms if row.status == "baseline_donors_available"),
        key=lambda row: (row.reform_year, row.country),
    )
    if not matched:
        raise ValueError("No eligible reform episodes for political audit")
    event_rows: list[PoliticalReformCoverage] = []
    donor_rows: list[PoliticalDonorCoverage] = []
    for event in matched:
        counts = _coverage(event.country, event.reform_year, political)
        event_donors = sorted(
            (
                donor
                for donor in candidates
                if (donor.country, donor.reform_year)
                == (event.country, event.reform_year)
            ),
            key=lambda donor: donor.donor,
        )
        if len(event_donors) != event.n_pre_censor_donors:
            raise ValueError("Political audit changed the pre-censor donor set")
        for donor in event_donors:
            donor_counts = _coverage(donor.donor, event.reform_year, political)
            donor_rows.append(
                PoliticalDonorCoverage(
                    event.country,
                    event.reform_year,
                    donor.donor,
                    donor.first_censored_horizon,
                    *donor_counts,
                    all(count == PRE_YEARS for count in donor_counts),
                )
            )
        all_donors = [
            row
            for row in donor_rows
            if (row.country, row.reform_year) == (event.country, event.reform_year)
        ]
        h5_donors = [
            row
            for row in all_donors
            if row.first_censored_horizon is None
            or row.first_censored_horizon > HORIZON
        ]
        if len(h5_donors) != event.n_horizon_five_donors:
            raise ValueError("Political audit changed horizon-five donor censoring")
        event_rows.append(
            PoliticalReformCoverage(
                event.country,
                event.reform_year,
                *counts,
                all(count == PRE_YEARS for count in counts),
                len(all_donors),
                sum(row.donor_all_three_pre5_complete for row in all_donors),
                len(h5_donors),
                sum(row.donor_all_three_pre5_complete for row in h5_donors),
            )
        )

    potential = len(PAPER_COUNTRIES) * (STUDY_END_YEAR - STUDY_START_YEAR + 1)
    source_years = len(PAPER_COUNTRIES) * (STUDY_END_YEAR - SOURCE_START_YEAR + 1)
    status_counts = Counter(row.orientation_status for row in political.values())
    h5_complete = sum(
        row.n_horizon_five_donors_all_three_complete for row in event_rows
    )
    report: dict[str, object] = {
        "status": "pre_tax_cut_political_institutional_coverage_only",
        "source_dataset": "IDB Database of Political Institutions DPI2023",
        "source_url": DPI_CSV_URL,
        "dataset_url": DPI_DATASET_URL,
        "codebook_url": DPI_CODEBOOK_URL,
        "doi": DPI_DOI,
        "license": "CC BY 4.0",
        "study_years": [STUDY_START_YEAR, STUDY_END_YEAR],
        "source_start_year": SOURCE_START_YEAR,
        "n_potential_country_years": potential,
        "n_potential_country_years_before_dpi_start": potential - source_years,
        "n_selected_source_country_years": len(political),
        "n_with_known_orientation": sum(
            row.executive_orientation is not None for row in political.values()
        ),
        "n_with_checks": sum(row.checks is not None for row in political.values()),
        "n_with_electoral_competitiveness": sum(
            row.electoral_competitiveness is not None for row in political.values()
        ),
        "orientation_status_counts": dict(sorted(status_counts.items())),
        "n_reforms_original_observed": prior["n_original_reform_events"],
        "n_reforms_baseline_eligible": len(matched),
        "n_reforms_full_pre5_ideology": sum(
            row.n_treated_pre5_orientation == PRE_YEARS for row in event_rows
        ),
        "n_reforms_full_pre5_checks": sum(
            row.n_treated_pre5_checks == PRE_YEARS for row in event_rows
        ),
        "n_reforms_full_pre5_electoral": sum(
            row.n_treated_pre5_electoral == PRE_YEARS for row in event_rows
        ),
        "n_reforms_full_pre5_all_three": sum(
            row.treated_all_three_pre5_complete for row in event_rows
        ),
        "n_pre_censor_donor_pairs": len(candidates),
        "n_pre_censor_pairs_full_pre5_all_three": sum(
            row.donor_all_three_pre5_complete for row in donor_rows
        ),
        "n_horizon_five_donor_pairs": sum(
            row.n_horizon_five_donors for row in event_rows
        ),
        "n_horizon_five_pairs_full_pre5_all_three": h5_complete,
        "n_horizon_five_reforms_with_both_treated_and_donor_coverage": sum(
            row.treated_all_three_pre5_complete
            and row.n_horizon_five_donors_all_three_complete > 0
            for row in event_rows
        ),
        "reconciled_to_original_donor_risk_sets": (
            len(candidates) == prior["n_pre_censor_donor_event_pairs"]
            and sum(row.n_horizon_five_donors for row in event_rows)
            == prior["by_horizon"][-1]["n_donor_reform_pairs_at_risk"]
        ),
        "time_order": (
            "DPI January-1 snapshots, using only country years t-5 through t-1"
        ),
        "orientation": "right/center/left are nominal categories, not a numeric score",
        "missingness": (
            "DPI unknown 0, not-applicable -999/NA and empty are distinct; "
            "none is a usable ideology category"
        ),
        "inference": (
            "none; coverage and provenance only, no re-matching or causal effects"
        ),
        "caveats": [
            "DPI begins in 1975; full five-year history is impossible before 1980.",
            (
                "Executive party orientation is not implemented fiscal policy "
                "or a tax-cut date."
            ),
            (
                "Swiss collegial executive authority has no single coded "
                "party orientation."
            ),
            "DPI values may be revised; compare original-source SHA-256 across runs.",
            (
                "Covariate coverage alone does not establish exchangeability "
                "or positivity."
            ),
        ],
    }
    if not report["reconciled_to_original_donor_risk_sets"]:
        raise ValueError("Source-dependent political audit changed donor risk sets")
    return event_rows, donor_rows, report


def save_political_audit(
    output_dir: Path,
    political: dict[Key, PoliticalYear],
    reform_rows: list[PoliticalReformCoverage],
    donor_rows: list[PoliticalDonorCoverage],
    report: dict[str, object],
    source_meta: dict[str, object],
    *,
    source_panel: Path,
    frozen_reference: Path | None = None,
) -> None:
    """Write limited CC-BY-derived controls and source-verifiable coverage."""
    data = {**report, **source_meta}
    data["joined_panel_sha256"] = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    # A source-vintage SHA and fixed algorithm make this manifest reproducible.
    if frozen_reference is not None:
        reference = json.loads(frozen_reference.read_text(encoding="utf-8"))
        if data["joined_panel_sha256"] == reference.get("joined_panel_sha256"):
            if (
                data["n_reforms_baseline_eligible"]
                != reference["n_eligible_reforms_before_future_censoring"]
                or data["n_pre_censor_donor_pairs"]
                != reference["n_pre_censor_donor_event_pairs"]
                or data["n_horizon_five_donor_pairs"]
                != reference["by_horizon"][-1]["n_donor_reform_pairs_at_risk"]
            ):
                raise ValueError(
                    "Frozen matching risk sets differ at same source vintage"
                )
            data["frozen_reference_status"] = "same_source_validated"
        else:
            data["frozen_reference_status"] = "source_vintage_changed"

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dpi_political_controls.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [*PoliticalYear.__dataclass_fields__]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for country in sorted(PAPER_COUNTRIES):
            for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1):
                row = political.get((country, year))
                if row is None:
                    row = PoliticalYear(
                        country,
                        year,
                        None,
                        "out_of_source_period"
                        if year < SOURCE_START_YEAR
                        else "not_in_source",
                        None,
                        None,
                    )
                writer.writerow(asdict(row))
    for filename, rows, row_type in (
        ("political_reform_coverage.csv", reform_rows, PoliticalReformCoverage),
        ("political_donor_coverage.csv", donor_rows, PoliticalDonorCoverage),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(row_type.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output_dir / "political_coverage_audit.json").write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build a no-effects coverage artifact from the published IDB DPI2023."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--frozen-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    panel = load_panel(args.panel, "top1_share_pct", ())
    macro = read_macro_controls(args.panel, set(panel))
    if args.source_csv is None:
        with tempfile.TemporaryDirectory(prefix="dpi2023-source-") as temp:
            source_csv = Path(temp) / "DPI2023.csv"
            download_dpi(source_csv)
            political, source_meta = parse_dpi(source_csv)
    else:
        political, source_meta = parse_dpi(args.source_csv)
    events, donors, report = audit_political_coverage(panel, macro, political)
    save_political_audit(
        args.output_dir,
        political,
        events,
        donors,
        report,
        source_meta,
        source_panel=args.panel,
        frozen_reference=args.frozen_reference,
    )
    print(
        "DPI2023 complete three-control prehistory: "
        f"{report['n_reforms_full_pre5_all_three']} of "
        f"{report['n_reforms_baseline_eligible']} baseline reforms"
    )
    print("CAUTION: coverage only; no ideology score or causal estimation")


if __name__ == "__main__":
    main()
