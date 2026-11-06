"""Audit Forbes March 2026 published country counts against frozen aggregates.

The independent March source publishes only the top-ten country counts used
here. A matching worldwide total or top-ten subset NEVER proves that the
November person-list endpoint reproduces the entire March 2026 vintage.

Run::

    python forbes_march_vintage.py --out-dir results/forbes_march_vintage

No new poverty regression, residence inference or person-level extraction is
authorized by this source comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, TypedDict

from forbes_iso3_review import build_ledger, read_csv


SOURCE: Path = Path(__file__).resolve().parent
MARCH_TOP10: Path = SOURCE / "references" / "forbes_2026_march_top10_official.csv"
MARCH_TOP10_SHA256: str = (
    "e16f905c860c76ff261c8b0be5876c720f27f812c140acb5560f6f598b227572"
)
FROZEN_AUDIT: Path = (
    SOURCE / "evidence" / "forbes-2026-run-42099896845" / "audit.json"
)
MARCH_SOURCE_URL: str = (
    "https://www.forbes.com/sites/samellefson/2026/03/11/"
    "the-countries-with-the-most-billionaires/"
)
MARCH_RELEASE_URL: str = (
    "https://www.forbes.com/sites/pr/2026/03/10/"
    "forbes-40th-annual-worlds-billionaires-list-elon-musk-is-worlds-richest-person-ever-recorded/"
)
MARCH_PUBLICATION_DATE: str = "2026-03-11"
MARCH_WEALTH_DATE: str = "2026-03-01"
MARCH_GLOBAL_TOTAL: int = 3_428
MARCH_AREAS: int = 80


@dataclass(frozen=True)
class MarchEntry:
    """A factual, source-linked published March country count."""

    rank: int
    label: str
    iso3: str
    count: int


class ComparisonRow(TypedDict):
    """Preserve unknown March counts as empty rather than invented zeros."""

    primary_label: str
    iso3: str
    primary_citizenship_count: int
    secondary_citizenship_count: str
    primary_minus_secondary: str
    march_published_count: str
    primary_minus_march: str
    secondary_minus_march: str
    march_comparison_status: str
    secondary_match_type: str
    territory_review: str
    approved_for_regression: str


def march_top_ten() -> dict[str, MarchEntry]:
    """Validate exactly ten ranked facts from Forbes' March 11 publication."""
    records = read_csv(MARCH_TOP10, MARCH_TOP10_SHA256)
    expected = {
        "rank", "forbes_label", "iso3", "published_billionaires",
        "net_worth_as_of_date", "publication_date", "source_url",
    }
    result: dict[str, MarchEntry] = {}
    ranks: set[int] = set()
    codes: set[str] = set()
    for record in records:
        if not expected.issubset(record):
            raise ValueError("Forbes March ten-country source is missing columns")
        label = record["forbes_label"]
        code = record["iso3"]
        rank_text = record["rank"]
        count_text = record["published_billionaires"]
        if (
            not label or label in result or code in codes
            or not re.fullmatch(r"[A-Z]{3}", code)
            or not re.fullmatch(r"[1-9][0-9]*", rank_text)
            or not re.fullmatch(r"[1-9][0-9]*", count_text)
            or record["net_worth_as_of_date"] != MARCH_WEALTH_DATE
            or record["publication_date"] != MARCH_PUBLICATION_DATE
            or record["source_url"] != MARCH_SOURCE_URL
        ):
            raise ValueError("Forbes March reference contains an invalid entry")
        rank = int(rank_text)
        ranks.add(rank)
        codes.add(code)
        result[label] = MarchEntry(rank, label, code, int(count_text))
    if len(result) != 10 or ranks != set(range(1, 11)):
        raise ValueError("Forbes March top-ten source must cover ranks 1 to 10")
    if sum(item.count for item in result.values()) != 2_494:
        raise ValueError("Published Forbes March top-ten count changed unexpectedly")
    return result


def audit_march_vintage() -> tuple[dict[str, Any], list[ComparisonRow]]:
    """Compare independently published March counts with two frozen series.

    The 70 other country counts remain unverified against a historical Forbes
    release, irrespective of whether primary and secondary agree with each
    other. Never interpret non-publication as zero.
    """
    march = march_top_ten()
    rows = build_ledger()  # Validates three source-byte SHA-256 digests.
    audit = json.loads(FROZEN_AUDIT.read_text(encoding="utf-8"))
    source_sha = audit.get("source_sha256")
    if audit["schema"]["n_persons"] != MARCH_GLOBAL_TOTAL:
        raise ValueError("Live source does not contain the published global total")
    if audit["coverage"]["citizenship"]["distinct_labels"] != MARCH_AREAS:
        raise ValueError("Unexpected live source country-label coverage")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("Live Forbes source digest is missing or malformed")

    output: list[ComparisonRow] = []
    conflicts: list[dict[str, Any]] = []
    for item in rows:
        label: str = item["primary_label"]
        primary: int = item["primary_count"]
        secondary_str: str = item["secondary_count"]
        published: MarchEntry | None = march.get(label)
        if published is not None and published.iso3 != item["iso3_candidate"]:
            raise ValueError(f"Published Forbes/UN code conflict for {label}")
        if published is not None and primary != published.count:
            conflicts.append({
                "country": label,
                "iso3": published.iso3,
                "march_published": published.count,
                "november_api": primary,
                "difference": primary - published.count,
            })
        output.append({
            "primary_label": label,
            "iso3": item["iso3_candidate"],
            "primary_citizenship_count": primary,
            "secondary_citizenship_count": secondary_str,
            "primary_minus_secondary": item["difference_from_secondary"],
            "march_published_count": str(published.count) if published else "",
            "primary_minus_march": str(primary - published.count) if published else "",
            "secondary_minus_march": (
                str(int(secondary_str) - published.count)
                if published and secondary_str else ""
            ),
            "march_comparison_status": (
                "november_api_conflicts_with_march" if published and primary != published.count
                else "consistent_on_this_country_only" if published
                else "not_independently_checked_against_march"
            ),
            "secondary_match_type": item["match_type"],
            "territory_review": item["territory_review"],
            "approved_for_regression": "false",
        })
    if {item["primary_label"] for item in output if item["march_published_count"]} != set(march):
        raise ValueError("Published March country observations are not all aligned")
    comparable = [item for item in output if item["secondary_citizenship_count"]]
    missing = [item for item in output if not item["secondary_citizenship_count"]]
    changed = [item for item in comparable if int(item["primary_minus_secondary"]) != 0]
    primary_total = sum(item["primary_citizenship_count"] for item in output)
    secondary_total = sum(int(item["secondary_citizenship_count"]) for item in comparable)
    comparable_delta = sum(int(item["primary_minus_secondary"]) for item in comparable)
    missing_total = sum(item["primary_citizenship_count"] for item in missing)
    secondary_published_matches = sum(
        item["secondary_minus_march"] == "0"
        for item in output if item["march_published_count"]
    )
    result: dict[str, Any] = {
        "primary_source_sha256": source_sha,
        "primary_retrieved_at_utc": audit["retrieved_at_utc"],
        "march_published_source_url": MARCH_SOURCE_URL,
        "march_global_release_url": MARCH_RELEASE_URL,
        "march_top10_transcription_sha256": hashlib.sha256(MARCH_TOP10.read_bytes()).hexdigest(),
        "march_wealth_as_of_date": MARCH_WEALTH_DATE,
        "march_publication_date": MARCH_PUBLICATION_DATE,
        "global_people_march_published": MARCH_GLOBAL_TOTAL,
        "primary_people_november_api": primary_total,
        "primary_distinct_labels": len(output),
        "published_march_top10_labels": len(march),
        "published_march_top10_count_total": sum(item.count for item in march.values()),
        "march_top10_primary_matches": len(march) - len(conflicts),
        "march_top10_primary_conflicts": sorted(conflicts, key=lambda row: row["country"]),
        "march_top10_secondary_matches": secondary_published_matches,
        "secondary_people_total": secondary_total,
        "primary_minus_secondary_total": primary_total - secondary_total,
        "comparable_secondary_labels": len(comparable),
        "comparable_secondary_disagreements": len(changed),
        "comparable_primary_minus_secondary": comparable_delta,
        "comparable_absolute_difference": sum(
            abs(int(item["primary_minus_secondary"])) for item in comparable
        ),
        "primary_labels_absent_from_secondary": [item["primary_label"] for item in missing],
        "people_absent_from_secondary": missing_total,
        "historic_full_80_country_vintage_verified": False,
        "november_api_exact_march_distribution": False,
        "approved_for_publication_or_regression": False,
        "interpretation": (
            "The published March top ten contradicts November API citizenship counts "
            "for three countries. Neither matching global totals nor matching "
            "the other seven top-ten countries establishes a historical freeze. "
            "The secondary table matches the published March top ten only; "
            "the remaining 70 countries are not independently verified."
        ),
    }
    if (
        result["primary_minus_secondary_total"] != comparable_delta + missing_total
        or len(conflicts) != 3 or secondary_published_matches != 10
        or len(changed) != 35 or comparable_delta != 64 or missing_total != 5
    ):
        raise ValueError("Frozen Forbes evidence no longer satisfies recorded audit facts")
    return result, output


def write_report(output_dir: Path) -> dict[str, Any]:
    """Export an offline report without rewriting an existing audit."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evidence: {output_dir}")
    summary, comparisons = audit_march_vintage()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "country_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ComparisonRow.__annotations__))
        writer.writeheader()
        writer.writerows(comparisons)
    return summary


def main() -> None:
    """Validate references and write a non-publication diagnostic artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_report(args.out_dir)
    print(
        f"March top ten: {summary['march_top10_primary_matches']} concordant, "
        f"{len(summary['march_top10_primary_conflicts'])} conflict; "
        "historic full-vintage validation remains blocked."
    )


if __name__ == "__main__":
    main()
