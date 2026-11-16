"""Reconcile all 17 archived INE 0012340 Dim2 slices with frozen AMP counts.

Each municipality response is a separately captured source view. Matching
these views establishes count reproducibility and reciprocal consistency,
not the direction of commuting, residential moves or a housing causal effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from extract_commuting import OUTPUT as FROZEN_PAIRS, municipality_codes_from_panel
from ine_commuting_partition_reconciliation import parse_dim2_slice, validate_manifest
from ine_commuting_reconciliation import read_frozen_pairs

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = ROOT / "data"

# These are immutable GitHub Actions captures, not reconstructed source data.
# Each archive contains the original manifest and raw JSON response bytes.
SOURCE_RUN_BY_DICO: dict[str, str] = {
    "0104": "43404869490",
    "0107": "43404869490",
    "0109": "43412012492",
    "0113": "43414048813",
    "0116": "43419199498",
    "0119": "43422198121",
    "1304": "43513828242",
    "1306": "43581552155",
    "1308": "43581552155",
    "1310": "43582490308",
    "1312": "43594835001",
    "1313": "43594835001",
    "1314": "43594835001",
    "1315": "43650564745",
    "1316": "43650564745",
    "1317": "43650564745",
    "1318": "43887818291",
}


def compare_source_views(
    observations: Mapping[str, Mapping[str, int]],
    frozen: Mapping[tuple[str, str], int],
    names: Mapping[str, str],
) -> dict[str, Any]:
    """Compare all 272 source views and 136 independently reciprocal pairs."""
    codes = sorted(names)
    if len(codes) != 17 or set(observations) != set(codes):
        raise ValueError("Exactly 17 different AMP municipality source slices required")
    if any(set(observations[code]) != set(codes) for code in codes):
        raise ValueError("Every source slice must contain all 17 AMP municipalities")
    if any(observations[code][code] != 0 for code in codes):
        raise ValueError("Every official source slice must have a zero diagonal")
    if any(
        type(value) is not int or value < 0
        for source in observations.values()
        for value in source.values()
    ):
        raise ValueError("Official source counts must be nonnegative integers")

    expected_pairs = set(combinations(codes, 2))
    if set(frozen) != expected_pairs:
        raise ValueError("Frozen third-party data must contain exactly 136 AMP pairs")

    comparisons: list[dict[str, Any]] = []
    for first, second in sorted(expected_pairs):
        # Both official directions were captured from different HTTP responses.
        forward, backward = observations[first][second], observations[second][first]
        expected = frozen[first, second]
        comparisons.append({
            "municipality_a_code": first,
            "municipality_a": names[first],
            "municipality_b_code": second,
            "municipality_b": names[second],
            "official_a_to_b_view": forward,
            "official_b_to_a_view": backward,
            "frozen_third_party_interactions": expected,
            "source_views_symmetric": forward == backward,
            "both_source_views_match_frozen": forward == backward == expected,
        })

    differences = [row for row in comparisons if not row["both_source_views_match_frozen"]]
    asymmetric = [row for row in comparisons if not row["source_views_symmetric"]]
    return {
        "status": "match" if not differences else "discrepancies_detected",
        "official_offdiagonal_source_views": len(codes) * (len(codes) - 1),
        "frozen_undirected_pairs": len(expected_pairs),
        "reciprocal_pairs_checked": len(comparisons),
        "reciprocal_pairs_symmetric": len(comparisons) - len(asymmetric),
        "both_source_views_matching_frozen": len(comparisons) - len(differences),
        "reciprocal_discrepancies": asymmetric,
        "frozen_comparison_discrepancies": differences,
        "pair_comparisons": comparisons,
    }


def build_report(data_directory: Path = DEFAULT_DATA_DIRECTORY) -> dict[str, Any]:
    """Validate source bytes/receipts, then assemble a reproducible audit."""
    names = municipality_codes_from_panel()
    if len(names) != 17 or set(SOURCE_RUN_BY_DICO) != set(names):
        raise ValueError("The immutable source registry must cover exactly the AMP panel")

    observations: dict[str, dict[str, int]] = {}
    sources: list[dict[str, Any]] = []
    for code, run_id in sorted(SOURCE_RUN_BY_DICO.items()):
        archive_name = f"ine_0012340_pr_run_{run_id}"
        source_name = f"ine_0012340_dim2_{code}_candidate.json"
        archive = data_directory / archive_name
        source_bytes = (archive / source_name).read_bytes()
        manifest_bytes = (archive / "partition_manifest.json").read_bytes()
        # Fail closed if either the original capture receipt or source shape fails.
        validate_manifest(manifest_bytes, source_bytes, code, names)
        counts, records = parse_dim2_slice(source_bytes, code, names)
        if records != 347:
            raise ValueError(f"Official source slice {code} has unexpected record count")
        observations[code] = counts
        sources.append({
            "municipality_code": code,
            "municipality": names[code],
            "github_actions_run_id": run_id,
            "source_file": f"data/{archive_name}/{source_name}",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_bytes": len(source_bytes),
            "capture_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_records": records,
            "amp_counterparts": len(counts),
        })

    result = compare_source_views(observations, read_frozen_pairs(), names)
    return {
        "evidence_class": "17_independently_captured_ine_dim2_municipality_slices",
        "indicator_id": "0012340",
        "reference_period": "2021",
        "scope": "2021 Census municipal work/study interactions in 17 AMP municipalities",
        "source_municipality_slices": len(sources),
        "source_records_per_slice": 347,
        "frozen_third_party_csv_sha256": hashlib.sha256(FROZEN_PAIRS.read_bytes()).hexdigest(),
        "independent_source_authentication_by_offline_script": False,
        "residential_migration_or_displacement_evidence": False,
        "housing_cascade_causality_established": False,
        "single_official_full_matrix_download_claimed": False,
        "source_slices": sources,
        **result,
    }


def run(output_json: Path, data_directory: Path = DEFAULT_DATA_DIRECTORY) -> dict[str, Any]:
    """Write the JSON audit, including row-level mismatches when present."""
    report = build_report(data_directory)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the source comparison offline and fail if any pair disagrees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    args = parser.parse_args(argv)
    report = run(args.output_json, args.data_dir)
    print(f"INE 0012340: {report['both_source_views_matching_frozen']}/136 "
          "AMP work/study pairs match in both source views")
    if report["status"] != "match":
        parser.exit(1, "INE source-view discrepancies detected.\n")


if __name__ == "__main__":
    main()
