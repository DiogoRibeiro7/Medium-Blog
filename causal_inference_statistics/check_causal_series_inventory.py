"""Check the complete archived-data inventory for the 20-part causal series.

Run from anywhere: ``python causal_inference_statistics/check_causal_series_inventory.py``.
This fast structural check does not rerun experiments or establish causal validity.
For numerical reproduction, also run ``python verify_causal_series.py --recompute``.
"""

from __future__ import annotations

import csv
from pathlib import Path


SERIES_ROOT: Path = Path(__file__).resolve().parent
PARTS: tuple[int, ...] = tuple(range(1, 21))
SUPPLEMENTAL_TABLES: tuple[str, ...] = (
    "part1/coverage_targets.csv",
    "part2/randomization_uncertainty.csv",
    "part3/matching_diagnostics.csv",
    "part4/weight_sensitivity.csv",
    "part5/population_coefficients.csv",
    "part7/weak_iv_inference_results.csv",
    "part8/rd_bias_corrected_results.csv",
    "part8/rd_falsification_results.csv",
    "part9/sc_preperiod_holdout_results.csv",
    "part13/validation_results.csv",
    "part15/msm_validation_results.csv",
    "part20/validation_results.csv",
)


def check_table(path: Path) -> list[str]:
    """Return schema/row errors for one archive without interpreting its statistics."""
    if not path.is_file():
        return [f"missing table: {path}"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or any(not field or not field.strip() for field in fields):
            return [f"invalid header: {path}"]
        if len(fields) != len(set(fields)):
            return [f"duplicate header: {path}"]
        rows = list(reader)
    if not rows:
        return [f"empty result table: {path}"]
    if any(None in row for row in rows):
        return [f"extra fields in data rows: {path}"]
    if any(value is None for row in rows for value in row.values()):
        return [f"missing fields in data rows: {path}"]
    return []


def audit_inventory(root: Path = SERIES_ROOT) -> tuple[int, list[str]]:
    """Check 20 primary pairs, all discovered CSVs, and the supplemental manifest.

    Returns:
        Number of checked CSVs and descriptive errors (empty if consistent).
    """
    errors: list[str] = []
    expected: set[str] = set(SUPPLEMENTAL_TABLES)
    for number in PARTS:
        folder = root / f"part{number}"
        for filename in ("article.md", "experiment.py"):
            if not (folder / filename).is_file():
                errors.append(f"missing part {number} file: {filename}")
        expected.add(f"part{number}/results.csv")

    discovered = {file.relative_to(root).as_posix() for file in root.glob("part*/*.csv")}
    for unexpected in sorted(discovered - expected):
        errors.append(f"untracked supplemental table: {unexpected}")
    for table in sorted(expected):
        errors.extend(check_table(root / table))
    return len(expected), errors


def main() -> int:
    """Print the audit outcome and return a nonzero exit code on a problem."""
    count, problems = audit_inventory()
    print(f"Inventory: {len(PARTS)} parts, {count} archived CSV tables")
    for problem in problems:
        print(f"ERROR: {problem}")
    print("PASS: inventory is complete" if not problems else f"FAIL: {len(problems)} issue(s)")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
