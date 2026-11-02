"""Verify Monte Carlo result tables for causal-inference Parts 6–10.

Run from the repository root:

    python verify_identification_results.py
    python verify_identification_results.py --recompute

Quick validation needs only the standard library. A full rerun requires NumPy,
SciPy and scikit-learn and uses temporary directories to protect published CSVs.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
PARTS: tuple[int, ...] = (6, 7, 8, 9, 10)
KEYS: dict[int, tuple[str, ...]] = {
    6: ("scenario",),
    7: ("scenario", "method"),
    8: ("scenario", "bandwidth"),
    9: ("scenario", "method"),
    10: ("propensity_correct", "outcome_correct", "method"),
}
EXPECTED_ROWS: dict[int, int] = {6: 3, 7: 8, 8: 12, 9: 9, 10: 16}
TRUTHS: dict[int, float] = {6: -1.0, 7: 1.0, 8: -1.0, 9: -2.0, 10: 1.0}
TOLERANCE: float = 1e-8


def load_rows(path: Path) -> list[dict[str, str]]:
    """Read a nonempty, rectangular CSV with unique column names."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AssertionError(f"{path}: bad header")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise AssertionError(f"{path}: missing or malformed rows")
    return rows


def numeric_match(a: str, b: str) -> bool:
    """Compare floating values, including intentional NaN diagnostics."""
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def validate(part: int, rows: list[dict[str, str]]) -> None:
    """Check unique experimental conditions and applicable statistical identities."""
    if len(rows) != EXPECTED_ROWS[part]:
        raise AssertionError(f"Part {part}: wrong row count {len(rows)}")
    labels = [tuple(row[key] for key in KEYS[part]) for row in rows]
    if len(set(labels)) != len(labels):
        raise AssertionError(f"Part {part}: duplicate experimental conditions")
    for row in rows:
        truth = TRUTHS[part]
        if part == 6:
            mean_key, truth_key = "mean_did", None
            if int(row["replications"]) != 500 or int(row["n"]) != 1000:
                raise AssertionError("Part 6: incorrect simulation design")
            for field in ("coverage", "placebo_rejection"):
                if not 0 <= float(row[field]) <= 1:
                    raise AssertionError(f"Part 6: invalid probability {field}")
        elif part == 8:
            mean_key, truth_key = "mean_estimate", "true_effect"
            if int(row["replications"]) != 400 or int(row["sample_size"]) != 2000:
                raise AssertionError("Part 8: incorrect simulation design")
            if not 0 <= float(row["coverage"]) <= 1:
                raise AssertionError("Part 8: invalid coverage")
        else:
            mean_key, truth_key = "mean", "true_effect"
            count_key = "replicates" if part == 10 else "replications"
            expected_count = {7: 500, 9: 300, 10: 200}[part]
            if int(row[count_key]) != expected_count:
                raise AssertionError(f"Part {part}: incorrect replication count")
        if truth_key and not math.isclose(float(row[truth_key]), truth, abs_tol=TOLERANCE):
            raise AssertionError(f"Part {part}: incorrect stated estimand")
        if not math.isclose(
            float(row[mean_key]) - truth, float(row["bias"]), abs_tol=TOLERANCE
        ):
            raise AssertionError(f"Part {part}: bias identity violated")
        if part in (7, 9, 10):
            if float(row["rmse"]) + TOLERANCE < abs(float(row["bias"])):
                raise AssertionError(f"Part {part}: RMSE below absolute bias")
        if part == 10 and not (
            float(row["q05"]) <= float(row["mean"]) <= float(row["q95"])
        ):
            raise AssertionError("Part 10: invalid quantile ordering")


def verify(*, recompute: bool) -> None:
    """Check published tables and optionally rerun seeded experiments."""
    for part in PARTS:
        folder = ROOT / "causal_inference_statistics" / f"part{part}"
        published = load_rows(folder / "results.csv")
        validate(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                subprocess.run(
                    [sys.executable, str(script)], check=True,
                    capture_output=True, text=True, timeout=1200,
                )
                recreated = load_rows(script.with_name("results.csv"))
                if len(recreated) != len(published):
                    raise AssertionError(f"Part {part}: row count changed")
                for index, (old, new) in enumerate(
                    zip(published, recreated, strict=True), start=1
                ):
                    if old.keys() != new.keys():
                        raise AssertionError(f"Part {part}: schema changed")
                    for field, value in old.items():
                        if not numeric_match(value, new[field]):
                            raise AssertionError(
                                f"Part {part} row {index} {field}: "
                                f"committed={value!r}, regenerated={new[field]!r}"
                            )
        print(f"Part {part}: {'reproduced and validated' if recompute else 'validated'}")


def main() -> None:
    """Parse full-rerun opt-in and execute verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Rerun full simulations")
    args = parser.parse_args()
    verify(recompute=args.recompute)


if __name__ == "__main__":
    main()
