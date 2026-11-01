"""Audit the published Monte Carlo results for causal-inference Parts 1–5.

Run from repository root:

    python verify_foundations_results.py
    python verify_foundations_results.py --recompute

The quick mode validates the structure and arithmetic of committed summaries.
The optional full mode reruns all experiments in isolated temporary directories,
leaving the committed CSVs untouched. Requires NumPy, SciPy and scikit-learn.
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
PARTS: tuple[int, ...] = (1, 2, 3, 4, 5)
EXPECTED_ROWS: dict[int, int] = {1: 9, 2: 9, 3: 10, 4: 30, 5: 24}
TOLERANCE: float = 1e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    """Load a nonempty rectangular results table with unique headers."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise AssertionError(f"{path}: malformed CSV header")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise AssertionError(f"{path}: missing or malformed rows")
    return rows


def close(a: str, b: str) -> bool:
    """Compare numeric cells exactly enough for cross-platform minor noise."""
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def validate(part: int, rows: list[dict[str, str]]) -> None:
    """Check design dimensions, unique cells and relevant bias identities."""
    if len(rows) != EXPECTED_ROWS[part]:
        raise AssertionError(f"Part {part}: unexpected row count: {len(rows)}")
    if any(not all(field is not None for field in row.values()) for row in rows):
        raise AssertionError(f"Part {part}: missing field")
    if part == 1:
        labels = {(row["design"], row["n"]) for row in rows}
        if labels != {(d, str(n)) for d in ("naive", "adjusted", "randomized")
                     for n in (100, 1000, 10000)}:
            raise AssertionError("Part 1: missing design or sample size")
    if part == 2:
        labels = {(row["design"], row["n"]) for row in rows}
        if labels != {(d, str(n)) for d in
                     ("selected", "randomized", "randomized_adjusted")
                     for n in (200, 1000, 5000)}:
            raise AssertionError("Part 2: missing design or sample size")
    if part == 3:
        labels = {(row["hidden_confounding"], row["metric"]) for row in rows}
        if len(labels) != len(rows):
            raise AssertionError("Part 3: repeated confounding/metric combination")
        if any(int(row["n"]) != 1500 or int(row["replications"]) != 300
               for row in rows):
            raise AssertionError("Part 3: inconsistent sample size or repetitions")
    if part == 4:
        labels = {(row["strength"], row["metric"]) for row in rows}
        if len(labels) != len(rows):
            raise AssertionError("Part 4: repeated strength/metric combination")
    if part == 5:
        labels = {(row["n"], row["scenario"], row["model"]) for row in rows}
        if len(labels) != len(rows):
            raise AssertionError("Part 5: repeated sample/scenario/model combination")
    for row in rows:
        if part in (1, 2):
            true_effect = 1.0 if part == 1 else -1.0
            mean_key = "mean_est" if part == 1 else "mean_effect"
            if not math.isclose(
                float(row[mean_key]) - true_effect, float(row["bias"]),
                abs_tol=1.1e-4,
            ):
                raise AssertionError(f"Part {part}: rounded bias identity failed: {row}")
            if not (0 <= float(row["coverage"]) <= 1):
                raise AssertionError(f"Part {part}: invalid coverage: {row}")
        if part == 5:
            expected = float(row["mean"]) - float(row["target_total_effect"])
            if not math.isclose(expected, float(row["bias"]), abs_tol=TOLERANCE):
                raise AssertionError(f"Part 5: bias identity failed: {row}")
            if float(row["rmse"]) + TOLERANCE < abs(float(row["bias"])):
                raise AssertionError(f"Part 5: RMSE smaller than absolute bias: {row}")


def compare(part: int, old: list[dict[str, str]], new: list[dict[str, str]]) -> None:
    """Compare all stored and reproduced cells, including metadata."""
    if len(old) != len(new):
        raise AssertionError(f"Part {part}: row count changed")
    for index, (before, after) in enumerate(zip(old, new, strict=True), start=1):
        if before.keys() != after.keys():
            raise AssertionError(f"Part {part}: CSV schema changed")
        for field in before:
            if not close(before[field], after[field]):
                raise AssertionError(
                    f"Part {part}, row {index}, {field}: "
                    f"published={before[field]!r}, computed={after[field]!r}"
                )


def verify(*, recompute: bool) -> None:
    """Validate results and opt into complete Monte Carlo reproduction."""
    for part in PARTS:
        folder = ROOT / "causal_inference_statistics" / f"part{part}"
        published = read_csv(folder / "results.csv")
        validate(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                subprocess.run(
                    [sys.executable, str(script)], check=True,
                    capture_output=True, text=True, timeout=1200,
                )
                regenerated = read_csv(script.with_name("results.csv"))
                compare(part, published, regenerated)
        print(f"Part {part}: {'reproduced and validated' if recompute else 'validated'}")


def main() -> None:
    """Provide a cheap default audit with explicit full rerun switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Rerun every Monte Carlo experiment")
    opts = parser.parse_args()
    verify(recompute=opts.recompute)


if __name__ == "__main__":
    main()
