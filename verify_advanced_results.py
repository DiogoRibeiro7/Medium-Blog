"""Verify published simulation results for causal inference Parts 12–14.

Run from the repository root:

    python verify_advanced_results.py
    python verify_advanced_results.py --recompute

The fast check needs only the standard library. The full check reruns the
experiments with their original seeds in temporary directories; it requires
NumPy and scikit-learn.
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
PARTS: tuple[int, ...] = (12, 13, 14)
ROW_COUNTS: dict[int, int] = {12: 11, 13: 8, 14: 18}
KEYS: dict[int, tuple[str, ...]] = {
    12: ("type", "method"),
    13: ("design", "method"),
    14: ("n", "design", "prior_precision"),
}
TOLERANCE: float = 1e-8


def rows_from(path: Path) -> list[dict[str, str]]:
    """Read a well-formed CSV while retaining its field order."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AssertionError(f"{path}: bad headers")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise AssertionError(f"{path}: empty or malformed table")
    return rows


def equivalent(a: str, b: str) -> bool:
    """Compare text exactly and numerical cells within precision tolerance."""
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def check_summary(part: int, records: list[dict[str, str]]) -> None:
    """Validate row dimensions, unique treatments, and estimand identities."""
    if len(records) != ROW_COUNTS[part]:
        raise AssertionError(f"Part {part}: expected {ROW_COUNTS[part]} rows")
    keys = [tuple(row[field] for field in KEYS[part]) for row in records]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"Part {part}: duplicate experimental setting")
    repetitions = {12: 400, 13: 20, 14: 300}[part]
    for row in records:
        if int(row["replications"]) != repetitions:
            raise AssertionError(f"Part {part}: inconsistent simulation count")
        if part == 12:
            if row["type"] not in ("prediction", "policy", "group_effect"):
                raise AssertionError(f"Part 12: unknown result type: {row['type']}")
            if not math.isfinite(float(row["mean"])):
                raise AssertionError("Part 12: non-finite mean")
        else:
            truth_key = "truth" if part == 13 else "true_effect"
            truth = float(row[truth_key])
            if not math.isclose(truth, 1.0, abs_tol=TOLERANCE):
                raise AssertionError(f"Part {part}: wrong stated causal effect")
            bias = float(row["mean"]) - truth
            if not math.isclose(bias, float(row["bias"]), abs_tol=TOLERANCE):
                raise AssertionError(f"Part {part}: bias arithmetic mismatch")
            if float(row["rmse"]) + TOLERANCE < abs(bias):
                raise AssertionError(f"Part {part}: RMSE below absolute bias")
            if part == 14:
                for field in ("causal_coverage", "positive_posterior_fraction"):
                    if not 0 <= float(row[field]) <= 1:
                        raise AssertionError(f"Part 14: invalid fraction {field}")
                if float(row["posterior_sd"]) <= 0:
                    raise AssertionError("Part 14: posterior SD must be positive")


def verify(*, recompute: bool) -> None:
    """Validate archived summaries, with optional complete seeded reruns."""
    for part in PARTS:
        folder = ROOT / "causal_inference_statistics" / f"part{part}"
        published = rows_from(folder / "results.csv")
        check_summary(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                subprocess.run(
                    [sys.executable, str(script)], check=True,
                    capture_output=True, text=True, timeout=1200,
                )
                reproduced = rows_from(script.with_name("results.csv"))
                if len(reproduced) != len(published):
                    raise AssertionError(f"Part {part}: row count drift")
                for line, (old, new) in enumerate(
                    zip(published, reproduced, strict=True), start=1
                ):
                    if old.keys() != new.keys():
                        raise AssertionError(f"Part {part}: column changes")
                    for field in old:
                        if not equivalent(old[field], new[field]):
                            raise AssertionError(
                                f"Part {part} row {line} {field}: "
                                f"published={old[field]!r}, reproduced={new[field]!r}"
                            )
        print(f"Part {part}: {'reproduced and validated' if recompute else 'validated'}")


def main() -> None:
    """Parse opt-in full reproduction flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Run full Monte Carlo experiments")
    opts = parser.parse_args()
    verify(recompute=opts.recompute)


if __name__ == "__main__":
    main()
