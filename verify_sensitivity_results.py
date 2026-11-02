"""Check published Monte Carlo results for causal-inference Parts 16–20.

Run from the repository root:

    python verify_sensitivity_results.py
    python verify_sensitivity_results.py --recompute

The fast path only uses the Python standard library. The full path reruns
each original experiment in an isolated temporary directory and requires
NumPy, SciPy and scikit-learn.
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
PARTS: tuple[int, ...] = (16, 17, 18, 19, 20)
ROWS: dict[int, int] = {16: 8, 17: 9, 18: 12, 19: 7, 20: 4}
KEYS: dict[int, tuple[str, ...]] = {
    16: ("strength", "model"),
    17: ("scenario", "test"),
    18: ("scenario", "method"),
    19: ("scenario", "method"),
    20: ("method",),
}
REPLICATIONS: dict[int, int] = {16: 400, 17: 500, 18: 300, 19: 400, 20: 500}
TOLERANCE: float = 1e-8


def read_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV and reject missing, repeated or malformed headers and rows."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AssertionError(f"{path}: invalid CSV headers")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise AssertionError(f"{path}: incomplete result table")
    return rows


def same(a: str, b: str) -> bool:
    """Compare scalar CSV values with a strict numeric tolerance."""
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, abs_tol=TOLERANCE, rel_tol=TOLERANCE)


def check_summary(part: int, data: list[dict[str, str]]) -> None:
    """Check experimental conditions, metadata and basic effect identities."""
    if len(data) != ROWS[part]:
        raise AssertionError(f"Part {part}: unexpected row count")
    labels = [tuple(row[key] for key in KEYS[part]) for row in data]
    if len(set(labels)) != len(labels):
        raise AssertionError(f"Part {part}: duplicate experimental condition")
    for row in data:
        if int(row["replications"]) != REPLICATIONS[part]:
            raise AssertionError(f"Part {part}: incorrect replication count")
        if part == 20:
            if (int(row["n_train"]), int(row["n_test"])) != (1200, 3000):
                raise AssertionError("Part 20: train/test design mismatch")
            if not math.isclose(float(row["capacity"]), 0.2, abs_tol=TOLERANCE):
                raise AssertionError("Part 20: capacity mismatch")
            if not math.isclose(
                float(row["mean_population_gain"]),
                float(row["capacity"]) * float(row["mean_benefit"]),
                abs_tol=TOLERANCE,
            ):
                raise AssertionError("Part 20: population gain identity violated")
            if float(row["mean_regret"]) < -TOLERANCE:
                raise AssertionError("Part 20: negative regret")
            continue
        if part == 19:
            truth_key = "target"
        else:
            truth_key = "true_effect"
        truth = float(row[truth_key])
        if not math.isclose(
            float(row["mean"]) - truth, float(row["bias"]), abs_tol=TOLERANCE
        ):
            raise AssertionError(f"Part {part}: bias identity violated")
        if float(row["rmse"]) + TOLERANCE < abs(float(row["bias"])):
            raise AssertionError(f"Part {part}: RMSE below absolute bias")
        if part in (16, 17):
            if not 0 <= float(row["coverage"]) <= 1:
                raise AssertionError(f"Part {part}: coverage outside [0, 1]")
        if part == 17 and not 0 <= float(row["reject_zero"]) <= 1:
            raise AssertionError("Part 17: invalid rejection frequency")
        if part == 18 and not 0 <= float(row["mean_observed_fraction"]) <= 1:
            raise AssertionError("Part 18: invalid observation fraction")
        if part == 19 and not 0 <= float(row["target_support_fraction"]) <= 1:
            raise AssertionError("Part 19: invalid target support fraction")
        if part == 16:
            strength = float(row["strength"])
            theoretical = (
                1.5 * strength / (1.0 + strength**2)
                if row["model"] == "observed_only" else 0.0
            )
            if not math.isclose(
                float(row["theoretical_observed_bias"]), theoretical,
                abs_tol=TOLERANCE,
            ):
                raise AssertionError("Part 16: analytical bias identity violated")


def verify(*, recompute: bool) -> None:
    """Validate every table, optionally reproducing the full computation."""
    for part in PARTS:
        folder = ROOT / "causal_inference_statistics" / f"part{part}"
        published = read_rows(folder / "results.csv")
        check_summary(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                subprocess.run(
                    [sys.executable, str(script)], capture_output=True,
                    text=True, check=True, timeout=1200,
                )
                generated = read_rows(script.with_name("results.csv"))
                if len(generated) != len(published):
                    raise AssertionError(f"Part {part}: row count drift")
                for line, (old, new) in enumerate(
                    zip(published, generated, strict=True), start=1
                ):
                    if old.keys() != new.keys():
                        raise AssertionError(f"Part {part}: changed CSV columns")
                    for key in old:
                        if not same(old[key], new[key]):
                            raise AssertionError(
                                f"Part {part} row {line} {key}: "
                                f"published={old[key]!r}, generated={new[key]!r}"
                            )
        print(f"Part {part}: {'reproduced and validated' if recompute else 'validated'}")


def main() -> None:
    """Parse full-reproduction opt-in and run verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Rerun seeded experiments")
    opts = parser.parse_args()
    verify(recompute=opts.recompute)


if __name__ == "__main__":
    main()
