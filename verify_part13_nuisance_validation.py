"""Verify the independent Part 13 nuisance-basis validation artifact.

Run from the repository root:

    python verify_part13_nuisance_validation.py
    python verify_part13_nuisance_validation.py --recompute

The default validation only uses the standard library. A full rerun requires
NumPy and reproduces 1,000 replications for each design in a temporary folder.
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
SOURCE: Path = ROOT / "causal_inference_statistics" / "part13"
SCHEMA: tuple[str, ...] = (
    "design", "method", "n", "replications", "truth", "mean",
    "bias", "sd", "rmse",
)
CONDITIONS: set[tuple[str, str]] = {
    (design, method)
    for design in ("linear", "nonlinear")
    for method in ("linear_basis", "correct_basis", "oracle")
}
TOLERANCE: float = 1e-8


def read_results(path: Path) -> list[dict[str, str]]:
    """Load exactly six finite, correctly labelled validation rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SCHEMA:
            raise AssertionError(f"{path}: unexpected CSV schema")
        rows = list(reader)
    if len(rows) != len(CONDITIONS):
        raise AssertionError(f"{path}: expected six rows")
    labels = [(row["design"], row["method"]) for row in rows]
    if set(labels) != CONDITIONS or len(set(labels)) != len(labels):
        raise AssertionError(f"{path}: missing or duplicate validation condition")
    for row in rows:
        if set(row) != set(SCHEMA):
            raise AssertionError(f"{path}: unexpected columns")
        if int(row["n"]) != 800 or int(row["replications"]) != 1000:
            raise AssertionError(f"{path}: invalid simulation dimensions")
        for field in SCHEMA[2:]:
            value = float(row[field])
            if not math.isfinite(value):
                raise AssertionError(f"{path}: nonfinite {field}")
        truth = float(row["truth"])
        mean = float(row["mean"])
        bias = float(row["bias"])
        sd = float(row["sd"])
        rmse = float(row["rmse"])
        if not math.isclose(truth, 1.0, abs_tol=TOLERANCE):
            raise AssertionError(f"{path}: incorrect true effect")
        if not math.isclose(mean - truth, bias, abs_tol=TOLERANCE):
            raise AssertionError(f"{path}: inconsistent bias")
        expected_rmse = math.sqrt(bias**2 + sd**2 * (1000 - 1) / 1000)
        if not math.isclose(rmse, expected_rmse, abs_tol=TOLERANCE):
            raise AssertionError(f"{path}: inconsistent RMSE")
    return rows


def verify(*, recompute: bool) -> None:
    """Check summary arithmetic and optionally reproduce every CSV cell."""
    committed = read_results(SOURCE / "validation_results.csv")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="causal-part13-validation-") as tmp:
            script = Path(tmp) / "validate_nuisance_basis.py"
            script.write_bytes((SOURCE / "validate_nuisance_basis.py").read_bytes())
            run = subprocess.run(
                [sys.executable, str(script)], check=False,
                capture_output=True, text=True, timeout=1200,
            )
            if run.returncode:
                raise RuntimeError(
                    f"Part 13 validation script exited {run.returncode}\n"
                    f"stderr:\n{run.stderr or '(empty)'}\n"
                    f"stdout:\n{run.stdout or '(empty)'}"
                )
            regenerated = read_results(script.with_name("validation_results.csv"))
            for index, (before, after) in enumerate(
                zip(committed, regenerated, strict=True), start=1
            ):
                for field in SCHEMA:
                    if field in ("design", "method"):
                        equal = before[field] == after[field]
                    else:
                        equal = math.isclose(
                            float(before[field]), float(after[field]),
                            rel_tol=TOLERANCE, abs_tol=TOLERANCE,
                        )
                    if not equal:
                        raise AssertionError(
                            f"Row {index} {field}: published={before[field]!r}, "
                            f"regenerated={after[field]!r}"
                        )
    print("Part 13 nuisance-basis validation: "
          + ("reproduced and validated" if recompute else "validated"))


def main() -> None:
    """Provide a fast default and opt-in complete reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Rerun all validation replicates")
    args = parser.parse_args()
    verify(recompute=args.recompute)


if __name__ == "__main__":
    main()
