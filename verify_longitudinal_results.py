"""Verify the published longitudinal Monte Carlo tables in Parts 11 and 15.

Run from the repository root:

    python verify_longitudinal_results.py
    python verify_longitudinal_results.py --recompute

The default mode validates the committed summaries and their mathematical
identities without fitting models. The optional mode reproduces both experiments
in temporary directories and compares every reported numeric cell.
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
PARTS: tuple[int, ...] = (11, 15)
TRUE_EFFECT: float = 0.7 + 1.1 + 1.3 * 1.2
NUMERIC_ABS_TOL: float = 1e-8


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a simulation CSV, rejecting missing or duplicate columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate headers in {path}")
        records = list(reader)
    if not records or any(None in row for row in records):
        raise ValueError(f"Malformed or empty results table: {path}")
    return records


def validate_summary(part: int, rows: list[dict[str, str]]) -> None:
    """Validate design, effect identities, and the interpretation of ESS rows."""
    identity = "method" if part == 11 else "metric"
    expected = (
        {"g_correct", "g_omit_l1", "naive_adjustment"}
        if part == 11
        else {"msm_ipw", "g_formula", "direct_adjustment", "ess_11", "ess_00"}
    )
    labels = [row[identity] for row in rows]
    if set(labels) != expected or len(labels) != len(expected):
        raise AssertionError(f"Part {part}: unexpected or repeated estimator labels: {labels}")
    expected_n, expected_reps = ((2000, 300) if part == 11 else (2200, 160))
    for row in rows:
        label = row[identity]
        if (int(row["n"]), int(row["replications"])) != (expected_n, expected_reps):
            raise AssertionError(f"Part {part} {label}: design mismatch")
        is_ess = label.startswith("ess_")
        truth_key = "true_effect" if part == 11 else "true_total_effect"
        if is_ess:
            if row[truth_key] or row["bias"] or row["rmse"]:
                raise AssertionError(f"Part {part} {label}: ESS is not an effect estimate")
            if not (0 < float(row["mean"]) <= expected_n):
                raise AssertionError(f"Part {part} {label}: implausible effective sample size")
            continue
        truth = float(row[truth_key])
        mean = float(row["mean"])
        bias = float(row["bias"])
        sd = float(row["sd"])
        rmse = float(row["rmse"])
        if not math.isclose(truth, TRUE_EFFECT, abs_tol=NUMERIC_ABS_TOL):
            raise AssertionError(f"Part {part} {label}: incorrect structural truth")
        if not math.isclose(mean - truth, bias, abs_tol=NUMERIC_ABS_TOL):
            raise AssertionError(f"Part {part} {label}: bias identity violated")
        # The reported sd uses ddof=1 whereas RMSE uses denominator REPS.
        expected_rmse = math.sqrt(bias * bias + sd * sd * (expected_reps - 1) / expected_reps)
        if not math.isclose(rmse, expected_rmse, abs_tol=NUMERIC_ABS_TOL):
            raise AssertionError(f"Part {part} {label}: RMSE identity violated")
        if part == 11 and not (float(row["q05"]) <= mean <= float(row["q95"])):
            raise AssertionError(f"Part {part} {label}: invalid empirical quantiles")


def compare_recomputed(
    committed: list[dict[str, str]], reproduced: list[dict[str, str]], part: int
) -> None:
    """Compare labels, cells and floating-point statistics across runs."""
    if len(committed) != len(reproduced):
        raise AssertionError(f"Part {part}: changed number of rows")
    for old, new in zip(committed, reproduced, strict=True):
        if old.keys() != new.keys():
            raise AssertionError(f"Part {part}: changed CSV schema")
        for field in old:
            if old[field] == new[field]:
                continue
            try:
                equivalent = math.isclose(
                    float(old[field]), float(new[field]),
                    rel_tol=NUMERIC_ABS_TOL, abs_tol=NUMERIC_ABS_TOL
                )
            except ValueError:
                equivalent = False
            if not equivalent:
                raise AssertionError(
                    f"Part {part} {old.get('method', old.get('metric'))} "
                    f"{field}: committed={old[field]!r}, recomputed={new[field]!r}"
                )


def verify(*, recompute: bool) -> None:
    """Validate each published table and optionally repeat both simulations."""
    for part in PARTS:
        folder = ROOT / "causal_inference_statistics" / f"part{part}"
        published = read_table(folder / "results.csv")
        validate_summary(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                subprocess.run(
                    [sys.executable, str(script)],
                    check=True, capture_output=True, text=True, timeout=600
                )
                regenerated = read_table(script.with_name("results.csv"))
                compare_recomputed(published, regenerated, part)
        print(f"Part {part}: {'reproduced and validated' if recompute else 'validated'}")


def main() -> None:
    """Parse the explicit opt-in computational verification flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Rerun full Monte Carlo simulations")
    arguments = parser.parse_args()
    verify(recompute=arguments.recompute)


if __name__ == "__main__":
    main()
