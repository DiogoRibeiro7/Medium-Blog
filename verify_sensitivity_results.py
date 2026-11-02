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
SCHEMAS: dict[int, tuple[str, ...]] = {
    16: ("strength", "model", "n", "replications", "true_effect", "mean",
         "bias", "rmse", "mean_se", "coverage", "theoretical_observed_bias"),
    17: ("scenario", "test", "n", "replications", "true_effect", "mean",
         "bias", "rmse", "mean_se", "coverage", "reject_zero"),
    18: ("scenario", "method", "n", "replications", "true_effect", "mean",
         "bias", "sd", "rmse", "mean_observed_fraction"),
    19: ("scenario", "method", "n_trial", "n_target", "replications", "target",
         "mean", "bias", "rmse", "sd", "target_support_fraction"),
    20: ("method", "replications", "n_train", "n_test", "capacity", "true_ate",
         "mean_benefit", "sd", "mean_regret", "mean_population_gain"),
}
EXPECTED_LABELS: dict[int, set[tuple[str, ...]]] = {
    16: {(strength, model) for strength in ("0.0", "0.25", "0.5", "1.0")
         for model in ("observed_only", "oracle_adjusted")},
    17: {(scenario, test) for scenario in (
        "no_confounding", "shared_confounder", "outcome_specific_confounder"
    ) for test in ("target", "negative_control_outcome", "placebo_exposure")},
    18: {(scenario, method) for scenario in ("mcar", "mar", "mnar")
         for method in ("full_data", "complete_case", "response_ipw",
                        "outcome_regression")},
    19: {(scenario, method) for scenario in (
        "measured_effect_modifier", "omitted_effect_modifier", "nonoverlap"
    ) for method in (("trial", "transported", "restricted")
                     if scenario == "nonoverlap" else ("trial", "transported"))},
    20: {(method,) for method in (
        "random", "high_risk", "estimated_benefit", "oracle"
    )},
}
REPLICATIONS: dict[int, int] = {16: 400, 17: 500, 18: 300, 19: 400, 20: 500}
TOLERANCE: float = 1e-8


def read_rows(path: Path, part: int) -> list[dict[str, str]]:
    """Load CSV and reject unexpected schemas or nonfinite numeric cells."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SCHEMAS[part]:
            raise AssertionError(f"{path}: unexpected CSV schema")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise AssertionError(f"{path}: incomplete result table")
    labels = set(KEYS[part])
    for line, row in enumerate(rows, start=2):
        for field, value in row.items():
            if value is None or not value.strip():
                raise AssertionError(f"{path}:{line}: empty {field}")
            if field not in labels:
                try:
                    number = float(value)
                except ValueError as error:
                    raise AssertionError(
                        f"{path}:{line}: nonnumeric {field}: {value!r}"
                    ) from error
                if not math.isfinite(number):
                    raise AssertionError(f"{path}:{line}: nonfinite {field}")
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
    if set(labels) != EXPECTED_LABELS[part] or len(labels) != len(set(labels)):
        raise AssertionError(f"Part {part}: unexpected or missing experimental condition")
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
        published = read_rows(folder / "results.csv", part)
        check_summary(part, published)
        if recompute:
            with tempfile.TemporaryDirectory(prefix=f"causal-part{part}-") as tmp:
                script = Path(tmp) / "experiment.py"
                script.write_bytes((folder / "experiment.py").read_bytes())
                completed = subprocess.run(
                    [sys.executable, str(script)], capture_output=True,
                    text=True, check=False, timeout=1200,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"Part {part}: experiment failed with exit status "
                        f"{completed.returncode}\n"
                        f"stderr:\n{completed.stderr or '(empty)'}\n"
                        f"stdout:\n{completed.stdout or '(empty)'}"
                    )
                generated = read_rows(script.with_name("results.csv"), part)
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
