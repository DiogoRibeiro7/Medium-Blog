"""Paired, small-sample sensitivity of the frozen midterm model comparisons.

The ten chronological test elections are the units of comparison. Every
resample preserves the *pair* of losses for a model and OLS in the same year.
This is descriptive resampling, not a calibrated confidence interval: ten
electoral cycles cannot establish independent, exchangeable forecast errors.

Both published Gallup transcriptions are examined with identical resampled
year indices, and no 2026 election outcome enters any calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from analysis import ROOT
from model_extensions import MODEL_NAMES, ModelEvidence, ModelFold, compute_model_evidence

Array = npt.NDArray[np.float64]
BOOTSTRAP_SEED = 20261031
BOOTSTRAP_DRAWS = 20_000
CANDIDATES = tuple(name for name in MODEL_NAMES if name != "ols")
SOURCE_NAMES = ("primary", "alternative_gallup")


@dataclass(frozen=True)
class PairedLoss:
    """Per-year squared-loss differences: candidate minus reference OLS."""

    source: str
    year: int
    actual: int
    ols_squared_error: float
    student_t_minus_ols: float
    ridge_minus_ols: float
    recency_wls_minus_ols: float


@dataclass(frozen=True)
class LeaveOneYearOut:
    """RMSE difference after dropping one common historical test year."""

    source: str
    candidate: str
    omitted_year: int
    delta_rmse: float


@dataclass(frozen=True)
class PairedComparison:
    """Descriptive comparisons, with percentiles of paired year resampling."""

    source: str
    candidate: str
    count: int
    delta_mse: float
    delta_rmse: float
    bootstrap_q025: float
    bootstrap_q975: float
    bootstrap_fraction_better: float
    omitted_year_delta_min: float
    omitted_year_delta_max: float
    omitted_year_reversal_count: int
    lower_absolute_error_count: int
    largest_loss_difference_year: int
    largest_loss_difference_share: float


@dataclass(frozen=True)
class PairedEvidence:
    """Auditable per-election, omission, bootstrap, and metadata outputs."""

    losses: tuple[PairedLoss, ...]
    comparisons: tuple[PairedComparison, ...]
    omissions: tuple[LeaveOneYearOut, ...]
    summary: dict[str, object]


def _validated(folds: Iterable[ModelFold]) -> tuple[ModelFold, ...]:
    """Reject mismatched, duplicated, or nonfinite held-out predictions."""

    rows = tuple(folds)
    if len(rows) < 3:
        raise ValueError("At least three paired held-out elections are required")
    years = [r.year for r in rows]
    if years != sorted(set(years)):
        raise ValueError("Held-out elections must be sorted and unique")
    for row in rows:
        if row.year >= 2026:
            raise ValueError("Prospective 2026 election outcomes cannot enter training or evaluation")
        if not all(math.isfinite(float(getattr(row, name))) for name in MODEL_NAMES):
            raise ValueError("All forecasts must be finite")
    return rows


def _squared_errors(rows: tuple[ModelFold, ...]) -> dict[str, Array]:
    """Compute all model losses from the identical realized election years."""

    actual: Array = np.array([row.actual for row in rows], dtype=float)
    return {
        name: (np.array([getattr(row, name) for row in rows], dtype=float) - actual) ** 2
        for name in MODEL_NAMES
    }


def _reversal_count(delta: float, omitted: Array) -> int:
    """Count exclusions that strictly reverse the observed RMSE ranking."""

    if delta == 0.0:
        return 0
    return int(np.count_nonzero(omitted * delta < 0))


def compute_paired_evidence(
    models: ModelEvidence | None = None,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> PairedEvidence:
    """Resample ten *paired* test years and examine omission sensitivity.

    Every candidate is compared with OLS on exactly the same original folds.
    The generator produces one draw matrix, reused across candidates and
    Gallup variants. Nothing is refitted on resampled test outcomes: the
    historical forward forecasts were already fitted strictly in the past.
    """

    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ValueError("Bootstrap draws must be an integer of at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")

    source = models if models is not None else compute_model_evidence()
    versions = {
        "primary": _validated(source.folds),
        "alternative_gallup": _validated(source.alternative_folds),
    }
    target_rows = [(row.year, row.actual, row.train_count) for row in versions["primary"]]
    if target_rows != [
        (row.year, row.actual, row.train_count) for row in versions["alternative_gallup"]
    ]:
        raise ValueError("Source variants must share identical held-out target rows")

    count = len(target_rows)
    generator = np.random.default_rng(seed)
    # Resample test-year indices once; a full candidate/reference pair travels together.
    indices = generator.integers(0, count, size=(draws, count))

    losses: list[PairedLoss] = []
    comparisons: list[PairedComparison] = []
    omissions: list[LeaveOneYearOut] = []
    baseline_rmse: dict[str, float] = {}
    for source_name in SOURCE_NAMES:
        rows = versions[source_name]
        squared = _squared_errors(rows)
        reference = squared["ols"]
        reference_rmse = float(np.sqrt(np.mean(reference)))
        baseline_rmse[source_name] = reference_rmse

        for i, row in enumerate(rows):
            losses.append(PairedLoss(
                source=source_name,
                year=row.year,
                actual=row.actual,
                ols_squared_error=float(reference[i]),
                student_t_minus_ols=float(squared["student_t"][i] - reference[i]),
                ridge_minus_ols=float(squared["ridge"][i] - reference[i]),
                recency_wls_minus_ols=float(squared["recency_wls"][i] - reference[i]),
            ))

        sampled_reference: Array = np.sqrt(np.mean(reference[indices], axis=1))
        for name in CANDIDATES:
            candidate = squared[name]
            year_differences: Array = candidate - reference
            delta_rmse = float(np.sqrt(np.mean(candidate)) - reference_rmse)
            sampled: Array = np.sqrt(np.mean(candidate[indices], axis=1))
            bootstrap_delta: Array = sampled - sampled_reference

            # Removing one year from both error series retains the pairing.
            omitted_deltas: Array = (
                np.sqrt((np.sum(candidate) - candidate) / (count - 1))
                - np.sqrt((np.sum(reference) - reference) / (count - 1))
            )
            omissions.extend(
                LeaveOneYearOut(source_name, name, row.year, float(omitted_deltas[i]))
                for i, row in enumerate(rows)
            )
            most_influential = int(np.argmax(np.abs(year_differences)))
            total_abs = float(np.sum(np.abs(year_differences)))
            comparisons.append(PairedComparison(
                source=source_name,
                candidate=name,
                count=count,
                delta_mse=float(np.mean(year_differences)),
                delta_rmse=delta_rmse,
                bootstrap_q025=float(np.quantile(bootstrap_delta, 0.025)),
                bootstrap_q975=float(np.quantile(bootstrap_delta, 0.975)),
                bootstrap_fraction_better=float(np.mean(bootstrap_delta < 0)),
                omitted_year_delta_min=float(np.min(omitted_deltas)),
                omitted_year_delta_max=float(np.max(omitted_deltas)),
                omitted_year_reversal_count=_reversal_count(delta_rmse, omitted_deltas),
                lower_absolute_error_count=int(np.count_nonzero(candidate < reference)),
                largest_loss_difference_year=rows[most_influential].year,
                largest_loss_difference_share=(
                    float(abs(year_differences[most_influential]) / total_abs)
                    if total_abs > 0 else 0.0
                ),
            ))

    summary = {
        "as_of": "2026-10-31",
        "method": "paired_year_bootstrap_and_leave_one_year_out",
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "bootstrap_quantiles": [0.025, 0.975],
        "historical_forward_years": [row[0] for row in target_rows],
        "forward_count": count,
        "candidate_names": list(CANDIDATES),
        "source_names": list(SOURCE_NAMES),
        "baseline_ols_rmse": baseline_rmse,
        "model_comparisons": [asdict(row) for row in comparisons],
        "interpretation": (
            "Descriptive paired-year resampling, not calibrated confidence intervals; "
            "electoral cycles may be dependent/nonexchangeable, and historical "
            "model specification choices precede this diagnostic."
        ),
    }
    return PairedEvidence(tuple(losses), tuple(comparisons), tuple(omissions), summary)


def _write_csv(path: Path, rows: Iterable[object], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save_paired_evidence(evidence: PairedEvidence, output: Path) -> None:
    """Write three tidy tables and the provenance-bearing numerical snapshot."""

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_loss_differences.csv", evidence.losses,
               tuple(PairedLoss.__dataclass_fields__))
    _write_csv(output / "paired_model_comparisons.csv", evidence.comparisons,
               tuple(PairedComparison.__dataclass_fields__))
    _write_csv(output / "paired_leave_one_year_out.csv", evidence.omissions,
               tuple(LeaveOneYearOut.__dataclass_fields__))
    (output / "paired_uncertainty_summary.json").write_text(
        json.dumps(evidence.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Output root containing results/")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    evidence = compute_paired_evidence(draws=args.draws, seed=args.seed)
    save_paired_evidence(evidence, args.output / "results")
    for row in evidence.comparisons:
        print(f"{row.source:18s} {row.candidate:11s}: RMSE difference "
              f"{row.delta_rmse:+.3f}, paired-resample 2.5–97.5% "
              f"[{row.bootstrap_q025:+.3f}, {row.bootstrap_q975:+.3f}], "
              f"year omissions reversing sign={row.omitted_year_reversal_count}")


if __name__ == "__main__":
    main()
