"""Stress-test false cure discovery when the population cure fraction is zero.

This is a confirmatory extension of the frozen mixture-cure simulation. It
uses independent random seeds and separate result filenames, preserving all
outputs from the original positive-cure experiment.

Run from the cure-fraction-identifiability directory:

    python no_cure_audit.py --reps 100 --workers 4
    python no_cure_audit.py --summary-only

The audit evaluates two distinct errors: AIC selecting any cure model when
the truth has no cured units, and reporting a cure fraction above 0.10 after
the article's combined agreement/AIC diagnostic passes.
"""

from __future__ import annotations

import argparse
import itertools
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from simulate import Q95, run_cell

ROOT: Final[Path] = Path(__file__).resolve().parent
RESULTS: Final[Path] = ROOT / "results"
LATENCIES: Final[tuple[str, ...]] = (
    "lognormal-0.5",
    "lognormal-1.0",
    "weibull-1.5",
)
SIZES: Final[tuple[int, ...]] = (250, 1000)
RATIOS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0)
TRUE_PI: Final[float] = 0.0
FALSE_CURE_THRESHOLD: Final[float] = 0.10
FAMILY_AGREEMENT_THRESHOLD: Final[float] = 0.05
SEED_OFFSET: Final[int] = 98_000
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "pi",
        "latency",
        "n",
        "ratio",
        "rep",
        "events",
        "pi_best",
        "family_gap",
        "cure_preferred",
        "converged",
    }
)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a two-sided Wilson binomial interval for Monte Carlo rates."""
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    rate: float = successes / trials
    denominator: float = 1 + z * z / trials
    center: float = (rate + z * z / (2 * trials)) / denominator
    half_width: float = (
        z
        * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def validate_raw(raw: pd.DataFrame) -> None:
    """Reject missing, empty or non-null experiment data."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame")
    missing: set[str] = set(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing audit columns: {', '.join(sorted(missing))}")
    if raw.empty or raw[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Audit data must be nonempty and contain no null required values")
    if not np.allclose(raw["pi"].to_numpy(dtype=float), TRUE_PI):
        raise ValueError("The no-cure audit requires pi=0 in every dataset")


def annotate(raw: pd.DataFrame) -> pd.DataFrame:
    """Attach predeclared rule outcomes without changing source fit columns."""
    validate_raw(raw)
    out: pd.DataFrame = raw.copy()
    positive: pd.Series = out["pi_best"] > FALSE_CURE_THRESHOLD
    preferred: pd.Series = out["cure_preferred"] == 1
    agreed: pd.Series = out["family_gap"] < FAMILY_AGREEMENT_THRESHOLD
    out["false_positive_fit"] = positive.astype(int)
    out["screen_pass"] = (preferred & agreed).astype(int)
    out["false_cure_report"] = (preferred & agreed & positive).astype(int)
    return out


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate false-positive rates by prespecified latency/sample/window."""
    annotated: pd.DataFrame = annotate(raw)
    rows: list[dict[str, float | int | str]] = []
    # Explicit grouping avoids pandas-version-dependent groupby.apply behavior.
    for (latency, n, ratio), group in annotated.groupby(
        ["latency", "n", "ratio"], sort=True
    ):
        observed: int = len(group)
        false_reports: int = int(group["false_cure_report"].sum())
        ci_low, ci_high = wilson_interval(false_reports, observed)
        rows.append(
            {
                "latency": str(latency),
                "n": int(n),
                "ratio": float(ratio),
                "reps": observed,
                "mean_event_fraction": float((group["events"] / group["n"]).mean()),
                "mean_pi_best": float(group["pi_best"].mean()),
                "false_positive_fit": float(group["false_positive_fit"].mean()),
                "aic_prefers_cure": float(group["cure_preferred"].mean()),
                "screen_pass": float(group["screen_pass"].mean()),
                "false_cure_report": float(group["false_cure_report"].mean()),
                "false_report_ci_low": ci_low,
                "false_report_ci_high": ci_high,
                "all_fits_converged": float(group["converged"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["latency", "n", "ratio"]).reset_index(
        drop=True
    )


def run_experiment(reps: int, workers: int) -> pd.DataFrame:
    """Run one independent pi=0 Monte Carlo grid."""
    if not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0:
        raise ValueError("reps must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    grid: list[tuple[str, int, float]] = list(
        itertools.product(LATENCIES, SIZES, RATIOS)
    )
    jobs: list[tuple[float, str, int, float, int, int]] = [
        (TRUE_PI, latency, n, ratio, reps, SEED_OFFSET + index)
        for index, (latency, n, ratio) in enumerate(grid)
    ]
    if workers == 1:
        parts: list[list[dict[str, object]]] = [run_cell(job) for job in jobs]
    else:
        with Pool(processes=workers) as pool:
            parts = pool.map(run_cell, jobs, chunksize=1)
    raw: pd.DataFrame = pd.DataFrame(
        record for part in parts for record in part
    )
    validate_raw(raw)
    return raw


def format_summary(raw: pd.DataFrame, cells: pd.DataFrame) -> str:
    """Make a compact, stable report without speculative conclusions."""
    annotated: pd.DataFrame = annotate(raw)
    statements: list[str] = [
        "No-cure stress test: pi = 0 for every simulated dataset",
        f"Datasets: {len(raw)}; cells: {len(cells)}; replicates per cell: {int(cells.reps.min())}",
        f"Latency 95th percentile: {Q95:g}; ratios: {', '.join(f'{r:g}' for r in RATIOS)}",
        "False cure report = AIC prefers cure AND lognormal/Weibull cure fractions agree within 0.05 AND selected cure fraction > 0.10.",
        "Intervals: two-sided 95% Wilson binomial intervals (100 Monte Carlo replicates per cell).",
        f"Across grid: AIC prefers cure {annotated.cure_preferred.mean():.3f}; "
        f"screen passes {annotated.screen_pass.mean():.3f}; "
        f"false cure reported {annotated.false_cure_report.mean():.3f}; "
        f"all four fits converge {annotated.converged.mean():.3f}.",
        "",
    ]
    for _, row in cells.iterrows():
        statements.append(
            f"{row.latency} n={int(row.n):4d} R={row.ratio:4.2f}: "
            f"observed event share {row.mean_event_fraction:.3f}; "
            f"mean selected pi {row.mean_pi_best:.3f}; "
            f"pi>0.10 {row.false_positive_fit:.3f}; "
            f"AIC cure {row.aic_prefers_cure:.3f}; "
            f"combined screen {row.screen_pass:.3f}; "
            f"false report {row.false_cure_report:.3f}; "
            f"95% interval [{row.false_report_ci_low:.3f}, {row.false_report_ci_high:.3f}]; "
            f"converged {row.all_fits_converged:.3f}"
        )
    return "\n".join(statements) + "\n"


def main() -> int:
    """Generate or summarize the prespecified experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild summary from the archived no_cure_raw.csv",
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        raw: pd.DataFrame = pd.read_csv(RESULTS / "no_cure_raw.csv")
    else:
        raw = run_experiment(reps=args.reps, workers=args.workers)
        raw.to_csv(RESULTS / "no_cure_raw.csv", index=False, float_format="%.9g")
    cells: pd.DataFrame = summarize(raw)
    cells.to_csv(RESULTS / "no_cure_cells.csv", index=False, float_format="%.9g")
    report: str = format_summary(raw, cells)
    (RESULTS / "no_cure_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
