"""Paired synchronized quarterly-block bootstrap for country comparisons.

The time-block index is identical for all countries. It preserves observed
contemporaneous co-movement while approximately retaining serial dependence.
This is descriptive resampling of complete-case (X, y) rows, not a structural
GDP shock bootstrap or a causal country ranking.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.comparison import FOCUS_COUNTRIES, common_sample_designs
from eu_okun.contract import EurostatDataError, quarter_sequence
from eu_okun.stability import holm_adjust


@dataclass(frozen=True)
class PairResult:
    """Observed difference and percentile interval from paired block resamples."""

    country_a: str
    country_b: str
    estimate_a_minus_b: float
    ci95_low: float
    ci95_high: float
    valid_replicates: int
    requested_replicates: int
    bootstrap_sign_p: float
    holm_sign_p: float
    block_length: int


def _ols_sum(x: np.ndarray, y: np.ndarray) -> float:
    """Full-rank two-lag OLS sum; invalid draws must not pass silently."""
    if x.shape[0] < 3 * x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Insufficient rank or bootstrap sample size.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[1:4].sum())


def paired_block_comparisons(
    designs: dict[str, tuple[np.ndarray, np.ndarray]],
    replicates: int = 999,
    block_length: int = 8,
    seed: int = 20261015,
) -> list[PairResult]:
    """Circular moving-block bootstrap with shared draws for every country.

    The sign-frequency p-value is an approximate descriptive sign test, not
    a null-centered bootstrap significance test for country equality.
    """
    if replicates < 199 or block_length < 2 or len(designs) < 2:
        raise ValueError("Need >=199 replicates, blocks >=2 and at least two countries.")
    sizes = {x.shape[0] for x, _ in designs.values()}
    if len(sizes) != 1:
        raise ValueError("Country designs must share identical quarterly rows.")
    n = sizes.pop()
    if n < max(30, 4 * block_length):
        raise ValueError("Series too short for requested block length.")
    names = tuple(sorted(designs))
    observed = {key: _ols_sum(*designs[key]) for key in names}
    differences: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in itertools.combinations(names, 2)
    }
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block_length)
    for _ in range(replicates):
        origins = rng.integers(0, n, size=blocks)
        sample = np.concatenate([
            (origin + np.arange(block_length)) % n for origin in origins
        ])[:n]
        estimates: dict[str, float] = {}
        for name in names:
            x, y = designs[name]
            try:
                estimates[name] = _ols_sum(x[sample, :], y[sample])
            except ValueError:
                break
        if len(estimates) != len(names):
            continue
        for (a, b), series in differences.items():
            series.append(estimates[a] - estimates[b])
    output: list[PairResult] = []
    raw_p: list[float] = []
    for (a, b), values in differences.items():
        valid = len(values)
        if valid < int(0.9 * replicates):
            raise EurostatDataError(
                f"Too many rank-deficient bootstrap draws for {a}/{b}: {valid}"
            )
        draws = np.asarray(values, dtype=float)
        difference = observed[a] - observed[b]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        # Two-sided sign probability is descriptive. Report it explicitly
        # rather than mislabeling it as a null-centered bootstrap test.
        p = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
        raw_p.append(p)
        output.append(PairResult(
            a, b, difference, float(lo), float(hi), valid, replicates,
            p, 1.0, block_length,
        ))
    from dataclasses import replace

    corrected = holm_adjust(raw_p)
    return [replace(row, holm_sign_p=p) for row, p in zip(output, corrected, strict=True)]


def run_country_pairs(
    data_dir: Path, output_dir: Path, replicates: int = 999,
) -> int:
    """Validate Eurostat source and run block-length sensitivity for focus pairs."""
    audit = audit_snapshot(data_dir)
    period = audit["analysis_period"]
    first, last = period["start"], period["end"]
    if not isinstance(first, str) or not isinstance(last, str):
        raise EurostatDataError("Invalid source coverage period.")
    quarters = quarter_sequence(first, last)
    panel = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    matrices, dates = common_sample_designs(
        panel, quarters, countries=FOCUS_COUNTRIES, lag_orders=(2,),
    )
    samples = {geo: matrices[(geo, 2, False)] for geo in FOCUS_COUNTRIES}
    results: list[PairResult] = []
    for length in (4, 8, 12):
        results.extend(paired_block_comparisons(
            samples, replicates=replicates, block_length=length
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": "Synchronized circular block bootstrap of paired complete-case rows",
        "caveat": "Descriptive; not a structural causal comparison or a formal ranking",
        "quarter_sample": dates,
        "seed": 20261015,
        "replicates": replicates,
        "snapshot_manifest_sha256": hashlib.sha256(
            (data_dir / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "pairs": [asdict(result) for result in results],
    }
    (output_dir / "country_pairs.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return len(results)
