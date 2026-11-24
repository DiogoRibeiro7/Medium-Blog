"""Audit pre-reform outcome and macro convex support, not causal effects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.donor_support import audit_future_donor_exposure
from causal_tax_cuts.macro_matching import (
    MacroValues,
    pre_reform_means,
    read_macro_controls,
)

TOL = 1e-9
CASES = (("FIN", 1986), ("DEU", 2008))


def in_hull(matrix: np.ndarray) -> bool:
    """Test whether standardized donor differences contain zero."""
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite two-dimensional donor matrix")
    n, p = matrix.shape
    target = np.r_[np.zeros(p), 1.0]
    for k in range(1, min(n, p + 1) + 1):
        for indices in itertools.combinations(range(n), k):
            system = np.vstack((matrix[list(indices)].T, np.ones(k)))
            weights = np.linalg.lstsq(system, target, rcond=None)[0]
            if np.min(weights) < -TOL:
                continue
            weights = np.maximum(weights, 0)
            if weights.sum() <= 0:
                continue
            weights /= weights.sum()
            if np.max(np.abs(system @ weights - target)) <= TOL:
                return True
    return False


def profile(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
    country: str,
    year: int,
    last_lag: int,
    mode: str,
) -> np.ndarray:
    """Build macro means and strictly pre-onset outcome features."""
    years = list(range(year - 5, year - last_lag + 1))
    # The existing macro helper averages the five years before its anchor.
    # Keep the same original macro window for all outcome specifications.
    macro = pre_reform_means(controls, country, year, 5)
    if macro is None:
        raise ValueError("Missing eligible macro history")
    values = np.array([panel[country, y].outcome for y in years])
    if mode == "macro":
        return np.asarray(macro)
    if mode == "level_slope":
        x = np.arange(len(values), dtype=float)
        slope = float(np.dot(x - x.mean(), values) / np.sum((x - x.mean()) ** 2))
        return np.r_[macro, values.mean(), slope]
    if mode == "trajectory":
        return np.r_[macro, values]
    raise ValueError("Unknown feature specification")


def audit(
    panel: dict[Key, Observation], controls: dict[Key, MacroValues]
) -> list[dict[str, object]]:
    """Keep original donor eligibility fixed and vary pre-outcome features."""
    _, candidates, _, _ = audit_future_donor_exposure(panel, controls)
    rows: list[dict[str, object]] = []
    for country, year in CASES:
        entries = sorted(
            (c for c in candidates if (c.country, c.reform_year) == (country, year)),
            key=lambda c: c.donor,
        )
        if not entries:
            raise ValueError(f"Missing original candidate: {country} {year}")
        for last_lag in (1, 2):
            for mode in ("macro", "level_slope", "trajectory"):
                target = profile(panel, controls, country, year, last_lag, mode)
                full = np.vstack([
                    profile(panel, controls, c.donor, year, last_lag, mode)
                    for c in entries
                ])
                scales = np.std(np.vstack((target, full)), axis=0)
                scales = np.where(scales < 1e-10, 1.0, scales)
                matrix = (full - target) / scales
                for horizon in (0, 5):
                    keep = [
                        i for i, c in enumerate(entries)
                        if c.first_censored_horizon is None
                        or c.first_censored_horizon > horizon
                    ]
                    subset = matrix[keep]
                    supported = in_hull(subset)
                    if mode == "macro" and not supported:
                        raise ValueError("Original strict macro support no longer reproduces")
                    # Deletion cannot turn an infeasible hull into a feasible one.
                    deletions = sum(
                        in_hull(np.delete(subset, i, axis=0))
                        for i in range(len(keep))
                    ) if supported else 0
                    rows.append({
                        "country": country,
                        "year": year,
                        "horizon": horizon,
                        "last_outcome_lag": last_lag,
                        "mode": mode,
                        "dimensions": len(target),
                        "donors": len(keep),
                        "supported": supported,
                        "supported_single_donor_deletions": deletions,
                    })
    return rows


def main() -> None:
    """Write diagnostic CSV and source-hashed manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    rows = audit(panel, read_macro_controls(args.panel, set(panel)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "outcome_hull.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "status": "descriptive_geometry_only",
        "rows": rows,
    }
    (args.output_dir / "outcome_hull.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
