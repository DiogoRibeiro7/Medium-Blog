"""Chronological held-out pre-reform prediction, not causal estimation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.convex_approximation import project
from causal_tax_cuts.donor_support import audit_future_donor_exposure
from causal_tax_cuts.macro_matching import (
    MacroValues,
    read_macro_controls,
)
from causal_tax_cuts.outcome_hull import CASES


def predict(
    target: np.ndarray, donors: np.ndarray, holdout: np.ndarray
) -> tuple[float, float]:
    """Fit weights on training outcomes only and predict one held-out year."""
    if (
        target.ndim != 1
        or donors.ndim != 2
        or donors.shape[1] != target.size
        or holdout.shape != (donors.shape[0],)
        or target.size < 2
        or not donors.shape[0]
        or not all(np.isfinite(x).all() for x in (target, donors, holdout))
    ):
        raise ValueError("Invalid training or held-out arrays")
    scale = np.std(np.vstack((target, donors)), axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    weights = project((donors - target) / scale)
    return float(weights @ holdout), float(1.0 / (weights @ weights))


def audit(
    panel: dict[Key, Observation], controls: dict[Key, MacroValues]
) -> list[dict[str, object]]:
    """Use expanding training windows and the original fixed donor pools."""
    _, candidates, _, _ = audit_future_donor_exposure(panel, controls)
    rows: list[dict[str, object]] = []
    for country, year in CASES:
        for horizon in (0, 5):
            names = sorted(
                c.donor
                for c in candidates
                if (c.country, c.reform_year) == (country, year)
                and (
                    c.first_censored_horizon is None
                    or c.first_censored_horizon > horizon
                )
            )
            if not names:
                raise ValueError("Missing original donor pool")
            for lag in (3, 2, 1):
                held_year = year - lag
                years = range(year - 5, held_year)
                target = np.array([panel[country, y].outcome for y in years])
                donors = np.array(
                    [[panel[name, y].outcome for y in years] for name in names]
                )
                held = np.array([panel[name, held_year].outcome for name in names])
                actual = panel[country, held_year].outcome
                prediction, effective = predict(target, donors, held)
                predictions = {
                    "convex_outcome": prediction,
                    "equal_weight": float(held.mean()),
                    "persistence": float(target[-1]),
                }
                for method, value in predictions.items():
                    rows.append(
                        {
                            "country": country,
                            "year": year,
                            "horizon": horizon,
                            "holdout_year": held_year,
                            "training_years": len(target),
                            "method": method,
                            "actual_pp": actual,
                            "prediction_pp": value,
                            "error_pp": value - actual,
                            "donor_count": len(names),
                            "effective_donors": (
                                effective if method == "convex_outcome" else None
                            ),
                        }
                    )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate three chronological holdouts per case, pool and method."""
    summaries: list[dict[str, object]] = []
    for country, year in CASES:
        for horizon in (0, 5):
            for method in ("convex_outcome", "equal_weight", "persistence"):
                errors = np.array(
                    [
                        float(row["error_pp"])
                        for row in rows
                        if (row["country"], row["year"]) == (country, year)
                        and row["horizon"] == horizon
                        and row["method"] == method
                    ]
                )
                if len(errors) != 3:
                    raise ValueError("Expected three chronological holdouts")
                summaries.append(
                    {
                        "country": country,
                        "year": year,
                        "horizon": horizon,
                        "method": method,
                        "rmse_pp": float(np.sqrt(np.mean(errors**2))),
                        "mae_pp": float(np.mean(np.abs(errors))),
                    }
                )
    return summaries


def main() -> None:
    """Publish pre-period prediction errors and input provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    rows = audit(panel, read_macro_controls(args.panel, set(panel)))
    manifest = {
        "status": "conditional_pre_period_prediction_only",
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "rows": rows,
        "summary": summarize(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "heldout_prefit.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
