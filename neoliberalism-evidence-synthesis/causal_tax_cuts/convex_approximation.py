"""Quantify pre-onset convex approximation error without effect estimation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.donor_support import audit_future_donor_exposure
from causal_tax_cuts.macro_matching import MacroValues, read_macro_controls
from causal_tax_cuts.outcome_hull import CASES, profile


def project(matrix: np.ndarray) -> np.ndarray:
    """Find simplex weights minimizing squared norm of the weighted rows.

    Enumerate affinely independent faces of at most p+1 vertices. A
    closest point has such a representation by Caratheodory's theorem.
    Ties retain the first encountered certificate, not unique weights.
    """
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Expected finite nonempty donor matrix")
    n, p = matrix.shape
    best = np.zeros(n)
    best[0] = 1.0
    loss = float(np.dot(matrix[0], matrix[0]))
    for k in range(1, min(n, p + 1) + 1):
        for indices in itertools.combinations(range(n), k):
            face = matrix[list(indices)]
            delta = (face[1:] - face[0]).T
            tail = np.linalg.lstsq(delta, -face[0], rcond=None)[0]
            weights = np.r_[1.0 - tail.sum(), tail]
            if np.min(weights) < -1e-10:
                continue
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            residual = weights @ face
            candidate = float(residual @ residual)
            if candidate < loss:
                loss = candidate
                best = np.zeros(n)
                best[list(indices)] = weights
    # First-order optimality over the simplex independently checks the fit.
    residual = best @ matrix
    gradient = matrix @ residual
    if float(best @ gradient - gradient.min()) > 1e-8 * max(1.0, loss):
        raise ValueError("Convex projection failed optimality check")
    return best


def metrics(
    raw: np.ndarray, scaled: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    """Report separate pre-fit errors and a certificate's concentration."""
    residual = weights @ scaled
    original = weights @ raw
    return {
        "macro_standardized_rmse": float(np.sqrt(np.mean(residual[:3] ** 2))),
        "outcome_standardized_rmse": float(np.sqrt(np.mean(residual[3:] ** 2))),
        "outcome_rmse_pp": float(np.sqrt(np.mean(original[3:] ** 2))),
        "raw_feature_residuals": original.tolist(),
        "standardized_feature_residuals": residual.tolist(),
        "effective_donors": float(1.0 / (weights @ weights)),
        "max_weight": float(weights.max()),
    }


def audit(
    panel: dict[Key, Observation], controls: dict[Key, MacroValues]
) -> list[dict[str, object]]:
    """Compare outcome-only and equal-block joint pre-fit at h=0 and h=5."""
    _, candidates, _, _ = audit_future_donor_exposure(panel, controls)
    rows: list[dict[str, object]] = []
    for country, year in CASES:
        entries = sorted(
            (c for c in candidates if (c.country, c.reform_year) == (country, year)),
            key=lambda c: c.donor,
        )
        if not entries:
            raise ValueError("Missing original candidate")
        target = profile(panel, controls, country, year, 1, "trajectory")
        full = np.vstack(
            [profile(panel, controls, c.donor, year, 1, "trajectory") for c in entries]
        )
        scale = np.std(np.vstack((target, full)), axis=0)
        scale = np.where(scale < 1e-10, 1.0, scale)
        for horizon in (0, 5):
            keep = [
                i
                for i, c in enumerate(entries)
                if c.first_censored_horizon is None
                or c.first_censored_horizon > horizon
            ]
            raw = full[keep] - target
            scaled = raw / scale
            for mode in ("outcome_only", "equal_blocks"):
                design = scaled[:, 3:] / np.sqrt(5)
                if mode == "equal_blocks":
                    design = np.column_stack((scaled[:, :3] / np.sqrt(3), design))
                weights = project(design)
                loss = float(np.linalg.norm(weights @ design))
                deletion_losses = [
                    float(np.linalg.norm(project(reduced) @ reduced))
                    for i in range(len(keep))
                    if len(keep) > 1
                    for reduced in [np.delete(design, i, axis=0)]
                ]
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "horizon": horizon,
                        "mode": mode,
                        "objective_norm": loss,
                        "worst_deletion_norm": max(deletion_losses, default=None),
                        "donors": [entries[i].donor for i in keep],
                        "weights": weights.tolist(),
                        **metrics(raw, scaled, weights),
                    }
                )
    return rows


def main() -> None:
    """Write source-hashed, strictly pre-treatment fit diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    rows = audit(panel, read_macro_controls(args.panel, set(panel)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "status": "pre_fit_only_no_causal_effect",
        "rows": rows,
    }
    (args.output_dir / "convex_approximation.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
