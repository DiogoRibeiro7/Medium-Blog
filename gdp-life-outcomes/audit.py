"""Check that regenerated numerical evidence matches the archived WDI study.

This tool deliberately ignores PNG byte identity because installed rendering
fonts can vary. It checks the numeric tables and archived data fingerprint,
and requires every publication figure to exist and be nonempty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_FILES = (
    "cross_section.csv", "paired_changes.csv", "adjustment_sensitivity.csv",
    "covariate_coverage.csv", "gdp_band_dispersion.csv", "regional_spearman.csv",
    "outcome_coverage.csv",
)
FIGURES = (
    "01_2023_country_scatter.png", "02_changes_not_causality.png",
    "03_life_residuals.png", "04_model_sensitivity.png",
    "05_outcome_coverage.png",
)


def compare_csv(original: Path, generated: Path) -> None:
    """Compare ordered output tables and numeric cells with roundoff tolerance."""

    left = pd.read_csv(original)
    right = pd.read_csv(generated)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        raise ValueError(f"Changed table shape or columns: {original.name}")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            a = left[column].to_numpy(dtype=float)
            b = right[column].to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True):
                raise ValueError(f"Numerical discrepancy in {original.name}: {column}")
        elif not left[column].fillna("<MISSING>").equals(right[column].fillna("<MISSING>")):
            raise ValueError(f"Text/categorical discrepancy in {original.name}: {column}")


def verify(reference: Path, candidate: Path, panel: Path, figures: Path) -> None:
    """Audit data provenance and the independent regeneration of results."""

    for filename in RESULT_FILES:
        compare_csv(reference / filename, candidate / filename)
    provenance = json.loads((panel.parent / "provenance.json").read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
    if actual_hash != provenance["panel_sha256"]:
        raise ValueError("Archived WDI data do not match their declared SHA-256")
    summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
    if summary["data_sha256"] != actual_hash:
        raise ValueError("Generated results use a different WDI panel")
    for filename in FIGURES:
        path = figures / filename
        if not path.is_file() or path.stat().st_size < 2000:
            raise ValueError(f"Missing/empty figure: {path}")


def main() -> None:
    """Run the audit from a CI job or local shell."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results"))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--panel", type=Path, default=Path("data/wdi_panel.csv"))
    parser.add_argument("--figures", required=True, type=Path)
    args = parser.parse_args()
    verify(args.reference, args.candidate, args.panel, args.figures)
    print("All archived numerical results and provenance checks passed")


if __name__ == "__main__":
    main()
