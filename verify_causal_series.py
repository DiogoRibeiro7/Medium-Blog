"""Run the consolidated reproducibility audit for all 20 causal-inference parts.

Usage from repository root:

    python verify_causal_series.py
    python verify_causal_series.py --recompute

The default checks committed CSV structure and summary identities with the
standard library. Full reproduction requires NumPy, SciPy and scikit-learn.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable

AUDIT_MODULES: tuple[str, ...] = (
    "verify_foundations_results",            # Parts 1–5
    "verify_identification_results",         # Parts 6–10
    "verify_longitudinal_results",           # Parts 11 and 15
    "verify_advanced_results",               # Parts 12–14
    "verify_sensitivity_results",            # Parts 16–20
    "verify_part13_nuisance_validation",      # Separate six-row validation
)


def run(*, recompute: bool) -> None:
    """Run every independent audit; raise on the first failed check."""
    if not isinstance(recompute, bool):
        raise TypeError("recompute must be a boolean")
    for module_name in AUDIT_MODULES:
        module = importlib.import_module(module_name)
        verify: Callable[..., None] | None = getattr(module, "verify", None)
        if not callable(verify):
            raise TypeError(f"{module_name} does not expose callable verify()")
        print(f"Checking {module_name}...", flush=True)
        verify(recompute=recompute)
    print("All 20 main experiments and Part 13 secondary validation passed.")


def main() -> None:
    """Parse audit mode and report verification failures through exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="Rerun all original Monte Carlo experiments in temporary directories",
    )
    arguments = parser.parse_args()
    run(recompute=arguments.recompute)


if __name__ == "__main__":
    main()
