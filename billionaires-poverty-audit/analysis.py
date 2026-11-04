"""Validate country-level data and audit the billionaire-density/poverty association.

This is a descriptive model, not a causal identification strategy.
Only verified, explicitly supplied CSV data are used. No synthetic input is
silently substituted for absent country observations.

Usage:
    python analysis.py --input data/countries.csv --output results/models.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

Attribution = Literal["residence", "citizenship"]
REQUIRED_COLUMNS: tuple[str, ...] = (
    "country",
    "iso3",
    "billionaires",
    "population",
    "poverty_pct",
    "gdp_ppp_per_capita",
    "poverty_year",
    "gdp_year",
    "billionaire_method",
)


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, Attribution]:
    """Reject invalid or mixed-attribution country records before fitting.

    Raises:
        ValueError: If fields, country identifiers, numeric domains or
            billionaire-attribution types are inconsistent.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")

    data = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    if data.empty:
        raise ValueError("No country rows were supplied")
    for field in ("country", "iso3", "billionaire_method"):
        data[field] = data[field].astype("string").str.strip()
        if data[field].isna().any() or data[field].eq("").any():
            raise ValueError(f"{field} contains empty values")
    if not data["iso3"].str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError("iso3 values must be uppercase three-letter codes")
    if data["iso3"].duplicated().any():
        raise ValueError("Each country ISO3 must occur once per analysis")

    methods = set(data["billionaire_method"].tolist())
    if not methods.issubset({"residence", "citizenship"}) or len(methods) != 1:
        raise ValueError("Provide one and only one valid billionaire_method")
    method: Attribution = "residence" if "residence" in methods else "citizenship"

    numeric_columns = (
        "billionaires",
        "population",
        "poverty_pct",
        "gdp_ppp_per_capita",
        "poverty_year",
        "gdp_year",
    )
    for field in numeric_columns:
        data[field] = pd.to_numeric(data[field], errors="raise").astype("float64")
        if not np.isfinite(data[field].to_numpy(dtype="float64")).all():
            raise ValueError(f"{field} contains non-finite values")

    for field in ("billionaires", "population", "poverty_year", "gdp_year"):
        if not (data[field] % 1 == 0).all():
            raise ValueError(f"{field} must contain integers")
    if (data["billionaires"] < 0).any():
        raise ValueError("billionaires must be non-negative")
    if (data["population"] <= 0).any():
        raise ValueError("population must be strictly positive")
    if not data["poverty_pct"].between(0, 100).all():
        raise ValueError("poverty_pct must be between zero and 100")
    if (data["gdp_ppp_per_capita"] <= 0).any():
        raise ValueError("gdp_ppp_per_capita must be strictly positive")
    if not data["poverty_year"].between(1990, 2030).all():
        raise ValueError("poverty_year outside supported 1990–2030 range")
    if not data["gdp_year"].between(1990, 2030).all():
        raise ValueError("gdp_year outside supported 1990–2030 range")
    return data, method


def model_summary(response: np.ndarray, regressors: pd.DataFrame) -> dict[str, Any]:
    """Fit intercept OLS with heteroskedasticity-robust HC3 inference."""
    if response.ndim != 1 or response.size != len(regressors):
        raise ValueError("Response and regressors have incompatible dimensions")
    if response.size < 4:
        raise ValueError("At least four observations are needed")
    matrix = sm.add_constant(regressors, has_constant="add")
    fit = sm.OLS(response, matrix, missing="raise").fit(cov_type="HC3")
    # Validate finite estimands; do not report unstable/infinite estimates.
    if not np.isfinite(fit.params).all() or not math.isfinite(float(fit.rsquared)):
        raise ValueError("Regression failed to produce finite estimates")
    return {
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "r_squared_adjusted": float(fit.rsquared_adj),
        "coefficients": {key: float(value) for key, value in fit.params.items()},
        "hc3_standard_errors": {key: float(value) for key, value in fit.bse.items()},
        "hc3_p_values": {key: float(value) for key, value in fit.pvalues.items()},
    }


def analyse(frame: pd.DataFrame) -> dict[str, Any]:
    """Analyse positive-density countries on one consistent complete-case sample.

    The positive-density restriction reproduces the domain of log10(B).
    Excluded zero-count countries are counted rather than imputed.
    """
    data, method = validate_frame(frame)
    density = (
        data["billionaires"].to_numpy(dtype=float)
        * 10_000_000.0
        / data["population"].to_numpy(dtype=float)
    )
    positive_mask = density > 0.0
    n_zero = int((~positive_mask).sum())
    active = data.loc[positive_mask].copy()
    if active.shape[0] < 6:
        raise ValueError("Need at least six distinct positive-density countries")
    x_b = np.log10(density[positive_mask])
    x_g = np.log10(active["gdp_ppp_per_capita"].to_numpy(dtype=float))
    y = active["poverty_pct"].to_numpy(dtype=float)
    if np.std(x_b) <= 1e-12 or np.std(x_g) <= 1e-12:
        raise ValueError("Country covariates require nonzero variation")
    raw = model_summary(y, pd.DataFrame({"log10_billionaires_per_10m": x_b}))
    gdp = model_summary(y, pd.DataFrame({"log10_gdp_ppp_pc": x_g}))
    both = model_summary(
        y,
        pd.DataFrame(
            {
                "log10_billionaires_per_10m": x_b,
                "log10_gdp_ppp_pc": x_g,
            }
        ),
    )
    return {
        "evidence": "observed descriptive associations; not causal effects",
        "billionaire_method": method,
        "n_input_countries": int(len(data)),
        "n_zero_billionaire_countries_excluded_from_log_fit": n_zero,
        "n_model_countries": int(len(active)),
        "poverty_year_min": int(active["poverty_year"].min()),
        "poverty_year_max": int(active["poverty_year"].max()),
        "gdp_year_min": int(active["gdp_year"].min()),
        "gdp_year_max": int(active["gdp_year"].max()),
        "models": {"billionaires_only": raw, "gdp_only": gdp, "both": both},
        "incremental_r2_billionaires_conditional_on_gdp": (
            both["r_squared"] - gdp["r_squared"]
        ),
    }


def main() -> None:
    """Run the CSV-driven analysis from a command-line invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Verified country CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination JSON summary"
    )
    arguments = parser.parse_args()
    if not arguments.input.is_file():
        parser.error(f"Verified input CSV is missing: {arguments.input}")
    frame = pd.read_csv(arguments.input)
    summary = analyse(frame)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {summary['n_model_countries']} observed-country fits to {arguments.output}")


if __name__ == "__main__":
    main()
