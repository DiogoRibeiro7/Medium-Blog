"""Audit mixed poverty-survey vintages in the exploratory 54-country panel.

This is an observational citizenship-based sensitivity analysis, not a
reconstruction of the original residence-based 64-country figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observed_figures import BILLIONAIRE, GDP, fit, load_panel

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "data" / "observed_citizenship_54_2026.csv"


def audit(frame: pd.DataFrame, *, min_countries: int = 12) -> dict[str, Any]:
    """Estimate survey-age sensitivity with clearly exposed sample attrition."""
    required = {"iso3", "poverty_year", "gdp_year", "population", BILLIONAIRE, GDP, "poverty_pct"}
    if not isinstance(frame, pd.DataFrame) or required - set(frame.columns):
        raise ValueError("Input requires the validated regression-panel columns")
    if not isinstance(min_countries, int) or isinstance(min_countries, bool) or min_countries < 8:
        raise ValueError("min_countries must be an integer >= 8")
    if frame["iso3"].duplicated().any() or frame[list(required - {"iso3"})].isna().any().any():
        raise ValueError("Duplicate countries or missing model values")
    if (frame["poverty_year"] > frame["gdp_year"]).any():
        raise ValueError("Poverty survey cannot be newer than paired GDP observation")
    if not np.isfinite(frame[["poverty_year", "gdp_year"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite source years")
    data = frame.copy()
    data["survey_lag"] = data["gdp_year"] - data["poverty_year"]
    if (data["survey_lag"] < 0).any():
        raise ValueError("Negative source-year lag")
    distribution = [
        {"poverty_year": int(year), "countries": int(len(group)),
         "population": int(group["population"].sum())}
        for year, group in data.groupby("poverty_year", sort=True)
    ]
    base = fit(data, [BILLIONAIRE, GDP])
    models: list[dict[str, Any]] = []
    for max_lag in sorted({int(value) for value in data["survey_lag"]}):
        subset = data.loc[data["survey_lag"] <= max_lag].copy()
        record: dict[str, Any] = {
            "max_years_between_poverty_and_gdp": max_lag,
            "n": len(subset),
            "excluded_iso3": sorted(set(data["iso3"]) - set(subset["iso3"])),
            "fitted": len(subset) >= min_countries,
        }
        if record["fitted"]:
            model = fit(subset, [BILLIONAIRE, GDP])
            record.update({
                "billionaire_coef_pp_per_log10": model["coef"][BILLIONAIRE],
                "hc3_p": model["hc3_p"][BILLIONAIRE],
                "hc3_ci95": model["hc3_ci95"][BILLIONAIRE],
                "r_squared": model["r_squared"],
            })
        models.append(record)
    data["survey_lag_centered"] = data["survey_lag"] - data["survey_lag"].mean()
    continuous = fit(data, [BILLIONAIRE, GDP, "survey_lag_centered"])
    return {
        "classification": "exploratory_citizenship_not_causal",
        "n": len(data),
        "poverty_year_distribution": distribution,
        "min_poverty_year": int(data["poverty_year"].min()),
        "max_poverty_year": int(data["poverty_year"].max()),
        "max_survey_lag": int(data["survey_lag"].max()),
        "baseline": base,
        "survey_lag_adjusted": continuous,
        "maximum_lag_restrictions": models,
        "warning": (
            "Different survey years and selected-country composition prevent "
            "a fixed-time causal comparison; threshold estimates use changing samples."
        ),
    }


def write_results(results: dict[str, Any], directory: Path) -> None:
    """Write reproducible metadata without overwriting an existing output."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    records = results["maximum_lag_restrictions"]
    fields = ["max_years_between_poverty_and_gdp", "n", "fitted",
              "billionaire_coef_pp_per_log10", "hc3_p", "r_squared", "excluded_iso3"]
    with (directory / "lag_sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: ",".join(record[key]) if key == "excluded_iso3"
                             else record.get(key, "") for key in fields})


def main() -> None:
    """Run the frozen-panel survey-vintage audit offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = audit(load_panel(args.input))
    write_results(results, args.output)
    print(json.dumps({
        "n": results["n"],
        "poverty_year_range": [results["min_poverty_year"], results["max_poverty_year"]],
        "max_survey_lag": results["max_survey_lag"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
