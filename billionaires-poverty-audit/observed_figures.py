"""Render eight exploratory, provenance-checked Medium figures.

The 54-country panel is an incomplete, secondary Forbes 2026 citizenship
sample. It is NOT a replication of the 64-country residence-based chart.
Run from the study directory:

    python observed_figures.py --input data/observed_citizenship_54_2026.csv \
        --output figures/citizenship_exploratory

All fitted relationships are descriptive. GDP is a diagnostic control, not
an identified causal adjustment. No internet requests are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

FROZEN_SHA256 = "db18001307fc89c77275f9d6147c23ce5f33b9e35e8da640bef82ce725b6fbda"
BILLIONAIRE = "log10_billionaires_per_10m"
GDP = "log10_gdp_ppp_per_capita"
N_COUNTRIES = 54
FIGURES: dict[str, str] = {
    "01_raw_citizenship": "Billionaires by citizenship and poverty: raw association",
    "02_gdp_benchmark": "Economic development is the stronger descriptive benchmark",
    "03_model_fit": "Goodness of fit in the same country sample",
    "04_gdp_partial": "GDP-adjusted partial association",
    "05_specification_curve": "How the coefficient changes under alternative specifications",
    "06_leave_one_out": "Every country omitted in turn",
    "07_poverty_years": "The poverty observations come from different years",
    "08_predictor_relationship": "Development and billionaire density are themselves associated",
}


def load_panel(path: Path) -> pd.DataFrame:
    """Validate the exact observed panel recovered from GitHub Actions.

    Raises:
        ValueError: When source bytes, country counts, attribution or basic
            statistical domains are inconsistent with the frozen sample.
    """
    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(f"Observed panel not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError("Frozen panel SHA-256 mismatch; refuse model calculation")
    frame = pd.read_csv(path).sort_values("iso3").reset_index(drop=True)
    required = {
        "country", "iso3", "billionaires", "population", "poverty_pct",
        "gdp_ppp_per_capita", "poverty_year", "billionaire_method", "forbes_year",
    }
    if required - set(frame.columns):
        raise ValueError(f"Missing panel columns: {sorted(required-set(frame.columns))}")
    if len(frame) != N_COUNTRIES or frame.iso3.duplicated().any():
        raise ValueError("Expected exactly 54 different ISO3 countries")
    if not frame.billionaire_method.eq("citizenship").all():
        raise ValueError("Cannot silently use residence figures")
    if not frame.forbes_year.eq(2026).all():
        raise ValueError("Expected Forbes 2026 citizenship count vintage")
    for col in ("billionaires", "population", "poverty_pct", "gdp_ppp_per_capita", "poverty_year"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {col}")
    if (frame.billionaires <= 0).any() or (frame.population <= 0).any():
        raise ValueError("Log density requires positive count/population")
    if (frame.gdp_ppp_per_capita <= 0).any() or not frame.poverty_pct.between(0, 100).all():
        raise ValueError("Out-of-domain GDP or poverty value")
    frame[BILLIONAIRE] = np.log10(frame.billionaires * 1e7 / frame.population)
    frame[GDP] = np.log10(frame.gdp_ppp_per_capita)
    frame["log_population"] = np.log10(frame.population)
    frame["gdp_squared"] = frame[GDP] ** 2
    return frame


def fit(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit same-sample OLS or population-weighted WLS, with HC3 errors."""
    if len(frame) <= len(predictors) + 4:
        raise ValueError("Insufficient observations")
    X = sm.add_constant(frame[list(predictors)], has_constant="add")
    y = frame.poverty_pct.to_numpy(dtype=float)
    if weights is None:
        result = sm.OLS(y, X).fit(cov_type="HC3")
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(frame),) or not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError("Weights must be finite, positive and row-aligned")
        result = sm.WLS(y, X, weights=w).fit(cov_type="HC3")
    return {
        "n": int(result.nobs),
        "r_squared": float(result.rsquared),
        "model_type": "population-weighted WLS" if weights is not None else "equal-country OLS",
        "coef": {k: float(v) for k, v in result.params.items()},
        "hc3_p": {k: float(v) for k, v in result.pvalues.items()},
        "hc3_ci95": {
            key: [float(x) for x in result.conf_int().loc[key].tolist()]
            for key in result.params.index
        },
    }


def analyse(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute baseline, alternative estimands and 54 exclusion fits."""
    models = {
        "raw": fit(frame, [BILLIONAIRE]),
        "gdp": fit(frame, [GDP]),
        "joint": fit(frame, [BILLIONAIRE, GDP]),
        "population_weighted": fit(frame, [BILLIONAIRE, GDP],
                                   weights=frame.population.to_numpy(dtype=float)),
        "gdp_quadratic": fit(frame, [BILLIONAIRE, GDP, "gdp_squared"]),
        "gdp_population": fit(frame, [BILLIONAIRE, GDP, "log_population"]),
        "poverty_2022_plus": fit(frame.loc[frame.poverty_year >= 2022], [BILLIONAIRE, GDP]),
        "poverty_2023_plus": fit(frame.loc[frame.poverty_year >= 2023], [BILLIONAIRE, GDP]),
    }
    loo = [{
        "excluded_iso3": iso,
        "coef": fit(frame.loc[frame.iso3 != iso], [BILLIONAIRE, GDP])["coef"][BILLIONAIRE],
    } for iso in frame.iso3.tolist()]
    return {
        "disclaimer": "secondary citizenship sample; descriptive, not causal or the original chart",
        "n": len(frame),
        "models": models,
        "incremental_r2": models["joint"]["r_squared"] - models["gdp"]["r_squared"],
        "leave_one_out": loo,
        "poverty_year_counts": {
            str(int(y)): int(n)
            for y, n in frame.poverty_year.value_counts().sort_index().items()
        },
    }


def _new_plot(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Use consistent, spacious dimensions for all figures."""
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    ax.set_title(title, fontsize=16, pad=16)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    return fig, ax


def _note(ax: plt.Axes, detail: str) -> None:
    """Mark each result as an exploratory citizenship comparison."""
    ax.text(0, -0.16, detail, transform=ax.transAxes, fontsize=9, wrap=True)


def _scatter(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    """Plot all countries and identify some reference points."""
    ax.scatter(frame[x], frame[y], alpha=0.78)
    for iso3 in ("USA", "CHN", "IND", "NGA", "CHE", "BRA", "PRT"):
        subset = frame.loc[frame.iso3 == iso3]
        if not subset.empty:
            point = subset.iloc[0]
            ax.annotate(iso3, (point[x], point[y]), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)


def render(
    frame: pd.DataFrame, results: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Save eight PNG/SVG pairs and a verified inventory of figure hashes."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refuse to overwrite prior figures: {output}")
    output.mkdir(parents=True, exist_ok=True)
    models = results["models"]
    plots: list[tuple[str, plt.Figure]] = []

    fig, ax = _new_plot(FIGURES["01_raw_citizenship"])
    _scatter(ax, frame, BILLIONAIRE, "poverty_pct")
    x = np.linspace(frame[BILLIONAIRE].min(), frame[BILLIONAIRE].max(), 100)
    ax.plot(x, models["raw"]["coef"]["const"] + models["raw"]["coef"][BILLIONAIRE]*x)
    ax.set(xlabel="log10(billionaires per 10m, citizenship)",
           ylabel="Share below $30/day, 2021 PPP (%)")
    _note(ax, "54 countries; secondary Forbes citizenship counts; not the original residence chart.")
    plots.append(("01_raw_citizenship", fig))

    fig, ax = _new_plot(FIGURES["02_gdp_benchmark"])
    _scatter(ax, frame, GDP, "poverty_pct")
    x = np.linspace(frame[GDP].min(), frame[GDP].max(), 100)
    ax.plot(x, models["gdp"]["coef"]["const"] + models["gdp"]["coef"][GDP]*x)
    ax.set(xlabel="log10(GDP per capita, PPP)", ylabel="Share below $30/day, PPP (%)")
    _note(ax, "Same 54 countries as the billionaire model, but years of poverty surveys vary.")
    plots.append(("02_gdp_benchmark", fig))

    fig, ax = _new_plot(FIGURES["03_model_fit"])
    keys = ["raw", "gdp", "joint"]
    labels = ["Billionaires", "GDP", "GDP + billionaires"]
    bars = ax.barh(labels, [models[k]["r_squared"] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for b, k in zip(bars, keys):
        ax.text(b.get_width()+0.015, b.get_y()+b.get_height()/2,
                f"{models[k]['r_squared']:.3f}", va="center")
    ax.set_xlabel("In-sample R²")
    _note(ax, "Same n=54 for all models. No causal interpretation follows from R².")
    plots.append(("03_model_fit", fig))

    fig, ax = _new_plot(FIGURES["04_gdp_partial"])
    z = sm.add_constant(frame[[GDP]], has_constant="add")
    xres = sm.OLS(frame[BILLIONAIRE], z).fit().resid.to_numpy()
    yres = sm.OLS(frame.poverty_pct, z).fit().resid.to_numpy()
    ax.scatter(xres, yres, alpha=0.75)
    span = np.linspace(min(xres), max(xres), 100)
    ax.plot(span, models["joint"]["coef"][BILLIONAIRE]*span)
    ax.set(xlabel="Billionaire-density residual after GDP",
           ylabel="Poverty residual after GDP (pp)")
    _note(ax, "Frisch–Waugh–Lovell partial association, not an identified treatment effect.")
    plots.append(("04_gdp_partial", fig))

    fig, ax = _new_plot(FIGURES["05_specification_curve"])
    specifications = [
        ("raw", "Unadjusted"), ("joint", "GDP adjusted"),
        ("population_weighted", "Population weighted"),
        ("gdp_quadratic", "GDP squared"), ("gdp_population", "GDP + log population"),
        ("poverty_2022_plus", "Poverty observed 2022+"),
        ("poverty_2023_plus", "Poverty observed 2023+"),
    ]
    for i, (k, label) in enumerate(specifications):
        value = models[k]["coef"][BILLIONAIRE]
        lo, hi = models[k]["hc3_ci95"][BILLIONAIRE]
        ax.errorbar(value, i, xerr=np.array([[value-lo], [hi-value]]),
                    fmt="o", capsize=4)
    ax.set_yticks(range(len(specifications)), [label for _, label in specifications])
    ax.invert_yaxis()
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Billionaire coefficient, percentage points per 10x; HC3 95% CI")
    _note(ax, "Different models/estimands and sometimes different samples: not causal effects.")
    plots.append(("05_specification_curve", fig))

    fig, ax = _new_plot(FIGURES["06_leave_one_out"])
    ax.hist([r["coef"] for r in results["leave_one_out"]], bins=12, edgecolor="black")
    ax.axvline(models["joint"]["coef"][BILLIONAIRE], linestyle="--")
    ax.set(xlabel="GDP-adjusted billionaire coefficient with one country omitted",
           ylabel="Number of refits")
    _note(ax, "54 leave-one-out refits; uncertainty from shared confounding remains.")
    plots.append(("06_leave_one_out", fig))

    fig, ax = _new_plot(FIGURES["07_poverty_years"])
    years = results["poverty_year_counts"]
    ax.bar(list(years), list(years.values()))
    ax.set(xlabel="OWID observation year (not necessarily survey interview year)",
           ylabel="Countries")
    _note(ax, "Country observations span 2020–2025; Forbes counts describe 2026.")
    plots.append(("07_poverty_years", fig))

    fig, ax = _new_plot(FIGURES["08_predictor_relationship"])
    _scatter(ax, frame, GDP, BILLIONAIRE)
    ax.set(xlabel="log10(GDP per capita, PPP)",
           ylabel="log10(billionaires per 10m, citizenship)")
    _note(ax, "Development is associated with both measured variables in this sample.")
    plots.append(("08_predictor_relationship", fig))

    manifest: dict[str, Any] = {}
    for name, fig in plots:
        png = output / f"{name}.png"
        svg = output / f"{name}.svg"
        fig.savefig(png, dpi=240, bbox_inches="tight")
        fig.savefig(svg, format="svg", bbox_inches="tight")
        plt.close(fig)
        data = png.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG generated: {name}")
        width, height = struct.unpack(">II", data[16:24])
        if width < 2200:
            raise ValueError(f"Figure is below publication width: {name}")
        manifest[name] = {
            "title": FIGURES[name], "png": png.name, "svg": svg.name,
            "width_px": width, "height_px": height,
            "png_sha256": hashlib.sha256(data).hexdigest(),
            "svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
        }
    return manifest


def main() -> None:
    """Validate frozen data, create figures and archive descriptive results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load_panel(args.input)
    results = analyse(frame)
    figures = render(frame, results, args.output)
    report = {
        "source_run": 41866282683,
        "dataset_sha256": FROZEN_SHA256,
        "data_status": "secondary citizenship counts; not original chart",
        "results": results,
        "figures": figures,
    }
    (args.output/"figure_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    print(f"Generated {len(figures)} figure pairs from {len(frame)} countries")


if __name__ == "__main__":
    main()
