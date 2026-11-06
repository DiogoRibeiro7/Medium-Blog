"""Descriptive replication and sensitivity analysis of GDP and living standards.

These are between-country associations and paired-country changes, *not*
identified effects of changing GDP, redistribution, or public expenditure.
The archival WDI panel is the only input; no live API calls occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


GDP = "NY.GDP.PCAP.PP.KD"
LIFE = "SP.DYN.LE00.IN"
UNDER_FIVE = "SH.DYN.MORT"
ELECTRICITY = "EG.ELC.ACCS.ZS"
WATER = "SH.H2O.SMDW.ZS"
GINI = "SI.POV.GINI"
HEALTH = "SH.XPD.GHED.GD.ZS"
POPULATION = "SP.POP.TOTL"
YEAR = 2023
BASE_YEAR = 2000
OUTCOMES: dict[str, tuple[str, str, str]] = {
    "life_expectancy": (LIFE, "Life expectancy (years)", "level"),
    "under_five_mortality": (UNDER_FIVE, "Under-five mortality / 1,000", "log2"),
    "electricity_access": (ELECTRICITY, "Electricity access (%)", "level"),
    "safely_managed_water": (WATER, "Safely managed water (%)", "level"),
}
REQUIRED_COLUMNS = {
    "country_code", "country_name", "region", "income_group", "year",
    "indicator_id", "value",
}


def load_panel(path: Path) -> pd.DataFrame:
    """Read and validate the original, unreconciled WDI country-year panel."""

    frame = pd.read_csv(path, dtype={"country_code": "string", "indicator_id": "string"})
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED_COLUMNS - set(frame.columns))}")
    if frame.empty or frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Data must be nonempty and contain no nulls in required fields")
    if frame.duplicated(["country_code", "year", "indicator_id"]).any():
        raise ValueError("Duplicate country-year-indicator observations")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite WDI values")
    if frame["year"].min() < 1960 or frame["year"].max() > 2026:
        raise ValueError("Unexpected observation year")
    return frame


def wide_year(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """Pivot annual indicators, preserving one row per genuine country."""

    rows = panel.loc[panel["year"] == year]
    meta = rows.drop_duplicates("country_code")[
        ["country_code", "country_name", "region", "income_group"]
    ].set_index("country_code")
    values = rows.pivot(index="country_code", columns="indicator_id", values="value")
    return meta.join(values, how="inner").reset_index()


def last_reported(panel: pd.DataFrame, indicator: str, first: int, last: int) -> pd.DataFrame:
    """Return latest nonmissing indicator and its actual year within a strict window."""

    rows = panel.loc[
        (panel["indicator_id"] == indicator)
        & (panel["year"].between(first, last))
    ].sort_values(["country_code", "year"])
    rows = rows.drop_duplicates("country_code", keep="last")
    return rows[["country_code", "year", "value"]].rename(
        columns={"value": indicator, "year": f"{indicator}_vintage"}
    )


def hc3_ols(design: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    """OLS with HC3 sandwich intervals; design includes an explicit intercept.

    HC3 describes sampling uncertainty *conditional on the selected design*.
    It is not a correction for omitted variables or cross-country dependence.
    """

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(y) != x.shape[0]:
        raise ValueError("Invalid design matrix/outcome shapes")
    n, p = x.shape
    if n <= p + 2 or np.linalg.matrix_rank(x) != p:
        raise ValueError("Insufficient sample or rank-deficient design")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid nonfinite design or outcome")
    cross_inverse = np.linalg.inv(x.T @ x)
    beta = cross_inverse @ x.T @ y
    residual = y - x @ beta
    hat = np.einsum("ij,jk,ik->i", x, cross_inverse, x)
    if np.any(hat >= 1 - 1e-9):
        raise ValueError("Excessive leverage prevents HC3 inference")
    scaled = residual / (1 - hat)
    sandwich = cross_inverse @ ((x.T * (scaled * scaled)) @ x) @ cross_inverse
    se = np.sqrt(np.maximum(0, np.diag(sandwich)))
    crit = float(t.ppf(0.975, df=n - p))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(n), "parameters": beta.tolist(), "se": se.tolist(),
        "lower": (beta - crit * se).tolist(), "upper": (beta + crit * se).tolist(),
        "r_squared": float(1 - float(np.sum(residual**2)) / ss_total)
        if ss_total > 0 else float("nan"),
    }


def _clean_for_fit(frame: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Select complete cases with positive GDP, never substituting missing data."""

    selected = frame.dropna(subset=indicators).copy()
    if GDP in selected and GDP in indicators:
        selected = selected.loc[selected[GDP] > 0]
    return selected


def _specification(frame: pd.DataFrame, y: np.ndarray, controls: list[str]) -> dict[str, Any]:
    """Fit the GDP coefficient per doubling, with optional additional controls."""

    columns = [np.ones(len(frame)), np.log2(frame[GDP].to_numpy(dtype=float))]
    for name in controls:
        columns.append(frame[name].to_numpy(dtype=float))
    return hc3_ols(np.column_stack(columns), y)


def _wls_slope_per_doubling(frame: pd.DataFrame, y: np.ndarray) -> float | None:
    """Population-weighted *descriptive* slope, without an inferential interval."""

    rows = frame.loc[frame[POPULATION] > 0] if POPULATION in frame else frame.iloc[:0]
    if len(rows) < 15:
        return None
    pos = frame.index.get_indexer(rows.index)
    x = np.column_stack([np.ones(len(rows)), np.log2(rows[GDP].to_numpy())])
    weights = rows[POPULATION].to_numpy(dtype=float)
    weights = weights / np.mean(weights)
    beta = np.linalg.lstsq(x * np.sqrt(weights[:, None]), y[pos] * np.sqrt(weights), rcond=None)[0]
    return float(beta[1])


def cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate four measured outcomes on the same 2023 GDP definition."""

    results: list[dict[str, object]] = []
    for label, (indicator, _name, transform) in OUTCOMES.items():
        sample = _clean_for_fit(frame, [GDP, indicator]).reset_index(drop=True)
        if transform == "log2":
            sample = sample.loc[sample[indicator] > 0].reset_index(drop=True)
        y = sample[indicator].to_numpy(dtype=float)
        transformed = np.log2(y) if transform == "log2" else y
        fit = _specification(sample, transformed, [])
        rho, _ = spearmanr(sample[GDP], y)
        results.append({
            "outcome": label, "year": YEAR, "n_countries": len(sample),
            "y_transform": transform,
            "coefficient_per_gdp_doubling": fit["parameters"][1],
            "hc3_lower": fit["lower"][1], "hc3_upper": fit["upper"][1],
            "r_squared": fit["r_squared"], "spearman_rho": float(rho),
            "population_weighted_coefficient": _wls_slope_per_doubling(sample, transformed),
        })
    return pd.DataFrame(results)


def paired_changes(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe 2000–2023 within-country differences; NOT a panel causal estimator."""

    columns = ["country_code", "country_name", "region", GDP, LIFE, UNDER_FIVE]
    common = old[columns].merge(
        new[columns], on="country_code", suffixes=("_2000", "_2023")
    )
    results: list[dict[str, object]] = []
    scatter_rows: list[pd.DataFrame] = []
    for label, indicator, transform in [
        ("life_expectancy", LIFE, "level"),
        ("under_five_mortality", UNDER_FIVE, "log2"),
    ]:
        fields = [f"{GDP}_2000", f"{GDP}_2023", f"{indicator}_2000", f"{indicator}_2023"]
        sample = common.dropna(subset=fields).copy()
        sample = sample.loc[(sample[fields] > 0).all(axis=1)].copy()
        change_x = np.log2(sample[f"{GDP}_2023"]) - np.log2(sample[f"{GDP}_2000"])
        start = sample[f"{indicator}_2000"].to_numpy(dtype=float)
        end = sample[f"{indicator}_2023"].to_numpy(dtype=float)
        change_y = (np.log2(end) - np.log2(start)) if transform == "log2" else (end - start)
        result = hc3_ols(np.column_stack([np.ones(len(sample)), change_x]), change_y)
        rho, _ = spearmanr(change_x, change_y)
        results.append({
            "outcome": label, "first_year": BASE_YEAR, "last_year": YEAR,
            "n_countries": len(sample), "y_change_transform": transform,
            "slope_per_gdp_doubling": result["parameters"][1],
            "hc3_lower": result["lower"][1], "hc3_upper": result["upper"][1],
            "r_squared": result["r_squared"], "spearman_rho": float(rho),
        })
        scatter_rows.append(pd.DataFrame({
            "country_code": sample["country_code"], "outcome": label,
            "delta_log2_gdp": change_x, "delta_outcome": change_y,
        }))
    return pd.DataFrame(results), pd.concat(scatter_rows, ignore_index=True)


def sensitivity(frame: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit matched-sample model variants, exposing Gini/reporting selection."""

    gini_latest = last_reported(panel, GINI, 2018, YEAR)
    health_latest = last_reported(panel, HEALTH, 2020, YEAR)
    joined = frame.merge(gini_latest, on="country_code", how="left", suffixes=("", "_recent_gini"))
    joined = joined.merge(health_latest, on="country_code", how="left", suffixes=("", "_recent_health"))
    # Exact-year columns have the same names; use recent-vintage aliases explicitly.
    recent_gini = f"{GINI}_recent_gini" if GINI in frame.columns else GINI
    recent_health = f"{HEALTH}_recent_health" if HEALTH in frame.columns else HEALTH
    joined["recent_gini"] = joined[recent_gini]
    joined["recent_health"] = joined[recent_health]
    baseline = _clean_for_fit(frame, [GDP, LIFE])
    matched = _clean_for_fit(joined, [GDP, LIFE, "recent_gini", "recent_health"])
    coverage = pd.DataFrame([{
        "gdp_life_countries": int(len(baseline)),
        "gini_available_2018_2023": int(joined.loc[joined[GDP].notna() & joined[LIFE].notna(), "recent_gini"].notna().sum()),
        "health_available_2020_2023": int(joined.loc[joined[GDP].notna() & joined[LIFE].notna(), "recent_health"].notna().sum()),
        "complete_cases": int(len(matched)),
        "gini_earliest_vintage": int(matched[f"{GINI}_vintage"].min()) if len(matched) else None,
        "health_earliest_vintage": int(matched[f"{HEALTH}_vintage"].min()) if len(matched) else None,
    }])
    if len(matched) < 25:
        raise ValueError(f"Only {len(matched)} observations with both Gini and health expenditure")
    all_model = _specification(baseline, baseline[LIFE].to_numpy(dtype=float), [])
    match_model = _specification(matched, matched[LIFE].to_numpy(dtype=float), [])
    control_model = _specification(
        matched, matched[LIFE].to_numpy(dtype=float), ["recent_gini", "recent_health"]
    )
    models = [
        ("all_countries_gdp_only", all_model),
        ("matched_countries_gdp_only", match_model),
        ("matched_plus_gini_and_health", control_model),
    ]
    estimates = pd.DataFrame([{
        "model": name, "n_countries": result["n"],
        "log2_gdp_coefficient": result["parameters"][1],
        "hc3_lower": result["lower"][1], "hc3_upper": result["upper"][1],
        "r_squared": result["r_squared"],
        "gini_coefficient": result["parameters"][2] if len(result["parameters"]) > 2 else None,
        "health_expenditure_coefficient": result["parameters"][3] if len(result["parameters"]) > 3 else None,
    } for name, result in models])
    return estimates, coverage


def dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify how life expectancy varies among similarly productive countries."""

    out: list[dict[str, object]] = []
    for lower, upper in [(3000, 10000), (10000, 25000), (25000, 50000), (50000, 100000)]:
        selected = frame.loc[(frame[GDP] >= lower) & (frame[GDP] < upper) & frame[LIFE].notna()]
        if len(selected) < 5:
            continue
        years = selected[LIFE].to_numpy(dtype=float)
        out.append({
            "gdp_lower_inclusive": lower, "gdp_upper_exclusive": upper,
            "n_countries": len(selected),
            "life_p10": float(np.quantile(years, 0.1)),
            "life_median": float(np.median(years)),
            "life_p90": float(np.quantile(years, 0.9)),
            "p90_minus_p10_years": float(np.quantile(years, 0.9) - np.quantile(years, 0.1)),
        })
    return pd.DataFrame(out)


def region_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate the country-level relationship by World Bank region."""

    selected = _clean_for_fit(frame, [GDP, LIFE])
    records: list[dict[str, object]] = []
    for name, group in selected.groupby("region"):
        if len(group) < 8:
            continue
        rho, _ = spearmanr(group[GDP], group[LIFE])
        records.append({"region": name, "n_countries": len(group), "spearman_rho": float(rho)})
    return pd.DataFrame(records)


def outcome_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Show the data selection behind every 2023 panel."""

    valid = frame.loc[frame[GDP] > 0]
    return pd.DataFrame([{
        "indicator": label, "gdp_countries": len(valid),
        "complete_country_pairs": int(valid[indicator].notna().sum()),
        "coverage_fraction": float(valid[indicator].notna().mean()),
    } for label, (indicator, _, _) in OUTCOMES.items()])


def _save_figure(fig: Any, path: Path) -> None:
    """Export stable white-background PNG graphics suitable for Medium."""

    fig.savefig(path, dpi=115, facecolor="white", bbox_inches="tight", metadata={"Software": "GDP-Welfare reproducibility study"})
    plt.close(fig)
    # A palette PNG is substantially smaller and remains sharp for web charts.
    try:
        from PIL import Image

        with Image.open(path) as png:
            reduced = png.convert("RGB").quantize(colors=128)
            reduced.save(path, optimize=True)
    except ImportError:
        pass  # A normal RGB PNG is still valid if optional Pillow is absent.


def make_figures(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    cross: pd.DataFrame,
    adjusted: pd.DataFrame,
    coverage: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Generate five independently interpretable, labelled empirical figures."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (label, (indicator, title, transform)) in zip(axes.flat, OUTCOMES.items()):
        subset = _clean_for_fit(frame, [GDP, indicator])
        subset = subset.loc[subset[indicator] > 0] if transform == "log2" else subset
        ax.scatter(subset[GDP], subset[indicator], s=15, alpha=0.7, color="#397ba8", edgecolors="none")
        ax.set_xscale("log")
        if transform == "log2":
            ax.set_yscale("log")
        ax.set_title(f"{title} (n={len(subset)})")
        ax.set_xlabel("GDP per capita, 2023 PPP international $ (log scale)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    fig.suptitle("Four country-level associations, one shared predictor (2023 WDI)", fontsize=14)
    _save_figure(fig, figure_dir / "01_2023_country_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, outcome, ylabel in zip(
        axes, ["life_expectancy", "under_five_mortality"],
        ["Change in life expectancy (years)", "Change in log2 under-five mortality"],
    ):
        subset = paired.loc[paired["outcome"] == outcome]
        ax.scatter(subset["delta_log2_gdp"], subset["delta_outcome"], s=18, alpha=0.6)
        if len(subset) > 2:
            p = np.polyfit(subset["delta_log2_gdp"], subset["delta_outcome"], 1)
            xx = np.linspace(subset["delta_log2_gdp"].min(), subset["delta_log2_gdp"].max(), 100)
            ax.plot(xx, np.polyval(p, xx), color="#d1495b", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"2000–2023 paired-country changes (n={len(subset)})")
        ax.set_xlabel("Change in log2 GDP per capita")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    _save_figure(fig, figure_dir / "02_changes_not_causality.png")

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    sample = _clean_for_fit(frame, [GDP, LIFE])
    xx = np.log2(sample[GDP].to_numpy())
    yy = sample[LIFE].to_numpy()
    beta = np.polyfit(xx, yy, 1)
    residuals = yy - np.polyval(beta, xx)
    ax.scatter(sample[GDP], residuals, s=20, alpha=0.65)
    ax.axhline(0, color="#d1495b", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_title("Variation unexplained by the two-variable GDP fit (2023)")
    ax.set_xlabel("GDP per capita, 2023 PPP international $ (log scale)")
    ax.set_ylabel("Life expectancy residual (years)")
    ax.grid(alpha=0.2)
    _save_figure(fig, figure_dir / "03_life_residuals.png")

    fig, ax = plt.subplots(figsize=(10, 5.1), constrained_layout=True)
    ordered = adjusted.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(ordered))
    vals = ordered["log2_gdp_coefficient"].to_numpy()
    ax.errorbar(
        vals, y, xerr=[vals - ordered["hc3_lower"], ordered["hc3_upper"] - vals],
        fmt="o", color="#295f84", capsize=5, markersize=7,
    )
    ax.set_yticks(y, [x.replace("_", " ") for x in ordered["model"]])
    ax.set_xlabel("GDP coefficient: associated life-years per doubling (95% HC3 interval)")
    ax.set_title("Selection and covariate adjustment change the descriptive coefficient")
    ax.grid(axis="x", alpha=0.2)
    for i, row in ordered.iterrows():
        ax.annotate(f"n={int(row['n_countries'])}", (row["log2_gdp_coefficient"], i), xytext=(10, 10), textcoords="offset points")
    _save_figure(fig, figure_dir / "04_model_sensitivity.png")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    names = [x.replace("_", " ") for x in coverage["indicator"]]
    vals = 100 * coverage["coverage_fraction"].to_numpy()
    bars = ax.barh(names, vals, color="#3c7699", alpha=0.9)
    for bar, frac in zip(bars, vals):
        ax.text(min(frac + 1, 97), bar.get_y() + bar.get_height() / 2, f"{frac:.1f}%", va="center")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Countries with outcome data among those reporting 2023 GDP (%)")
    ax.set_title("Not all published-looking scatter plots have the same country sample")
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, figure_dir / "05_outcome_coverage.png")


def _write_csv(frame: pd.DataFrame, target: Path) -> None:
    """Write stable CSV output with explicit numeric rounding."""

    frame.to_csv(target, index=False, float_format="%.10g")


def run(panel_path: Path, output_dir: Path, figures_dir: Path) -> dict[str, object]:
    """Recompute all results and figures from the captured WDI observation file."""

    panel = load_panel(panel_path)
    current = wide_year(panel, YEAR)
    old = wide_year(panel, BASE_YEAR)
    if len(current) < 150:
        raise ValueError("Too few observed countries in the target year")
    cross = cross_section(current)
    changes, paired = paired_changes(old, current)
    adjusted, covariate_coverage = sensitivity(current, panel)
    dispersions = dispersion(current)
    regions = region_correlations(current)
    coverage = outcome_coverage(current)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("cross_section.csv", cross), ("paired_changes.csv", changes),
        ("adjustment_sensitivity.csv", adjusted),
        ("covariate_coverage.csv", covariate_coverage),
        ("gdp_band_dispersion.csv", dispersions),
        ("regional_spearman.csv", regions), ("outcome_coverage.csv", coverage),
    ]:
        _write_csv(table, output_dir / name)
    make_figures(current, paired, cross, adjusted, coverage, figures_dir)
    summary: dict[str, object] = {
        "data_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "analysis_year": YEAR, "change_start_year": BASE_YEAR,
        "source": "World Bank WDI snapshot; original 2026 OWID chart not directly replicated",
        "number_of_wdi_rows": int(len(panel)),
        "country_count_2023_any_indicator": int(len(current)),
        "outcomes": cross.astype(object).where(pd.notna(cross), None).to_dict(orient="records"),
        "paired_changes": changes.astype(object).where(pd.notna(changes), None).to_dict(orient="records"),
        "sensitivity": adjusted.astype(object).where(pd.notna(adjusted), None).to_dict(orient="records"),
        "coverage": covariate_coverage.astype(object).where(pd.notna(covariate_coverage), None).to_dict(orient="records"),
        "important_caveat": (
            "All estimates are descriptive. No intervention is identified; covariance, "
            "within-country differences and conditioning do not establish causation."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """CLI entry point for deterministic analysis of a fixed snapshot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/wdi_panel.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    result = run(args.panel, args.results_dir, args.figures_dir)
    print(json.dumps({
        "rows": result["number_of_wdi_rows"],
        "countries": result["country_count_2023_any_indicator"],
        "source_sha256": result["data_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
