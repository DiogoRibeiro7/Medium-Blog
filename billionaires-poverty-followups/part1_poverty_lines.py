"""Part 1: billionaire density across poverty lines and survey types.

Reproduces every number and figure in the first follow-up article. Inputs:

* the SHA-256-locked 54-country citizenship panel from the first article
  (``../billionaires-poverty-audit/data/observed_citizenship_54_2026.csv``);
* World Bank PIP survey estimates fetched by ``fetch_pip.py`` (release
  20260922, 2021 PPP, survey years only), hash-checked against
  ``data/pip_manifest.json``.

Each panel country is matched to the PIP survey estimate that reproduces its
$30 headcount (same year, national coverage, same welfare type), so every
poverty line describes the same survey the first article used.

Usage, from this directory::

    python part1_poverty_lines.py          # compute, print, write results and figures
    python part1_poverty_lines.py --check  # fail if any number differs from results/part1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "billionaires-poverty-audit"
sys.path.insert(0, str(AUDIT))
from observed_figures import BILLIONAIRE as B, GDP as G, load_panel  # noqa: E402
from stats_tools import fractional_logit, loo_rmse_fractional, loo_rmse_ols, ols_hc3  # noqa: E402

PANEL = AUDIT / "data" / "observed_citizenship_54_2026.csv"
DATA = HERE / "data"
RESULTS = HERE / "results" / "part1"
FIGURES = HERE / "figures" / "part1"
REL_TOL = 1e-6
BASE_LINE = 30.0
KEY = ["country_code", "reporting_year", "welfare_type", "reporting_level"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE, PALE = "#e1e0d9", "#c3c2b7", "#ffffff", "#b7d3f6"
SOURCE = ("Data: World Bank PIP, release 20260922 (2021 PPP, survey years); Forbes 2026 billionaires by "
          "citizenship (secondary transcription); WDI GDP per person (PPP). 54 countries; exploratory, not causal.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pip(data: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the PIP tables after checking their hashes against the manifest."""
    manifest = json.loads((data / "pip_manifest.json").read_text(encoding="utf-8"))
    for name, key in (("pip_surveys.csv.gz", "pip_surveys_sha256"), ("pip_headcounts.csv.gz", "pip_headcounts_sha256")):
        if _sha(data / name) != manifest[key]:
            raise ValueError(f"{name} does not match the manifest hash")
    return pd.read_csv(data / "pip_surveys.csv.gz"), pd.read_csv(data / "pip_headcounts.csv.gz"), manifest


def build_panel(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each panel country to its PIP survey and attach every poverty line."""
    at30 = heads[(heads.poverty_line == BASE_LINE) & (heads.reporting_level == "national")]
    cand = panel.merge(at30, left_on=["iso3", "poverty_year"], right_on=["country_code", "reporting_year"])
    cand["gap"] = (100 * cand.headcount - cand.poverty_pct).abs()
    chosen = cand.loc[cand.gap < 1e-4].sort_values(["iso3", "welfare_type"]).groupby("iso3").head(1)
    if len(chosen) != len(panel):
        raise ValueError(f"matched {len(chosen)} of {len(panel)} panel countries to a PIP survey")
    keys = chosen[KEY]
    base = panel.merge(keys, left_on="iso3", right_on="country_code").merge(surveys, on=KEY)
    base["consumption"] = (base.welfare_type == "consumption").astype(float)
    base["log10_mean"] = np.log10(base["mean"])
    base["log10_median"] = np.log10(base["median"])
    base["bottom40"] = base[[f"decile{k}" for k in range(1, 5)]].sum(axis=1)
    for k in range(1, 11):
        base[f"log10_decile{k}"] = np.log10(10 * base[f"decile{k}"] * base["mean"])
    base = base.sort_values("iso3").reset_index(drop=True)
    long = heads.merge(keys, on=KEY)[["country_code", "poverty_line", "headcount"]]
    wide = long.pivot(index="country_code", columns="poverty_line", values="headcount").mul(100)
    wide = wide.loc[base.iso3]
    return base, wide


def at_line(base: pd.DataFrame, wide: pd.DataFrame, line: float) -> pd.DataFrame:
    frame = base.copy()
    frame["poverty_pct"] = wide[line].to_numpy()
    return frame


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def sweep(base: pd.DataFrame, wide: pd.DataFrame) -> list[dict[str, Any]]:
    """Models re-estimated at every poverty line."""
    rows = []
    for line in wide.columns:
        f = at_line(base, wide, float(line))
        raw, gdp, joint = (ols_hc3(f, "poverty_pct", c) for c in ([B], [G], [B, G]))
        jc = ols_hc3(f, "poverty_pct", [B, G, "consumption"])
        fg = fractional_logit(f, "poverty_pct", [B, G], B)
        fc = fractional_logit(f, "poverty_pct", [B, G, "consumption"], B)
        fm = fractional_logit(f, "poverty_pct", [B, "log10_mean"], B)
        rows.append({
            "line": float(line),
            "n_below_1pct": int((f.poverty_pct < 1).sum()),
            "n_above_99pct": int((f.poverty_pct > 99).sum()),
            "mean_headcount": float(f.poverty_pct.mean()),
            "r2_billionaire": raw["r_squared"], "r2_gdp": gdp["r_squared"], "r2_joint": joint["r_squared"],
            "incremental_r2": joint["r_squared"] - gdp["r_squared"],
            "raw_slope": raw["coef"][B],
            "ols_gdp_coef": joint["coef"][B], "ols_gdp_p": joint["p"][B],
            "ols_gdp_type_coef": jc["coef"][B], "ols_gdp_type_p": jc["p"][B],
            "frac_gdp_ame": fg["ame"], "frac_gdp_ci95": fg["ci95"],
            "frac_gdp_type_ame": fc["ame"], "frac_gdp_type_ci95": fc["ci95"],
            "frac_mean_ame": fm["ame"], "frac_mean_ci95": fm["ci95"],
            "loo_frac_mean": loo_rmse_fractional(f, "poverty_pct", ["log10_mean"]),
            "loo_frac_mean_b": loo_rmse_fractional(f, "poverty_pct", [B, "log10_mean"]),
            "loo_frac_gdp": loo_rmse_fractional(f, "poverty_pct", [G]),
            "loo_frac_gdp_b": loo_rmse_fractional(f, "poverty_pct", [B, G]),
            "loo_frac_gdp_type": loo_rmse_fractional(f, "poverty_pct", [G, "consumption"]),
            "loo_frac_gdp_type_b": loo_rmse_fractional(f, "poverty_pct", [B, G, "consumption"]),
        })
    return rows


def welfare_pairs(surveys: pd.DataFrame, heads: pd.DataFrame) -> dict[str, Any]:
    """Country-years in which PIP reports both an income and a consumption survey."""
    nat = surveys[surveys.reporting_level == "national"]
    means = nat.pivot_table(index=["country_code", "reporting_year"], columns="welfare_type", values="mean").dropna()
    ratio = means.consumption / means.income
    out: dict[str, Any] = {
        "n_pairs": int(len(ratio)),
        "n_countries": int(means.index.get_level_values(0).nunique()),
        "countries": sorted(means.index.get_level_values(0).unique().tolist()),
        "year_min": int(means.index.get_level_values(1).min()),
        "year_max": int(means.index.get_level_values(1).max()),
        "mean_ratio_mean": float(ratio.mean()),
        "mean_ratio_median": float(ratio.median()),
        "share_consumption_lower": float((ratio < 1).mean()),
    }
    for line in (8.3, 30.0):
        hh = heads[(heads.reporting_level == "national") & (heads.poverty_line == line)]
        hh = hh.pivot_table(index=["country_code", "reporting_year"], columns="welfare_type", values="headcount").dropna()
        diff = 100 * (hh.consumption - hh.income)
        out[f"headcount_gap_{line:g}"] = {"mean": float(diff.mean()), "median": float(diff.median()),
                                          "min": float(diff.min()), "max": float(diff.max())}
    return out


def distribution(base: pd.DataFrame) -> dict[str, Any]:
    """Billionaire coefficients on welfare levels and shares, given GDP and survey type."""
    ctrl = [B, G, "consumption"]
    out: dict[str, Any] = {}
    for y in ("log10_mean", "log10_median", "gini", "decile10", "bottom40"):
        fit = ols_hc3(base, y, ctrl)
        out[y] = {"coef": fit["coef"][B], "p": fit["p"][B], "ci95": fit["ci95"][B], "r_squared": fit["r_squared"],
                  "consumption_coef": fit["coef"]["consumption"]}
    out["deciles"] = []
    for k in range(1, 11):
        fit = ols_hc3(base, f"log10_decile{k}", ctrl)
        out["deciles"].append({"decile": k, "coef": fit["coef"][B], "p": fit["p"][B], "ci95": fit["ci95"][B]})
    out["mean_ratio_by_type"] = {
        t: float((g["mean"] * 365 / g.gdp_ppp_per_capita).median()) for t, g in base.groupby("welfare_type")}
    return out


def given_survey_mean(base: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    """Poverty at $30 explained by the survey mean, and what billionaire density adds."""
    f = at_line(base, wide, BASE_LINE)
    only = ols_hc3(f, "poverty_pct", ["log10_mean"])
    both = ols_hc3(f, "poverty_pct", [B, "log10_mean"])
    frac = fractional_logit(f, "poverty_pct", [B, "log10_mean"], B)
    return {
        "r2_survey_mean_only": only["r_squared"], "r2_with_billionaires": both["r_squared"],
        "ols_coef": both["coef"][B], "ols_p": both["p"][B], "ols_ci95": both["ci95"][B],
        "frac_ame": frac["ame"], "frac_ci95": frac["ci95"],
        "loo_frac_mean": loo_rmse_fractional(f, "poverty_pct", ["log10_mean"]),
        "loo_frac_mean_b": loo_rmse_fractional(f, "poverty_pct", [B, "log10_mean"]),
    }


def at_thirty(base: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    """Survey-type results at the first article's $30 line."""
    f = at_line(base, wide, BASE_LINE)
    jc = ols_hc3(f, "poverty_pct", [B, G, "consumption"])
    fc = fractional_logit(f, "poverty_pct", [B, G, "consumption"], B)
    split = {}
    for t, g in f.groupby("welfare_type"):
        fit = ols_hc3(g, "poverty_pct", [B, G])
        split[t] = {"n": fit["n"], "coef": fit["coef"][B], "p": fit["p"][B], "r_squared": fit["r_squared"]}
    return {
        "n_income": int((f.welfare_type == "income").sum()),
        "n_consumption": int((f.welfare_type == "consumption").sum()),
        "consumption_countries": sorted(f.loc[f.welfare_type == "consumption", "iso3"].tolist()),
        "ols_type": {"coef": jc["coef"][B], "p": jc["p"][B], "ci95": jc["ci95"][B],
                     "consumption_coef": jc["coef"]["consumption"], "consumption_p": jc["p"]["consumption"],
                     "r_squared": jc["r_squared"]},
        "frac_type": fc,
        "loo_ols_gdp_type": loo_rmse_ols(f, "poverty_pct", [G, "consumption"]),
        "loo_ols_gdp_type_b": loo_rmse_ols(f, "poverty_pct", [B, G, "consumption"]),
        "split": split,
    }


def income_alternative(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame) -> dict[str, Any]:
    """Replace the Philippine and Russian consumption surveys by their same-year income surveys."""
    base, wide = build_panel(panel, surveys, heads)
    f = at_line(base, wide, BASE_LINE)
    for iso in ("PHL", "RUS"):
        year = int(base.loc[base.iso3 == iso, "reporting_year"].iloc[0])
        row = heads[(heads.country_code == iso) & (heads.reporting_year == year) & (heads.welfare_type == "income")
                    & (heads.reporting_level == "national") & (heads.poverty_line == BASE_LINE)]
        f.loc[f.iso3 == iso, "poverty_pct"] = 100 * float(row.headcount.iloc[0])
        f.loc[f.iso3 == iso, "consumption"] = 0.0
    joint = ols_hc3(f, "poverty_pct", [B, G])
    frac = fractional_logit(f, "poverty_pct", [B, G], B)
    return {"ols_gdp_coef": joint["coef"][B], "ols_gdp_p": joint["p"][B], "frac_gdp_ame": frac["ame"],
            "frac_gdp_ci95": frac["ci95"], "PHL_income_30": float(f.loc[f.iso3 == "PHL", "poverty_pct"].iloc[0]),
            "RUS_income_30": float(f.loc[f.iso3 == "RUS", "poverty_pct"].iloc[0])}


def compute(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    base, wide = build_panel(panel, surveys, heads)
    rows = sweep(base, wide)
    return {
        "pip_release": next(v["release_version"] for v in manifest["versions"] if v["ppp_version"] == "2021"),
        "pip_retrieved_utc": manifest["retrieved_utc"],
        "pip_survey_estimates": int(len(surveys)),
        "lines": [float(c) for c in wide.columns],
        "n": int(len(base)),
        "max_match_gap_pp": float((wide[BASE_LINE] - panel.set_index("iso3").loc[base.iso3, "poverty_pct"].to_numpy()).abs().max()),
        "sweep": rows,
        "at_thirty": at_thirty(base, wide),
        "income_alternative": income_alternative(panel, surveys, heads),
        "distribution": distribution(base),
        "given_survey_mean": given_survey_mean(base, wide),
        "welfare_pairs": welfare_pairs(surveys, heads),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fmt(value: float, digits: int = 1, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": SECONDARY, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 10, "text.parse_math": False,
    })


def _frame(fig: plt.Figure, title: str, subtitle: str, top: float = 0.86, bottom: float = 0.17,
           left: float = 0.08, right: float = 0.97, source: str = SOURCE) -> None:
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=11, color=SECONDARY, va="top", linespacing=1.35)
    fig.text(0.012, 0.012, source, fontsize=8, color=MUTED, va="bottom", wrap=True)
    fig.subplots_adjust(top=top - 0.045 * subtitle.count("\n"), bottom=bottom, left=left, right=right)


def _line_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ticks = [2, 3, 5, 8.3, 15, 30, 50, 100]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlim(1.85, 108)


def _official_lines(ax: plt.Axes, top: float, labels: bool = True) -> None:
    for line, text in ((3.0, "$3"), (8.3, "$8.30"), (30.0, "$30")):
        ax.axvline(line, color=BASELINE, lw=1, zorder=1)
        if labels:
            ax.text(line * 1.03, top, text, fontsize=9, color=MUTED, va="top")


def fig_curves(base: pd.DataFrame, wide: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    lines = np.array(wide.columns, dtype=float)
    highlight = {"NGA": "Nigeria", "IND": "India", "CHN": "China", "BRA": "Brazil", "PRT": "Portugal", "USA": "United States"}
    for iso, row in wide.iterrows():
        if iso not in highlight:
            ax.plot(lines, row.to_numpy(), color=PALE, lw=1.1, zorder=2)
    for iso, name in highlight.items():
        y = wide.loc[iso].to_numpy()
        ax.plot(lines, y, color=BLUE, lw=2, zorder=3)
        k = int(np.searchsorted(lines, {"NGA": 4.2, "IND": 6.0, "CHN": 17.5, "BRA": 25.0, "PRT": 40.0, "USA": 70.0}[iso]))
        ax.scatter([lines[k]], [y[k]], color=BLUE, s=26, zorder=4)
        ax.text(lines[k] * 1.05, y[k] - 1.0, name, fontsize=9, color=SECONDARY, va="top")
    _official_lines(ax, 104)
    _line_axis(ax)
    ax.set_ylim(-2, 106)
    ax.set_xlabel("Poverty line, 2021 PPP dollars per person per day (log scale)")
    ax.set_ylabel("Share of population below the line (%)")
    _frame(fig, "Each country is a curve, and a poverty line picks one point on it",
           "Share below each line for the 54 countries, from the same survey the first article used at $30.\n"
           "Pale lines: the other 48 countries.")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_r2(n: dict[str, Any], out: Path) -> None:
    rows = n["sweep"]
    x = [r["line"] for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for key, colour, label in (("r2_gdp", BLUE, "GDP per person only"), ("r2_billionaire", ORANGE, "Billionaire density only")):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=4.5, zorder=3)
        ax.text(x[-1] * 1.03, y[-1], label, fontsize=9.5, color=SECONDARY, va="center")
    _official_lines(ax, 0.99)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R² across 54 countries")
    ax2.plot(x, [r["n_below_1pct"] for r in rows], color=BLUE, lw=2, zorder=3)
    ax2.plot(x, [r["n_above_99pct"] for r in rows], color=ORANGE, lw=2, zorder=3)
    ax2.text(6.3, 22, "countries with under 1% below the line", fontsize=9, color=SECONDARY)
    ax2.text(40, 22, "countries with over 99% below the line", fontsize=9, color=SECONDARY)
    _official_lines(ax2, 0, labels=False)
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("Countries")
    for a in (ax, ax2):
        _line_axis(a)
    ax.set_xlim(1.85, 200)
    ax2.set_xlabel("Poverty line, 2021 PPP dollars per day (log scale)")
    _frame(fig, "GDP explains more than billionaire density at every poverty line",
           "Top: R² of the poverty share on one variable. Bottom: how many countries sit at the floor or the ceiling.",
           top=0.87, right=0.83)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_adjusted(n: dict[str, Any], out: Path) -> None:
    rows = n["sweep"]
    x = np.array([r["line"] for r in rows])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1.15]})
    for key, colour, label in (("frac_gdp", BLUE, "controlling for GDP"),
                               ("frac_gdp_type", ORANGE, "controlling for GDP and survey type")):
        est = np.array([r[f"{key}_ame"] for r in rows])
        lo = np.array([r[f"{key}_ci95"][0] for r in rows])
        hi = np.array([r[f"{key}_ci95"][1] for r in rows])
        if key == "frac_gdp_type":
            ax.fill_between(x, lo, hi, color=colour, alpha=0.14, lw=0)
        ax.plot(x, est, color=colour, lw=2, marker="o", ms=4, label=label, zorder=3)
    ax.axhline(0, color=BASELINE, lw=1.2)
    _official_lines(ax, 7.6)
    ax.set_ylabel("Points per tenfold density\n(average marginal effect)")
    ax.legend(loc="lower left")
    ax.set_ylim(-16, 8)
    for key, colour in (("gdp", BLUE), ("gdp_type", ORANGE)):
        d = [r[f"loo_frac_{key}_b"] - r[f"loo_frac_{key}"] for r in rows]
        ax2.plot(x, d, color=colour, lw=2, marker="o", ms=4, zorder=3)
    ax2.axhline(0, color=BASELINE, lw=1.2)
    _official_lines(ax2, 0, labels=False)
    ax2.set_ylabel("Change in out-of-\nsample error (points)")
    ax2.text(1.95, 0.52, "above zero: adding billionaires makes prediction worse", fontsize=9, color=SECONDARY)
    ax2.set_ylim(-1.4, 0.7)
    for a in (ax, ax2):
        _line_axis(a)
    ax2.set_xlabel("Poverty line, 2021 PPP dollars per day (log scale)")
    _frame(fig, "After GDP, the billionaire association is small at every line",
           "Fractional logit, 54 countries; band: 95% interval of the model with survey type. Bottom: leave-\n"
           "one-country-out error with billionaire density minus error without it.", top=0.87, left=0.13)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_pairs(surveys: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    nat = surveys[surveys.reporting_level == "national"]
    means = nat.pivot_table(index=["country_code", "reporting_year"], columns="welfare_type", values="mean").dropna()
    fig, ax = plt.subplots(figsize=(9, 6.0))
    lim = [3, 50]
    ax.plot(lim, lim, color=BASELINE, lw=1.2, zorder=1)
    ax.text(26, 43, "grey line: equal means", fontsize=9, color=MUTED)
    ax.scatter(means.income, means.consumption, color=BLUE, s=40, edgecolors=SURFACE, linewidths=1, zorder=3)
    latest = means.reset_index().sort_values("reporting_year").groupby("country_code").tail(1)
    for _, r in latest.iterrows():
        if r.country_code in {"POL", "RUS", "PHL", "NIC", "HTI", "LCA"}:
            ax.annotate(f"{r.country_code} {r.reporting_year}", (r.income, r.consumption), xytext=(5, -10),
                        textcoords="offset points", fontsize=8.5, color=SECONDARY)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator([3, 5, 10, 20, 40]))
        a.set_minor_locator(NullLocator())
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator([3, 5, 10, 20, 40]))
        a.set_minor_locator(NullLocator())
        a.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Income survey: mean per person per day (2021 PPP, log scale)")
    ax.set_ylabel("Consumption survey, same country and year")
    p = n["welfare_pairs"]
    _frame(fig, "In the same country and year, consumption surveys usually record less",
           f"{p['n_pairs']} country-years in {p['n_countries']} countries with both survey types ({p['year_min']}–{p['year_max']}). "
           f"Median ratio {p['mean_ratio_median']:.2f}.\nLabels mark the latest pair for selected countries.",
           left=0.10, source="Data: World Bank PIP, release 20260922 (2021 PPP, survey years), national estimates; every "
           "country-year with both an income and a consumption survey.")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_deciles(n: dict[str, Any], out: Path) -> None:
    dec = n["distribution"]["deciles"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.array([d["decile"] for d in dec])
    est = np.array([d["coef"] for d in dec])
    lo = np.array([d["ci95"][0] for d in dec])
    hi = np.array([d["ci95"][1] for d in dec])
    pct = lambda v: 100 * (10**v - 1)  # noqa: E731
    for xi, a, b in zip(x, lo, hi):
        ax.plot([xi, xi], [pct(a), pct(b)], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(x, pct(est), color=BLUE, s=58, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.axhline(0, color=BASELINE, lw=1.2)
    ax.set_xticks(x, ["Poorest\n10%"] + [str(k) for k in range(2, 10)] + ["Richest\n10%"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Difference in average welfare per tenfold\nbillionaire density (%), given GDP and survey type")
    d = n["distribution"]
    ax.text(0.99, 0.97, f"Gini: {fmt(d['gini']['coef'], 4, sign=True)} (p = {d['gini']['p']:.2f})\n"
            f"Bottom-40% share: {fmt(100 * d['bottom40']['coef'], 2, sign=True)} points (p = {d['bottom40']['p']:.2f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=SECONDARY)
    _frame(fig, "The association shifts every decile by about the same amount",
           "Each point is a separate regression of log average welfare in one decile. Lines: HC3 95% intervals.",
           left=0.13)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_given_mean(base: pd.DataFrame, wide: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    f = at_line(base, wide, BASE_LINE)
    import statsmodels.api as sm
    X = sm.add_constant(f[["log10_mean"]])
    fit = sm.GLM(f.poverty_pct / 100, X, family=sm.families.Binomial()).fit()
    resid = f.poverty_pct - 100 * fit.predict(X)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5.2))
    for t, colour in (("income", BLUE), ("consumption", ORANGE)):
        g = f[f.welfare_type == t]
        ax.scatter(g["mean"], g.poverty_pct, color=colour, s=44, edgecolors=SURFACE, linewidths=1, zorder=3, label=f"{t} survey")
        ax2.scatter(10 ** g[B], resid[g.index], color=colour, s=44, edgecolors=SURFACE, linewidths=1, zorder=3)
    xs = np.logspace(np.log10(f["mean"].min()), np.log10(f["mean"].max()), 200)
    ax.plot(xs, 100 * fit.predict(sm.add_constant(pd.DataFrame({"log10_mean": np.log10(xs)}))), color=INK, lw=2, zorder=2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlabel("Survey mean per person per day (log scale)")
    ax.set_ylabel("Share below $30 a day (%)")
    ax.legend(loc="lower left")
    ax2.axhline(0, color=BASELINE, lw=1.2)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_locator(FixedLocator([0.2, 1, 5, 20, 50]))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.set_xlabel("Billionaires per 10 million people (log scale)")
    ax2.set_ylabel("Unexplained poverty (points)")
    g = n["given_survey_mean"]
    ax.set_title(f"Survey mean alone: R² {g['r2_survey_mean_only']:.2f}", fontsize=10.5, color=INK, loc="left")
    ax2.set_title(f"Billionaire slope {fmt(g['ols_coef'])} (p = {g['ols_p']:.2f})", fontsize=10.5, color=INK, loc="left")
    _frame(fig, "Once the survey mean is known, billionaire density says nothing about poverty",
           "Left: poverty at $30 against the survey's average welfare, with a fractional-logit curve.\n"
           "Right: what that curve leaves unexplained, against billionaire density.", top=0.83, bottom=0.19)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = ("fig1_poverty_curves.png", "fig2_r2_by_line.png", "fig3_adjusted_by_line.png",
                "fig4_survey_type_pairs.png", "fig5_decile_shift.png", "fig6_given_survey_mean.png")


def render(panel: pd.DataFrame, surveys: pd.DataFrame, heads: pd.DataFrame, n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    base, wide = build_panel(panel, surveys, heads)
    fig_curves(base, wide, outdir / FIGURE_FILES[0])
    fig_r2(n, outdir / FIGURE_FILES[1])
    fig_adjusted(n, outdir / FIGURE_FILES[2])
    fig_pairs(surveys, n, outdir / FIGURE_FILES[3])
    fig_deciles(n, outdir / FIGURE_FILES[4])
    fig_given_mean(base, wide, n, outdir / FIGURE_FILES[5])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_lines(n: dict[str, Any]) -> list[str]:
    s = {r["line"]: r for r in n["sweep"]}
    t, d, g, w, alt = n["at_thirty"], n["distribution"], n["given_survey_mean"], n["welfare_pairs"], n["income_alternative"]
    lines = [f"PIP release {n['pip_release']} retrieved {n['pip_retrieved_utc']}; {n['pip_survey_estimates']} survey estimates; "
             f"{n['n']} countries matched (max $30 gap {n['max_match_gap_pp']:.2e} pp)",
             f"Survey type: {t['n_income']} income, {t['n_consumption']} consumption: {', '.join(t['consumption_countries'])}"]
    for line in n["lines"]:
        r = s[line]
        lines.append(
            f"${line:g}: floor {r['n_below_1pct']} ceil {r['n_above_99pct']} mean {r['mean_headcount']:.1f} | R2 B {r['r2_billionaire']:.3f} "
            f"G {r['r2_gdp']:.3f} J {r['r2_joint']:.3f} inc {r['incremental_r2']:.4f} | raw {r['raw_slope']:.2f} | OLS|G {r['ols_gdp_coef']:.2f} "
            f"(p {r['ols_gdp_p']:.3f}) OLS|G,type {r['ols_gdp_type_coef']:.2f} (p {r['ols_gdp_type_p']:.3f}) | AME|G {r['frac_gdp_ame']:.2f} "
            f"[{r['frac_gdp_ci95'][0]:.2f}, {r['frac_gdp_ci95'][1]:.2f}] AME|G,type {r['frac_gdp_type_ame']:.2f} "
            f"[{r['frac_gdp_type_ci95'][0]:.2f}, {r['frac_gdp_type_ci95'][1]:.2f}] | LOO G {r['loo_frac_gdp']:.3f} G+B {r['loo_frac_gdp_b']:.3f} "
            f"G,type {r['loo_frac_gdp_type']:.3f} G,type+B {r['loo_frac_gdp_type_b']:.3f} | AME|mean {r['frac_mean_ame']:.2f} "
            f"[{r['frac_mean_ci95'][0]:.2f}, {r['frac_mean_ci95'][1]:.2f}] LOO mean {r['loo_frac_mean']:.3f} mean+B {r['loo_frac_mean_b']:.3f}")
    lines += [
        f"$30 with survey type: OLS B {t['ols_type']['coef']:.2f} (p {t['ols_type']['p']:.3f}, CI {t['ols_type']['ci95'][0]:.2f} to "
        f"{t['ols_type']['ci95'][1]:.2f}); consumption {t['ols_type']['consumption_coef']:.2f} (p {t['ols_type']['consumption_p']:.4f}); "
        f"R2 {t['ols_type']['r_squared']:.3f}; frac AME {t['frac_type']['ame']:.2f} [{t['frac_type']['ci95'][0]:.2f}, "
        f"{t['frac_type']['ci95'][1]:.2f}] coef p {t['frac_type']['coef_p']:.3f}; LOO OLS {t['loo_ols_gdp_type']:.3f} -> {t['loo_ols_gdp_type_b']:.3f}",
        "Split by type: " + "; ".join(f"{k} n={v['n']} B {v['coef']:.2f} p {v['p']:.3f} R2 {v['r_squared']:.3f}" for k, v in t["split"].items()),
        f"PHL/RUS income alternative: $30 PHL {alt['PHL_income_30']:.1f} RUS {alt['RUS_income_30']:.1f}; OLS|G {alt['ols_gdp_coef']:.2f} "
        f"(p {alt['ols_gdp_p']:.3f}); AME|G {alt['frac_gdp_ame']:.2f} [{alt['frac_gdp_ci95'][0]:.2f}, {alt['frac_gdp_ci95'][1]:.2f}]",
        "Levels and shares given GDP and type: " + "; ".join(
            f"{k} B {v['coef']:.4f} (p {v['p']:.3f}, CI {v['ci95'][0]:.4f} to {v['ci95'][1]:.4f}; consumption {v['consumption_coef']:.4f})"
            for k, v in d.items() if isinstance(v, dict) and "coef" in v),
        "Deciles (log10): " + "; ".join(f"{x['decile']}: {x['coef']:.4f} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}] p {x['p']:.3f}" for x in d["deciles"]),
        f"Survey mean / daily GDP, median by type: {d['mean_ratio_by_type']}",
        f"Given survey mean at $30: R2 mean only {g['r2_survey_mean_only']:.3f}, with B {g['r2_with_billionaires']:.3f}; OLS B {g['ols_coef']:.2f} "
        f"(p {g['ols_p']:.3f}, CI {g['ols_ci95'][0]:.2f} to {g['ols_ci95'][1]:.2f}); frac AME {g['frac_ame']:.2f} [{g['frac_ci95'][0]:.2f}, "
        f"{g['frac_ci95'][1]:.2f}]; LOO frac {g['loo_frac_mean']:.3f} -> {g['loo_frac_mean_b']:.3f}",
        f"Welfare pairs: {w['n_pairs']} country-years, {w['n_countries']} countries ({', '.join(w['countries'])}), {w['year_min']}-{w['year_max']}; "
        f"consumption/income mean ratio mean {w['mean_ratio_mean']:.3f} median {w['mean_ratio_median']:.3f}; consumption lower in "
        f"{100 * w['share_consumption_lower']:.1f}%; headcount gap $8.30 {w['headcount_gap_8.3']}; $30 {w['headcount_gap_30']}",
    ]
    return lines


def _compare(new: Any, old: Any, path: str, problems: list[str]) -> None:
    if isinstance(new, dict):
        if not isinstance(old, dict) or set(new) != set(old):
            problems.append(f"{path}: keys differ")
            return
        for k in new:
            _compare(new[k], old[k], f"{path}.{k}", problems)
    elif isinstance(new, list):
        if not isinstance(old, list) or len(new) != len(old):
            problems.append(f"{path}: list differs")
            return
        for i, (a, b) in enumerate(zip(new, old)):
            _compare(a, b, f"{path}[{i}]", problems)
    elif isinstance(new, float) or isinstance(old, float):
        if not math.isclose(float(new), float(old), rel_tol=REL_TOL, abs_tol=1e-9):
            problems.append(f"{path}: {new!r} != {old!r}")
    elif new != old:
        problems.append(f"{path}: {new!r} != {old!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    panel = load_panel(PANEL)
    surveys, heads, manifest = load_pip()
    numbers = json.loads(json.dumps(compute(panel, surveys, heads, manifest)))
    lines = summary_lines(numbers)
    print("\n".join(lines))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part1/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(panel, surveys, heads, numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
