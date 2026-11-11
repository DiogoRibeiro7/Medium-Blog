"""Reproduce every number and figure in the revised billionaire-poverty article.

The input is the frozen, SHA-256-locked 54-country panel
(``data/observed_citizenship_54_2026.csv``): Forbes 2026 billionaire counts
attributed by citizenship (a secondary transcription), the OWID/World Bank
share of people below US$30 a day in 2021 PPP, WDI PPP GDP per person and
population. It is an exploratory comparison, not a replication of the
64-country residence-based chart that motivated the article, and nothing
below identifies a causal effect.

Usage, from this directory::

    python publication_analysis.py                  # compute, print, write
    python publication_analysis.py --check          # compare with committed

``--check`` recomputes everything and fails if any number differs from
``results/publication/numbers.json`` beyond a relative tolerance of 1e-6.
No network access is used.
"""

from __future__ import annotations

import argparse
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
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from observed_figures import BILLIONAIRE, GDP, load_panel

HERE = Path(__file__).resolve().parent
PANEL = HERE / "data" / "observed_citizenship_54_2026.csv"
RESULTS = HERE / "results" / "publication"
FIGURES = HERE / "publication_figures"
SEED = 20261111
N_PLACEBO = 20_000
REL_TOL = 1e-6

# Reference palette (dataviz skill, light mode), validated all-pairs for the
# three categorical slots used here; text never takes a series colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SOURCE = (
    "Data: Forbes 2026 billionaires by citizenship (secondary transcription); World Bank PIP via "
    "Our World in Data, share below $30/day (2021 PPP); WDI GDP per person (PPP) and population. "
    "54 countries; exploratory, not causal."
)
LABELLED = {"NGA", "IND", "CHN", "BRA", "PRT", "KAZ", "ISR", "USA", "CHE", "LUX", "IDN", "RUS"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def ols_hc3(
    frame: pd.DataFrame, cols: Sequence[str], weights: np.ndarray | None = None
) -> dict[str, Any]:
    """Fit OLS (or WLS) of the poverty share on ``cols`` with HC3 inference."""
    X = sm.add_constant(frame[list(cols)], has_constant="add")
    y = frame.poverty_pct.to_numpy(dtype=float)
    model = sm.OLS(y, X) if weights is None else sm.WLS(y, X, weights=weights)
    fit = model.fit(cov_type="HC3")
    ci = fit.conf_int()
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    return {
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "coef": {k: float(v) for k, v in fit.params.items()},
        "hc3_p": {k: float(v) for k, v in fit.pvalues.items()},
        "hc3_ci95": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fit.params.index},
        "fitted_above_100": int((fitted > 100).sum()),
        "fitted_below_0": int((fitted < 0).sum()),
        "fitted_max": float(fitted.max()),
        "fitted_min": float(fitted.min()),
        "countries_fitted_above_100": sorted(frame.iso3[fitted > 100].tolist()),
    }


def fractional_logit(frame: pd.DataFrame, cols: Sequence[str]) -> dict[str, Any]:
    """Papke-Wooldridge fractional logit with a sandwich covariance.

    Returns coefficients, robust p-values and, for the billionaire term, the
    average marginal effect in percentage points per tenfold increase with a
    delta-method 95% interval.
    """
    X = sm.add_constant(frame[list(cols)], has_constant="add")
    y = frame.poverty_pct.to_numpy(dtype=float) / 100.0
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC0")
    beta = fit.params.to_numpy()
    cov = fit.cov_params().to_numpy()
    Xm = X.to_numpy(dtype=float)
    p = 1.0 / (1.0 + np.exp(-Xm @ beta))
    w = p * (1.0 - p)
    out: dict[str, Any] = {
        "n": int(fit.nobs),
        "coef": {k: float(v) for k, v in fit.params.items()},
        "robust_p": {k: float(v) for k, v in fit.pvalues.items()},
        "fitted_min_pct": float(100 * p.min()),
        "fitted_max_pct": float(100 * p.max()),
    }
    if BILLIONAIRE in cols:
        j = list(X.columns).index(BILLIONAIRE)
        ame = 100.0 * float(w.mean()) * beta[j]
        grad = 100.0 * beta[j] * (w * (1.0 - 2.0 * p)) @ Xm / len(p)
        grad[j] += 100.0 * float(w.mean())
        se = float(math.sqrt(grad @ cov @ grad))
        out["ame_billionaire_pp_per_tenfold"] = float(ame)
        out["ame_se"] = se
        out["ame_ci95"] = [float(ame - 1.959963984540054 * se), float(ame + 1.959963984540054 * se)]
    return out


def loo_cv(frame: pd.DataFrame, cols: Sequence[str]) -> dict[str, float]:
    """Leave-one-country-out prediction error (percentage points) for OLS."""
    y = frame.poverty_pct.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(dtype=float) for c in cols])
    errors = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        errors[i] = y[i] - X[i] @ beta
    return {"rmse": float(np.sqrt(np.mean(errors**2))), "mae": float(np.mean(np.abs(errors)))}


def loo_cv_fractional(frame: pd.DataFrame, cols: Sequence[str]) -> dict[str, float]:
    """Leave-one-country-out prediction error for the fractional logit."""
    errors = []
    for i in range(len(frame)):
        train = frame.drop(index=frame.index[i])
        X = sm.add_constant(train[list(cols)], has_constant="add")
        fit = sm.GLM(train.poverty_pct / 100.0, X, family=sm.families.Binomial()).fit()
        x = np.r_[1.0, frame.iloc[i][list(cols)].to_numpy(dtype=float)]
        pred = 100.0 / (1.0 + math.exp(-float(x @ fit.params.to_numpy())))
        errors.append(frame.poverty_pct.iloc[i] - pred)
    e = np.asarray(errors)
    return {"rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e)))}


def placebo_proxies(frame: pd.DataFrame, n_draws: int, seed: int) -> dict[str, Any]:
    """Replace billionaire density with random 'development proxies'.

    Each proxy has exactly the sample mean, standard deviation and Pearson
    correlation with log GDP of log billionaire density, but its remaining
    variation is random noise orthogonal to log GDP. The distribution of the
    raw R-squared shows how well a pure, noisy development indicator would fit.
    """
    rng = np.random.default_rng(seed)
    y = frame.poverty_pct.to_numpy(dtype=float)
    g = frame[GDP].to_numpy(dtype=float)
    b = frame[BILLIONAIRE].to_numpy(dtype=float)
    zg = (g - g.mean()) / g.std()
    zb = (b - b.mean()) / b.std()
    r = float(np.mean(zg * zb))
    basis = np.column_stack([np.ones_like(zg), zg])
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = np.empty(n_draws)
    coef = np.empty(n_draws)
    for k in range(n_draws):
        e = rng.standard_normal(len(y))
        e -= basis @ np.linalg.lstsq(basis, e, rcond=None)[0]
        e = (e - e.mean()) / e.std()
        x = b.mean() + b.std() * (r * zg + math.sqrt(1.0 - r * r) * e)
        X1 = np.column_stack([np.ones_like(x), x])
        res = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        r2[k] = 1.0 - float(res @ res) / sst
        X2 = np.column_stack([np.ones_like(x), x, g])
        coef[k] = np.linalg.lstsq(X2, y, rcond=None)[0][1]
    observed_raw = ols_hc3(frame, [BILLIONAIRE])["r_squared"]
    observed_coef = ols_hc3(frame, [BILLIONAIRE, GDP])["coef"][BILLIONAIRE]
    return {
        "n_draws": n_draws,
        "seed": seed,
        "corr_billionaire_gdp": r,
        "r2_mean": float(r2.mean()),
        "r2_median": float(np.median(r2)),
        "r2_p05": float(np.percentile(r2, 5)),
        "r2_p95": float(np.percentile(r2, 95)),
        "r2_max": float(r2.max()),
        "share_r2_at_least_observed": float(np.mean(r2 >= observed_raw)),
        "share_abs_adjusted_coef_at_least_observed": float(np.mean(np.abs(coef) >= abs(observed_coef))),
        "share_adjusted_coef_at_most_observed": float(np.mean(coef <= observed_coef)),
        "_r2_draws": r2,
    }


def gdp_bands(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Within-band raw slopes after splitting countries into GDP thirds."""
    bands = pd.qcut(frame[GDP], 3, labels=["lowest", "middle", "highest"])
    rows = []
    for label in ("lowest", "middle", "highest"):
        part = frame.loc[bands == label]
        fit = ols_hc3(part, [BILLIONAIRE])
        rows.append({
            "band": label,
            "n": fit["n"],
            "gdp_min": float(part.gdp_ppp_per_capita.min()),
            "gdp_max": float(part.gdp_ppp_per_capita.max()),
            "poverty_min": float(part.poverty_pct.min()),
            "poverty_max": float(part.poverty_pct.max()),
            "slope": fit["coef"][BILLIONAIRE],
            "hc3_p": fit["hc3_p"][BILLIONAIRE],
            "r_squared": fit["r_squared"],
            "iso3": sorted(part.iso3.tolist()),
        })
    return rows


def leave_one_out(frame: pd.DataFrame) -> dict[str, Any]:
    """GDP-adjusted billionaire coefficient with each country omitted."""
    coefs = {
        iso: ols_hc3(frame.loc[frame.iso3 != iso], [BILLIONAIRE, GDP])["coef"][BILLIONAIRE]
        for iso in frame.iso3
    }
    lo = min(coefs, key=coefs.get)
    hi = max(coefs, key=coefs.get)
    return {"min": coefs[lo], "min_excluded": lo, "max": coefs[hi], "max_excluded": hi,
            "all_negative": bool(all(v < 0 for v in coefs.values()))}


def influence(frame: pd.DataFrame, top: int = 5) -> list[dict[str, Any]]:
    """Countries with the largest Cook's distance in the GDP-adjusted model."""
    X = sm.add_constant(frame[[BILLIONAIRE, GDP]], has_constant="add")
    cooks = sm.OLS(frame.poverty_pct, X).fit().get_influence().cooks_distance[0]
    order = np.argsort(cooks)[::-1][:top]
    return [{"iso3": frame.iso3.iloc[i], "cooks_distance": float(cooks[i])} for i in order]


def compute(frame: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    """Every number quoted in the article, from the frozen panel."""
    frame = frame.copy()
    frame["lag"] = frame.gdp_year - frame.poverty_year
    pop = frame.population.to_numpy(dtype=float)
    raw = ols_hc3(frame, [BILLIONAIRE])
    gdp = ols_hc3(frame, [GDP])
    joint = ols_hc3(frame, [BILLIONAIRE, GDP])
    lags = {}
    for max_lag in (2, 3, 4, 5):
        part = frame.loc[frame.lag <= max_lag]
        fit = ols_hc3(part, [BILLIONAIRE, GDP])
        lags[str(max_lag)] = {
            "n": fit["n"], "coef": fit["coef"][BILLIONAIRE],
            "hc3_p": fit["hc3_p"][BILLIONAIRE], "hc3_ci95": fit["hc3_ci95"][BILLIONAIRE],
        }
    placebo = placebo_proxies(frame, N_PLACEBO, SEED)
    corr_bg = placebo["corr_billionaire_gdp"]
    numbers = {
        "n": int(len(frame)),
        "billionaires_total": int(frame.billionaires.sum()),
        "density_per_10m_min": float((frame.billionaires * 1e7 / frame.population).min()),
        "density_per_10m_max": float((frame.billionaires * 1e7 / frame.population).max()),
        "poverty_year_counts": {str(int(k)): int(v) for k, v in frame.poverty_year.value_counts().sort_index().items()},
        "gdp_year_counts": {str(int(k)): int(v) for k, v in frame.gdp_year.value_counts().sort_index().items()},
        "lag_counts": {str(int(k)): int(v) for k, v in frame.lag.value_counts().sort_index().items()},
        "correlations": {
            "billionaire_gdp": corr_bg,
            "poverty_gdp": float(np.corrcoef(frame.poverty_pct, frame[GDP])[0, 1]),
            "poverty_billionaire": float(np.corrcoef(frame.poverty_pct, frame[BILLIONAIRE])[0, 1]),
        },
        "implied_proxy_r2": gdp["r_squared"] * corr_bg**2,
        "models": {
            "raw": raw,
            "gdp": gdp,
            "joint": joint,
            "joint_gdp_squared": ols_hc3(frame.assign(gdp_sq=frame[GDP] ** 2), [BILLIONAIRE, GDP, "gdp_sq"]),
            "joint_log_population": ols_hc3(frame, [BILLIONAIRE, GDP, "log_population"]),
            "joint_population_weighted": ols_hc3(frame, [BILLIONAIRE, GDP], weights=pop),
            "fractional_logit_raw": fractional_logit(frame, [BILLIONAIRE]),
            "fractional_logit_gdp": fractional_logit(frame, [GDP]),
            "fractional_logit_joint": fractional_logit(frame, [BILLIONAIRE, GDP]),
        },
        "incremental_r2": joint["r_squared"] - gdp["r_squared"],
        "loo_cv": {
            "intercept_only": loo_cv(frame, []),
            "billionaire_only": loo_cv(frame, [BILLIONAIRE]),
            "gdp_only": loo_cv(frame, [GDP]),
            "joint": loo_cv(frame, [BILLIONAIRE, GDP]),
            "fractional_logit_gdp": loo_cv_fractional(frame, [GDP]),
            "fractional_logit_joint": loo_cv_fractional(frame, [BILLIONAIRE, GDP]),
        },
        "lag_restrictions": lags,
        "gdp_bands": gdp_bands(frame),
        "leave_one_out": leave_one_out(frame),
        "influence": influence(frame),
        "placebo": {k: v for k, v in placebo.items() if not k.startswith("_")},
    }
    return numbers, placebo["_r2_draws"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": SECONDARY,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "text.parse_math": False,
        "axes.unicode_minus": True,
    })


def fmt(value: float, digits: int = 1, sign: bool = False) -> str:
    """Format a number with a typographic minus sign."""
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "\u2212")


def _frame(fig: plt.Figure, title: str, subtitle: str, top: float = 0.86, bottom: float = 0.17,
           adjust: bool = True) -> None:
    """Title, subtitle (may contain one line break) and source line."""
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=11, color=SECONDARY, va="top", linespacing=1.35)
    fig.text(0.012, 0.012, SOURCE, fontsize=8, color=MUTED, va="bottom", wrap=True)
    if adjust:
        lines = subtitle.count("\n")
        fig.subplots_adjust(top=top - 0.045 * lines, bottom=bottom, left=0.08, right=0.97)


def _density_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _gdp_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ticks = [5_000, 10_000, 20_000, 50_000, 100_000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:g}k"))


def _label(ax: plt.Axes, x: float, y: float, text: str, dx: int = 6, dy: int = 5, ha: str = "left") -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                color=SECONDARY, ha=ha)


OFFSETS = {  # manual nudges keep labels off neighbouring markers
    "NGA": (6, -12), "IND": (3, 7), "IDN": (6, -12), "CHN": (6, 5), "BRA": (6, 4),
    "RUS": (6, 4), "KAZ": (-6, 6, "right"), "PRT": (6, 4), "ISR": (6, 4), "USA": (-6, 7, "right"),
    "CHE": (6, 5), "LUX": (6, 4),
}


def _labels(ax: plt.Axes, frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, which: set[str]) -> None:
    for xi, yi, iso in zip(x, y, frame.iso3):
        if iso in which:
            off = OFFSETS.get(iso, (6, 4))
            _label(ax, xi, yi, iso, off[0], off[1], off[2] if len(off) > 2 else "left")


def fig_raw(frame: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    density = frame.billionaires * 1e7 / frame.population
    fig = plt.figure(figsize=(9, 5.8))
    ax = fig.add_axes([0.08, 0.17, 0.80, 0.68])
    cax = fig.add_axes([0.905, 0.17, 0.016, 0.68])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    norm = LogNorm(vmin=5_000, vmax=135_000)
    sc = ax.scatter(density, frame.poverty_pct, c=frame.gdp_ppp_per_capita, cmap=cmap, norm=norm,
                    s=62, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    raw = n["models"]["raw"]["coef"]
    xs = np.logspace(np.log10(density.min()), np.log10(density.max()), 200)
    ax.plot(xs, raw["const"] + raw[BILLIONAIRE] * np.log10(xs), color=INK, lw=2, zorder=2)
    ax.axhline(100, color=BASELINE, lw=1)
    ax.axhline(0, color=BASELINE, lw=1)
    _density_axis(ax)
    ax.set_ylim(-8, 125)
    ax.set_xlabel("Billionaires per 10 million people, by citizenship (log scale)")
    ax.set_ylabel("Share living below $30 a day, 2021 PPP (%)")
    _labels(ax, frame, density.to_numpy(), frame.poverty_pct.to_numpy(), LABELLED)
    ax.text(0.33, 114, "Fitted line rises above 100% here", ha="left", va="bottom", fontsize=9,
            color=SECONDARY)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_ticks([5_000, 10_000, 20_000, 50_000, 100_000])
    cbar.set_ticklabels(["$5k", "$10k", "$20k", "$50k", "$100k"])
    cbar.ax.minorticks_off()
    cbar.set_label("GDP per person, PPP", color=SECONDARY)
    cbar.outline.set_visible(False)
    m = n["models"]["raw"]
    _frame(fig, "More billionaires per person, lower poverty: the raw association",
           f"Slope {fmt(m['coef'][BILLIONAIRE])} points per tenfold increase in density, R² {m['r_squared']:.2f}. "
           "Darker points are richer countries.", adjust=False)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_placebo(n: dict[str, Any], draws: np.ndarray, out: Path) -> None:
    p = n["placebo"]
    raw_r2 = n["models"]["raw"]["r_squared"]
    gdp_r2 = n["models"]["gdp"]["r_squared"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.hist(draws, bins=np.linspace(0.30, 0.82, 66), color="#86b6ef", rwidth=0.82, zorder=2)
    top = ax.get_ylim()[1] * 1.32
    ax.set_ylim(0, top)
    for value, colour, text, ha, dx in (
        (p["r2_mean"], MUTED, f"Average placebo\nR² {p['r2_mean']:.3f}", "right", -0.006),
        (raw_r2, ORANGE, f"Billionaire density\nR² {raw_r2:.3f}", "left", 0.006),
        (gdp_r2, INK, f"GDP per person\nR² {gdp_r2:.3f}", "right", -0.006),
    ):
        ax.axvline(value, color=colour, lw=2, zorder=3)
        ax.text(value + dx, top * 0.97, text, ha=ha, va="top", fontsize=9.5, color=SECONDARY)
    ax.set_xlim(0.30, 0.84)
    ax.set_xlabel("R² of poverty share on one variable (same 54 countries)")
    ax.set_ylabel("Placebo proxies")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    count = round(p["share_r2_at_least_observed"] * p["n_draws"])
    _frame(fig, "A random development proxy fits almost as well as billionaire density",
           f"{p['n_draws']:,} synthetic variables with the same mean, spread and {p['corr_billionaire_gdp']:.2f} correlation "
           f"with log GDP as billionaire density.\n{count:,} of them ({100 * count / p['n_draws']:.2f}%) fit the poverty "
           "share at least as well.")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_bands(frame: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    density = frame.billionaires * 1e7 / frame.population
    bands = pd.qcut(frame[GDP], 3, labels=["lowest", "middle", "highest"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.0), sharey=True)
    names = {"lowest": "Lowest GDP third", "middle": "Middle GDP third", "highest": "Highest GDP third"}
    for ax, band in zip(axes, n["gdp_bands"]):
        mask = (bands == band["band"]).to_numpy()
        d = density[mask]
        ax.scatter(d, frame.poverty_pct[mask], color=BLUE, s=46, edgecolors=SURFACE, linewidths=1.1, zorder=3)
        xs = np.logspace(np.log10(d.min()), np.log10(d.max()), 50)
        const = np.mean(frame.poverty_pct[mask]) - band["slope"] * np.mean(np.log10(d))
        ax.plot(xs, const + band["slope"] * np.log10(xs), color=INK, lw=2, zorder=2)
        _density_axis(ax)
        ax.set_xlim(0.12, 90)
        ax.set_ylim(-2, 104)
        ax.set_title(f"{names[band['band']]}\n${band['gdp_min']/1000:.0f}k–${band['gdp_max']/1000:.0f}k per person",
                     fontsize=10.5, color=INK, loc="left")
        ax.text(0.03, 0.04 if band["band"] != "highest" else 0.6,
                f"slope {fmt(band['slope'])}, R² {band['r_squared']:.2f}\nHC3 p = {band['hc3_p']:.2f}",
                transform=ax.transAxes, fontsize=9.5, color=SECONDARY)
    axes[0].set_ylabel("Share below $30 a day (%)")
    axes[1].set_xlabel("Billionaires per 10 million people, by citizenship (log scale)")
    _frame(fig, "Among countries at similar income levels, billionaire density explains little",
           "18 countries per panel, split by GDP per person. Lines are within-panel least-squares fits.",
           top=0.78, bottom=0.19)
    fig.subplots_adjust(left=0.07, wspace=0.08)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_partial(frame: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    z = sm.add_constant(frame[[GDP]], has_constant="add")
    xres = sm.OLS(frame[BILLIONAIRE], z).fit().resid.to_numpy()
    yres = sm.OLS(frame.poverty_pct, z).fit().resid.to_numpy()
    joint = n["models"]["joint"]
    slope = joint["coef"][BILLIONAIRE]
    lo, hi = joint["hc3_ci95"][BILLIONAIRE]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    xs = np.linspace(xres.min() - 0.05, xres.max() + 0.05, 100)
    ax.fill_between(xs, lo * xs, hi * xs, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.plot(xs, slope * xs, color=INK, lw=2, zorder=2)
    ax.scatter(xres, yres, color=BLUE, s=52, edgecolors=SURFACE, linewidths=1.1, zorder=3)
    ax.axhline(0, color=BASELINE, lw=1)
    named = {row["iso3"] for row in n["influence"]} | {"USA", "CHE", "IND", "CHN", "PRT"}
    offsets = {"NPL": (6, -4), "NGA": (6, 4), "KAZ": (6, -2), "IRL": (6, -11), "GEO": (6, 4),
               "USA": (6, 4), "CHE": (6, 4), "IND": (6, 4), "CHN": (-7, 2, "right"), "PRT": (6, 4)}
    for xi, yi, iso in zip(xres, yres, frame.iso3):
        if iso in named:
            off = offsets.get(iso, (6, 4))
            _label(ax, xi, yi, iso, off[0], off[1], off[2] if len(off) > 2 else "left")
    ax.set_xlabel("Billionaire density not predicted by GDP (log10 units)")
    ax.set_ylabel("Poverty share not predicted by GDP (points)")
    ax.text(0.99, 0.97, f"Slope {fmt(slope)}, HC3 95% interval {fmt(lo)} to {fmt(hi, sign=True)}, "
            f"p = {joint['hc3_p'][BILLIONAIRE]:.2f}\nShaded wedge: slopes inside that interval",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=SECONDARY)
    _frame(fig, "What is left after GDP: a shallow slope with a wide interval",
           "Both axes are residuals from a regression on log GDP per person (added-variable plot).\n"
           "Labelled: the five most influential countries and five reference points.")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_bounded(frame: pd.DataFrame, n: dict[str, Any], out: Path) -> None:
    gdp_fit = n["models"]["gdp"]["coef"]
    frac = n["models"]["fractional_logit_gdp"]["coef"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.axhspan(100, 145, color="#f0efec", zorder=0)
    ax.axhspan(-15, 0, color="#f0efec", zorder=0)
    ax.text(140_000, 141, "Impossible: more than 100% of people", fontsize=9, color=SECONDARY, va="top",
            ha="right")
    ax.text(5_100, -13.5, "Impossible: below 0%", fontsize=9, color=SECONDARY, ha="left", va="bottom")
    ax.scatter(frame.gdp_ppp_per_capita, frame.poverty_pct, color=BLUE, s=46, edgecolors=SURFACE,
               linewidths=1.1, zorder=3, label="Countries")
    gx = np.logspace(np.log10(frame.gdp_ppp_per_capita.min()), np.log10(frame.gdp_ppp_per_capita.max()), 300)
    lg = np.log10(gx)
    ax.plot(gx, gdp_fit["const"] + gdp_fit[GDP] * lg, color=INK, lw=2, zorder=2, label="Straight line (OLS)")
    ax.plot(gx, 100 / (1 + np.exp(-(frac["const"] + frac[GDP] * lg))), color=ORANGE, lw=2, zorder=2,
            label="Fractional logit (bounded)")
    for xi, yi, iso in zip(frame.gdp_ppp_per_capita, frame.poverty_pct, frame.iso3):
        if iso in {"NPL", "NGA", "LUX", "IRL", "KAZ"}:
            dx, dy, ha = {"NPL": (4, -13, "left"), "NGA": (4, -13, "left"), "LUX": (-6, -4, "right"),
                          "IRL": (-6, 4, "right")}.get(iso, (4, 5, "left"))
            _label(ax, xi, yi, iso, dx, dy, ha)
    cluster = frame.loc[frame.iso3.isin(["IND", "LBN", "PHL"])]
    ax.annotate("IND, LBN, PHL", (cluster.gdp_ppp_per_capita.mean(), 97), xytext=(10_300, 84), fontsize=9,
                color=SECONDARY, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    _gdp_axis(ax)
    ax.set_ylim(-15, 145)
    ax.set_xlabel("GDP per person, PPP (log scale)")
    ax.set_ylabel("Share below $30 a day (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    g = n["models"]["gdp"]
    _frame(fig, "A straight line predicts impossible poverty rates inside the sample",
           f"The GDP-only straight line predicts more than 100% for {g['fitted_above_100']} countries (up to "
           f"{g['fitted_max']:.0f}%) and below 0% for {g['fitted_below_0']}.\nThe fractional logit cannot leave the 0–100% range.")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_prediction(n: dict[str, Any], out: Path) -> None:
    cv = n["loo_cv"]
    rows = [
        ("Sample average only", cv["intercept_only"]["rmse"]),
        ("Billionaire density only", cv["billionaire_only"]["rmse"]),
        ("GDP per person only", cv["gdp_only"]["rmse"]),
        ("GDP + billionaire density", cv["joint"]["rmse"]),
        ("GDP, fractional logit", cv["fractional_logit_gdp"]["rmse"]),
        ("GDP + billionaires, fractional logit", cv["fractional_logit_joint"]["rmse"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, [r[1] for r in rows], height=0.42, color=BLUE, zorder=2)
    for yv, (_, value) in zip(ypos, rows):
        ax.text(value + 0.4, yv, f"{value:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 42)
    ax.set_xlabel("Leave-one-country-out prediction error, root mean square (percentage points)")
    _frame(fig, "Predicting a left-out country: billionaires add almost nothing to GDP",
           "Each country is predicted from a model fitted to the other 53. Lower is better.",
           top=0.83, bottom=0.17)
    fig.subplots_adjust(left=0.30)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def fig_sensitivity(n: dict[str, Any], out: Path) -> None:
    m = n["models"]
    lags = n["lag_restrictions"]
    rows = [
        ("No GDP control (n = 54)", m["raw"]["coef"][BILLIONAIRE], m["raw"]["hc3_ci95"][BILLIONAIRE], False),
        ("GDP control, baseline (n = 54)", m["joint"]["coef"][BILLIONAIRE], m["joint"]["hc3_ci95"][BILLIONAIRE], True),
        ("+ GDP squared", m["joint_gdp_squared"]["coef"][BILLIONAIRE], m["joint_gdp_squared"]["hc3_ci95"][BILLIONAIRE], False),
        ("+ log population", m["joint_log_population"]["coef"][BILLIONAIRE], m["joint_log_population"]["hc3_ci95"][BILLIONAIRE], False),
        ("Fractional logit, average effect", m["fractional_logit_joint"]["ame_billionaire_pp_per_tenfold"],
         m["fractional_logit_joint"]["ame_ci95"], False),
        ("Population-weighted", m["joint_population_weighted"]["coef"][BILLIONAIRE],
         m["joint_population_weighted"]["hc3_ci95"][BILLIONAIRE], False),
    ] + [
        (f"Poverty and GDP years ≤ {k} apart (n = {v['n']})", v["coef"], v["hc3_ci95"], False)
        for k, v in lags.items() if k != "5"
    ]
    loo = n["leave_one_out"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ypos = np.arange(len(rows) + 1)[::-1]
    for yv, (label, est, ci, base) in zip(ypos[:-1], rows):
        colour = ORANGE if base else BLUE
        ax.plot(ci, [yv, yv], color=colour, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([est], [yv], color=colour, s=58, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.text(62, yv, fmt(est), va="center", ha="right", fontsize=9.5, color=INK)
    ax.plot([loo["min"], loo["max"]], [ypos[-1], ypos[-1]], color=MUTED, lw=6, solid_capstyle="round", zorder=2)
    ax.text(62, ypos[-1], f"{fmt(loo['min'])} to {fmt(loo['max'])}", va="center", ha="right", fontsize=9.5,
            color=INK)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    ax.set_yticks(ypos, [r[0] for r in rows] + ["Drop one country (range, 54 refits)"])
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-60, 63)
    ax.set_xticks([-60, -40, -20, 0, 20, 40])
    ax.set_xlabel("Poverty points per tenfold increase in billionaire density (95% interval)")
    _frame(fig, "How the billionaire coefficient moves with each modelling choice",
           "Orange: the GDP-adjusted baseline. Lines: 95% intervals. Subsets overlap, so they are not independent tests.",
           top=0.85, bottom=0.15)
    fig.subplots_adjust(left=0.37, right=0.97)
    fig.savefig(out, dpi=250)
    plt.close(fig)


FIGURE_FILES = {
    "fig1_raw_association.png": "raw association coloured by GDP",
    "fig2_placebo_proxies.png": "placebo development proxies",
    "fig3_added_variable.png": "added-variable plot after GDP",
    "fig4_within_gdp_thirds.png": "within-GDP-third slopes",
    "fig5_out_of_sample.png": "leave-one-out prediction error",
    "fig6_bounded_outcome.png": "straight line versus fractional logit",
    "fig7_coefficient_sensitivity.png": "coefficient across specifications",
}


def render(frame: pd.DataFrame, numbers: dict[str, Any], draws: np.ndarray, outdir: Path) -> None:
    """Write the seven publication figures as PNG."""
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_raw(frame, numbers, outdir / "fig1_raw_association.png")
    fig_placebo(numbers, draws, outdir / "fig2_placebo_proxies.png")
    fig_bands(frame, numbers, outdir / "fig4_within_gdp_thirds.png")
    fig_partial(frame, numbers, outdir / "fig3_added_variable.png")
    fig_bounded(frame, numbers, outdir / "fig6_bounded_outcome.png")
    fig_prediction(numbers, outdir / "fig5_out_of_sample.png")
    fig_sensitivity(numbers, outdir / "fig7_coefficient_sensitivity.png")


# ---------------------------------------------------------------------------
# Reporting and checking
# ---------------------------------------------------------------------------

def summary_lines(n: dict[str, Any]) -> list[str]:
    """Human-readable list of every number quoted in the article."""
    m, cv, p = n["models"], n["loo_cv"], n["placebo"]
    B = BILLIONAIRE
    lines = [
        f"Countries: {n['n']}; billionaires in panel: {n['billionaires_total']}",
        f"Density range per 10m: {n['density_per_10m_min']:.3f} to {n['density_per_10m_max']:.2f}",
        f"Poverty observation years: {n['poverty_year_counts']}",
        f"GDP years: {n['gdp_year_counts']}; poverty-to-GDP gap (years): {n['lag_counts']}",
        f"corr(log B, log GDP) = {n['correlations']['billionaire_gdp']:.4f}; corr(P, log GDP) = "
        f"{n['correlations']['poverty_gdp']:.4f}; corr(P, log B) = {n['correlations']['poverty_billionaire']:.4f}",
        f"Raw OLS: slope {m['raw']['coef'][B]:.3f}, intercept {m['raw']['coef']['const']:.3f}, R2 {m['raw']['r_squared']:.4f}, "
        f"HC3 95% CI {m['raw']['hc3_ci95'][B][0]:.2f} to {m['raw']['hc3_ci95'][B][1]:.2f}",
        f"GDP-only OLS: slope {m['gdp']['coef'][GDP]:.3f}, R2 {m['gdp']['r_squared']:.4f}",
        f"Joint OLS: B {m['joint']['coef'][B]:.3f} (HC3 p {m['joint']['hc3_p'][B]:.3f}, CI {m['joint']['hc3_ci95'][B][0]:.2f} "
        f"to {m['joint']['hc3_ci95'][B][1]:.2f}); GDP {m['joint']['coef'][GDP]:.3f}; R2 {m['joint']['r_squared']:.4f}",
        f"Incremental R2: {n['incremental_r2']:.4f}; implied proxy R2 (R2_gdp x r^2): {n['implied_proxy_r2']:.4f}",
        f"Fitted outside 0-100: raw {m['raw']['fitted_above_100']} above (max {m['raw']['fitted_max']:.1f}), "
        f"{m['raw']['fitted_below_0']} below (min {m['raw']['fitted_min']:.1f}); GDP-only {m['gdp']['fitted_above_100']} above "
        f"{m['gdp']['countries_fitted_above_100']} (max {m['gdp']['fitted_max']:.1f}), {m['gdp']['fitted_below_0']} below "
        f"(min {m['gdp']['fitted_min']:.1f}); joint {m['joint']['fitted_above_100']} above (max {m['joint']['fitted_max']:.1f}), "
        f"{m['joint']['fitted_below_0']} below",
        f"+GDP squared: B {m['joint_gdp_squared']['coef'][B]:.3f} (p {m['joint_gdp_squared']['hc3_p'][B]:.3f}), R2 "
        f"{m['joint_gdp_squared']['r_squared']:.4f}",
        f"+log population: B {m['joint_log_population']['coef'][B]:.3f} (p {m['joint_log_population']['hc3_p'][B]:.3f})",
        f"Population-weighted: B {m['joint_population_weighted']['coef'][B]:.3f} (p {m['joint_population_weighted']['hc3_p'][B]:.3f}, "
        f"CI {m['joint_population_weighted']['hc3_ci95'][B][0]:.1f} to {m['joint_population_weighted']['hc3_ci95'][B][1]:.1f})",
        f"Fractional logit raw: AME {m['fractional_logit_raw']['ame_billionaire_pp_per_tenfold']:.2f} pp/tenfold",
        f"Fractional logit joint: B coef {m['fractional_logit_joint']['coef'][B]:.4f} (robust p "
        f"{m['fractional_logit_joint']['robust_p'][B]:.3f}); AME {m['fractional_logit_joint']['ame_billionaire_pp_per_tenfold']:.2f} "
        f"pp/tenfold, 95% CI {m['fractional_logit_joint']['ame_ci95'][0]:.2f} to {m['fractional_logit_joint']['ame_ci95'][1]:.2f}; "
        f"fitted range {m['fractional_logit_joint']['fitted_min_pct']:.1f}% to {m['fractional_logit_joint']['fitted_max_pct']:.1f}%",
        "LOO-CV RMSE (MAE): " + "; ".join(f"{k} {v['rmse']:.2f} ({v['mae']:.2f})" for k, v in cv.items()),
        "Lag restrictions: " + "; ".join(
            f"<= {k}y n={v['n']} B {v['coef']:.3f} p {v['hc3_p']:.3f} CI {v['hc3_ci95'][0]:.2f} to {v['hc3_ci95'][1]:.2f}"
            for k, v in n["lag_restrictions"].items()),
        "GDP thirds: " + "; ".join(
            f"{b['band']} n={b['n']} GDP ${b['gdp_min']:,.0f}-${b['gdp_max']:,.0f} poverty {b['poverty_min']:.1f}-{b['poverty_max']:.1f}% "
            f"slope {b['slope']:.2f} p {b['hc3_p']:.3f} R2 {b['r_squared']:.3f}" for b in n["gdp_bands"]),
        f"Leave-one-out: {n['leave_one_out']['min']:.2f} (without {n['leave_one_out']['min_excluded']}) to "
        f"{n['leave_one_out']['max']:.2f} (without {n['leave_one_out']['max_excluded']}); all negative: {n['leave_one_out']['all_negative']}",
        "Most influential (Cook's D): " + ", ".join(f"{r['iso3']} {r['cooks_distance']:.3f}" for r in n["influence"]),
        f"Placebo ({p['n_draws']} draws, seed {p['seed']}): R2 mean {p['r2_mean']:.4f}, median {p['r2_median']:.4f}, "
        f"5-95% {p['r2_p05']:.4f}-{p['r2_p95']:.4f}, max {p['r2_max']:.4f}; share >= observed {p['share_r2_at_least_observed']:.4f}; "
        f"share |adjusted coef| >= observed {p['share_abs_adjusted_coef_at_least_observed']:.4f}; share adjusted coef <= observed "
        f"{p['share_adjusted_coef_at_most_observed']:.4f}",
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
    parser.add_argument("--check", action="store_true", help="compare with the committed numbers.json")
    parser.add_argument("--figures", type=Path, default=FIGURES, help="figure output folder")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    frame = load_panel(PANEL)
    numbers, draws = compute(frame)
    numbers = json.loads(json.dumps(numbers))
    lines = summary_lines(numbers)
    print("\n".join(lines))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/publication/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(frame, numbers, draws, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
