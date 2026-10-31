"""Seeded simulations of spurious regression and genuine cointegration.

Independent random walks can produce conventional OLS significance without
any common data-generating mechanism. A genuinely cointegrated system has a
stationary long-run residual and an error-correction representation. This
module calibrates *specific finite-sample tests* by simulation; its critical
values are not universal econometric reference tables.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]
BLUE, RED, TEAL, ORANGE, GRAY, PURPLE = (
    "#26629c", "#b6494f", "#188a7a", "#cb8138", "#596271", "#845c9c"
)


@dataclass(frozen=True)
class Design:
    """Specified experiment and simple stationary-cointegrated process."""

    seed: int = 20261031
    horizon: int = 250
    long_run_beta: float = 1.6
    adjustment_rho: float = 0.65
    residual_noise: float = 0.7
    stationary_rho: float = 0.60
    nominal_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise TypeError("horizon must be an integer")
        if self.horizon < 40:
            raise ValueError("horizon must be at least 40")
        for field in ("long_run_beta", "adjustment_rho", "residual_noise",
                      "stationary_rho", "nominal_alpha"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not 0 < self.nominal_alpha < 1:
            raise ValueError("nominal_alpha must be strictly between 0 and 1")
        if not -1 < self.adjustment_rho < 1 or not -1 < self.stationary_rho < 1:
            raise ValueError("stationarity requires both AR coefficients in (-1,1)")
        if self.residual_noise <= 0:
            raise ValueError("residual_noise must be positive")


@dataclass(frozen=True)
class OLSResult:
    """One-intercept simple least-squares fit, including a *naive* t test."""

    intercept: float
    slope: float
    slope_se_naive: float
    t_naive: float
    p_naive: float
    r_squared: float
    durbin_watson: float
    fitted: FloatArray
    residuals: FloatArray


@dataclass(frozen=True)
class Process:
    """An I(1) common driver X with stationary or shifting residual relation."""

    x: FloatArray
    y: FloatArray
    latent_residual: FloatArray
    beta_path: FloatArray


def _series(y: FloatArray) -> FloatArray:
    """Validate and standardize a one-dimensional finite numeric series."""

    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 4 or not np.isfinite(arr).all():
        raise ValueError("series must be a finite one-dimensional vector of length at least 4")
    return arr


def ols_levels(y: FloatArray, x: FloatArray) -> OLSResult:
    """Regression with intercept; t/p presume IID errors and may be invalid.

    The t test is purposefully retained to demonstrate its failure in levels
    regressions between independent unit-root processes. It is not corrected
    for dependence, and a small p-value is not evidence of causal association.
    """

    yy, xx = _series(y), _series(x)
    if yy.shape != xx.shape:
        raise ValueError("x and y must share a length")
    centered = xx - xx.mean()
    ssx = float(centered @ centered)
    if ssx <= 1e-14:
        raise ValueError("predictor must have nonzero variance")
    xmat = np.column_stack((np.ones(xx.size), xx))
    coeff, *_ = np.linalg.lstsq(xmat, yy, rcond=None)
    fitted = xmat @ coeff
    residuals = yy - fitted
    rss = float(residuals @ residuals)
    tss = float((yy - yy.mean()) @ (yy - yy.mean()))
    if tss <= 1e-14:
        raise ValueError("response must have nonzero variance")
    sigma = math.sqrt(rss / (xx.size - 2))
    slope_se = sigma / math.sqrt(ssx)
    if slope_se <= 0:
        raise ValueError("slope standard error must be positive")
    t_stat = float(coeff[1] / slope_se)
    return OLSResult(
        intercept=float(coeff[0]), slope=float(coeff[1]),
        slope_se_naive=float(slope_se), t_naive=t_stat,
        p_naive=float(2 * student_t.sf(abs(t_stat), df=xx.size - 2)),
        r_squared=float(1 - rss / tss),
        durbin_watson=float(np.diff(residuals) @ np.diff(residuals) / rss),
        fitted=fitted, residuals=residuals,
    )


def independent_walks(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Independent I(1) zero-drift Gaussian series with no causal link."""

    if not isinstance(n, int) or n < 40:
        raise ValueError("n must be an integer >= 40")
    return np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))


def independent_iid(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Independent stationary white-noise control series."""

    if not isinstance(n, int) or n < 40:
        raise ValueError("n must be an integer >= 40")
    return rng.normal(size=n), rng.normal(size=n)


def ar1(n: int, rho: float, sd: float, rng: np.random.Generator) -> FloatArray:
    """Generate a stationary Gaussian AR(1) with a stationary initial draw."""

    if not isinstance(n, int) or n < 40 or not -1 < rho < 1 or sd <= 0:
        raise ValueError("AR(1) requires n>=40, |rho|<1 and sd>0")
    values = np.empty(n, dtype=float)
    values[0] = rng.normal(scale=sd / math.sqrt(1 - rho * rho))
    noise = rng.normal(scale=sd, size=n - 1)
    for t in range(1, n):
        values[t] = rho * values[t - 1] + noise[t - 1]
    return values


def cointegrated_pair(n: int, rng: np.random.Generator, design: Design,
                      break_index: int | None = None, post_beta: float | None = None) -> Process:
    """Construct Y_t = beta_t X_t + U_t with X I(1), U stationary AR(1)."""

    if not isinstance(n, int) or n < 40:
        raise ValueError("n must be an integer >= 40")
    if break_index is not None and not 10 <= break_index < n - 10:
        raise ValueError("break_index must have >=10 observations on each side")
    if break_index is not None and (post_beta is None or not math.isfinite(post_beta)):
        raise ValueError("post_beta must be finite when break_index is set")
    x = np.cumsum(rng.normal(size=n))
    u = ar1(n, design.adjustment_rho, design.residual_noise, rng)
    beta = np.full(n, design.long_run_beta)
    if break_index is not None:
        beta[break_index:] = float(post_beta)
    return Process(x=x, y=beta * x + u, latent_residual=u, beta_path=beta)


def dickey_fuller_t(series: FloatArray, intercept: bool = True) -> float:
    """Lag-zero DF coefficient t statistic, with an optional constant.

    The regression is Delta Z_t = c + gamma Z_(t-1) + e_t. No lagged
    differences are added: calibrated critical values apply ONLY to this
    exact IID-innovation design, sample length and deterministic term choice.
    """

    z = _series(series)
    if z.size < 40:
        raise ValueError("Dickey-Fuller illustration requires >=40 observations")
    lag, delta = z[:-1], np.diff(z)
    features = np.column_stack((np.ones(lag.size), lag)) if intercept else lag[:, None]
    coefficient, *_ = np.linalg.lstsq(features, delta, rcond=None)
    residual = delta - features @ coefficient
    dof = delta.size - features.shape[1]
    scale = float(residual @ residual) / dof
    inv = np.linalg.inv(features.T @ features)
    se = math.sqrt(scale * float(inv[-1, -1]))
    return float(coefficient[-1] / se)


def engle_granger_t(x: FloatArray, y: FloatArray) -> float:
    """Two-step cointegration residual DF t statistic, no residual intercept.

    Because residuals are first estimated using OLS, the statistic does NOT
    follow an ordinary univariate Dickey-Fuller null distribution.
    """

    return dickey_fuller_t(ols_levels(y, x).residuals, intercept=False)


def ecm_fit(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Estimate Delta Y = c + b Delta X + lambda uhat_(t-1) + error.

    The estimated long-run residual uhat comes from OLS(Y on 1,X). These
    illustrative coefficients are not causal unless substantive assumptions
    provide an independent causal identification argument.
    """

    xx, yy = _series(x), _series(y)
    if xx.shape != yy.shape:
        raise ValueError("x and y must have equal length")
    longrun = ols_levels(yy, xx)
    mat = np.column_stack((np.ones(xx.size - 1), np.diff(xx), longrun.residuals[:-1]))
    target = np.diff(yy)
    coef, *_ = np.linalg.lstsq(mat, target, rcond=None)
    unexplained = target - mat @ coef
    return {
        "longrun_slope": longrun.slope,
        "delta_x_slope": float(coef[1]),
        "adjustment_lambda": float(coef[2]),
        "residual_sd": float(np.sqrt(unexplained @ unexplained / (target.size - 3))),
        "r_squared": float(1 - (unexplained @ unexplained) /
                           ((target - target.mean()) @ (target - target.mean()))),
    }


def _table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write consistent UTF-8 CSV evidence and reject nonfinite statistics."""

    if not rows:
        raise ValueError("cannot write an empty result table")
    keys = list(rows[0])
    for row in rows:
        if list(row) != keys:
            raise ValueError("inconsistent result-table columns")
        if any(isinstance(value, (float, np.floating)) and not np.isfinite(value)
               for value in row.values()):
            raise ValueError("nonfinite statistics are not publishable")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: Any, path: Path) -> None:
    """Save and close a publication PNG to limit memory consumption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)


def _out(root: Path, name: str, kind: str, quick: bool) -> Path:
    suffix = "_quick" if quick else ""
    extension = ".csv" if kind == "results" else ".png"
    return root / kind / f"{name}{suffix}{extension}"


def _summarize_null(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group empirical false rejections using the *conventional* IID t test."""

    summary: list[dict[str, Any]] = []
    for name in ("random_walk_levels", "random_walk_differences", "independent_iid"):
        subset = [r for r in rows if r["case"] == name]
        summary.append({"case": name, "replicates": len(subset),
                        "mean_r_squared": float(np.mean([r["r_squared"] for r in subset])),
                        "median_r_squared": float(np.median([r["r_squared"] for r in subset])),
                        "false_rejection_rate": float(np.mean([r["nominal_reject"] for r in subset])),
                        "mean_durbin_watson": float(np.mean([r["durbin_watson"] for r in subset])),
                        "median_abs_t": float(np.median(np.abs([r["t_naive"] for r in subset])))})
    return summary


def run_study(root: Path, quick: bool = False) -> dict[str, float]:
    """Generate eight original figures and fifteen reproducible CSV tables."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    root.mkdir(parents=True, exist_ok=True)
    design = Design()
    rng = np.random.default_rng(design.seed)
    horizon = design.horizon
    main_n = 90 if quick else 1400
    unit_n = 150 if quick else 1800
    scaling_n = 60 if quick else 260
    break_n = 65 if quick else 300

    # Choose a clear, explicitly selected visual example. Selection affects
    # only the illustrative path, never the Monte Carlo null-rejection rates.
    for attempt in range(1, 200):
        x_demo, y_demo = independent_walks(horizon, rng)
        demo_fit = ols_levels(y_demo, x_demo)
        if demo_fit.p_naive < 1e-5 and demo_fit.r_squared > .25:
            break
    else:
        raise RuntimeError("could not find a predeclared illustrative example")
    illustration = [{"time": t, "x_random_walk": float(x_demo[t]),
                     "y_random_walk": float(y_demo[t]),
                     "naive_fitted_y": float(demo_fit.fitted[t]),
                     "naive_residual": float(demo_fit.residuals[t]),
                     "illustrative_selection_attempt": attempt}
                    for t in range(horizon)]
    _table(_out(root, "illustrative_null_path", "results", quick), illustration)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    axes[0].plot(x_demo, color=BLUE, label="Independent X random walk", lw=2)
    axes[0].plot(y_demo, color=ORANGE, label="Independent Y random walk", lw=2)
    axes[0].legend(frameon=False)
    _style(axes[0], "Two independent accumulated noise sequences", "Time index", "Level")
    axes[1].scatter(x_demo, y_demo, alpha=.4, s=14, color=BLUE)
    order = np.argsort(x_demo)
    axes[1].plot(x_demo[order], demo_fit.fitted[order], color=RED, lw=2)
    _style(axes[1], f"Naive level regression: R²={demo_fit.r_squared:.2f}, p={demo_fit.p_naive:.1g}",
           "X level", "Y level")
    _save(fig, _out(root, "fig1_independent_random_walks", "figures", quick))

    null_rows: list[dict[str, Any]] = []
    for replicate in range(main_n):
        xr, yr = independent_walks(horizon, rng)
        xs, ys = independent_iid(horizon, rng)
        for name, x, y in (("random_walk_levels", xr, yr),
                           ("random_walk_differences", np.diff(xr), np.diff(yr)),
                           ("independent_iid", xs, ys)):
            result = ols_levels(y, x)
            null_rows.append({"replicate": replicate, "case": name, "n": int(x.size),
                "slope": result.slope, "r_squared": result.r_squared,
                "t_naive": result.t_naive, "p_naive": result.p_naive,
                "nominal_reject": int(result.p_naive < design.nominal_alpha),
                "durbin_watson": result.durbin_watson})
    _table(_out(root, "null_regression_replicates", "results", quick), null_rows)
    null_summary = _summarize_null(null_rows)
    _table(_out(root, "null_regression_summary", "results", quick), null_summary)

    fig, ax = plt.subplots(figsize=(9.6, 5))
    vals = np.array([r["t_naive"] for r in null_rows if r["case"] == "random_walk_levels"])
    vals_iid = np.array([r["t_naive"] for r in null_rows if r["case"] == "independent_iid"])
    bins = np.linspace(-30, 30, 81)
    ax.hist(vals, bins=bins, density=True, alpha=.52, label="Independent random walks", color=RED)
    ax.hist(vals_iid, bins=bins, density=True, alpha=.63, label="Independent IID series", color=BLUE)
    tcrit = float(student_t.ppf(1 - design.nominal_alpha / 2, horizon - 2))
    ax.axvline(-tcrit, linestyle="--", color=GRAY)
    ax.axvline(tcrit, linestyle="--", color=GRAY, label="Conventional 5% cutoffs")
    ax.legend(frameon=False)
    _style(ax, "Conventional t statistics do not keep their IID reference law", "Naive slope t statistic", "Density")
    _save(fig, _out(root, "fig2_naive_t_distribution", "figures", quick))

    t_hist_rows: list[dict[str, Any]] = []
    for case, sample in (("random_walk_levels", vals), ("independent_iid", vals_iid)):
        hist, edges = np.histogram(sample, bins=bins, density=True)
        t_hist_rows.extend({"case": case, "left": float(l), "right": float(u),
                            "density": float(v)} for l, u, v in zip(edges[:-1], edges[1:], hist))
    _table(_out(root, "naive_t_histogram", "results", quick), t_hist_rows)

    fig, ax = plt.subplots(figsize=(9.6, 5))
    r2_bins = np.linspace(0, 1, 51)
    for case, color, label in (("random_walk_levels", RED, "RW in levels"),
                               ("random_walk_differences", TEAL, "RW first differences"),
                               ("independent_iid", BLUE, "IID control")):
        sample = [r["r_squared"] for r in null_rows if r["case"] == case]
        ax.hist(sample, bins=r2_bins, density=True, alpha=.49, label=label, color=color)
    ax.legend(frameon=False)
    _style(ax, "R² from unrelated trends can look unusually persuasive", "Regression R²", "Density")
    _save(fig, _out(root, "fig3_r_squared_nulls", "figures", quick))

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    names = ["Independent RW\nin levels", "First differences\nof same RW", "Independent IID\ncontrol"]
    rates = [r["false_rejection_rate"] for r in null_summary]
    ax.bar(names, rates, color=[RED, TEAL, BLUE], alpha=.82)
    ax.axhline(design.nominal_alpha, color=GRAY, linestyle="--", label="Nominal α = 0.05")
    ax.set_ylim(0, max(.15, max(rates) * 1.12))
    ax.legend(frameon=False)
    for index, rate in enumerate(rates):
        ax.text(index, rate + .012, f"{100 * rate:.1f}%", ha="center", fontsize=11)
    _style(ax, "False rejection under three zero-relationship designs", "Data-generating process", "Fraction p < 0.05")
    _save(fig, _out(root, "fig4_false_discovery_rates", "figures", quick))

    unit_rows: list[dict[str, Any]] = []
    for rep in range(unit_n):
        x, y = independent_walks(horizon, rng)
        unit_rows.append({"replicate": rep, "df_x_with_intercept": dickey_fuller_t(x, True),
                          "eg_residual_without_intercept": engle_granger_t(x, y)})
    _table(_out(root, "unit_root_null_replicates", "results", quick), unit_rows)
    df_null = np.array([r["df_x_with_intercept"] for r in unit_rows])
    eg_null = np.array([r["eg_residual_without_intercept"] for r in unit_rows])
    dfcrit, egcrit = float(np.quantile(df_null, .05)), float(np.quantile(eg_null, .05))
    cal_rows = [
        {"test": "DF_level_intercept_lag0", "n": horizon, "null_replicates": unit_n,
         "empirical_lower_5pct": dfcrit, "mean_t": float(df_null.mean()),
         "null_process": "independent_gaussian_random_walks"},
        {"test": "EG_residual_no_intercept_lag0", "n": horizon, "null_replicates": unit_n,
         "empirical_lower_5pct": egcrit, "mean_t": float(eg_null.mean()),
         "null_process": "independent_gaussian_random_walks"},
    ]
    _table(_out(root, "calibrated_critical_values", "results", quick), cal_rows)
    fig, ax = plt.subplots(figsize=(9.6, 5))
    test_bins = np.linspace(-6, 3, 65)
    ax.hist(df_null, bins=test_bins, density=True, color=BLUE, alpha=.50, label="DF: observed RW level")
    ax.hist(eg_null, bins=test_bins, density=True, color=ORANGE, alpha=.52,
            label="EG: estimated OLS residual")
    ax.axvline(dfcrit, color=BLUE, linestyle="--", label=f"DF 5%: {dfcrit:.2f}")
    ax.axvline(egcrit, color=ORANGE, linestyle="--", label=f"EG 5%: {egcrit:.2f}")
    ax.legend(frameon=False)
    _style(ax, "A residual cointegration test has a different null law", "Lag-zero residual autoregression t statistic", "Density")
    _save(fig, _out(root, "fig5_df_versus_eg_null", "figures", quick))

    process = cointegrated_pair(horizon, rng, design)
    longrun = ols_levels(process.y, process.x)
    ecm = ecm_fit(process.x, process.y)
    eg_alt = engle_granger_t(process.x, process.y)
    coin_rows = [{"time": t, "x": float(process.x[t]), "y": float(process.y[t]),
                  "true_stationary_residual": float(process.latent_residual[t]),
                  "fitted_longrun_residual": float(longrun.residuals[t]),
                  "true_longrun_beta": design.long_run_beta}
                 for t in range(horizon)]
    _table(_out(root, "cointegrated_path", "results", quick), coin_rows)
    diagnostic_items = {**ecm, "true_longrun_beta": design.long_run_beta,
        "true_adjustment_lambda": design.adjustment_rho - 1,
        "eg_residual_t": eg_alt, "simulated_eg_critical": egcrit,
        "reject_no_cointegration": float(eg_alt < egcrit)}
    _table(_out(root, "cointegration_diagnostics", "results", quick),
           [{"metric": name, "value": value} for name, value in diagnostic_items.items()])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    axes[0].plot(process.x, color=BLUE, label="X: common stochastic trend", linewidth=1.7)
    axes[0].plot(process.y, color=ORANGE, label="Y: beta X + stationary U", linewidth=1.6)
    axes[0].legend(frameon=False)
    _style(axes[0], "Common trend does not itself establish causation", "", "Levels")
    axes[1].plot(process.latent_residual, color=TEAL, linewidth=1.7,
                 label="True stationary deviation")
    axes[1].plot(longrun.residuals, color=PURPLE, linestyle="--", alpha=.75,
                 label="Estimated equilibrium residual")
    axes[1].axhline(0, color=GRAY, linewidth=.8)
    axes[1].legend(frameon=False)
    _style(axes[1], "The cointegrating residual mean-reverts", "Time index", "Y minus fitted beta X")
    _save(fig, _out(root, "fig6_genuine_cointegration", "figures", quick))

    sizes = (100, 200, 400, 800)
    scaling_rows: list[dict[str, Any]] = []
    for n in sizes:
        for rep in range(scaling_n):
            co = cointegrated_pair(n, rng, design)
            independent_x, independent_y = independent_walks(n, rng)
            beta_co = ols_levels(co.y, co.x).slope
            beta_null = ols_levels(independent_y, independent_x).slope
            scaling_rows.extend([
                {"n": n, "replicate": rep, "system": "cointegrated",
                 "estimated_beta": beta_co, "reference_beta": design.long_run_beta,
                 "absolute_error": abs(beta_co - design.long_run_beta)},
                {"n": n, "replicate": rep, "system": "independent_walks",
                 "estimated_beta": beta_null, "reference_beta": 0.,
                 "absolute_error": abs(beta_null)},
            ])
    _table(_out(root, "slope_scaling_replicates", "results", quick), scaling_rows)
    scaling_summary: list[dict[str, Any]] = []
    for n in sizes:
        for system in ("cointegrated", "independent_walks"):
            errors = np.array([r["absolute_error"] for r in scaling_rows
                               if r["n"] == n and r["system"] == system])
            scaling_summary.append({"n": n, "system": system, "replicates": scaling_n,
                                    "median_absolute_beta_error": float(np.median(errors)),
                                    "mean_absolute_beta_error": float(errors.mean()),
                                    "p90_absolute_beta_error": float(np.quantile(errors, .9))})
    _table(_out(root, "slope_scaling_summary", "results", quick), scaling_summary)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for case, color, label in (("cointegrated", TEAL, "Genuine cointegrated relation"),
                               ("independent_walks", RED, "Independent random-walk slope")):
        yy = [next(r["median_absolute_beta_error"] for r in scaling_summary
                   if r["n"] == n and r["system"] == case) for n in sizes]
        ax.loglog(sizes, yy, marker="o", linewidth=2.3, color=color, label=label)
    ax.legend(frameon=False)
    ax.set_xticks(sizes, labels=[str(n) for n in sizes])
    _style(ax, "More observations can sharpen a real equilibrium—not a spurious one",
           "Time-series length", "Median absolute slope error (log scale)")
    _save(fig, _out(root, "fig7_slope_convergence", "figures", quick))

    break_length = 420
    break_point = 252
    beta_after = 2.3
    broken = cointegrated_pair(break_length, rng, design, break_index=break_point,
                              post_beta=beta_after)
    pooled = ols_levels(broken.y, broken.x)
    window = 95
    rolling: list[dict[str, Any]] = []
    for end in range(window, break_length + 1):
        start = end - window
        fit = ols_levels(broken.y[start:end], broken.x[start:end])
        rolling.append({"end_time": end - 1, "window_start": start,
                        "estimated_beta": fit.slope,
                        "true_beta_end": float(broken.beta_path[end - 1])})
    break_path = [{"time": t, "x": float(broken.x[t]), "y": float(broken.y[t]),
                   "true_beta": float(broken.beta_path[t]),
                   "true_stationary_u": float(broken.latent_residual[t]),
                   "pooled_residual": float(pooled.residuals[t]),
                   "break_index": break_point} for t in range(break_length)]
    _table(_out(root, "broken_relation_path", "results", quick), break_path)
    _table(_out(root, "rolling_break_estimates", "results", quick), rolling)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(range(break_length), broken.beta_path, color=RED, linestyle="--",
                 lw=2.4, label="True beta by regime")
    axes[0].plot([r["end_time"] for r in rolling],
                 [r["estimated_beta"] for r in rolling], color=BLUE,
                 linewidth=1.8, label=f"Rolling {window}-observation OLS")
    axes[0].axvline(break_point, color=GRAY, alpha=.8)
    axes[0].legend(frameon=False)
    _style(axes[0], "A structural break changes the long-run relation", "", "Long-run slope")
    axes[1].plot(pooled.residuals, color=PURPLE, label="One pooled long-run residual")
    axes[1].axvline(break_point, color=GRAY, alpha=.8)
    axes[1].legend(frameon=False)
    _style(axes[1], "A single equilibrium leaves regime-dependent residual structure",
           "Time index", "Pooled fitted residual")
    _save(fig, _out(root, "fig8_breaks_and_instability", "figures", quick))

    break_rows: list[dict[str, Any]] = []
    for rep in range(break_n):
        sample = cointegrated_pair(break_length, rng, design,
                                  break_index=break_point, post_beta=beta_after)
        all_fit = ols_levels(sample.y, sample.x)
        pre = ols_levels(sample.y[:break_point], sample.x[:break_point])
        post = ols_levels(sample.y[break_point:], sample.x[break_point:])
        pooled_eg = engle_granger_t(sample.x, sample.y)
        break_rows.append({"replicate": rep, "true_beta_early": design.long_run_beta,
                           "true_beta_late": beta_after,
                           "estimated_beta_early": pre.slope,
                           "estimated_beta_late": post.slope,
                           "pooled_beta": all_fit.slope,
                           "pooled_eg_t": pooled_eg,
                           "pooled_eg_reject_using_stable_null_calibration": int(
                               pooled_eg < egcrit)})
    _table(_out(root, "break_replicates", "results", quick), break_rows)
    early_error = np.array([r["estimated_beta_early"] - design.long_run_beta for r in break_rows])
    late_error = np.array([r["estimated_beta_late"] - beta_after for r in break_rows])
    break_summary = [
        {"metric": "true_slope_change", "value": beta_after - design.long_run_beta},
        {"metric": "mean_estimated_early_slope", "value": float(np.mean([r["estimated_beta_early"] for r in break_rows]))},
        {"metric": "mean_estimated_late_slope", "value": float(np.mean([r["estimated_beta_late"] for r in break_rows]))},
        {"metric": "rmse_early_slope", "value": float(np.sqrt(np.mean(early_error ** 2)))},
        {"metric": "rmse_late_slope", "value": float(np.sqrt(np.mean(late_error ** 2)))},
        {"metric": "pooled_eg_rejection_fraction_with_stable_null_cutoff", "value": float(
            np.mean([r["pooled_eg_reject_using_stable_null_calibration"] for r in break_rows]))},
        {"metric": "break_replicates", "value": float(break_n)},
    ]
    _table(_out(root, "break_summary", "results", quick), break_summary)

    summary = {
        "null_replicates": float(main_n),
        "rw_levels_false_rejection": null_summary[0]["false_rejection_rate"],
        "rw_differences_false_rejection": null_summary[1]["false_rejection_rate"],
        "iid_false_rejection": null_summary[2]["false_rejection_rate"],
        "mean_rw_levels_r2": null_summary[0]["mean_r_squared"],
        "median_rw_levels_r2": null_summary[0]["median_r_squared"],
        "df_null_critical_5pct": dfcrit,
        "eg_null_critical_5pct": egcrit,
        "cointegration_residual_t": eg_alt,
        "cointegrated_ecm_lambda": ecm["adjustment_lambda"],
        "cointegrated_ecm_beta": ecm["delta_x_slope"],
        "structural_break_true_beta_change": beta_after - design.long_run_beta,
        "illustration_attempt": float(attempt),
    }
    _table(_out(root, "study_summary", "results", quick),
           [{"metric": k, "value": v} for k, v in summary.items()])
    return summary


def main() -> None:
    """Generate all reproducible evidence from a normal CLI invocation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduced counts with suffixed output files")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Output root for figures and CSV evidence")
    args = parser.parse_args()
    for key, value in run_study(args.output, quick=args.quick).items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
