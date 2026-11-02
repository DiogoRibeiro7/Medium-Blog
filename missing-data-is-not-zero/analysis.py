"""Nine seeded studies of missing-data mechanisms, estimation, and deployment.

All data are synthetic. Full mode writes nine original figures and eighteen
CSV evidence tables. Quick mode writes suffixed files without overwriting
canonical evidence. Missingness indicator R=1 always means *observed*.
"""

from __future__ import annotations

import argparse
import csv
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
SEED = 20261101
EXPERIMENTS = (
    "zero", "mechanisms", "mar_recovery", "variance", "intervals",
    "conditional", "sensitivity", "indicator", "positivity",
)
FIGURES = (
    "fig1_zero_is_not_missing", "fig2_missingness_mechanisms",
    "fig3_mar_recovery", "fig4_variance_attenuation",
    "fig5_interval_coverage", "fig6_conditional_regression",
    "fig7_mnar_sensitivity", "fig8_indicator_drift",
    "fig9_positivity_and_weights",
)
REPETITIONS = {
    "zero": 24, "mechanisms": 24, "mar_recovery": 24, "variance": 24,
    "intervals": 300, "conditional": 24, "sensitivity": 24,
    "indicator": 24, "positivity": 24,
}
Row = dict[str, str | int | float | None]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Design:
    """Output target, run intensity, and reproducible base seed."""

    root: Path = ROOT
    quick: bool = False
    seed: int = SEED


def check_vector(value: NDArray[np.floating], minimum: int = 3) -> FloatArray:
    """Accept a finite numerical vector with at least the requested rows."""

    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError("expected a finite one-dimensional vector")
    return a


def check_mask(mask: NDArray[np.bool_], n: int) -> NDArray[np.bool_]:
    """Require a Boolean observation indicator with both groups present."""

    a = np.asarray(mask)
    if a.dtype != np.dtype(bool) or a.ndim != 1 or len(a) != n:
        raise ValueError("observation indicator must be a Boolean n-vector")
    if a.sum() < 3 or (~a).sum() < 3:
        raise ValueError("both observed and missing groups must have three rows")
    return a


def stream(seed: int, name: str, rep: int) -> np.random.Generator:
    """Independent stable SeedSequence stream for each study and replicate."""

    if not isinstance(seed, int) or seed < 0 or rep < 0 or name not in EXPERIMENTS:
        raise ValueError("invalid seed, experiment name, or replicate")
    return np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(name.encode("ascii")), rep])
    )


def observation(
    rep: int, condition: str, metric: str, value: float,
    theory: float | None = None,
) -> Row:
    """Construct a long-format finite Monte Carlo observation."""

    if not condition or not metric:
        raise ValueError("nonempty condition and metric required")
    if not np.isfinite(value) or (theory is not None and not np.isfinite(theory)):
        raise ValueError("scores and theory must be finite")
    return {"replicate": rep, "condition": condition, "metric": metric,
            "value": float(value), "theory": theory}


def summarize(rows: list[Row]) -> list[Row]:
    """Mean and 1.96-MCSE intervals over *independent outer replications*."""

    if not rows:
        raise ValueError("cannot summarize empty results")
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), str(row["metric"])), []).append(row)
    result = []
    for (condition, metric), group in sorted(groups.items()):
        theory = {r["theory"] for r in group}
        if len(theory) != 1:
            raise ValueError("theory differs within condition/metric")
        values = np.array([float(r["value"]) for r in group])
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        mean = float(values.mean())
        se = sd / math.sqrt(len(values))
        result.append({
            "condition": condition, "metric": metric, "n": len(values),
            "mean": mean, "sd": sd, "se": se,
            "lower95": mean - 1.96 * se, "upper95": mean + 1.96 * se,
            "theory": group[0]["theory"],
        })
    return result


def lookup(rows: list[Row], condition: str, metric: str,
           field: str = "mean") -> float:
    """Retrieve one summary number by its full condition/metric key."""

    found = [r for r in rows if r["condition"] == condition and r["metric"] == metric]
    if len(found) != 1 or found[0].get(field) is None:
        raise KeyError((condition, metric, field))
    return float(found[0][field])


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write a deterministic UTF-8 CSV with uniform field schema."""

    if not rows:
        raise ValueError("empty CSV")
    fields = list(rows[0])
    if any(set(r) != set(fields) for r in rows):
        raise ValueError("inconsistent CSV schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hajek_mean(y: FloatArray, observed: NDArray[np.bool_],
               probabilities: FloatArray, lower: float = 0.0) -> float:
    """Self-normalized IPW (Hájek) mean using observed outcomes only.

    Probability model is supplied explicitly; this estimator does not
    identify MNAR mechanisms from observed data. The optional floor is a
    declared bias-variance sensitivity operation, not a free correction.
    """

    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) < 6:
        raise ValueError("outcomes must be a one-dimensional sample")
    r = check_mask(observed, len(values))
    if not np.isfinite(values[r]).all():
        raise ValueError("observed outcome values must be finite")
    pi = np.asarray(probabilities, dtype=float)
    if pi.shape != values.shape or not np.isfinite(pi).all():
        raise ValueError("invalid observation probabilities")
    if not 0 <= lower < 1 or (pi <= 0).any() or (pi > 1).any():
        raise ValueError("probabilities must be in (0,1]")
    weights = 1 / np.maximum(pi[r], lower)
    return float(np.sum(weights * values[r]) / np.sum(weights))


def effective_sample_size(weights: FloatArray) -> float:
    """Kish effective sample size for positive finite nonzero weights."""

    w = check_vector(weights)
    if (w <= 0).any():
        raise ValueError("weights must be positive")
    return float(w.sum()**2 / np.sum(w**2))


def normal_multiple_imputation(
    rng: np.random.Generator,
    observed_values: FloatArray, missing_n: int, m: int = 30,
    known_variance: float = 1.0,
) -> tuple[float, float]:
    """Proper normal-model MI of a population mean under MCAR.

    With an improper flat prior for the unknown mean, draw a posterior mean
    for each imputation, then missing observations conditionally on it. Each
    complete-data analysis uses known outcome variance / total n, and Rubin's
    within + between formula pools the mean and variance. This special-case
    implementation is not a generic MICE package.
    """

    x = check_vector(observed_values)
    if missing_n < 1 or m < 2 or known_variance <= 0:
        raise ValueError("need missing values, m >= 2, and positive variance")
    nobs = len(x)
    n = nobs + missing_n
    posterior_sd = math.sqrt(known_variance / nobs)
    means = []
    for _ in range(m):
        mu = float(x.mean() + rng.normal(scale=posterior_sd))
        missing = rng.normal(loc=mu, scale=math.sqrt(known_variance), size=missing_n)
        means.append(float((x.sum() + missing.sum()) / n))
    q = np.asarray(means)
    ubar = known_variance / n
    between = float(q.var(ddof=1))
    total = ubar + (1 + 1/m) * between
    return float(q.mean()), float(total)


def zero_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Confusing missingness with numerical zero biases a positive mean."""

    n = 400 if quick else 1800
    x = 2.0 + rng.normal(size=n)
    result = []
    for q in (0.1, 0.3, 0.6):
        observed = rng.random(n) >= q
        checked = check_mask(observed, n)
        coded_zero = np.where(checked, x, 0.)
        cond = f"{q:.1f}"
        result.extend((
            observation(rep, cond, "zero_filled_mean", float(coded_zero.mean()), 2*(1-q)),
            observation(rep, cond, "complete_case_mean", float(x[checked].mean()), 2.0),
            observation(rep, cond, "true_sample_mean", float(x.mean()), 2.0),
        ))
    return result


def mechanisms_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """MAR depends on observed Z; MNAR also depends on unseen outcome Y."""

    n = 550 if quick else 2100
    z = rng.normal(size=n)
    y = 1 + z + rng.normal(scale=.8, size=n)
    laws = {
        "MCAR": np.full(n, .62),
        "MAR": expit(.5 - 1.2*z),
        "MNAR": expit(.5 - 1.2*y),
    }
    result = []
    for name, pi in laws.items():
        observed = check_mask(rng.random(n) < pi, n)
        # Use only the fully observed auxiliary variable in the fitted model.
        observed_model = LogisticRegression(C=1000., max_iter=200).fit(
            z.reshape(-1, 1), observed.astype(int)
        )
        fitted_pi = np.clip(observed_model.predict_proba(z.reshape(-1, 1))[:, 1],
                            .005, 1.0)
        cc = float(y[observed].mean() - y.mean())
        adjusted = hajek_mean(y, observed, fitted_pi) - float(y.mean())
        result.extend((
            observation(rep, name, "cc_bias", cc, 0.0 if name == "MCAR" else None),
            observation(rep, name, "auxiliary_ipw_bias", adjusted,
                        0.0 if name in {"MCAR", "MAR"} else None),
            observation(rep, name, "missing_fraction", float((~observed).mean())),
        ))
    return result


def mar_recovery_study(rng: np.random.Generator, rep: int,
                       quick: bool) -> list[Row]:
    """Recover a marginal mean when R is MAR conditional on fully observed Z."""

    n = 450 if quick else 1600
    z = rng.normal(size=n)
    y = 1 + 1.2*z + rng.normal(scale=.9, size=n)
    pi = expit(.2 - 1.5*z)
    observed = check_mask(rng.random(n) < pi, n)
    linear = LinearRegression().fit(z[observed, None], y[observed])
    methods = {
        "complete_case": float(y[observed].mean()),
        "oracle_ipw": hajek_mean(y, observed, pi),
        "outcome_standardization": float(linear.predict(z[:, None]).mean()),
    }
    true_mean = float(y.mean())
    result = []
    for name, est in methods.items():
        result.extend((
            observation(rep, name, "error", est - true_mean, 0.0 if name != "complete_case" else None),
            observation(rep, name, "squared_error", (est - true_mean)**2),
        ))
    return result


def variance_study(rng: np.random.Generator, rep: int,
                   quick: bool) -> list[Row]:
    """Mean filling preserves a mean under MCAR but suppresses dispersion."""

    n = 500 if quick else 1800
    x = 2 + rng.normal(size=n)
    result = []
    for q in (.1, .3, .5, .7):
        observed = check_mask(rng.random(n) >= q, n)
        mean_fill = np.where(observed, x, float(x[observed].mean()))
        cond = f"{q:.1f}"
        result.extend((
            observation(rep, cond, "complete_case_variance",
                        float(np.var(x[observed], ddof=1)), 1.0),
            observation(rep, cond, "mean_fill_variance",
                        float(np.var(mean_fill, ddof=1)), 1.0-q),
            observation(rep, cond, "full_sample_variance",
                        float(np.var(x, ddof=1)), 1.0),
        ))
    return result


def intervals_study(rng: np.random.Generator, rep: int,
                    quick: bool) -> list[Row]:
    """Coverage: treating mean-filled cells as observed versus proper MI."""

    n = 120
    true_mean = 2.0
    x = true_mean + rng.normal(size=n)
    result = []
    for q in (.2, .5, .75):
        r = check_mask(rng.random(n) >= q, n)
        observed = x[r]
        na = len(observed)
        filled = np.where(r, x, observed.mean())
        # Incorrect inference: filled cells counted as independent observations.
        naive_mu = float(filled.mean())
        naive_se = float(filled.std(ddof=1) / math.sqrt(n))
        cc_mu = float(observed.mean())
        cc_se = 1 / math.sqrt(na)  # known variance = 1 in this designed model
        mi_mu, mi_variance = normal_multiple_imputation(
            rng, observed, n - na, m=20, known_variance=1.0,
        )
        methods = {
            "single": (naive_mu, naive_se),
            "complete_case": (cc_mu, cc_se),
            "multiple": (mi_mu, math.sqrt(mi_variance)),
        }
        for method, (est, se) in methods.items():
            cond = f"{q:.2f}:{method}"
            result.extend((
                observation(rep, cond, "coverage",
                            float(abs(est - true_mean) <= 1.96 * se)),
                observation(rep, cond, "interval_width", float(2 * 1.96 * se)),
                observation(rep, cond, "mean_error", est - true_mean),
            ))
    return result


def conditional_study(rng: np.random.Generator, rep: int,
                      quick: bool) -> list[Row]:
    """Complete-case regression can retain a slope while losing a mean."""

    n = 450 if quick else 1800
    x = rng.normal(size=n)
    y = 1 + 2*x + rng.normal(scale=.8, size=n)
    result = []
    for strength in (0.0, .5, 1.0, 2.0):
        pi = expit(.5 - strength*x)
        observed = check_mask(rng.random(n) < pi, n)
        beta = float(LinearRegression().fit(
            x[observed, None], y[observed]
        ).coef_[0])
        result.extend((
            observation(rep, f"{strength:.1f}", "slope_error", beta - 2, 0.0),
            observation(rep, f"{strength:.1f}", "mean_error",
                        float(y[observed].mean() - y.mean()),
                        0.0 if strength == 0 else None),
        ))
    return result


def sensitivity_study(rng: np.random.Generator, rep: int,
                      quick: bool) -> list[Row]:
    """MNAR pattern-mixture delta is an assumption, not learned from records."""

    n = 500 if quick else 2100
    y = rng.normal(size=n)
    pi = expit(.3 + 1.2*y)
    observed = check_mask(rng.random(n) < pi, n)
    obs_fraction = float(observed.mean())
    obs_mean = float(y[observed].mean())
    unseen_mean = float(y[~observed].mean())  # oracle; unavailable to analysis
    oracle_delta = unseen_mean - obs_mean
    result = []
    for delta in (-2.0, -1.6, -1.2, -.8, -.4, 0.0):
        condition = f"{delta:.1f}"
        assumed_mean = obs_mean + (1-obs_fraction)*delta
        result.extend((
            observation(rep, condition, "assumed_population_mean", assumed_mean),
            observation(rep, condition, "full_population_mean", float(y.mean()), 0.0),
            observation(rep, condition, "oracle_pattern_delta", oracle_delta),
        ))
    return result


def indicator_study(rng: np.random.Generator, rep: int,
                    quick: bool) -> list[Row]:
    """The missingness flag predicts severity only while the workflow is stable."""

    ntr, nte = (250, 450) if quick else (650, 1200)
    z = rng.normal(size=ntr + nte)
    latent = rng.normal(size=ntr + nte)
    y = 1.2*z + 2.0*latent + rng.normal(scale=.7, size=len(z))
    missing = (latent + rng.normal(scale=.25, size=len(z))) > 0
    full = Ridge(alpha=1.).fit(
        np.column_stack((z[:ntr], missing[:ntr].astype(float))), y[:ntr]
    )
    clean = Ridge(alpha=1.).fit(z[:ntr, None], y[:ntr])
    original = np.column_stack((z[ntr:], missing[ntr:].astype(float)))
    shifted = np.column_stack((z[ntr:], rng.integers(0, 2, size=nte).astype(float)))
    scores = {
        "indicator_original": r2_score(y[ntr:], full.predict(original)),
        "indicator_shifted": r2_score(y[ntr:], full.predict(shifted)),
        "no_indicator_original": r2_score(y[ntr:], clean.predict(z[ntr:, None])),
        "no_indicator_shifted": r2_score(y[ntr:], clean.predict(z[ntr:, None])),
    }
    return [observation(rep, key, "test_r2", float(score))
            for key, score in scores.items()]


def positivity_study(rng: np.random.Generator, rep: int,
                     quick: bool) -> list[Row]:
    """Near-zero observation probabilities destabilize inverse weights."""

    n = 750 if quick else 2600
    z = rng.normal(size=n)
    y = 2 + z + rng.normal(scale=.8, size=n)
    target = float(y.mean())
    result = []
    for strength in (0., .8, 1.6, 2.4):
        pi = expit(-strength*z)
        observed = check_mask(rng.random(n) < pi, n)
        cc = float(y[observed].mean())
        raw = hajek_mean(y, observed, pi)
        trimmed = hajek_mean(y, observed, pi, lower=.05)
        w = 1 / pi[observed]
        cond = f"{strength:.1f}"
        for method, estimate in (("complete_case", cc), ("oracle_ipw", raw),
                                 ("clipped_ipw", trimmed)):
            result.extend((
                observation(rep, cond, f"{method}_error", estimate - target),
                observation(rep, cond, f"{method}_squared_error",
                            (estimate - target)**2),
            ))
        result.extend((
            observation(rep, cond, "relative_ess",
                        effective_sample_size(w) / observed.sum()),
            observation(rep, cond, "observed_fraction", float(observed.mean())),
        ))
    return result


STUDIES: dict[str, Callable[[np.random.Generator, int, bool], list[Row]]] = {
    "zero": zero_study,
    "mechanisms": mechanisms_study,
    "mar_recovery": mar_recovery_study,
    "variance": variance_study,
    "intervals": intervals_study,
    "conditional": conditional_study,
    "sensitivity": sensitivity_study,
    "indicator": indicator_study,
    "positivity": positivity_study,
}


def series(ax: plt.Axes, summary: list[Row], cond: list[str],
           metric: str, label: str, color: str,
           x: list[float] | None = None) -> None:
    """Add mean and normal ±1.96-MCSE bars to an axes."""

    xx = x if x is not None else list(range(len(cond)))
    means = [lookup(summary, c, metric) for c in cond]
    errors = [1.96 * lookup(summary, c, metric, "se") for c in cond]
    ax.errorbar(xx, means, yerr=errors, label=label, color=color,
                linewidth=2.3, marker="o", capsize=3.5)


def plot_study(name: str, summary: list[Row], filename: Path) -> None:
    """Render the independently generated figure for a canonical study."""

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 170,
    })
    navy, orange, teal = "#153856", "#d66a35", "#008b87"
    fig, ax = plt.subplots(figsize=(9.3, 5.3), layout="constrained")
    if name == "zero":
        c = ["0.1", "0.3", "0.6"]
        x = [float(z) for z in c]
        series(ax, summary, c, "zero_filled_mean", "Missing coded as zero", orange, x)
        series(ax, summary, c, "complete_case_mean", "Observed-only mean", navy, x)
        ax.axhline(2, color=teal, ls="--", label="True population mean")
        ax.plot(x, [2*(1-q) for q in x], ls=":", color=orange, alpha=.8)
        ax.set(xlabel="Missing fraction (MCAR)", ylabel="Estimated mean",
               title="1  |  A missing measurement is not the value zero")
    elif name == "mechanisms":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), layout="constrained")
        c = ["MCAR", "MAR", "MNAR"]
        for a, metric, heading in zip(
            axes, ("cc_bias", "auxiliary_ipw_bias"),
            ("Complete-case mean bias", "IPW using observed Z only"),
        ):
            values = [lookup(summary, k, metric) for k in c]
            errors = [1.96*lookup(summary, k, metric, "se") for k in c]
            a.bar(c, values, yerr=errors, capsize=3,
                  color=[teal, navy, orange])
            a.axhline(0, c="#666666", lw=1)
            a.set_title(heading)
            a.set_ylabel("Error relative to complete sample")
        fig.suptitle("2  |  The missingness mechanism changes what can be recovered",
                     fontsize=15)
    elif name == "mar_recovery":
        c = ["complete_case", "oracle_ipw", "outcome_standardization"]
        values = [lookup(summary, k, "error") for k in c]
        errors = [1.96*lookup(summary, k, "error", "se") for k in c]
        ax.bar(["Complete\ncases", "Oracle\nIPW", "Outcome model\n+ standardize"],
               values, yerr=errors, capsize=3, color=[orange, navy, teal])
        ax.axhline(0, c="#666666", lw=1)
        ax.set(ylabel="Error against complete-sample mean",
               title="3  |  An observed auxiliary variable supports MAR correction")
    elif name == "variance":
        c = ["0.1", "0.3", "0.5", "0.7"]
        x = [float(k) for k in c]
        series(ax, summary, c, "complete_case_variance", "Observed-case variance", navy, x)
        series(ax, summary, c, "mean_fill_variance", "Mean-filled variance", orange, x)
        ax.plot(x, [1-k for k in x], "--", c=orange, alpha=.65,
                label="Approximate mean-fill expectation")
        ax.axhline(1., ls="--", c=navy, alpha=.55)
        ax.set(xlabel="Missing fraction (MCAR)", ylabel="Sample variance",
               title="4  |  Replacing unknown cells by a mean erases dispersion")
    elif name == "intervals":
        c = [".20", ".50", ".75"]
        xx = np.array([.2, .5, .75])
        for offset, method, label, color in (
            (-.018, "single", "Single mean fill, naive SE", orange),
            (0.0, "complete_case", "Complete-case, valid SE", navy),
            (.018, "multiple", "Proper normal-model MI", teal),
        ):
            keys = [f"{q}:{method}" for q in ("0.20", "0.50", "0.75")]
            yy = [lookup(summary, k, "coverage") for k in keys]
            ee = [1.96*lookup(summary, k, "coverage", "se") for k in keys]
            ax.errorbar(xx+offset, yy, yerr=ee, marker="o",
                        label=label, color=color, lw=2, capsize=3)
        ax.axhline(.95, c="#666666", ls="--", label="Nominal 95%")
        ax.set(xlabel="Missing fraction (MCAR)", ylabel="Empirical interval coverage",
               title="5  |  Imputing a value is not recovering uncertainty")
        ax.set_ylim(0, 1.04)
    elif name == "conditional":
        c = ["0.0", "0.5", "1.0", "2.0"]
        x = [float(k) for k in c]
        series(ax, summary, c, "slope_error", "Conditional regression slope error", navy, x)
        series(ax, summary, c, "mean_error", "Marginal outcome-mean error", orange, x)
        ax.axhline(0, c="#666666", ls="--")
        ax.set(xlabel="Strength of MAR selection on observed X",
               ylabel="Error from full-data target",
               title="6  |  Valid conditional slopes can coexist with biased means")
    elif name == "sensitivity":
        c = ["-2.0", "-1.6", "-1.2", "-0.8", "-0.4", "0.0"]
        x = [float(k) for k in c]
        series(ax, summary, c, "assumed_population_mean",
               "Pattern-mixture assumption", orange, x)
        series(ax, summary, c, "full_population_mean",
               "Hidden complete-data truth (simulation)", navy, x)
        ax.axhline(0, c="#777777", ls=":")
        ax.set(xlabel="Assumed missing-minus-observed mean (δ)",
               ylabel="Estimated population mean",
               title="7  |  MNAR sensitivity is an assumption, not an imputation trick")
    elif name == "indicator":
        c = ["indicator_original", "indicator_shifted",
             "no_indicator_original", "no_indicator_shifted"]
        values = [lookup(summary, k, "test_r2") for k in c]
        errors = [1.96 * lookup(summary, k, "test_r2", "se") for k in c]
        ax.bar(["Indicator\noriginal policy", "Indicator\nnew policy",
                "No indicator\noriginal policy", "No indicator\nnew policy"],
               values, yerr=errors, color=[navy, orange, teal, teal], capsize=3)
        ax.axhline(0, c="#666666")
        ax.set(ylabel="Held-out R²",
               title="8  |  Missingness flags can encode a fragile workflow")
    elif name == "positivity":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10.7, 5.2), layout="constrained")
        c = ["0.0", "0.8", "1.6", "2.4"]
        x = [float(k) for k in c]
        for method, label, color in (
            ("complete_case", "Complete case", orange),
            ("oracle_ipw", "Oracle IPW", navy),
            ("clipped_ipw", "Clipped IPW", teal),
        ):
            vals = [math.sqrt(lookup(summary, k, f"{method}_squared_error")) for k in c]
            axes[0].plot(x, vals, label=label, color=color, lw=2, marker="o")
        axes[0].set(xlabel="Selection strength on Z", ylabel="RMSE for sample mean",
                    title="Bias and finite-sample instability")
        axes[0].legend(frameon=False, fontsize=9)
        series(axes[1], summary, c, "relative_ess",
               "Kish effective-n / observed-n", navy, x)
        axes[1].set(xlabel="Selection strength on Z", ylabel="Relative effective sample size",
                    title="Concentrated inverse weights")
        fig.suptitle("9  |  Near-positivity failures limit what weighting can rescue",
                     fontsize=15)
    else:
        plt.close(fig)
        raise KeyError(name)
    for a in fig.axes:
        a.grid(axis="y", color="#e5e9ed", linewidth=.7)
        a.set_axisbelow(True)
        if len(fig.axes) == 1 and name in {
            "zero", "variance", "intervals", "conditional", "sensitivity",
        }:
            a.legend(frameon=False, fontsize=9)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run(design: Design = Design()) -> dict[str, list[Row]]:
    """Regenerate all canonical tables/figures or non-overwriting quick outputs."""

    if not isinstance(design.root, Path):
        raise TypeError("output root must be a pathlib.Path")
    if not isinstance(design.seed, int) or design.seed < 0:
        raise ValueError("seed must be a nonnegative int")
    suffix = "_quick" if design.quick else ""
    summaries: dict[str, list[Row]] = {}
    for i, name in enumerate(EXPERIMENTS):
        rows = []
        reps = (10 if name == "intervals" else 3) if design.quick else REPETITIONS[name]
        for rep in range(reps):
            rows.extend(STUDIES[name](stream(design.seed, name, rep), rep, design.quick))
        summary = summarize(rows)
        write_csv(design.root / "results" / f"{name}_replicates{suffix}.csv", rows)
        write_csv(design.root / "results" / f"{name}_summary{suffix}.csv", summary)
        plot_study(name, summary,
                   design.root / "figures" / f"{FIGURES[i]}{suffix}.png")
        summaries[name] = summary
    return summaries


def main() -> None:
    """CLI for deterministic evidence regeneration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="short independent replications")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    summaries = run(Design(root=args.output, quick=args.quick, seed=args.seed))
    for name in EXPERIMENTS:
        print(f"{name}: {len(summaries[name])} condition-metric combinations")


if __name__ == "__main__":
    main()
