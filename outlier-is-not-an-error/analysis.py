"""Nine reproducible studies of robustness, influence and anomaly-score limits.

All observations are synthetic. Full mode writes nine original figures and
eighteen versioned CSV files. Quick mode never overwrites full-mode evidence.
Independent outer repetitions, rather than pooled individual rows, define
Monte Carlo uncertainty. The word 'outlier' is not an error label here.
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
from scipy.optimize import brentq
from scipy.stats import binom, chi2, norm, theilslopes, trim_mean, t
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor


ROOT = Path(__file__).resolve().parent
SEED = 20261102
EXPERIMENTS = (
    "contamination", "tails", "intervals", "regression", "leverage",
    "multivariate", "minority", "tail_risk", "shift",
)
FIGURES = (
    "fig1_contamination_and_location", "fig2_heavy_tail_sample_size",
    "fig3_coverage_and_estimands", "fig4_vertical_outliers_regression",
    "fig5_leverage_is_not_influence", "fig6_robust_multivariate_distance",
    "fig7_legitimate_minority_flagged", "fig8_tail_deletion_changes_risk",
    "fig9_thresholds_under_shift",
)
REPETITIONS = {
    "contamination": 30, "tails": 30, "intervals": 500,
    "regression": 30, "leverage": 30, "multivariate": 24,
    "minority": 24, "tail_risk": 30, "shift": 30,
}
Row = dict[str, str | int | float | None]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Design:
    """Output destination, quick/full intensity, and independent base seed."""

    root: Path = ROOT
    quick: bool = False
    seed: int = SEED


def check_vector(x: NDArray[np.floating], minimum: int = 3) -> FloatArray:
    """Reject nonfinite, multidimensional, or too-short estimator input."""

    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError("expected a finite one-dimensional vector")
    return a


def stream(seed: int, name: str, rep: int) -> np.random.Generator:
    """Stable independent SeedSequence stream; unaffected by loop ordering."""

    if not isinstance(seed, int) or seed < 0 or rep < 0 or name not in EXPERIMENTS:
        raise ValueError("invalid seed, study name, or replicate")
    return np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(name.encode("ascii")), rep])
    )


def observation(rep: int, condition: str, metric: str, value: float,
                theory: float | None = None) -> Row:
    """Validate one finite long-format experimental observation."""

    if not condition or not metric or rep < 0:
        raise ValueError("nonempty keys and nonnegative replicate required")
    if not math.isfinite(value) or (theory is not None and not math.isfinite(theory)):
        raise ValueError("observation and theory must be finite")
    return {"replicate": rep, "condition": condition, "metric": metric,
            "value": float(value), "theory": theory}


def summarize(rows: list[Row]) -> list[Row]:
    """Across-independent-replication mean and normal 95% Monte Carlo CI."""

    if not rows:
        raise ValueError("empty replication set")
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), str(row["metric"])), []).append(row)
    result = []
    for (condition, metric), group in sorted(groups.items()):
        theories = {r["theory"] for r in group}
        if len(theories) != 1:
            raise ValueError("theory must agree within a condition and metric")
        replicates = [int(r["replicate"]) for r in group]
        if len(replicates) != len(set(replicates)):
            raise ValueError("one observation per outer replication and key")
        values = np.asarray([r["value"] for r in group], dtype=float)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        mean = float(np.mean(values))
        se = sd / math.sqrt(len(values))
        result.append({"condition": condition, "metric": metric, "n": len(values),
                       "mean": mean, "sd": sd, "se": se,
                       "lower95": mean - 1.96*se, "upper95": mean + 1.96*se,
                       "theory": group[0]["theory"]})
    return result


def lookup(rows: list[Row], condition: str, metric: str,
           field: str = "mean") -> float:
    """Get one summary cell by exact key, with loud missing-value failure."""

    hits = [r for r in rows if r["condition"] == condition and r["metric"] == metric]
    if len(hits) != 1 or hits[0].get(field) is None:
        raise KeyError((condition, metric, field))
    return float(hits[0][field])


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write consistently ordered CSV evidence with LF line endings."""

    if not rows:
        raise ValueError("cannot write empty evidence")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("inconsistent CSV columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def huber_location(x: FloatArray, threshold: float = 1.345,
                   scale: float = 1.) -> float:
    """Solve sum clip((x-theta)/scale, -threshold, threshold)=0.

    This function deliberately uses supplied, known scale. It does not
    estimate scale robustly from the contaminated sample, and the article
    must not describe it as a fully adaptive Huber location estimator.
    """

    values = check_vector(x)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return lo
    psi = lambda theta: float(np.clip((values-theta)/scale,
                                      -threshold, threshold).sum())
    return float(brentq(psi, lo, hi, xtol=1e-12))


def contaminated_normal(rng: np.random.Generator, n: int, eps: float,
                        displacement: float) -> FloatArray:
    """Independent mixture (1-eps) N(0,1) + eps N(displacement,1)."""

    if n < 3 or not 0 <= eps < 1 or not math.isfinite(displacement):
        raise ValueError("invalid contamination mixture")
    shifted = rng.random(n) < eps
    return rng.normal(size=n) + displacement*shifted


def mixture_median(eps: float, displacement: float) -> float:
    """Exact population median of the two-unit-variance-normal mixture."""

    if not 0 <= eps < 1 or displacement <= 0:
        raise ValueError("invalid mixture parameters")
    f = lambda x: (1-eps)*norm.cdf(x) + eps*norm.cdf(x-displacement) - .5
    return float(brentq(f, -10., displacement+10.))


def sign_median_interval(x: FloatArray, alpha: float = .05) -> tuple[float, float]:
    """Distribution-free conservative sign/order-statistic CI for median.

    Requires IID draws from a continuous population with a unique median.
    k is the greatest integer for which P{Bin(n,.5)<=k-1}<=alpha/2.
    """

    values = check_vector(x)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    n = len(values)
    k = int(binom.ppf(alpha/2, n, .5))
    if k < 1 or k > n//2:
        raise ValueError("sample too small for a finite median interval")
    ordered = np.sort(values)
    return float(ordered[k-1]), float(ordered[n-k])


def ols_fit(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Intercept and slope for one-predictor OLS with explicit validation."""

    xx, yy = check_vector(x), check_vector(y)
    if len(xx) != len(yy) or np.ptp(xx) < 1e-12:
        raise ValueError("equal-length vectors with variation in x required")
    slope = float(np.sum((xx-xx.mean())*(yy-yy.mean())) / np.sum((xx-xx.mean())**2))
    return float(yy.mean()-slope*xx.mean()), slope


def regression_diagnostics(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Hat leverages and Cook's D with p=2, OLS residual MSE, full fit.

    Cook D=e_i^2/(p*MSE) * h_ii/(1-h_ii)^2; the MSE uses n-p degrees
    of freedom. This is a linear-model diagnostic, not an error posterior.
    """

    xx, yy = check_vector(x), check_vector(y)
    if len(xx) != len(yy) or len(xx) < 5:
        raise ValueError("regression diagnostics need >=5 paired rows")
    a, b = ols_fit(xx, yy)
    leverage = 1/len(xx) + (xx-xx.mean())**2 / np.sum((xx-xx.mean())**2)
    residual = yy - a - b*xx
    mse = float(np.sum(residual**2)/(len(xx)-2))
    if mse <= 0 or np.any(leverage >= 1):
        raise ValueError("degenerate residual variance or hat matrix")
    cooks = residual**2/(2*mse) * leverage/(1-leverage)**2
    return leverage.astype(float), cooks.astype(float)


def expected_shortfall_lognormal(p: float, sigma: float) -> tuple[float, float]:
    """Lognormal(mu=0,sigma) q_p and ES_p=E[Y|Y>=q_p]."""

    if not 0 < p < 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid tail probability or scale")
    z = float(norm.ppf(p))
    quantile = math.exp(sigma*z)
    expected_shortfall = math.exp(.5*sigma**2)*norm.sf(z-sigma)/(1-p)
    return quantile, expected_shortfall


def population_distances(x: NDArray[np.floating], robust: bool,
                         seed: int = 0) -> FloatArray:
    """Classical or MCD squared Mahalanobis distances on a 2D sample."""

    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 20 or not np.isfinite(a).all():
        raise ValueError("expected finite n-by-2 data, n>=20")
    # A singular design makes classical Mahalanobis inversion undefined and
    # invalidates the full-rank scatter model used by MCD as well.
    if np.linalg.matrix_rank(a-a.mean(axis=0)) < 2:
        raise ValueError("expected a full-rank two-dimensional sample")
    if robust:
        estimator = MinCovDet(support_fraction=.7, random_state=seed).fit(a)
        return np.asarray(estimator.mahalanobis(a), dtype=float)
    center = np.mean(a, axis=0)
    covariance = np.cov(a, rowvar=False)
    centered = a-center
    return np.asarray(np.einsum("ni,ij,nj->n", centered,
                                np.linalg.inv(covariance), centered), dtype=float)


def threshold_alert(scores: FloatArray, budget: float) -> NDArray[np.bool_]:
    """Deterministic top-score budget, never called an error probability."""

    a = check_vector(scores)
    if not 0 < budget < 1:
        raise ValueError("budget must be strictly inside (0,1)")
    k = max(1, int(math.ceil(len(a)*budget)))
    selected = np.zeros(len(a), dtype=bool)
    selected[np.argsort(a, kind="stable")[-k:]] = True
    return selected


def contamination_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Sensitivity of four location estimators to shifted mixture mass."""

    rows = []
    for eps in (0., .01, .03, .10, .20):
        x = contaminated_normal(rng, 180 if quick else 500, eps, 10.)
        cond = f"{eps:.2f}"
        for method, value in (
            ("mean", float(x.mean())), ("median", float(np.median(x))),
            ("trimmed_10pct", float(trim_mean(x, .1))),
            ("huber_known_scale", huber_location(x)),
        ):
            theory = 10*eps if method == "mean" else None
            rows.append(observation(rep, cond, method, value, theory))
    return rows


def tails_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Per-outer-rep 75th percentile |location error| across 15 inner draws.

    The Cauchy *mean* is undefined. Zero is a location parameter for all
    three families, not a claimed finite population mean for the Cauchy.
    """

    rows = []
    for family in ("normal", "t3", "cauchy"):
        for n in (40, 200, 800):
            errors: dict[str, list[float]] = {
                "q75_abs_error_sample_mean": [], "q75_abs_error_median": [],
                "q75_abs_error_trimmed_10pct": [],
            }
            for _ in range(5 if quick else 15):
                if family == "normal":
                    x = rng.normal(size=n)
                elif family == "t3":
                    x = rng.standard_t(df=3, size=n)/math.sqrt(3.)
                else:
                    x = rng.standard_cauchy(size=n)
                errors["q75_abs_error_sample_mean"].append(abs(float(np.mean(x))))
                errors["q75_abs_error_median"].append(abs(float(np.median(x))))
                errors["q75_abs_error_trimmed_10pct"].append(
                    abs(float(trim_mean(x, .1))))
            for method, error in errors.items():
                rows.append(observation(rep, f"{family}:{n}", method,
                                        float(np.quantile(error, .75))))
    return rows


def intervals_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Coverage for *different* mixture targets, mean and median respectively."""

    rows = []
    n = 80
    for eps in (0., .02, .05):
        x = contaminated_normal(rng, n, eps, 20.)
        target_mean, target_median = eps*20, mixture_median(eps, 20.)
        half = float(t.ppf(.975, n-1)*np.std(x, ddof=1)/math.sqrt(n))
        center = float(np.mean(x))
        median_lo, median_hi = sign_median_interval(x)
        cond = f"{eps:.2f}"
        for method, coverage, width in (
            ("mean_student_t", float(center-half <= target_mean <= center+half), 2*half),
            ("median_sign", float(median_lo <= target_median <= median_hi),
             median_hi-median_lo),
        ):
            rows.append(observation(rep, f"{cond}:{method}", "coverage", coverage, .95))
            rows.append(observation(rep, f"{cond}:{method}", "width", width))
    return rows


def regression_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Random vertical contamination biases intercept; slope need not be biased."""

    rows = []
    for eps in (0., .02, .08, .20):
        n = 170 if quick else 360
        x = rng.normal(size=n)
        y = 1. + 2.*x + rng.normal(scale=.7, size=n)
        y = y + 18.*(rng.random(n) < eps)
        a_ols, b_ols = ols_fit(x, y)
        huber = HuberRegressor(epsilon=1.35, max_iter=200).fit(x[:, None], y)
        b_theil, a_theil, _, _ = theilslopes(y, x, method="separate")
        cond = f"{eps:.2f}"
        for method, intercept, slope in (
            ("ols", a_ols, b_ols),
            ("huber", float(huber.intercept_), float(huber.coef_[0])),
            ("theil_sen", float(a_theil), float(b_theil)),
        ):
            rows.append(observation(rep, cond, f"{method}_intercept_error",
                                    abs(intercept-1.)))
            rows.append(observation(rep, cond, f"{method}_slope_error",
                                    abs(slope-2.)))
    return rows


def leverage_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """A high-leverage on-line row need not exert high parameter influence."""

    x = rng.normal(size=80 if quick else 150)
    y = 1.+2.*x+rng.normal(scale=.7, size=len(x))
    rows = []
    for case in ("ordinary", "on_line", "off_line"):
        if case == "ordinary":
            xx, yy = x.copy(), y.copy()
            target_idx = int(np.argmax(np.abs(xx)))
        else:
            xx = np.r_[x, 12.]
            yy = np.r_[y, 25. if case == "on_line" else -20.]
            target_idx = len(xx)-1
        h, cooks = regression_diagnostics(xx, yy)
        a_full, b_full = ols_fit(xx, yy)
        keep = np.arange(len(xx)) != target_idx
        a_minus, b_minus = ols_fit(xx[keep], yy[keep])
        rows.extend((
            observation(rep, case, "leverage", h[target_idx]),
            observation(rep, case, "cooks_distance", cooks[target_idx]),
            observation(rep, case, "leave_one_out_slope_shift", abs(b_full-b_minus)),
            observation(rep, case, "leave_one_out_intercept_shift", abs(a_full-a_minus)),
        ))
    return rows


def multivariate_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Contaminant masking in fitted covariance versus MCD robust scatter."""

    rows = []
    covariance = np.asarray([[1., .65], [.65, 1.]])
    n = 190 if quick else 380
    cutoff = float(chi2.ppf(.99, df=2))
    for eps in (.02, .10, .20):
        n_bad = max(1, round(n*eps))
        clean = rng.multivariate_normal([0., 0.], covariance, size=n-n_bad)
        shifted = rng.multivariate_normal([5., 5.], covariance*.2, size=n_bad)
        a = np.vstack((clean, shifted))
        bad = np.r_[np.zeros(len(clean), dtype=bool), np.ones(n_bad, dtype=bool)]
        for method, squared in (
            ("classical", population_distances(a, False)),
            ("mcd", population_distances(a, True, seed=rep)),
        ):
            flagged = squared > cutoff
            cond = f"{eps:.2f}"
            rows.append(observation(rep, cond, f"{method}_tpr", float(flagged[bad].mean())))
            rows.append(observation(rep, cond, f"{method}_fpr", float(flagged[~bad].mean())))
    return rows


def minority_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """A valid small mixture component absorbs much of a 5% alert budget."""

    rows = []
    n = 450 if quick else 950
    for share in (.01, .05, .15, .25):
        n_min = max(1, round(n*share))
        major = rng.normal(size=(n-n_min, 2))
        minor = rng.normal(loc=(4., 0.), scale=.65, size=(n_min, 2))
        values = np.vstack((major, minor))
        minority = np.r_[np.zeros(len(major), dtype=bool), np.ones(len(minor), dtype=bool)]
        forest = IsolationForest(n_estimators=80 if quick else 100,
                                 max_samples=min(256, n), random_state=rep,
                                 n_jobs=1).fit(values)
        anomaly_scores = -forest.score_samples(values)
        flags = threshold_alert(anomaly_scores, .05)
        cond = f"{share:.2f}"
        rows.extend((
            observation(rep, cond, "minority_flagged_fraction", float(flags[minority].mean())),
            observation(rep, cond, "majority_flagged_fraction", float(flags[~minority].mean())),
            observation(rep, cond, "flagged_share_valid_minority",
                        float(minority[flags].mean())),
            observation(rep, cond, "score_cutoff", float(anomaly_scores[flags].min())),
        ))
    return rows


def tail_risk_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Deleting/capping legitimate tail draws changes target risk quantities."""

    sigma = 1.1
    true_mean = math.exp(sigma**2/2)
    true_q, true_es = expected_shortfall_lognormal(.99, sigma)
    x = rng.lognormal(0., sigma, size=2500 if quick else 9000)
    cap = float(np.quantile(x, .99))
    methods = {"raw": x, "drop_top_1pct": x[x <= cap],
               "winsor_top_1pct": np.minimum(x, cap)}
    rows = []
    for method, observed in methods.items():
        threshold = float(np.quantile(observed, .99))
        es = float(np.mean(observed[observed >= threshold]))
        for metric, value, truth in (
            ("mean_ratio", float(np.mean(observed))/true_mean, 1.),
            ("q99_ratio", threshold/true_q, 1.),
            ("es99_ratio", es/true_es, 1.),
        ):
            rows.append(observation(rep, method, metric, value, truth))
    return rows


def shift_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Fixed threshold drifts; recentering budget can hide changing background."""

    n = 2500 if quick else 11000
    base = rng.normal(size=n)
    baseline_threshold = float(np.quantile(np.abs(base), .99))
    rows = []
    for name in ("normal", "t3_unit_var", "sd_1.5", "location_0.6"):
        if name == "normal":
            background = rng.normal(size=n)
        elif name == "t3_unit_var":
            background = rng.standard_t(df=3, size=n)/math.sqrt(3.)
        elif name == "sd_1.5":
            background = rng.normal(scale=1.5, size=n)
        else:
            background = rng.normal(loc=.6, size=n)
        anomalies = rng.normal(loc=5., scale=1., size=250 if quick else 500)
        updated_threshold = float(np.quantile(np.abs(background), .99))
        for method, threshold in (("fixed", baseline_threshold),
                                  ("recalibrated", updated_threshold)):
            # Exactly 1% of a continuous current-background sample lies above
            # its recalibrated 99% quantile. That is an operational budget,
            # not the analytic tail rate of a shifted fixed threshold.
            theory = .01 if method == "recalibrated" else None
            rows.append(observation(rep, name, f"{method}_background_alert_rate",
                                    float(np.mean(np.abs(background)>threshold)), theory))
            rows.append(observation(rep, name, f"{method}_anomaly_recall",
                                    float(np.mean(np.abs(anomalies)>threshold))))
    return rows


STUDIES: dict[str, Callable[[np.random.Generator, int, bool], list[Row]]] = {
    "contamination": contamination_study,
    "tails": tails_study,
    "intervals": intervals_study,
    "regression": regression_study,
    "leverage": leverage_study,
    "multivariate": multivariate_study,
    "minority": minority_study,
    "tail_risk": tail_risk_study,
    "shift": shift_study,
}


def series(ax: plt.Axes, summary: list[Row], cond: list[str],
           metric: str, label: str, color: str,
           x: list[float] | None = None) -> None:
    """Plot outer-replicate estimate and ±1.96 standard errors."""

    xx = x if x is not None else list(range(len(cond)))
    yy = [lookup(summary, c, metric) for c in cond]
    se = [1.96*lookup(summary, c, metric, "se") for c in cond]
    ax.errorbar(xx, yy, yerr=se, marker="o", capsize=3,
                lw=2.1, color=color, label=label)


def plot_study(name: str, summary: list[Row], filename: Path) -> None:
    """Render one independently generated, titled, labeled original graphic."""

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 11, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    navy, orange, teal, purple = "#163858", "#d66a35", "#00887f", "#7c5499"
    fig, ax = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
    if name == "contamination":
        conditions = ["0.00", "0.01", "0.03", "0.10", "0.20"]
        x = [float(c) for c in conditions]
        for metric, label, color in (
            ("mean", "Sample mean", orange), ("median", "Median", navy),
            ("trimmed_10pct", "10% trimmed mean", teal),
            ("huber_known_scale", "Huber (known scale)", purple),
        ):
            series(ax, summary, conditions, metric, label, color, x)
        ax.plot(x, 10*np.asarray(x), linestyle="--", color="#555555",
                label="Mixture population mean = 10ε")
        ax.set(xlabel="Shifted-mixture probability ε", ylabel="Location estimate",
               title="1  |  A small contaminated fraction moves the mean")
    elif name == "tails":
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), layout="constrained",
                                 sharey=True)
        for ax, family, label in zip(axes, ("normal", "t3", "cauchy"),
                                      ("Normal", "Student t (ν=3), variance 1",
                                       "Cauchy, undefined mean")):
            x = [40., 200., 800.]
            cond = [f"{family}:{int(n)}" for n in x]
            for method, method_label, color in (
                ("q75_abs_error_sample_mean", "Mean of sample", orange),
                ("q75_abs_error_median", "Median", navy),
                ("q75_abs_error_trimmed_10pct", "10% trim", teal),
            ):
                series(ax, summary, cond, method, method_label, color, x)
            ax.set(xscale="log", yscale="log", xlabel="Sample size",
                   title=label)
        axes[0].set_ylabel("75th percentile |estimate − 0| (15 inner draws)")
        axes[2].legend(frameon=False, fontsize=9)
        fig.suptitle("2  |  More data do not stabilize a Cauchy sample mean", fontsize=15)
    elif name == "intervals":
        conditions = ["0.00", "0.02", "0.05"]
        x = [float(c) for c in conditions]
        for method, label, color in (
            ("mean_student_t", "Student-t CI for mixture mean", orange),
            ("median_sign", "Sign CI for mixture median", navy),
        ):
            series(ax, summary, [f"{c}:{method}" for c in conditions],
                   "coverage", label, color, x)
        ax.axhline(.95, c="#555555", ls="--", label="Nominal 95%")
        ax.set(xlabel="Probability of +20 draw", ylabel="Empirical target coverage",
               ylim=(0., 1.02),
               title="3  |  Honest uncertainty requires naming the estimand")
    elif name == "regression":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.1), layout="constrained")
        cond = ["0.00", "0.02", "0.08", "0.20"]
        x = [float(c) for c in cond]
        for ax, target, title in zip(axes, ("intercept_error", "slope_error"),
                                     ("Intercept error |â−1|", "Slope error |b̂−2|")):
            for method, label, color in (("ols", "OLS", orange),
                                         ("huber", "Huber regression", navy),
                                         ("theil_sen", "Theil–Sen", teal)):
                series(ax, summary, cond, f"{method}_{target}", label, color, x)
            ax.set(xlabel="Vertical contamination probability", ylabel=title,
                   title=title)
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle("4  |  Random vertical outliers mostly bias the OLS intercept",
                     fontsize=14)
    elif name == "leverage":
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.9), layout="constrained")
        c = ["ordinary", "on_line", "off_line"]
        display = ["Largest ordinary |x|", "Extreme x, on line", "Extreme x, off line"]
        for ax, metric, heading in zip(axes,
            ("leverage", "cooks_distance", "leave_one_out_slope_shift"),
            ("Hat leverage hᵢᵢ", "Cook's distance Dᵢ", "Slope change if deleted"),
        ):
            means = [lookup(summary, k, metric) for k in c]
            errors = [1.96*lookup(summary, k, metric, "se") for k in c]
            ax.bar(range(3), means, yerr=errors, capsize=3,
                   color=[navy, teal, orange])
            ax.set_xticks(range(3), display, rotation=15, ha="right")
            ax.set_title(heading)
        fig.suptitle("5  |  Leverage, residual and influence are different objects",
                     fontsize=15)
    elif name == "multivariate":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), layout="constrained")
        c = ["0.02", "0.10", "0.20"]
        x = [float(k) for k in c]
        for ax, outcome, heading in zip(axes, ("tpr", "fpr"),
                                        ("Flagged shifted-cluster rows",
                                         "Flagged original Gaussian rows")):
            for method, label, color in (("classical", "Classical covariance", orange),
                                         ("mcd", "MCD robust covariance", navy)):
                series(ax, summary, c, f"{method}_{outcome}", label, color, x)
            ax.set(xlabel="Shifted mixture share", ylabel="Fraction flagged",
                   ylim=(-.03, 1.03), title=heading)
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("6  |  Contamination can mask itself in classical covariance",
                     fontsize=15)
    elif name == "minority":
        c = ["0.01", "0.05", "0.15", "0.25"]
        x = [float(k) for k in c]
        series(ax, summary, c, "minority_flagged_fraction", "Valid minority flagged", orange, x)
        series(ax, summary, c, "majority_flagged_fraction", "Majority flagged", navy, x)
        ax.axhline(.05, c="#555555", ls="--", label="Total alert budget (5%)")
        ax.set(xlabel="Legitimate minority prevalence", ylabel="Within-group alert rate",
               ylim=(-.03, 1.03),
               title="7  |  Rarity can be mistaken for an actionable anomaly")
    elif name == "tail_risk":
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.9), layout="constrained")
        c = ["raw", "drop_top_1pct", "winsor_top_1pct"]
        display = ["Raw sample", "Delete top 1%", "Cap top 1%"]
        for ax, metric, title in zip(axes, ("mean_ratio", "q99_ratio", "es99_ratio"),
                                     ("Mean", "99th percentile", "99% expected shortfall")):
            values = [lookup(summary, k, metric) for k in c]
            errors = [1.96*lookup(summary, k, metric, "se") for k in c]
            ax.bar(range(3), values, yerr=errors, capsize=3,
                   color=[navy, orange, teal])
            ax.axhline(1., color="#666666", ls="--")
            ax.set_xticks(range(3), display, rotation=15, ha="right")
            ax.set(title=title, ylabel="Estimate / full-population truth")
        fig.suptitle("8  |  Tail surgery rewrites the quantity you claim to estimate",
                     fontsize=15)
    elif name == "shift":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.0), layout="constrained")
        c = ["normal", "t3_unit_var", "sd_1.5", "location_0.6"]
        labels = ["Normal", "t(3), unit variance", "SD ×1.5", "Mean +0.6"]
        positions = np.arange(len(c))
        for ax, outcome, heading in zip(axes,
            ("background_alert_rate", "anomaly_recall"),
            ("Unlabeled background alert rate", "Recall on planted +5 observations"),
        ):
            for off, method, color in ((-.18, "fixed", orange),
                                       (.18, "recalibrated", navy)):
                means = [lookup(summary, k, f"{method}_{outcome}") for k in c]
                errors = [1.96*lookup(summary, k, f"{method}_{outcome}", "se") for k in c]
                ax.bar(positions+off, means, width=.34, yerr=errors,
                       color=color, capsize=3, label=method)
            ax.set_xticks(positions, labels, rotation=16, ha="right")
            ax.set(title=heading, ylabel="Fraction", ylim=(0, 1.03))
        axes[0].axhline(.01, c="#555555", ls="--", label="1% budget")
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle("9  |  Recalibration maintains a budget, not the meaning of an alarm",
                     fontsize=15)
    else:
        plt.close(fig)
        raise KeyError(name)
    for axis in fig.axes:
        axis.grid(axis="y", color="#e5e9ed", linewidth=.7)
        axis.set_axisbelow(True)
    if name in {"contamination", "intervals", "minority"}:
        ax.legend(frameon=False, fontsize=9)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run(design: Design = Design()) -> dict[str, list[Row]]:
    """Generate nine experiments, their raw replications, summaries and figures."""

    if not isinstance(design.root, Path):
        raise TypeError("output root must be a pathlib.Path")
    if not isinstance(design.seed, int) or design.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not design.quick and design.seed != SEED and design.root.resolve() == ROOT.resolve():
        raise ValueError("non-canonical full seed requires an explicit output directory")
    suffix = "_quick" if design.quick else ""
    summaries: dict[str, list[Row]] = {}
    for i, name in enumerate(EXPERIMENTS):
        repetitions = (16 if name == "intervals" else 3) if design.quick else REPETITIONS[name]
        observations: list[Row] = []
        for rep in range(repetitions):
            observations.extend(STUDIES[name](stream(design.seed, name, rep),
                                              rep, design.quick))
        summary = summarize(observations)
        write_csv(design.root/"results"/f"{name}_replicates{suffix}.csv", observations)
        write_csv(design.root/"results"/f"{name}_summary{suffix}.csv", summary)
        plot_study(name, summary, design.root/"figures"/f"{FIGURES[i]}{suffix}.png")
        summaries[name] = summary
    return summaries


def main() -> None:
    """Command-line entry point for full or non-overwriting quick regeneration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    summaries = run(Design(root=args.output, quick=args.quick, seed=args.seed))
    for name in EXPERIMENTS:
        print(f"{name}: {len(summaries[name])} condition-metric combinations")


if __name__ == "__main__":
    main()
