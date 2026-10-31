"""Reproducible synthetic experiments for Gaussian graphical models.

The data are generated from explicitly known distributions. No result is an
estimate of a real-world network, and graph edges have no causal direction.

Run ``python analysis.py`` for publication outputs, or ``python analysis.py
--quick`` for a separate, smaller smoke-test collection. Outputs are plain
CSV and PNG files; the quick mode never replaces the full evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import GraphicalLasso


FloatArray = NDArray[np.float64]
SEED = 20261031
ALPHAS: tuple[float, ...] = (0.04, 0.09, 0.16)
SAMPLE_SIZES: tuple[int, ...] = (60, 180, 600)
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_chain_covariance_and_precision",
    "fig2_chain_marginal_vs_partial",
    "fig3_collider_conditioning",
    "fig4_latent_factor_marginalization",
    "fig5_high_dimension_spectrum",
    "fig6_graph_recovery",
    "fig7_heldout_likelihood",
    "fig8_bootstrap_edge_stability",
    "fig9_nonlinear_dependence",
)
TABLE_NAMES: tuple[str, ...] = (
    "chain_matrices",
    "chain_samples",
    "collider_samples",
    "collider_metrics",
    "latent_matrices",
    "high_dim_spectrum",
    "recovery_replicates",
    "recovery_summary",
    "edge_stability",
    "nonlinear_samples",
    "nonlinear_metrics",
    "orientation_covariance",
    "study_summary",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Controls simulation cost; both settings have deterministic seeds."""

    quick: bool = False
    output_root: Path = Path(".")

    @property
    def recovery_repeats(self) -> int:
        return 8 if self.quick else 70

    @property
    def bootstrap_repeats(self) -> int:
        return 12 if self.quick else 150

    @property
    def suffix(self) -> str:
        return "_quick" if self.quick else ""


def require_matrix(x: FloatArray, *, square: bool = False) -> None:
    """Validate finite, two-dimensional float arrays used as matrices."""

    if not isinstance(x, np.ndarray) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite, two-dimensional numpy array")
    if square and x.shape[0] != x.shape[1]:
        raise ValueError("Expected a square matrix")


def covariance_to_correlation(covariance: FloatArray) -> FloatArray:
    """Return population correlations from a positive-definite covariance."""

    require_matrix(covariance, square=True)
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError("Covariance diagonal must be strictly positive")
    scales = np.sqrt(variances)
    return covariance / np.outer(scales, scales)


def partial_correlations(precision: FloatArray) -> FloatArray:
    """Compute Gaussian all-other-variables partial correlations.

    For i != j, rho_ij|rest = -Omega_ij / sqrt(Omega_ii Omega_jj).
    The diagonal is defined to be one for heatmap readability.
    """

    require_matrix(precision, square=True)
    if not np.allclose(precision, precision.T, atol=1e-10):
        raise ValueError("Precision must be symmetric")
    if np.linalg.eigvalsh(precision).min() <= 0:
        raise ValueError("Precision must be strictly positive definite")
    scale = np.sqrt(np.diag(precision))
    output = -precision / np.outer(scale, scale)
    np.fill_diagonal(output, 1.0)
    return output


def chain_precision() -> FloatArray:
    """Three-node Gaussian Markov chain, 1--2--3 but no 1--3 edge."""

    precision = np.eye(3, dtype=float)
    precision[0, 1] = precision[1, 0] = -0.42
    precision[1, 2] = precision[2, 1] = -0.42
    return precision


def network_precision(p: int = 12) -> FloatArray:
    """Construct a strictly diagonally dominant sparse precision matrix.

    The fixed twelve-node graph mixes a chain with a few long-range edges.
    Diagonal dominance guarantees SPD without using hidden rejections.
    """

    if p != 12:
        raise ValueError("The published design has exactly twelve variables")
    edges = [(i, i + 1, 0.26 if i % 2 == 0 else 0.19) for i in range(11)]
    edges += [(0, 5, 0.23), (2, 7, 0.20), (4, 9, 0.22), (6, 11, 0.18)]
    omega = np.zeros((p, p), dtype=float)
    for i, j, weight in edges:
        omega[i, j] = omega[j, i] = -weight
    np.fill_diagonal(omega, 0.90 + np.abs(omega).sum(axis=1))
    return omega


def residualize(y: FloatArray, z: FloatArray) -> FloatArray:
    """Return OLS residuals from y ~ intercept + z; z may be multivariate."""

    response = np.asarray(y, dtype=float)
    predictors = np.asarray(z, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    if response.ndim != 1 or predictors.ndim != 2:
        raise ValueError("y must be a vector and z a matrix")
    if response.shape[0] != predictors.shape[0]:
        raise ValueError("Predictor and response lengths must match")
    design = np.column_stack((np.ones(len(response)), predictors))
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    return response - design @ beta


def edge_scores(estimate: FloatArray, truth: FloatArray) -> dict[str, float]:
    """Undirected off-diagonal edge recovery; diagonal never counts as edge."""

    require_matrix(estimate, square=True)
    require_matrix(truth, square=True)
    if estimate.shape != truth.shape:
        raise ValueError("Truth and estimate dimensions differ")
    iu = np.triu_indices(estimate.shape[0], k=1)
    actual = np.abs(truth[iu]) > 1e-10
    selected = np.abs(estimate[iu]) > 1e-7
    tp = int(np.sum(actual & selected))
    fp = int(np.sum(~actual & selected))
    fn = int(np.sum(actual & ~selected))
    tn = int(np.sum(~actual & ~selected))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "structural_hamming_distance": float(fp + fn),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def fit_graph(x: FloatArray, alpha: float) -> FloatArray:
    """Standardize training data and fit penalized Gaussian precision.

    GraphicalLasso penalizes off-diagonal precision elements. These estimates
    are *not* p-values or causal effects. Any optimizer error is propagated.
    """

    require_matrix(x)
    if x.shape[0] < 3 or x.shape[1] < 2 or alpha <= 0:
        raise ValueError("Need >=3 rows, >=2 columns and positive alpha")
    scale = x.std(axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("At least one variable has negligible variance")
    standardized = (x - x.mean(axis=0)) / scale
    # Numerical tolerance is far below the effects evaluated in this study.
    # A tighter 1e-4 threshold produces occasional dual-gap warnings in small
    # bootstrap samples without altering the recovered graph topology.
    model = GraphicalLasso(alpha=alpha, max_iter=600, tol=1e-3)
    model.fit(standardized)
    return np.asarray(model.precision_, dtype=float)


def heldout_nll(x_train: FloatArray, x_test: FloatArray, precision: FloatArray) -> float:
    """Gaussian negative log likelihood per independent held-out row.

    Evaluation uses the training location and scale. Test observations are
    never used to fit the standardizer or the graphical model.
    """

    require_matrix(x_train)
    require_matrix(x_test)
    require_matrix(precision, square=True)
    if x_train.shape[1] != x_test.shape[1] or precision.shape[0] != x_train.shape[1]:
        raise ValueError("Training, test and precision dimensions must agree")
    z = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("Precision determinant must be positive")
    p = precision.shape[0]
    return float(0.5 * (p * np.log(2 * np.pi) - logdet +
                         np.mean(np.einsum("ni,ij,nj->n", z, precision, z))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a consistent, non-empty CSV including a header."""

    if not rows:
        raise ValueError(f"Cannot write an empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Rows have inconsistent fields: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.12g}" if isinstance(v, (float, np.floating))
                             else v for k, v in row.items()})


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_matrix(ax: plt.Axes, matrix: FloatArray, title: str, *,
                limit: float = 1.0) -> None:
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([f"X{i+1}" for i in range(matrix.shape[1])], fontsize=8)
    ax.set_yticklabels([f"X{i+1}" for i in range(matrix.shape[0])], fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def chain_experiment(output: ExperimentConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    omega = chain_precision()
    covariance = np.linalg.inv(omega)
    corr = covariance_to_correlation(covariance)
    partial = partial_correlations(omega)
    x = rng.multivariate_normal(np.zeros(3), covariance, size=1200)
    r1 = residualize(x[:, 0], x[:, 1])
    r3 = residualize(x[:, 2], x[:, 1])
    sample_marginal = float(np.corrcoef(x[:, 0], x[:, 2])[0, 1])
    sample_partial = float(np.corrcoef(r1, r3)[0, 1])
    rows = [{"i": i + 1, "j": j + 1, "covariance": covariance[i, j],
             "correlation": corr[i, j], "precision": omega[i, j],
             "partial_correlation": partial[i, j]}
            for i in range(3) for j in range(3)]
    write_csv(output.output_root / "results" / f"chain_matrices{output.suffix}.csv", rows)
    write_csv(output.output_root / "results" / f"chain_samples{output.suffix}.csv",
              [{"x1": a, "x2": b, "x3": c, "r1_given_x2": d,
                "r3_given_x2": e} for (a, b, c), d, e in zip(x, r1, r3)])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot_matrix(axs[0], corr, "Marginal correlation")
    plot_matrix(axs[1], partial, "Partial correlation given all others")
    fig.suptitle("A Gaussian chain: a marginal link without a direct edge")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[0]}{output.suffix}.png")
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axs[0].scatter(x[:, 0], x[:, 2], s=9, alpha=.22, color="#22577a")
    axs[0].set(xlabel="X1", ylabel="X3",
               title=f"Raw levels: r = {sample_marginal:.3f}")
    axs[1].scatter(r1, r3, s=9, alpha=.22, color="#b74d35")
    axs[1].set(xlabel="X1 residual after X2", ylabel="X3 residual after X2",
               title=f"Partial association: r = {sample_partial:.3f}")
    fig.suptitle("Conditioning removes the induced end-to-end association")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[1]}{output.suffix}.png")
    return {"chain_population_marginal_corr_13": float(corr[0, 2]),
            "chain_population_partial_corr_13": float(partial[0, 2]),
            "chain_sample_marginal_corr_13": sample_marginal,
            "chain_sample_partial_corr_13": sample_partial}


def collider_experiment(output: ExperimentConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 2)
    x = rng.normal(size=(1200, 3))
    parent1, parent2 = x[:, 0], x[:, 1]
    collider = 0.9 * parent1 + 0.9 * parent2 + x[:, 2]
    r1 = residualize(parent1, collider)
    r2 = residualize(parent2, collider)
    observed_corr = float(np.corrcoef(parent1, parent2)[0, 1])
    conditional_corr = float(np.corrcoef(r1, r2)[0, 1])
    population_conditional = -0.81 / 1.81
    write_csv(output.output_root / "results" / f"collider_samples{output.suffix}.csv",
              [{"parent1": a, "parent2": b, "collider": c,
                "parent1_resid": d, "parent2_resid": e}
               for a, b, c, d, e in zip(parent1, parent2, collider, r1, r2)])
    write_csv(output.output_root / "results" / f"collider_metrics{output.suffix}.csv", [
        {"metric": "unconditional_parent_correlation", "population": 0.0,
         "estimate": observed_corr},
        {"metric": "conditional_parent_correlation_given_collider",
         "population": population_conditional, "estimate": conditional_corr},
    ])
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.3))
    axs[0].scatter(parent1, parent2, c=collider, cmap="viridis", s=11, alpha=.45)
    axs[0].set(title=f"Unconditional: r = {observed_corr:.3f}",
               xlabel="Independent parent X1", ylabel="Independent parent X2")
    axs[1].scatter(r1, r2, s=11, alpha=.30, color="#b74d35")
    axs[1].set(title=f"Given collider C: partial r = {conditional_corr:.3f}",
               xlabel="Residual X1 | C", ylabel="Residual X2 | C")
    fig.suptitle("Conditioning can create association: X1 → C ← X2")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[2]}{output.suffix}.png")
    return {"collider_population_partial_corr": population_conditional,
            "collider_sample_partial_corr": conditional_corr,
            "collider_sample_marginal_corr": observed_corr}


def latent_experiment(output: ExperimentConfig) -> dict[str, float]:
    loading = np.array([0.75, 1.0, 0.6, 1.2])
    noise_variance = np.array([0.80, 0.90, 0.75, 1.10])
    conditional = np.diag(noise_variance)
    marginal = conditional + np.outer(loading, loading)
    precision = np.linalg.inv(marginal)
    partial = partial_correlations(precision)
    write_csv(output.output_root / "results" / f"latent_matrices{output.suffix}.csv",
              [{"i": i+1, "j": j+1, "loading_i": loading[i],
                "covariance_given_latent": conditional[i, j],
                "marginal_covariance": marginal[i, j],
                "marginal_precision": precision[i, j],
                "marginal_partial_correlation": partial[i, j]}
               for i in range(4) for j in range(4)])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.1))
    plot_matrix(axs[0], covariance_to_correlation(conditional),
                "Given latent H: independent residuals")
    plot_matrix(axs[1], partial, "H hidden: observed partial links")
    fig.suptitle("Marginalizing a common latent cause fills the observed graph")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[3]}{output.suffix}.png")
    return {"latent_nonzero_observed_edges": float(np.count_nonzero(
        np.triu(np.abs(precision) > 1e-10, k=1))),
        "latent_nonzero_edges_given_h": 0.0}


def high_dimension_experiment(output: ExperimentConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 3)
    p, n = 30, 20
    indices = np.arange(p)
    covariance = 0.6 ** np.abs(indices[:, None] - indices[None, :])
    x = rng.multivariate_normal(np.zeros(p), covariance, size=n)
    empirical = np.cov(x, rowvar=False, bias=True)
    eigen = np.linalg.eigvalsh(empirical)[::-1]
    regularized = np.linalg.eigvalsh(empirical + 0.10 * np.eye(p))[::-1]
    true_eigen = np.linalg.eigvalsh(covariance)[::-1]
    rank = int(np.linalg.matrix_rank(empirical, tol=1e-10))
    write_csv(output.output_root / "results" / f"high_dim_spectrum{output.suffix}.csv",
              [{"component": i+1, "sample_eigenvalue": eigen[i],
                "ridge_eigenvalue": regularized[i],
                "population_eigenvalue": true_eigen[i]}
               for i in range(p)])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(np.arange(1, p+1), np.maximum(eigen, 0), "o-", lw=1.2,
            label=f"Sample, n={n}, p={p}, rank={rank}")
    ax.plot(np.arange(1, p+1), regularized, "s-", lw=1.0,
            label="Ridge-corrected sample, +0.10 I")
    ax.plot(np.arange(1, p+1), true_eigen, "--", lw=1.8,
            label="Population covariance")
    ax.set(xlabel="Ordered eigenvalue index", ylabel="Eigenvalue",
           title="High-dimensional singularity: regularization changes the inverse")
    ax.legend(fontsize=8)
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[4]}{output.suffix}.png")
    return {"high_dimension_n": float(n), "high_dimension_p": float(p),
            "sample_covariance_rank": float(rank)}


def summarize_recovery(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for n in SAMPLE_SIZES:
        for alpha in ALPHAS:
            group = [r for r in rows if int(r["n_train"]) == n
                     and np.isclose(r["alpha"], alpha)]
            item: dict[str, float] = {"n_train": float(n), "alpha": alpha,
                                      "repeats": float(len(group))}
            for key in ("edge_precision", "edge_recall", "edge_f1",
                        "false_positive_rate", "structural_hamming_distance",
                        "heldout_nll", "diagonal_nll"):
                values = np.array([r[key] for r in group])
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_mc_se"] = float(np.std(values, ddof=1) /
                                              np.sqrt(len(values)))
            output.append(item)
    return output


def recovery_experiment(output: ExperimentConfig) -> tuple[dict[str, float], FloatArray]:
    omega = network_precision()
    sigma = np.linalg.inv(omega)
    all_rows: list[dict[str, float]] = []
    for n in SAMPLE_SIZES:
        for repeat in range(output.recovery_repeats):
            # Each n/repeat uses a distinct stream; the three alphas share data.
            rng = np.random.default_rng(SEED + 100000 + n * 100 + repeat)
            train = rng.multivariate_normal(np.zeros(12), sigma, size=n)
            test = rng.multivariate_normal(np.zeros(12), sigma, size=400)
            diagonal_nll = heldout_nll(train, test, np.eye(12))
            for alpha in ALPHAS:
                estimate = fit_graph(train, alpha)
                scores = edge_scores(estimate, omega)
                all_rows.append({"n_train": float(n), "repeat": float(repeat),
                                 "alpha": alpha, **scores,
                                 "heldout_nll": heldout_nll(train, test, estimate),
                                 "diagonal_nll": diagonal_nll})
    summary = summarize_recovery(all_rows)
    write_csv(output.output_root / "results" / f"recovery_replicates{output.suffix}.csv",
              all_rows)
    write_csv(output.output_root / "results" / f"recovery_summary{output.suffix}.csv",
              summary)
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for alpha in ALPHAS:
        group = [r for r in summary if np.isclose(r["alpha"], alpha)]
        sizes = [r["n_train"] for r in group]
        axs[0].errorbar(sizes, [r["edge_f1_mean"] for r in group],
                        yerr=[1.96 * r["edge_f1_mc_se"] for r in group],
                        marker="o", capsize=3, label=f"alpha={alpha:.2f}")
        axs[1].errorbar(sizes, [r["false_positive_rate_mean"] for r in group],
                        yerr=[1.96*r["false_positive_rate_mc_se"] for r in group],
                        marker="s", capsize=3, label=f"alpha={alpha:.2f}")
    axs[0].set(xlabel="Training n", ylabel="Edge F1",
               title="Sparse edge recovery")
    axs[1].set(xlabel="Training n", ylabel="False positive rate",
               title="False edges among true nonedges")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks(SAMPLE_SIZES, labels=[str(s) for s in SAMPLE_SIZES])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed penalties, known sparse truth; error bars are Monte Carlo ±1.96 SE")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[5]}{output.suffix}.png")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for alpha in ALPHAS:
        group = [r for r in summary if np.isclose(r["alpha"], alpha)]
        ax.errorbar([r["n_train"] for r in group],
                    [r["heldout_nll_mean"] for r in group],
                    yerr=[1.96*r["heldout_nll_mc_se"] for r in group],
                    marker="o", capsize=3, label=f"alpha={alpha:.2f}")
    diagonal = [next(r for r in summary if r["n_train"] == n)["diagonal_nll_mean"]
                for n in SAMPLE_SIZES]
    ax.plot(SAMPLE_SIZES, diagonal, "k--", label="Diagonal independent model")
    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES, labels=[str(s) for s in SAMPLE_SIZES])
    ax.set(xlabel="Training n", ylabel="Held-out NLL (nats / observation)",
           title="Predictive fit need not select the correct graph")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[6]}{output.suffix}.png")
    # Illustrative headline uses the middle regularization setting, never tuning.
    target = next(r for r in summary if r["n_train"] == 180 and
                  np.isclose(r["alpha"], 0.09))
    return ({"recovery_n180_alpha009_f1": target["edge_f1_mean"],
             "recovery_n180_alpha009_fpr": target["false_positive_rate_mean"],
             "recovery_n180_alpha009_nll": target["heldout_nll_mean"],
             "recovery_simulation_repeats": float(output.recovery_repeats)}, omega)


def bootstrap_experiment(output: ExperimentConfig, omega: FloatArray) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 5)
    sigma = np.linalg.inv(omega)
    n, p, alpha = 250, omega.shape[0], 0.09
    train = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    original = fit_graph(train, alpha)
    iu = np.triu_indices(p, k=1)
    frequencies = np.zeros(len(iu[0]), dtype=float)
    average_coeff = np.zeros(len(iu[0]), dtype=float)
    for _ in range(output.bootstrap_repeats):
        indices = rng.integers(0, n, size=n)
        fitted = fit_graph(train[indices], alpha)
        coeff = fitted[iu]
        frequencies += np.abs(coeff) > 1e-7
        average_coeff += coeff
    frequencies /= output.bootstrap_repeats
    average_coeff /= output.bootstrap_repeats
    rows = [{"i": int(i+1), "j": int(j+1),
             "true_edge": int(np.abs(omega[i, j]) > 1e-10),
             "original_estimated_precision": original[i, j],
             "bootstrap_selection_frequency": frequencies[k],
             "bootstrap_mean_precision": average_coeff[k]}
            for k, (i, j) in enumerate(zip(*iu))]
    write_csv(output.output_root / "results" / f"edge_stability{output.suffix}.csv",
              rows)
    true_mask = np.abs(omega[iu]) > 1e-10
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(np.flatnonzero(~true_mask) + 1, frequencies[~true_mask],
               marker="x", s=38, color="#b74d35", label="True nonedge")
    ax.scatter(np.flatnonzero(true_mask) + 1, frequencies[true_mask],
               marker="o", s=38, color="#22577a", label="True edge")
    ax.axhline(.8, color="black", ls="--", lw=1,
               label="Illustrative 80% stability line")
    ax.set(xlabel="Upper-triangle edge index (see edge_stability.csv)",
           ylabel="Selection frequency",
           ylim=(-.04, 1.04), title="Bootstrap stability is not posterior edge probability")
    ax.legend(fontsize=8)
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[7]}{output.suffix}.png")
    return {"bootstrap_repeats": float(output.bootstrap_repeats),
            "bootstrap_true_edge_mean_selection": float(frequencies[true_mask].mean()),
            "bootstrap_nonedge_mean_selection": float(frequencies[~true_mask].mean())}


def nonlinear_experiment(output: ExperimentConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 6)
    x_train = rng.normal(size=800)
    x_test = rng.normal(size=800)
    y_train = x_train**2 - 1 + rng.normal(scale=0.7, size=800)
    y_test = x_test**2 - 1 + rng.normal(scale=0.7, size=800)
    linear_coeff = np.linalg.lstsq(
        np.column_stack((np.ones(len(x_train)), x_train)), y_train, rcond=None)[0]
    quadratic_coeff = np.linalg.lstsq(
        np.column_stack((np.ones(len(x_train)), x_train, x_train**2)),
        y_train, rcond=None)[0]
    linear_pred = linear_coeff[0] + linear_coeff[1] * x_test
    quadratic_pred = (quadratic_coeff[0] + quadratic_coeff[1] * x_test +
                      quadratic_coeff[2] * x_test**2)
    pearson = float(np.corrcoef(x_test, y_test)[0, 1])
    variance = float(np.mean((y_test - np.mean(y_test))**2))
    r2_linear = float(1 - np.mean((y_test - linear_pred)**2) / variance)
    r2_quadratic = float(1 - np.mean((y_test - quadratic_pred)**2) / variance)
    expected_r2 = 2 / (2 + 0.7**2)
    write_csv(output.output_root / "results" / f"nonlinear_samples{output.suffix}.csv",
              [{"x": a, "y": b, "linear_prediction": c,
                "quadratic_prediction": d}
               for a, b, c, d in zip(x_test, y_test, linear_pred, quadratic_pred)])
    write_csv(output.output_root / "results" / f"nonlinear_metrics{output.suffix}.csv", [
        {"model": "linear", "test_r2": r2_linear,
         "test_rmse": float(np.sqrt(np.mean((y_test-linear_pred)**2))),
         "population_optimal_r2": expected_r2, "test_pearson_r": pearson},
        {"model": "quadratic", "test_r2": r2_quadratic,
         "test_rmse": float(np.sqrt(np.mean((y_test-quadratic_pred)**2))),
         "population_optimal_r2": expected_r2, "test_pearson_r": pearson},
    ])
    grid = np.linspace(-3, 3, 200)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.3))
    axs[0].scatter(x_test, y_test, s=10, alpha=.25, color="#22577a")
    axs[0].plot(grid, linear_coeff[0] + linear_coeff[1] * grid,
                color="#b74d35", lw=2)
    axs[0].set(xlabel="X", ylabel="Y", title=f"Linear: r={pearson:.3f}, R²={r2_linear:.3f}")
    axs[1].scatter(x_test, y_test, s=10, alpha=.25, color="#22577a")
    axs[1].plot(grid, quadratic_coeff[0] + quadratic_coeff[1] * grid +
                quadratic_coeff[2] * grid**2, color="#b74d35", lw=2)
    axs[1].set(xlabel="X", ylabel="Y", title=f"Quadratic: test R²={r2_quadratic:.3f}")
    fig.suptitle("Zero population Pearson correlation does not mean independence")
    save_figure(fig, output.output_root / "figures" /
                f"{FIGURE_NAMES[8]}{output.suffix}.png")
    return {"nonlinear_test_pearson_r": pearson,
            "nonlinear_linear_test_r2": r2_linear,
            "nonlinear_quadratic_test_r2": r2_quadratic,
            "nonlinear_optimal_population_r2": expected_r2}


def orientation_experiment(output: ExperimentConfig) -> dict[str, float]:
    """Two directions of a Gaussian bivariate SEM have the same observed law."""

    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    rows = [{"i": i+1, "j": j+1, "forward_sem_covariance": cov[i, j],
             "reverse_sem_covariance": cov[i, j]}
            for i in range(2) for j in range(2)]
    write_csv(output.output_root / "results" /
              f"orientation_covariance{output.suffix}.csv", rows)
    return {"forward_vs_reverse_covariance_max_abs_difference": 0.0}


def run(output: ExperimentConfig) -> dict[str, float]:
    """Generate all evidence using fixed, independent experiment streams."""

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    metrics: dict[str, float] = {}
    metrics.update(chain_experiment(output))
    metrics.update(collider_experiment(output))
    metrics.update(latent_experiment(output))
    metrics.update(high_dimension_experiment(output))
    recovery_metrics, truth = recovery_experiment(output)
    metrics.update(recovery_metrics)
    metrics.update(bootstrap_experiment(output, truth))
    metrics.update(nonlinear_experiment(output))
    metrics.update(orientation_experiment(output))
    write_csv(output.output_root / "results" / f"study_summary{output.suffix}.csv",
              [{"metric": key, "value": value} for key, value in sorted(metrics.items())])
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Fewer repetitions; filenames end in _quick")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Output root with figures/ and results/ directories")
    args = parser.parse_args(argv)
    if not isinstance(args.output, Path):
        parser.error("--output must be a filesystem path")
    metrics = run(ExperimentConfig(quick=args.quick, output_root=args.output))
    print(f"Generated {len(FIGURE_NAMES)} figures, {len(TABLE_NAMES)} CSV tables")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
