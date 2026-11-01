"""Reproducible synthetic experiments about clustering and interpretation.

The study distinguishes algorithmic partitions from generative mixture
components and scientific populations. Every stochastic result is seeded;
paired parameter interventions reuse observations within a replication.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Record = dict[str, int | float | str]
Rows = list[Record]
SEED = 20261101
EXPERIMENTS = (
    "elbow", "scale", "rings", "ellipses", "dimensions",
    "silhouette", "bic", "stability", "posterior",
)
FIGURES = (
    "fig1_inertia_is_not_rank", "fig2_scale_changes_partitions",
    "fig3_rings_and_density", "fig4_elliptical_mixtures",
    "fig5_irrelevant_dimensions", "fig6_silhouette_component_count",
    "fig7_mixture_bic", "fig8_stable_nonexistent_classes",
    "fig9_soft_assignments", "fig3b_rings_parameter_sensitivity",
)


@dataclass(frozen=True)
class Design:
    """Run controls with independent random streams for each study/repeat."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    seed: int = SEED
    quick: bool = False

    @property
    def repetitions(self) -> dict[str, int]:
        """Independent outer repetition counts, not nested fit initializations."""

        if self.quick:
            return {name: 3 for name in EXPERIMENTS}
        return {
            "elbow": 18, "scale": 24, "rings": 18, "ellipses": 24,
            "dimensions": 18, "silhouette": 24, "bic": 18,
            "stability": 24, "posterior": 24,
        }

    @property
    def suffix(self) -> str:
        """Keep smoke-test outputs separate from canonical full evidence."""

        return "_quick" if self.quick else ""

    def stream(self, experiment: str, replicate: int) -> np.random.Generator:
        """Return an independent, deterministic seed-sequence stream."""

        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        if replicate < 0:
            raise ValueError("replicate must be nonnegative")
        return np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, EXPERIMENTS.index(experiment) + 1, replicate]
            )
        )


def check_matrix(values: FloatArray, *, minimum: int = 3) -> FloatArray:
    """Validate finite numeric 2-D data before fitting a clustering model."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < minimum or x.shape[1] < 1:
        raise ValueError(f"Expected a matrix with at least {minimum} rows")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite matrix entries")
    return x


def kmeans(x: FloatArray, k: int, seed: int) -> KMeans:
    """Fit multiple-start Lloyd-style K-means to a finite matrix."""

    data = check_matrix(x)
    if not 1 <= k <= len(data):
        raise ValueError("Invalid number of clusters")
    return KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=250).fit(data)


def gaussian_mixture(x: FloatArray, k: int, seed: int) -> GaussianMixture:
    """Fit a regularized full-covariance Gaussian mixture with EM."""

    data = check_matrix(x)
    if not 1 <= k <= len(data):
        raise ValueError("Invalid number of mixture components")
    return GaussianMixture(
        n_components=k, covariance_type="full", reg_covar=1e-4,
        n_init=3, random_state=seed, max_iter=250, tol=1e-4,
    ).fit(data)


def marginal_standardize(x: FloatArray) -> FloatArray:
    """Apply within-sample marginal standardization, guarding zero variance."""

    data = check_matrix(x)
    scale = data.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError("Cannot standardize a constant feature")
    return (data - data.mean(axis=0)) / scale


def mixture_sample(
    rng: np.random.Generator,
    sizes: Sequence[int],
    means: Sequence[FloatArray],
    covariances: Sequence[FloatArray],
) -> tuple[FloatArray, IntArray]:
    """Generate labeled Gaussian draws without supplying labels to the fit."""

    if len(sizes) < 1 or len(sizes) != len(means) or len(sizes) != len(covariances):
        raise ValueError("Each component needs size, mean, and covariance")
    if any(n < 2 for n in sizes):
        raise ValueError("Each component needs at least two observations")
    blocks = [
        rng.multivariate_normal(np.asarray(mu), np.asarray(sigma), size=n)
        for n, mu, sigma in zip(sizes, means, covariances, strict=True)
    ]
    x = check_matrix(np.vstack(blocks))
    y = np.concatenate(
        [np.full(n, j, dtype=np.int64) for j, n in enumerate(sizes)]
    )
    return x, y


def rings_sample(
    rng: np.random.Generator, count: int = 600, noise: float = 0.065
) -> tuple[FloatArray, IntArray]:
    """Generate two connected, non-convex annuli with disjoint support bands."""

    if count < 20 or count % 2:
        raise ValueError("count must be even and at least 20")
    if noise <= 0:
        raise ValueError("noise must be positive")
    radius = np.repeat([1.0, 2.0], count // 2)
    angle = rng.uniform(0, 2 * np.pi, size=count)
    data = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    return data + rng.normal(scale=noise, size=(count, 2)), np.repeat(
        np.array([0, 1], dtype=np.int64), count // 2
    )


def mean_and_mcse(values: Sequence[float]) -> tuple[float, float]:
    """Calculate the mean and standard error across independent simulations."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite one-dimensional observations")
    if len(x) == 1:
        return float(x.mean()), math.nan
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize(rows: Rows, groups: tuple[str, ...], metrics: tuple[str, ...]) -> Rows:
    """Aggregate each metric with independent-replication MC standard errors."""

    if not rows or not groups or not metrics:
        raise ValueError("Summary needs rows, grouping keys, and metrics")
    keys = sorted({tuple(r[g] for g in groups) for r in rows})
    output: Rows = []
    for key in keys:
        selected = [r for r in rows if tuple(r[g] for g in groups) == key]
        record: Record = dict(zip(groups, key, strict=True))
        record["repetitions"] = len(selected)
        for metric in metrics:
            mu, se = mean_and_mcse([float(r[metric]) for r in selected])
            record[f"{metric}_mean"] = mu
            record[f"{metric}_mc_se"] = se
        output.append(record)
    return output


def write_csv(path: Path, rows: Rows) -> None:
    """Write rectangular tables in a deterministic, readable CSV format."""

    if not rows:
        raise ValueError("Cannot write an empty table")
    cols = list(rows[0])
    if any(set(row) != set(cols) for row in rows):
        raise ValueError("Rows must have identical columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: format(v, ".12g") if isinstance(v, float) else v
                for k, v in row.items()
            })


def experiment_elbow(design: Design) -> Rows:
    """The empirical K-means distortion always improves with added centers."""

    rows: Rows = []
    for rep in range(design.repetitions["elbow"]):
        rng = design.stream("elbow", rep)
        x, _ = mixture_sample(
            rng, [140, 140, 140],
            [np.array([-4., -1.]), np.array([0., 2.]), np.array([4., -1.])],
            [np.eye(2) * 1.3**2] * 3,
        )
        for k in range(1, 9):
            fit = kmeans(x, k, int(rng.integers(2**31 - 1)))
            rows.append({"replicate": rep, "k": k,
                         "inertia_per_observation": float(fit.inertia_ / len(x))})
    return rows


def experiment_scale(design: Design) -> Rows:
    """Paired nuisance-unit changes rotate a two-center partition."""

    rows: Rows = []
    for rep in range(design.repetitions["scale"]):
        rng = design.stream("scale", rep)
        labels = np.repeat(np.array([0, 1], dtype=np.int64), 220)
        x = np.column_stack((rng.normal(loc=2.0 * labels - 1.0, scale=.65),
                             rng.normal(scale=1.0, size=len(labels))))
        for factor in (.25, .5, 1., 2., 4.):
            shifted = x * np.array([1., factor])
            seed = int(rng.integers(2**31 - 1))
            raw = kmeans(shifted, 2, seed).labels_
            standardized = kmeans(marginal_standardize(shifted), 2, seed).labels_
            rows.append({"replicate": rep, "nuisance_scale": factor,
                         "raw_ari": float(adjusted_rand_score(labels, raw)),
                         "standardized_ari": float(
                             adjusted_rand_score(labels, standardized))})
    return rows


def experiment_rings(design: Design) -> Rows:
    """Density connectivity captures annuli which centroid Voronoi cells cut."""

    rows: Rows = []
    for rep in range(design.repetitions["rings"]):
        rng = design.stream("rings", rep)
        x, y = rings_sample(rng)
        baseline = adjusted_rand_score(
            y, kmeans(x, 2, int(rng.integers(2**31 - 1))).labels_
        )
        for eps in (.08, .12, .18, .30, .70):
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(x)
            found = len(set(labels) - {-1})
            rows.append({
                "replicate": rep, "eps": eps, "kmeans_ari": float(baseline),
                "dbscan_ari": float(adjusted_rand_score(y, labels)),
                "density_clusters": found,
                "assigned_fraction": float(np.mean(labels >= 0)),
            })
    return rows


def experiment_ellipses(design: Design) -> Rows:
    """Full-covariance likelihood can fit shapes missed by Euclidean centers."""

    rows: Rows = []
    for rep in range(design.repetitions["ellipses"]):
        rng = design.stream("ellipses", rep)
        x, y = mixture_sample(
            rng, [260, 260],
            [np.array([-.7, 0.]), np.array([.7, 0.])],
            [np.diag([.25**2, 2.1**2]), np.diag([.25**2, .3**2])],
        )
        raw = kmeans(x, 2, int(rng.integers(2**31 - 1)))
        model = gaussian_mixture(x, 2, int(rng.integers(2**31 - 1)))
        rows.append({"replicate": rep,
                     "kmeans_ari": float(adjusted_rand_score(y, raw.labels_)),
                     "gmm_ari": float(adjusted_rand_score(y, model.predict(x))),
                     "kmeans_inertia": float(raw.inertia_ / len(x)),
                     "gmm_bic": float(model.bic(x))})
    return rows


def experiment_dimensions(design: Design) -> Rows:
    """Appending independent features can bury a low-dimensional partition."""

    rows: Rows = []
    for rep in range(design.repetitions["dimensions"]):
        rng = design.stream("dimensions", rep)
        y = np.repeat(np.array([0, 1], dtype=np.int64), 170)
        signal = np.column_stack((
            rng.normal(loc=2.1 * y - 1.05, scale=.65),
            rng.normal(scale=.6, size=len(y)),
        ))
        all_noise = rng.normal(scale=1.65, size=(len(y), 150))
        for noise_dims in (0, 5, 20, 60, 150):
            x = np.column_stack((signal, all_noise[:, :noise_dims]))
            fit = kmeans(x, 2, int(rng.integers(2**31 - 1)))
            rows.append({"replicate": rep, "noise_dimensions": noise_dims,
                         "total_dimensions": x.shape[1],
                         "kmeans_ari": float(adjusted_rand_score(y, fit.labels_)),
                         "inertia_per_observation": float(fit.inertia_ / len(x))})
    return rows


def experiment_silhouette(design: Design) -> Rows:
    """Compact-separation scores need not equal generative component count."""

    rows: Rows = []
    for rep in range(design.repetitions["silhouette"]):
        rng = design.stream("silhouette", rep)
        x, y = mixture_sample(
            rng, [150, 150, 150],
            [np.array([-4., 0.]), np.array([1., 0.]), np.array([2., 0.])],
            [np.eye(2) * .8**2] * 3,
        )
        for k in (2, 3, 4, 5):
            fit = kmeans(x, k, int(rng.integers(2**31 - 1)))
            score = silhouette_score(x, fit.labels_, sample_size=350,
                                     random_state=rep + design.seed)
            rows.append({"replicate": rep, "k": k, "true_components": 3,
                         "silhouette": float(score),
                         "ari_to_generator": float(adjusted_rand_score(y, fit.labels_))})
    return rows


def experiment_bic(design: Design) -> Rows:
    """BIC balances Gaussian-mixture fit against declared parameter count."""

    rows: Rows = []
    for rep in range(design.repetitions["bic"]):
        rng = design.stream("bic", rep)
        x, y = mixture_sample(
            rng, [140, 160],
            [np.array([-2., -1.]), np.array([2., 1.])],
            [np.diag([.8**2, .5**2]), np.diag([.6**2, 1.1**2])],
        )
        for k in range(1, 6):
            model = gaussian_mixture(x, k, int(rng.integers(2**31 - 1)))
            rows.append({"replicate": rep, "k": k, "true_components": 2,
                         "bic": float(model.bic(x)),
                         "ari_to_generator": float(
                             adjusted_rand_score(y, model.predict(x)))})
    return rows


def experiment_stability(design: Design) -> Rows:
    """One continuous Gaussian can admit a stable two-region partition."""

    rows: Rows = []
    for rep in range(design.repetitions["stability"]):
        rng = design.stream("stability", rep)
        population = rng.multivariate_normal(
            [0., 0.], np.diag([3.0**2, .5**2]), size=650
        )
        # The same untouched anchors are scored by both bootstrap fits.
        anchors = population[:250]
        fit_pool = population[250:]
        a = fit_pool[rng.integers(len(fit_pool), size=len(fit_pool))]
        b = fit_pool[rng.integers(len(fit_pool), size=len(fit_pool))]
        fit_a = kmeans(a, 2, int(rng.integers(2**31 - 1)))
        fit_b = kmeans(b, 2, int(rng.integers(2**31 - 1)))
        labels_a, labels_b = fit_a.predict(anchors), fit_b.predict(anchors)
        rows.append({
            "replicate": rep, "known_gaussian_components": 1,
            "fitted_k": 2,
            "bootstrap_ari": float(adjusted_rand_score(labels_a, labels_b)),
            "anchor_small_cluster_fraction": float(
                min(np.mean(labels_a == 0), np.mean(labels_a == 1))
            ),
        })
    return rows


def experiment_posterior(design: Design) -> Rows:
    """Overlap creates uncertain responsibilities despite hard partitions."""

    rows: Rows = []
    for rep in range(design.repetitions["posterior"]):
        rng = design.stream("posterior", rep)
        for distance in (.6, 1.2, 2.0, 3.5):
            x, y = mixture_sample(
                rng, [210, 210],
                [np.array([-distance / 2, 0.]),
                 np.array([distance / 2, 0.])],
                [np.eye(2)] * 2,
            )
            model = gaussian_mixture(x, 2, int(rng.integers(2**31 - 1)))
            probabilities = model.predict_proba(x)
            max_p = np.max(probabilities, axis=1)
            rows.append({
                "replicate": rep, "mean_separation": distance,
                "gmm_ari": float(adjusted_rand_score(y, model.predict(x))),
                "ambiguous_fraction": float(np.mean(max_p < .8)),
                "mean_max_responsibility": float(max_p.mean()),
            })
    return rows


EXPERIMENT_FUNCTIONS: Mapping[str, Callable[[Design], Rows]] = {
    "elbow": experiment_elbow,
    "scale": experiment_scale,
    "rings": experiment_rings,
    "ellipses": experiment_ellipses,
    "dimensions": experiment_dimensions,
    "silhouette": experiment_silhouette,
    "bic": experiment_bic,
    "stability": experiment_stability,
    "posterior": experiment_posterior,
}

GROUPS: Mapping[str, tuple[str, ...]] = {
    "elbow": ("k",), "scale": ("nuisance_scale",), "rings": ("eps",),
    "ellipses": ("replicate",), "dimensions": ("noise_dimensions",),
    "silhouette": ("k",), "bic": ("k",), "stability": ("replicate",),
    "posterior": ("mean_separation",),
}
METRICS: Mapping[str, tuple[str, ...]] = {
    "elbow": ("inertia_per_observation",),
    "scale": ("raw_ari", "standardized_ari"),
    "rings": ("kmeans_ari", "dbscan_ari", "density_clusters", "assigned_fraction"),
    "ellipses": ("kmeans_ari", "gmm_ari"),
    "dimensions": ("kmeans_ari", "inertia_per_observation"),
    "silhouette": ("silhouette", "ari_to_generator"),
    "bic": ("bic", "ari_to_generator"),
    "stability": ("bootstrap_ari", "anchor_small_cluster_fraction"),
    "posterior": ("gmm_ari", "ambiguous_fraction", "mean_max_responsibility"),
}


def overall_summary(name: str, rows: Rows) -> Rows:
    """Return one summary row per experimental condition, never per replicate."""

    if name not in EXPERIMENTS:
        raise ValueError("Unknown experiment")
    if name in ("ellipses", "stability"):
        # Each row is an independent repeat of the same condition.
        metrics = METRICS[name]
        result: Record = {"scenario": name, "repetitions": len(rows)}
        for metric in metrics:
            mean, mcse = mean_and_mcse([float(x[metric]) for x in rows])
            result[metric + "_mean"] = mean
            result[metric + "_mc_se"] = mcse
        return [result]
    return summarize(rows, GROUPS[name], METRICS[name])


def _series(rows: Rows, xkey: str, ykey: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract an ordered mean/error series from a summary CSV representation."""

    ordered = sorted(rows, key=lambda row: float(row[xkey]))
    return (
        np.array([float(r[xkey]) for r in ordered]),
        np.array([float(r[ykey + "_mean"]) for r in ordered]),
        np.array([float(r[ykey + "_mc_se"]) for r in ordered]),
    )


def _error_curve(
    ax: plt.Axes, rows: Rows, xkey: str, ykey: str,
    label: str, color: str, marker: str = "o",
) -> None:
    """Display independent-run mean ± 1.96 MC standard errors."""

    x, y, se = _series(rows, xkey, ykey)
    ax.errorbar(
        x, y, yerr=1.96 * se, label=label, color=color,
        marker=marker, capsize=3, linewidth=2,
    )


def _save(fig: Figure, design: Design, filename: str) -> None:
    """Write a publication PNG and close its Figure to avoid state leakage."""

    target = design.root / "figures"
    target.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target / f"{filename}{design.suffix}.png", dpi=170)
    plt.close(fig)


def plot_elbow(design: Design, tables: Mapping[str, Rows]) -> None:
    """Show monotone distortion and its successive marginal reductions."""

    rows = tables["elbow"]
    x, y, se = _series(rows, "k", "inertia_per_observation")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    axes[0].errorbar(x, y, 1.96 * se, color="#196D84",
                     marker="o", linewidth=2, capsize=2)
    axes[0].axvline(3, color="#BB622C", linestyle="--", label="Generator: 3 Gaussians")
    axes[0].set(xlabel="K-means centers", ylabel="Within-cluster squared error / row",
                title="More centers always reduce sample distortion")
    axes[0].legend(fontsize=8)
    reductions = (y[:-1] - y[1:]) / y[:-1]
    axes[1].bar(x[1:], reductions, color="#196D84", width=.65)
    axes[1].set(xlabel="Centers after increment", ylabel="Relative distortion reduction",
                title="An elbow requires a judgment")
    _save(fig, design, FIGURES[0])


def plot_scale(design: Design, tables: Mapping[str, Rows]) -> None:
    """Display sensitivity to a nuisance feature's arbitrary units."""

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    rows = tables["scale"]
    _error_curve(ax, rows, "nuisance_scale", "raw_ari", "Covariance units",
                 "#BD5934")
    _error_curve(ax, rows, "nuisance_scale", "standardized_ari",
                 "Marginally standardized", "#1B7A81", "s")
    ax.set(xlabel="Multiplier on unrelated feature", ylabel="ARI to generated labels",
           title="The same observations, different numerical metric", ylim=(-.06, 1.05))
    ax.legend()
    _save(fig, design, FIGURES[1])


def plot_rings(design: Design, tables: Mapping[str, Rows]) -> None:
    """Show centroid partition and density-connectivity hyperparameter sweep."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x, y = rings_sample(np.random.default_rng(1309))
    km = kmeans(x, 2, 5).labels_
    db = DBSCAN(eps=.30, min_samples=5).fit_predict(x)
    for ax, label, name in zip(axes, (km, db),
                               ("K-means: two Voronoi cells",
                                "DBSCAN: density-connected rings"), strict=True):
        ax.scatter(x[:, 0], x[:, 1], c=label, cmap="viridis",
                   s=6, alpha=.7, rasterized=True)
        ax.set(xlabel="x", ylabel="y", title=name, aspect="equal")
    fig.suptitle("A non-convex generator; K-means has no ring-shaped cells", fontsize=11)
    _save(fig, design, FIGURES[2])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = tables["rings"]
    _error_curve(ax, rows, "eps", "dbscan_ari", "DBSCAN ARI", "#186D84")
    _error_curve(ax, rows, "eps", "kmeans_ari", "K-means ARI (fixed k=2)", "#C3633E")
    ax.set(xlabel="DBSCAN neighborhood radius ε",
           ylabel="Adjusted Rand index to generator", ylim=(-.08, 1.05),
           title="Density-connected partitions require a chosen scale")
    ax.legend()
    _save(fig, design, FIGURES[9])


def plot_ellipses(design: Design, tables: Mapping[str, Rows]) -> None:
    """Compare a Euclidean Voronoi partition with full-covariance mixture fit."""

    summary = tables["ellipses"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.2))
    means = [float(summary[x + "_mean"]) for x in ("kmeans_ari", "gmm_ari")]
    errors = [1.96 * float(summary[x + "_mc_se"]) for x in ("kmeans_ari", "gmm_ari")]
    axes[0].bar(["K-means", "GMM"], means, color=["#C3633E", "#186D84"],
                yerr=errors, capsize=4)
    axes[0].set(ylabel="ARI to generated components", ylim=(-.05, 1.05),
                title=f"{design.repetitions['ellipses']} independent samples")
    rng = np.random.default_rng(742)
    x, _ = mixture_sample(
        rng, [260, 260], [np.array([-.7, 0.]), np.array([.7, 0.])],
        [np.diag([.25**2, 2.1**2]), np.diag([.25**2, .3**2])],
    )
    fits = (kmeans(x, 2, 7).labels_, gaussian_mixture(x, 2, 7).predict(x))
    for ax, labels, title in zip(axes[1:], fits,
                                  ("Euclidean centers", "Full-covariance EM"),
                                  strict=True):
        ax.scatter(x[:, 0], x[:, 1], c=labels, s=7, alpha=.65,
                   cmap="viridis", rasterized=True)
        ax.set(xlabel="x", ylabel="y", title=title)
    _save(fig, design, FIGURES[3])


def plot_dimensions(design: Design, tables: Mapping[str, Rows]) -> None:
    """Display deterioration under appended independent nuisance dimensions."""

    fig, ax = plt.subplots(figsize=(8.3, 4.3))
    _error_curve(ax, tables["dimensions"], "noise_dimensions", "kmeans_ari",
                 "K-means ARI", "#186D84")
    ax.set(xlabel="Additional unrelated Gaussian dimensions",
           ylabel="ARI to the two generated groups",
           title="Separation becomes harder as irrelevant geometry grows",
           ylim=(-.08, 1.05))
    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    _save(fig, design, FIGURES[4])


def plot_silhouette(design: Design, tables: Mapping[str, Rows]) -> None:
    """Show a compactness criterion and its mismatch to generator labels."""

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    rows = tables["silhouette"]
    _error_curve(axes[0], rows, "k", "silhouette", "Mean silhouette", "#186D84")
    _error_curve(axes[1], rows, "k", "ari_to_generator", "Generator ARI", "#B15A37")
    for ax in axes:
        ax.axvline(3, color="gray", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Requested K-means centers")
        ax.set_xticks([2, 3, 4, 5])
    axes[0].set(ylabel="Average silhouette width",
                title="Geometric separation criterion")
    axes[1].set(ylabel="ARI against known generating labels",
                title="Generative labels are a different target")
    _save(fig, design, FIGURES[5])


def plot_bic(design: Design, tables: Mapping[str, Rows], reps: Mapping[str, Rows]) -> None:
    """Plot BIC selection and its frequency across independent datasets."""

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    rows = reps["bic"]
    diffs: list[float] = []
    ks: list[int] = []
    winners: list[int] = []
    for rep in range(design.repetitions["bic"]):
        group = [r for r in rows if int(r["replicate"]) == rep]
        low = min(float(r["bic"]) for r in group)
        winner = min(group, key=lambda r: float(r["bic"]))
        winners.append(int(winner["k"]))
        for item in group:
            ks.append(int(item["k"]))
            diffs.append(float(item["bic"]) - low)
    for k in range(1, 6):
        values = [v for v, kk in zip(diffs, ks, strict=True) if kk == k]
        mu, se = mean_and_mcse(values)
        axes[0].errorbar(k, mu, 1.96 * se, marker="o", capsize=3,
                         color="#196D84")
    axes[0].plot([], [], color="#196D84", label="ΔBIC to best within each sample")
    axes[0].set(xlabel="Gaussian components", ylabel="Mean ΔBIC (smaller is better)",
                title="Penalized fit is not direct proof of classes")
    axes[1].bar(range(1, 6), [winners.count(k) / len(winners) for k in range(1, 6)],
                color="#196D84")
    axes[1].set(xlabel="BIC-selected components", ylabel="Fraction of datasets",
                ylim=(0, 1.05), title="Selection varies across samples")
    for ax in axes:
        ax.axvline(2, color="#BD5934", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, 6))
    _save(fig, design, FIGURES[6])


def plot_stability(design: Design, tables: Mapping[str, Rows], reps: Mapping[str, Rows]) -> None:
    """High bootstrap agreement can occur for one continuous Gaussian."""

    values = [float(r["bootstrap_ari"]) for r in reps["stability"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    rng = np.random.default_rng(1828)
    sample = rng.multivariate_normal([0., 0.], np.diag([9., .25]), size=650)
    labels = kmeans(sample, 2, 7).labels_
    axes[0].scatter(sample[:, 0], sample[:, 1], c=labels, s=7,
                    cmap="viridis", alpha=.65, rasterized=True)
    axes[0].set(xlabel="x", ylabel="y", title="One Gaussian, two fitted regions")
    axes[1].hist(values, bins=np.linspace(0, 1, 11), color="#196D84",
                 edgecolor="white")
    axes[1].set(xlabel="Bootstrap ARI on common holdout anchors",
                ylabel="Independent repetitions", xlim=(0, 1),
                title="Stable split does not imply two populations")
    _save(fig, design, FIGURES[7])


def plot_posterior(design: Design, tables: Mapping[str, Rows]) -> None:
    """Show uncertainty of Gaussian responsibilities under component overlap."""

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    rows = tables["posterior"]
    _error_curve(axes[0], rows, "mean_separation", "ambiguous_fraction",
                 "Max responsibility < 0.8", "#BE633F")
    _error_curve(axes[1], rows, "mean_separation", "gmm_ari",
                 "Hard-label ARI", "#196D84")
    axes[0].set(xlabel="Separation of the two generating means",
                ylabel="Fraction with ambiguous fitted membership",
                ylim=(-.04, 1.05), title="Soft responsibility reveals overlap")
    axes[1].set(xlabel="Separation of the two generating means",
                ylabel="ARI to known generating component",
                ylim=(-.06, 1.05), title="A hard label hides assignment uncertainty")
    _save(fig, design, FIGURES[8])


def generate_figures(
    design: Design, summaries: Mapping[str, Rows], replicates: Mapping[str, Rows]
) -> None:
    """Render original figures from saved experiment summaries and draws."""

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    plot_elbow(design, summaries)
    plot_scale(design, summaries)
    plot_rings(design, summaries)
    plot_ellipses(design, summaries)
    plot_dimensions(design, summaries)
    plot_silhouette(design, summaries)
    plot_bic(design, summaries, replicates)
    plot_stability(design, summaries, replicates)
    plot_posterior(design, summaries)


def run(design: Design) -> dict[str, Rows]:
    """Run every study, write 18 CSVs, and render publication-ready figures."""

    if design.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if not isinstance(design.root, Path):
        raise TypeError("Output root must be a Path")
    summaries: dict[str, Rows] = {}
    replicates: dict[str, Rows] = {}
    for name in EXPERIMENTS:
        rows = EXPERIMENT_FUNCTIONS[name](design)
        result = overall_summary(name, rows)
        write_csv(design.root / "results" / f"{name}_replicates{design.suffix}.csv", rows)
        write_csv(design.root / "results" / f"{name}_summary{design.suffix}.csv", result)
        replicates[name], summaries[name] = rows, result
    generate_figures(design, summaries, replicates)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point with separate full and quick artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Short smoke run with separately suffixed outputs")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="Directory for generated figures and results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    config = Design(root=args.output, quick=args.quick, seed=args.seed)
    summary = run(config)
    print(f"Generated {len(FIGURES)} figures and {2 * len(EXPERIMENTS)} tables")
    for key, value in (
        ("scale_first_raw_ari", summary["scale"][0]["raw_ari_mean"]),
        ("ellipses_kmeans_ari", summary["ellipses"][0]["kmeans_ari_mean"]),
        ("ellipses_gmm_ari", summary["ellipses"][0]["gmm_ari_mean"]),
        ("high_dims_ari", summary["dimensions"][-1]["kmeans_ari_mean"]),
        ("stability_ari", summary["stability"][0]["bootstrap_ari_mean"]),
    ):
        print(f"{key}: {float(value):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

