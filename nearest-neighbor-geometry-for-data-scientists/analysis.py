"""Seeded geometry, neighbor, and metric experiments for article #17.

All data are synthetic. Examples intentionally change the generating
distribution between experiments; the observed patterns are not universal
theorems about every high-dimensional dataset.

Run ``python analysis.py`` to regenerate canonical PNG/CSV evidence, or
``python analysis.py --quick`` for separate, inexpensive smoke outputs.
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
from scipy.spatial.distance import cdist, pdist
from scipy.special import gammaln
from scipy.stats import norm, skew


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SEED = 20261031
DIMS: tuple[int, ...] = (2, 8, 32, 128, 512)
PROJECTIONS: tuple[int, ...] = (8, 16, 32, 64, 128)
NOISE_SCALES: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
LATENT_NOISE: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0)
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_pairwise_distance_distributions",
    "fig2_concentration_and_contrast",
    "fig3_knn_accuracy_vs_dimension",
    "fig4_nuisance_scale",
    "fig5_random_projection_distortion",
    "fig6_hubness_by_dimension",
    "fig7_hubness_and_norms",
    "fig8_intrinsic_spectral_dimension",
    "fig9_metric_choice_tasks",
)
TABLE_NAMES: tuple[str, ...] = (
    "distance_samples", "distance_summary",
    "neighbor_contrast_replicates", "neighbor_contrast_summary",
    "classification_replicates", "classification_summary",
    "scale_replicates", "scale_summary",
    "jl_replicates", "jl_summary",
    "hub_replicates", "hub_summary", "hub_example_counts",
    "intrinsic_replicates", "intrinsic_summary",
    "metric_replicates", "metric_summary", "study_summary",
)


@dataclass(frozen=True)
class Design:
    """Declared simulation effort and isolated output destination."""

    quick: bool = False
    output_root: Path = Path(".")

    @property
    def suffix(self) -> str:
        return "_quick" if self.quick else ""

    @property
    def repeats(self) -> dict[str, int]:
        if self.quick:
            return {"contrast": 3, "classification": 5, "scale": 5,
                    "projection": 4, "hub": 3, "intrinsic": 3,
                    "metric": 5}
        return {"contrast": 14, "classification": 36, "scale": 28,
                "projection": 20, "hub": 12, "intrinsic": 14,
                "metric": 36}


def require_matrix(x: FloatArray) -> None:
    """Reject invalid or non-finite distance inputs early."""

    if (not isinstance(x, np.ndarray) or x.ndim != 2 or 0 in x.shape
            or not np.isfinite(x).all()):
        raise ValueError("Expected a nonempty, finite, two-dimensional numpy array")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty, rectangular CSV without scientific-data coercion."""

    if not rows:
        raise ValueError(f"No data to write: {path}")
    cols = list(rows[0])
    if any(set(row) != set(cols) for row in rows):
        raise ValueError(f"Inconsistent CSV schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{value:.12g}"
                             if isinstance(value, (float, np.floating)) else value
                             for k, value in row.items()})


def save_figure(fig: plt.Figure, design: Design, name: str) -> None:
    path = design.output_root / "figures" / f"{name}{design.suffix}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize(replicates: list[dict[str, Any]], keys: tuple[str, ...],
              measures: tuple[str, ...]) -> list[dict[str, Any]]:
    """Compute across-independent-replicate means and Monte Carlo SEs."""

    if not replicates:
        raise ValueError("Cannot summarize an empty experiment")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in replicates:
        group_key = tuple(row[k] for k in keys)
        groups.setdefault(group_key, []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(keys, group_key))
        result["repeats"] = len(group)
        for name in measures:
            values = np.array([float(row[name]) for row in group])
            result[f"{name}_mean"] = float(np.mean(values))
            result[f"{name}_mc_se"] = float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            ) if len(values) > 1 else float("nan")
        output.append(result)
    return output


def gaussian_distance_moments(d: int) -> tuple[float, float, float]:
    """Exact moments of ||X-Y|| for independent N_d(0,I) draws.

    The distance has distribution sqrt(2)*chi_d; E[D²]=2d.
    """

    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError("Dimension must be a positive integer")
    mean = float(2 * np.exp(gammaln((d + 1) / 2) - gammaln(d / 2)))
    variance = max(0.0, 2 * d - mean**2)
    return mean, float(np.sqrt(variance)), float(np.sqrt(variance) / mean)


def distance_experiment(design: Design) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    samples: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for d in DIMS:
        # These pair differences are independent, not a pseudoreplicated
        # collection of all n(n-1)/2 dependent dataset pairs.
        delta = rng.normal(scale=np.sqrt(2), size=(2100, d))
        distances = np.linalg.norm(delta, axis=1)
        mean_theory, sd_theory, cv_theory = gaussian_distance_moments(d)
        summary.append({"dimension": d, "pairs": len(distances),
                        "distance_mean": float(distances.mean()),
                        "distance_sd": float(distances.std(ddof=1)),
                        "distance_cv": float(distances.std(ddof=1)/distances.mean()),
                        "theory_mean": mean_theory, "theory_sd": sd_theory,
                        "theory_cv": cv_theory,
                        "normalized_p05": float(np.quantile(distances / mean_theory, .05)),
                        "normalized_p95": float(np.quantile(distances / mean_theory, .95))})
        samples.extend({"dimension": d, "pair_index": i,
                        "distance": dist, "distance_over_population_mean": dist/mean_theory}
                       for i, dist in enumerate(distances))
    write_csv(design.output_root / "results" / f"distance_samples{design.suffix}.csv", samples)
    write_csv(design.output_root / "results" / f"distance_summary{design.suffix}.csv", summary)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(.4, 1.65, 62)
    for d in DIMS:
        x = [r["distance_over_population_mean"] for r in samples
             if r["dimension"] == d]
        ax.hist(x, bins=bins, histtype="step", density=True, lw=1.6,
                label=f"p={d}")
    ax.set(xlabel="Distance / exact population mean", ylabel="Probability density",
           title="Pairwise Euclidean distance concentrates in isotropic Gaussian space")
    ax.legend()
    save_figure(fig, design, FIGURE_NAMES[0])
    return {"distance_p2_cv": float(summary[0]["distance_cv"]),
            "distance_p512_cv": float(summary[-1]["distance_cv"]),
            "distance_p512_theory_cv": float(summary[-1]["theory_cv"])}


def contrast_experiment(design: Design) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for d in DIMS:
        for rep in range(design.repeats["contrast"]):
            rng = np.random.default_rng(SEED + 2000 + d*100 + rep)
            candidates = rng.normal(size=(260, d))
            queries = rng.normal(size=(100, d))
            distances = cdist(queries, candidates, metric="euclidean")
            minimum, maximum = distances.min(axis=1), distances.max(axis=1)
            contrast = (maximum-minimum)/minimum
            rows.append({"dimension": d, "repeat": rep, "candidates": 260,
                         "queries": 100, "mean_relative_contrast": float(contrast.mean()),
                         "median_relative_contrast": float(np.median(contrast))})
    summary = summarize(rows, ("dimension",), (
        "median_relative_contrast", "mean_relative_contrast"))
    write_csv(design.output_root / "results" /
              f"neighbor_contrast_replicates{design.suffix}.csv", rows)
    write_csv(design.output_root / "results" /
              f"neighbor_contrast_summary{design.suffix}.csv", summary)
    distance_summary = [{"dimension": d, "empirical_cv": 0.0,
                         "theory_cv": gaussian_distance_moments(d)[2]} for d in DIMS]
    rng = np.random.default_rng(SEED + 1)
    for row in distance_summary:
        d = int(row["dimension"])
        delta = rng.normal(scale=np.sqrt(2), size=(2100, d))
        ds = np.linalg.norm(delta, axis=1)
        row["empirical_cv"] = float(ds.std(ddof=1)/ds.mean())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(DIMS, [r["empirical_cv"] for r in distance_summary],
                "o-", label="Sample CV")
    axs[0].plot(DIMS, [r["theory_cv"] for r in distance_summary],
                "--", label="Exact Gaussian CV")
    axs[0].set(ylabel="SD(distance) / mean(distance)",
               title="Relative distance variability")
    for row in summary:
        axs[1].errorbar(row["dimension"], row["median_relative_contrast_mean"],
                        yerr=1.96*row["median_relative_contrast_mc_se"],
                        fmt="o", color="#b74d35", capsize=3)
    axs[1].plot([r["dimension"] for r in summary],
                [r["median_relative_contrast_mean"] for r in summary],
                color="#b74d35")
    axs[1].set(ylabel="Median query (farthest - nearest) / nearest",
               title="Finite-sample neighbor contrast", yscale="log")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks(DIMS, labels=[str(v) for v in DIMS])
        ax.set_xlabel("Ambient dimension p")
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle("Two distinct concentration diagnostics; candidate pool fixed at 260")
    save_figure(fig, design, FIGURE_NAMES[1])
    return {"contrast_p2": float(summary[0]["median_relative_contrast_mean"]),
            "contrast_p512": float(summary[-1]["median_relative_contrast_mean"])}


def knn_accuracy(x_train: FloatArray, y_train: IntArray,
                 x_test: FloatArray, y_test: IntArray, *,
                 neighbors: int = 7, distance: str = "euclidean") -> float:
    """Evaluate a deterministic, odd-k majority vote on independent test data.

    'norm_difference' is a pseudodistance: two distinct vectors can have
    the same norm. It is intentionally not called a metric on R^p.
    """

    require_matrix(x_train)
    require_matrix(x_test)
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Feature dimensions do not match")
    if len(y_train) != len(x_train) or len(y_test) != len(x_test):
        raise ValueError("Feature and label lengths differ")
    if (not isinstance(neighbors, int) or neighbors < 1
            or neighbors > len(y_train) or neighbors % 2 == 0):
        raise ValueError("Neighbors must be odd and between one and train size")
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_test, [0, 1]).all():
        raise ValueError("Only binary labels are supported")
    if distance == "norm_difference":
        matrix = np.abs(np.linalg.norm(x_test, axis=1)[:, None]
                        - np.linalg.norm(x_train, axis=1)[None, :])
    elif distance in ("euclidean", "cosine"):
        if distance == "cosine" and (np.any(np.linalg.norm(x_train, axis=1) == 0)
                                     or np.any(np.linalg.norm(x_test, axis=1) == 0)):
            raise ValueError("Cosine dissimilarity requires nonzero vectors")
        matrix = cdist(x_test, x_train, metric=distance)
    else:
        raise ValueError("Unknown distance rule")
    indices = np.argpartition(matrix, kth=neighbors-1, axis=1)[:, :neighbors]
    votes = y_train[indices].sum(axis=1)
    prediction = (votes > neighbors/2).astype(np.int64)
    return float(np.mean(prediction == y_test))


def generate_signal_and_nuisance(rng: np.random.Generator,
                                 n_train: int = 260, n_test: int = 160,
                                 max_dimension: int = 512
                                 ) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """Binary Gaussian classes differ only in the first coordinate.

    Bayes-optimal sign rule is x[0]>0, with accuracy Phi(0.85).
    Other coordinates are mutually independent N(0,1) nuisances.
    """

    n = n_train + n_test
    labels = rng.integers(0, 2, size=n, dtype=np.int64)
    x = rng.normal(size=(n, max_dimension))
    x[:, 0] += (2*labels-1)*0.85
    return x[:n_train], labels[:n_train], x[n_train:], labels[n_train:]


def classification_experiment(design: Design) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for rep in range(design.repeats["classification"]):
        rng = np.random.default_rng(SEED + 4000 + rep)
        train, ytrain, test, ytest = generate_signal_and_nuisance(rng)
        oracle = float(np.mean((test[:, 0] > 0).astype(np.int64) == ytest))
        for d in DIMS:
            accuracy = knn_accuracy(train[:, :d], ytrain, test[:, :d], ytest)
            rows.append({"dimension": d, "repeat": rep, "neighbors": 7,
                         "train_size": 260, "test_size": 160,
                         "knn_accuracy": accuracy, "oracle_accuracy": oracle})
    summary = summarize(rows, ("dimension",), ("knn_accuracy", "oracle_accuracy"))
    write_csv(design.output_root / "results" / f"classification_replicates{design.suffix}.csv", rows)
    write_csv(design.output_root / "results" / f"classification_summary{design.suffix}.csv", summary)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.errorbar(DIMS, [r["knn_accuracy_mean"] for r in summary],
                yerr=[1.96*r["knn_accuracy_mc_se"] for r in summary],
                marker="o", capsize=3, label="7-NN, Euclidean all included features")
    ax.plot(DIMS, [r["oracle_accuracy_mean"] for r in summary], "--",
            label="Oracle sign of first coordinate (same tests)")
    ax.axhline(norm.cdf(.85), color="gray", ls=":",
               label="Population Bayes accuracy, Phi(0.85)")
    ax.axhline(.5, color="black", lw=.8, ls="-.", label="Equal-prior guessing")
    ax.set(xscale="log", xlabel="Included dimension p", ylabel="Independent-test accuracy",
           ylim=(.38,.88),title="Irrelevant dimensions displace useful nearest neighbors")
    ax.set_xticks(DIMS, labels=[str(d) for d in DIMS]); ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, design, FIGURE_NAMES[2])
    return {"knn_p2_acc":float(summary[0]["knn_accuracy_mean"]),
            "knn_p512_acc":float(summary[-1]["knn_accuracy_mean"]),
            "classification_repeats": float(design.repeats["classification"]),
            "bayes_accuracy":float(norm.cdf(.85))}


def scale_experiment(design: Design) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    for rep in range(design.repeats["scale"]):
        rng = np.random.default_rng(SEED + 6000 + rep)
        train, ytrain, test, ytest = generate_signal_and_nuisance(
            rng, max_dimension=128)
        for scale in NOISE_SCALES:
            # Change only the 127 nuisance coordinates. Scaling the features
            # changes the Euclidean geometry, not the Bayes label mechanism.
            train_s = train.copy(); test_s = test.copy()
            train_s[:, 1:] *= scale; test_s[:, 1:] *= scale
            rows.append({"nuisance_scale": scale, "repeat": rep,
                         "knn_accuracy": knn_accuracy(train_s,ytrain,test_s,ytest)})
    summary = summarize(rows,("nuisance_scale",),("knn_accuracy",))
    write_csv(design.output_root / "results" / f"scale_replicates{design.suffix}.csv", rows)
    write_csv(design.output_root / "results" / f"scale_summary{design.suffix}.csv", summary)
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.errorbar([r["nuisance_scale"] for r in summary],
                [r["knn_accuracy_mean"] for r in summary],
                yerr=[1.96*r["knn_accuracy_mc_se"] for r in summary],
                marker="o", capsize=3, color="#22577a")
    ax.axhline(norm.cdf(.85),color="gray",ls="--",label="Bayes reference")
    ax.set(xlabel="Scale multiplying 127 irrelevant variables",
           ylabel="7-NN test accuracy",ylim=(.4,.9),
           title="Distances change when irrelevant features are rescaled")
    ax.grid(alpha=.2); ax.legend()
    save_figure(fig,design,FIGURE_NAMES[3])
    return {"scaled_noise_01_accuracy":float(summary[0]["knn_accuracy_mean"]),
            "scaled_noise_20_accuracy":float(summary[-1]["knn_accuracy_mean"])}


def jl_projection(x: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    """Gaussian random map N(0,1/k); no training or label information used."""

    require_matrix(x)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Target dimension must be positive")
    matrix = rng.normal(size=(x.shape[1], k)) / np.sqrt(k)
    return x @ matrix


def jl_experiment(design: Design) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 8)
    points = rng.normal(size=(170, 256))
    original = pdist(points)
    rows: list[dict[str, Any]] = []
    for k in PROJECTIONS:
        for rep in range(design.repeats["projection"]):
            projected_rng = np.random.default_rng(SEED + 8000 + k*100 + rep)
            mapped = jl_projection(points,k,projected_rng)
            distort = np.abs(pdist(mapped)-original)/original
            rows.append({"target_dimension":k,"repeat":rep,
                         "pairs":len(original),
                         "median_absolute_relative_error":float(np.median(distort)),
                         "p95_absolute_relative_error":float(np.quantile(distort,.95)),
                         "maximum_absolute_relative_error":float(np.max(distort))})
    summary=summarize(rows,("target_dimension",),(
        "median_absolute_relative_error","p95_absolute_relative_error",
        "maximum_absolute_relative_error"))
    write_csv(design.output_root / "results" / f"jl_replicates{design.suffix}.csv",rows)
    write_csv(design.output_root / "results" / f"jl_summary{design.suffix}.csv",summary)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for measure,label in (("median_absolute_relative_error","Median pair distortion"),
                           ("p95_absolute_relative_error","95th percentile pair distortion"),
                           ("maximum_absolute_relative_error","Maximum pair distortion")):
        ax.errorbar(PROJECTIONS,[r[f"{measure}_mean"] for r in summary],
                    yerr=[1.96*r[f"{measure}_mc_se"] for r in summary],
                    marker="o",capsize=3,label=label)
    ax.set(xlabel="Gaussian projection dimension k",ylabel="Absolute relative distance error",
           title="Projection preserves finite distances approximately, not perfectly")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    save_figure(fig,design,FIGURE_NAMES[4])
    return {"jl_k8_p95":float(summary[0]["p95_absolute_relative_error_mean"]),
            "jl_k128_p95":float(summary[-1]["p95_absolute_relative_error_mean"]),
            "projection_repeats":float(design.repeats["projection"])}


def k_occurrences(x: FloatArray, neighbors: int=10) -> IntArray:
    """Count times each point occurs among other points' k-NN lists."""

    require_matrix(x)
    n=len(x)
    if not isinstance(neighbors,int) or neighbors < 1 or neighbors >= n:
        raise ValueError("k must be between one and n-1")
    ds=cdist(x,x)
    np.fill_diagonal(ds,np.inf)  # Explicitly exclude self-neighbors.
    idx=np.argpartition(ds,kth=neighbors-1,axis=1)[:,:neighbors]
    counts=np.bincount(idx.ravel(),minlength=n).astype(np.int64)
    assert counts.sum()==n*neighbors
    return counts


def hubness_experiment(design: Design) -> dict[str, float]:
    rows: list[dict[str, Any]]=[]
    example: list[dict[str,Any]]=[]
    for d in DIMS:
        for rep in range(design.repeats["hub"]):
            rng=np.random.default_rng(SEED+10000+d*100+rep)
            x=rng.normal(size=(350,d))
            counts=k_occurrences(x,neighbors=10)
            norms=np.linalg.norm(x,axis=1)
            top=int(np.ceil(.05*len(counts)))
            rows.append({"dimension":d,"repeat":rep,"points":len(x),"neighbors":10,
                         "kocc_skewness":float(skew(counts,bias=False)),
                         "top5pct_occurrence_share":float(np.sort(counts)[-top:].sum()/counts.sum()),
                         "norm_occurrence_correlation":float(np.corrcoef(norms,counts)[0,1]),
                         "kocc_max":int(counts.max())})
            if rep==0 and d in (2,512):
                example.extend({"dimension":d,"point":i,"norm":norm,
                                "occurrences":int(count)}
                               for i,(norm,count) in enumerate(zip(norms,counts)))
    summary=summarize(rows,("dimension",),(
        "kocc_skewness","top5pct_occurrence_share",
        "norm_occurrence_correlation","kocc_max"))
    write_csv(design.output_root / "results" / f"hub_replicates{design.suffix}.csv",rows)
    write_csv(design.output_root / "results" / f"hub_summary{design.suffix}.csv",summary)
    write_csv(design.output_root / "results" / f"hub_example_counts{design.suffix}.csv",example)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for ax,measure,ylabel in ((axs[0],"kocc_skewness","Skewness of k-occurrence"),
                               (axs[1],"top5pct_occurrence_share","Share of all neighbor slots")):
        ax.errorbar(DIMS,[r[f"{measure}_mean"] for r in summary],
                    yerr=[1.96*r[f"{measure}_mc_se"] for r in summary],
                    marker="o",capsize=3)
        ax.set(xscale="log",xlabel="Gaussian ambient dimension",ylabel=ylabel)
        ax.set_xticks(DIMS,labels=[str(d) for d in DIMS]);ax.grid(alpha=.2)
    axs[0].set_title("K-occurrence asymmetry")
    axs[1].set_title("Neighbor-list concentration in top 5% of points")
    fig.suptitle("Hubness in this isotropic Gaussian design, k=10, n=350")
    save_figure(fig,design,FIGURE_NAMES[5])
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.2))
    for ax,d in zip(axs,(2,512)):
        selected=[r for r in example if r["dimension"]==d]
        ax.scatter([r["norm"] for r in selected],
                   [r["occurrences"] for r in selected],s=14,alpha=.42)
        ax.set(title=f"p={d}: norm vs neighbor appearances",
               xlabel="Distance from population center (norm)",
               ylabel="Number of appearances in k-NN lists")
    fig.suptitle("Central points can become hubs without being semantic prototypes")
    save_figure(fig,design,FIGURE_NAMES[6])
    return {"hub_p2_skew":float(summary[0]["kocc_skewness_mean"]),
            "hub_p512_skew":float(summary[-1]["kocc_skewness_mean"]),
            "hub_p2_share":float(summary[0]["top5pct_occurrence_share_mean"]),
            "hub_p512_share":float(summary[-1]["top5pct_occurrence_share_mean"]),
            "hub_repeats":float(design.repeats["hub"])}


def participation_ratio(covariance: FloatArray) -> float:
    """Spectral effective rank (tr Sigma)^2 / tr(Sigma^2)."""

    require_matrix(covariance)
    if covariance.shape[0]!=covariance.shape[1] or not np.allclose(
            covariance,covariance.T,atol=1e-8):
        raise ValueError("Expected a symmetric square covariance")
    eigen=np.linalg.eigvalsh(covariance)
    if eigen.min() < -1e-8 or np.sum(eigen)<=0:
        raise ValueError("Expected nonzero positive-semidefinite covariance")
    return float(eigen.sum()**2/np.sum(eigen**2))


def latent_experiment(design: Design) -> dict[str, float]:
    p,q,n=120,5,650
    rng=np.random.default_rng(SEED+12000)
    basis=np.linalg.qr(rng.normal(size=(p,q)))[0]
    rows: list[dict[str,Any]]=[]
    for scale in LATENT_NOISE:
        true_eigen=np.r_[np.repeat(3+scale**2,q),np.repeat(scale**2,p-q)]
        true_eff=float(true_eigen.sum()**2/np.sum(true_eigen**2))
        for rep in range(design.repeats["intrinsic"]):
            rnd=np.random.default_rng(SEED+13000+int(scale*100)*100+rep)
            z=rnd.normal(scale=np.sqrt(3),size=(n,q))
            x=z@basis.T+scale*rnd.normal(size=(n,p))
            empirical=np.cov(x,rowvar=False,bias=True)
            eig=np.linalg.eigvalsh(empirical)[::-1]
            threshold=int(np.searchsorted(np.cumsum(eig)/np.sum(eig),.90)+1)
            rows.append({"noise_sd":scale,"repeat":rep,"ambient_dimension":p,
                         "latent_rank":q,"sample_participation_rank":participation_ratio(empirical),
                         "population_participation_rank":true_eff,
                         "sample_components_for_90pct_variance":threshold})
    summary=summarize(rows,("noise_sd",),(
        "sample_participation_rank","population_participation_rank",
        "sample_components_for_90pct_variance"))
    write_csv(design.output_root / "results" / f"intrinsic_replicates{design.suffix}.csv",rows)
    write_csv(design.output_root / "results" / f"intrinsic_summary{design.suffix}.csv",summary)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.errorbar(LATENT_NOISE,[r["sample_participation_rank_mean"] for r in summary],
                yerr=[1.96*r["sample_participation_rank_mc_se"] for r in summary],
                marker="o",capsize=3,label="Sample spectral effective rank")
    ax.plot(LATENT_NOISE,[r["population_participation_rank_mean"] for r in summary],
            "--",label="Exact population spectral effective rank")
    ax.axhline(q,color="gray",ls=":",label="Generated latent factor count = 5")
    ax.set(xlabel="Independent isotropic measurement-noise SD",
           ylabel="Participation ratio",title="Latent factor count is not noisy spectral effective rank")
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    save_figure(fig,design,FIGURE_NAMES[7])
    return {"intrinsic_no_noise_effective_rank":float(summary[0]["sample_participation_rank_mean"]),
            "intrinsic_max_noise_effective_rank":float(summary[-1]["sample_participation_rank_mean"])}


def generate_metric_task(rng: np.random.Generator,task: str,
                         ntrain: int=280,ntest:int=160
                         )->tuple[FloatArray,IntArray,FloatArray,IntArray]:
    """Two intentionally different synthetic notions of similarity."""

    n=ntrain+ntest;p=12
    labels=rng.integers(0,2,size=n,dtype=np.int64)
    if task=="angular":
        centers=np.zeros((n,p));centers[np.arange(n),labels]=1.0
        radii=rng.lognormal(mean=0,sigma=1.25,size=n)
        x=radii[:,None]*(centers+rng.normal(scale=.45,size=(n,p)))
    elif task=="radial":
        directions=rng.normal(size=(n,p))
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        radii=np.where(labels==0,1.0,1.7)*rng.lognormal(0,.18,size=n)
        x=radii[:,None]*directions
    else:
        raise ValueError("Expected angular or radial task")
    return x[:ntrain],labels[:ntrain],x[ntrain:],labels[ntrain:]


def metric_experiment(design: Design) -> dict[str,float]:
    rows: list[dict[str,Any]]=[]
    methods=("euclidean","cosine","norm_difference")
    for task in ("angular","radial"):
        for rep in range(design.repeats["metric"]):
            rng=np.random.default_rng(SEED+15000+(0 if task=="angular" else 5000)+rep)
            train,ytrain,test,ytest=generate_metric_task(rng,task)
            for method in methods:
                rows.append({"task":task,"method":method,"repeat":rep,
                             "accuracy":knn_accuracy(train,ytrain,test,ytest,distance=method)})
    summary=summarize(rows,("task","method"),("accuracy",))
    write_csv(design.output_root / "results" / f"metric_replicates{design.suffix}.csv",rows)
    write_csv(design.output_root / "results" / f"metric_summary{design.suffix}.csv",summary)
    fig,axs=plt.subplots(1,2,figsize=(10.7,4.5),sharey=True)
    descriptions={"euclidean":"Euclidean","cosine":"Cosine dissimilarity",
                  "norm_difference":"Absolute norm difference"}
    for ax,task in zip(axs,("angular","radial")):
        data=[next(r for r in summary if r["task"]==task and r["method"]==m)
              for m in methods]
        ax.bar(range(3),[r["accuracy_mean"] for r in data],
               color=["#22577a","#4682a9","#b74d35"])
        ax.errorbar(range(3),[r["accuracy_mean"] for r in data],
                    yerr=[1.96*r["accuracy_mc_se"] for r in data],
                    fmt="none",ecolor="black",capsize=3)
        ax.set_xticks(range(3),labels=[descriptions[m] for m in methods],rotation=20)
        ax.set(title="Classes differ by direction" if task=="angular"
               else "Classes differ by radius",ylim=(0,1),ylabel="7-NN test accuracy")
        ax.axhline(.5,color="gray",ls="--",lw=1)
    fig.suptitle("Which dissimilarity is useful depends on the task's generative structure")
    save_figure(fig,design,FIGURE_NAMES[8])
    return {"angular_cosine_acc":float(next(r["accuracy_mean"] for r in summary if
                r["task"]=="angular" and r["method"]=="cosine")),
            "radial_norm_acc":float(next(r["accuracy_mean"] for r in summary if
                r["task"]=="radial" and r["method"]=="norm_difference")),
            "metric_repeats":float(design.repeats["metric"])}


def run(design: Design)->dict[str,float]:
    """Regenerate the complete synthetic evidence inventory."""

    plt.rcParams.update({"figure.dpi":110,"font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    out: dict[str,float]={}
    out.update(distance_experiment(design))
    out.update(contrast_experiment(design))
    out.update(classification_experiment(design))
    out.update(scale_experiment(design))
    out.update(jl_experiment(design))
    out.update(hubness_experiment(design))
    out.update(latent_experiment(design))
    out.update(metric_experiment(design))
    write_csv(design.output_root / "results" / f"study_summary{design.suffix}.csv",
              [{"metric":k,"value":v} for k,v in sorted(out.items())])
    return out


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",
                        help="Small independent run; filenames end in _quick")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent,
                        help="Output root for figures/ and results/")
    args=parser.parse_args(argv)
    result=run(Design(quick=args.quick,output_root=args.output))
    print(f"Generated {len(FIGURE_NAMES)} figures and {len(TABLE_NAMES)} CSV tables")
    for key,value in sorted(result.items()):
        print(f"{key}: {value:.6f}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
