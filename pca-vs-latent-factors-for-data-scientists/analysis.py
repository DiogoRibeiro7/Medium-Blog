"""Article #24: reproducible PCA, factor, and eigenstructure experiments.

All data are synthetic. The exact identities and model-dependent
asymptotics are separated from the finite experiments they motivate.
Run this script from its directory to regenerate 9 figures and 18 CSVs;
--quick writes isolated suffixed smoke-test artifacts.
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
from scipy.stats import norm

FloatArray = NDArray[np.float64]
SEED = 20261101
FIGURES = (
    "fig1_spiked_spectrum",
    "fig2_spike_alignment",
    "fig3_reconstruction_vs_denoising",
    "fig4_low_variance_label",
    "fig5_feature_scales",
    "fig6_outlier_leverage",
    "fig7_factor_rotation",
    "fig8_parallel_analysis",
    "fig9_anisotropic_noise",
)
TABLES = (
    "spectrum_eigenvalues", "spectrum_summary",
    "alignment_replicates", "alignment_summary",
    "reconstruction_replicates", "reconstruction_summary",
    "label_replicates", "label_summary",
    "scale_replicates", "scale_summary",
    "outlier_replicates", "outlier_summary",
    "rotation_replicates", "rotation_summary",
    "parallel_replicates", "parallel_summary",
    "anisotropic_replicates", "anisotropic_summary",
)


@dataclass(frozen=True)
class Design:
    """Output scope and independent Monte Carlo effort."""

    quick: bool = False
    root: Path = Path(".")

    @property
    def suffix(self) -> str:
        return "_quick" if self.quick else ""

    @property
    def repetitions(self) -> dict[str, int]:
        if self.quick:
            return {"spectrum": 3, "alignment": 3, "reconstruction": 3,
                    "label": 4, "scale": 3, "outlier": 3,
                    "rotation": 3, "parallel": 3, "anisotropic": 3}
        return {"spectrum": 18, "alignment": 18, "reconstruction": 18,
                "label": 36, "scale": 24, "outlier": 24,
                "rotation": 15, "parallel": 18, "anisotropic": 24}


def validate(x: FloatArray) -> None:
    """Require a finite, nonempty matrix of observations by columns."""

    if (not isinstance(x, np.ndarray) or x.ndim != 2
            or min(x.shape) == 0 or not np.isfinite(x).all()):
        raise ValueError("Expected a nonempty finite two-dimensional matrix")


def covariance_pca(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return descending covariance eigenvalues, unit vectors, and mean.

    Centering and the (n-1) denominator are explicit. Eigenvector signs
    are not intrinsically identified.
    """

    validate(x)
    if x.shape[0] < 2:
        raise ValueError("PCA requires at least two observations")
    mean = x.mean(axis=0)
    center = x - mean
    covariance = center.T @ center / (len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0), eigenvectors[:, order], mean


def projection_overlap(estimated: FloatArray, target: FloatArray) -> float:
    """Average squared principal-subspace alignment, in [0,1].

    Both matrices must have the same number of orthonormal columns.
    """

    validate(estimated)
    validate(target)
    if estimated.shape != target.shape:
        raise ValueError("Subspaces must have matching dimensions")
    for basis in (estimated, target):
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-6):
            raise ValueError("Subspace columns must be orthonormal")
    return float(np.sum((estimated.T @ target) ** 2) / target.shape[1])


def spiked_eigenvalue_prediction(theta: float, gamma: float) -> tuple[float, float]:
    """Large-(n,p) real Gaussian rank-one spiked-covariance reference.

    The top population covariance eigenvalue is 1+theta. Above theta>
    sqrt(gamma), return asymptotic sample eigenvalue and squared alignment.
    Below threshold, return noise upper edge and zero alignment. These
    are asymptotic approximations, not finite-sample identities.
    """

    if theta < 0 or gamma <= 0 or not np.isfinite((theta, gamma)).all():
        raise ValueError("theta must be nonnegative; gamma positive and finite")
    if theta <= np.sqrt(gamma):
        return (1 + np.sqrt(gamma)) ** 2, 0.0
    return ((1 + theta) * (1 + gamma / theta),
            (1 - gamma / theta**2) / (1 + gamma / theta))


def gaussian_spike(rng: np.random.Generator, n: int, p: int,
                   theta: float) -> FloatArray:
    """Generate rank-one signal along the first coordinate plus isotropic noise."""

    if n < 2 or p < 2 or theta < 0:
        raise ValueError("Invalid spiked-model design")
    x = rng.normal(size=(n, p))
    x[:, 0] += np.sqrt(theta) * rng.normal(size=n)
    return x


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Preserve a complete rectangular evidence table with stable precision."""

    if not rows:
        raise ValueError("Refusing to write an empty evidence table")
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("Inconsistent evidence table columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.12g}" if isinstance(value, (float, np.floating))
                else value for key, value in row.items()
            })


def summarize(rows: list[dict[str, Any]], grouping: tuple[str, ...],
              measures: tuple[str, ...]) -> list[dict[str, Any]]:
    """Group repeated experiments and report means and independent-run MC-SE."""

    if not rows:
        raise ValueError("No replicates")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in grouping), []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(grouping, key))
        item["repeats"] = len(group)
        for measure in measures:
            values = np.array([float(row[measure]) for row in group])
            item[measure + "_mean"] = float(values.mean())
            item[measure + "_mc_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else float("nan")
            )
        output.append(item)
    return output


def emit(design: Design, key: str, figure: plt.Figure,
         replicates: list[dict[str, Any]], summary: list[dict[str, Any]],
         figure_index: int) -> None:
    """Write one experiment's two data tables and one original figure."""

    result_dir = design.root / "results"
    write_csv(result_dir / f"{key}_replicates{design.suffix}.csv", replicates)
    write_csv(result_dir / f"{key}_summary{design.suffix}.csv", summary)
    figure_dir = design.root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{FIGURES[figure_index]}{design.suffix}.png",
                   dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def emit_spectrum(design: Design) -> dict[str, float]:
    """Empirical spectrum with a fixed isotropic-noise and spiked design."""

    n, p, theta = 320, 120, 3.0
    rng = np.random.default_rng(SEED + 1)
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    spectra: dict[str, FloatArray] = {}
    for scenario, spike in (("noise", 0.0), ("spike", theta)):
        for rep in range(design.repetitions["spectrum"]):
            x = gaussian_spike(rng, n, p, spike)
            eig, vectors, _ = covariance_pca(x)
            if rep == 0:
                spectra[scenario] = eig
            rows.extend({"scenario": scenario, "repeat": rep,
                         "component": j + 1, "eigenvalue": float(value)}
                        for j, value in enumerate(eig))
            outcomes.append({"scenario": scenario, "repeat": rep,
                             "top_eigenvalue": float(eig[0]),
                             "top_alignment": float(vectors[0, 0] ** 2),
                             "variance_explained_top": float(eig[0] / eig.sum())})
    summary = summarize(outcomes, ("scenario",), ("top_eigenvalue",
                                                  "top_alignment", "variance_explained_top"))
    # This pair of files deliberately has detailed single-run eigenvalues
    # and across-replication summary, with the full replicate-level
    # headline values retained separately in the spectrum table.
    summary_rows = [dict(x) for x in summary]
    for row in summary_rows:
        row["sample_size"] = n
        row["ambient_dimension"] = p
        row["population_spike"] = theta if row["scenario"] == "spike" else 0.0
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    for scenario, style in (("noise", "-"), ("spike", "-")):
        ax.plot(np.arange(1, p + 1), spectra[scenario], style, lw=1.7,
                label=f"{scenario.title()} covariance sample")
    gamma = p / (n - 1)
    ax.axhline((1 + np.sqrt(gamma))**2, color="#b74d35", ls="--",
               label="Asymptotic noise upper edge")
    ax.set(xlabel="Ranked sample covariance component",
           ylabel="Sample eigenvalue",
           title="A sample noise spectrum is not a list of population factors")
    ax.set_xlim(1, p)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    # Unlike the other experiments, spectrum_eigenvalues stores each component
    # of the displayed draws; spectrum_summary stores independent repetitions.
    write_csv(design.root / "results" / f"spectrum_eigenvalues{design.suffix}.csv", rows)
    write_csv(design.root / "results" / f"spectrum_summary{design.suffix}.csv", summary_rows)
    figdir = design.root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{FIGURES[0]}{design.suffix}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"spectrum_noise_edge":float((1 + np.sqrt(gamma))**2),
            "spectrum_spike_top":float(next(r for r in summary if
                r["scenario"] == "spike")["top_eigenvalue_mean"])}


def emit_alignment(design: Design) -> dict[str, float]:
    """Compare finite-sample leading PCs against real spiked-model limits."""

    n, dims = 280, (30, 100, 180)
    strengths = (0.2, 0.5, 1.0, 2.0, 4.0)
    rows: list[dict[str, Any]] = []
    for p in dims:
        gamma = p / n
        for theta in strengths:
            predicted_top, predicted_alignment = spiked_eigenvalue_prediction(theta, gamma)
            for rep in range(design.repetitions["alignment"]):
                rng = np.random.default_rng(SEED + 1000 + p * 10000 +
                                            int(theta * 1000) * 20 + rep)
                eig, vec, _ = covariance_pca(gaussian_spike(rng, n, p, theta))
                rows.append({"dimension": p, "theta": theta, "repeat": rep,
                             "gamma": gamma, "alignment": float(vec[0, 0] ** 2),
                             "top_eigenvalue":float(eig[0]),
                             "asymptotic_alignment": predicted_alignment,
                             "asymptotic_top": predicted_top})
    summary = summarize(rows, ("dimension", "theta"), ("alignment", "top_eigenvalue"))
    fig, axs = plt.subplots(1, 2, figsize=(11.3, 4.5))
    colors = ["#22577a", "#b74d35", "#318c73"]
    for p, color in zip(dims, colors):
        subset = [r for r in summary if r["dimension"] == p]
        axs[0].errorbar(strengths, [r["alignment_mean"] for r in subset],
                        yerr=[1.96*r["alignment_mc_se"] for r in subset],
                        marker="o", capsize=3, color=color, label=f"p={p}, n={n}")
        curve = np.linspace(.1, 4, 140)
        theory = [spiked_eigenvalue_prediction(t, p/n)[1] for t in curve]
        axs[0].plot(curve, theory, ls="--", lw=1, color=color, alpha=.7)
        axs[1].plot(strengths,[r["top_eigenvalue_mean"] for r in subset],
                    marker="o", color=color, label=f"p={p}")
    axs[0].set(xlabel="Population spike excess θ",
               ylabel="Squared leading-eigenvector alignment", ylim=(-.04,1.04),
               title="Finite samples versus high-dimensional limit")
    axs[1].set(xlabel="Population spike excess θ",
               ylabel="Mean largest sample eigenvalue",
               title="An outlying eigenvalue depends on p/n")
    for ax in axs: ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
    emit(design, "alignment", fig, rows, summary, 1)
    return {"alignment_p180_theta4":float(next(r["alignment_mean"] for r in
               summary if r["dimension"]==180 and r["theta"]==4.0))}


def emit_reconstruction(design: Design) -> dict[str, float]:
    """Track observed-data reconstruction and latent-signal denoising separately."""

    p, q, ntest = 80, 5, 240
    ks = (1, 3, 5, 8, 15, 30, 60, 80)
    train_sizes = (70, 280)
    rows: list[dict[str, Any]] = []
    for size in train_sizes:
        for rep in range(design.repetitions["reconstruction"]):
            rng = np.random.default_rng(SEED + 2000 + size * 100 + rep)
            basis = np.linalg.qr(rng.normal(size=(p, q)))[0]
            latent_train = rng.normal(scale=2, size=(size, q)) @ basis.T
            latent_test = rng.normal(scale=2, size=(ntest, q)) @ basis.T
            train = latent_train + rng.normal(size=(size, p))
            test = latent_test + rng.normal(size=(ntest, p))
            eig, vectors, mean = covariance_pca(train)
            for k in ks:
                v = vectors[:, :k]
                reconstructed = (test - mean) @ v @ v.T + mean
                rows.append({"train_size":size, "repeat":rep, "components":k,
                             "latent_mse":float(np.mean((reconstructed-latent_test)**2)),
                             "observed_mse":float(np.mean((reconstructed-test)**2)),
                             "train_explained_fraction":float(eig[:k].sum()/eig.sum())})
    summary = summarize(rows, ("train_size","components"),
                        ("latent_mse","observed_mse","train_explained_fraction"))
    fig, axs = plt.subplots(1, 2, figsize=(11.2,4.5))
    for size, color in ((70,"#b74d35"),(280,"#22577a")):
        x=[r for r in summary if r["train_size"]==size]
        for ax,key in ((axs[0],"observed_mse"),(axs[1],"latent_mse")):
            ax.errorbar(ks,[r[key+"_mean"] for r in x],
                        yerr=[1.96*r[key+"_mc_se"] for r in x],
                        marker="o",capsize=2,color=color,label=f"n_train={size}")
    axs[0].set(title="Observed-data reconstruction (keeps noise)",
               ylabel="Held-out MSE per coordinate",xlabel="Retained PCs")
    axs[1].set(title="Latent-signal denoising (oracle target)",
               ylabel="Held-out MSE per coordinate",xlabel="Retained PCs")
    for ax in axs:
        ax.axvline(q,ls=":",color="gray");ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    emit(design, "reconstruction", fig, rows, summary, 2)
    return {"reconstruction_280_k5_risk":float(next(r["latent_mse_mean"] for r
          in summary if r["train_size"]==280 and r["components"]==5))}


def pooled_linear_discriminant(train: FloatArray, labels: NDArray[np.int64],
                               test: FloatArray) -> NDArray[np.int64]:
    """Train a regularized equal-covariance linear classifier on training data."""

    validate(train); validate(test)
    if train.shape[1] != test.shape[1] or len(labels) != len(train):
        raise ValueError("Shape or label mismatch")
    if not np.isin(labels,[0,1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Both binary classes are required")
    mean0 = train[labels==0].mean(axis=0)
    mean1 = train[labels==1].mean(axis=0)
    residuals = train - np.where(labels[:,None]==1,mean1,mean0)
    cov = residuals.T@residuals / (len(train)-2)
    cov += .05*np.eye(train.shape[1])  # Explicit numerical regularization.
    direction = np.linalg.solve(cov, mean1-mean0)
    return ((test-(mean1+mean0)/2)@direction > 0).astype(np.int64)


def emit_label(design: Design) -> dict[str, float]:
    """Show why high-variance PCs can exclude nearly all label information."""

    p, ntrain, ntest = 14, 240, 180
    ks = (1, 3, 5, 9, 13, 14)
    rows: list[dict[str, Any]]=[]
    for rep in range(design.repetitions["label"]):
        rng=np.random.default_rng(SEED+3000+rep)
        labels=rng.integers(0,2,size=ntrain+ntest,dtype=np.int64)
        x=rng.normal(scale=2.2,size=(ntrain+ntest,p))
        x[:,-1]=.19*(2*labels-1)+rng.normal(scale=.17,size=len(labels))
        train,test=x[:ntrain],x[ntrain:]
        ytrain,ytest=labels[:ntrain],labels[ntrain:]
        eig,vec,mean=covariance_pca(train)
        for k in ks:
            ztrain=(train-mean)@vec[:,:k]
            ztest=(test-mean)@vec[:,:k]
            pred=pooled_linear_discriminant(ztrain,ytrain,ztest)
            rows.append({"components":k,"repeat":rep,"accuracy":float(np.mean(pred==ytest)),
                         "signal_included":int(k==p),
                         "explained_variance":float(eig[:k].sum()/eig.sum())})
    summary=summarize(rows,("components",),("accuracy","explained_variance"))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].errorbar(ks,[r["accuracy_mean"] for r in summary],
                    yerr=[1.96*r["accuracy_mc_se"] for r in summary],
                    marker="o",capsize=3,color="#22577a")
    axs[0].axhline(norm.cdf(.19/.17),ls="--",color="#b74d35",
                   label="Population Bayes accuracy")
    axs[0].axhline(.5,ls=":",color="gray",label="Equal-prior chance")
    axs[0].set(xlabel="Leading PCs given to supervised classifier",
               ylabel="Independent-test accuracy",ylim=(.35,1),
               title="Low variance does not imply low predictive value")
    axs[0].legend(fontsize=8)
    axs[1].plot(ks,[r["explained_variance_mean"] for r in summary],marker="o")
    axs[1].set(xlabel="Leading PCs",ylabel="Training variance explained",
               ylim=(0,1.03),title="Very high explained variance can miss the label")
    for ax in axs:ax.grid(alpha=.2)
    emit(design,"label",fig,rows,summary,3)
    return {"label_top13_accuracy":float(next(r["accuracy_mean"] for r in summary
            if r["components"]==13)),
            "label_all14_accuracy":float(next(r["accuracy_mean"] for r in summary
            if r["components"]==14))}


def emit_scale(design: Design) -> dict[str, float]:
    """Show that changing a nuisance coordinate's units rotates the leading PC."""

    scales=(.3,.8,1.5,3.0,6.0)
    rows: list[dict[str, Any]]=[]
    for rep in range(design.repetitions["scale"]):
        rng=np.random.default_rng(SEED+4000+rep)
        x=rng.normal(size=(300,18))
        x[:,1]=rng.normal(scale=2,size=len(x))
        for s in scales:
            y=x.copy();y[:,0]*=s
            eig,vec,_=covariance_pca(y)
            rows.append({"nuisance_scale":s,"repeat":rep,
                         "signal_alignment":float(vec[1,0]**2),
                         "nuisance_alignment":float(vec[0,0]**2),
                         "top_variance_fraction":float(eig[0]/eig.sum())})
    summary=summarize(rows,("nuisance_scale",),
                      ("signal_alignment","nuisance_alignment","top_variance_fraction"))
    fig,ax=plt.subplots(figsize=(9.2,4.7))
    for key,name,color in (("signal_alignment","Signal axis","#22577a"),
                           ("nuisance_alignment","Rescaled nuisance axis","#b74d35")):
        ax.errorbar(scales,[r[key+"_mean"] for r in summary],
                    yerr=[1.96*r[key+"_mc_se"] for r in summary],
                    marker="o",capsize=3,color=color,label=name)
    ax.axvline(2.,color="gray",ls=":",label="Equal population SD")
    ax.set(xlabel="Multiplier of one nuisance coordinate",ylabel="Squared leading-PC alignment",
           ylim=(-.06,1.06),title="A unit change can swap the leading principal direction")
    ax.grid(alpha=.2);ax.legend()
    emit(design,"scale",fig,rows,summary,4)
    return {"scale_small_signal_alignment":float(summary[0]["signal_alignment_mean"]),
            "scale_large_signal_alignment":float(summary[-1]["signal_alignment_mean"])}


def emit_outlier(design: Design) -> dict[str, float]:
    """Finite-sample sensitivity of a squared-error principal direction."""

    heights=(0.,8.,16.,24.,32.,40.)
    rows: list[dict[str, Any]]=[]
    example: dict[float,tuple[FloatArray,FloatArray]]={}
    for rep in range(design.repetitions["outlier"]):
        rng=np.random.default_rng(SEED+5000+rep)
        base=rng.normal(size=(180,2))*np.array([2.5,.7])
        for height in heights:
            augmented=np.vstack([base,[[0.,height]]])
            eig,vec,_=covariance_pca(augmented)
            if rep==0 and height in (0.,40.):
                example[height]=(augmented,vec[:,0])
            rows.append({"outlier_height":height,"repeat":rep,
                         "horizontal_alignment":float(vec[0,0]**2),
                         "top_variance_fraction":float(eig[0]/eig.sum())})
    summary=summarize(rows,("outlier_height",),
                      ("horizontal_alignment","top_variance_fraction"))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].errorbar(heights,[r["horizontal_alignment_mean"] for r in summary],
                    yerr=[1.96*r["horizontal_alignment_mc_se"] for r in summary],
                    marker="o",capsize=3,color="#b74d35")
    axs[0].set(xlabel="Added point's vertical coordinate",ylabel="Squared PC alignment to x-axis",
               ylim=(-.04,1.04),title="One point can rotate a least-squares direction")
    for height,color in ((0.,"#22577a"),(40.,"#b74d35")):
        points,v=example[height]
        axs[1].scatter(points[:,0],points[:,1],alpha=.18,s=9,color=color)
        axs[1].plot([-9*v[0],9*v[0]],[-9*v[1],9*v[1]],color=color,lw=2,
                    label=f"Height {height:g}")
    axs[1].set(xlabel="First coordinate",ylabel="Second coordinate",
               title="Representative contaminated PCs",ylim=(-10,42))
    axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    emit(design,"outlier",fig,rows,summary,5)
    return {"outlier_clean_alignment":float(summary[0]["horizontal_alignment_mean"]),
            "outlier_40_alignment":float(summary[-1]["horizontal_alignment_mean"])}


def emit_rotation(design: Design) -> dict[str, float]:
    """Exact orthogonal factor rotation changes loadings, not reconstructed data."""

    angles=(0.,np.pi/12,np.pi/6,np.pi/4,np.pi/3,np.pi/2)
    rows: list[dict[str, Any]]=[]
    shown:tuple[FloatArray,FloatArray]|None=None
    for rep in range(design.repetitions["rotation"]):
        rng=np.random.default_rng(SEED+6000+rep)
        basis=np.linalg.qr(rng.normal(size=(10,2)))[0]
        z=rng.normal(size=(200,2))
        original=z@basis.T
        for angle in angles:
            c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,-s],[s,c]])
            rotated_basis=basis@rotation
            rotated_z=z@rotation
            if rep==0 and angle==np.pi/4:shown=(basis,rotated_basis)
            rows.append({"angle_degrees":float(np.degrees(angle)),"repeat":rep,
                         "loadings_fro_difference":float(np.linalg.norm(basis-rotated_basis)),
                         "covariance_fro_difference":float(np.linalg.norm(
                             basis@basis.T-rotated_basis@rotated_basis.T)),
                         "max_data_difference":float(np.max(np.abs(
                             original-rotated_z@rotated_basis.T)))})
    summary=summarize(rows,("angle_degrees",),
                      ("loadings_fro_difference","covariance_fro_difference",
                       "max_data_difference"))
    assert shown is not None
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.5),sharey=True)
    for ax,mat,title in zip(axs,shown,("Original factor loadings","Rotated factor loadings")):
        ax.imshow(mat,vmin=-.75,vmax=.75,cmap="RdBu_r",aspect="auto")
        ax.set(xlabel="Latent factor index",ylabel="Observed coordinate",title=title)
        ax.set_xticks((0,1),labels=("1","2"))
    fig.suptitle("Different loading matrices can reconstruct identical observations")
    emit(design,"rotation",fig,rows,summary,6)
    return {"rotation_covariance_max_error":float(max(r["covariance_fro_difference"]
                                             for r in rows))}


def permute_columns(x: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Independently permute each column to destroy between-feature dependence."""

    validate(x)
    return np.column_stack([rng.permutation(x[:,j]) for j in range(x.shape[1])])


def emit_parallel(design: Design) -> dict[str, float]:
    """Horn-style parallel analysis versus a >1 eigenvalue counting heuristic."""

    n, dims = 150, (30,70)
    rows: list[dict[str, Any]]=[]
    for p in dims:
        for rep in range(design.repetitions["parallel"]):
            rng=np.random.default_rng(SEED+7000+p*100+rep)
            basis=np.linalg.qr(rng.normal(size=(p,3)))[0]
            signal=rng.normal(size=(n,3))*np.sqrt([5.,3.,1.])
            x=signal@basis.T+rng.normal(size=(n,p))
            eig,_,_=covariance_pca(x)
            nulls=np.array([covariance_pca(permute_columns(x,rng))[0]
                            for _ in range(12 if not design.quick else 4)])
            thresholds=np.quantile(nulls,.95,axis=0)
            pa=int(np.sum(eig>thresholds))
            # The >1 criterion is intentionally evaluated on *covariance*,
            # whose noise variance is explicitly one in the generating model.
            kaiser=int(np.sum(eig>1.))
            rows.append({"dimension":p,"repeat":rep,"sample_size":n,
                         "latent_factors":3,"parallel_selected":pa,
                         "eigen_gt_one_selected":kaiser,
                         "largest_sample_eigenvalue":float(eig[0]),
                         "largest_null_threshold":float(thresholds[0])})
    summary=summarize(rows,("dimension",),
                      ("parallel_selected","eigen_gt_one_selected"))
    fig,ax=plt.subplots(figsize=(8.7,4.5))
    offsets=(-.13,.13)
    for shift,key,label,color in ((offsets[0],"parallel_selected","Parallel analysis","#22577a"),
                                   (offsets[1],"eigen_gt_one_selected",">1 covariance eigenvalue","#b74d35")):
        ax.bar(np.arange(2)+shift,[r[key+"_mean"] for r in summary],
               width=.26,color=color,label=label)
        ax.errorbar(np.arange(2)+shift,[r[key+"_mean"] for r in summary],
                    yerr=[1.96*r[key+"_mc_se"] for r in summary],
                    fmt="none",ecolor="black",capsize=3)
    ax.axhline(3,ls="--",color="gray",label="Three generated latent factors")
    ax.set_xticks(range(2),labels=[f"p={p}, n={n}" for p in dims])
    ax.set(ylabel="Selected component count",
           title="Heuristics answer different questions under finite samples")
    ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    emit(design,"parallel",fig,rows,summary,7)
    return {"parallel_p70_mean":float(summary[-1]["parallel_selected_mean"]),
            "kaiser_p70_mean":float(summary[-1]["eigen_gt_one_selected_mean"])}


def emit_anisotropic(design: Design) -> dict[str, float]:
    """A high-variance noise coordinate can displace a true signal subspace."""

    scales=(.5,1.,1.5,2.5,4.)
    rows: list[dict[str, Any]]=[]
    p,n,q=50,260,3
    true_basis=np.eye(p)[:,2:5]
    for rep in range(design.repetitions["anisotropic"]):
        rng=np.random.default_rng(SEED+8000+rep)
        noise=rng.normal(size=(n,p))
        signal=rng.normal(size=(n,q))*np.sqrt([5.,3.5,2.5])
        for scale in scales:
            x=noise.copy()
            x[:,0]*=scale
            x+=signal@true_basis.T
            eig,vec,_=covariance_pca(x)
            rows.append({"nuisance_sd":scale,"repeat":rep,
                         "top3_signal_overlap":projection_overlap(vec[:,:3],true_basis),
                         "leading_nuisance_alignment":float(vec[0,0]**2),
                         "largest_eigenvalue":float(eig[0])})
    summary=summarize(rows,("nuisance_sd",),
                      ("top3_signal_overlap","leading_nuisance_alignment","largest_eigenvalue"))
    fig,ax=plt.subplots(figsize=(9.1,4.7))
    for key,label,color in (("top3_signal_overlap","Top-three signal subspace overlap","#22577a"),
                            ("leading_nuisance_alignment","Top PC on nuisance axis","#b74d35")):
        ax.errorbar(scales,[r[key+"_mean"] for r in summary],
                    yerr=[1.96*r[key+"_mc_se"] for r in summary],
                    marker="o",capsize=3,label=label,color=color)
    ax.set(xlabel="SD of a single anisotropic noise coordinate",
           ylabel="Projection alignment",ylim=(-.05,1.05),
           title="Large eigenvalues need not reflect latent signal")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    emit(design,"anisotropic",fig,rows,summary,8)
    return {"anisotropic_low_signal_overlap":float(summary[0]["top3_signal_overlap_mean"]),
            "anisotropic_high_signal_overlap":float(summary[-1]["top3_signal_overlap_mean"])}


def run(design: Design) -> dict[str, float]:
    """Produce every figure, data table, and prespecified headline quantity."""

    plt.rcParams.update({"figure.dpi":115,"font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    outcome: dict[str, float]={}
    for experiment in (emit_spectrum,emit_alignment,emit_reconstruction,
                       emit_label,emit_scale,emit_outlier,emit_rotation,
                       emit_parallel,emit_anisotropic):
        outcome.update(experiment(design))
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",
                        help="Fewer independent repetitions, isolated _quick outputs")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent,
                        help="Root for figures/ and results/ outputs")
    args=parser.parse_args(argv)
    values=run(Design(quick=args.quick,root=args.output))
    print(f"Generated {len(FIGURES)} figures and {len(TABLES)} tables")
    for key,value in sorted(values.items()):
        print(f"{key}: {value:.6f}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
