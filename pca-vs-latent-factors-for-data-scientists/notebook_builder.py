"""Construct the tracked, reproducible companion notebook with stable IDs."""

from __future__ import annotations

from pathlib import Path

import nbformat


def build() -> Path:
    """Write a notebook that independently reruns the quick study."""

    cells = [
        nbformat.v4.new_markdown_cell(r"""# A principal component is not a latent factor

This notebook accompanies **article #24**. All examples are **synthetic**.
The canonical tables use more independent repetitions than this notebook's
quick run. Nothing here establishes a fact about a specific production
representation or a human population.

Python 3.12: install the adjacent *requirements.txt* before executing.
"""),
        nbformat.v4.new_markdown_cell(r"""## 1. What PCA actually optimizes

For centered observations in matrix \(X\in\mathbb R^{n\times p}\), the
sample covariance is \(S=X^\top X/(n-1)\). A first principal component
maximizes \(v^\top S v\) subject to \(\|v\|=1\), which yields
\(Sv=\lambda_{\max}v\). This optimization does not mention labels,
causes, independent latent factors, or robustness.
"""),
        nbformat.v4.new_code_cell("""import csv
import tempfile
from pathlib import Path
import numpy as np
from IPython.display import Image, display
from analysis import (Design, FIGURES, TABLES, covariance_pca,
                      spiked_eigenvalue_prediction, run)

print('Spiked covariance references (theta=2, gamma=0.5):')
print(spiked_eigenvalue_prediction(2., .5))
sample = np.random.default_rng(8).normal(size=(40, 3))
eigenvalues, vectors, mean = covariance_pca(sample)
print('Sample eigenvalues:', eigenvalues)"""),
        nbformat.v4.new_markdown_cell(r"""## 2. Regenerate evidence in isolation

The full archived study uses many independent repetitions. This notebook
uses reduced effort with the same declared distributions and independent
random streams. It writes separate *quick* files to a temporary directory
and never overwrites the canonical figures or CSV tables.
"""),
        nbformat.v4.new_code_cell("""temporary = tempfile.TemporaryDirectory(prefix='pca_latent_')
out = Path(temporary.name)
summary = run(Design(quick=True, root=out))
print(f'{len(FIGURES)} figures and {len(TABLES)} tables')
for key, value in sorted(summary.items()):
    print(f'{key}: {value:.5f}')"""),
        nbformat.v4.new_markdown_cell(r"""## 3. Spiked eigenvalues and direction recovery

The Marchenko–Pastur upper edge for a pure-noise covariance with
aspect ratio \(\gamma=p/n\) is \((1+\sqrt\gamma)^2\). In a rank-one
spiked model with population eigenvalue \(1+\theta\), asymptotic
eigenvector alignment becomes nonzero above \(\theta=\sqrt\gamma\)
under its assumptions. These formulas are high-dimensional limits,
not exact statements for the finite simulations below.
"""),
        nbformat.v4.new_code_cell("""for fig in FIGURES[:2]:
    display(Image(filename=str(out / 'figures' / (fig + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 4. Reconstruction is not denoising or prediction

Every retained sample PC helps reconstruct observed held-out vectors.
But when the goal is to recover a known **latent noiseless signal**,
adding PCs beyond the signal subspace also restores measurement noise.
The second quantity exists here only because the synthetic generator
reveals the unobserved truth.
"""),
        nbformat.v4.new_code_cell("""display(Image(filename=str(out / 'figures' /
                            (FIGURES[2] + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 5. Unsupervised variance and arbitrary units

The class label is encoded in the smallest-variance coordinate in one
experiment. PCA is fit on training features only; the discriminant
classifier uses training labels only. The next experiment changes
only the units of one nuisance variable, rotating the leading component
without changing any semantic meaning of the data.
"""),
        nbformat.v4.new_code_cell("""for fig in FIGURES[3:5]:
    display(Image(filename=str(out / 'figures' / (fig + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 6. Outliers, rotations, and component selection

One extreme observation can determine a least-squares principal axis.
Orthogonal rotations of factor loadings and scores leave their product,
and the implied covariance under isotropic latent factors, unchanged.
Parallel analysis is a sample-dependent heuristic, not an oracle that
reveals the generating factor count in every sample.
"""),
        nbformat.v4.new_code_cell("""for fig in FIGURES[5:8]:
    display(Image(filename=str(out / 'figures' / (fig + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 7. Anisotropic measurement noise

An unexpectedly high sample eigenvalue can come from a high-variance
noise coordinate. The final figure reports subspace overlap against the
known signal axes, a quantity unavailable without synthetic ground truth.
"""),
        nbformat.v4.new_code_cell("""display(Image(filename=str(out / 'figures' /
                            (FIGURES[8] + '_quick.png'))))

with (out / 'results' / 'label_summary_quick.csv').open(newline='') as file:
    rows = list(csv.DictReader(file))
for row in rows:
    print('PCs', row['components'], 'accuracy',
          f"{float(row['accuracy_mean']):.3f}")"""),
        nbformat.v4.new_markdown_cell(r"""## 8. Scope and reproducibility

- All sources are original synthetic simulations with declared generators.
- Monte Carlo error bars summarize **independent repetitions**, not
  uncertainty about every possible real-world dataset.
- PCA optimizes squared Euclidean reconstruction; predictive adequacy,
  latent factor interpretation, and causality need additional assumptions.
- The 18 full-run CSVs are the versioned evidence; this notebook writes
  only temporary quick-run outputs.
- The path-scoped CI runs numerical tests and this notebook in a fresh
  kernel, and separately regenerates quick evidence.
"""),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"pca-factors-{i:02d}"
    book = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbformat.validate(book)
    destination = Path(__file__).resolve().parent / "pca_latent_factors.ipynb"
    nbformat.write(book, destination)
    return destination


if __name__ == "__main__":
    print(build())
