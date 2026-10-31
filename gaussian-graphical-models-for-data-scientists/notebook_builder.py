"""Build the companion notebook from audited, versioned code cells.

Run this script after editing notebook narrative. The generated notebook runs
its own quick simulation in a temporary directory, so it never overwrites the
full publication figures and tables.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell("""# A Missing Edge Is Not a Missing Relationship

## Gaussian graphical models, conditional independence, and the limits of network discovery

Companion to the long-form [article](article.md), by Diogo Ribeiro (October 2026).

**Scope:** All numerical examples here are synthetic. The graph is a set of
Gaussian conditional independence statements, not a causal diagram or a map
of real people, systems, genes, patients, or organizations. No internet access
or proprietary data are required after the listed dependencies are installed.
"""),
        nbf.v4.new_markdown_cell("""## 1. The precision identity

For jointly Gaussian $X\\sim N_p(\\mu,\\Sigma)$ with
$\\Omega=\\Sigma^{-1}$,
$X_i\\perp X_j\\mid X_{-\\{i,j\\}}$ if and only if $\\Omega_{ij}=0$.
For $i\\neq j$, the partial correlation is
$-\\Omega_{ij}/\\sqrt{\\Omega_{ii}\\Omega_{jj}}$.
This identity needs a positive-definite covariance and a Gaussian law.
"""),
        nbf.v4.new_code_cell("""import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from analysis import (ExperimentConfig, chain_precision,
                      covariance_to_correlation, partial_correlations, run)

omega = chain_precision()
sigma = np.linalg.inv(omega)
print('Marginal correlation of chain endpoints:',
      covariance_to_correlation(sigma)[0, 2])
print('Population partial correlation of endpoints:',
      partial_correlations(omega)[0, 2])
assert abs(partial_correlations(omega)[0, 2]) < 1e-12"""),
        nbf.v4.new_markdown_cell("""## 2. Regenerate the evidence

The full publication uses 70 independent repetitions per sample size and
150 bootstrap resamples. For notebook interactivity, this run uses eight
repetitions and twelve resamples, with **distinct filenames** and a separate
temporary output root. Do not interpret its small Monte Carlo summary as the
full-run published evidence.
"""),
        nbf.v4.new_code_cell("""working_directory = Path(tempfile.mkdtemp(prefix='ggm-notebook-'))
metrics = run(ExperimentConfig(quick=True, output_root=working_directory))
print('Generated figures:', len(list((working_directory / 'figures').glob('*.png'))))
print('Generated tables:', len(list((working_directory / 'results').glob('*.csv'))))
assert len(list((working_directory / 'figures').glob('*.png'))) == 9
assert len(list((working_directory / 'results').glob('*.csv'))) == 13"""),
        nbf.v4.new_markdown_cell("""## 3. A chain and a collider answer different questions

The chain has a marginal end-to-end correlation despite no conditional edge.
The collider has independent parents in the population, but residualizing
both against their common effect induces a negative conditional association.
Selecting on the collider can create the same problem in a real analysis.
"""),
        nbf.v4.new_code_cell("""for name in ('fig1_chain_covariance_and_precision',
             'fig3_collider_conditioning'):
    display(Image(filename=str(working_directory / 'figures' /
                               f'{name}_quick.png')))"""),
        nbf.v4.new_markdown_cell("""## 4. High-dimensional estimation is a different problem

When $p=30$ and $n=20$, the centered sample covariance has rank at most 19,
so direct inversion does not identify a unique sample precision. Penalized
optimization yields a well-defined estimate, but support recovery then
depends on regularization, sample size, signal strength, and model validity.
"""),
        nbf.v4.new_code_cell("""display(Image(filename=str(working_directory / 'figures' /
                           'fig5_high_dimension_spectrum_quick.png')))
display(Image(filename=str(working_directory / 'figures' /
                           'fig6_graph_recovery_quick.png')))"""),
        nbf.v4.new_markdown_cell("""## 5. Edge selection, predictive loss, and stability

The upper-triangle F1 score measures a known simulated support. Held-out
Gaussian likelihood measures prediction under a Gaussian law. Bootstrap
selection frequency is sensitivity to resampling **conditional on the fitted
procedure and observed sample**; none of them is automatically a probability
of causation or a calibrated posterior probability of edge existence.
"""),
        nbf.v4.new_code_cell("""display(Image(filename=str(working_directory / 'figures' /
                           'fig7_heldout_likelihood_quick.png')))
display(Image(filename=str(working_directory / 'figures' /
                           'fig8_bootstrap_edge_stability_quick.png')))"""),
        nbf.v4.new_markdown_cell("""## 6. Nonlinear dependence is invisible to Pearson correlation

With $X\\sim N(0,1)$, $Y=X^2-1+\\varepsilon$ and independent
$\\varepsilon\\sim N(0,0.7^2)$, the population covariance vanishes by
symmetry. Yet $\\mathbb{E}[Y\\mid X]=X^2-1$ and the population
best-predictor $R^2$ equals $2/(2+0.7^2)\\approx0.8032$.
"""),
        nbf.v4.new_code_cell("""display(Image(filename=str(working_directory / 'figures' /
                           'fig9_nonlinear_dependence_quick.png')))
print('Linear held-out R2:', round(metrics['nonlinear_linear_test_r2'], 3))
print('Quadratic held-out R2:', round(metrics['nonlinear_quadratic_test_r2'], 3))
assert metrics['nonlinear_quadratic_test_r2'] > 0.65"""),
        nbf.v4.new_markdown_cell("""## 7. What these experiments cannot show

- All numerical experiments are seeded, synthetic, and IID, except for the
  nonlinear but still independently drawn experiment.
- Regularization settings are fixed for illustration; the notebook does not
  claim oracle tuning or universal support-recovery rates.
- Observed Gaussian conditional independence is not proof of structural
  causation, intervention effects, or absence of mediated pathways.
- Latent confounders, sample selection, non-Gaussian structure, and temporal
  dynamics can each invalidate a causal reading of an undirected edge.
- For the **full** 70-repeat and 150-bootstrap results used in the article,
  run `python analysis.py` and consult the committed `results/*.csv`.
"""),
    ]
    notebook_path = ROOT / "gaussian_graphical_models.ipynb"
    with notebook_path.open("w", encoding="utf-8") as stream:
        nbf.write(notebook, stream)
    print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    build()
