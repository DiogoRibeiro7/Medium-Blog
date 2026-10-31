"""Build the tracked, self-contained companion notebook from versioned cells.

The notebook runs the *quick* experiment in a new temporary directory, never
overwriting the article's canonical full-run figures or evidence tables.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


def build() -> Path:
    """Write a valid notebook in a stable, reviewable location."""

    cells = [
        nbformat.v4.new_markdown_cell(r"""# A nearest neighbor is not necessarily similar

This is the executable companion to **article #16**, a study of distance
concentration, irrelevant dimensions, hubness, and task-dependent metrics.

All data in this notebook are **synthetic**. It independently runs the reduced
Monte Carlo configuration; the tracked CSVs and figures were generated using
the more extensively replicated full configuration. Neither configuration
provides evidence of generalization to particular real-world populations.

Reproduce with Python 3.12 and `pip install -r requirements.txt`.
"""),
        nbformat.v4.new_markdown_cell(r"""## 1. The exact Gaussian reference

For independent $X,Y\sim\mathcal N_p(0,I_p)$, their Euclidean distance
$D=\lVert X-Y\rVert_2$ has distribution $\sqrt{2}\,\chi_p$. Consequently

$$\mathbb E D=2\frac{\Gamma((p+1)/2)}{\Gamma(p/2)},\qquad
  \mathbb E D^2=2p,\qquad
  \mathrm{CV}(D)=\frac{\sqrt{2p-(\mathbb ED)^2}}{\mathbb ED}. $$

The coefficient of variation approaches $(2p)^{-1/2}$; this is a
distribution-specific result, not a universal claim about embeddings.
"""),
        nbformat.v4.new_code_cell("""import csv
import tempfile
from pathlib import Path

from IPython.display import Image, display
from analysis import Design, FIGURE_NAMES, TABLE_NAMES, gaussian_distance_moments, run

for dimension in (2, 8, 32, 128, 512):
    mean, sd, cv = gaussian_distance_moments(dimension)
    print(f'p={dimension:<3} E[D]={mean:8.3f} SD[D]={sd:6.3f} CV={cv:6.4f}')"""),
        nbformat.v4.new_markdown_cell(r"""## 2. Generate independent evidence

Run the `--quick` design into a temporary directory. The full study has 36
classification repetitions, 20 independent Gaussian projections per output
dimension, and 12 hubness repetitions. This quick kernel uses only 5, 4, and 3,
respectively. Monte Carlo standard errors in the quick output are necessarily
less stable. The random-number streams and all implementation details are in
the adjacent `analysis.py`.
"""),
        nbformat.v4.new_code_cell("""workspace = tempfile.TemporaryDirectory(prefix='nearest_neighbor_geometry_')
output_root = Path(workspace.name)
summary = run(Design(quick=True, output_root=output_root))
print(f'{len(FIGURE_NAMES)} original PNG figures; {len(TABLE_NAMES)} CSV evidence tables')
for key in ('bayes_accuracy', 'knn_p2_acc', 'knn_p512_acc',
            'hub_p2_share', 'hub_p512_share', 'jl_k8_p95', 'jl_k128_p95'):
    print(f'{key:42s}: {summary[key]:.4f}')"""),
        nbformat.v4.new_markdown_cell(r"""## 3. Pairwise distances and finite-pool contrast

The left plot uses independent Gaussian pair differences; the right adds a
*specific* candidate-pool experiment. Relative distance concentration and
farthest-vs-nearest contrast are different summaries. Notice how the finite
number of candidates is held fixed across dimensions.
"""),
        nbformat.v4.new_code_cell("""for figure in FIGURE_NAMES[:2]:
    display(Image(filename=str(output_root / 'figures' / (figure + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 4. A fixed Bayes problem, degraded Euclidean neighbors

The first coordinate contains the entire label signal:
$X_1\mid Y=y\sim\mathcal N((2y-1)0.85,1)$, while the remaining
coordinates are independent nuisance variables. For equal class priors,
the exact Bayes decision is $\hat Y=\mathbf 1\{X_1>0\}$ and population
accuracy is $\Phi(0.85)$. We use exactly the same simulated observations
for each truncated dimension within a repetition. Feature rescaling changes
geometry, not the conditional class mechanism.
"""),
        nbformat.v4.new_code_cell("""for figure in FIGURE_NAMES[2:4]:
    display(Image(filename=str(output_root / 'figures' / (figure + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 5. Distortion, hubness, and spectral dimension

The Johnson–Lindenstrauss experiment quantifies pair-distance distortion on
a fixed set of 170 points and independent random maps. It does not test
classification quality. Hubness counts how often each observation appears in
others' nearest-neighbor lists (self-neighbors excluded). Spectral participation
ratio estimates effective spread, not the count of generative latent factors.
"""),
        nbformat.v4.new_code_cell("""for figure in FIGURE_NAMES[4:8]:
    display(Image(filename=str(output_root / 'figures' / (figure + '_quick.png'))))"""),
        nbformat.v4.new_markdown_cell(r"""## 6. The similarity rule depends on the question

In the angular task, class label is encoded in *direction* with a wide radius
distribution. In the radial task, label is encoded in *norm* with random
directions. Absolute norm difference is **not** a metric on vectors: distinct
points can have identical norm. The tests compare prediction accuracy for
each generative task separately, without claiming a universal best distance.
"""),
        nbformat.v4.new_code_cell("""display(Image(filename=str(output_root / 'figures' /
                               (FIGURE_NAMES[8] + '_quick.png'))))

def table(name):
    with (output_root / 'results' / (name + '_quick.csv')).open(newline='') as handle:
        return list(csv.DictReader(handle))

print('Example classification summary rows:')
for row in table('classification_summary'):
    print(f"p={row['dimension']:>3} | 7-NN={float(row['knn_accuracy_mean']):.3f} | "
          f"MC-SE={float(row['knn_accuracy_mc_se']):.3f}")"""),
        nbformat.v4.new_markdown_cell(r"""## 7. Reproducibility and interpretation

- All generated samples are synthetic and independent across configured
  repetitions unless paired explicitly; the article distinguishes conditional
  finite-test fluctuations from Monte Carlo standard error over repetitions.
- Confidence whiskers in figures are descriptive `mean ± 1.96 × MC-SE`,
  not guarantees of nominal coverage under arbitrary data-generating processes.
- The fixed per-experiment choices of n, k, p, distribution, and noise scaling
  matter. Alter them to examine failure modes before exporting a conclusion.
- The full, versioned evidence is in `results/*.csv`; this notebook produces
  separate temporary outputs and therefore does not silently replace it.
- For tests, run `python -m unittest discover -s tests -v`. CI also runs this
  notebook in a fresh kernel.
"""),
    ]

    for i, cell in enumerate(cells):
        cell["id"] = f"geometry-{i:02d}"
    book = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbformat.validate(book)
    destination = Path(__file__).resolve().parent / "nearest_neighbor_geometry.ipynb"
    nbformat.write(book, destination)
    return destination


if __name__ == "__main__":
    print(build())
