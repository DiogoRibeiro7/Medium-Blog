# Causal Inference Is Not a Toolbox — Research Companion

Original figures, documented Python source, and validation checks for the
long-form article *Causal Inference Is Not a Toolbox: Why Identification Comes
Before Estimation*, by Diogo Ribeiro.

The complete editorial manuscript is kept in the private
research-articles-medium repository under the Statistics for Data Scientists
theme. This public companion contains no duplicate manuscript.

## What the figures demonstrate

| Figure | Constructed comparison | Identification lesson |
| --- | --- | --- |
| 01 — Same numbers, different worlds | Adopters retain 65%; non-adopters 50%; user composition alone generates the 30% relative gap | An association need not be a treatment effect |
| 02 — Universal rollout | One post-launch observed path with two plausible untreated trajectories | Without a credible counterfactual, a before/after difference does not identify impact |
| 03 — Which controls? | Confounder, mediator, and collider causal structures | Adding a predictive variable can change or bias a causal estimand |
| 04 — Nonparallel trends | Conventional DiD returns 10 percentage points, while one trend-continuation hypothesis implies 5 | A counterfactual trend must be defended, not inferred from the arithmetic |

These examples are intentionally synthetic and schematic. They are not based
on user-level data from any real product and support no empirical effect claim.

## Reproduce and validate

Requires Python 3.10+ and the packages listed in requirements.txt.

    python -m pip install -r requirements.txt
    python generate_figures.py
    python -m unittest discover -s tests -v

The generation script deterministically writes four publication PNGs and four
SVG versions to the figures directory. Tests verify the two numerical
identities and independently render all four graphics in a temporary folder.
No API key, private data, or network access is required after installation.

## Displayed mathematics for Medium

Medium does not render LaTeX equations directly. The article's fourteen
displayed equations are tracked in the repository-level
medium-equations/manifest.json under the key
statistics-for-data-scientists/causal-inference-is-not-a-toolbox. The PNGs are
stored in the corresponding medium-equations subdirectory and are referenced
from the editorial manuscript by a pinned public Git commit.

The canonical equation-rendering command from the medium-equations directory is:

    python render_equations.py statistics-for-data-scientists/causal-inference-is-not-a-toolbox

All text and figures distinguish observational differences from causal
effects, and every numerical example is labelled as constructed.
