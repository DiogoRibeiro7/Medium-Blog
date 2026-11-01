# Article #29 — A Feature Importance Score Is Not a Causal Effect

**Full title:** *A Feature Importance Score Is Not a Causal Effect: Permutation, Dependence, and the Limits of Model Explanations*

The [11,000+ word manuscript](article.md) develops mathematical estimands and
nine controlled synthetic experiments. The companion is an executable
illustration of model reliance and attribution assumptions, not a benchmark
of causal discovery and not an analysis of any field population. It produces
**nine original publication figures** and **eighteen machine-readable CSV
tables** (one replicate table and one summary per experiment).

## Reproduce

Use Python 3.12 from this directory:

    python -m pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
    RUN_NOTEBOOK_EXECUTION=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

The final command executes the notebook using an independent, **fresh kernel**.
It requires permission for local Jupyter IPC sockets. The path-scoped GitHub
Actions workflow performs this full test suite on relevant pull requests.

The canonical seed is **20261101**. The script supports:

    python analysis.py --quick
    python analysis.py --quick --output /tmp/importance-smoke
    python analysis.py --seed 73 --output /tmp/importance-sensitivity
    python notebook_builder.py

Quick mode uses three independent outer replicates and produces filenames
ending in _quick.csv and _quick.png. Full mode uses 18–24 independent outer
replicates and writes canonical filenames. The notebook invokes quick mode
in a temporary output directory; it does not modify canonical artifacts.

NumPy SeedSequence streams are keyed by the requested seed, a stable CRC32
experiment tag, and the outer replication index. The relevant study controls
are embedded alongside each function in [analysis.py](analysis.py).

## Evidence inventory

| Study / CSV prefix | Figure | Full-run design | Target |
| --- | --- | --- | --- |
| conditional | 1 | n=1,500, ρ=0, .3, .6, .9, .98; 24 repeats | Exact marginal versus conditional replacement risk |
| proxies | 2 | 380 train/1,000 test, four proxy-noise scales; 24 repeats | Fixed-model permutation versus retraining ablation |
| impurity | 3 | 440 train/850 test, RF with 90 trees; 18 repeats | Impurity split-credit versus held-out reliance |
| shapley | 4 | n=1,100, four correlations; 24 repeats | Two exact Shapley coalition games for f(x)=x₁ |
| interaction | 5 | 850 train/1,200 test, depth-two tree; 24 repeats | XOR univariate screening versus interaction |
| leakage | 6 | 500 train/1,000 test; 24 repeats | Retrospective outcome proxy versus deployment availability |
| shift | 7 | n=1,100, three feature-variance regimes; 24 repeats | Ranking under fixed oracle coefficients and shifted P(X) |
| subgroup | 8 | n=2,500, minority shares 5–50%; 24 repeats | Global average versus subgroup-specific reliance |
| selection | 9 | 180 train/250 selection/800 verification, p=10/40/80; 24 repeats | Winner-selection bias on independent null features |

Each CSV record includes a named condition and metric. Replicate files contain
the outer replicate ID and raw score. Summary files report independent
replicate count, mean, sample standard deviation, Monte Carlo standard error,
normal ±1.96-SE interval, and a known analytic benchmark when available.
Some conditions are paired within a replicate. Their **marginal** MC intervals
are not independent-sample confidence tests of a contrast. For a paired
contrast, analyze paired replicate-level differences.

All figures are derived from the same recorded summaries:

1. fig1_conditional_permutation.png
2. fig2_proxy_reliance.png
3. fig3_impurity_bias.png
4. fig4_shapley_games.png
5. fig5_xor_interaction.png
6. fig6_leakage_availability.png
7. fig7_importance_shift.png
8. fig8_subgroup_averaging.png
9. fig9_selection_optimism.png

## Exact identities and non-equivalences

With a centered standard-Gaussian pair of correlation ρ, oracle
f(x)=βx₁, and Y=βX₁+ε, the expected risk increase under a new marginal
X₁ draw is 2β². Under a new **conditional** X₁ draw given X₂, it is
2β²(1−ρ²). Conditional draws preserve pairwise dependence by construction;
neither replacement is automatically a real-world intervention.

With f(x₁,x₂)=x₁, the exact two-coordinate Shapley values are:

- Marginal-value game: (x₁, 0).
- Observational conditional-value game: (x₁−ρx₂/2, ρx₂/2).

Both games satisfy efficiency for their different coalition values. Neither
identifies the causal effect of changing X₂ on an observed outcome.

Other experiments show that retraining can compensate for correlated
proxies; RF impurity can reward continuous unrelated noise; zero marginal
correlation can coexist with a strong interaction; a leak can be unusable
at deployment; risk-based importance rankings depend on the evaluation
distribution; a subgroup-critical variable can have low global score; and
maximizing scores over null features creates validation-selection optimism.

## Limits and provenance

- These are **synthetic observations** with stated generator assumptions,
  not source-transcribed field measurements.
- Numbers and error bars describe the specified Monte Carlo designs. They
  are not estimates of prevalence, causality, safety, or fairness in any
  actual industry or population.
- The model, loss, sample distribution, feature dependence, perturbation
  semantics, and availability timeline must all be defined in a real use.
- The RF MDI fraction, permutation MSE increase, Shapley allocation, and
  R-squared are not in common units and must not be ranked against each other.
- The manuscript uses scholarly original sources and links each methodological
  family in its source-audit section. All article-specific numerical findings
  come from the committed CSVs.
- The deterministic notebook is built by [notebook_builder.py](notebook_builder.py)
  and verified against [feature_importance_is_not_causality.ipynb](feature_importance_is_not_causality.ipynb).

## Validation

The **107-test local suite** checks numerical identities, input validation,
independent seed streams, perturbation contracts, exact attribution
efficiency, nine studies, eighteen tables, nine nontrivial PNG figures,
canonical numerical results, summary intervals, notebook structure, and
the generated notebook's byte-level JSON structure. Two notebook-library
checks are skipped locally if unavailable or not enabled; CI installs
the dependencies and sets RUN_NOTEBOOK_EXECUTION=1 so both run.

The repository's global Git LFS CSV policy is overridden **only** for this
article's versioned result tables using a path-specific .gitattributes
entry. The workflow triggers only for files in this article directory or
its own workflow file, keeping unrelated article builds independent.
