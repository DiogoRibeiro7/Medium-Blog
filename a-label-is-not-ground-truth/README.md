# A Label Is Not Ground Truth — reproducible companion

Article #32 of the `Medium-Blog` long-form series. The [manuscript](article.md) addresses binary label noise, annotation quality, latent-variable aggregation, selective observation, test-label bias, entity leakage and label-policy changes. It includes original derivations and nine seeded experiments.

**All observations and latent outcomes are synthetic.** The simulation knows latent truth because it creates it. A deployed dataset ordinarily has no such oracle labels. Reported rates are demonstrations under the specified generators, not estimates of real-world annotation error or a recommendation to relabel any actual person.

## Reproduce

Use Python 3.12 in a virtual environment. The numerical stack is pinned to the versions used for the archived evidence: NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and scikit-learn 1.8.0. Notebook dependencies are bounded, rather than entirely pinned. Numerical outputs can still vary in their least significant digits across platforms and BLAS implementations.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python analysis.py
python -m unittest discover -s tests -v
```

The canonical seed is `20261102`. Each experiment gets an independent `SeedSequence` derived from the seed, a stable study key, and an outer-replication index. Full mode produces one `results/{study}_replicates.csv`, one `results/{study}_summary.csv`, and one `figures/{figure}.png` for each of nine studies. The summary contains outer-replication `mean`, `sd`, Monte Carlo `se`, `lower95`, `upper95`, and an analytic `theory` when available. The intervals are uncertainty for a simulated average, *not* sampling confidence intervals around one real dataset's unknown truth. Empty `theory` cells mean that no analytic reference was supplied.

For a quick smoke test that cannot overwrite canonical results:

```bash
python analysis.py --quick
```

Quick mode writes `_quick` suffixed CSVs/PNGs, with three outer repetitions and smaller datasets or ensembles. It uses the same estimators and plotting code. If you want to change the seed in **full** mode, use an explicit separate directory, for example `python analysis.py --seed 42 --output /tmp/label-study-seed42`. A different seed is rejected for full runs in the article's root directory.

The notebook is rebuilt deterministically using `python notebook_builder.py`. The checked-in `a_label_is_not_ground_truth.ipynb` is intentionally unexecuted; running it in Jupyter performs all nine quick studies under a temporary directory and displays the resulting images. To execute it in a *fresh kernel* through tests (as the path-scoped CI workflow does):

```bash
RUN_NOTEBOOK_EXECUTION=1 python -m unittest discover -s tests -v
```

The workflow [`a-label-is-not-ground-truth.yml`](../.github/workflows/a-label-is-not-ground-truth.yml) requires only `contents: read`, limits jobs to twenty minutes, and runs on matching pull requests or pushes to `main` (no duplicate push runs for every feature-branch update).

## Evidence inventory

| Figure | Experiment key | Full outer reps | What is held fixed and what is varied? |
| --- | --- | ---: | --- |
| `fig1_symmetric_label_noise.png` | `symmetric` | 30 | Independent symmetric training-label flips 0–30%; evaluate probabilities against untouched latent test outcomes |
| `fig2_asymmetric_prevalence.png` | `asymmetric` | 30 | Latent prevalence 25%; vary class-conditional false-positive/false-negative flip channels; correct only with oracle rates |
| `fig3_instance_dependent_noise.png` | `instance` | 30 | Exactly 12% labels flipped in each scenario, but selected uniformly, near the decision boundary, or at high latent margin |
| `fig4_annotator_aggregation.png` | `annotators` | 24 | Three or five conditionally independent workers of differing class-specific skill; majority versus binary Dawid–Skene EM |
| `fig5_correlated_annotator_failures.png` | `correlation` | 24 | Five workers with fixed 20% marginal error; vary the fraction of errors shared at item level, breaking EM independence |
| `fig6_selective_labels.png` | `selection` | 36 | High-score top 10–40% reviewed; compare reviewed-only prevalence with randomized audit plus Horvitz–Thompson estimate |
| `fig7_noisy_test_ranking_reversal.png` | `benchmark` | 30 | Model A is genuinely better; increasingly replace minority test labels with model B's prediction, biasing measured ranking |
| `fig8_entity_leakage.png` | `leakage` | 20 | Repeated entity fingerprints with random entity labels; row split versus truly unseen-entity split |
| `fig9_annotation_policy_drift.png` | `policy` | 30 | Fixed latent outcomes and predictions; change the labeler's false-positive/false-negative policy |

The canonical archive contains **18 CSVs** and **9 original publication PNGs**, plus this README, article, analysis source, notebook generator, executable notebook, pinned numerical dependencies, and tests. No external dataset download, secret, or cloud credential is required.

## Numerical cross-checks

- Symmetric training flips from 0% to 30% keep clean-test AUC near **0.827 to 0.825**, while Brier loss rises from **0.168 to 0.200**. AUC and probability calibration are different quantities.
- With 25% latent prevalence and flip rates `eta0=.05`, `eta1=.30`, the expected recorded positive fraction is **0.2125**; simulations give **0.213**. Oracle correction averages **0.251**, but oracle channel rates are not normally observable.
- Flipping exactly 12% high-margin training labels raises clean-test Brier loss to **0.220**, versus **0.167** when the same fraction is concentrated at the decision boundary. This is one synthetic model and corruption rule, not a universal ordering.
- For five heterogeneous independent annotators, majority accuracy is **0.923** and Dawid–Skene accuracy **0.963**. When workers share a 20% item-level error, majority and EM are both wrong around **20.1%** of the time.
- Reviewing only the top 20% of risk scores yields observed positive prevalence **0.641**, versus latent full-population prevalence **0.276**. A 10%-random-audit HT estimator averages **0.276**, but has greater Monte Carlo variability.
- Model A is truly more accurate (**0.8995** versus **0.8849**), yet replacing only 25% of minority labels with model B's prediction makes recorded test accuracy favor B (**0.9000** versus **0.8873**). Just **1.51%** of all labels are different from latent truth in that condition.
- The row-level split reports **0.984** accuracy, but unseen-entity holdout is **0.482** for deliberately random entity labels. The split unit matters.
- Model accuracy against latent truth stays **0.751** across label-policy conditions; recorded-label accuracy changes from **0.737** to **0.661** or **0.695** without changing predictions.

## Scope and nonidentification boundaries

The conditional noise-channel inversion assumes rates are known and their sum differs from one; the published examples use sums below one, while the correction function also supports negatively oriented, invertible channels. The EM algorithm assumes binary outcomes, complete worker votes, conditional independence, an oriented initialization, and regularized confusion matrices. It does not prove that the consensus equals truth. The shared-error study intentionally violates conditional independence. The review-audit estimator assumes true random audits of the unreviewed population with known inclusion probability; auditing only alerts would not support that correction. The model-ranking reversal is *intentionally preferential label replacement*, not independent random noise. The random entity labels in study eight are constructed to expose an evaluation split error, not to describe actual people. None of these simulations supplies an external validity guarantee.

The manuscript provides the precise definitions, source audit, interpretation limits, and a practical protocol for real-world label governance.

The notebook can be executed with the kernel working directory set to the
article directory or to the repository root. When the kernel starts in an
unrelated directory, point the environment variable LABEL_STUDY_DIR at the
absolute article directory before launching the notebook. Its import cell
checks for both the analysis source and companion notebook before importing.
