# Editorial manuscript separation

## Purpose

Medium manuscripts are maintained in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium) repository. `Medium-Blog` remains the public computational companion: code, tests, data provenance, research figures and reproducible outputs are retained here.

- Source snapshot imported: [`6e6f84c85931`](https://github.com/DiogoRibeiro7/Medium-Blog/commit/6e6f84c8593198fbd70811fd8d1915def98373a8).
- Later synchronized amendment: causal inference part 16 from [`d603e96c1986`](https://github.com/DiogoRibeiro7/Medium-Blog/commit/d603e96c19869780e04f38df294808368453d5f2).
- Private import: [PR #2](https://github.com/DiogoRibeiro7/research-articles-medium/pull/2), 90 manuscripts and editorial indexes.
- Private editable source templates and sections: [PR #17](https://github.com/DiogoRibeiro7/research-articles-medium/pull/17), merged before public generator cleanup.
- Public migration stages 1–8: separate 89 manuscript bodies by compatibility notices
  or source-file removals after private archiving.
- Deferred: 1 manuscript requiring further publication-generator or source-audit decoupling.

## Why stages are required

Executable publication checks in the public repository currently read some article files and generated publication outputs. Replacing those manuscripts immediately would break otherwise valid research CI. The code and its tests must first be decoupled from editorial content. A public compatibility notice is not a replacement for a manuscript in a scientific assertion.

The private import must be merged and verified before this public cleanup is merged. Subsequent cleanup PRs should make tests validate computational inputs, generated figures and results independently of editorial Markdown. Generated article templates and related publication scripts require individual assessment.

## Stage 1 — compatibility notices

- `a-correlation-is-not-a-mechanism/article.md`
- `anonymous-bank-redials/article.md`
- `billionaires-poverty-audit/article.md`
- `causal-transportability-for-data-scientists/article.md`
- `civility-and-political-change/article.md`
- `clustering-is-not-a-population/article.md`
- `conformal-prediction-for-data-scientists/article.md`
- `decision-focused-optimization-for-data-scientists/article.md`
- `dependent-data-uncertainty-for-data-scientists/article.md`
- `extreme-value-theory-for-data-scientists/article.md`
- `feature-importance-is-not-causality/article.md`
- `gaussian-graphical-models-for-data-scientists/article.md`
- `gilded-age-political-economy/article.md`
- `hotel-cancellation-replay/article.md`
- `hybrid-discrepancy-generators/article.md`
- `industrial-statistics-for-data-scientists/article.md`
- `inverse-problems-for-data-scientists/article.md`
- `mc-dropout-posterior/article.md`
- `measurement-error/article.md`
- `mechanistic-vs-data-driven-generators/article.md`
- `missing-data-is-not-zero/article.md`
- `mixed-effects-models/article.md`
- `monty-hall/part1/article.md`
- `monty-hall/part2/article.md`
- `monty-hall/part3/article.md`
- `monty-hall/part4/article.md`
- `multiple-imputation/article.md`
- `nearest-neighbor-geometry-for-data-scientists/article.md`
- `pca-vs-latent-factors-for-data-scientists/article.md`
- `portugal-household-spending/article.md`
- `portugal-iva-zero/article.md`
- `portugal-wage-counterfactual/article.md`
- `prescribing-individuals/article.md`
- `queueing-theory-for-data-scientists/article.md`
- `rare-event-detection-for-data-scientists/article.md`
- `regression-to-the-mean/part1/article.md`
- `regression-to-the-mean/part2/article.md`
- `regression-to-the-mean/part3/article.md`
- `regression-to-the-mean/part4/article.md`
- `reliability-statistics-for-data-scientists/article.md`
- `robust-regression/article.md`
- `sequential-change-detection-for-data-scientists/article.md`
- `significance-selection-winners-curse/article.md`
- `simpsons-paradox/part1/article.md`
- `simpsons-paradox/part2/article.md`
- `simpsons-paradox/part3/article.md`
- `simpsons-paradox/part4/article.md`
- `synthetic-data-evaluation/article.md`
- `tax-cuts-rich-historical-cycles/article.md`
- `tenure-by-age-at-hire/article.md`

## Reconciled compatibility notices (previously misclassified)

- `a-dataset-is-not-a-population/article.md`
- `a-label-is-not-ground-truth/article.md`
- `civil-rights-great-society/article.md`
- `financialization-before-2008/article.md`
- `great-recession-unequal-recovery/article.md`
- `new-deal-social-contract/article.md`
- `outlier-is-not-an-error/article.md`
- `pandemic-inflation-unequal-recovery/article.md`
- `progressive-era-reform-gap/article.md`
- `spurious-regression-cointegration/article.md`
- `stagflation-market-turn/article.md`
- `the-test-set-is-not-an-oracle/article.md`

## Stage 2 — causal inference series

- `causal_inference_statistics/part1/article.md`
- `causal_inference_statistics/part2/article.md`
- `causal_inference_statistics/part3/article.md`
- `causal_inference_statistics/part4/article.md`
- `causal_inference_statistics/part5/article.md`
- `causal_inference_statistics/part6/article.md`
- `causal_inference_statistics/part7/article.md`
- `causal_inference_statistics/part8/article.md`
- `causal_inference_statistics/part9/article.md`
- `causal_inference_statistics/part10/article.md`
- `causal_inference_statistics/part11/article.md`
- `causal_inference_statistics/part12/article.md`
- `causal_inference_statistics/part13/article.md`
- `causal_inference_statistics/part14/article.md`
- `causal_inference_statistics/part15/article.md`
- `causal_inference_statistics/part16/article.md`
- `causal_inference_statistics/part17/article.md`
- `causal_inference_statistics/part18/article.md`
- `causal_inference_statistics/part19/article.md`
- `causal_inference_statistics/part20/article.md`

## Stage 3 — metropolitan affordability manuscript

- `urban-affordability-cascade/article.md`

## Stage 4 — restaurant portion economics manuscript\n\n- `restaurant-portion-economics/publication/medium/article.md`\n- Seven modular editorial sections archived privately (research-articles-medium PR #20) and removed from this public repository.\n- Public article assembly now requires an explicit private checkout and writes only into that checkout; research tests and figures remain public.\n\n## Stage 5 — Iberian electricity manuscript

- `iberian-electricity-lp/publication/article.md`
- The editorial source template and article are archived in the private repository; public numerical-value checks and figure generation no longer require them.

## Stage 6 — routing on real roads

- `routing-on-real-roads/publication/article.md`
- The source template and full article are archived privately. The public publication command generates numeric values, research figures, and a short notice; scientific assertions and attribution remain independently testable.

## Stage 7 — European wheat and maize weather studies

- `eu-wheat-weather/publication/article.md` (removed from public main)
- `eu-wheat-weather/publication/maize/article.md` (removed from public main)
- Both original articles and editable templates were preserved privately in
  [PR #17](https://github.com/DiogoRibeiro7/research-articles-medium/pull/17).
- Numerical manifests, ten research figures, scientific tests, data licences
  and provenance remain public and independent of the editorial manuscripts.

## Stage 8 — EU Okun Law Medium article

- `eu-okun-law/publication/medium/article.md` and its template were removed
  in [PR #559](https://github.com/DiogoRibeiro7/Medium-Blog/pull/559),
  after private archiving.
- The scientific LaTeX manuscript, numerical manifests, tests, research figures,
  source provenance and bibliography validation remain public.

## Deferred manuscript paths

- `midterm-forecast-fit-vs-prediction/article.md`

## Historical links and privacy

Compatibility notices preserve most historical paths; others were intentionally removed after private archiving. The article prose is absent from the current public branch. Earlier public commits, forks, cached pages and prior pull-request diffs can still contain the manuscripts. This migration is an ownership split, not a history purge.
