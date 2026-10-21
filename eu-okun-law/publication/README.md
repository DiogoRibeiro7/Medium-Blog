# Publication: manuscript, Medium article and reproduction

This directory holds the final outputs of the EU-27 Okun's law study:

- the LaTeX manuscript;
- a Medium version generated from the same values;
- the figures and tables;
- the manifests that freeze data, analyses and environment;
- the audit of every prior output.

Nothing here is published automatically.

## Contents

| Path | What it is | Written by |
| --- | --- | --- |
| `manuscript/main.tex`, `manuscript/sections/*.tex` | The manuscript text | hand |
| `manuscript/references.bib` | Bibliography | hand, verified by code |
| `manuscript/generated/values.tex` | Every number quoted in the text, as `\val{key}` macros | `eu-okun publication` |
| `manuscript/generated/tab_*.tex` | Tables 2, 3, 4, 6 and the appendix tables | `eu-okun publication` |
| `manuscript/figures/*.pdf` | Figures for the manuscript | `eu-okun publication` |
| `medium/article.template.md` | The Medium text, with `{{key}}` placeholders | hand |
| `medium/article.md` | The Medium article with the values filled in | `eu-okun publication` |
| `medium/figures/*.png` | Figures for the Medium article | `eu-okun publication` |
| `manifest/data.json` | SHA-256 of every file of every frozen snapshot | `eu-okun publication` |
| `manifest/experiments.json` | Each analysis, registered or post hoc: protocol, report, commands, seeds, replication, output fingerprints | `eu-okun publication` |
| `manifest/environment.json` | Python requirement, locked package versions, lock-file hash, seeds | `eu-okun publication` |
| `manifest/values.json` | Each quoted value with its LaTeX and Markdown forms and its source file | `eu-okun publication` |
| `manifest/bibliography.json` | The result of resolving every DOI and URL | `eu-okun publication --verify-bibliography` |
| `AUDIT.md` | Snapshot audits, consistency checks and checks of quoted numbers | `eu-okun publication` |

## How the numbers reach the text

1. `collect_values` in `src/eu_okun/publication_values.py` reads every quoted number from
   the committed result tables under `docs/` and records its source file.
2. The build writes these values twice from the same source:
   - as LaTeX macros (`\val{by_mu}`);
   - into the Medium placeholders (`{{by_mu}}`).
3. Compilation stops on an undefined key, and so does the Medium rendering.
4. A test fails if a decimal number other than a protocol constant (a threshold or prior
   scale) is typed by hand in the manuscript or the Medium template.

To change the text, edit `manuscript/sections/*.tex` or `medium/article.template.md`.
Never edit a generated file: `--check` fails if one differs from what the code produces.

## Requirements

- Python 3.12 to 3.14 and Poetry 2.x (CI uses Poetry 2.5.1).
- A LaTeX distribution with `latexmk`, `pdflatex` and `bibtex`, for example TeX Live or
  MiKTeX. The manuscript needs these packages: `amsmath`, `booktabs`, `tabularx`,
  `placeins`, `natbib`, `caption`, `xcolor`, `hyperref`, `geometry` and `lmodern`.
- Internet access only to re-download data or to verify the bibliography.

## Reproduce the article from the committed results

Run from `eu-okun-law/`, one command per line:

```bash
poetry install --extras dev
poetry check --lock
poetry run eu-okun publication --check
poetry run pytest tests/test_publication.py
poetry run eu-okun publication --build
cd publication/manuscript
latexmk -pdf -interaction=nonstopmode main.tex
```

What each command does:

- `--check` runs the audit and regenerates every text output in memory. It fails if an
  audit check fails or a committed file differs.
- `--build` writes the text outputs and the figures.
- `latexmk` produces `publication/manuscript/main.pdf`.

The PDF is not committed. CI builds it and uploads it as the `eu-okun-manuscript`
artifact of the [estimates workflow](../../.github/workflows/eu-okun-law-estimates.yml).

To verify the bibliography again (internet needed):

```bash
poetry run eu-okun publication --verify-bibliography
```

This command:

- resolves every DOI at doi.org and compares the registered title, or the dataset code
  for Eurostat;
- requests every URL;
- rewrites `manifest/bibliography.json`.

## Reproduce the results from the frozen data

Every analysis reads only the frozen snapshots under `snapshots/`. To regenerate a result
table, run that analysis's commands below. Its protocol is the file named in
`manifest/experiments.json`.

The fitting commands marked "expensive" are the MCMC runs, which take from minutes to
about an hour each on a laptop CPU. Their draws are not bit-identical across platforms.
The committed outputs therefore come from one run, and an independent Linux replication
is recorded in each report.

Classical estimates and their review:

```bash
poetry run eu-okun review --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/review --report docs/CLASSICAL_RESULTS_REVIEW.md --published-dir snapshots/ci-results-38754743850 --provenance "CI artifact eu-okun-empirical-baselines, workflow run 38754743850 (commit e73ee68)"
```

Null-imposed bootstraps with errors from the unrestricted residuals (post hoc, not
registered):

```bash
poetry run eu-okun null-residuals --data-dir snapshots/eurostat-2026-10-15 --headline-dir docs/headline-inference --small-sample-dir docs/small-sample --output-dir docs/null-residuals --report docs/NULL_RESIDUALS.md
```

Small-sample inference for the post-2021 tests:

```bash
poetry run eu-okun small-sample --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/small-sample --report docs/SMALL_SAMPLE_INFERENCE.md --review-dir docs/review
```

The core robustness grid without the participation control:

```bash
poetry run eu-okun core-grid --grid docs/review/grid.csv --robustness docs/review/robustness.csv --output-dir docs/core-grid --report docs/CORE_GRID.md
```

Monthly smoothing in the quarterly unemployment rate:

```bash
poetry run eu-okun smoothing --data-dir snapshots/eurostat-2026-10-15 --monthly-dir snapshots/eurostat-monthly-2026-10-18 --review-dir docs/review --bayes-dir docs/bayes --output-dir docs/smoothing --report docs/SMOOTHING.md
```

Okun's law in hours, for every country, with and without the pandemic:

```bash
poetry run eu-okun hours --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/heterogeneity-2026-10-18 --case-study docs/heterogeneity/elasticities.csv --output-dir docs/hours --report docs/HOURS.md
```

A common-correlated-effects mean-group panel:

```bash
poetry run eu-okun cce --data-dir snapshots/eurostat-2026-10-15 --case-study docs/heterogeneity/elasticities.csv --bayes docs/bayes/country_estimates.csv --output-dir docs/cce --report docs/CCE.md
```

Asymmetry between expansions and contractions:

```bash
poetry run eu-okun asymmetry --data-dir snapshots/eurostat-2026-10-15 --case-study docs/heterogeneity/elasticities.csv --output-dir docs/asymmetry --report docs/ASYMMETRY.md
```

Measurement noise or finance, for Luxembourg:

```bash
poetry run eu-okun gdp-measurement --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/gdp-measurement-2026-10-19 --output-dir docs/gdp-measurement --report docs/GDP_MEASUREMENT.md
```

Small-sample inference for the classical headline sums:

```bash
poetry run eu-okun headline-inference --data-dir snapshots/eurostat-2026-10-15 --grid docs/review/grid.csv --output-dir docs/headline-inference --report docs/HEADLINE_INFERENCE.md
```

Hierarchical Bayesian estimates, checks, robustness, dynamics and regimes (the two
`bayes-fit` commands are expensive):

```bash
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes --retry-warmup 3000
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes --specs time_effects,time_effects_participation,participation_no_common_factor,lags_1,dynamic_factor,ardl,ardl_dynamic_factor,regime_shift,regime_shift_final_gdp,pandemic_excluded_dynamic,regime_shift_dynamic,drifting_association --retry-warmup 3000
poetry run eu-okun bayes-export --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15
poetry run eu-okun bayes-report --fits-dir results/bayes --output-dir docs/bayes --report docs/HIERARCHICAL_BAYES.md --data-dir snapshots/eurostat-2026-10-15 --replication-dir docs/bayes/replication-linux
poetry run eu-okun bayes-recovery --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes-recovery
poetry run eu-okun bayes-checks --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/bayes-checks --report docs/BAYES_MODEL_CHECKS.md --recovery-dir results/bayes-recovery
poetry run eu-okun bayes-robustness --fits-dir results/bayes --output-dir docs/bayes-robustness --report docs/BAYES_ROBUSTNESS.md --pairs-csv docs/review/pairs_review.csv
poetry run eu-okun bayes-dynamics --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/bayes-dynamics --report docs/BAYES_DYNAMICS.md
poetry run eu-okun bayes-regime-recovery --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes-regime-recovery
poetry run eu-okun bayes-regimes --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/bayes-regimes --report docs/BAYES_REGIMES.md --recovery-dir results/bayes-regime-recovery
```

Conditional predictive comparison (the fit is expensive):

```bash
poetry run eu-okun predictive-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/predictive
poetry run eu-okun predictive-report --data-dir snapshots/eurostat-2026-10-15 --fits-dir results/predictive --output-dir docs/predictive --report docs/PREDICTIVE_COMPARISON.md
```

One-quarter-ahead forecasts (the fit is expensive):

```bash
poetry run eu-okun forecast-bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/forecast-bayes
poetry run eu-okun forecast-report --data-dir snapshots/eurostat-2026-10-15 --fits-dir results/forecast-bayes --output-dir docs/forecast-comparison --report docs/FORECAST_COMPARISON.md
```

Local projections on externally identified shocks, quarterly and monthly:

```bash
poetry run eu-okun identification --data-dir snapshots/eurostat-2026-10-15 --shocks-dir snapshots/external-shocks-2026-10-16 --output-dir docs/identification --report docs/IDENTIFICATION_RESULTS.md
poetry run eu-okun monthly-identification --data-dir snapshots/eurostat-monthly-2026-10-18 --shocks-dir snapshots/external-shocks-2026-10-16 --quarterly-dir docs/identification --output-dir docs/monthly-identification --report docs/MONTHLY_IDENTIFICATION_RESULTS.md
```

Shift-share feasibility:

```bash
poetry run eu-okun shift-share --data-dir snapshots/shift-share-2026-10-18 --eurostat-dir snapshots/eurostat-2026-10-15 --output-dir docs/shift-share --report docs/SHIFT_SHARE_FEASIBILITY.md
```

Sign-restricted SVAR:

```bash
poetry run eu-okun sign-restrictions --data-dir snapshots/eurostat-2026-10-15 --deflator-dir snapshots/eurostat-deflator-2026-10-18 --output-dir docs/sign-restrictions --report docs/SIGN_RESTRICTIONS_RESULTS.md
```

Case study of Portugal, Spain, the Netherlands and Luxembourg:

```bash
poetry run eu-okun heterogeneity --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/heterogeneity-2026-10-18 --output-dir docs/heterogeneity --report docs/HETEROGENEITY_CASE_STUDY.md
```

Post-hoc check, not registered (added after external reviews on 2026-10-19 and
2026-10-20): the identification and shift-share feasibility results with the pandemic
window widened by one period. The registered results remain primary; see
[docs/WINDOW_SENSITIVITY.md](../docs/WINDOW_SENSITIVITY.md).

```bash
poetry run eu-okun window-sensitivity --output-dir docs/window-sensitivity --report docs/WINDOW_SENSITIVITY.md
```

After any of these, rebuild the publication with `poetry run eu-okun publication --build`.
Then read `AUDIT.md` and the diff of `manifest/values.json` before trusting the text: a
changed value may need different words.

## Re-download the data (optional)

Eurostat and the OECD revise published data, so a new download gives a new snapshot, not
the frozen one. The fetch commands are listed in the project [README](../README.md). The
audit compares each frozen snapshot with its own manifest. It does not compare a frozen
snapshot with today's data.

## Platform notes

- Text files are hashed with LF line endings, so the manifests are identical on Windows
  and Linux. Snapshot files are hashed byte for byte, as in their own manifests.
- A rebuild on another platform changes floating-point results in their last digits
  (about 1e-16 relative). It also changes numerical-error measurements, such as the
  classical review's verification gaps, by more.
  - Reports and outputs are therefore recorded by a fingerprint: the SHA-256 of the text
    with every decimal number rounded to 10 significant digits, and those below 1e-9 in
    magnitude set to zero.
  - The raw values in `manifest/values.json` are rounded the same way.
  - A rebuild on Linux of everything the offline test suite regenerates leaves both
    manifests unchanged, while a change in any result changes the fingerprint.
- PNG figures are excluded from the output hashes because their bytes depend on the
  platform's font rendering. PDF figures are written without a creation date.

## Medium

`medium/article.md` contains no LaTeX and no tables, and the build refuses to write it if
either appears. Medium does not import Markdown files, so the article is posted by hand:

1. paste the text into the editor;
2. upload each PNG from `medium/figures/` where the article references it.

Publication is left to the author.
