<p align="center">
  <img src="assets/project-logo.png" alt="Medium-Blog project logo" width="160" height="160">
</p>

# Medium-Blog

[![GitHub issues](https://img.shields.io/github/issues/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/issues)
[![GitHub stars](https://img.shields.io/github/stars/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Code, notebooks, data-analysis examples, and reproducible case studies accompanying articles and technical notes.

This repository has accumulated material over many years. It is intentionally treated as a **research and article monorepo**, not as one Python application with one dependency set or one execution environment.

## Editorial manuscript migration

The editorial sources for 90 Medium articles have been imported into the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium) repository. This public repository continues to hold the computational studies, figures, code, tests and reproducibility records.

The separation is staged to protect CI: 89 manuscript bodies have been separated from public main through compatibility notices or removal after private archiving; 1 manuscript path still requires independent publication-generator or source-audit refactoring. See [migration tracking](EDITORIAL_MIGRATION.md). Historical public Git commits remain accessible.

## Repository model

The repository contains two broad classes of material:

1. **Maintained projects** — active or recently modernized studies with explicit environments, documentation, tests, reproducibility checks, and project-scoped automation where appropriate.
2. **Historical article material** — notebooks, scripts, figures, datasets, and small project directories retained primarily as the public computational record behind older articles.

Historical paths are part of the public archive. They are not renamed merely to make the root directory uniform, because external articles, bookmarks, and citations may point to those paths.

See [`docs/REPOSITORY_ARCHITECTURE.md`](docs/REPOSITORY_ARCHITECTURE.md) for the repository contract and [`REPOSITORY_ROADMAP.md`](REPOSITORY_ROADMAP.md) for the modernization programme.

## Maintained projects

### Blood-pressure missingness

[`Blood_Pressure_Missingness/`](Blood_Pressure_Missingness/)

**Missing Data Is Not Empty Space: Statistical Analysis of an Irregular Blood-Pressure Tracker**

A privacy-preserving statistical case study of irregular longitudinal blood-pressure measurements. It is currently the reference maintained project in this repository and includes:

- an installable project-local Python package;
- privacy-safe aggregate public data;
- source, privacy, schema, and reproduction contracts;
- regression and notebook tests;
- secret-backed refresh automation;
- package-native analysis and validation entry points;
- project-specific architecture and roadmap documentation.

It is the reference for engineering quality in new maintained work, but its exact structure is **not** imposed retroactively on historical notebooks.

### Forecasting after the cutoff

[`post-cutoff-forecasting/`](post-cutoff-forecasting/)

**Time-series foundation models against classical baselines on Portuguese electricity load published after the models were released**

Chronos-2, TimesFM 2.5 and TiRex against seasonal naive and ETS baselines on hourly Portuguese load. The evaluation window starts after the last of the three models was released, and the results are compared with earlier windows that the models could have seen. The project has its own package, tests on synthetic data, a CI workflow, and a manually started workflow that runs the full experiment and opens a pull request with the results.

## Research studies in development

### Backward evaluation of forecasting models

[`backward-evaluation/`](backward-evaluation/) — fits a forecasting model on every block of a series and scores it on every other block, earlier ones included. The train-period × test-period matrix identifies level, volatility and persistence changes from its shape and asymmetry; its last row becomes a rule for the training window, tested forward on simulated breaks and on US industrial production and CPI inflation, 1961–2026. Reusable functions, tests and a CI workflow that reruns everything.



### Evidence-weighted pooling of the backward window (pre-registered test)

[`backward-evaluation-pooling/`](backward-evaluation-pooling/) — a pre-registered test of whether weighting the backward-window forecast by the strength of the evidence for a change keeps the gains after lasting changes and removes the losses after temporary episodes. Simulations with regimes and episodes, a premise test (does a stronger rejection predict a lasting change?) and a forward test on eight US series not used before. Verdict: not supported.

[`backward-evaluation-episode-end/`](backward-evaluation-episode-end/) — a pre-registered test of an end-of-episode check for the backward training-window rule: when the walk stops, ask whether the most recent year looks like the rejected past again and, if so, train on all the data. Simulations with Gaussian and Student-t models, ten test series from Japan, Canada, the UK, Germany and Switzerland, and the ten US series as an in-sample development set. Verdict: not supported; a pre-registered prediction that changes in rates and exchange rates persist more than changes in real activity held in direction.

[`backward-evaluation-generality/`](backward-evaluation-generality/) — a pre-registered test of whether three findings about backward training windows, developed on monthly macroeconomic series, hold on daily data from other domains: 25 countries' electricity load (Open Power System Data) and 304 Wikipedia articles' page views, forecast one day ahead by an autoregression, a seasonal linear model, Holt–Winters and LightGBM with Gaussian and Student-t distributions. Sharded computation in GitHub Actions. Pooling over window lengths beats the backward rules in all 8 domain × model cells; the end-of-episode check and the heavy-tail findings do not travel.

[`backward-evaluation-model-choice/`](backward-evaluation-model-choice/) — a pre-registered test of backward evaluation as a way to choose which forecasting model to deploy, against the usual forward backtest: every 91 days, four candidates (autoregression, seasonal linear model, Holt–Winters, LightGBM) are scored on the recent past or fitted on the recent year and scored on older data, and the best is deployed. Test set of 245 Wikipedia articles from seven lists fixed before the data were downloaded; development set of 25 electricity-load series and 304 articles. Verdict: the backward choice is worse than the forward one in all seven lists; pooling the candidates beats both.

[`backward-evaluation-guide/`](backward-evaluation-guide/) — the last part of the backward-evaluation series: `bwdiag.py`, a single-file module that computes the train-period × test-period matrix of any forecasting model and splits every cell into level (removed by a constant shift), spread (removed by a rescaling) and structure (what neither removes), with regime splits and real-time monitoring. Worked readings on simulated changes, US inflation and industrial production, electricity load and 549 Wikipedia articles, and one forward check fixed in advance: a level-normalised LightGBM, suggested by the matrix, beats LightGBM on 239 of 245 test articles.

### Statistical evidence and decision value

[statistical-evidence-and-decisions/](statistical-evidence-and-decisions/) —
a reproducible notebook study comparing PCA, coefficient inference,
regularisation, supervised components and decision utility. A follow-up
robustness notebook uses paired simulations of missing data, contaminated
training outcomes and sample-size sensitivity to test which conclusions survive.


### Classifier drift in Hansard

[`hansard-classifier-drift/`](hansard-classifier-drift/) — validates a dictionary classifier of the Chair's disciplinary interventions in the House of Commons, 1979–2021, era by era against a blind hand-labelled sample, and splits the change in its counts into a change in behaviour and a change in the instrument's precision and recall. Stage 1 (corpus, tags, naive rates, sample) runs on GitHub Actions; the corrected estimates wait for the hand labels.

### Okun's law across the EU-27

[`eu-okun-law/`](eu-okun-law/) — independently investigate the relationship between quarterly GDP growth, unemployment and labour-force participation across 27 countries. Data provenance and validation are implemented; econometric estimates and causal analysis are planned, not yet claimed.

## Historical archive

The remainder of the repository contains article-supporting material across statistics, machine learning, mathematics, data engineering, finance, forecasting, networks, and applied data science.

Some items are complete reproductions of published analyses. Others are exploratory notebooks, code fragments, figures, small datasets, or teaching-style examples. Their presence should not be interpreted as a claim that every historical artifact is currently executable under one modern environment.

A machine-readable catalogue with explicit maintenance and reproducibility status is the next repository-level modernization step.

## Reproducibility levels

| Tier | Meaning |
| --- | --- |
| **A — Maintained** | Reproducible environment, documented execution path, tests or validation gates where appropriate, and active maintenance. |
| **B — Reproducible archive** | Historical material with enough pinned context or instructions to reproduce the main result, but not necessarily under active development. |
| **C — Reference archive** | Preserved historical material that remains useful as an article/code reference but is not guaranteed to run unchanged today. |
| **D — Unclassified** | Material not yet audited during the repository modernization programme. |

`Blood_Pressure_Missingness` is the first Tier-A project. Existing historical material remains unclassified until it is actually reviewed.

## Rules for new work

For **new** projects added to this repository:

- use stable, URL-friendly names;
- avoid spaces, trailing spaces, and non-breaking spaces in new paths;
- keep dependencies local to the project rather than introducing a repository-wide Python environment;
- include a project README explaining the question, data, execution path, and outputs;
- keep generated artifacts close to the project that owns them;
- add project-scoped CI only when the project has something meaningful to validate;
- do not modify unrelated historical material to satisfy a repository-wide style preference.

These rules are prospective. Historical naming is preserved unless a separate migration has a strong reason, a compatibility plan, and explicit review.

## Automation

GitHub Actions in this repository are **project scoped**. A workflow should declare the paths it owns and should not imply that unrelated historical notebooks share its environment or quality guarantees.

At present, the maintained automated analysis belongs to `Blood_Pressure_Missingness`. Repository-wide automation should remain limited to lightweight structural checks and catalogue consistency rather than trying to execute every historical notebook.

## Modernization principle

> **Classify first, document second, automate third, and only then consider moving or deleting historical material.**

See [`REPOSITORY_ROADMAP.md`](REPOSITORY_ROADMAP.md) for the phased plan.

## License

Unless a subproject states otherwise, repository content is licensed under the [Apache License 2.0](LICENSE).
