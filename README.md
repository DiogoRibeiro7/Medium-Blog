# Medium-Blog

[![GitHub issues](https://img.shields.io/github/issues/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/issues)
[![GitHub stars](https://img.shields.io/github/stars/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Code, notebooks, data-analysis examples, and reproducible case studies accompanying articles and technical notes.

This repository has accumulated material over many years. It is intentionally treated as a **research and article monorepo**, not as one Python application with one dependency set or one execution environment.

## Repository model

The repository contains two broad classes of material:

1. **Maintained projects** — active or recently modernized studies with explicit environments, documentation, tests, reproducibility checks, and project-scoped automation where appropriate.
2. **Historical article material** — notebooks, scripts, figures, datasets, and small project directories retained primarily as the public computational record behind older articles.

Historical paths are part of the public archive. They are not renamed merely to make the root directory uniform, because external articles, bookmarks, and citations may point to those paths.

See [`docs/REPOSITORY_ARCHITECTURE.md`](docs/REPOSITORY_ARCHITECTURE.md) for the repository contract and [`REPOSITORY_ROADMAP.md`](REPOSITORY_ROADMAP.md) for the modernization programme.

## Editorial queue

New article work is tracked in [`EDITORIAL_QUEUE.md`](EDITORIAL_QUEUE.md). Drafts use stable, URL-friendly project directories so each piece can grow independently into references, code, data, figures, or reproducibility material when needed.

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
