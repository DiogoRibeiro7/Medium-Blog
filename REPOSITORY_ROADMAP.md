# Medium-Blog repository modernization roadmap

## Status

The repository is a long-lived public archive of notebooks, code, figures, datasets, and article-support material created across different periods and conventions. It should not be treated as one Python application or forced into a single runtime.

The modernization goal is therefore:

```text
preserve historical public paths
        +
make maintained projects reproducible
        +
turn the repository root into a curated catalogue
        +
add repository-wide hygiene without rewriting history
```

`Blood_Pressure_Missingness` is the current reference implementation for a maintained, reproducible project inside the repository. It is a reference for engineering quality, privacy, reproducibility, and CI discipline, but it is not a template that every historical notebook must adopt.

## Non-negotiable migration rules

1. **Do not break historical article links unnecessarily.** Existing paths may be referenced from Medium posts, external pages, bookmarks, or citations and must be treated as public interfaces until proven otherwise.
2. **Do not mass-rename legacy folders or notebooks for aesthetic consistency.** New work follows modern naming rules; historical work may remain historically named.
3. **Do not create one global Python environment for the entire repository.** Maintained projects own their own dependencies and execution contracts.
4. **Do not claim legacy material is reproducible unless it has been checked.** Catalogue status must distinguish maintained, archived, historical, and unknown projects.
5. **Do not delete data or artifacts merely because they look old or untidy.** Deletion requires provenance and usage checks.
6. **Repository-wide cleanup must remain separable from scientific changes.** Structural maintenance should not silently change calculations or conclusions inside projects.

## Target repository model

The repository should evolve into a curated article/research monorepo with three conceptual classes of content.

### Maintained projects

Projects with an explicit README, dependency boundary, reproducible entry point, tests or validation where appropriate, and a clear maintenance status.

`Blood_Pressure_Missingness` is the first example.

### Stable historical articles

Published article-support material that should remain available at its historical path. These projects may use old dependencies, naming conventions, or notebook styles. They are preserved primarily for reference and link stability.

### Unclassified legacy material

Content whose purpose, publication status, dependencies, or external-link surface has not yet been audited. It remains untouched until classified.

The root catalogue, rather than physical relocation, should provide the primary organization layer.

---

## Phase 1 — Establish the repository contract

### Objective

Make the root of the repository explain what the repository is, what is maintained, and what guarantees users can expect.

### Work

- Rewrite the root `README.md` as the repository landing page.
- Fix stale or malformed badges and avoid advertising one Python version as if it applied repository-wide.
- Describe the repository as a collection of article and research projects rather than a single application.
- Highlight maintained projects separately from historical material.
- Add a short repository-maintenance policy.
- Add `docs/REPOSITORY_ARCHITECTURE.md` defining the maintained/historical/unclassified model.
- Link this roadmap from the root README.

### Exit condition

A new visitor can determine, from the repository root alone:

- what the repository contains;
- which projects are actively maintained;
- which material is archival;
- where reproducibility guarantees apply;
- where to find project-specific instructions.

### Non-goals

- renaming old directories;
- moving notebooks;
- changing article code;
- introducing repository-wide dependencies.

---

## Phase 2 — Build a machine-readable repository catalogue

### Objective

Replace implicit knowledge about dozens of folders and notebooks with an auditable inventory.

### Work

Create a machine-readable catalogue, for example:

```text
data/repository_catalogue.json
```

Each entry should eventually record fields such as:

```text
path
kind
status
topic
language
primary_runtime
article_url
readme_present
reproducibility_status
last_audited
notes
```

Recommended status vocabulary:

```text
maintained
stable-historical
archived
unclassified
```

Recommended reproducibility vocabulary:

```text
verified
partial
historical-environment
not-checked
not-applicable
```

Generate a human-readable catalogue from the machine-readable source rather than maintaining two independent inventories.

### First audit pass

The first pass should classify metadata only. It should not attempt to execute every notebook.

Capture obvious repository-level anomalies such as:

- spaces and non-breaking spaces in paths;
- trailing whitespace in names;
- standalone root notebooks;
- standalone root data files and figures;
- unusually large tracked notebooks or artifacts;
- empty or ambiguous files such as `test.py`;
- projects with and without READMEs.

### Exit condition

Every top-level content item is represented in the catalogue and has at least a provisional status.

---

## Phase 3 — Define conventions for all new work

### Objective

Stop adding new entropy while leaving historical paths stable.

### New-project naming

For new projects use lowercase kebab-case directory names:

```text
article-topic-name/
```

Avoid spaces, non-breaking spaces, trailing whitespace, `-main` suffixes copied from downloaded archives, and generic names such as `Untitled1.ipynb`.

### Minimum maintained-project structure

A maintained Python project should normally contain only the pieces appropriate to that project, but the preferred shape is:

```text
project-name/
  README.md
  pyproject.toml
  package_or_src/
  tests/
  data/          # only public/safe data
  figures/
  docs/
```

Notebook-first projects may be simpler, but must still document their environment and execution path.

### Project-local dependency rule

Dependencies belong to the maintained project that uses them. The repository root should not accumulate a synthetic `requirements.txt` merely to make Dependabot or CI convenient.

### Exit condition

All newly created projects follow documented naming, dependency, provenance, and README conventions.

---

## Phase 4 — Repair repository-level automation

### Objective

Make automation reflect the monorepo structure instead of pretending the repository has one Python dependency graph.

### Work

- Keep project-specific workflows path-scoped.
- Preserve the blood-pressure workflows as project-specific automation.
- Fix Dependabot so Python dependency updates target actual maintained dependency roots rather than `/` when no root Python project exists.
- Keep GitHub Actions dependency updates at repository level.
- Add a lightweight repository-hygiene workflow that does not execute historical notebooks.

The hygiene workflow should check only structural invariants, for example:

- root README links to maintained projects and roadmap;
- catalogue JSON parses and follows its schema;
- every catalogue path exists;
- no **new** top-level path violates naming conventions unless explicitly classified as historical;
- no accidental secrets or obviously forbidden private source files are introduced;
- maintained projects referenced by the catalogue have their required metadata files.

### Important boundary

Repository-wide CI must not suddenly require every historical notebook to run under a modern Python version. That would convert an archive-preservation problem into an endless dependency-repair project.

### Exit condition

Automation mirrors the repository's actual maintenance model and does not generate meaningless root-level dependency updates.

---

## Phase 5 — Curate the root without breaking historical paths

### Objective

Reduce root-level visual noise while preserving URLs and provenance.

### Work

Audit standalone root files and classify each as one of:

```text
historical article artifact
project-owned artifact
repository infrastructure
orphan candidate
```

Examples requiring audit include root-level notebooks, PNG/JPEG figures, CSV/XLSX/RData files, and ad hoc test files.

For artifacts that clearly belong to a project but cannot be moved safely because of external links, prefer catalogue metadata and documentation over relocation.

For genuine orphan candidates:

1. search repository references;
2. check git history/provenance;
3. check likely Medium/external use where practical;
4. document the decision;
5. delete only in a dedicated cleanup PR.

### Exit condition

Every root-level non-infrastructure file has an explicit catalogue classification and unexplained clutter no longer accumulates.

---

## Phase 6 — Progressive legacy-project audits

### Objective

Improve valuable historical projects without turning modernization into a rewrite of the entire repository.

### Prioritization

Audit projects based on value and likelihood of reuse rather than alphabetically.

Suggested priority score:

```text
priority = publication relevance
         + technical/research value
         + external-link importance
         + expected reuse
         - migration risk
```

A legacy audit should answer:

- What article or research question does this support?
- Is there an external publication link?
- What runtime/language was used?
- Are required datasets present and legally/publicly shareable?
- Does the notebook/code still execute?
- Are there hidden assumptions or obsolete APIs?
- Is the result worth maintaining, preserving as historical, or archiving?

### Possible outcomes

#### Promote to maintained

Add a project README, explicit environment, reproducible execution, tests/validation where justified, and project-scoped CI.

#### Preserve as stable historical

Document the original environment and purpose but do not spend engineering effort modernizing it.

#### Archive

Retain for provenance but mark clearly as no longer maintained or recommended.

### Exit condition

The most valuable legacy projects have intentional statuses rather than accidental ones.

---

## Phase 7 — Reproducibility tiers

### Objective

Use a realistic reproducibility standard instead of a binary "works/does not work" label.

### Tier A — maintained and verified

- documented environment;
- deterministic or explicitly stochastic reproduction contract;
- tests/validation as appropriate;
- CI;
- provenance for data and external sources.

### Tier B — reproducible historical

- documented environment or lockfile;
- known execution instructions;
- no promise of continuous modernization.

### Tier C — preserved historical

- purpose and provenance documented;
- original artifacts retained;
- execution may require obsolete dependencies or external data.

### Tier D — unclassified

- not yet audited.

`Blood_Pressure_Missingness` should remain Tier A.

### Exit condition

The root catalogue exposes a reproducibility tier for every audited project.

---

## Phase 8 — Repository size and artifact policy

### Objective

Prevent uncontrolled growth without rewriting git history casually.

### Work

- identify large notebooks and generated outputs;
- distinguish source notebooks from rendered/generated artifacts;
- establish a policy for notebook output stripping for **new** projects where appropriate;
- avoid committing large derived datasets that can be regenerated reliably;
- document when binary data legitimately belongs in git;
- consider Git LFS only for future assets that genuinely require it, not as a reflexive migration of history.

Do not rewrite repository history merely to reduce size unless there is a compelling operational or security reason.

### Exit condition

New content has a clear artifact-size policy and repository growth becomes intentional.

---

## Phase 9 — Publication and provenance metadata

### Objective

Make the repository useful as a map between code, analyses, and published writing.

### Work

For audited projects, record when available:

- article title;
- Medium/publication URL;
- publication date;
- code path;
- primary topic;
- data sources;
- citation or attribution requirements;
- reproducibility tier;
- maintenance status.

Optionally generate topic views such as:

```text
statistics
machine-learning
time-series
causal-inference
finance
data-engineering
mathematics
Portugal
```

### Exit condition

The repository acts as a discoverable research/article portfolio rather than a filesystem that visitors must reverse-engineer.

---

## Phase 10 — Release and maintenance discipline

### Objective

Keep the repository orderly after the initial modernization.

### Maintenance rules

For every new substantial article/project:

1. add/update the catalogue entry;
2. provide a meaningful README or notebook introduction;
3. declare whether the project is maintained or historical;
4. document dependencies locally;
5. keep generated/private data boundaries explicit;
6. add project-scoped CI when continuous reproducibility is promised;
7. link the published article when available.

For repository-level changes:

- prefer small reviewable PRs;
- separate path moves/deletions from code changes;
- preserve public paths unless migration value clearly exceeds breakage risk;
- update the catalogue and root README when maintenance status changes.

---

## Proposed implementation order

The recommended sequence is:

```text
1. root README + repository architecture document
2. machine-readable catalogue and schema
3. conventions for new projects
4. Dependabot and repository-hygiene CI
5. root-file audit
6. prioritized legacy-project audits
7. reproducibility tiers and publication metadata
8. artifact/size policy
9. ongoing maintenance
```

This order deliberately puts **discovery and classification before movement or deletion**.

## Immediate next PRs

### PR A — repository landing page

Rewrite the root README and add `docs/REPOSITORY_ARCHITECTURE.md`. No project paths move.

### PR B — catalogue bootstrap

Add the catalogue schema and enumerate every top-level item with provisional status `unclassified`, while marking known infrastructure and `Blood_Pressure_Missingness` explicitly.

### PR C — automation alignment

Correct Dependabot and add catalogue/hygiene validation without attempting to execute historical projects.

### PR D — root artifact audit

Classify standalone root notebooks, data files, figures, and test files. This PR should record decisions, not perform broad deletions.

Only after those four slices should individual legacy projects begin promotion, archival, or cleanup.

## Definition of success

The repository modernization is successful when:

```text
historical links remain stable
AND maintained projects are reproducible
AND every top-level item is classified
AND the root README is a useful catalogue
AND automation matches actual project boundaries
AND new work follows consistent conventions
AND legacy modernization is intentional rather than cosmetic
```

The goal is not to make every historical artifact look new. The goal is to make the repository **legible, trustworthy, maintainable, and honest about what is and is not reproducible**.
