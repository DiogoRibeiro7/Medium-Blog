# Repository Architecture

## Purpose

`Medium-Blog` is a long-lived public archive of computational material supporting articles, technical notes, experiments, and research case studies.

It is **not** one software application and should not be forced into one repository-wide runtime, dependency graph, or packaging model.

The repository architecture therefore optimizes for four things:

1. preservation of historical public paths;
2. clear ownership boundaries for maintained projects;
3. honest reproducibility claims;
4. incremental modernization without rewriting the archive for cosmetic consistency.

## Core invariant

Historical paths are treated as public interfaces unless there is explicit evidence that changing them is safe.

That means repository modernization must not casually rename or move directories, notebooks, figures, or data files simply to normalize naming conventions. Older Medium articles, external links, citations, bookmarks, and code references may depend on those paths.

For new work, the repository uses modern conventions prospectively.

## Content classes

Every top-level item will eventually be classified into one of the following ownership classes.

### Maintained project

A maintained project has an identified scientific or technical purpose and an explicit maintenance boundary. Depending on the project, that may include:

- a project README;
- a local environment definition;
- package or module structure;
- tests or validation gates;
- reproducible public artifacts;
- project-scoped GitHub Actions;
- privacy or data-source contracts where necessary;
- a local roadmap.

`Blood_Pressure_Missingness` is the first maintained reference project.

### Reproducible archive

Historical material whose principal result can still be reproduced from documented dependencies, data, and commands, but which is not under active feature development.

### Reference archive

Historical material retained because it supports or documents an article, technique, example, or past implementation, but which is not guaranteed to execute unchanged under current environments.

### Unclassified

Material that has not yet been audited. Unclassified does not mean low quality or broken; it means that the repository has not yet made a verified maintenance or reproducibility claim about it.

## Reproducibility tiers

The repository uses four explicit tiers:

- **Tier A — Maintained:** active maintenance, documented environment and execution path, with tests or validation where meaningful.
- **Tier B — Reproducible archive:** main result reproducible with documented historical context, without a promise of active development.
- **Tier C — Reference archive:** useful preserved material with no guarantee of current executability.
- **Tier D — Unclassified:** not yet reviewed.

Tiers are assigned from evidence, not inferred from directory names, age, language, or apparent complexity.

## Dependency ownership

There is no repository-wide Python environment.

Dependencies belong to the project that uses them. A maintained Python project may define its own `pyproject.toml`, requirements, lock file, or equivalent. R, JavaScript, Java, or other projects may use their ecosystem-native dependency definitions.

Root-level dependency automation must target real maintained environments. It must not imply that all historical Python notebooks share one supported dependency set.

## Automation ownership

GitHub Actions workflows should be scoped to the project or repository contract they validate.

Project workflows should:

- use path filters where useful;
- install only the environment owned by that project;
- avoid executing unrelated historical material;
- state clearly which outputs or invariants they protect.

Repository-wide workflows should be lightweight. Appropriate examples include:

- catalogue/schema validation;
- checks for newly introduced invalid path names;
- broken internal catalogue links;
- repository artifact-size policy checks.

A repository-wide workflow should not attempt to execute every notebook.

## Naming policy

Historical paths are preserved by default.

New paths should:

- use lowercase or otherwise consistently readable URL-safe names;
- prefer hyphens or underscores over spaces;
- never contain trailing spaces;
- never contain non-breaking spaces or other visually ambiguous whitespace;
- be descriptive enough to identify the project without opening it.

A historical rename requires a separate reviewed migration and should document compatibility consequences.

## Root-directory policy

The current root contains both historical directories and loose notebooks, figures, datasets, and utility files. These will be **classified before they are reorganized**.

The modernization programme must not bulk-move root artifacts before answering:

1. which article or project owns the artifact;
2. whether external links may target it;
3. whether it is source data, generated output, or disposable scratch material;
4. whether it should remain preserved even if it is no longer executable;
5. whether a compatibility strategy is needed.

## Data and privacy

Each maintained project owns its data boundary.

Public repositories must not rely on a vague repository-wide assumption that all committed data is safe to publish. Projects handling private, licensed, secret-backed, or externally refreshed data should define explicit rules for what may be committed.

`Blood_Pressure_Missingness` is the current reference implementation: private source rows and timestamps remain outside public artifacts while privacy-safe aggregates and reproducible public diagnostics are committed.

## Generated artifacts

Generated outputs should live with the project that owns them whenever practical.

Future repository policy will distinguish:

- source inputs;
- reproducible generated outputs worth versioning;
- large generated artifacts better stored elsewhere;
- scratch outputs that should not be committed.

Until that policy is complete, historical artifacts are preserved rather than deleted solely because they appear generated.

## Documentation hierarchy

Repository-level documentation defines cross-project rules only.

The intended hierarchy is:

```text
README.md
REPOSITORY_ROADMAP.md
docs/
  REPOSITORY_ARCHITECTURE.md
<project>/
  README.md
  ROADMAP.md              # where needed
  docs/                    # project-specific architecture/methods
```

Project documentation overrides generic repository guidance when the project has a justified local requirement.

## Reference maintained project

`Blood_Pressure_Missingness` demonstrates the strongest current engineering standard in this repository:

```text
private source
    -> validated project-local ingestion
    -> privacy-safe aggregate snapshot
    -> package-native analyses
    -> validation and sensitivity diagnostics
    -> reproducible public artifacts
    -> project-scoped CI
```

This is a **reference quality bar**, not a mandatory directory template. A small mathematical notebook does not need a package, secrets workflow, or elaborate CI merely to resemble the blood-pressure project.

The appropriate engineering depth should follow the scientific and operational risk of each project.

## Change policy

Repository modernization follows this order:

```text
inventory
    -> classify
    -> document ownership
    -> establish reproducibility claim
    -> add appropriate automation
    -> only then consider relocation or deletion
```

Structural changes that can break historical links require explicit review.

## Current architecture state

At present:

- `Blood_Pressure_Missingness` is Tier A / maintained;
- existing historical paths remain preserved;
- the root repository does not claim one common Python version or environment;
- existing GitHub Actions are project-scoped to the blood-pressure analysis;
- the remaining top-level content is considered unclassified until the catalogue/audit phase assigns evidence-based status.

The next architecture milestone is a machine-readable repository catalogue that makes these classifications visible and checkable without changing historical paths.
