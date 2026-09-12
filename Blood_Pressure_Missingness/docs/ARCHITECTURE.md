# Blood-pressure analysis architecture

The project has grown from a single notebook into a small statistical-analysis system. This document defines the canonical structure reached by the staged refactor, without changing statistical results or privacy guarantees.

## Canonical package

```text
Blood_Pressure_Missingness/
├── blood_pressure_missingness/
│   ├── public_analysis.py
│   ├── analyses/
│   │   ├── observation_process.py
│   │   ├── gap_aware.py
│   │   ├── day_influence.py
│   │   ├── episode_observation.py
│   │   ├── episode_time_form.py
│   │   ├── temporal_dependence.py
│   │   ├── temperature.py
│   │   └── temperature_sensitivity.py
│   ├── data_sources/
│   │   └── google_sheets.py
│   └── validation/
│       ├── current_influence.py
│       └── current_narrative.py
├── data/
├── figures/
├── tests/
└── blood-pressure-missingness.ipynb
```

## Responsibility boundaries

### Public analysis

`public_analysis.py` owns the privacy-safe day-indexed snapshot model, public validation, descriptive statistics, primary HC3 trends, state-space analysis, and public figure generation. It must not require private credentials or raw row-level health records.

### Analyses

`analyses/` contains sensitivity analyses and diagnostics. These modules may depend on the public-analysis layer and, where mathematically justified, on another analysis module. They should not own authentication or secret management.

### Data sources

`data_sources/` owns private ingestion and source-specific validation. Raw Google Sheet rows and timestamps stay in memory and are transformed into privacy-safe outputs before anything is written.

### Validation

`validation/` contains snapshot-sensitive scientific and narrative gates. These are refresh/release safeguards rather than estimators.

### Notebook and reporting

The notebook is a narrative surface over tested modules. It must not contain an independent implementation of the statistics.

## Dependency direction

The intended dependency direction is:

```text
private data source
      │
      ▼
privacy-safe artifacts ──► public analysis ──► sensitivity analyses
                                  │                    │
                                  └──────────┬─────────┘
                                             ▼
                                      validation gates
                                             │
                                             ▼
                                      notebook/reporting
```

Temperature analysis is a deliberate secret-backed exception: it consumes private timestamps in memory, joins them to external weather, and publishes only aggregate diagnostics.

## Refactor stages

### Stage 1 — package boundary

Canonical package paths were introduced while retaining the historical top-level modules temporarily.

### Stage 2 — invert the compatibility layer

Implementations were moved into the package, historical cross-module imports were replaced with package imports, and `sys.path` manipulation was removed from supported tests.

### Stage 3 — preserve numerical contracts

Privacy, output-schema, and deterministic-reproduction contracts were added so structural changes could be separated from scientific changes.

### Stage 4 — canonical supported interfaces

The notebook, workflows, reproduction commands, validation gates, and regression tests now use the installed canonical package directly. The historical root forwarding shims and temporary compatibility helper have been removed because no supported surface depends on them.

## Non-negotiable refactor rule

A structural refactor must not silently change:

- an estimating equation;
- a weighting rule;
- a robust-covariance choice;
- a privacy boundary;
- a published JSON schema;
- a narrative conclusion tied to the current snapshot.

Any such change belongs in a separate scientific PR after the architecture refactor is green.
