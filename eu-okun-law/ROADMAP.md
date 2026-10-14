# Research roadmap

## Phase 0 — Reproducible foundation (this PR)

- [x] Define research questions, units and separate estimands.
- [x] Select official quarterly Eurostat series and fixed EU-27 cohort.
- [x] Add project-local environment and offline quality tests.
- [x] Download/parse JSON-stat, preserve missing cells and flags.
- [x] Write provenance manifest and country-quarter CSV.
- [ ] Run a real live Eurostat download; review coverage and flags.

## Phase 1 — Baselines and data audit

- [x] Country-level coverage and quality-flag JSON audit (offline; live run pending).
- [ ] Missingness visualization and review of survey breaks.
- [ ] Validate labour-force survey comparability across 2021.
- [x] Implement distributed-lag OLS with HAC uncertainty; lags 0, 1, 2, 4 (real-data execution pending).
- [x] Implement paired specifications with/without participation (real-data comparison pending).
- [ ] Time-ordered out-of-sample forecasting against persistence.
- [ ] Document preliminary associations without causal claims.

## Phase 2 — Hierarchical Bayes

- [ ] Country partial pooling, prior sensitivity and common shocks.
- [ ] Posterior predictive/convergence checks and robust errors.
- [ ] Joint posterior rank probabilities, not point ranks alone.
- [ ] Compare predictive performance with classical baselines.

## Phase 3 — Structural tests

- [ ] Formal shock identification, with demand/supply distinctions.
- [ ] Impulse responses with stated units and assumptions.
- [ ] Regime sensitivity, structural breaks and mechanisms.
- [ ] Detailed alternative hypotheses for NL and LU.

## Phase 4 — Medium article

- [ ] Freeze data, environment and all model settings.
- [ ] Generate every chart and table directly from scripts.
- [ ] Report support, contradiction and uncertainty impartially.
- [ ] Write and review the mathematical article before publication.

Green offline CI is not evidence of a successful live data download.
No pre-selected country results, artificial observations or causal
claims without identification.
