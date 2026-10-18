# Research roadmap

## Phase 0 — Reproducible foundation

- [x] Define research questions, units and separate estimands.
- [x] Select official quarterly Eurostat series and fixed EU-27 cohort.
- [x] Add project-local environment and offline quality tests.
- [x] Download/parse JSON-stat, preserve missing cells and flags.
- [x] Write provenance manifest and country-quarter CSV.
- [x] Run a real live Eurostat download; review coverage and flags (CI workflows;
      reviewed in [docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md), section 2).

## Phase 1 — Baselines and data audit

All classical results below were verified against an independent implementation on
the frozen snapshot `snapshots/eurostat-2026-10-15` (review, section 3).

- [x] Country-level coverage and quality-flag JSON audit.
- [x] Missingness visualization and review of survey breaks (review, section 2).
- [x] Review labour-force survey comparability across 2021: break-corrected back series
      from 2009; Spain and France flagged "definition differs" from 2021-Q1; Netherlands
      phase-in during 2021 (review, section 2.5).
- [x] Implement distributed-lag OLS with HAC uncertainty; lags 0, 1, 2, 4.
- [x] Implement paired specifications with/without participation.
- [x] Implement expanding-window pseudo-out-of-sample forecasting against persistence.
- [x] Implement common-sample cumulative coefficient HAC sensitivity and report.
- [x] Implement dependence-aware paired country block bootstrap.
- [x] Implement formal pre/post HAC interaction stability tests.
- [x] Implement economic-regime sensitivity with sample-size diagnostics.
- [x] Independent verification and sensitivity audit of the classical results.
- [x] Document preliminary associations without causal claims
      ([docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md)).

Open items raised by the review:

- [ ] Make the treatment of 2020-2021 an explicit, pre-registered choice for headline
      tables (the published comparison includes it and is dominated by it).
- [ ] Use Eurostat flags in estimation, not only in sensitivity checks (ES/FR from 2021,
      DE 2020, series breaks).
- [x] Small-sample-valid inference for the post-2021 window: null-imposed sieve and
      block bootstraps of the HAC t-statistic and a size study
      ([docs/SMALL_SAMPLE_INFERENCE.md](docs/SMALL_SAMPLE_INFERENCE.md)). None of the 14
      published Holm-significant changes survives; the normal reference rejects 22.8% at
      a nominal 5%.
- [ ] Re-test the post-2021 stability once more quarters are available (needs a new
      snapshot).
- [ ] Dependence-aware tests of forecast accuracy (paired loss differentials). Done for
      the conditional predictive comparison (panel Diebold-Mariano on cross-country mean
      differentials); the t-1 forecasts of `eu-okun forecast` still use approximate
      country-level ratios.
- [ ] Hours- or employment-based measures, including cross-border workers for Luxembourg.
- [ ] Compare results across Eurostat vintages.

## Phase 2 — Hierarchical Bayes

Pre-registered in [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md); results in
[docs/HIERARCHICAL_BAYES.md](docs/HIERARCHICAL_BAYES.md).

- [x] Country partial pooling, prior sensitivity and common shocks (nine fixed
      specifications; eight pass the convergence criteria, `no_pooling` fails).
- [x] Posterior predictive/convergence checks and robust (Student-t) errors.
- [x] Joint posterior rank probabilities, not point ranks alone.
- [x] Additive shared quarter effects with and without participation, prior predictive
      checks, posterior recovery on synthetic data and ArviZ NetCDF posterior artifacts
      (amendment 1; [docs/BAYES_MODEL_CHECKS.md](docs/BAYES_MODEL_CHECKS.md)).
- [x] Robustness of country differences and ranks across specifications, including
      the lag order K = 1 and the focus countries against the classical bootstrap
      (amendment 2; [docs/BAYES_ROBUSTNESS.md](docs/BAYES_ROBUSTNESS.md)).
- [x] One-quarter-ahead forecasts of hierarchical Bayesian forecasting models against the
      PR #149 classical models (nothing dated t; retrospective, not real-time)
      ([docs/FORECAST_COMPARISON.md](docs/FORECAST_COMPARISON.md)).
- [x] Compare predictive performance with classical baselines (expanding-window refits):
      pre-registered in [PREDICTIVE_PROTOCOL.md](PREDICTIVE_PROTOCOL.md), results in
      [docs/PREDICTIVE_COMPARISON.md](docs/PREDICTIVE_COMPARISON.md). Conditional on the
      quarter's GDP growth; not a real-time forecast.
- [x] Address the misfits the checks found, under a new registered protocol (amendment
      3): an AR(1) common factor and lagged unemployment changes (ARDL). The combination
      fixes the persistence misfit in a local and a Linux run; the AR(1) factor alone sits
      on the threshold. Both factor specifications are better supported than `primary` by
      PSIS-LOO ([docs/BAYES_DYNAMICS.md](docs/BAYES_DYNAMICS.md)). The factor stays
      correlated with average GDP growth.
- [x] Time-varying or regime-specific coefficients (amendment 4): post-2021 shifts of
      the coefficient sums, partially pooled, and a common random walk. No credible
      average or country shift. The drift is better supported than a fixed coefficient by
      PSIS-LOO, but its period contrasts are not credible
      ([docs/BAYES_REGIMES.md](docs/BAYES_REGIMES.md)).

## Phase 3 — Structural tests

- [x] Identification study: the problem, a causal graph, recursive / long-run / sign /
      external-instrument strategies, testable assumptions and public data
      ([docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md)). Productivity,
      labour-supply and country-specific GDP shocks are not identifiable.
- [x] Impulse responses with stated units and assumptions for the two externally
      identified common shocks (ECB monetary policy, OPEC oil supply news); the
      shock-conditional Okun coefficients are uninformative
      ([docs/IDENTIFICATION_RESULTS.md](docs/IDENTIFICATION_RESULTS.md)).
- [x] Monthly outcomes (unemployment, industrial production) for more power, under a new
      registered protocol ([MONTHLY_IDENTIFICATION_PROTOCOL.md](MONTHLY_IDENTIFICATION_PROTOCOL.md);
      [docs/MONTHLY_IDENTIFICATION_RESULTS.md](docs/MONTHLY_IDENTIFICATION_RESULTS.md)).
      Direct monthly survey estimates exist for only 8 countries. The first stages are no
      stronger than the quarterly ones, and every primary ratio set is unbounded.
- [x] A shift-share foreign-demand instrument for country-specific GDP shocks, under a
      pre-registered feasibility rule ([SHIFT_SHARE_PROTOCOL.md](SHIFT_SHARE_PROTOCOL.md);
      [docs/SHIFT_SHARE_FEASIBILITY.md](docs/SHIFT_SHARE_FEASIBILITY.md)). Only distant,
      non-European demand is defensible. After country and quarter effects its expected
      first-stage t is 0.19 against 3.16: no go, so no IV is estimated.
- [x] Structural VAR sign restrictions for demand, technology and labour-supply shocks,
      with robust Bayesian identified sets and the recursive orderings as a non-causal
      sensitivity measure ([SIGN_RESTRICTIONS_PROTOCOL.md](SIGN_RESTRICTIONS_PROTOCOL.md);
      [docs/SIGN_RESTRICTIONS_RESULTS.md](docs/SIGN_RESTRICTIONS_RESULTS.md)). The
      demand and technology Okun-ratio sets are unbounded, and labour supply and demand
      differ only at impact, by assumption.
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
