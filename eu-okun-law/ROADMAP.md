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
- [x] Hours- or employment-based measures, including cross-border workers for Luxembourg
      ([docs/HETEROGENEITY_CASE_STUDY.md](docs/HETEROGENEITY_CASE_STUDY.md)). In Luxembourg, domestic employment (including commuters) and resident
      employment are equally flat; output per hour absorbs the GDP fluctuations.
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
- [x] Detailed alternative hypotheses for NL and LU, and PT against ES. These were
      pre-registered in [HETEROGENEITY_PROTOCOL.md](HETEROGENEITY_PROTOCOL.md), with
      results in [docs/HETEROGENEITY_CASE_STUDY.md](docs/HETEROGENEITY_CASE_STUDY.md). One of 14 predictions is supported: the Netherlands absorbs more
      through output per hour than Spain. The Luxembourg measurement mismatch is not
      supported.

## Phase 4 — Medium article

- [x] Freeze data, environment and all model settings. The manifests in
      [publication/manifest/](publication/manifest/) list every snapshot file, every
      analysis (protocol, commands, seeds, replication, output hashes), the locked
      environment and every quoted value with its source.
- [x] Generate every chart and table directly from scripts (`eu-okun publication`). The
      two hand-written tables (data sources and the comparison with the earlier claims)
      take every number from the same generated values.
- [x] Audit all prior outputs: snapshot re-derivation, consistency between reports, and
      the numbers quoted in this roadmap, the README and the identification study
      ([publication/AUDIT.md](publication/AUDIT.md)). The audit corrected two quoted
      ranges.
- [x] Report support, contradiction and uncertainty impartially, keeping association,
      prediction and causation apart.
- [x] Write the mathematical article in LaTeX
      ([publication/manuscript/](publication/manuscript/)) and the Medium version generated
      from the same values ([publication/medium/article.md](publication/medium/article.md)).
      The bibliography is verified against doi.org.
- [ ] Author review and publication. Nothing is published automatically.

### External review, 2026-10-19

Done:

- [x] Report the credibly-negative count as 22 or 15 depending on the unpooled fit at the
      R-hat threshold, and correct the explanation (OLS excludes zero for five of the
      seven countries in between).
- [x] Quote the range of the average sum across accepted fits and the dynamic model's
      value; state the estimand (beyond the common European cycle) in the abstract.
- [x] Correct the forecasting conclusions: small decided gains without the pandemic
      targets; the one conditional gain is fragile; nothing beats a no-change forecast.
- [x] Remove internal contradictions in the article; rewrite the Medium version in prose;
      plain meaning of the magnitude.
- [x] Reframe pre-registration as sequential, stage by stage, on the same data; reframe
      identification as uninformative designs; add GDP measurement error as an untested
      rival explanation for Luxembourg; add the Holm count and the classical-inference
      caveats.
- [x] Post-hoc pandemic-window sensitivity ([docs/WINDOW_SENSITIVITY.md](docs/WINDOW_SENSITIVITY.md)).
- [x] Rename the SVAR variance share (it is of the unemployment change).
- [x] Guard hand-typed integers and percentages in the article; freeze the CI outputs that
      the classical review compares against; check in CI that the review reproduces.

Open:

- [ ] Cite the motivating analysis, or state precisely where and when it appeared
      (needs the author).
- [ ] Archive the protocols with an external timestamp (for example, a DOI deposit).
- [x] Small-sample inference for the classical headline sums, registered in
      [HEADLINE_INFERENCE_PROTOCOL.md](HEADLINE_INFERENCE_PROTOCOL.md); results in
      [docs/HEADLINE_INFERENCE.md](docs/HEADLINE_INFERENCE.md). The normal reference
      over-rejects (22.0% at a nominal 5%); under null-imposed sieve and block bootstraps
      20 of 27 sums are negative, but only 1 survives a Holm adjustment over the countries,
      against 21 with normal critical values. The registered bootstrap is conservative for
      large sums (post-hoc diagnostic).
- [x] Test measurement noise and the financial sector for Luxembourg (and Ireland, Malta),
      registered in [GDP_MEASUREMENT_PROTOCOL.md](GDP_MEASUREMENT_PROTOCOL.md); results in
      [docs/GDP_MEASUREMENT.md](docs/GDP_MEASUREMENT.md). Both explanations fail for
      Luxembourg: its hours-output correlation does not rise year on year, and removing
      finance changes nothing. Ireland shows negatively autocorrelated quarterly growth.
- [ ] Ireland's modified domestic demand (Irish CSO) and annual national accounts, not yet
      used.
- [x] Checked whether `une_rt_q` inherits monthly smoothing, registered in
      [SMOOTHING_PROTOCOL.md](SMOOTHING_PROTOCOL.md); results in
      [docs/SMOOTHING.md](docs/SMOOTHING.md). None of the six countries with smoothed
      monthly series has a quarterly rate that is the average of its monthly rate; the
      persistence misfit and the Dutch delay are not concentrated in them. Luxembourg's
      quarterly changes are strongly negatively autocorrelated (survey noise).
- [ ] Gap-form Okun's law against Ball, Leigh and Loungani (2017); asymmetry; hours-based
      Okun's law for every country; a common-correlated-effects mean-group panel; a
      labour-force-weighted EU aggregate.
- [x] Kept the participation control out of the core robustness grid, registered in
      [CORE_GRID_PROTOCOL.md](CORE_GRID_PROTOCOL.md); results in
      [docs/CORE_GRID.md](docs/CORE_GRID.md). The published verdicts are reproduced; on
      the 12 participation-free estimates 18 countries are robustly negative (17
      published), and only Romania changes. The manuscript and article use these verdicts.

Green offline CI is not evidence of a successful live data download.
No pre-selected country results, artificial observations or causal
claims without identification.
