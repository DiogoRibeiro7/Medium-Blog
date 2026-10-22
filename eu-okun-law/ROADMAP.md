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

- [x] Make the treatment of 2020-2021 an explicit choice for headline tables. It cannot
      be pre-registered after the fact. The headline excludes 2020-2021, a choice made
      after the full-sample estimates had been seen, and the abstract, introduction,
      appendix and limitations say so. Both samples are reported throughout. Later
      protocols (hours, asymmetry, CCE, aggregate) fixed their own treatment in advance.
- [x] Use Eurostat flags in estimation, not only in sensitivity checks. The study has
      one definition-break rule (`review.flagged_change`,
      [MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md)).
  - The Bayesian likelihood applies the rule.
  - The classical headline keeps its registered sample and is reported under the rule
    as a registered sensitivity ([docs/MEASUREMENT.md](docs/MEASUREMENT.md)): 17
    significant sums instead of 20.
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
- [x] Compare results across Eurostat vintages, as far as Eurostat allows
      ([MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md), [docs/MEASUREMENT.md](docs/MEASUREMENT.md)).
  - No historical country vintages are published, so the check uses the latest release
    (2026-10-22).
  - The labour-force series were unrevised. GDP was revised in 134 cells in 6 countries.
  - No count or classification of the headline inference changes.

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
- [x] Diagnostics of the reported results, after an external assessment
      ([BAYES_DIAGNOSTICS_PROTOCOL.md](BAYES_DIAGNOSTICS_PROTOCOL.md),
      [docs/BAYES_DIAGNOSTICS.md](docs/BAYES_DIAGNOSTICS.md)).
  - Four further seeds and exact leave-one-out for the 171 cells with high Pareto k:
    8 of the 12 registered PSIS-LOO comparisons are established.
  - Both factor models beat `primary` in every run. The amended model against `ardl`
    alone is not established.
  - Influential cells and a calibrated (halved) prior scale move the mean sums by less
    than 0.01. They add a few borderline countries to the credibly negative set.
  - The persistence fix of `ardl_dynamic_factor` is lost under the halved scale, which
    also narrows the prior of the factor's persistence.

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
- [x] Regime sensitivity, structural breaks and mechanisms, across several studies:
  - **Classical regime sensitivity and pre/post stability tests** (phase 1). None of the
    post-2021 changes survives small-sample inference
    ([docs/SMALL_SAMPLE_INFERENCE.md](docs/SMALL_SAMPLE_INFERENCE.md)).
  - **Regime-specific and drifting Bayesian coefficients** (amendment 4): no credible
    shift ([docs/BAYES_REGIMES.md](docs/BAYES_REGIMES.md)).
  - **Asymmetry between expansions and contractions:** not determined
    ([docs/ASYMMETRY.md](docs/ASYMMETRY.md)).
  - **Mechanism: job retention in the pandemic**, through hours
    ([docs/HOURS.md](docs/HOURS.md)).
  - **Re-testing with more post-2021 quarters** is the open item above.
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

- [x] Cite the motivating analysis: not cited, by the author's choice (an informal
      social-media post). The text treats its statements as hypotheses and does not
      depend on it.
- [ ] Archive the protocols with an external timestamp (for example, a DOI deposit).
      Deferred by the author's choice.
- [x] Small-sample inference for the classical headline sums, registered in
      [HEADLINE_INFERENCE_PROTOCOL.md](HEADLINE_INFERENCE_PROTOCOL.md); results in
      [docs/HEADLINE_INFERENCE.md](docs/HEADLINE_INFERENCE.md). The normal reference
      over-rejects (22.0% at a nominal 5%); under null-imposed sieve and block bootstraps
      20 of 27 sums are negative, but only 1 survives a Holm adjustment over the countries,
      against 21 with normal critical values. The registered bootstrap is conservative for
      large sums (post-hoc diagnostic).
- [x] Test measurement noise and the financial sector for Luxembourg (and Ireland, Malta),
      registered in [GDP_MEASUREMENT_PROTOCOL.md](GDP_MEASUREMENT_PROTOCOL.md); results in
      [docs/GDP_MEASUREMENT.md](docs/GDP_MEASUREMENT.md). Neither explanation is
      supported for Luxembourg. Removing finance changes nothing. Noise is not established:
      one fingerprint points towards it before adjustment, and the other points away.
      Ireland shows negatively autocorrelated quarterly growth, but so do comparison
      countries (Sweden lower).
- [ ] Ireland's modified domestic demand (Irish CSO) and annual national accounts, not yet
      used.
- [x] Checked whether `une_rt_q` inherits monthly smoothing, registered in
      [SMOOTHING_PROTOCOL.md](SMOOTHING_PROTOCOL.md); results in
      [docs/SMOOTHING.md](docs/SMOOTHING.md). The inheritance rule does not discriminate:
      it also rejects 5 of the 8 direct-estimate countries. Eurostat's `une_rt_m` metadata,
      which benchmarks the monthly series to the quarterly survey, is the better evidence.
      There is no evidence that the persistence misfit or the Dutch delay comes from
      smoothing, but with 6 countries against 8 the tests have little power. Luxembourg's
      quarterly changes are strongly negatively autocorrelated (survey noise), which makes
      its long-run "weak" result an artefact.
- [x] Hours-based Okun's law for every country with hours data (26), with and without the
      pandemic, registered in [HOURS_PROTOCOL.md](HOURS_PROTOCOL.md); results in
      [docs/HOURS.md](docs/HOURS.md). The registered tests of the job-retention pattern
      are not determined, and could not have been, because of a p-value floor of 0.067.
      An amendment registered after that result supports the pattern for hours and hours
      per worker, but not for persons.
- [x] Asymmetry between expansions and contractions, registered in
      [ASYMMETRY_PROTOCOL.md](ASYMMETRY_PROTOCOL.md); results in
      [docs/ASYMMETRY.md](docs/ASYMMETRY.md). Every point estimate has unemployment
      responding more strongly to contractions (pooled -0.10 [-0.19, +0.03]), but neither
      registered test, nor any sensitivity setting, decides it.
- [x] A common-correlated-effects mean-group panel (Pesaran 2006), registered in
      [CCE_PROTOCOL.md](CCE_PROTOCOL.md); results in [docs/CCE.md](docs/CCE.md).
      Beyond the common European cycle the mean sum is -0.17 [-0.22, -0.08], against
      -0.29 for the plain mean group. Both registered tests are supported in every
      sensitivity setting, and the result is close to the Bayesian factor model.
- [x] A labour-force-weighted EU aggregate, registered in
      [AGGREGATE_PROTOCOL.md](AGGREGATE_PROTOCOL.md); results in
      [docs/AGGREGATE.md](docs/AGGREGATE.md); new snapshot
      `snapshots/eu-aggregate-2026-10-21`.
  - The unweighted mean is -0.29, the labour-force-weighted mean -0.33, and the EU
    aggregate -0.45 [-0.57, -0.25].
  - Neither registered difference is decided; 3 of 8 sensitivity tests are.
- [ ] Gap-form Okun's law against Ball, Leigh and Loungani (2017). Needs a choice of
      potential-output measure.
- [x] Kept the participation control out of the core robustness grid, registered in
      [CORE_GRID_PROTOCOL.md](CORE_GRID_PROTOCOL.md); results in
      [docs/CORE_GRID.md](docs/CORE_GRID.md). The published verdicts are reproduced; on
      the 12 participation-free estimates 18 countries are robustly negative (17
      published), and only Romania changes. The manuscript and article use these verdicts.
- [x] External review of 2026-10-20, text and one post-hoc computation:
  - [x] Null-imposed bootstraps with errors from the unrestricted residuals, and
        false-discovery-rate counts ([docs/NULL_RESIDUALS.md](docs/NULL_RESIDUALS.md),
        not registered).
    - The normal reference's size falls from 22.0% to 16.6%, and the registered
      bootstrap still holds its size.
    - The unrestricted variant over-rejects (8.0%).
    - The country count runs from 1 (Holm, or Benjamini-Yekutieli) to 19
      (Benjamini-Hochberg).
    - The post-2021 result is unchanged.
  - [x] Luxembourg's long-run "weak" result labelled an artefact of survey noise.
  - [x] Smoothing: Eurostat metadata cited, and "no evidence that" replaces "do not drive".
  - [x] Employment protection: 6 correlations, not 18. None is decided in the primary
        variant.
  - [x] Ireland's noise prediction is not distinctive. Noise is "not established" rather
        than "fails".
  - [x] The reading of the coefficient sum in the background section.
  - [x] Smaller wording points:
    - the revision history is dropped from the article;
    - Belgium and Germany are excluded from "not short of evidence";
    - long-run and two-lag sums are kept apart;
    - Okun's GNP;
    - the association paragraph is split.
  - [x] Hand-typed counts replaced by values.
- [x] Remaining code items from that review:
  - [x] The window-sensitivity and null-residuals outputs are guarded by regeneration
        tests and listed in a post-hoc registry (`POST_HOC`, fingerprinted in
        `experiments.json`).
  - [x] Linux rebuilds no longer look stale. Manifests record numeric fingerprints, and
        raw values are rounded to 10 significant digits (magnitudes below 1e-9 set to
        zero). This was checked by regenerating on Linux (WSL).
  - [x] The country-level Newey-West in the quarterly and monthly identification
        studies counts lags in calendar periods. It is recorded in both change logs.
  - [x] The shift-share origin window is added to the window sensitivity: P1's expected
        t is 0.19 either way, and the decision is unchanged.
  - [x] Actions are pinned to commit SHAs, at the versions they already used.

### External assessment, 2026-10-22

- [x] P0, measurement: one definition-break rule, and the headline under it and on the
      latest release ([docs/MEASUREMENT.md](docs/MEASUREMENT.md)).
- [x] P0, Bayesian diagnostics: seeds, exact leave-one-out, influence and prior
      calibration ([docs/BAYES_DIAGNOSTICS.md](docs/BAYES_DIAGNOSTICS.md)).
- [x] P1, inferential claims:
  - **Origins.** Each analysis is marked as an item of the initial plan or a follow-up
    (prompted by earlier results or a review), in the pre-registration table, the
    introduction, the text where it appears and the article.
  - **Pandemic headline.** The pandemic-excluded headline is stated as a choice made
    after the full-sample estimates, which cannot be registered retrospectively. This is
    in the abstract, the introduction, the appendix and the limitations.
  - **Small-sample results.** They lead:
    - the country table shows bootstrap-t intervals and the registered small-sample
      verdicts;
    - the abstract, the introduction and the article state that none of the 14
      post-2021 changes survives.
  - **Country pairs.** The robust country differences are presented as pairwise
    statements, each at its own threshold and not a ranking: in the heterogeneity
    section, the discussion, the conclusions and the article.
- [ ] P1, provenance and independent review: an econometric review of the manuscript by
      a person, not another automated check.

Green offline CI is not evidence of a successful live data download.
No pre-selected country results, artificial observations or causal
claims without identification.
