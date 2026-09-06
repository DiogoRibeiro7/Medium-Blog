# Blood-pressure missingness roadmap

## Status

**Phase 1 is complete.**

The current analysis has reached the point where additional variants of the same regressions are unlikely to add proportionate scientific value. The project now has a coherent, privacy-preserving analysis of an irregular longitudinal tracker, together with live Google Sheet refreshes, executable documentation, and CI gates that protect both numerical and narrative consistency.

The current programme already covers:

- privacy-safe day-indexed aggregation from the private source;
- source-quality validation and sessionisation;
- equal-day HC3 descriptive trends;
- sensitivity to unequal observation intensity;
- gap-aware decomposition into within- and between-episode components;
- single-observed-day influence and leave-one-day-out robustness;
- episode-level sensitivity to sampling intensity;
- episode-level sensitivity to alternative within-episode time forms;
- state-space visualisation over the full calendar grid;
- calendar-gap-aware residual-dependence diagnostics;
- automatic live-source refreshes without committing raw health rows;
- scientific and narrative drift gates in CI;
- executable Medium-facing notebook validation.

The current findings should therefore be treated as a **completed descriptive case study for this snapshot**, not as an invitation to keep adding specifications until a preferred result appears.

## Reopen criteria

Substantive statistical work should resume only when at least one of the following changes materially:

1. **The observed-day sample grows meaningfully.** A handful of extra measurements on already dense days is much less informative than additional independently observed calendar days.
2. **The observation design changes prospectively.** More regular measurement opportunities, clearer sampling rules, or consistently recorded context would support questions that the retrospective tracker cannot identify.
3. **New information becomes available inside a previously unobserved region.** Without observations inside the long gap, its transition timing remains unidentified.
4. **A genuinely new estimand is proposed.** New work should answer a different scientific question rather than merely swap one robust covariance estimator, polynomial degree, or weighting rule for another.
5. **A second comparable series becomes available.** Replication across another period or independently designed series would justify hierarchical or cross-series work.

A routine Google Sheet refresh that only adds a small number of readings should update the public aggregates and narrative through the existing workflow, but does not by itself justify a new modelling phase.

## Future work

### 1. Prospective observation design

Highest priority if the tracker continues.

Define a prospective measurement protocol before analysing future data. The statistical goal is to make observation opportunities more interpretable, not to maximise the raw number of readings.

Useful design improvements include:

- pre-specified measurement windows or opportunities;
- explicit recording of why an expected observation was missed;
- consistent contextual fields when they are part of the research question;
- separation between scheduled measurements and symptom- or event-triggered measurements;
- stable rules for repeated readings and session boundaries.

This would make the observation process itself estimable rather than forcing it to remain an unidentified nuisance.

### 2. Joint modelling of outcome and observation process

Only after enough prospective information exists.

The current analysis can show that sampling intensity is associated with observed pressure, but it cannot identify the missingness mechanism or true observation probabilities. A later phase could model measurement events and outcomes jointly, for example through a recurrent-event or point-process observation model coupled to the longitudinal outcome.

This should not be attempted from the current aggregate snapshot because the relevant row-level timing and observation-process information is deliberately excluded from the public data.

### 3. Irregular-time stochastic dependence

The current residual diagnostics establish that row-order lags are not valid calendar-time lags. With a substantially larger number of observed days, the next step could move from descriptive exact-lag correlations to an explicitly irregular-time dependence model.

Candidate directions include:

- continuous-time autoregressive or Ornstein-Uhlenbeck-type residual processes;
- Gaussian-process residual covariance indexed by actual elapsed time;
- state-space models whose transition covariance depends on calendar-time distance.

Any such model should preserve the large gaps rather than collapsing the observed subset into an equally spaced series.

### 4. Richer state-space comparison

The current local-linear-trend model is intentionally illustrative. With more data, future work could compare a small set of pre-specified state-space structures using predictive diagnostics rather than selecting whichever model produces the most attractive trajectory.

Possible extensions include stochastic level-only, local-linear-trend, and continuous-time variants. Missing calendar days must remain missing observations in every specification.

### 5. Prospective episode-transition analysis

The current 32-day gap prevents identification of when the pre/post level difference arose. A future design with denser observations around transitions could support questions about gradual versus abrupt changes.

This would then justify formal transition or change-point models. Applying such methods retrospectively to an interval with no observations would still be inappropriate.

### 6. Session-level and within-session modelling

If private row-level processing remains available, a separate private analysis could study within-session repeatability and measurement variance while publishing only aggregate outputs.

Potential questions include:

- how much variation occurs within versus between sessions;
- whether the first reading in a session differs systematically from later readings;
- how session-level uncertainty should propagate into daily summaries.

This should remain separate from the public day-level snapshot unless a privacy-safe sufficient-statistic representation is designed first.

### 7. Replication rather than specification expansion

If the methodological article is extended, replication should take priority over adding further sensitivity models to the same 26 observed days.

Useful replication targets would be another independently observed time window or another dataset with a similar irregular-observation problem. The purpose would be to test whether the methodological lessons generalise, not whether the same numerical blood-pressure result repeats.

### 8. Publication and reproducibility polish

Low statistical risk and suitable as maintenance work:

- keep the notebook and README aligned with refreshed public aggregates;
- preserve the live-source privacy boundary;
- keep generated diagnostic JSON files reproducible from the public snapshot;
- add archival release tags for article versions if the Medium post is published or materially revised;
- optionally add a compact methods diagram showing source → privacy-safe aggregation → diagnostics → article.

## Work explicitly deferred

The following are not priorities for the current sample:

- adding higher-degree polynomials or splines simply because they are available;
- naïve interpolation across the 32-day missing interval;
- ordinary row-order Newey-West/HAC covariance on the observed subset;
- selecting a preferred trajectory by trying many functional forms after seeing the results;
- declaring MCAR, MAR, or MNAR from the observed tracker alone;
- interpreting the episode contrast as evidence of a change occurring at a known time inside the gap;
- causal attribution of the observed level difference;
- treating more readings on the same observed day as equivalent to more independently observed days;
- medical or clinical conclusions from this statistical case study.

## Maintenance rule

Until one of the reopen criteria is met, the project should stay in **maintenance mode**:

1. refresh from the private Google Sheet when needed;
2. review any changed privacy-safe aggregate PR;
3. update the public narrative when the drift gate requires it;
4. fix reproducibility, privacy, or correctness issues;
5. otherwise leave the statistical model set unchanged.

That boundary is intentional. The next meaningful gain should come from **better information or a genuinely new question**, not from another estimator applied to the same small irregular sample.
