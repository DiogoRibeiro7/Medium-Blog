# Research protocol: does affordability pressure propagate outward from Porto?

**Status:** proposed next-stage empirical programme, **not preregistered**
and **not yet a causal study**. Frozen descriptive baseline: 31 October 2026.

## Questions and estimands

The first empirical object is *new-contract rental-price pressure* in destination
municipality \(j\) following a change in housing affordability in an origin
municipality \(i\). A secondary object is residential relocation \(M_{ij,t}\)
from \(i\) to \(j\), if a credible and privacy-safe origin–destination data
source can be obtained. A third outcome is the generalized burden
\(C_{hj}=R_{hj}+T_{hj}+v_h\tau_{hj}\) for household type \(h\).

Do **not** conflate these outcomes. A destination price increase without
identified origin–destination moves does not estimate displacement.

### Hypotheses

1. **H1, temporal spillovers:** a plausibly exogenous shock to origin
   affordability changes subsequent destination housing demand and/or prices.
2. **H2, transport-network mediation:** effects are more concentrated among
   accessible substitutes than among geographically nearby but less-connected
   alternatives.
3. **H3, supply moderation:** greater suitable vacant capacity or housing
   supply responsiveness reduces downstream price pressure and onward movement.
4. **H4, total burden:** a subset of households moving outward loses enough
   time or pays enough transport costs that lower rent does not improve its
   generalized economic position.

These are hypotheses, not findings.

## Geographic definitions

- Primary descriptive administrative sample: the 17 municipalities legally
  comprising the Porto Metropolitan Area (AMP).
- Required sensitivity 1: official commuting-based functional urban area,
  if a reproducible crosswalk and observed matrix are available.
- Required sensitivity 2: distance and public-transport accessibility matrices
  constructed from dated, versioned route data.
- Required sensitivity 3: finer freguesia-level geography wherever compatible
  rent, price and denominator data are published.

The AMP administrative perimeter is a governance unit; it is not proof of
residential substitutability. Do not infer residence changes from Census
commuting-to-work flows.

## Source ledger to assemble

| Quantity | Preferred publisher | First-release availability | Next-stage check |
| --- | --- | --- | --- |
| Median sold-price EUR/m² | INE, via pinned housing project | 2019–2025 | Transaction mix and geographic methodology |
| Median new-contract rent EUR/m² | INE, via pinned housing project | 2020–2025 | Contract counts, suppression and vintage |
| Annual municipal population | INE | 2019–2025 | Revision status and net vs gross movements |
| Fiscal-household median income | INE / Autoridade Tributária | 2019–2024 | Household concept and tax definition |
| Permits, completions, stock | INE | Existing panel partial | Lags, gross vs net stock, vacancy |
| True residential origin–destination moves | INE Census/administrative, subject to availability | **Missing** | Definition, coverage, privacy and frequency |
| Commuting origin–destination | INE Census or transport authority | **Missing** | Not a substitute for residential moves |
| Scheduled and observed travel-time network | Official transport publisher, dated timetable data | **Missing** | Generalized cost, service frequency, reliability |
| Household expenditure / burden distribution | INE and validated surveys | **Missing** | Tenant/owner split; income and time valuation |
| Employment, interest rates and policy changes | INE, Eurostat, Banco de Portugal, municipalities | To assemble | Common-shock controls and event calendar |

All new inputs require exact source URL, indicator identifier, extraction date,
geographic identifiers, original bytes, licence/redistribution review, checksum
and missingness audit. Never fill official suppression with zero.

## Planned analysis sequence

### Stage A — Source and descriptive validation

1. Confirm the published municipal series against original INE data and
   archived snapshots; reconcile 17 names to stable municipal codes.
2. Document each definition and series break. Do not stitch different
   methodologies into a continuous index without a defensible bridge.
3. Compare levels and changes in sale prices, new-contract rents,
   population, permits, completions and available income measures.
4. Report unweighted and justified weighted summaries separately.
5. Compare nominal outcomes with an explicitly cited deflator.

### Stage B — Spatial and timing diagnostics

1. Construct a defensible geographical adjacency matrix, a travel-time
   matrix and, if available, separate observed commuting and moving matrices.
2. Plot local and neighbouring changes with uncertainty and missingness.
3. Explore contemporaneous and lead–lag associations at resolutions the
   underlying data support. Do not fit many annual lags to seven years.
4. Compare local-only forecasts with network-augmented forecasts using
   rolling-origin validation and a simple benchmark.
5. Assess sensitivity to the definition of the housing market and to
   peripheral municipalities that are poor substitutes for Porto.

### Stage C — Identification, only if a suitable shock exists

1. Specify the treatment, exposure, population at risk and estimand before
   selecting a model.
2. Identify a plausibly exogenous local policy or demand/supply shock, then
   explain why it is not simply responding to expected price appreciation.
3. Construct comparison units and inspect pretreatment trajectories.
4. Use event-study, differences-in-differences, synthetic-control or
   instrumental-variable methods *only when their identification assumptions
   can be defended*.
5. Report spillovers to comparison units, inference under spatial/serial
   dependence, anticipation and the limits of external validity.

If no credible shock and mobility data are found, conclude with
**descriptive and predictive evidence only**, not a causal effect estimate.

### Stage D — Household welfare and policy scenarios

1. Define income-specific and household-type-specific generalized costs.
2. Use realistic transport service and time values with ranges, not a
   unique convenient coefficient.
3. Model supply and accessible affordable-housing alternatives.
4. Evaluate tradeoffs in rents, residential stability, transport burden,
   access to employment, construction timing and public expenditure.
5. Disclose all parameters fitted from observations versus assumed for
   counterfactual scenarios.

## Failure and falsification conditions

- No directional origin–destination movement: the proposed relocation channel
  is unsupported by those data.
- Price co-movement disappears after appropriate common-shock controls:
  a simple spillover interpretation weakens.
- Results depend on arbitrary spatial weights or a hand-picked subset:
  report fragility rather than a selected specification.
- No differential pretrend, no plausible exogenous shock, or serious
  anticipation: do not report a causal treatment effect.
- High capacity and little onward search: cascade attenuates; the proposed
  universal mechanism is rejected.
- Net generalized-cost savings for a household group: acknowledge that
  relocation can improve its welfare.

## Reporting standard

Every table and figure must identify: primary publisher, statistic and units,
geography, period, nominal/real status, sample and missingness, evidence class
(observed / derived / hypothetical / estimated), and uncertainty when relevant.

The article and code must be updated together. Do not turn a diagnostic
correlation, model illustration or hypothetical displacement count into a
Portuguese fact. Preserve the exact source commit and dataset SHA-256 for
reproducing the published version.
