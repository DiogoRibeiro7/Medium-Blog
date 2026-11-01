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
| Portugal all-items HICP annual average (2025=100) | Eurostat `prc_hicp_ainr`, official frozen JSON-stat | **Observed 2019–2025, source updated 17 Sep 2026** | National inflation normalization; **not** municipal inflation, income or housing burden |
| Annual municipal population | INE | 2019–2025 | Revision status and net vs gross movements |
| Fiscal-household median income | INE / Autoridade Tributária | 2019–2024 | Household concept and tax definition |
| Permits, completions, stock | INE | Existing panel partial | Lags, gross vs net stock, vacancy |
| Residential origin categories (inbound to Porto) | INE Censos 2021 via Município do Porto official open-data archive | **Observed 2021 Porto and seven parishes, referenced to 31 Dec 2019** | Full category/denominator audit; source does not publish origin municipality in this extract |
| True directed intermunicipal residential origin–destination moves | INE Census/administrative, subject to availability | **Missing** | Both origin and destination municipality, scope, privacy and timing |
| Commuting interactions (unordered pairs) | INE Census 2021, indicator 0012340, via pinned third-party workbook | **Preliminary 2021 17-municipality extract** | Exact symmetry; independent direct-INE verification still pending; not directed relocations |
| Scheduled and observed travel-time network | Official transport publisher, dated timetable data | **Missing** | Generalized cost, service frequency, reliability |
| Household expenditure / burden distribution | INE and validated surveys | **Missing** | Tenant/owner split; income and time valuation |
| Employment, interest rates and policy changes | INE, Eurostat, Banco de Portugal, municipalities | To assemble | Common-shock controls and event calendar |

All new inputs require exact source URL, indicator identifier, extraction date,
geographic identifiers, original bytes, licence/redistribution review, checksum
and missingness audit. Never fill official suppression with zero.

**Stage B checkpoint (31 October 2026).** The companion now includes a
source-hashed 2021 municipal commuting **interaction** extract and exploratory
permutation tests. The original 17×17 table is symmetric, so no commuting
direction or household moves can be reconstructed from it. The source workbook
is pinned to an external GitHub commit attributed to INE; the direct INE JSON
endpoint returned HTTP 503 at extraction and remains a verification task.
Moran's I tests on 17 endpoint growth rates are hypothesis-generating only;
the 2021 network is not a pretreatment instrument for 2019–2025 changes.

**Residential checkpoint (31 October 2026).** The official Porto municipal
open-data archive includes a fixed INE table of residential provenance from
31 December 2019 to the Census reference date (19 April 2021). We have pinned
the ZIP and member XLS hashes, extracted 11 nested geographies, independently
reconciled all seven parishes against Porto, and created transparent in-mover
rate denominators. The published category "other municipality" does not
identify the actual origin. Therefore this evidence **cannot** test Porto
residents leaving for Gaia, Maia, Valongo, or other suburbs. Nor is its single
retrospective window a longitudinal moving panel or a causal response to
2019–2025 prices. A directed residential OD source remains the priority.

**Partial-identification checkpoint (31 October 2026).** The official
2021 AMP intermunicipal destination margin is 124,618, versus 15,724 in
Porto alone, leaving 108,894 at the other 16 destinations. For the exchange
between Porto and those 16 municipalities, let \(x\) count their 2021
residents who previously lived in Porto and let \(y\) count Porto's 2021
residents who previously lived in the other 16. With no observed
origin-municipality detail, the *sharp identified set* from these two
destination margins alone is

\[
0\leq x\leq 108{,}894,\quad 0\leq y\leq 15{,}724,\quad
-15{,}724\leq x-y\leq108{,}894.
\]

The sign of within-AMP net exchange is **not identified**. The bound is a
logical constraint among **people resident in AMP in 2021**, not a frequentist
interval, a model estimate, an annual net-migration rate, all departures from
Porto or a displacement estimate. The companion's Figure 13 and machine-
readable allocations give an explicit proof by counterexample, including
opposite directions under identical observed margins. Census 2021 individual
question Q10 confirms that previous residence was asked; inspected published
INE indicator 0011631 provides migration *categories*, not municipality pairs;
indicator 0012340 is about *commuting*. INE annual municipal population-
estimation methodology 113 describes *estimated internal migration balances*,
which are also not pair-level residential flows. A published/releasable
municipality×municipality retrospective cross-tabulation remains an open
source-access question; do not substitute a modelled allocation.

**INE access checkpoint (31 October 2026).** Question 10 of the original
Census 2021 individual questionnaire explicitly requests the *municipality
and parish names* for previous residence in a different municipality. INE
documents a user service for custom statistical cross-tabulations, and an
INE–FCT–DGEEC protocol and safe-centre routes for eligible accredited
scientific research. This is a realistic *request route*, not a guarantee of
field availability, approval, price or public redistribution. The independent
April 2026 INE newsletter confirms that research safe centres are active,
including a Porto facility. The public request-form link did not load during
this work session. All access and affiliation requirements must be confirmed
before any submission. See [OD_DATA_ACCESS.md](OD_DATA_ACCESS.md) for a
reviewable request text, the 17-destination×18-origin disclosure-safe grouping,
the expected 15,724/124,618 nested reconciliation and the planned response
checks. The accompanying [residential_od_intake.py](residential_od_intake.py)
requires exact published counts or explicitly suppressed cells. Unit tests
are synthetic and cannot stand in for collected migration data.

**Annual timing / rolling-prediction checkpoint (31 October 2026).** The
fixed 2019–2025 sold-price and 2020–2025 new-contract rent medians now
support an audited Stage B first-passage calculation and an exploratory
**2023–2025 one-year-ahead** forecasting exercise. The extra predictor is
the *2021 undirected commuting-weighted* lagged neighbor growth **minus
the preceding-year metropolitan mean growth**; a two-feature baseline
already contains own and metropolitan lagged growth. Fixed
train-standardized ridge regressions are re-estimated in each rolling
origin without using held-out outcomes. In 51 municipality-year forecasts
per outcome, network augmentation changes pooled MAE from **5.257 to
5.188** log-growth points for sold-price medians, but worsens it from
**5.790 to 8.141** for new-contract rents. Alternative network weights
do not reverse the qualitative rent result. The 999 network-label shuffles
are an **informal predictive benchmark**, not calibrated randomization
inference or causal significance. All 17 new-contract rent medians crossed
+25% versus 2020 during the *published year* 2023; this annual nominal
threshold pattern cannot establish a diffusion sequence or mobility.
The OD data request remains **unsent**, and temporal association does not
establish displacement or a directed propagation parameter.

**Retrospective forecast robustness checkpoint (after the annual timing
stage).** We retain the original fixed ridge specifications and score three
unfitted benchmarks on the same held-out years: no annual growth, own
prior-year growth, and the unweighted 17-municipality *prior-year mean*.
The prior metropolitan mean yields pooled **3.855** MAE log-growth
percentage points for new-contract rents, beating both ridge baseline
(5.790) and network augmentation (8.141). All 17 municipalities have a
negative *individual mean* network contribution to rent forecasting across
the three tested years. The small sale gain (+0.069) vanishes (−0.002)
when the 2025 **evaluation year** is omitted and is slightly negative
(−0.010) when Porto and its network links are excluded **and the models
are refitted**. These deletion exercises are descriptive sensitivity
checks, not confidence intervals or treatments. Most critically, the
2023 test fold's sole training target year is 2022: its metropolitan prior
mean feature is constant across rows and collinear with the intercept.
In addition, publication-vintage availability has not been audited; a
lagged calendar year is not guaranteed to be a real-time-available data
release. The evidence strengthens the need to evaluate common metropolitan
factors before interpreting any remaining network correlation. It neither
identifies a causal common shock nor disproves household displacement.

**National consumer inflation checkpoint (1 November 2026).** An official
Eurostat Portugal annual **all-items HICP** 2019–2025 response was
retrieved at an exact five-dimension API selection (`prc_hicp_ainr`,
`A`, `INX_A_AVG`, `TOTAL`, `PT`) and frozen by SHA-256. The 2025=100
index implies **20.38%** general consumer inflation since 2019 and
**20.53%** since 2020. Deflating each municipal median by the national
index changes the observed timing diagnostic: all **17** rent medians
first passed nominal +25% relative to 2020 in **2023**, but only
**one** passed HICP-deflated +25% that year; **ten** first did so in
2024 and **six** in 2025. The unweighted median municipality new-rent
change over 2020–2025 is **+60.48% nominal** versus **+33.15%
HICP-deflated**. No real income, local cost-of-living index, fixed-
quality housing index or housing moves are measured by this conversion.
The appearance of a universal 2023 nominal threshold was partly sensitive
to the inflation convention; this neither identifies a common shock nor
disproves outward displacement. Figures 18–19, frozen input bytes and
source/derived checksum records document the result.

**Matched fiscal-income checkpoint (1 November 2026).** The same fixed
municipal extract also contains annual median gross IRS-declared income
per fiscal household. We joined that *municipal statistic* to the
median EUR/m²/month of newly signed rental contracts only in
**2020–2024** (85 matched municipality-years); 2019 new-rent medians and
2025 fiscal incomes remain missing. For each municipality and a stated
illustrative floor area, the synthetic level
\(Q=100(12A R/Y)\) is the quotient of two separate published medians,
**not an actual tenant burden**. The relative 2020–2024 quotient
increase is independent of floor area and HICP if both numerator and
denominator receive the same HICP adjustment. All **17** municipalities
show a higher 2024 than 2020 quotient; the unweighted median of
municipality relative increases is **+15.17%**, with Porto at **+30.55%**.
These values do not measure household disposable income, tenant-level
affordability, or migration. Source definitions follow the
[INE/AT local-income technical note](https://www.ine.pt/documentos/municipios/rnlocal/ano2022/1B0.pdf)
and the pinned upstream panel. Figures 20–21 and the code-generated
tables document the sensitivity to 50/70/90 m² *assumed* sizes.

## Planned analysis sequence

### Stage 0 — Obtain or document denial of directed residential counts

1. Ask INE whether an **aggregated anonymized custom tabulation** of Census
   2021 question 10 can be made available and whether a no-cost academic
   route applies. Do not submit without the researcher's approval.
2. Request 2019 origin municipality × 2021 AMP destination municipality,
   with all non-AMP Portuguese origins safely aggregated into one category.
3. Require official geographic version, universe, count status, privacy
   treatment, reference dates, citation, license, and source bytes/checksum.
4. Validate complete scope and Porto/AMP totals. If suppressed, keep residual
   uncertainty as a bound. If unavailable, maintain the original broad
   non-identification result instead of inferring moves from commuting.
5. Distinguish bilateral location change from *causally induced* housing
   displacement; an OD table does not identify move motives.

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
