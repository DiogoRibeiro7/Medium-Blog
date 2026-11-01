# The Suburb Is Not an Escape — reproducible Greater Porto housing analysis

Companion project for the long-form Medium manuscript
**[The Suburb Is Not an Escape: How Housing Unaffordability Can Spread Across a Metropolitan Area](article.md)**.

This is a **descriptive study and methodological foundation**, not a causal
estimate of residential displacement. It combines a fixed official-data-derived
municipality panel with two separately classified hypothetical models.

## Scope

- Geography: all **17 legally defined municipalities of the Porto Metropolitan
  Area (AMP)**, as listed by the [AMP](https://www.amp.pt/en/amp-instituicao/area-metropolitana-do-porto-or-lei-no-752013/).
- Observed grid: **17 municipalities × 7 years (2019–2025) = 119 rows**.
- Median sold-dwelling price (EUR/m²): 2019–2025.
- Median new-contract housing rent (EUR/m²/month): 2020–2025.
- Resident population: 2019–2025.
- Median fiscal-household annual gross income: 2019–2024.
- Building completions: retained in the input, **not** used to claim an estimated
  supply elasticity.
- Hypothetical: household absorption by three abstract tiers; illustrative
  generalized commuting and housing cost by distance.
- General-inflation comparison: frozen official **Portugal annual average
  all-items HICP** (2019–2025, 2025=100), used as a *national deflator* of
  published municipal medians, **not** an income-based housing-burden measure.
- Matched-year fiscal-income screening proxy: 2020–2024 municipality new-
  contract rent medians / median annual *gross declared IRS fiscal-household*
  income, at explicitly hypothetical 50/70/90 m² dwelling sizes. This is a
  **ratio of two group medians, not the rent burden of actual households**.

The intended next stage is an empirical study of actual residential mobility,
spatial connectivity, housing stock, and transport burden. It is described in
[the research protocol](RESEARCH_PROTOCOL.md). A follow-up companion now adds
**preliminary Census-derived commuting connectivity** and exploratory spatial
association diagnostics (below). There are still **no observed residential
origin–destination household flows**, estimated propagation coefficients, or
causal treatment effects in this project.

## Reproduce

Requirements: Python 3.12 and the pinned dependencies.

    cd urban-affordability-cascade
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py

To avoid modifying tracked outputs during a check:

    python analysis.py --output /tmp/porto-housing-reproduction

The script writes eight original PNG figures into figures/, three CSV tables
into results/, and a JSON provenance summary into results/.
It uses no API requests, random numbers, credentials, or untracked inputs.
The executable notebook model_walkthrough.ipynb offers a complementary walkthrough.

## Fixed observational extract and provenance

| Field | Recorded identity |
| --- | --- |
| Upstream project | [DiogoRibeiro7/portugal-housing-generations](https://github.com/DiogoRibeiro7/portugal-housing-generations) |
| Upstream file | results/integrated_panel.csv |
| Pinned upstream commit | 0407faed1f08209b3d7682e19fc57e7120e62879 |
| Original upstream Git blob | 9bade430163315e7c5da323251daa0715e1ffede |
| This project's fixed subset | data/amp_municipal_panel_2019_2025.csv |
| SHA-256 of fixed subset | 88502e15b55a4d46536eed42ddc1924d30f555498d2c4a7e7a1b41fbeed71c66 |
| Freeze date | 31 October 2026 |
| Extraction rule | Keep the 17 officially defined AMP names and years 2019–2025; preserve all 18 upstream columns |

The upstream project records the INE source identifiers, archived snapshots,
metadata-vintage boundaries, missingness and methodology. The local subset
contains transformed official statistical data, **not** original respondent- or
transaction-level records. Read the upstream [source contracts](https://github.com/DiogoRibeiro7/portugal-housing-generations/tree/0407faed1f08209b3d7682e19fc57e7120e62879/data)
before extending the panel. In particular, sale medians are not constant-quality
house-price indices, new-lease rents do not describe incumbent contracts, and
resident counts are not observed migration flows.

Missing 2019 new-lease rents and missing 2025 fiscal-household income are
retained as missing. They are never extrapolated, interpolated or converted to
zero. The analysis's exact-period comparisons enforce those distinctions.

## Figure evidence ledger

| Figure | File | Interpretation | Evidence class |
| --- | --- | --- | --- |
| 1 | fig1_sale_price_growth_2019_2025.png | Growth in each municipality's median dwelling-sale price, 2019–2025 | Observed, derived |
| 2 | fig2_new_lease_rent_growth_2020_2025.png | Growth in median new-contract rent, 2020–2025 | Observed, derived |
| 3 | fig3_sale_price_index_2019_2025.png | Each municipality's sale-price series rebased to 2019=100 | Observed, derived |
| 4 | fig4_illustrative_value_income_ratio.png | 100 m² hypothetical purchase value / median annual fiscal income, 2019 vs 2024 | Derived proxy; **not** mortgage affordability |
| 5 | fig5_new_lease_rent_trajectories.png | Annual new-contract rents in six selected municipalities | Observed, derived |
| 6 | fig6_population_vs_price_growth.png | Municipal population vs median sold-dwelling price changes | Observed, descriptive association |
| 7 | fig7_hypothetical_cascade_absorption.png | Different assumed fractions of households absorbed in three abstract tiers | **Hypothetical**, no fitted relocation |
| 8 | fig8_hypothetical_generalized_cost.png | Assumed rent savings vs monetary travel and value of commuting time | **Hypothetical**, no measured Porto journey |
| 9 | fig9_2021_commuting_interactions.png | Undirected work/study links from 2021, not changing residence | Observed interaction pairs; third-party extract |
| 10 | fig10_commuting_moran_null.png | Null permutation distributions for exploratory spatial association | Descriptive, conditional on fixed 2021 network |
| 11 | fig11_parish_intermunicipal_in_mobility.png | Porto parish shares with 2019 residence in another municipality | Observed destination margins, not OD flows |
| 12 | fig12_parish_residential_mover_mix.png | Previous-residence categories among 2021 parish movers | Observed composition, not OD flows |
| 13 | fig13_residential_od_identification_set.png | All directed Porto↔rest-AMP allocations allowed by observed margins | **Sharp logical bounds and hypothetical feasible worlds; not observed OD** |
| 14 | fig14_nominal_housing_first_passage.png | Earliest reported annual median crossing four fixed nominal thresholds, with right-censored municipalities | **Observed median comparisons; no mover or displacement observations** |
| 15 | fig15_rolling_housing_prediction_mae.png | Held-out annual growth forecast errors with and without one fixed commuting-network lag | **Exploratory prediction; not a causal transmission test** |
| 16 | fig16_rolling_housing_naive_benchmarks.png | Fixed, unfitted zero-change, own-persistence and metropolitan-prior-mean forecasts vs two ridge models | **Retrospective held-out predictive benchmark; not a causal test** |
| 17 | fig17_network_prediction_municipal_influence.png | Municipality-level contribution to baseline-minus-network error, averaged over three held-out years | **Heterogeneous prediction error; not observed moves** |
| 18 | fig18_nominal_vs_hicp_deflated_housing_growth.png | Unweighted median of 17 municipalities' nominal and national-HICP-deflated housing median changes | **Observed inputs; derived consumer-price adjustment, not affordability** |
| 19 | fig19_nominal_vs_real_housing_first_passage.png | First nominal versus HICP-deflated rent/price threshold years, with not-yet-crossed cells | **Descriptive annual first passage; not residential migration** |
| 20 | fig20_synthetic_rent_fiscal_income_70m2.png | Paired 2020/2024 synthetic annual new-rent / gross fiscal-income ratios for a stipulated 70 m² dwelling | **Derived cross-statistic ratio; not a measured household rental burden** |
| 21 | fig21_rent_vs_fiscal_income_growth_2020_2024.png | Municipal new-contract rent median growth vs gross fiscal-income median growth | **Descriptive separate-population medians, not tenant income** |
| 22 | fig22_porto_relative_rent_discounts_2020_2025.png | Porto-relative percentage discount in every other AMP municipality, 2020 and 2025 | **Independent municipal new-rent medians; not a set of available moving choices** |
| 23 | fig23_porto_rent_discount_and_real_savings_annual.png | Annual median/IQR of relative discounts and illustrative 70 m² rent savings adjusted to 2020 consumer euros | **Synthetic rent-only comparisons; no actual moving or travel expenses** |
| 24 | fig24_porto_link_intensity_vs_rent_discount_compression.png | 2021 Porto-pair interactions per 1,000 comparator residents against 2020–2025 percentage-point rent-discount compression | **Undirected work/study links, not resident movers or causal exposure** |
| 25 | fig25_porto_link_rank_sensitivity_windows.png | Spearman correlations over 2020/2021/2022/2023–2025 windows and three Porto-link exposure normalizations | **Exploratory window sensitivity; shared Porto denominator and dependent municipalities** |
| 26 | fig26_porto_geographic_distance_vs_rent_discount_compression.png | Official CAOP2025 municipal centroid distance to Porto against rent-discount compression | **Administrative centroid straight-line distance, not travel time or housing-search accessibility** |
| 27 | fig27_porto_commuting_gradient_distance_adjustment.png | Unadjusted and centroid-distance-conditional Spearman coefficients over all three network exposures and four windows | **Severe proximity/commuting rank collinearity; partial association is not a causal effect** |

The figure titles, manuscript captions and generated tables repeat the
evidence-class distinction. There are no charts of invented historical
municipality values. The illustrative experiments do not label tiers with
real municipalities.

## Results tables

- results/amp_growth_summary.csv — 17 observed municipality endpoint
  comparisons and a clearly named illustrative 100 m² / income proxy.
- results/hypothetical_cascade.csv — three tiers in two parameter scenarios;
  each row says hypothetical.
- results/hypothetical_commuting_cost.csv — assumed housing, transport, time
  and total cost by distance; each row says hypothetical.
- results/provenance.json — fixed source identity, observational coverage,
  and machine-readable false flags for migration data and causal estimation.

The municipal medians of growth rates in the article are unweighted across
17 municipalities. They are *not* equivalent to the growth of a pooled or
transaction-weighted metropolitan median.

The temporal module adds `results/sequential_housing_first_passage.csv`,
`results/sequential_housing_rolling_predictions.csv` (102 held-out
municipality-years across both outcomes),
`results/sequential_housing_rolling_summary.csv` (fold and pooled MAE), and
`results/sequential_housing_provenance.json` (source identity, model contract,
network-label benchmark and alternative-weight sensitivity).

## Mathematics and integrity checks

For the descriptive percentage change, the program uses
100 × (endpoint / baseline − 1) and rejects a zero baseline. It verifies
municipality identity, one row per municipality-year, the complete 17 × 7
grid, positive published prices and rents, expected missingness, and pinned
Porto reference values.

The hypothetical relocation experiment uses:

\[
I_{k+1}=I_k(1-a_k),\qquad A_k=I_ka_k,
\]

with initial households \(I_1=1000\), assumed tier absorption fractions
\(a_k \in [0,1]\), and exact conservation
\(\sum_k A_k+I_{K+1}=1000\). Under constrained assumptions
[0.20, 0.20, 0.25], 480 remain outside the third tier. Under more responsive
assumptions [0.55, 0.60, 0.75], 45 remain. These values are mathematical
consequences of chosen inputs, not predictions.

The hypothetical commuting calculation uses a monthly rent function
\(1100-600(1-e^{-d/25})\), transport \(35+4d\), and time valuation \(5.5d\),
where \(d\) is a one-way distance in kilometres. None of these coefficients
is estimated from Portuguese transport or expenditure surveys.

The original 13 unit tests check provenance-specific reference values, missingness,
duplicate rejection, numerical growth, conservation, parameter bounds,
comparative statics, costs and reproducible outputs. Fifteen additional tests
audit the Census-derived pairs, undirected counting, Moran calculations,
permutation determinism, and new generated artifacts.

## Follow-up: observed commuting connectivity (2021)

The additional module [commuting_network.py](commuting_network.py) joins a
fixed 2021 **work/study commuting interaction** matrix to the same 17 AMP
municipalities. Its source is [INE Census 2021 indicator 0012340](https://www.ine.pt/xurl/indx/0012340/PT),
**as transcribed into** [a third-party workbook](https://github.com/andrebrito0/OD_Data_Portugal_Census_2021)
at pinned commit `a3bd023b8eab56962cb377d1f07c6279cde67007`.
The source workbook SHA-256 is
`7d6dbb58be8356c868c9d0d548a81db27025228f662c5c09d71cd4ba1ee7dace`.
The direct INE JSON API returned HTTP 503 during extraction, so the workbook
has **not yet been independently reconciled with the official API**. The
derived 136-pair CSV and detailed provenance are included for audit. Credit
INE and the workbook's author; the INE data portal lists CC BY 4.0.

Crucially, the 17×17 source matrix is **exactly symmetric**, with 136 positive
unordered municipality pairs and zeros on its diagonal. The derived CSV keeps
one row per unordered pair. It reports a total of **339,378 pair
interactions** within the AMP subset (the full symmetric matrix would double
that count). The strongest link is Porto–Vila Nova de Gaia (**39,960** reported
pair interactions). These counts do not identify commuter direction,
within-municipality commuting, household relocation, or displaced tenants.

Reproduce the frozen extract after obtaining the pinned upstream workbook:

    python extract_commuting.py --source-xlsx /path/to/all_data.xlsx \
      --output /tmp/reproduced_commuting_pairs.csv \
      --provenance /tmp/reproduced_commuting_provenance.json

Then regenerate the two additional observed-data figures and diagnostics:

    python commuting_network.py --output /tmp/porto-commuting-check

The reproducible outputs are:

- `figures/fig9_2021_commuting_interactions.png`: log-colour heatmap of
  **observed** undirected interactions (not migration);
- `figures/fig10_commuting_moran_null.png`: prespecified fixed-seed,
  two-sided random-label null comparisons for municipal growth rates;
- `results/commuting_spatial_diagnostics.csv`: raw weights (primary), square-root
  weights and top-three strongest-neighbor union (sensitivities), plus excluding
  Porto from the raw-weight calculation;
- `results/commuting_municipal_exposure.csv`: each municipality's observed
  interaction strength and the fixed-network-weighted mean of *other*
  municipalities' sale/rent growth;
- `results/commuting_diagnostics_provenance.json`: source identity, interpretation,
  configuration and numerical results.

With 9,999 random-label permutations, global Moran's I using **raw symmetric
interactions** is **−0.197** for sold-price growth 2019–2025 (two-sided
`p = 0.303`) and **+0.156** for new-contract rent growth 2020–2025
(`p = 0.114`). These are exploratory correlations among just 17 municipalities,
**not evidence of causal price contagion or non-contagion**. The fixed 2021
commuting network overlaps the periods whose endpoints are compared, so it
cannot establish temporal order. Most importantly, binary adjacency is
complete (every pair has a positive count); converting this matrix to a binary
graph would make Moran's I degenerate under permutation. We therefore do not
claim binary-neighbor significance or infer residential moving flows. Removing
Porto also changes the *sign* of the raw-weight sale-price association, so a
single metropolitan summary is not a stable account of the mechanism.

## Follow-up: observed residential origins, Porto Census 2021

The [official Porto open-data portal](https://dadosabertos.cm-porto.pt/dataset/indicadores-relativos-populacao-residente-no-porto-resultados-definitivos-censos-2021)
redistributes an **INE Census 2021** table about the whereabouts of current
residents on **31 December 2019**, classified into same home, changed home within
the same parish, another parish in the same municipality, another municipality,
abroad, and not yet born. [Question 10 of the Census questionnaire](https://censos.ine.pt/scripts/censos_css_js/quest/PT_Q_Individual_Censos2021_INE.pdf)
confirms the exact question and the 19 April 2021 Census reference date. This
interval is *longer than one year* and cannot be converted into an annual rate.

Unlike the commuting matrix, these are genuinely **residential-origin counts**,
but the accessible table supplies only 11 nested geographic rows: Portugal,
Norte, AMP, Porto, and Porto's seven contemporary parishes. Crucially, its
"other municipality" category does **not name which origin municipality**.
The extract therefore measures inbound origins for 2021 Porto residents, **not
Porto-to-suburb exits** or a 17×17 residential-migration matrix.

The public archive is fixed at SHA-256
`52f447ed1e1e159132ca94c9719d74ab7553f3eefaee4107f391ea0cd54fdf14`;
the selected XLS member at SHA-256
`0d82ac3a057202c3781ec73ebfbb32ee29d3c60f0c86e9b64a3e1b2b8d47def5`.
The dedicated [extract_residential_mobility.py](extract_residential_mobility.py)
validates these hashes and the original INE dimension labels, preserves stable
2021 geographic codes, and rejects suppression or non-integer counts.

Porto counted **231,800** residents in 2021, of whom **2,073** had not yet
been born on the prior reference date. Among the remaining **229,727** people,
**15,724 (6.84%)** had lived in a *different municipality* at the earlier date;
**4,345 (1.89%)** had lived abroad. This is an observed inbound-residence
comparison, not a count of residents priced out. The municipality's seven
parishes sum **exactly** to the municipality in all eight independently checked
population categories. Their other-municipality shares range from **5.92%**
(Aldoar–Foz–Nevogilde) to **8.40%** (Cedofeita–Santo Ildefonso–Sé–Miragaia–
São Nicolau–Vitória union). These differences alone cannot identify why people
moved or whether anyone was displaced.

Reproduce from the frozen public ZIP, which need not enter version control:

    python extract_residential_mobility.py --source-zip /path/to/archive.zip \
      --output /tmp/reproduced_residential.csv \
      --provenance /tmp/reproduced_residential_source.json

And regenerate the observed diagnostic tables and figures:

    python residential_mobility.py --output /tmp/porto-residential-check

The committed outputs include:

- `data/porto_census2021_residential_mobility.csv` and a paired source-provenance
  JSON, with all 11 geographic margins and explicit 2019-eligible denominators;
- `figures/fig11_parish_intermunicipal_in_mobility.png`, an observed comparison
  of the seven parishes with Porto and AMP reference rates;
- `figures/fig12_parish_residential_mover_mix.png`, a separate category
  composition **among movers only**, never confused with all residents;
- `results/residential_mobility_parish_summary.csv` and
  `results/residential_mobility_reconciliation.json`, including exact
  accounting controls and unobserved-flow limitations.

Twelve additional tests enforce category partitions, the full parish-to-city
reconciliation, source schema, hashes and publication artifacts. The combined
test suite had **40 tests** at the observed-residential-baseline checkpoint.

## Follow-up: sharp bounds on the missing Porto-to-suburb flow

The official Census origin categories tell us **how many 2021 residents of a
destination lived in another municipality on 31 December 2019**, but not the
specific origin. This produces a formal *partial-identification* problem.
The nested official counts above imply:

- **15,724** intermunicipal in-movers now living in Porto;
- **124,618** in-movers now living in any AMP municipality, including Porto;
- **108,894 = 124,618 − 15,724** in-movers now living in the other **16** AMP
  municipalities, whose previous municipality could have been Porto, another
  AMP municipality, or a municipality outside AMP.

Let **x** be the unobserved number of these people now living in the other
16 municipalities who lived in Porto in 2019. Let **y** be current Porto
residents who lived in one of the other 16 municipalities. With these
destination margins alone:

\[
0 \leq x \leq 108{,}894,\qquad 0 \leq y \leq 15{,}724,
\qquad -15{,}724 \leq x-y \leq 108{,}894.
\]

These bounds are *sharp given only the published margins*: every integer
point in the rectangle can be completed by assigning the remaining in-movers
to other origins, which are not identified in the extract. Two explicit
allocations both reconcile exactly to the observations:

- `x=25,000`, `y=5,000`: hypothetical net **20,000 toward the other AMP
  municipalities**;
- `x=5,000`, `y=12,000`: hypothetical net **7,000 toward Porto**.

The very different stories require no change to the observed Porto or AMP
counts. Therefore even the **sign** of the exchange cannot be determined.
The large right-hand endpoint is a logical ceiling, *not* an estimate of
outward displacement. Neither this range nor the picture is a confidence
interval or probability distribution. It covers only people recorded as
2021 AMP residents with an intermunicipal prior residence, not moves out of
the region, international movement, deaths, households, or move motives.

Reproduce the construction and Figure 13 without network access:

    python residential_identifiability.py --output /tmp/porto-residential-od-bounds

This produces:

- `results/residential_od_identification_bounds.json` with the observed
  margins, sharp bounds, source hashes and explicit inference limitations;
- `results/residential_od_feasible_worlds.csv` with five fully reconciled
  **hypothetical** allocations and their residual-origin categories;
- `figures/fig13_residential_od_identification_set.png` showing the rectangle
  of feasible allocations and the full net-exchange bound.

Nine additional tests check the sharp endpoints, signs, accounting identities,
invalid inputs and deterministic artifacts: **49 tests in total**.

## Next empirical step: request and validate actual residential OD counts

The [data-access and intake specification](OD_DATA_ACCESS.md) now describes
two documented INE routes: a special statistical cross-tabulation request,
and an accredited scientific-research route whose eligibility must be
confirmed. Census 2021 question 10 explicitly requested the **named
municipality** of previous residence from people who had moved from another
municipality. The available published extract groups that detail into
"other municipality", so the cross-tabulation still needs authorization.

The target is a **17-destination × 18-origin** aggregated table:

- Destinations: the 17 official AMP 2021 DICO municipalities.
- Origins: those 17 plus `REST_PT` for all other Portuguese municipalities.
- Counts: individual 2021 residents formerly living in another Portuguese
  municipality on **31 December 2019**, not commuters or households.
- Privacy: explicit `observed`, `suppressed` and `structural_zero` statuses;
  suppressed cells remain unknown, never zero-imputed.

The new [residential_od_intake.py](residential_od_intake.py) validates a real
authorized aggregate source **once it is obtained**, with a pinned SHA-256,
exact DICO scope, a complete 306-cell grid, source metadata and reconciliation
to **15,724 for Porto** and **124,618 for AMP**. With suppressions, it computes
conservative sharp count bounds from the published marginal totals and the
reported cell mask. It rejects changed universe, wrong dates, duplicate or
missing municipality pairs, malformed integers, and personal-level columns.

The command is intentionally unusable without a supplied authorized source:

    python residential_od_intake.py \
      --source-csv /path/to/approved_aggregate_od.csv \
      --metadata-json /path/to/approved_release_metadata.json \
      --output-json /tmp/porto-residential-od-audit.json

**No real directed OD table has been received or committed.** Fourteen new
tests use only disposable synthetic 18×17 tables; **63 total tests pass**.
No statistical request has been submitted or external party contacted by
this repository update. The manuscript remains a descriptive study until
such a source is validated.

### Audited access routes for an actual directed OD table

| Source | Available information | Missing for our estimand |
| --- | --- | --- |
| [INE Census 2021 individual questionnaire, Q10](https://censos.ine.pt/scripts/censos_css_js/quest/PT_Q_Individual_Censos2021_INE.pdf) | Earlier usual residence question | Published privacy-safe cross-tabulation of previous **and** current municipality |
| [INE Census indicator 0011631](https://www.ine.pt/xurl/indx/0011631/PT) | Current place, sex and previous-residence migration **category** | Named pair of municipalities; the public catalog metadata gives no OD dimension |
| [Municipality of Porto official Census archive](https://dadosabertos.cm-porto.pt/dataset/indicadores-relativos-populacao-residente-no-porto-resultados-definitivos-censos-2021) | Porto/parish residential origin margins, with exact checksums | Destination-specific original municipality and departures from Porto |
| [INE indicator 0012340](https://www.ine.pt/xurl/indx/0012340/PT) | Work/study *commuting* interactions | Residential relocations (commutes are not moves) |
| [INE annual resident population estimates: methodological document 113](https://www.pordata.pt/sites/default/files/methodological_documentation/Documento_metodologico_estimativas_anuais_da_populacao_residente_113.pdf) | Estimated municipal demographic components and internal-migration balance | Bilateral migration flows, individual reasons for moving |

These are the sources inspected, not a claim that no other table or restricted
access route exists. The next empirical requirement is a disclosed,
privacy-safe cross-tab of **municipality of residence on 31 December 2019 ×
municipality on 19 April 2021**, with a clear universe, cell suppression,
geographic version and access/licensing terms. To test housing displacement
rather than mere exchange, origin/tenure and housing-cost information or a
defensible causal design would also be needed.

## Observed annual timing and rolling prediction (Stage B)

The [sequential_pressure.py](sequential_pressure.py) follow-up reuses the
**source-hashed** 17-municipality annual panel and the fixed 2021
**undirected work/study interaction** weights. It adds no new data source.
Run it offline, with 999 deterministic network-label shuffles, using:

    python sequential_pressure.py --output /tmp/porto-sequential-check --shuffles 999

The observed first-passage table reports the earliest *published year* in
which a municipality median crosses a **nominal** cumulative threshold:
sales use 2019 as baseline (+25/50/75/100%), while new-contract rents
use 2020 (+25/50/75%). An empty cell means the threshold had **not** been
reached through 2025; it is not imputed or assumed never to occur. All
**17** municipalities first exceeded the +25% new-contract rent threshold
in **2023**. For sold-price medians, +50% was first reached by **2** in
2022, **8** in 2023, **6** in 2024, and **1** in 2025. Porto's median first
reached sale +50% in 2023 and new-contract rent +50% in 2025. These
aggregate thresholds provide neither a date for a household move nor
evidence that one municipality drove another's price.

To examine whether the observed connectivity helps *predict* later changes,
the outcome is 100× the annual log ratio of successive medians. For each
municipality and target year the fixed two-feature baseline contains its
**own preceding-year growth** and the **AMP mean of the preceding-year
growth**. The extra network feature is its 2021 interaction-weighted mean
of **other** municipalities' preceding-year growth, *minus the AMP
preceding-year mean*. The difference isolates the network-specific departure
from a common metro change; it is not a measured spatial spillover.

Both regressions use fixed ridge penalty **1.0**, with coefficients and
feature standardization learned strictly from training years. Targets from
**2022** form the first training fold; **2023**, **2024** and **2025** are
tested one year at a time on 17 municipalities, without using any test-year
outcome for its prediction. The short time series yields just **three
held-out calendar years**, 51 municipality-years per outcome, and correlated
observations; its predictive uncertainty is large and is not quantified by
pretending the 51 observations are independent.

| Held-out target | Sales baseline MAE | Sales + network | New-rent baseline MAE | New-rent + network |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 3.13 | 3.10 | 7.15 | 10.08 |
| 2024 | 4.86 | 4.89 | 6.15 | 10.37 |
| 2025 | 7.78 | 7.57 | 4.07 | 3.97 |
| **Pooled** | **5.26** | **5.19** | **5.79** | **8.14** |

All values are MAE in **annual log-growth percentage points**, not median
house prices or percentages of displaced households. For sale medians the
network reduces pooled MAE by only **0.069** point. For new-contract rent
it *increases* pooled MAE by **2.351** points. Alternative square-root and
top-three-union weights leave the price improvement small (+0.057 and
+0.043), and the rent result worse (−2.817 and −1.654). In an informal
999-draw network-label shuffle, the fractions with as much or more pooled
improvement were **0.094** for sales and **0.989** for rents. These fractions
are **not calibrated p-values**: municipality identities cannot reasonably
be treated as exchangeable assignments, and the fixed 2021 network overlaps
the outcomes' overall periods.

The strongest defensible finding is **synchronized nominal new-rent
thresholds but no consistent incremental network forecasting advantage**.
Neither the threshold heatmap nor the forecasts identify an outward
housing-affordability epidemic. Actual directed residential flows,
household-level affordability and an identification strategy are still
missing. Thirteen new tests enforce timing, censoring, model inputs,
prediction leakage and deterministic outputs: **76 total tests**.

## Forecast robustness: do simpler alternatives do better?

The [forecast_robustness.py](forecast_robustness.py) companion deliberately
does **not select new network coefficients or retune the existing model**.
Instead, it compares the same 2023–2025 held-out annual-median predictions
with three transparent, unfitted benchmarks: **zero annual log growth**,
**each municipality's previous-year annual log growth**, and **the
unweighted AMP mean of previous-year annual log growth**. No target-year
outcome enters any benchmark. All values are mean absolute error in
annual log-growth percentage points (not percentages of households moving).

Reproduce the two new figures and four auditable files offline:

    python forecast_robustness.py --output /tmp/porto-forecast-robustness

| Held-out pooled 2023–2025 model | Median sale-price MAE | New-contract rent MAE |
| --- | ---: | ---: |
| No annual change | 12.65 | 11.36 |
| Own prior-year growth (persistence) | 8.10 | 5.33 |
| AMP prior-year mean growth | 5.60 | **3.85** |
| Own + AMP prior-year ridge | 5.26 | 5.79 |
| Add 2021 commuting-network deviation | **5.19** | 8.14 |

For **new-contract rents**, the *unfitted AMP prior mean* beats both ridge
models in each of the three annual folds. The network feature worsens
absolute error in **all 17 municipalities when each is averaged over the
three test years**. These outcomes are consistent with a broad common
metropolitan pattern, but they do not identify the reason it occurs.
For sales, deleting 2025 from the **evaluation** changes the network's
pooled improvement from +0.069 to **−0.002**; its modest advantage is not
stable to which forecast years are scored.

Deleting any municipality's three *held-out evaluation cells* (without
refitting) leaves sale-network gains in **+0.048 to +0.100** and rent gains
in **−2.485 to −2.280** log-growth points. A separate ablation actually
**refits both ridge models on the remaining 16 municipalities**, removes
Porto from the network and recomputes the metropolitan mean: sale-network
improvement becomes **−0.010**, rent improvement **−2.077**. This is a
sample/network-scope sensitivity, **not** a counterfactual effect of Porto.

The 2023 forecast fold has an additional identifiable design limitation:
its training targets come *only from 2022*, so the AMP mean prior-growth
feature is **constant across training municipalities** and is collinear
with the intercept. It cannot be independently estimated in this first
fold. Moreover, the source observations describe the previous reference
year, but the actual *publication dates* of the municipal medians have not
been reconstructed. These are retrospective one-year-ahead diagnostics,
not verified real-time forecasts.

The new outputs are `results/forecast_robustness_benchmark_mae.csv`
(eight fold/pooled rows),
`results/forecast_robustness_municipality_deletions.csv` (34 evaluation
deletions), `results/forecast_robustness_year_deletions.csv` (six), and
`results/forecast_robustness_provenance.json` (source hash, fixed
predictors, Porto ablation, fold design rank and limitations). Eleven
additional unit tests validate identities and artifacts: **87 tests pass**.
No bootstrap interval, randomization p-value, housing spillover coefficient
or displacement estimate is inferred from these short, dependent series.

## Fixed Eurostat HICP and nominal-versus-deflated housing growth

The next [real_housing_pressure.py](real_housing_pressure.py) check adds an
**official Eurostat input** that was absent from the initial fixed panel:
[HICP annual indices and rates, `prc_hicp_ainr`](https://doi.org/10.2908/PRC_HICP_AINR),
Portugal (`PT`), annual (`A`), all-items (`TOTAL` under ECOICOP 2018),
unit `INX_A_AVG`, **2025=100**, years **2019–2025**. This is the *current*
series; the older `prc_hicp_aind` has been retired. The Eurostat dataset's
last update in our frozen response is **17 September 2026**. It was
retrieved on **1 November 2026**. The original 4,213-byte official
JSON-stat response is committed at
`data/portugal_eurostat_hicp_2019_2025.json` with SHA-256
`11742a14bd70d43c22e7170b07b65081f582d2a85eeac22005448f20327bd117`.
Eurostat [authorizes statistical-data reuse with attribution](https://ec.europa.eu/eurostat/help/copyright-notice).

Reconstruct the seven-row validated HICP extract and its source identity:

    python extract_hicp.py --output /tmp/porto-hicp-check

Then reproduce the housing arithmetic, figures 18–19 and three CSV results:

    python real_housing_pressure.py --output /tmp/porto-real-housing-check

The calculation for each municipality's original annual median \(P\) is:

\[
G^{\mathrm{deflated}}_{i,t}
=100\left[\frac{P_{i,t}/P_{i,b}}{H_t/H_b}-1\right],
\]

where \(H\) is **Portugal's annual average all-items HICP** and \(b\)
is **2019 for sale medians** or **2020 for new-contract rent medians**.
The national deflator is shared by all 17 municipalities. The yearly
national index is **83.07** in 2019, **82.97** in 2020 and **100.00** in
2025: general consumer prices rose **20.38%** from 2019 and **20.53%**
from 2020 to 2025, on this source vintage.

| Unweighted median across 17 municipalities | Nominal growth to 2025 | HICP-deflated growth to 2025 |
| --- | ---: | ---: |
| Sale-price median change since 2019 | 102.97% | **68.61%** |
| New-contract rent median change since 2020 | 60.48% | **33.15%** |

The initial observation that **all 17** municipality new-contract rent
medians first crossed **nominal +25%** in 2023 changes after deflation:
only **one** first crosses +25% HICP-deflated in 2023; **ten** first cross
in 2024 and **six** in 2025. Porto crosses **real +25% in 2025**, not
2023. At a +50% rent threshold, all 17 nominal series crossed by 2025,
but only **one** HICP-deflated series did. For sale medians, **17**
crossed nominal +50% by 2025, compared with **14** deflated; Porto's
sale median had **not yet** crossed real +50% by 2025. Missing 2019
rent remains unobserved. A *first observed crossing* is not an
intrayear date or guarantee of staying above a threshold.

These are **consumer-basket-adjusted comparisons** of municipality
annual medians, not constant-quality housing-price series, regional
cost-of-living indices, real salaries, mortgage service ratios or
the rent burdens of individual households. National HICP includes
consumer actual rents but excludes imputed owner-occupied housing
costs. Common inflation can materially shift threshold dates without
identifying common shocks, causal effects or moving flows. The article
now makes this distinction explicit.

The added results are `results/real_housing_annual_municipal_medians.csv`
(13 annual outcome-year rows), `results/real_housing_first_passage.csv`
(17 municipality rows), `results/real_housing_municipal_endpoints.csv`
(34 municipality-outcome rows) and `results/real_housing_provenance.json`.
The paired HICP source and derivation are also stored in `data/`, with
independently audited SHA-256 hashes. Twelve new tests enforce source
identity, exact dimensions, missingness, arithmetic and artifacts:
**99 tests in total**.

## Matched fiscal-income and new-contract rent proxy, 2020–2024

The next [fiscal_income_pressure.py](fiscal_income_pressure.py) analysis
compares two existing, independently defined statistics in the already
checksummed municipal panel. It adds **no invented observations** or new
external data. Annual median newly contracted rent is EUR/m²/month;
the municipal `income` variable is annual median **gross IRS-declared
income per fiscal household**, in EUR. INE's [local-income statistical
publication](https://www.ine.pt/documentos/municipios/rnlocal/ano2022/1B0.pdf)
(technical note) and its [2024 release](https://webinq.ine.pt/home.aspx)
explain the administrative tax-return basis. These statistics do not cover
identical populations. In particular, the fiscal-return population is
**not a survey of new renters** and is not the complete set of residents.

For municipality \(i\), publication year \(t\) and **assumed** dwelling
floor area \(A\) in m², the synthetic ratio is

\[
Q_{i,t}(A)=100\,\frac{12 A R_{i,t}}{Y_{i,t}},
\]

where \(R\) is the municipality median of new-contract EUR/m²/month
and \(Y\) is the municipality median of gross declared annual fiscal-
household income. The relative change is

\[
100\left[\frac{R_{i,t}/R_{i,2020}}{Y_{i,t}/Y_{i,2020}}-1\right].
\]

The *relative change* does not depend on the assumed size. The
*level* does. We expose **50, 70 and 90 m²** scenarios rather than
calling a chosen size a typical tenancy. Dividing both nominal rent
and nominal income by the same HICP also cancels exactly, so the
ratio is unaffected by the national general-inflation convention.
This differs from deflating rents alone, which was the previous study.

Across 17 municipalities, the unweighted median of their individual
2020–2024 rent growth rates is **+47.59%** and fiscal-income growth
rates **+29.90%**. The median of the 17 *individual ratio changes* is
**+15.17%**. All **17/17** ratios worsened by 2024; Porto's synthetic
ratio grew **+30.55%**. For an assumed **70 m²** new contract in Porto,
the quotient was **54.75%** in 2020 and **71.48%** in 2024, a **+16.73
percentage-point** level increase. These percentages **must not be
reported as observed household rent burdens**, even when they exceed
standard affordability thresholds: the numerators and denominators
come from different populations and different median operators.

Reproduce 85 annual municipality-year rows, the 17 endpoint comparisons,
five unweighted annual summaries, a source ledger, and figures 20–21:

    python fiscal_income_pressure.py --output /tmp/porto-fiscal-income-check

The saved files are `results/fiscal_income_rent_annual_municipal_proxy.csv`,
`results/fiscal_income_rent_municipal_endpoints.csv`,
`results/fiscal_income_rent_annual_summary.csv`, and
`results/fiscal_income_rent_provenance.json`. The 13 new tests cover
exact-period pairing, guards, level-versus-growth area dependence,
inflation cancellation, medians, evidence boundaries and outputs:
**112 tests in total**.

This proxy does not account for actual contract floor areas, new-renter
incomes, household size, after-tax disposable income, incumbent rents,
non-IRS incomes or travel costs. Income is missing for **2025**, so no
2025 income-adjusted claim is made. Neither the direction nor the cause
of any residential move follows from this all-municipality association.

## Porto-to-other-municipality rent gaps, 2020–2025

The subsequent [porto_rent_options.py](porto_rent_options.py) module
asks a different question: **does a lower new-contract rent median
outside Porto still exist, and does its *relative discount* move with
its *absolute euro saving*?** It uses only two already frozen inputs:
the same fixed 17-municipality INE-derived rent panel and Eurostat's
Portugal national all-items annual average HICP. There is **no new
housing inventory, matching of actual homes, residence change or
measured transport cost**.

For each of the other 16 AMP municipalities \(j\) and year \(t\):

\[
D_{j,t}=100(1-R_{j,t}/R_{\mathrm{Porto},t}), \qquad
S_{j,t}(A)=A(R_{\mathrm{Porto},t}-R_{j,t}),
\]

where \(R\) is the published new-contract median in EUR/m²/month
and \(A\) is an **assumed** constant floor area (50, 70 or 90 m²).
The euro saving \(S\) is illustrative and *rent only*. To compare
its purchasing-power value over time, report
\(S_{j,t}^{2020€}=S_{j,t}H_{2020}/H_t\) using the national HICP
(2020 index **82.97**, 2025 index **100.00**). The same-year
percentage discount is already independent of a common deflator
and of floor area.

The 2020–2025 descriptive result is subtle:

| Unweighted comparison across the 16 other AMP municipalities | 2020 | 2025 |
| --- | ---: | ---: |
| Median relative new-rent discount to Porto | 48.22% | **46.60%** |
| Median hypothetical 70 m² monthly rent saving, nominal EUR | €302.75 | **€460.95** |
| Median hypothetical 70 m² monthly saving, 2020 consumer EUR | €302.75 | **€382.45** |
| Municipalities with lower new-rent medians than Porto | 16/16 | **16/16** |

Among the 16 endpoint pairs, the relative percentage discount
**narrowed in 13**, yet the hypothetical **real euro gap increased
in all 16**. This is algebraically compatible: relative and
absolute differences are not the same measure. The annual sequence
is not monotonic (the median relative discount is **49.47% in
2023**, higher than its 2020 value). No spatial price wave or
displacement event is inferred from these comparisons.

Generate **96** municipality-year rows, 16 endpoint rows, six annual
summaries, the provenance ledger and figures 22–23:

    python porto_rent_options.py --output /tmp/porto-rent-options-check

Outputs are the annual comparison, municipal endpoint, annual summary
and provenance files under results/porto_rent_options_*. Fourteen
additional tests check the official source pin, 2020/2025 endpoints,
denominator and floor-area invariants, negative-gap support,
reproducibility and missing-data rejection: **126 tests total**.

The other AMP municipalities are **administrative comparators**,
not demonstrated housing-search substitutes. Ratios of annual
municipal rent medians need not compare equivalent dwellings.
Savings exclude moving costs, jobs, transport, time, household income,
vacancy and rental availability, so they do not measure net benefits
for anyone asked to move out of Porto.

## Porto-link intensity and rent-discount compression (figures 24–25)

The reproducible follow-up tests whether the 2021 **undirected**
Porto work/study pair intensity aligns with percentage-point compression
of the comparator's Porto-relative new-contract rent discount through
2025. The 16-municipality 2020–2025 Spearman values are **−0.179412**
for raw Porto pair counts, **−0.297059** for counts per 1,000 2021
municipal residents and **−0.270588** for Porto's share of the
municipality's within-AMP pair strength. Raw Porto pair counts are
very closely ranked with population (**+0.973529**). With 2023 rather
than 2020 as the baseline, all three correlations become weakly
positive (about +0.16, +0.02 and +0.05). The 16 primary
population-normalized leave-one-out correlations remain negative.

Regenerate figures 24–25, the 16-row exposure table, 12 rank/shuffle
benchmarks, 16 leave-one-out comparisons and a source/caveat ledger:

    python porto_network_gradient.py --output /tmp/porto-network-gradient --shuffles 9999

Results are saved as
results/porto_network_gradient_municipal_exposures.csv,
results/porto_network_gradient_rank_benchmarks.csv,
results/porto_network_gradient_leave_one_out.csv and
results/porto_network_gradient_provenance.json. The fixed-seed
random-label shuffle tail fractions are deliberately labeled **not
inferential p-values**; they assume implausible municipal label
exchangeability. Population normalization is not a commuter share.
The source is a checksummed *third-party* Census-derived workbook
attributed to INE, not a verified direct INE download. Counts are
undirected work/study interactions, not residential OD moves; the
2021 network overlaps the 2020–2025 outcome. These correlations
are descriptive and cannot establish displacement or rent spillovers.
Fourteen additional unit tests cover rankings, source-linked exact
values, shuffles, window stability, deletions and regenerated outputs:
**140 tests total**.

## Official DGT geography and Porto-link distance sensitivity (figures 26–27)

The official [DGT CAOP2025 municipal geometry collection](https://ogcapi.dgterritorio.gov.pt/collections/municipios)
supplies a geographic benchmark independent of the previously
checksummed, third-party commuter-link workbook. The pinned OGC API
request filters NUTS3 code **11A** and requests **EPSG:3763** projected
MultiPolygon coordinates. The code in [extract_caop.py](extract_caop.py)
computes area-weighted centroids of the 17 administrative municipal
polygons, subtracts interior holes, rejects malformed geometry and
validates the official listed area to 2.5%. The checked-in frozen
17-row centroid CSV has SHA-256
**a05cfb55ffba89dc9cc33756480470005fb771c35e824b331f81759e02a80966**.
The response bytes SHA-256
**9eb0060c8bdd60c785800ed43e79bf166e646969bc3e694719e44cc81aef917b**
is recorded alongside the exact request and the attribution required
by the official DGT **CC-BY 4.0** license in
data/amp_caop2025_municipal_centroids_provenance.json.

This geometry is current as of **CAOP2025**, not a reconstruction of
each 2020–2025 municipal perimeter. It yields 16 positive Euclidean
municipal-centroid distances from Porto (about 7.10–43.64 km), not
road routes, centre-to-centre commuting time, housing-search
accessibility or the location of the median resident.

For the 2020–2025 rent-discount compression outcome, geographic
distance alone has Spearman **+0.176471**. Distances are very
closely ranked against 2021 Porto link strength: raw count
**−0.852941**, count per 1,000 municipal residents **−0.950000**,
and the within-AMP link share **−0.797059**. Conditioning each
commuting exposure and outcome's ranks on distance produces a
purely descriptive partial Spearman: for 2020–2025, raw counts
**−0.056232**, per-resident links **−0.421057** and within-AMP
share **−0.218583**. In the per-resident case, distance and commuting
exposure are so nearly collinear that partial correlations must not
be interpreted as independent channel effects. Neither the direct
nor adjusted comparisons identify geographic housing displacement.

Offline reproduction of the 16-row municipal join, 16 rank/window
diagnostics, 16 leave-one-out partial-rank values, ledger and figures:

    python porto_distance_sensitivity.py --output /tmp/porto-distance-sensitivity

The four files appear under results/porto_distance_sensitivity_*.
To *re-query* the official geometry deliberately (network required)
and compare the source checksum before replacing the frozen derivative:

    python extract_caop.py --download --output /tmp/caop-refresh

This extraction should **not** be run on every CI build: reproducible
tests and figures use the pinned CSV, which is checked against its
recorded SHA-256. The new extractor and distance sensitivity add
**21 unit tests, 161 total**.

## Boundaries before publication

- Prices of sold homes are not quality-adjusted repeated-sales observations.
- New-contract rents are not the average rents paid by incumbent tenants.
- Municipal population counts cannot identify domestic relocation origins.
- The fixed geographic AMP perimeter does not equal a measured commuting
  or housing-search market.
- A descriptive spatial lag or temporal ordering is not a causal spillover.
- 2021 commuting interactions are undirected and do not reveal where households
  moved. The cached external workbook has not been cross-checked against a
  fresh official INE API download.
- The separate Porto residential-origin counts do not identify which
  municipality anybody came from, Porto's outflows, household income, rent,
  search motives, or a directed movement network.
- The 2019–2025 municipal sample is too short for a high-dimensional
  multi-lag causal model without further sources and identification.
- No household-level rent or generalized transport burden is measured here.
- Do not use the hypothetical curves to infer the number of people actually
  displaced, transmission elasticities, or policy effect sizes.
- The article is a manuscript for review and **is not automatically published
  to Medium** when a pull request is opened or merged.

## Literature

See the article's cited source list. The theoretical starting points include
Guerrieri, Hartley and Hurst (2010/2013), Couture et al. (2019), Asquith,
Mast and Reed (2019), the OECD's geography-of-demand analysis, and official
INE, AMP and IHRU statistical sources.
