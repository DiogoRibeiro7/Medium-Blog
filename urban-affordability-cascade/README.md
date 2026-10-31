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

The intended next stage is an empirical study of actual residential mobility,
spatial connectivity, housing stock, and transport burden. It is described in
[the research protocol](RESEARCH_PROTOCOL.md). There are currently **no**
observed origin–destination household flow data, estimated spatial propagation
coefficients, or causal treatment effects in this project.

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

The 13 unit tests check provenance-specific reference values, missingness,
duplicate rejection, numerical growth, conservation, parameter bounds,
comparative statics, costs and reproducible outputs.

## Boundaries before publication

- Prices of sold homes are not quality-adjusted repeated-sales observations.
- New-contract rents are not the average rents paid by incumbent tenants.
- Municipal population counts cannot identify domestic relocation origins.
- The fixed geographic AMP perimeter does not equal a measured commuting
  or housing-search market.
- A descriptive spatial lag or temporal ordering is not a causal spillover.
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
