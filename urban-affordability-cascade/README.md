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
