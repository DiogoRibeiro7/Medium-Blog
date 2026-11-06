# Research protocol: effective supply, external demand and housing prices

**Status:** exploratory design; not preregistered; no country-level causal effect has been estimated. Analysis vintage 6 November 2026.

## 1. Questions and target parameters

Primary question: what is the causal effect of an externally induced increase in *occupied or occupiable* long-term dwellings per private household on a local area's quality-adjusted real asking/sold prices and new-contract rents over years 1–5?

Secondary: does the effect vary with pre-shock investment demand and vacant-stock availability, and who experiences it (incumbent tenants, prospective buyers, low-income households, existing owners)? These mechanisms require **actual measured proxies**; birthplace of a buyer is not a valid direct indicator of speculative investment.

For geography `g`, let `S` be suitably usable long-term housing stock and `H` private households. A candidate treatment is a clearly measured, exogenously shifted `log(S/H)`. Outcomes are (a) real transaction-price HPI, (b) new-tenancy rent index, (c) household-specific housing cost burden and (d) mobility, each defined independently. A change in a national real HPI is not the same as a change in renter affordability.

## 2. Baseline data and boundaries

- Four independent country-year series 2015–2025: Portugal, Spain, France, Germany. EU27 remains a reference, not independent extra observations.
- Existing Eurostat numerator measures **permits**. Until comparable completions, demolitions and occupancy are acquired, this dataset is descriptive, not a treatment panel.
- Portugal Pordata new-build flow is not directly substitutable for a same-year building permit; it cannot be pasted into an international `permit` column.
- Bulgaria's annual completions and **Q2** annual HPI growth cannot be assigned a meaningful same-year treatment ordering; Q1 2022 changes the HPI data-source methodology.
- Investment demand must be separately operationalised, distinguishing non-resident tax status, property purpose, institutional ownership and cash-financed purchases. The 29% foreign-born 2024 category is not a direct investment variable.
- Effective vacancy requires repair status, availability/market status, quality, municipality and opportunity cost—not simply Census “vacant” categories.

## 3. Design requirements before estimating an effect

1. Define a geographic housing market and substitute area; test alternative boundaries (municipalities, travel-time catchments and functional urban areas).
2. Assemble at least several years **before** and after a plausibly external supply-policy change; document the rule and dates, land availability, expected exposure, construction lag and any anticipatory land/price effects.
3. Construct a comparison group with pre-trend evidence and event-study leads/lags. If matched localities diverged before the event, reject a simple parallel-trend interpretation.
4. Measure the *first stage* from the intervention into **completions and occupied long-term stock**, not merely permits.
5. For difference-in-differences, report modern staggered-adoption treatment effects instead of blindly pooling two-way fixed effects when treatment timing or effects differ.
6. For land/approval instruments, argue the exclusion restriction with direct local policy knowledge. A planning change may affect amenities, density, transport and neighbourhood expectations independently of completions.
7. Account for spatial spillovers: untreated neighbours may absorb movers or experience displaced demand, invalidating an untreated-control interpretation.
8. Predefine outcomes (sales, new rents, incumbent rents, household burdens) and separate nominal, CPI-deflated and income-adjusted comparisons.
9. Test heterogeneous effects by quantiles, investor exposure and actually measured vacancy. Do not use birthplace to infer motivation.
10. Publish identified confidence intervals with clustering appropriate to the actual treatment-assignment unit; report finite-sample robustness and placebo-event distributions.

## 4. Hypotheses to falsify

| Hypothesis | Observable implication | Important alternative |
| --- | --- | --- |
| Added effective dwellings moderate price pressure | Post-shock exposed areas show lower quality-adjusted growth than defensible controls | Supply was built where demand was already about to weaken |
| More idle but usable stock buffers demand | Larger post-shock moderation where genuinely rentable vacant stock exists | Vacant stock is spatially or quality mismatched |
| External investment demand modifies supply effects | Heterogeneity by **measured** nonresident/property-use demand | High-income resident household formation or local wage growth |
| Moving chains widen benefit | Registration-linked move chains reach lower-price submarkets | Relocation chains truncate, or filtering/amenity effects differ |

## 5. Negative controls and sensitivity

- Placebo outcome: comparable non-housing prices not expected to respond to additional homes in the same time frame.
- Placebo intervention dates and treated-place reassignments.
- Pre-trend and anticipation windows (construction announced years before completion).
- Exclude suspected methodology-break years and provisional construction observations.
- Compare local areas with and without strong travel-time connections to the new supply.
- Re-estimate with newly signed rent contracts, repeat-sale hedonic indices, and distributional rent burdens where definitions permit.
- Report results separately for market-rate and public/social-housing additions.
- Test attrition due to demolition, second homes and changing property use; never equate total gross completions with occupied net stock.

## 6. Possible future falsification of claims in the circulating post

The broad “only construction is insufficient” proposition is conceptually compatible with many empirical outcomes and needs a sharper operational target. The falsifiable claim would be: *In a given market and horizon, a particular verified increase in effective long-term stock had no price-moderating effect relative to its defensible counterfactual, after demand shocks and spillovers were accounted for*. The chart of aggregate completed homes versus the contemporaneous annual price-growth rate does not test it.

The alternative claim “any new home automatically causes absolute nominal prices to fall” is not a defensible baseline either. A valid design asks by how much observed prices or rents **differ from an identified no-intervention world**, not whether the observed price level falls below last year's level.
