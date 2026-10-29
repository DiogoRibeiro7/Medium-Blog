# Portugal housing, generations, and political claims

This is a reproducible *descriptive* check of selected assertions in José Jordão's
28 October 2026 article, **Habitação, Reforma ou Revolução**. It does not establish
that housing conditions cause any voting pattern or political instability.

## Run (Python 3.10+, standard library only)

```bash
cd portugal-housing-generations
python analysis.py
python -m unittest discover -p 'test_*.py'
```

`results.json` contains the annual HPI, income, normalized price/income index,
and arithmetic summaries. To refresh a
Eurostat source, download the URLs below and replace the dated snapshots; update
the filename and the script/test defaults as part of a reviewed change. The
snapshot is intentionally pinned to avoid silent changes when Eurostat revises
historical values.

## Sources and definitions

1. Eurostat `prc_hpi_a`: Portugal, purchase `TOTAL`, unit `I15_A_AVG`,
   frequency `A`, annual house price index, 2015=100. Source JSON:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hpi_a?geo=PT&purchase=TOTAL&unit=I15_A_AVG
   Accessed 2026-10-29; source `updated` timestamp is preserved in JSON.
2. Banco de Portugal, *Boletim Económico*, October 2023, p. 30:
   https://www.bportugal.pt/sites/default/files/anexos/pdf-boletim/be_out2023_p.pdf
   Owner-occupied-family share when the family representative is aged 25–34:
   birth cohorts 1977–1986: 61%; 1987–1996: 42%. These are rounded,
   published figures, **not individual homeownership rates**.
3. OECD, *Economic Surveys: Portugal 2026*, housing chapter:
   https://www.oecd.org/en/publications/oecd-economic-surveys-portugal-2026_025b3445-en/full-report/tackling-portugal-s-housing-affordability-challenge-promoting-sustainable-and-inclusive-housing_c920df36.html
   The 2021 dwelling stock: 12% vacant and 19% used as holiday/secondary homes.
   These categories do **not** indicate immediately available market supply.
4. Eurostat `ilc_di03`: Portugal, `statinfo=MED_EI`, `unit=EUR`,
   `age=TOTAL`, `sex=T`; annual median equivalised net income of the overall
   population, not young people's earnings:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_di03?geo=PT&age=TOTAL&sex=T&statinfo=MED_EI&unit=EUR
   Accessed 2026-10-29; dataset update timestamp preserved.
5. Eurostat `ilc_di03` with identical filters but age `Y18-24` and `Y25-49`,
   pinned as two additional JSON-stat source responses. The 25–49 bracket is
   wider than millennials, and neither series isolates wage-earners or homebuyers.
   Both are repeated age cross-sections, **not longitudinal birth cohorts**.
6. Eurostat `ilc_lvho07a`, `geo=PT`, `sex=T`, `rskpovth=TOTAL`,
   `unit=PC`, ages TOTAL, Y18-24 and Y25-29:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvho07a?geo=PT&sex=T&rskpovth=TOTAL&unit=PC
   This is the percentage of people in a household where net housing costs
   exceed 40% of disposable household income (net of housing allowances).
   Accessed 2026-10-29; raw age-category JSON-stat snapshot and metadata pinned.
7. Eurostat `ilc_lvps08`, people aged 18–34 living with parents, with
   selected groups 18–24, 18–34 and 25–34, sex total, percent:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvps08?geo=PT&sex=T&unit=PC
8. Eurostat `yth_demo_030`, estimated age when 50% have left the parental
   home, sex total:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/yth_demo_030?geo=PT&sex=T
   Both responses accessed 2026-10-29 and pinned with their source timestamps.
9. Eurostat `ilc_lvho02`, population shares by housing tenure, selected
   `geo=PT`, `hhcomp=TOTAL`, `rskpovth=TOTAL`, `unit=PC`:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvho02?geo=PT&rskpovth=TOTAL&hhcomp=TOTAL&unit=PC
   The source includes owner with/without outstanding housing loan and
   tenants paying market rent or reduced rent/occupying free. The percentages
   are of **people**, not dwellings or individual ownership titles.
10. Eurostat `sts_cobp_a`, annual unadjusted building permits for the *number
    of dwellings* (`indic_bt=BPRM_DW`) in residential buildings excluding
    residences for communities (`cpa2_1=CPA_F41001_X_410014`), measured in
    thousands (`unit=THS`):
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_cobp_a?geo=PT&indic_bt=BPRM_DW&cpa2_1=CPA_F41001_X_410014&s_adj=NSA&unit=THS
    Accessed 2026-10-29. The snapshot contains Eurostat preliminary (`p`)
    flags for 2022–2025; these are preserved in `results.json`.
11. INE, *Estatísticas das obras concluídas*, as republished by PORDATA,
    *Fogos concluídos em construções novas para habitação familiar: total e por
    tipologia do fogo*, nationwide annual total:
    https://www.pordata.pt/portugal/fogos+concluidos+em+construcoes+novas+para+habitacao+familiar+total+e+por+tipologia+do+fogo-189
    The 1995–2025 published total column was manually transcribed to the pinned
    CSV on 2026-10-29 because the INE JSON endpoint returned HTTP 503. This
    is not an original machine-generated INE export. PORDATA identifies 2021–2025
    as preliminary; those flags are retained in the CSV and output. INE
    metadata: https://ine.pt/bddXplorer/htdocs/minfo.jsp?lingua=PT&var_cd=0000076
12. Eurostat `lfst_hhnhtych`: total private households in Portugal,
    `phhcomp=TOTAL`, `agechild=TOTAL`, `n_child=TOTAL`, `unit=THS_HH`:
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfst_hhnhtych?geo=PT&unit=THS_HH&agechild=TOTAL&n_child=TOTAL&phhcomp=TOTAL
    These are household estimates from the Labour Force Survey, not an
    administrative register of dwelling addresses. Eurostat marks structural
    breaks in 2011 and 2021 (`b`). The dated source JSON retains these flags.
13. Eurostat `demo_pjan`: total usually resident population on 1 January,
    Portugal, all ages and both sexes, number of persons:
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_pjan?geo=PT&age=TOTAL&sex=T
    This is a date-specific demographic stock, not a calendar-year average.
    Both sources accessed 2026-10-29, with source update timestamps pinned.

## Relative price/income index

For each year with both series, compute

`100 * (HPI[year] / HPI[2015]) / (income[year] / income[2015])`.

This yields a *national relative price/income growth proxy* (2015=100), not
the number of median-income years required to purchase a typical home. It uses
the house-price transaction index, not a fixed dwelling's sale price, and
income from an equivalised household survey population, not borrower income.
The European Union Statistics on Income and Living Conditions income reference
period and the index's calendar year may differ; nominal index comparisons are
descriptive and should not be construed as an exact affordability measure.

`results.json` also records this index separately for the Eurostat 18–24 and
25–49 income groups, with each age group's own 2015 baseline. These index
levels should **not** be used to compare absolute affordability across ages:
the normalization erases their initial income levels, while housing needs,
household sizes, wealth, and borrowing access vary by group.

## Direct housing-cost stress measure

The `housing_overburden_pct_by_age` result adds a direct but different outcome.
It measures the share above a 40% housing-cost threshold *among all people*,
including young adults in their parents' homes. It is not a measure of the
financial barrier to leaving home or buying a first dwelling. Such selection
effects mean falling observed overburden rates can coexist with worsening
entry affordability and rising house prices. Do not equate the two trends or
infer a causal relationship from their aggregate difference.

## Leaving the parental home

`parental_coresidence_pct_by_age` records the repeated cross-sectional share
living with parents. `estimated_leaving_parental_home_age` is a model-derived
threshold (age when 50% no longer co-reside), not the mean age of individual
departure events. The 18–34 share was 62.9% in 2015, 69.8% in 2020, and
61.0% in 2025: it did not rise monotonically alongside house prices.

Neither measure identifies *why* someone remains with parents. Family norms,
education, partnership, employment and housing conditions can all matter.
These observations must not be used as proof that high prices caused delayed
emancipation or that housing access has become unproblematic.

## Tenure composition and aggregation

`population_tenure_pct` provides population shares by tenure status, with
internal sum checks allowing for published rounding. The 2015→2025 owner
share moved from 74.8% to 71.2%, while market-price renters increased from
12.7% to 16.3%. Owners with an outstanding mortgage fell from 36.5% to 31.5%
of the population; owners without one rose from 38.3% to 39.7%.

These are population *stock* categories: they cannot identify first-time
buyers, entry prices, or wealth transfers. Do not add parent categories to
their subcategories or assume they are a cohort transition matrix.

## New residential building permits

`dwellings_permitted_thousands` tracks dwellings covered by new-building
permits, not the number of permit documents or the number of homes completed.
The published series drops from 73.9 thousand dwellings in 2005 to 8.6
thousand in 2015, then rises to 40.9 thousand in 2025 (preliminary). The
2015–2025 change is about +376%, but 2015 is a depressed comparison year;
2025 remains well below the 2005 observation. Permits are neither a measure
of net additions to habitable stock nor of dwelling affordability. Annual
national permits must not be causally compared with house-price inflation:
there are completion lags, geographic differences, replacement and
demolition, and unobserved demand and financing conditions.

## Completed new family dwellings

`completed_new_family_dwellings` adds completed dwellings in *new construction*
from INE's annual works-completion statistics. Counts are units, not buildings;
they exclude rehabilitation and do not measure the net change in occupied
housing stock. They peaked at 125,708 in 2002, reached 7,148 in 2015,
and recovered to 27,301 in 2025 (preliminary). The 2015 trough makes the
2015–2025 percentage increase unusually sensitive to the baseline.

Permits and completions refer to different stages of construction, may be
revised independently, and do **not** form matched cohorts. Consequently,
dividing completions in year *t* by permits in year *t* is *not* a permit
conversion rate: a dwelling completed this year may have been permitted in
an earlier year. We report the two annual series separately and make no
causal or conversion-rate inference from their difference.

## Population, households and construction intensity

`completed_new_dwellings_construction_intensity` reports annual completed
new-family dwellings divided by two independently observed stocks: population
on 1 January and number of private households (both standardized per 1,000).
In 2015 there were 0.688 completions per 1,000 residents and 1.743 per
1,000 households; in 2025 the figures were 2.540 and 5.984, respectively.

The denominator measures need *imperfectly*. Household formation itself can
be suppressed by high housing costs, while additional demand can arise from
internal migration, location, second homes, the condition of existing homes,
and changes in household size. The private-household series is survey-based
and has Eurostat break flags in 2011 and 2021; its apparent jumps across
those years must not be interpreted as pure demographic changes. Nor should
the change in annual household stock be subtracted mechanically from annual
completions to declare a national housing shortage: the former is a stock
estimate and the latter a selected gross construction flow, and geographic
matching, replacement, vacancy, and conversion are unobserved.

## What the calculations support

- Nominal house-price-index changes and a relative house-price/income-growth
  proxy for overlapping years; no mortgage interest or down payment adjustment.
- A 19 percentage-point ownership gap between two cohorts at the same age.
- The stated breakdown of non-primary-residence dwellings in 2021.

## What they do not support

- A causal effect of speculation, immigration, or regulation on housing prices.
- A claim that young people reject democracy because of housing costs.
- The assertion that non-votes for PS/PSD indicate opposition to democracy.
- Aggregating these different vintages and populations into a single causal model.

Next research step: assemble comparable municipality/region-year price, income,
construction, credit and demographic panels with explicit source revisions; only
then specify a causal design and sensitivity checks. Electoral analysis additionally
requires individual-level evidence to avoid ecological inference.
