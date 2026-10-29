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

The workflow `.github/workflows/portugal-housing-generations.yml` runs the tests
and rebuilds `results.json` on every pull request; the build fails if the
committed file differs. `fetch_mortgage_rates.py` is the only script that touches
the network. It is run by that workflow's manual `fetch-mortgage-rates` job,
dispatched on a branch other than `main`, which commits the dated response and
its retrieval record to that branch for review; `analysis.py` never downloads.

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
14. ECB MFI interest rate statistics (MIR), series
    `MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N`, *Cost of borrowing for households for
    house purchase*, Portugal, monthly, percent per annum (`UNIT=PCPA`), source
    agency code `PT2` (Banco de Portugal), January 2003 to August 2026:
    https://data-api.ecb.europa.eu/service/data/MIR/M.PT.B.A2C.AM.R.A.2250.EUR.N?format=csvdata
    Retrieved 2026-10-29T14:49:43Z by `fetch_mortgage_rates.py` on a GitHub
    Actions runner (response `Last-Modified: Thu, 01 Oct 2026 08:00:00 GMT`).
    The raw SDMX-CSV is stored byte for byte, with its original CRLF line
    endings (`.gitattributes` marks it `-text`). Next to it,
    `*.meta.json` records the URL, retrieval time, HTTP status, headers and
    SHA-256; `analysis.py` refuses to run if the file no longer matches.
    Definition, from the response's `TITLE_COMPL`: annualised agreed rate
    (AAR) / narrowly defined effective rate (NDER) on *new business* lending to
    households and NPISH for house purchase, in euro, by monetary financial
    institutions; the total across rate-fixation periods is weighted by new
    business volumes with a moving average. Every observation has status `A`
    (normal); the response carries no break flags. The ECB documents one change
    in content: from reference period December 2014 renegotiated loans are also
    reported separately (Regulation ECB/2013/34), and "new business" has always
    included renegotiations of existing loans, not only first-time borrowing:
    https://data.ecb.europa.eu/methodology/bank-interest-rates
15. Banco de Portugal, macroprudential recommendation on new consumer and
    housing credit (in force since July 2018; amended version applies to
    agreements from 1 April 2022): maximum original maturity of 40 years for
    borrowers aged 30 or under, 37 years from 31 to 35 and 35 years above 35, with
    average maturity converging to 30 years by the end of 2022. Used only to
    choose the loan terms below:
    https://www.bportugal.pt/sites/default/files/documents/2024-01/31012022_Recomendacao_versao_alteradora_EN.pdf
16. INE, EU-SILC quality report for Portugal (Eurostat metadata `ilc_simsilc_pt`):
    income refers to the calendar year *before* the survey; the 2024 survey
    introduced administrative tax data for contributory old-age pensions, a break
    that lowers pension values. The Eurostat `ilc_di03` snapshots flag only a
    2004 break (`b`):
    https://ec.europa.eu/eurostat/cache/metadata/en/ilc_simsilc_pt.htm

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

## Mortgage financing cost and the initial payment

`initial_payment_to_income_index_2015_100` turns the price/income index into the
first monthly instalment on a new loan relative to income. For calendar year
*t* and base year 2015:

`100 * (HPI[t] / HPI[2015]) * (a(r[t]) / a(r[2015])) / (Y[t+k] / Y[2015+k])`

where `a(r)` is the level-payment annuity factor for a loan at annual rate *r*
(compounded monthly) over 30 years, *r* is the unweighted mean of the twelve
monthly cost-of-borrowing values (2026 has eight months and gets no annual
value), and *Y* is median equivalised net income. Two alignments are reported:
`survey_year_alignment` (*k* = 0) pairs prices with the EU-SILC survey of the
same year, as the price/income index above does; and
`income_reference_year_alignment` (*k* = 1) pairs them with the next survey,
whose income refers to year *t*, so the last year it covers is 2024. With a
constant rate the index reduces exactly to the price/income index.
`initial_payment_to_income_index_term_sensitivity_2015_100` repeats the
income-reference-year version with 25-, 35- and 40-year terms.
`initial_payment_to_income_change_log_points` splits the change between two
years into price, financing-cost and income terms (100 × natural-log ratios,
which add up exactly: total = price + financing cost − income).

What the numbers show (2015 = 100; survey-year alignment first, then
income-reference-year alignment):

- The annual mean cost of borrowing fell from 2.38% in 2015 to 0.83% in 2021,
  rose to 4.13% in 2023 and was 3.94% in 2024 and 2.99% in 2025. It was 5.43%
  in 2008.
- From 2015 to 2021 the price/income index rose to 128.4. The initial-payment
  index ended at 103.7 (108.7): lower rates offset 21.4 log points of a 52.4
  log-point price rise.
- From 2021 to 2023 the payment index rose to 183.1 (178.2), +56.9 (+49.5) log
  points. The financing-cost term, +43.5, was larger than the price term, +19.8.
- 2023 is the highest year of the series under both alignments and every term
  tested, above 2008 (161.4; 165.2), the first year of the price index and the
  year with the highest annual cost of borrowing (5.43%). In 2025 the index was
  166.5, still 66% above 2015, with prices up 24.9 log points since 2023,
  financing cost down 14.2 and income up 20.2.
- Longer terms make the payment more sensitive to the rate. With a 40-year
  term the 2023 index is 188.4; with 25 years, 172.8. Whether 2024 lies above
  2008 depends on the term: it does at 25 and 30 years (161.6 against 157.1;
  166.2 against 165.2) and does not at 35 and 40 years (170.6 against 172.9;
  174.8 against 180.3).

What it is **not**:

- Not a repayment burden of any household. The loan is assumed proportional to
  the house price index (constant loan-to-value and dwelling quality); fees,
  insurance, taxes and the deposit are left out; the rate is an AAR/NDER, not the
  annual percentage rate of charge.
- Not the payment path of a cohort. On a variable-rate or short-fixation loan
  the instalment moves with market rates after the first reset; the index
  describes the cost of entry only, at the average rate of the year of entry.
- The rate covers renegotiated as well as first-time loans and is a national
  average over borrowers who obtained credit; it says nothing about those who
  were refused, and the 2018 macroprudential limits changed who qualifies.
- The decomposition is an accounting identity, not a counterfactual. Rates and
  house prices are jointly determined, so "prices without the rate fall" is not
  identified by these series; nor does the index measure an effect of interest
  rates on household formation, ownership or voting.
- Income is one national median for the whole population. The INE-documented
  2024 break in pension values affects it, the base years differ between the
  two alignments, and young buyers' incomes are not observed here.

## What the calculations support

- Nominal house-price-index changes, a relative house-price/income-growth
  proxy for overlapping years, and an initial-mortgage-payment/income index
  that adds the cost of borrowing on new loans; neither includes the deposit.
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
