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
