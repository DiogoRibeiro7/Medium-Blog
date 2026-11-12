# Minimum wages, the middle of the distribution and in-work poverty in Portugal and the South

Computational companion to Part 3 of a Medium series on Europe's low-wage path. Question: when the statutory minimum wage rises faster than the rest of the distribution, does it become the wage that most of the bottom half is paid, and does it keep the people who earn it out of poverty? The repository compares the real growth of the minimum wage and of median earnings in nine euro-area countries, follows Portugal's minimum wage, average salary and poverty line to 2025, breaks down in-work poverty by household, contract, working time and employment status, and divides the net income of full-time minimum-wage households by the EU at-risk-of-poverty line.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Eurostat and OECD snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig4_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/minimum-wage-in-work-poverty.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

Seventeen requests made on 12 November 2026; `data/fetch_log.txt` records the URL, HTTP status, size and SHA-256 of each response, and the raw responses are kept gzipped in `data/raw/`.

Eurostat (JSON-stat dissemination API):

- `earn_mw_cur`: statutory monthly minimum wages, January and July, euro and national currency. For Portugal, Spain and Greece, where the minimum is paid fourteen times a year, Eurostat reports a twelve-month equivalent (× 14/12).
- `prc_hicp_aind`: harmonised consumer price index, annual average.
- `earn_ses_mdeci`, `earn_ses10_mdeci`, `earn_ses14_mdeci`, `earn_ses18_mdeci`, `earn_ses22_mdeci`: 10th percentile, median and 90th percentile of monthly earnings of full-time employees in October, Structure of Earnings Survey 2006–2022.
- `nama_10_fte`: average full-time adjusted salary per employee, national accounts, to 2025.
- `ilc_li01`: at-risk-of-poverty thresholds (60% of median equivalised disposable income) for one adult, two adults, and two adults with two children under 14.
- `ilc_iw01`, `ilc_iw02`, `ilc_iw03`, `ilc_iw05`, `ilc_iw07`: in-work at-risk-of-poverty rates by activity status, household type, work intensity, type of contract and working time.
- `ilc_li02`: at-risk-of-poverty rate of the whole population.

OECD Data Explorer (TaxBEN model):

- `DSD_TAXBEN_IMW@DF_IMW`: gross earnings and net family income of full-time minimum-wage earners by household type (children aged 4 and 6; social assistance claimed where eligible; no housing benefit; minimum wage averaged over the calendar year).
- `DSD_TAXBEN_HOURSPOV@DF_HOURSPOV`: weekly hours of work at the minimum wage needed to escape poverty, defined by the OECD as 50% of median disposable income, for a family claiming minimum-income benefits (no housing or temporary into-work benefits).

Eurostat and OECD data are reused under their terms; they are not relicensed by the repository's licence.

## Definitions

- **Real growth, 2006–2022.** Log change of the July minimum wage and of the SES percentiles, minus the log change of the national HICP. Restricted to the nine countries that were in the euro area and had a statutory minimum wage over the whole window (Portugal, Spain, Greece, France, Belgium, the Netherlands, Ireland, Luxembourg, Slovenia), so that euro amounts can be deflated with national prices. The SES 2022 deciles include enterprises with fewer than ten employees, the earlier waves do not; the median for 2006–2018 is reported as a check on comparable coverage.
- **Minimum as paid.** For Portugal, Spain and Greece the Eurostat value × 12/14, the monthly amount actually paid, which is what the SES October earnings measure.
- **Poverty line by income year.** EU-SILC measures income in the calendar year before the survey, so the threshold published for survey year t + 1 is compared with TaxBEN incomes of year t. The threshold for a single parent with two children is 1.6 × the one-adult threshold (modified OECD scale: 1 + 0.3 + 0.3); the published thresholds for two adults (1.5) and two adults with two children (2.1) are checked against the same scale.
- **Net income / poverty line.** TaxBEN net family income at a full-time minimum wage divided by the threshold for the same household. Above 1 the household is above the at-risk-of-poverty line. Households: single adult; single parent with two children; couple with one earner (the other adult unemployed without contributory benefits), without and with two children; couple with two minimum-wage earners and two children.
- **In-work at-risk-of-poverty rate.** Share of people aged 18–64 whose most frequent activity in the income year was employment and who live in a household below the 60% threshold. It is a household measure applied to individuals.

## Caveats

- TaxBEN households are stylised: full-time work all year, no other income, children aged 4 and 6, no housing benefit. Real households differ in hours, months worked, other income and benefit take-up.
- The cross-section of net income against the poverty line covers the 17 euro-area countries with a minimum wage in 2024, so that incomes and thresholds are in the same currency.
- In-work poverty rates come from EU-SILC samples; year-to-year changes of less than a point in a single country are within sampling noise.
- The SES coverage change in 2022 mostly affects the 10th percentile; the article uses the median and the minimum.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
