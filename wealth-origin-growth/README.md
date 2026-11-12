# Inherited and self-made billionaire wealth and growth within countries

Computational companion to a Medium article, and the sequel to `wealth-concentration-misallocation`. That project argued that concentrated wealth is inefficient to the extent that it is decoupled from the ability to use it, and found that inherited billionaire wealth predicted slower growth than self-made wealth across countries in all 24 variants it tried, but never significantly, with only three Forbes waves (1996, 2001, 2014). This project asks the same question with annual Forbes lists and within countries.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # run by the data workflow; see below
python analysis.py           # results/*.csv and results/summary.txt (about 1.5 minutes)
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. `.github/workflows/wealth-origin-growth.yml` reruns the analysis and fails if that file changes.

## Data (`fetch.py`, `data/`)

- **Forbes annual billionaire lists, 2002–2026**, from the Forbes person-list API (`/forbesapi/person/billionaires/{year}/position/true.json`). The script aggregates every list to citizenship × year *before* anything is written: number of billionaires and summed net worth, split by Forbes' `selfMade` flag and by its 1–10 `selfMadeRank` score. No person-level row is saved. `data/forbes_coverage.csv` records, per year, the HTTP status, list size, field names and the share of people with each field filled.
- **World Inequality Database**, Penn World Table 11.0 and **World Bank WDI**, as in the first project (WID wealth and income shares with per-observation quality scores; PWT growth, TFP and income levels; GDP in current dollars and country codes).

The workspace where the analysis was written cannot reach these hosts, so `.github/workflows/wealth-origin-growth-data.yml` runs `fetch.py` on GitHub's runners and commits the tidy tables; `data/fetch_log.txt` holds URL, status, size and SHA-256 of every response. Two Forbes years (2021, 2026) timed out on the first run and came through on a second.

**What the Forbes API can and cannot be used for.** The lists it returns before 2010 are incomplete: 333 people for 2002 against 497 on the published list, 908 for 2008 against 1,125. The `selfMade` flag is mostly empty before 2010 and, in 2010, is filled for everyone but true for only 2.2%, a default rather than a classification. From 2011 it is complete and between 65% and 72% of people are flagged self-made each year. The 1–10 self-made score covers about a fifth of people from 2015 and is not used. The analysis therefore uses the 2011–2023 lists (2024–2026 have no growth data yet). The flag is a property of each person's current profile, so it is the same for a person in every year.

## Analysis (`analysis.py`)

**Panel.** Countries with at least one million people in PWT 11.0 and a WDI GDP figure, list years 2011–2023: 1,926 country-years in 149 countries, 802 of them with at least one billionaire (75 countries). Inherited wealth is billionaire net worth with the self-made flag false; self-made wealth with the flag true; each is expressed as a percentage of the previous year's GDP in current dollars. Countries without billionaires enter with zeros. Citizenships outside the panel (Taiwan, Macao, Monaco, Liechtenstein, Cyprus and five smaller ones) hold 1.5% of listed wealth.

**Main test.** Annualised growth of real GDP per capita or TFP from t to t + h, regressed on inherited and on total billionaire wealth, so that the coefficient on inherited wealth is directly the difference between the inherited and the self-made coefficients. 144 fits: h ∈ {1, 3, 5} × {GDP per capita, TFP} × {log(1 + % of GDP), % of GDP} × {all countries; countries with billionaires in at least 7 of 13 years; all but financial centres} × {no control; income level at t} × {country and year fixed effects; year effects only}. Standard errors clustered by country (CR1).

**Joint test.** Simonsohn, Simmons and Nelson (2020): the estimated difference is removed from the outcome in every fit, countries are resampled jointly (one draw shared by all fits), and the 144 fits are re-estimated, 1,000 times; the observed median, counts of significant estimates and Stouffer's Z are compared with their null distributions, overall and by estimator, horizon and outcome.

**Placebo.** The same fixed-effects regression with past growth (t − h to t) as the outcome, and a country bootstrap of forward minus backward coefficients. If composition only tracked growth that had already happened, the two would look alike.

**WID check.** Conjecture 1 of the first article in its literal form: in country-years with a source-based WID top-1% wealth share (`data_quality > 0`), does the association between that share and subsequent growth depend on the inherited share of the country's billionaire wealth? 147 country-years in 26 countries, mostly 2011–2017.

## Caveats

- Billionaire wealth is a measure of the extreme top tail, valued at market prices, and moves with stock markets. Country and year effects absorb level differences and global cycles, not country-specific booms.
- Only about 7.5% of the variance of log(1 + inherited % of GDP) is within countries; the fixed-effects estimates rest on that.
- Growth windows overlap for h > 1; clustering by country accounts for the resulting serial correlation, not for everything else.
- Every result is an association. The placebo is a check on one alternative explanation, not an identification strategy.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository. Forbes, WID, PWT and WDI data are reused under their providers' terms.
