# Housing losses and inflation: how much of the loss never shows in the price

Computational companion to a Medium article. The question is how often owners lose money on housing and how they lose it. The analysis looks at the same houses through three lenses:

- **N**, the nominal price index: the number on the sign;
- **R**, the real price index: the nominal price deflated by consumer prices;
- **T**, the real total return: nominal capital gain plus net rental return, deflated.

A **hidden loss** is a real loss while the nominal index did not fall. The thesis tested is that most real losses on housing hide in inflation. When inflation is low, a loss has to appear as a lower nominal price, so it is visible.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # also run by the data workflow; see below
python analysis.py           # results/*.csv and results/summary.txt (a few seconds)
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. `.github/workflows/housing-losses-inflation.yml` reruns the analysis and fails if that file changes.

## Data (`fetch.py`, `data/`)

- **Jordà–Schularick–Taylor Macrohistory Database, release 6** (annual, 1870–2020, 18 countries). The script keeps only the columns used here: CPI, the Knoll–Schularick–Steger nominal house price index (`hpnom`), and the housing-returns series (`housing_capgain`, `housing_rent_rtn`, `housing_rent_yd`, `housing_tr`).
- **OECD analytical house price indicators** (quarterly, seasonally adjusted, 2015 = 100, 1970-Q1 to 2026-Q3). The series used are the nominal index (HPI), the real index (RHP, deflated by the private consumption deflator), the rent index (RPI) and the price-to-rent ratio. The analysis keeps the 37 OECD members that report the index.

macrohistory.net cannot be reached from the workspace where the analysis was written. The committed JST extract was therefore built from a copy of `JSTdatasetR6.csv` published in a public GitHub repository. Before it was used, it was checked against a second, independent public copy: all 2,718 rows match, with numeric differences below 1e-7. `data/fetch_log.txt` records each source with its size and SHA-256.

`.github/workflows/housing-losses-inflation-data.yml` runs `fetch.py` on GitHub's runners. There it downloads the official `JSTdatasetR6.xlsx` and a fresh OECD extract, then reruns the analysis and the figures, and commits whatever changed. The JST download is best effort, because macrohistory.net sometimes refuses scripted requests; if it fails, the committed extract is kept. The OECD request is required.

## Two price series inside JST

JST R6 carries two nominal house price series. They are identical for most countries but not for all. Country-years where they differ by more than 2 percentage points (`results/jst_price_series_gap.csv`):

- Germany, 34 (mostly the 1870s);
- Italy, 29 (1970s and 1980s);
- Portugal, 19 (1989–2013);
- Spain, 17 (1970s and 1980s);
- Japan, 4;
- Sweden, 2;
- Denmark, 1.

The analysis therefore uses two sets:

- the **price set** (18 countries) uses `hpnom` for N and R;
- the **returns set** (16 countries) uses the price series behind `housing_tr` for N, R and T. That way T − R is exactly the rental return.

Every holding-period and episode table is reported for both sets.

For Portugal the two series diverge badly. Between 1989 and 2013 the national index (`hpnom`, identical to the OECD index) rose 126% in total, while the returns series rose 408%. From 2014 the two coincide. The RoRE appendix (Jordà et al. 2019) lists a 2016 University of Bonn master's thesis on Portuguese house prices (Azevedo, *House Prices in Portugal, 1930 to 2015*) among its references. The Portugal series here therefore uses the JST returns series up to 1988 and the national index from 1989.

## Analysis (`analysis.py`)

**[1] Single years.** This section counts the country-years in which the nominal price fell, the real price fell, and the real price fell while the nominal price did not. Results are split by era and by that year's inflation.

**[2] Holding periods.** Every country and every purchase year is treated as a purchase. Horizons are 1, 5, 10 and 20 years for JST, and 1, 5 and 10 years for the OECD quarterly data. For each lens the share of purchases showing a loss is computed, along with the share of real losses that were hidden. These are split by era of purchase and by average inflation over the holding. Windows overlap. The 95% intervals resample countries with replacement (2,000 draws, seed 20261113).

**[3] Episodes.** An episode is a real price drawdown of at least 10%. It runs from a running real peak to the lowest point before that peak is regained, or to the end of the data. Sensitivity runs use 5% and 20%. The log real fall *L* splits into a visible part and a hidden part:

- the visible part is the fall of the nominal index from the real peak to the trough, capped at *L*;
- the hidden part is the rest, which inflation produced.

Each episode also records the largest nominal drawdown *inside* it, because a buyer at another date may have seen a nominal fall that the peak-to-trough split hides.

**[4] Inflation, visibility and speed.** This section reports the Spearman correlation between an episode's hidden share and inflation during the fall, with countries resampled for the interval. It also regresses log(years from peak to trough) on log real fall and annual log inflation, with SE clustered by country. The main specification sets aside two hyperinflation episodes (above 50% a year): Germany 1913–22 and Japan 1943–45.

**[5] Two busts.** OECD episodes with a real peak in 2005–2009 are compared with those peaking in 2020 or later. This section also gives, for every OECD member, the largest real fall from a post-2020 peak.

**[6] Rents.** The T lens is applied to holding periods and to drawdowns. Because later research finds JST's rental yields too high, the section also shows what happens when JST's net rental return is cut by 25% and by 50%:

- Chambers, Spaenjers and Steiner (2021) find a 2.3% real net return on UK residential property held by Oxbridge colleges, 1901–1983;
- Eichholtz, Korevaar, Lindenthal and Tallec (2021) find 4.0% for Paris and 4.8% for Amsterdam, lower than the returns JST implies.

**[7] 2021–2025.** JST ends in 2020. For the 16 returns countries the series is extended with OECD annual averages: price growth from HPI and inflation from the consumption deflator. The net yield is rolled forward from JST's 2020 yield with the OECD price-to-rent ratio, which is the rent–price method JST use. Australia and the United States have no published price-to-rent ratio for 2025 yet, so their last value is carried forward.

**[8] Portugal.** Annual series, 1948–2025, with two rent variants:

- **J** uses JST's rent return to 2020, rolled forward after that;
- **O** applies the rent–price method from 1989 to the OECD rent and price indices, anchored to JST's 2015 yield.

**[9] Gains.** For each country, the share of cumulative nominal appreciation that was consumer-price inflation.

## Caveats

- National indices average many transactions. An individual owner's gain or loss depends on the house, the city and the dates.
- JST deflates by the CPI; the OECD real index uses the private consumption deflator. The 2021–2025 extension switches from the first to the second.
- The lens is the asset, not the household. An owner with a fixed nominal mortgage passes part of a hidden real loss to the lender; taxes, transaction costs and leverage are not modelled.
- The visible/hidden split is measured from the real peak, which is the point of view of a buyer at the top. `max_nominal_drawdown_inside` gives the largest nominal fall anyone could have seen during the episode.
- Rental yields before recent decades are reconstructed. They come from a benchmark yield and rent and price indices, not from observed rents and prices of the same houses. For Portugal JST scale rental growth up by a factor of 1.2 over the rent-control years 1974–2005. The T lens is the least certain of the three.
- The speed regression is an association across episodes from different eras. It is consistent with nominal prices that adjust slowly, but it does not identify that mechanism.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Code: Apache-2.0, as the rest of the repository. The JST extract in `data/jst_housing.csv` keeps the JST licence (CC BY-NC-SA 4.0, macrohistory.net). OECD data are reused under the OECD's terms.

## References

- Jordà, Ò., Schularick, M. and Taylor, A. M. (2017). Macrofinancial History and the New Business Cycle Facts. *NBER Macroeconomics Annual* 31. Database: macrohistory.net, release 6.
- Jordà, Ò., Knoll, K., Kuvshinov, D., Schularick, M. and Taylor, A. M. (2019). The Rate of Return on Everything, 1870–2015. *Quarterly Journal of Economics* 134(3).
- Knoll, K., Schularick, M. and Steger, T. (2017). No Price Like Home: Global House Prices, 1870–2012. *American Economic Review* 107(2).
- Chambers, D., Spaenjers, C. and Steiner, E. (2021). The Rate of Return on Real Estate: Long-Run Micro-Level Evidence. *Review of Financial Studies* 34(8).
- Eichholtz, P., Korevaar, M., Lindenthal, T. and Tallec, R. (2021). The Total Return and Risk to Residential Real Estate. *Review of Financial Studies* 34(8).
- OECD, Analytical house price indicators (dataset DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES).
