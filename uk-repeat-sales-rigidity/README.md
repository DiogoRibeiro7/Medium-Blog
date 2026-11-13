# Do sellers refuse nominal losses? Repeat sales in England and Wales, 1995–2026

Computational companion to a Medium article, and the follow-up to `housing-losses-inflation`. That analysis found that most real losses on housing never appear as a lower nominal price. It conjectured that nominal prices resist falling most when inflation is low but positive. This folder tests that conjecture on individual sales: every pair of consecutive standard sales of the same home in England and Wales since 1995, 11.5 million in the main sample.

Two signs would show the owner's purchase price acting as a floor:

- more resales at exactly the old price than at nearby round differences;
- a jump in the density of the price change at zero: fewer resales just below the old price than just above it.

The same estimator applied where nothing should happen is the placebo. That means the real zero (purchase price plus inflation) and nominal changes of ±7.5%, ±10% and ±15%.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data                      # UK monthly CPI (OECD), committed
python fetch.py --out data --ppd /scratch/ppd   # also the 5.3 GB Price Paid Data file, not committed
python build.py --ppd /scratch/ppd/pp-complete.csv --out data --temp /scratch/duck
python analysis.py                              # results/*.csv and results/summary.txt
python figures.py                               # figures/fig1_*.png ... fig4_*.png
python -m pytest -q tests
```

`.github/workflows/uk-repeat-sales-rigidity-data.yml` runs the download and the build on GitHub's runners and commits the aggregate tables. On those runners the 5.3 GB file downloads in under a minute and DuckDB builds the pairs in about thirty seconds. `.github/workflows/uk-repeat-sales-rigidity.yml` reruns the tests and the analysis, and fails if `results/summary.txt` changes.

## Data

- **HM Land Registry Price Paid Data**, complete file: every residential sale in England and Wales lodged for registration since January 1995. Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the Open Government Licence v3.0. The file carries full addresses, so it is never committed. `build.py` reads it on the runner and writes only aggregate counts; no address or individual transaction leaves the script.
- **UK consumer price index**, monthly, 2015 = 100, from the OECD's consumer price dataset. It is published to one decimal, so the real change of a pair is measured to about 0.1%.

## Pairs (`build.py`)

Only standard sales (PPD category A) of houses and flats are used. Category A excludes repossessions, buy-to-let purchases identified by a mortgage, and sales to companies throughout: before October 2013 those transactions were not in the data at all, and since then they are category B.

A home is identified by postcode, primary and secondary address and street. Consecutive sales of the same home form a pair, and a pair is kept when:

- property type and tenure are unchanged;
- the second sale is not a new build;
- the two sales are at least 183 days apart;
- both prices are at least £10,000;
- the absolute log change is below log(10), and below 0.5 per year held.

Pairs whose first sale was a new build (1.49 million) are set aside, because the new-build premium is lost at resale. The main sample has 11,541,235 pairs.

The script writes these aggregates to `data/`:

- `pairs_hist.csv`: histograms of the nominal log change, the real log change, and the real log change without exact resales, in 0.5% bins from −40% to +40%. They are split by year of resale, region (London, South, North and Midlands, Wales) and holding band.
- `pairs_cells.csv`: counts of nominal and real losses, hidden losses and resales at exactly the old price, with medians.
- `pairs_diffs.csv`: resales at exactly the old price ± £0, 1,000, 2,500, 5,000 and 10,000.
- `pairs_monthly.csv` and `sales_monthly.csv`: monthly counts.

Regions are approximate. London is the Greater London county. Wales is the Welsh counties or the CF, LL, LD, NP and SA postcode areas. "South" is the rest of the South East, East and South West by postcode area, and everything else is "North and Midlands".

## The estimator (`analysis.py`)

The estimator works on the histogram of log price changes, in 0.5% bins.

1. Within 5% of the cutoff, fit a straight line to the log count separately on each side, by weighted least squares with Poisson weights.
2. The jump θ is the right intercept minus the left intercept at the cutoff.
3. With exact old-price resales removed from the first bin above zero, 1 − e^(−θ) is the share of resales that would have landed just below the old price and are missing.

Simulations in `tests/test_analysis.py` check the estimator:

- it recovers a simulated anchoring probability to within a few points in weak, normal and strong markets;
- it finds nothing where nothing was simulated.

The real-zero placebo uses holdings of 5–20 years. In shorter holdings the real zero sits within a few percent of the nominal zero, so the two windows overlap.

The inflation test has two parts:

- **Year by year:** the missing share is compared across years of resale.
- **Region-years:** a weighted regression of θ on CPI inflation in the resale year and on the median nominal change of 2–5-year holdings, a measure of how strong the local market is. It includes region fixed effects, with standard errors clustered by year.

## Caveats

- Address matching is imperfect, and work done between two sales is not observed; extensions and conversions inflate the second price.
- The data are registrations. Recent months are incomplete until late registrations arrive, so 2026 is partial and volume in the last months is understated.
- Only 2022 and 2023 had inflation above 4%, so the test of the inflation conjecture rests on two years. The same two years also saw interest rates rise quickly, so inflation and a rate shock cannot be separated.
- The missing share measures the floor among homes that were sold. Owners who did not sell because they would have taken a loss are invisible here, except through total volume.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Code: Apache-2.0, as the rest of the repository. The aggregate tables are derived from HM Land Registry Price Paid Data under the Open Government Licence v3.0. OECD data are reused under the OECD's terms.

## References

- Genesove, D. and Mayer, C. (2001). Loss Aversion and Seller Behavior: Evidence from the Housing Market. *Quarterly Journal of Economics* 116(4), 1233–1260.
- Erlandsen, S. K. and Juelsrud, R. E. (2023). Downward nominal house price rigidity: Evidence from three centuries of data on housing transactions. Norges Bank Working Paper 1/2023.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics* 142(2), 698–714.
- HM Land Registry, Price Paid Data and its explanatory notes, gov.uk.
