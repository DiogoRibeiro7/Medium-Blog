# Is the purchase price a floor in France too? Repeat sales in the DVF files, 2021–2025

Computational companion to a Medium article. It replicates `uk-repeat-sales-rigidity` in a second country. That analysis found that in England and Wales about a quarter of the resales that would have landed just below the seller's purchase price are missing.

This folder asks the same question of French repeat sales from the open DVF files (*demandes de valeurs foncières*), with the same estimator, and puts the two countries side by side. The comparison uses the same resale years (2021–2025) and the same holding periods (under five years).

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data                       # French monthly CPI (OECD), committed
python fetch.py --out data --dvf /scratch/dvf    # also the geolocated DVF files, not committed
python build.py --dvf /scratch/dvf --cpi data/fr_cpi_monthly.csv --out data --temp /scratch/duck
python analysis.py                               # results/*.csv and results/summary.txt
python figures.py                                # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`analysis.py` also reads the committed aggregates of `../uk-repeat-sales-rigidity/data` (England and Wales) and `../housing-losses-inflation/data/oecd_house_prices.csv` (OECD house price indices).

The heavy steps run on GitHub's runners:

- `.github/workflows/france-repeat-sales-floor-data.yml` downloads the DVF files, builds the aggregates and commits them. The five yearly files are about 540 MB compressed, and the build takes a few minutes.
- `.github/workflows/france-repeat-sales-floor.yml` reruns the tests and the analysis, and fails if `results/summary.txt` changes.

## Data

- **DVF, geolocated version** (Etalab, data.gouv.fr), from the DGFiP land registry files. It holds every sale of land and buildings registered in France except in Alsace-Moselle and Mayotte. The DGFiP publishes a rolling window of about five years. At the time of the run, the geolocated files covered January 2021 to December 2025; the first half of 2026 existed only in the raw DGFiP format and is not used.
  - Licence Ouverte / Open Licence 2.0; source DGFiP.
  - The DGFiP terms forbid re-identifying the parties, and the files carry addresses and cadastral parcel numbers. They are therefore never committed: `build.py` reads them on the runner and writes only aggregate counts.
- **French consumer price index**, monthly, 2015 = 100, OECD consumer price dataset.

`data/fetch_log.txt` records every URL, its HTTP status, size and SHA-256, and the dataset's last-modified date on data.gouv.fr.

## Repeat sales (`build.py`)

DVF has one row per combination of deed, parcel, building unit and land-use type, with the deed's price repeated on each row. The script collapses rows to deeds. It keeps deeds that transfer exactly one dwelling, under one disposition and one price:

- one house or one flat;
- any number of outbuildings, cellars or parking spaces;
- no commercial unit.

The property key is:

- for a house, the set of cadastral parcels sold;
- for a flat, its parcel and the set of co-ownership lot numbers sold (the flat and its annexes). Flats with more than five lots are dropped, because DVF lists only five.

A key sold twice on the same day is dropped as ambiguous.

Consecutive deeds with the same key form a pair. The chain includes every kind of deed, so an exchange or an auction between two sales breaks the pair rather than being skipped. The main sample keeps a pair when:

- both deeds are ordinary sales (*Vente*). Pairs whose first sale was off plan (*VEFA*, new builds) or that involve an auction (*Adjudication*) are kept apart;
- the dwelling type and number of main rooms are the same at both sales, and the built surface (and, for houses, the land area) differs by less than 10%. The England and Wales data cannot make this check; here it removes 8.8% of the pairs, mostly extensions;
- the two sales are at least 183 days apart and both prices are at least €10,000;
- the absolute log price change is below log(10), and below 0.5 per year held.

Main sample: 166,780 repeat sales resold between July 2021 and December 2025.

Outputs in `data/` (counts only):

- `pairs_hist.csv`: histograms of the log price change in 0.5% bins, by resale year, region, dwelling type and holding band. Kinds: nominal; real; real without exact resales; and real for pairs with at least 10% cumulative inflation (the real-zero placebo).
- `pairs_cells.csv`, `pairs_diffs.csv`: losses, exact resales and round euro differences.
- `dep_hist.csv`, `dep_cells.csv`: the same by département.
- `pairs_qq.csv`: pairs by quarter of purchase and resale.
- `pairs_monthly.csv`, `sales_monthly.csv`, `price_m2_quarterly.csv`: monthly counts, and the median price per m² of all single-dwelling sales.

Regions:

- Paris (75);
- the inner ring (92, 93, 94);
- the rest of Île-de-France (77, 78, 91, 95);
- nine départements with a large city (06, 13, 31, 33, 34, 35, 44, 59, 69);
- the overseas départements;
- the rest of France.

## Estimators (`analysis.py`)

1. **The estimator of the England and Wales analysis, unchanged.** Weighted least squares of the log count on the bin centre, on each side of zero within 5%, with resales at exactly the old price removed from the first bin above zero. The missing share is 1 − e^(−θ).
2. **A donut version.** The bins within 1% of zero are left out on both sides. In both countries, resales at exactly the old price draw mass from both sides of it, so this version does not need to know where the exact resales came from.
3. **A scan** of the donut estimator over cutoffs from −15% to +15%. It tests whether French sellers anchor on the price plus the notary fees (about 7–8% of the price of an older dwelling), and how the jump at zero compares with the jumps that round price ratios produce elsewhere.
4. **Region-years** in both countries. A weighted regression of the jump on local market strength (the median nominal change of 2–5-year holds, or the share of resales at a nominal loss), with a France indicator.
5. **A bunching estimator.** A degree-7 polynomial fitted outside −10% to +2% gives a counterfactual density. The missing mass below the old price is compared with the excess at it.
6. **Simulations** of each estimator under three cases: a floor that converts would-be losses into sales at the old price, a floor that keeps them off the market, and rounding to the old price.

`tests/test_analysis.py` checks the estimators on these simulations. `tests/test_build.py` checks the pair construction on synthetic DVF rows (the real column layout, invented sales).

## Caveats

- Five years of data allow only short holds, so the comparison with England and Wales is restricted to holds under five years.
- The DVF price is the price in the deed. Depending on the deed, it may leave out agency fees paid by the buyer and furniture sold with the dwelling, so part of the price change is measurement noise.
- Renovation without a change of surface is not visible. About a fifth of the pairs in both countries resold at least 30% (log) above the purchase price within five years. A repeat-sales index built from these pairs rises far faster than the official indices, so it is not used for the market path.
- Region-years for France exist only for 2023–2025, the years with enough holds of two years or more.

## Outputs

- `results/summary.txt`: every number used in the article.
- `results/losses_by_year.csv`, `exact_price.csv`, `jumps_by_group.csv`, `jump_sensitivity.csv`, `jump_scan.csv`, `region_years.csv`, `departement_quartiles.csv`, `near_zero.csv`, `bunching_observed.csv`, `bunching_grid.csv`, `simulations.csv`, `oecd_hpi.csv`, `volume_12m.csv`.
- `figures/fig1_market.png` … `fig5_worlds.png`.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Code: Apache-2.0, as the rest of the repository. The aggregate tables are derived from the DVF files (DGFiP, Licence Ouverte 2.0) and, for the comparison, from HM Land Registry Price Paid Data (Open Government Licence v3.0). OECD data are reused under the OECD's terms.
