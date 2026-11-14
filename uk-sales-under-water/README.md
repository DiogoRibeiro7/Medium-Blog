# Owners under water and the fall in house sales: postcode districts in England and Wales

Computational companion to a Medium article, the third follow-up to `uk-repeat-sales-rigidity`. That analysis found a floor at the owner's nominal purchase price and suggested, without testing it, that owners below their purchase price sell less often, which would make sales fall further when nominal prices fall. This folder tests that with every standard sale in England and Wales since 1995.

Every standard sale since 1995 starts an ownership spell at its address; the next standard sale of the same address ends it. Each year from 2000 to 2025 (and each month from 2005), the owners whose spell is running form the stock, and the resale rate is the share of them who sell. Each owner's position is m, the change in the repeat-sales index of their postcode area since they bought: the nominal gain the index implies.

- **Timing**: the monthly resale rate halved between the summer of 2007 and the first quarter of 2008, when 4% of owners were below their purchase price. The share under water peaked near 40% in 2009, while the resale rate was already recovering.
- **Lock-in curve**: within the same postcode area and year, and for owners with the same years since purchase, type and new-build status, the probability of reselling falls smoothly as m falls, with no step at zero. In 2008–2012, owners 10% under water sold about 16% less often than owners at 0 to +2%, and owners 20% under water about 37% less.
- **How much it mattered**: if owners below their purchase price had sold like owners just above it, 3% of the 2007–2009 fall would have been avoided. A wider version, owners below +10% selling like owners at +10–20%, adds 9–12% to the resale rate in 2009–2012.
- **Districts**: comparisons of postcode districts within the same area change sign with the controls and fail placebo tests; they cannot separate the position of owners from the composition and earlier boom of each district.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --ppd /scratch/ppd                                             # 5.3 GB, not committed
python build.py --ppd /scratch/ppd/pp-complete.csv --out data --temp /scratch/duck
python analysis.py        # results/*.csv and results/summary.txt (every number in the article)
python figures.py
python -m pytest -q tests  # synthetic sales with a planted lock-in
```

`.github/workflows/uk-sales-under-water-data.yml` runs the download and the build on GitHub's runners and commits the aggregates in `data/`. The analysis runs from those aggregates in under a minute.

## Outputs

- `data/hazard_cells.csv.gz`: stock and resales by postcode area, year, whole years since purchase, 2% band of m, flat or house, new build or not.
- `data/district_year.csv.gz`: stock and resales by postcode district and year, split by band of m and of years since purchase.
- `data/monthly_position.csv`: monthly stock and resales by region, band of m and holding band.
- `data/area_index.csv`, `data/national_monthly.csv`, `data/build_log.txt`.
- `results/lock_in_curve.csv`, `results/counterfactual*.csv`, `results/district_tests.csv`, `results/district_binned.csv`, `results/monthly_series.csv`, `results/summary.txt`.

## Data

**HM Land Registry Price Paid Data**, complete file. Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the Open Government Licence v3.0. The file carries full addresses, so it is never committed; only the aggregates in `data/` are.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt
