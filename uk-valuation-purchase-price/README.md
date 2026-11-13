# Does a valuation model need the price the owner paid? Repeat sales in England and Wales

Computational companion to a Medium article, and the follow-up to `uk-repeat-sales-rigidity`. That analysis found a floor at the owner's nominal purchase price: fewer resales just below it than just above, and more at exactly the old price. This folder asks what the floor means for a valuation model.

Every repeat sale in England and Wales since 1996 is valued at its sale date by three models, each trained only on earlier years and tested on three later periods (2008–2012, 2017–2019, 2022–2025):

- **M0, the indexed valuation**: the previous price carried forward by a monthly repeat-sales index of the home's postcode area;
- **M1**, a gradient-boosted correction to M0 that sees the indexed value, the home's type, tenure, holding period, area, its place in the area's price distribution and recent local performance, but not where the indexed value sits relative to the purchase price;
- **M2**, the same model plus that position: the nominal gain the index implies for the owner.

The models are compared on point accuracy, on the coverage of conformalised 80% intervals, and on the probability that a sale lands below, at or just above the purchase price.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --ppd /scratch/ppd                                          # 5.3 GB, not committed
python build.py --ppd /scratch/ppd/pp-complete.csv --pairs /scratch/pairs.parquet --out data --temp /scratch/duck
python model.py --pairs /scratch/pairs.parquet --data data --out results
python -m pytest -q tests
```

`.github/workflows/uk-valuation-purchase-price-data.yml` runs the download, the build and the models on GitHub's runners and commits only aggregates: the area index, and the tables under `results/`. `.github/workflows/uk-valuation-purchase-price.yml` reruns the tests on synthetic data with a planted floor.

## Data

**HM Land Registry Price Paid Data**, complete file. Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the Open Government Licence v3.0. The file carries full addresses, so it is never committed, and neither are the pairs built from it.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt
