# Lead time in import-dependence screening

Code and data for the article *Lead Time in Import-Dependence Screening: Forecasting Which EU Imports Will Come to Rely on One Outside Supplier, 2002–2025*.

The screen is the single-supplier variant of the European Commission's 2021 dependency screen: a product is flagged when its largest supplier outside the EU provides more than 40% of extra-EU imports, extra-EU imports are more than half of all EU imports, and extra-EU imports exceed total EU exports. This folder builds an annual panel of every HS 6-digit product from 2002 to 2025, describes how products enter the screen, and tests whether entries can be forecast from earlier data, out of time.

## Scripts

- `fetch_panel.py`: downloads the panel from Eurostat Comext (run by GitHub Actions, one range of years per job).
- `panel.py`: builds the panel, flags each product and year, dates durable entries (flagged in two consecutive years after three unflagged years) and traces the run-up to each entry.
- `warning.py`: the forecasts. For every product unflagged in t-2, t-1 and t, the target is a durable entry within three years. Five scores: a near-miss rule (no fitting), a logistic model on year-t levels, a logistic model with changes and supplier dynamics, gradient-boosted trees, and the boosted-tree probability times the value at stake. Rolling evaluation: the scores for year T come from models fitted on origins whose outcomes were known by T. Also: product-cluster bootstrap intervals, lead times before each entry, feature-group ablation and permutation importance, and a five-year horizon.
- `figures.py`: the nine figures.
- `check_results.py`: compares regenerated summaries with the committed ones within a small tolerance.
- `tests/`: the event definition, the eligibility and target windows, the out-of-time split and the near-miss rule.

## Data

Fetched by GitHub Actions (`.github/workflows/dependency-early-warning-data.yml`) into `data/parts/`, from Comext DS-045409 (EU trade since 1988 by HS2-4-6 and CN8), reporter EU27_2020:

- `totals_*.csv.gz`: imports and exports with partners outside (EXT_EU27_2020) and inside (INT_EU27_2020) the EU, every HS 6-digit code and year.
- `summary_*.csv.gz`: extra-EU imports in euros and in 100 kg, the Herfindahl index of outside suppliers (all partners) and the number of suppliers, by product and year.
- `partners_*.csv.gz`: the ten largest outside suppliers of each product and year, in value and quantity, and the remainder.

## Reproducing

```
python -m pip install -r requirements.txt
python -m pytest -q tests
python panel.py && python warning.py && python figures.py
```

`warning.py` takes about two minutes. Product labels come from `../europe-deindustrialisation/data/hs6_labels.csv`.

## Points to keep in mind

- Codes are used as reported. The Harmonized System was revised in 2007, 2012, 2017 and 2022, so some codes start or stop; a product-origin is scored only if its code exists from t-3 to t+4. 4,478 of the 6,421 codes exist in every year.
- The target is the screen as defined, flaws included. Total EU exports include intra-EU trade, so re-exports and production inside the EU can keep a product with a dominant outside supplier off the screen; most products on the watchlist that did not enter stayed dominated by one supplier.
- Imports are attributed to the country of origin; a component made in one country and assembled in another counts for the second.
- Durable entries can be dated from 2005 (three clean years are needed) to 2024 (the next year must be observed).
