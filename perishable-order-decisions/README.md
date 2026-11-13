# Algorithms against the decisions around them: daily orders of perishable goods

Companion code for an article that tests a conjecture from operational forecasting: the choices made around a model
(what it is trained on, how a forecast becomes an order) move more value than the choice of algorithm. The test
uses FreshRetailNet-50K, 50,000 store-product series of a fresh-food retailer with hourly sales and hourly
out-of-stock status, and scores every order by the cost of waste and of lost sales.

`PLAN.md` is the pre-registration, committed before the data were downloaded.

## Files

- `PLAN.md`: the pre-registration.
- `fetch.py`: downloads the dataset at a recorded revision (`sources/REVISION`) and checks the files (`sources/FILES`).
- `feed/data.py`: the dataset as a series x day panel, with the structural checks.
- `feed/recover.py`: the coding of the hourly out-of-stock status, hourly profiles and recovered demand.
- `feed/forecast.py`: the four algorithms (two simple rules, two LightGBM models).
- `feed/decide.py`: training targets, order rules and the cost of an order.
- `evaluate.py`: the registered test; writes `results/`.
- `tests/`: unit tests on synthetic data with the dataset's column layout.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python fetch.py --data data --sources sources
python evaluate.py --data data --out results
```

`.github/workflows/perishable-order-decisions.yml` runs the same steps and commits the sources, results and
figures. The data (about 4.85 million rows) are not committed.

## Data

FreshRetailNet-50K, Dingdong-Inc, licensed CC BY 4.0: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K.
Wang, Y. et al. (2025), FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery
and Forecasting in Fresh Retail, arXiv:2505.16319.
