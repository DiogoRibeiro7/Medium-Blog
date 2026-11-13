# Algorithms against the decisions around them: daily orders of perishable goods

Companion code for an article that tests a conjecture from operational forecasting: the choices made around a model
(what it is trained on, how a forecast becomes an order) move more value than the choice of algorithm. The test
uses FreshRetailNet-50K, 50,000 store-product series of a fresh-food retailer with hourly sales and hourly
out-of-stock status, and scores every order by the cost of waste and of lost sales.

`PLAN.md` is the pre-registration, committed before the data were downloaded.

## Results in one paragraph

With a symmetric cost (0.5), the algorithm explained 86% of the variation in cost over the 24 combinations of
algorithm and decision layer, and LightGBM was clearly cheapest. When a lost sale costs more than a wasted unit, the
decision layer took over: 68% of the variation at 0.7 and 98% at 0.9. The 4-week mean with the full layer (recovered
demand, per-series quantile) cost less than either LightGBM model with the default layer (sales, order the forecast)
at both ratios (H1 supported), and the layer explained more of the variation than the algorithm (H2 supported).
Training on recovered demand lowered cost in all 24 comparisons, but one interval touched zero (H3 not supported);
the algorithms ranked the same by cost and by WAPE under the full layer (H4 not supported). Most of the layer effect
is the order rule. Descriptive analyses added after the results show that the cost ratio 0.9 conclusions hold
however strongly demand is recovered, while those at 0.7 and the benefit of the recovered target under quantile
rules need a substantial part of the recovery to be right.

## Files

- `PLAN.md`: the pre-registration.
- `fetch.py`: downloads the dataset at a recorded revision (`sources/REVISION`) and checks the files (`sources/FILES`).
- `feed/data.py`: the dataset as a series x day panel, with the structural checks.
- `feed/recover.py`: the coding of the hourly out-of-stock status, hourly profiles and recovered demand.
- `feed/forecast.py`: the four algorithms (two simple rules, two LightGBM models).
- `feed/decide.py`: training targets, order rules and the cost of an order.
- `evaluate.py`: the registered test; writes `results/` (`summary.txt` holds every registered number).
- `details.py`: descriptive analyses added after the results (`results/details.txt`); not part of the registered
  test.
- `figures.py`, `figures/`: the figures of the article.
- `tests/`: unit tests on synthetic data with the dataset's column layout.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python fetch.py --data data --sources sources
python evaluate.py --data data --out results --orders data/orders.npz
python details.py --orders data/orders.npz --results results --out results/details.txt
python figures.py --results results --out figures
```

`.github/workflows/perishable-order-decisions.yml` runs the same steps and commits the sources, results and
figures. The data (about 4.85 million rows) are not committed.

## Data

FreshRetailNet-50K, Dingdong-Inc, licensed CC BY 4.0: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K.
Wang, Y. et al. (2025), FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery
and Forecasting in Fresh Retail, arXiv:2505.16319.
