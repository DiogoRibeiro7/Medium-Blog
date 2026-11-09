# Testing a machine learning pipeline end to end

Companion code for the Medium article *Testing a Machine Learning Pipeline End to End: Leakage, Splits and Train–Serve Skew in a Flight-Disruption Model*.

A gradient-boosted model predicts, two hours before the scheduled departure, whether a flight leaving New York in 2013 will be disrupted: cancelled, diverted, or more than 15 minutes late on arrival. The pipeline exists in a reviewed version and in six variants, each with one defect that real pipelines carry:

1. the flight's own departure delay used as a feature;
2. a congestion window anchored to the departure hour instead of the prediction time;
3. route and aircraft disruption rates computed from all labels before the split;
4. a shuffled split instead of a split in time;
5. cancelled and diverted flights dropped from the data;
6. the hour taken from the UTC timestamp while serving reads the local schedule.

Every variant is trained on January to August, tuned on September, tested on October and November and then judged on December with features computed the way a serving path can compute them. Eight test layers are pointed at every variant: data validation (pandera), unit tests on fixtures, a property-based test (Hypothesis), a future-perturbation test, a null-label permutation test, a split and protocol audit, a train/serve parity test against an independent per-flight serving path, and a gate on the reported AUC.

## Data

The `nycflights13` Python package (0.0.3, CC0) ships the CSV files of Hadley Wickham's R package: flights from the Bureau of Transportation Statistics, hourly weather, aircraft from the FAA registry. `mlp/data.py` reads the files from the installed package with declared types. Nothing is downloaded at run time and no data is committed.

## Layout

- `mlp/data.py`: loader, base table with prediction times (T = scheduled departure minus two hours) and labels.
- `mlp/features.py`: batch feature SQL; each defect is a flag on `Variant`.
- `mlp/serving.py`: per-flight serving path, every query bound to T.
- `mlp/model.py`: model, periods, protocol, serving-time frame for December, day-cluster bootstrap.
- `mlp/checks.py`: every test layer, written so it can be pointed at any variant.
- `tests/`: the pytest suite of the reviewed pipeline (12 tests, about 20 seconds on two cores).
- `analysis.py`: prints every number in the article to `results/output.txt`.
- `compare_output.py`: compares a fresh run with the committed one (counts exact, model outputs within two units of the last printed digit).
- `figures.py`: the five figures, from `results/figure_data.json` and the December predictions saved by `analysis.py`.

## Run

From this folder, with Python 3.13:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python analysis.py > results/output.txt        # about 6 minutes on two cores
python figures.py
```

`results/output.txt` is the run the article quotes. CI reruns the test suite and the analysis, compares the fresh output with the committed file and redraws the figures.

## Notes

- The model is scikit-learn's `HistGradientBoostingClassifier` with fixed hyperparameters and no early stopping. Its predictions depend on `random_state` because the binning step subsamples 200,000 of the 224,910 training rows; `analysis.py` reports how much.
- The serving path is deliberately separate code. It is slow (one set of queries per flight) and is only used on samples.
- The future-perturbation test rewrites, in a copy of the tables, every departure, arrival and weather observation that was not known at a cut-off, then rebuilds the features of the flights whose prediction time falls just before it.
- The wind-speed reading of 1,048.4 mph at Newark (2013-02-12 08:00 UTC) is treated as missing; the schema rejects wind speeds above 100 mph.

## License

Apache-2.0, as the rest of the repository. The data is CC0.
