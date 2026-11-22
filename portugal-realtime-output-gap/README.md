# Real-Time Output Gaps and Fiscal Rules: Portugal and the EU, 2011–2022

Reproduction code for the Medium article that follows [`portugal-fiscal-rules`](../portugal-fiscal-rules/). That piece assumed that real-time estimates of the business cycle miss half of each year's surprise, and found that the assumption decides how often cycle-based rules squeeze the budget in a downturn. This one measures the real-time errors. It compares seventeen archived releases of the European Commission's AMECO database with the current one, then replays the six rules with the errors it finds.

## Run

```bash
pip install -r requirements.txt
python fetch_data.py      # 17 archived releases + the current one into raw/, each checked against its SHA-256
python build_data.py      # data/vintages.csv (EU-27 vintage panel) and data/portugal.csv
python run_analysis.py    # results/results.json, results/summary.txt, figures/*.png (about 40 seconds)
python -m unittest discover -s tests -v
```

`raw/` and `data/` are rebuilt at every run and are not committed. CI (`.github/workflows/portugal-realtime-output-gap.yml`) lints, fetches with a cache, checks every digest, runs the tests, reruns the analysis and fails if `results/summary.txt` changes.

## Data

- **Archived releases.** Spring and autumn 2011–2018 and autumn 2021, from the legacy AMECO path. They are registered in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data) as `ec-ameco-vintages-2011-2021` and pinned in `data_sources/ec-ameco-vintages-2011-2021.json`. The 2019–2020, spring 2021 and 2022–2025 releases are not available on that path.
- **Current release.** Autumn 2026, the benchmark: `ec-ameco-autumn-2026`, pinned in `data_sources/ec-ameco-autumn-2026.json`.

The series, for the 27 member states, are:

- the output gap (`AVGDGP`);
- real and potential GDP (`OVGD`, `OVGDP`);
- the structural balance (`UBLGAPS`, from spring 2013);
- the cyclically adjusted balance and primary balance (`UBLGAP`, `UBLGBP`);
- the headline balance (`UBLGE`).

## What the code does

- **Errors by age** (`revisions.py`). The age of an estimate is the release year minus the target year: −1 is the autumn forecast a budget is built on, 0 the same-year estimate, 1 a year later, and so on. For each age, the error is the estimate minus the autumn 2026 estimate, for target years up to 2022.
- **Error process.** For ages −1 and 0, the error is regressed on the year's innovation in the current gap, measured with each country's own persistence. The slope is the share of the surprise that the real-time estimate misses. The constant gives the bias, and the residual gives noise and its persistence. The output also includes:
  - the compression slope (estimate on outcome);
  - a 90% interval for the share missed, from a 2,000-draw bootstrap over countries;
  - checks without the autumn 2021 release, without Ireland, and with a pooled persistence.
- **Decomposition.** Revisions to the same-year estimate split into revisions to real growth and to potential growth.
- **Structural effort.** The year's change in the structural balance as estimated that autumn, against the current estimate. The output records how often the verdict on a 0.5-point improvement changes.
- **Rules replay** (`rules.py`, extended from `portugal-fiscal-rules`). The six rules, calibrated to an average primary surplus of 2.5% in 2028–2035 on the same 10,000 futures, under six measurement scenarios:
  - exact measurement;
  - the earlier assumption that half of each surprise is missed;
  - EU errors for the same-year estimate;
  - EU errors at budget time;
  - Portugal's errors at budget time;
  - EU budget-time errors halved.

  It also covers Portugal's bias when no one knows about it, and a sweep of the share missed.

## Outputs

- `results/results.json`: every number quoted in the article.
- `results/summary.txt`: rounded headline numbers, compared in CI.
- `figures/fig1_portugal_vintages.png`: Portugal's output gap in each autumn release.
- `figures/fig2_errors_by_age.png`: the EU distribution of errors by age, and Portugal's average.
- `figures/fig3_compression.png`: budget-time forecasts against current estimates.
- `figures/fig4_effort_verdicts.png`: real-time against current structural effort, and the 0.5-point verdict.
- `figures/fig5_rules_by_measurement.png`: the six rules under exact, Portuguese and EU measurement errors.
- `figures/fig6_unknown_bias.png`: what Portugal's bias does to effort and to the chance of 60% when nobody corrects it.

## Licence

Code: same licence as the repository. Data: European Commission, AMECO, re-used under Commission Decision 2011/833/EU with attribution.
