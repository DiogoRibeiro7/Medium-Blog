# Forecasting after the cutoff

**Time-series foundation models against classical baselines on Portuguese electricity load published after the models were released**

Zero-shot forecasting results are usually reported on benchmarks whose data may have been part of the models' training corpora. This project evaluates three foundation models on a test window that none of them can have seen: hourly Portuguese electricity load from November 2025 onwards, after the release of the last of the three. It then asks whether the models' advantage over classical baselines in that window is the same as in earlier windows that they could have seen.

The project is repository-first. The article is written from the numbers this code produces, and nothing in it comes from elsewhere.

## Question and estimand

For each model *m* and window *W*, the main quantity is the ratio of total absolute error to that of the weekly seasonal naive forecast, over every scored origin in the window:

relative MAE(*m*, *W*) = Σ |error of *m*| / Σ |error of seasonal naive (week)|

The comparison of interest is how this ratio changes between two sets of windows:

- the windows every model could have seen (2022-23 and 2023-24, pooled);
- the window after every release (2025-26).

It is reported as a percentage change with a 95% block-bootstrap interval. The same is reported against MSTL + ETS, the classical model built for both the daily and the weekly cycle.

Two further results set the scale for that change:

- **Placebo contrast.** The same change between 2022-23 and 2023-24, two windows that every model could have seen. This is how much the ratio moves from one year to the next for reasons that have nothing to do with a release.
- **Block-length sensitivity.** The interval for the main contrast recomputed with bootstrap blocks of 7, 14, 28 and 56 days. Strongly persistent errors make short blocks too optimistic.

How to read the result:

- A change for the foundation models no larger than the classical baselines' change, or than the placebo change, means the earlier windows did not flatter them.
- A larger change is consistent with direct or indirect leakage. It is equally consistent with the series having changed in ways that suit some models more than others, so it is read alongside the window-by-window results and the error by lead day.

## Data

- **Series:** hourly load for Portugal (`Load` in the Energy-Charts `public_power` response), in MW, indexed by the start of each hour in UTC. If the API returns quarter-hourly values for part of the period, each hour is the mean of its four values; an hour with any value missing is treated as missing. The resolution of every request is recorded in the manifest.
- **Source:** [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE), token-free API. Licence CC BY 4.0, attribution "energy-charts.info".
- **Snapshot:** `data/pt_load_hourly.csv`, from 2022-01-01, with `data/snapshot_manifest.json` recording the requests, the download time and the SHA-256 of the file. The last three days before a download are dropped, so preliminary values are not scored.
- **Masked hours:** the Iberian blackout of 28 April 2025 (10:00–22:00 UTC). These hours are never scored. In a model's context window they are replaced by the same hour one week earlier.
- **Holidays:** Portuguese national public holidays from the `holidays` package (country `PT`, category public), in Lisbon local time.

## Models

| Name in results | Model | Weights | Released | Licence |
|---|---|---|---|---|
| `chronos2` | Chronos-2 (120M) | `amazon/chronos-2` | 20 Oct 2025 | Apache-2.0 |
| `chronos2_holidays` | Chronos-2 with a public-holiday flag as a known future covariate | same | | |
| `timesfm25` | TimesFM 2.5 (200M) | `google/timesfm-2.5-200m-pytorch` | 15 Sep 2025 | Apache-2.0 |
| `tirex` | TiRex (35M) | `NX-AI/TiRex` | 29 May 2025 | NXAI Community License |
| `seasonal_naive_week` | Seasonal naive, period 168 | | | reference |
| `seasonal_naive_day` | Seasonal naive, period 24 | | | |
| `ets` | AutoETS, season 24 | | | |
| `mstl_ets` | MSTL (24, 168) with an AutoETS trend | | | |

None of the three foundation models has a published training-data cutoff, so the release date is the upper bound used here.

Weights repositories can be updated after a release. Each run therefore records, in `results/run_manifest.json`:

- the exact revision of each model's weights;
- the SHA-256 of the weights file;
- the date of the last commit that changed any weights file in the repository.

The run stops if that date falls after the start of the 2025-26 window. The window-status table in `results/RESULTS.md` uses the later of the release date and that commit date. To reproduce a run exactly, copy the recorded revisions into `config/experiment.toml`.

The foundation models are used zero-shot on CPU. Two departures from each model's defaults apply to all of them:

- **Context length.** Every model gets the same 2,048-hour context, rather than its own default (8,192 hours for Chronos-2; the TimesFM README example uses 1,024). 2,048 is the context TiRex was trained with and is within the range of the other two.
- **Other settings.** These are the defaults. For TimesFM 2.5, they are the forecast settings shown in its README.

The classical models are refitted on every context with [statsforecast](https://github.com/Nixtla/statsforecast).

## Design

| | |
|---|---|
| Origins | every day at 00:00 UTC |
| Context | 2,048 hours (about 12 weeks), the same for every model |
| Horizon | 168 hours, scored as day-ahead (lead day 1) and week-ahead (lead days 1–7) |
| Windows | 2022-23, 2023-24, 2024-25 and 2025-26, each from 1 November to 30 September |
| Point forecast | the 0.5 quantile after the quantiles at each hour are sorted (for the classical models, whose intervals are symmetric, this is also the mean) |
| Quantiles | 0.1, 0.2, …, 0.9 |
| Metrics | MAE, RMSE, MAPE, bias, MASE (seasonal naive scale on the context), weighted quantile loss, 80% interval coverage |
| Uncertainty | circular moving-block bootstrap over origins, 14-day blocks (7–56 as a sensitivity check), 2,000 replications, models paired, windows resampled independently |

The windows cover the same calendar months, so a comparison between windows is not a comparison between seasons. An origin is dropped when its 168-hour forecast period contains a masked or missing hour; `results/origins.csv` lists every candidate origin and the reason for each exclusion.

## Running it

### On GitHub Actions

The workflow **Post-cutoff forecasting run** (`.github/workflows/post-cutoff-forecasting-run.yml`) is started by hand from the Actions tab. It takes the following steps:

1. downloads the series;
2. validates it;
3. smoke-tests each model on three origins;
4. runs the full experiment on CPU;
5. writes the tables and figures and runs the consistency checks;
6. opens a pull request with `data/`, `results/` and `figures/`.

The full quantile forecasts are kept as a run artifact for 90 days. If a model download is refused because its Hugging Face repository requires accepting a licence, accept it on Hugging Face and add a `HF_TOKEN` repository secret.

**Post-cutoff forecasting CI** runs the tests on synthetic data on every change to this folder. Once results are committed, it also re-checks them against the committed snapshot. A pull request opened by the run workflow does not start other workflows, so CI runs on the results after they are merged. The run workflow has already run the same checks before opening the pull request.

### Locally

Run from this folder:

```bash
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[models,dev]"
postcutoff fetch            # download the snapshot
postcutoff validate-data    # missing, masked and anomalous hours
postcutoff smoke            # each model on three origins
postcutoff all              # run, summarize, figures, check
```

`postcutoff run --models seasonal_naive_week chronos2` runs a subset; note that it rewrites `results/scores.csv` with that subset only.

Stored forecasts in `forecasts/` are reused when the model specification, the inputs, the package versions and this package's version are all unchanged. `--force` recomputes them.

## Outputs

| File | Contents |
|---|---|
| `results/scores.csv` | sums of errors per origin, model and lead day; every metric is computed from these |
| `results/summary.json` | all reported numbers with bootstrap intervals |
| `results/RESULTS.md` | the same, as tables |
| `results/metrics.csv` | accuracy by window, horizon and model |
| `results/origins.csv` | every candidate origin, scored or excluded, with the reason |
| `results/examples.csv` | full forecasts for the example weeks used in the figures |
| `results/run_manifest.json` | packages, weights revisions and commit dates, timings, data and config hashes |
| `results/pip-freeze.txt` | the exact environment of the GitHub Actions run |
| `figures/` | the article's figures |

## Checks

`postcutoff check` recomputes key numbers by a second route and fails on any disagreement:

- the seasonal naive errors, recomputed directly from the snapshot;
- every ratio and every change in the summary (both horizons, both references, MAE and WQL, main and placebo contrasts), recomputed from the score rows with plain sums;
- the time alignment of every model's stored forecasts: each median must be closest to the target at lag 0, not one or more hours off;
- that every model was scored on every origin;
- that the run used the snapshot on disk;
- that the weights revision of each foundation model was recorded.

## Limitations

- **Unknown training data.** A window before a release may or may not have been in a model's training data.
  - None of the training-data descriptions we could read names an ENTSO-E or Portuguese national load series. The full dataset table of the Chronos-2 paper could not be checked.
  - TiRex's corpus does include the UCI ElectricityLoadDiagrams20112014 dataset: 370 clients, 2011–2014, with timestamps in Portuguese time.
  - The comparison therefore measures whether being *possibly seen* made a difference, not how much was seen.
- **One series.** This is a single national series, so the results speak about this series. They are not a ranking of the models.
- **The series moves.** Rooftop solar, prices and weather differ between windows. Ratios to a baseline refitted on the same context absorb part of this, but not all of it. The placebo contrast shows how large such year-to-year changes are.
- **Intervals.** The intervals reflect sampling variation within each window, not the variation between years. If errors persist over many weeks, 14-day blocks understate the uncertainty, which is why the block-length sensitivity is reported.
- **Calendar effects.** The models receive no weather data. Only `chronos2_holidays` is told about holidays. It gets the flag as a category, so a holiday adds information only when the context contains one.

## Licence

Code: Apache-2.0, as the rest of the repository. Data: CC BY 4.0, energy-charts.info (Fraunhofer ISE). Model weights are downloaded at run time under their own licences and are not redistributed here.
