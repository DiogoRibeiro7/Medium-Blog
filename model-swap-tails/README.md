# Swappable models, scored where the risk is: European, US and Chinese models in one day-ahead load pipeline

Companion code for a Medium article on building forecasting pipelines in which the models can be swapped, so that European models (Mistral's Timestral and Ministral, EuroLLM, NXAI's TiRex, The Forecasting Company's t0) can replace US and Chinese ones. The thesis is that **averages hide the risk**: models that look interchangeable on average error can differ a lot on the few days that carry most of the risk, and those days decide whether a swap is safe.

The project is repository-first. The article is written from the numbers this code produces.

## The pipeline

One pipeline with two model slots, run in nine EU countries (Portugal, Spain, France, Belgium, the Netherlands, Germany, Austria, Italy, Poland):

1. **Text step.** A language model reads a short calendar note for a day and labels it `workday`, `saturday_like` or `sunday_like`. The note says what holidays fall on that day and on the two days either side, nationwide or regional (with the regions), in the country's own language. The model has to supply what the calendar does not say: whether a Friday after a Thursday holiday is a bridge day, how much of Germany a holiday in six states takes out, how the days between Christmas and New Year behave.
2. **Forecasting step.** A time-series foundation model forecasts tomorrow's hourly load from the history up to 18:00 local time the day before, zero-shot, on CPU. Models that take known future covariates get the day type (and the temperature forecast) for the horizon.

Each slot is filled in turn by European, US and Chinese models. The same harness, the same inputs and the same scoring apply to all of them.

## Question and measures

For each forecaster:

- **WAPE**: sum of absolute errors over sum of load, over every scored hour of every country;
- **P95 and CVaR95 of daily WAPE**: the 95th percentile, and the mean over the worst 5% of country-days;
- **WAPE by kind of day**: ordinary workdays and weekends, national, regional and optional holidays, days next to a national holiday, 24 December to 1 January, extreme-temperature days, days with a clock change;
- **WAPE on informative days**: days on which demand behaved differently from what the weekday alone implies (see below).

Rank agreement (Kendall tau) between the ranking by WAPE and the rankings by CVaR95 and by holiday error measures how much the average hides. Intervals come from a circular moving-block bootstrap over dates (7-day blocks, 2,000 replications), drawn jointly for every country and model so that contrasts are paired.

For each language model: the share of days labelled as demand turned out, on all the days it was asked about and on the informative ones, with a cluster bootstrap over country-weeks, against two rules (the weekday alone; a national holiday flag).

The two steps together: for each forecaster with covariates, the error on informative days when the day type comes from nothing, from a national holiday flag, from each language model, and from the ex-post truth (an upper bound, not a forecast).

### The ex-post day type

A day's mean load is compared with the median of ordinary days (no holiday of any kind, not next to a national holiday, outside 20 December to 6 January) within 28 days on either side: the same weekday for a working day, Saturdays, and Sundays. The label is the nearest of the three levels. It uses load after the day, so it is used only to score and for the oracle variant. A day is **informative** when its label differs from the weekday's (a Wednesday that behaved like a Sunday, a Friday like a Saturday).

## Models

Forecasters:

| Key | Model | Developer | Origin | Weights licence | Covariates |
|---|---|---|---|---|---|
| `timestral_209m` | Timestral 209M | Mistral AI | Europe (France) | Modified MIT: not for companies above 20 million dollars of monthly revenue | yes |
| `t0_gamma` | t0-gamma | The Forecasting Company | Europe (Paris; Y Combinator-backed) | recorded at run time from the model card | yes |
| `tirex` | TiRex (35M) | NXAI | Europe (Austria) | NXAI Community License | no |
| `chronos2` | Chronos-2 (120M) | Amazon | United States | Apache-2.0 | yes |
| `timesfm25` | TimesFM 2.5 (200M) | Google | United States | Apache-2.0 | no |
| `sundial` | Sundial base (128M) | Tsinghua University | China | Apache-2.0 | no |
| `similar_day` | Most recent day of the same type | – | reference | – | yes |
| `seasonal_naive` | Same hour one week earlier | – | reference | – | no |

The operators' own day-ahead load forecasts, as published on Energy-Charts, are scored as the reference where they cover at least 90% of the scored days.

Language models (served by [Ollama](https://ollama.com), quantised builds (mostly 4-bit; the exact format is recorded), CPU, temperature 0, fixed seed, output constrained to a JSON schema with the three labels):

| Key | Model | Developer | Origin |
|---|---|---|---|
| `ministral3_8b` | Ministral 3 8B | Mistral AI | Europe (France) |
| `eurollm_9b` | EuroLLM 9B Instruct | EuroLLM consortium | Europe (EU) |
| `gemma4_12b` | Gemma 4 12B | Google | United States |
| `llama31_8b` | Llama 3.1 8B | Meta | United States |
| `qwen35_9b` | Qwen3.5 9B | Alibaba | China |
| `qwen3_8b` | Qwen3 8B | Alibaba | China |

Origin is where the developer is based. It is the dimension the article is about, not a judgement on quality. Each run records the exact weights revision and declared licence of every forecaster (Hugging Face) and the digest and licence text of every language model (Ollama), in `results/run_manifest.json`.

Departures from each model's defaults, applied to all:

- **Context.** Up to 8,760 hours (a year, so the context holds last year's holidays), capped at each model's documented maximum: 2,048 for TiRex (its training context), 2,880 for Sundial, 8,192 for Chronos-2.
- **Hourly data.** Energy-Charts returns 15-minute load for some countries and hourly for others; everything is averaged to hourly so that the countries compare.

## Data

Downloaded on every run and never committed. The sources are registered as external references in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data) (`energy-charts-electricity-load`, `open-meteo-previous-runs-forecasts`).

- **Load**: Energy-Charts (Fraunhofer ISE) `public_power`, series `Load`, per country. CC BY 4.0, attribution energy-charts.info.
- **Operators' day-ahead load forecasts**: Energy-Charts `public_power_forecast`, `production_type=load`, `forecast_type=day-ahead`. Same licence.
- **Temperature**: Open-Meteo Previous Runs API, `temperature_2m_previous_day2` (the forecast made two days earlier, so that every value was available before the 18:00 cutoff), equal-weight mean over three to six large cities per country. CC BY 4.0, attribution Open-Meteo.com. The API keeps data from January 2024.
- **Holidays**: the `holidays` package (0.106), national, subdivision and optional or bank categories, in each country's language. Islands with their own grids (Azores, Madeira, Canary Islands) and overseas territories are excluded. Italy's provincial patron-saint holidays are kept for the ten largest provincial capitals only.
- **Masked hours**: the Iberian blackout of 28 April 2025 (Portugal and Spain). Never scored; inside a context window replaced by the same hour a week earlier.

`data/manifest.json` records every request and the SHA-256 of every cached file. The run copies it into `results/run_manifest.json`.

## Running it

### On GitHub Actions

**Model swap tails run** (`.github/workflows/model-swap-tails-run.yml`) is started by hand from the Actions tab. It:

1. plans the jobs from `config/experiment.toml`;
2. downloads the data and checks missing and anomalous hours;
3. asks each language model about every candidate day, one job per model and year, with Ollama on the runner (answers already committed in `labels/` are reused);
4. runs each forecaster in its own environment (Sundial needs an older `transformers`), one job per country, after a smoke test on three days;
5. writes the tables, figures and checks, and pushes `results/`, `figures/` and `labels/` to the branch `automation/model-swap-tails-run`.

The inputs restrict any step to some models or countries. If a model's Hugging Face repository asks you to accept its terms, accept them and add an `HF_TOKEN` repository secret. Full forecasts stay in the run's artifacts for 90 days.

**Model swap tails CI** runs the lint and the tests (synthetic data, no downloads) on every change to the folder.

### Locally

```bash
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,timestral,chronos2]"   # the extras for the models you want
swaptails fetch                                        # about 25 minutes: Energy-Charts allows 2 requests a minute
swaptails validate
swaptails notes --countries de --show 3                # what the language models read
ollama pull ministral-3:8b && swaptails labels --llm ministral3_8b
swaptails forecast --model timestral_209m
swaptails summarize && swaptails figures && swaptails check --lenient
```

`swaptails synth` writes synthetic data in place of the downloads, for a dry run without network access. Any OpenAI-compatible server can stand in for Ollama (`backend = "openai"` and `base_url` in the config).

## Outputs

- `results/summary.json`: every number the article uses.
- `results/RESULTS.md`: the main tables.
- `results/run_manifest.json`: data requests and hashes, model revisions and licences, language-model digests, timings.
- `results/figure_data.json`, `figures/*.png`: the figures and the data behind them.
- `labels/<model>.jsonl`: every language-model answer, with the note's hash, the raw output and the latency.

## Limits

- **One year, nine countries.** 2025 for the forecasts, 2024 and 2025 for the labels.
- **Contamination.** None of the forecasters publishes its training data; 2025 load may be in it. The comparison is between models, all exposed to the same risk.
- **Small language models only.** 8–12 billion parameters, quantised, on CPU, which is what a pipeline without a GPU or an external API can run. Larger models of the same families may do better.
- **The ex-post label is coarse.** Three levels compared on daily mean load; partial reductions (a regional holiday covering a third of a country) fall to the nearest level.
- **No retraining.** Every model is used zero-shot, as released.

## Licence

Code: Apache-2.0. Data: Energy-Charts (CC BY 4.0) and Open-Meteo (CC BY 4.0). Model weights are not redistributed and keep their own licences.
