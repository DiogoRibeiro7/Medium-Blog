# Shift constraints against forecast quality: staffing NYC 311

Companion code for an article that tests whether the shift rules or the forecast limit a call-centre roster more.
Hourly counts of agent-handled calls to New York City's 311 line (NYC Open Data, 311 Call Center Inquiry), 2016 to
2019, are forecast in four ways (two simple rules, a LightGBM model and the actual calls), turned into hourly
requirements with Erlang C at 311's own target (80% of calls answered within 30 seconds), and covered by rosters
under five shift regimes, each the cheapest roster from an integer programme. The cost of a roster is its excess and
shortfall of agent-hours against the requirement of the calls that actually arrived.

`PLAN.md` is the pre-registration, committed before any call count was downloaded.

## Results in one paragraph

With a perfect forecast, the shift rules alone force idle agent-hours worth 12.5% of the hours the calls needed with
8-hour shifts starting at any hour, 18.7% with weekly tours of five consecutive days, 48.9% with three fixed shift
starts, and 2.6% when a quarter of the hours can be 4-hour shifts. Under the two rigid regimes, that cost is larger
than what the forecast errors add (H1 supported, 8 of 8). The saving from a better forecast was smaller under weekly
tours than with hourly staffing, but established only when a shortfall weighs three times an idle hour (H2 not
supported). With 4-hour shifts available the forecast is the larger cost; with 8-hour shifts at any start it is not
(H3 not supported). The binding hours are 01:00, 09:00 and 17:00: eight hours apart, set by the shift length.

## Files

- `PLAN.md`: the pre-registration.
- `probe.py`: recorded the dataset's column schema (`sources/schema.json`) before the plan; no data values.
- `fetch.py`: counts the calls per clock hour on the portal, month by month (`sources/hourly_calls.csv`, with its
  SHA-256 in `sources/FILES`).
- `feed/calls.py`: the hourly grid, the test weeks and the structural checks.
- `feed/erlang.py`: Erlang C service level and staffing requirement.
- `feed/forecast.py`: the four forecasts.
- `feed/schedule.py`: the five shift regimes as integer programmes, and shadow prices.
- `evaluate.py`: the registered test; writes `results/` (`summary.txt` holds every registered number).
- `details.py`: descriptive analyses added after the results (`results/details.txt`); not part of the registered
  test.
- `figures.py`, `figures/`: the figures of the article.
- `tests/`: unit tests on synthetic data.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python fetch.py --sources sources
python evaluate.py --sources sources --out results
```

`.github/workflows/staffing-shift-constraints.yml` runs the same steps and commits the sources, results and figures.

## Data

NYC Open Data, 311 Call Center Inquiry (wewp-mm3p), https://data.cityofnewyork.us/d/wewp-mm3p. The dataset notes
that since March 2020 it does not fully represent agent-handled calls; the test uses 2016 to 2019.
