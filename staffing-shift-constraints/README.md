# Shift constraints against forecast quality: staffing NYC 311

Companion code for an article that tests whether the shift rules or the forecast limit a call-centre roster more.
Hourly counts of agent-handled calls to New York City's 311 line (NYC Open Data, 311 Call Center Inquiry), 2016 to
2019, are forecast in four ways (two simple rules, a LightGBM model and the actual calls), turned into hourly
requirements with Erlang C at 311's own target (80% of calls answered within 30 seconds), and covered by rosters
under five shift regimes, each the cheapest roster from an integer programme. The cost of a roster is its excess and
shortfall of agent-hours against the requirement of the calls that actually arrived.

`PLAN.md` is the pre-registration, committed before any call count was downloaded.

## Files

- `PLAN.md`: the pre-registration.
- `probe.py`: recorded the dataset's column schema (`sources/schema.json`) before the plan; no data values.
- `fetch.py`: counts the calls per clock hour on the portal, month by month (`sources/hourly_calls.csv`, with its
  SHA-256 in `sources/FILES`).
- `feed/calls.py`: the hourly grid, the test weeks and the structural checks.
- `feed/erlang.py`: Erlang C service level and staffing requirement.
- `feed/forecast.py`: the four forecasts.
- `feed/schedule.py`: the five shift regimes as integer programmes, and shadow prices.
- `evaluate.py`: the registered test; writes `results/`.
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
