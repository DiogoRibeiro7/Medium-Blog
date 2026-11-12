# Does the agreement of two counts predict the smallest loss they can catch?

Companion code for a follow-up to Part 4 of *Code Quality for Data Work* (Level Up Coding). Part 4
(`../data-tests-limits`) ended with a conjecture: the 95th percentile of how far the logarithm of
delivered / outside count strays from its twelve-month median, on clean months, predicts the smallest random
loss of rows that a check against the outside count catches nine times in ten at a 5% alarm budget (about
twice that percentile). It was read off the four NYC TLC feeds, so it is tested here on four new pairs of
counts:

- Citi Bike trip files against the operator's monthly operating reports to NYC DOT;
- Citi Bike trip files against NYC DOT's quarterly table of trips by month;
- Chicago rideshare trips (Transportation Network Providers) against the trips summed over the vehicle
  records;
- the same trips against the trips summed over the driver records.

The hypotheses and every setting are in `PLAN.md`, committed before any count from these sources was fetched,
with one change made before any result was seen (recorded there).

## Files

- `PLAN.md`: the pre-registration.
- `feed/sources.py`: snapshot of the outside counts (operator reports, NYC DOT table, Chicago portal counts)
  and, before `PLAN.md` existed, of names, sizes and HTTP status only (`sources/METADATA.json`).
- `feed/citibike.py`: downloads the Citi Bike archives of a year and counts the rows by month of trip start.
- `feed/detect.py`: Part 4's detector (unchanged), the prediction rules and the loss grid.
- `evaluate.py`: the test; writes `results/summary.txt`, `results/pairs.json`, `results/curves.csv` and
  `results/ratios.csv`.
- `figures.py`, `figures/`: the figures of the article.
- `tests/`: unit tests, including one that reproduces two of Part 4's published detection rates.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m feed.sources --out sources
for y in $(seq 2013 2026); do python -m feed.citibike --year $y --bucket sources/citibike_bucket.json --out counts; done
python evaluate.py --counts counts --sources sources --part4 ../data-tests-limits --out results
python figures.py
```

`.github/workflows/second-count-prediction.yml` runs the counting as one job per year (at most four at once,
finished years cached so a rerun resumes), runs the test and the figures, and commits sources, counts,
results and figures.

## Data

Citi Bike System Data (<https://citibikenyc.com/system-data>), the Citi Bike monthly operating reports
(<https://citibikenyc.com/system-data/operating-reports>), NYC DOT's Bike Share Usage Data Report
(<https://home4.nyc.gov/html/dot/html/bicyclists/bikestats.shtml>), and the City of Chicago's Transportation
Network Providers datasets (trips m6dm-c72p, n26f-ihde and 6dvr-xwnh; vehicles bc6b-sq4u; drivers j6wf-834c;
<https://data.cityofchicago.org>). Each is used under its publisher's terms; the Citi Bike data under the
NYCBS Data Use Policy.
