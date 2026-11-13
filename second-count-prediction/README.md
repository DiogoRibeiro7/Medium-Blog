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

The hypotheses and every setting are in `PLAN.md`, committed before any count from these sources was fetched;
the changes made before any result was seen are recorded there.

## Results in one paragraph

As registered (`results/summary.txt`): H1 held in all four pairs and H2 in all four, but for the two Citi Bike
pairs only through the "never reached" clause, because the registered rule for duplicate CSVs counted some
months twice (2013, 2018: a month held both as one file and as parts) and dropped equal-size parts in others
(2019, 2024 to 2026). The two outside counts are what showed it. Post hoc (`results/corrected/`), with every
CSV recounted and a corrected layout rule (`feed/layout.py`), and everything else unchanged: the observed
smallest loss caught 90% of the time is within a factor of two of the p95 rule for Chicago vehicles (3.6%
predicted, 5.0% observed), Chicago drivers (0.73%, 0.79%) and Citi Bike against the operator's reports (14.8%,
25.1%), but not for Citi Bike against NYC DOT's table (10.7% predicted, 3.2% observed), whose large deviations
are confined to 2013-2015; the p95 rule is closer than the Gaussian rule in all four pairs.

## Files

- `PLAN.md`: the pre-registration.
- `feed/sources.py`: snapshot of the outside counts (operator reports, NYC DOT table, Chicago portal counts)
  and, before `PLAN.md` existed, of names, sizes and HTTP status only (`sources/METADATA.json`).
- `feed/citibike.py`: downloads the Citi Bike archives of a year and counts the rows by month of trip start.
- `feed/layout.py`: which CSVs of an archive make up a month: the registered rule and the corrected one.
- `feed/detect.py`: Part 4's detector (unchanged), the prediction rules and the loss grid.
- `evaluate.py`: the test; writes `results/summary.txt`, `results/pairs.json`, `results/curves.csv` and
  `results/ratios.csv` as registered, and the same files under `results/corrected/` for the post hoc counts.
- `diagnose.py`: post hoc, prints the sentence an operator-report total was read from
  (`results/diagnosis.txt`).
- `sources/FROZEN`: the outside counts are the registered run's snapshot and are not fetched again.
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
