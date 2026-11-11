# Testing data in production: test the change, not the table

Companion code for Part 3 of *Code Quality for Data Work* (Level Up Coding): *Which Data Tests
Catch a Broken Delivery? Seven Years of NYC Taxi Files, Two Families of Checks and Fourteen
Planted Faults*.

The running example is the monthly yellow-taxi trip files of the NYC Taxi and Limousine
Commission, January 2019 to September 2026, treated as a production feed: each monthly file is a
delivery. Each delivery is reduced to a profile (about 7 KB of JSON), and two families of checks
run on the profiles:

- **table checks** (15): rules on one delivery, written from the TLC data dictionary of
  18 March 2025, each in a strict and a 1%-tolerant variant, plus a volume floor;
- **change checks** (14): the same delivery against the 12 before it, as robust z-scores
  (median and median absolute deviation, with per-group floors), firing at |z| > 5.

Both families run on every real delivery from 2020-01 (81) and on fourteen faulted copies of
every delivery from 2023-01 (630 faulted deliveries).

## Results in one paragraph

Table checks: 8.0 alarms per real delivery, no delivery without one; 155 of 630 faulted
deliveries caught (14% of the silent faults). Change checks: 1.8 alarms per real delivery,
28 of 81 without any; 544 of 630 faulted deliveries caught (82% of the silent faults); no
faulted delivery caught by a table check and missed by every change check. The change checks
missed a random loss of 15% of rows (2 of 45). Full numbers: `results/summary.txt`; the
follow-up on the real alarms (duplicates in January 2020, a vendor with drop-off before pick-up
in 2020–21, near-empty days in September 2022 and 2023, three recording conventions behind the
reconciliation failures): `results/diagnosis.txt`.

## Settings fixed before the real profiles were computed

These are in the first commit of this folder, before the commit that added the real profiles.

- Change checks: robust z-score against the trailing 12 deliveries, with the per-group floors
  in `feed/checks.py`; a check fires at |z| > 5.
- Table checks: rules from the TLC data dictionary; strict variants fail on any violating row,
  tolerant variants on more than 1% of rows; volume floor 1,000,000 rows.
- A fault counts as detected by a check if the check fires on the faulted delivery and not on
  the clean delivery of the same month.

`feed/diagnose.py` and the alternative completeness metric discussed in the article were added
after the results were seen and are labelled as follow-ups.

## Files

- `feed/profile.py`: download one monthly file and reduce it to a profile with DuckDB.
- `feed/faults.py`: the fourteen faults, each a SQL view over one month's rows.
- `feed/checks.py`: the table checks, the change checks, the floors and the robust z-score.
- `feed/run.py`: profile a range of months and their faulted copies; `feed/bundle.py` packs the
  profiles into `profiles/real.jsonl.gz` and `profiles/fault.jsonl.gz`.
- `evaluate.py`: run every check; writes `results/summary.txt`, `results/real_alarms.csv`,
  `results/fault_detection.csv.gz` and `results/k_sweep.csv`.
- `feed/diagnose.py`: follow the real alarms back to the rows of ten months;
  `results/diagnosis.txt`.
- `figures.py`, `figures/`: the four figures of the article.
- `tests/`: unit tests on synthetic months.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m feed.run --months 2019-01:2026-09 --faults-from 2023-01 --out profiles   # downloads ~6 GB in total
python -m feed.bundle profiles
python evaluate.py --profiles profiles --out results
python -m feed.diagnose --out results/diagnosis.txt \
    --months 2019-01,2019-02,2020-01,2020-10,2022-09,2023-09,2024-03,2025-11,2026-06,2026-09
python figures.py
```

`.github/workflows/data-tests-in-production.yml` runs the profiling as 11 parallel jobs (about
17 minutes), runs the evaluation and the follow-up, and commits the profiles, results and
figures. On a pull request it compares the regenerated summary with the committed one. Every
profile records the URL, size and SHA-256 of the file it came from.

## Data

TLC Trip Record Data, yellow taxi trip records, downloaded from
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. The files are not redistributed
here; only the profiles computed from them are.
