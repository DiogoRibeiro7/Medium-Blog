# Testing data in production: test the change, not the table

Companion code for Part 3 of *Code Quality for Data Work* (Level Up Coding). Work in progress.

The running example is the monthly yellow-taxi trip files of the NYC Taxi and Limousine
Commission, January 2019 to September 2026, treated as a production feed: each monthly file is
a delivery. Every check is run on every real delivery, and on fourteen faulted copies of every
delivery from January 2023 (`feed/faults.py`).

## Settings fixed before the real profiles were computed

- Change checks: robust z-score (median and MAD) against the trailing 12 deliveries, with the
  per-group floors in `feed/checks.py`; a check fires at |z| > 5.
- Table checks: rules from the TLC data dictionary of 18 March 2025; strict variants fail on any
  violating row, tolerant variants on more than 1% of rows; volume floor 1,000,000 rows.
- A fault counts as detected by a check if the check fires on the faulted delivery and not on
  the clean delivery of the same month.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m feed.run --months 2019-01:2026-09 --faults-from 2023-01 --out profiles   # downloads ~6 GB in total
python -m feed.bundle profiles
python evaluate.py --profiles profiles --out results
```

`.github/workflows/data-tests-in-production.yml` runs the profiling as 11 parallel jobs and
commits the profiles and results.
