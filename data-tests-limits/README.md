# What a delivery cannot vouch for: dictionary rules and random row losses

Companion code for Part 4, the last part, of *Code Quality for Data Work* (Level Up Coding). Part 3
(`../data-tests-in-production`) ended with two conjectures about the limits of data tests computed from a
delivery. This part tests both on four NYC TLC trip feeds (yellow taxi, green taxi, FHV and high-volume FHV),
monthly files from 2019 to 2026-09. The hypotheses and every setting are in `PLAN.md`, committed before any
profile or report value was computed.

- **E1, dictionary rules.** Rules written from the 18 March 2025 data dictionaries of yellow, FHV and
  high-volume FHV (`feed/contracts.py`, `feed/rules.py`), strict and 1%-tolerant, on every monthly file.
- **E2, random losses.** Losses of 1% to 20% of a delivery's rows against two detectors that use only the
  delivery's own history (volume, seasonal forecast) and two that use an outside count: the TLC monthly
  report's trips per day (NYC Open Data, dataset v6kb-cqej), as a history-based check and as a fixed 1% check.

## Results in one paragraph

E1: every delivery of every feed fails at least two strict rules and at least one tolerant rule; the strict
rules that fail on every delivery are 5 of 7 (yellow), 2 of 6 (FHV) and 2 of 6 (high-volume FHV), so H1a
(more than half) is not supported; alarms are more frequent the further a delivery is from the dictionary's
date (H1b supported), but the deliveries after the dictionary fail as many rules, and in March 2026 the
high-volume airport fee moved from the dictionary's $2.50 to $3.50. E2: the delivery-only detectors catch at
most 14% of 5% losses and at most 74% of 20% losses (H2a supported); the outside count catches 92% to 95% of 5%
losses in yellow, green and high-volume FHV, and 95% of 1% losses in high-volume FHV, but 22% in FHV, whose
files hold 5% to 32% fewer trips than the report from 2022 (H2b not supported). The outside count also flags
real deliveries: yellow November 2022 (-6.8%) and September 2023 (-8.9%), green November 2022 (-3.4%) and
high-volume FHV June 2026 (-2.2%). Full numbers: `results/summary.txt` and `results/details.txt`.

## Files

- `PLAN.md`: the pre-registration, with the one change made after it (row counts from the profiles).
- `feed/sources.py`: snapshot of the sources: the monthly report (NYC Open Data), the dictionaries where
  nyc.gov serves them to the runner (it does not), and, with `--footers`, the Parquet footers.
- `feed/contracts.py`: one contract per feed, with the dictionary sentence behind each field.
- `feed/profile.py`, `feed/run.py`: download a monthly file and reduce it to the counts the rules need.
- `feed/rules.py`: the seven rule kinds, strict and tolerant, judged on a profile.
- `feed/counts.py`: delivered and outside counts, the four detectors, the matched budget.
- `evaluate.py`: both experiments; writes `results/summary.txt`, `results/rules_by_delivery.csv.gz`,
  `results/losses.csv` and `results/scores.csv`.
- `details.py`: the descriptive numbers the article quotes that `evaluate.py` does not print (field-level
  shares, the file-versus-report difference month by month, the noise of each detector, the outside-count
  alarms on the real deliveries, and two yellow-taxi checks against Part 3's profiles); writes
  `results/details.txt`. Not part of the pre-registered analysis.
- `figures.py`, `figures/`: the three figures of the article.
- `tests/`: unit tests on synthetic months.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m feed.sources --out sources
for feed in yellow green fhv fhvhv; do python -m feed.run --feed $feed --months 2019-01:2026-09 --out profiles; done
python evaluate.py --bundle --profiles profiles --sources sources --out results
python details.py --profiles profiles --sources sources --results results --out results
python figures.py
```

`.github/workflows/data-tests-limits.yml` runs the profiling as one job per feed and year (at most eight at
once, with finished profiles cached so a rerun resumes), runs both experiments, and commits sources, profiles,
results and figures. On a pull request it compares the regenerated summary with the committed one, using the
committed report snapshot.

## Data

TLC Trip Record Data (<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>) and the TLC monthly
report on NYC Open Data (<https://data.cityofnewyork.us/d/v6kb-cqej>). The trip files are not redistributed
here; only the profiles computed from them and a snapshot of the monthly report are.
