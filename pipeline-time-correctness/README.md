# Correctness Over Time in Data Pipelines

Public companion to a five-part series on where correct pipelines give wrong numbers because of time. Every record has three dates: when the event happened, when the pipeline received it, and when it was corrected. A pipeline that keeps only one of them cannot reproduce what it reported yesterday, and cannot say how wrong today's number is.

1. **History** (`part1/`): how four fields met the same problem. Insurance run-off triangles and the chain ladder, epidemic reporting delays and nowcasting, valid time and transaction time in temporal databases, slowly changing dimensions in warehouses, and event time, watermarks and triggers in stream processing. Each idea is rebuilt in code, and the published numbers are recomputed where they exist.
2. **Mechanism** (`part2/`): five mechanisms simulated where the right answer is known. How much of the latest periods is missing for a given delay distribution; what a watermark's slack and allowed lateness trade, and what a nowcast does instead of waiting; what an incremental load loses with a high-water mark on event time, and what insert-only, upsert and an append-only version log keep when records are corrected; what training on final values and joining the current version of a dimension do to a model; and what a backfill does to values already sent.
3. **Real data** (`part3/`): three public sources that kept every version. The Robert Koch Institute's daily COVID-19 case files (1,034 files, 15.7 GB), the German COVID-19 Hospitalization Nowcast Hub's reporting triangle and the nowcasts ten models published from November 2021 to April 2022, and quarterly real-GDP vintages for the US, the euro area, Japan and Switzerland. It measures the delay distributions, the shortfall in the latest periods, the errors of real nowcasts and the size of the restatements.
4. **Practice** (`part4/`): a small daily pipeline in DuckDB that lands the RKI files of Part 3 in an append-only table with their publication date and load time, loads them incrementally on arrival, publishes each 7-day count with a nowcast, its estimated completeness and an interval from its own past errors, compares weeks at equal maturity, and builds training rows on both time axes. It is replayed day by day from June 2021 to August 2023, and every one of its 822 runs is then recomputed and compared with what it stored.
5. **Fixes** (`part5/`): three changes to the Part 4 pipeline, tested by replaying all 822 mornings with each change alone and in every combination: a content check that counts each date's age from the latest file that brought anything new, a calendar of Germany's nationwide public holidays, and a fallback that uses the same weekday a week earlier instead of a level four to eight weeks old.

The manuscripts are kept elsewhere; this repository holds the code, data, outputs and figures.

## Part 1

```
cd pipeline-time-correctness
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 60 tests (11 for Part 1, 19 for Part 2, 15 for Part 3, 9 for Part 4, 6 for Part 5)
cd part1
python history_numbers.py > output.txt         # every number in the article
python figures.py                              # figures/*.png
```

- `history.py`: the functions, one section per field.
  - A. Taylor–Ashe run-off triangle: chain ladder written in SQL (DuckDB) and Mack's (1993) standard errors. The test suite checks the total reserve (18,680,856), its standard error (2,447,095, 13%) and the per-year percentages of Mack's Table 3.
  - B. The 630 HUS hospitalizations of the 2011 STEC O104:H4 outbreak in Germany: reporting delays, the epidemic curve as known on given days, and a chain-ladder nowcast on the daily reporting triangle.
  - C. A bitemporal price table (valid time and transaction time) and the restatement it implies.
  - D. A simulated warehouse: Type 1 against Type 2 dimensions, with a point-in-time `ASOF JOIN`.
  - E. Twelve stream elements: event-time windows, a heuristic watermark, late firings, allowed lateness, and the three accumulation modes.
- `history_numbers.py` prints every number; `output.txt` is its committed output, and CI reruns it and compares.
- `figures.py` draws the five figures from the same functions.
- `fetch.py` rebuilds `data/*.csv` from the R packages (needs `pyreadr`); `data/fetch_log.txt` records the source URLs and SHA-256 hashes.

### Data

- `data/taylor_ashe.csv`: accumulated claims, 10 accident years × 10 development years (55 cells observed). Source: Taylor, G. C. and Ashe, F. R. (1983), *Journal of Econometrics* 23, 37–61, as `GenInsLong` in the R package ChainLadder (GPL ≥ 2), [mages/ChainLadder@41f4e94](https://github.com/mages/ChainLadder/tree/41f4e949e5c3bffd7a18af2c0eaa98d6bae2da2f).
- `data/hus_o104_2011.csv`: hospitalization date and date the hospitalization reached the Robert Koch Institute, 630 HUS cases. Delays above 15 days are set to 15 in the source. Source: `husO104Hosp` in the R package surveillance (GPL-2), [cran/surveillance@8e07579](https://github.com/cran/surveillance/tree/8e07579097e463f828b48ff7fb34b2945d1fb374), the reporting triangle of Höhle and an der Heiden (2014), *Biometrics* 70(4), 993–1002.

Sections C–E use small constructed examples, fixed in the code; section D's simulation uses a fixed seed.

## Part 2

```
cd pipeline-time-correctness/part2
python mechanism_numbers.py > output.txt       # every number in the article
python figures.py                              # figures/*.png
```

- `mechanism.py`: the functions, one section per mechanism.
  - Delays: three log-normal reporting delays (fast, medium and slow; medians 0.6, 2 and 6 days), counted in whole days from event date to report date and capped at 30 days, plus a "backlog" version of the medium delays (median 3 days).
  - M1. Right truncation: the share of the latest days still missing, the week-over-week change seen on the last day against the true change, and the growth rate at which the last week looks flat.
  - M2. Waiting against estimating: the share of a window in its on-time pane for a given watermark slack, the share dropped for a given allowed lateness, and a chain-ladder nowcast against publishing raw counts, with delays that stay the same and delays that lengthen.
  - M3. Loads, on a simulated shop (about 169,000 orders over 90 days, 10% taken offline, 8% later refunded or cancelled): a high-water mark on event time against one on arrival time, written once as a vectorized rule and once as SQL; and insert-only, upsert and append-only version-log loads run every morning in DuckDB.
  - M4. Features: a model trained on final counts and served on counts as known on the day; a churn model whose plan is joined from the current (Type 1) dimension or as of the snapshot (`ASOF JOIN`).
  - M5. Backfills: a daily aggregate rerun by appending and by replacing partitions; how much the restated values differ from what was sent; and a week-over-week alert read live, from the daily table, and across the backfill boundary.
- `mechanism_numbers.py` prints every number; `output.txt` is its committed output, and CI reruns it and compares.
- `figures.py` draws the six figures from the same functions.

Everything is simulated with fixed seeds and money is kept in integer cents, so the output is the same on every machine. No data is downloaded.

## Part 3

```
cd pipeline-time-correctness/part3
python fetch_rki.py      # about an hour: reads 1,034 daily RKI files (15.7 GB) and keeps aggregates in data/
python fetch_hub.py      # a few minutes: the hospitalisation triangle and the nowcasts, from the hub's repository
python fetch_gdp.py      # seconds: GDP vintages (needs pyreadr)
python realdata_numbers.py > output.txt
python figures.py
```

The fetch scripts are pinned to a commit of each source repository and log the git blob id (and, for the RKI files, the size) of every file they read. Their outputs are committed in `data/`, so the numbers, the figures and the tests run without the network.

- `data/rki_melde.csv.gz`, `data/rki_onset.csv.gz`: for each publication from 2 April 2021 to 31 December 2023 (and the first of each month after that, to 10 November 2026), the cases by reporting date (Meldedatum) and by onset date for the 120 days before, with the cases flagged new and removed in that file.
- `data/rki_state.csv.gz`: the same by federal state, for the 28 days before each publication.
- `data/rki_snapshots.csv.gz`: cases by reporting date for every date, in the file of the first day of each month and in the last file.
- `data/hub_triangle.csv.gz`: the Nowcast Hub's triangle of hospitalised cases (all ages, Germany and the 16 states), with the change at each delay from 0 to 80 days and after.
- `data/hub_nowcasts.csv.gz`: each model's nowcast for Germany, all ages, forecast dates 22 November 2021 to 29 April 2022.
- `data/gdp_vintages.csv`: real GDP by vintage (2002 Q4 to 2024 Q4) for the US, the euro area, Japan and Switzerland, from the rOpenSci package reviser (real-time database of Indergand and Leist, 2014).
- `realdata.py`: the functions (completeness by age, the 7-day sums in pandas and in SQL, the check of the change flags against the files, the chain ladder (pooled and by weekday) on real triangles, the hub evaluation, GDP revision statistics, and as-of and first-release queries in SQL).
- `realdata_numbers.py` prints every number in the article; `output.txt` is its committed output.
- `figures.py` draws the seven figures.

Sources and terms: the RKI case files are published by the Robert Koch Institute under CC BY 4.0; the hub's triangle and nowcasts come from the public repository KITmetricslab/hospitalization-nowcast-hub (Wolffram et al., 2023; a stable release is on Zenodo, record 7828604), whose triangle is built from RKI data; reviser is MIT-licensed. Check each source's terms before reusing the extracted files.

## Part 4

```
cd pipeline-time-correctness/part4
python practice_numbers.py > output.txt   # about four minutes: replays 822 daily runs, then recomputes each one
python figures.py                         # figures/*.png (replays the pipeline again)
```

The input is `part3/data/rki_state.csv.gz`: for each RKI publication from 2 April 2021 to 31 December 2023, the cases by federal state and reporting date for the 28 days before the publication. `write_incoming` writes it out as one CSV per publication, as a source would deliver it, and gives each file an arrival time (07:00 on its date; the file of 15 March 2022 arrives two days late). Nothing is downloaded.

- `pipeline.py`: the pipeline, in three schemas.
  - `landing`: `files` (one row per file name and SHA-256, with the publication date and the load time) and `cases` (the rows of each file, copied unchanged). Insert-only. A file whose content is already landed is skipped; a file republished under the same name with new content is landed as a new version.
  - `core`: the table macro `known_at(as_of)`, which returns, for each publication, the latest version loaded by `as_of`; and `delay_profile`, the share of the 28-day count known at each age (0 to 28 days) for each weekday, estimated at each run from the four latest complete dates of that weekday.
  - `mart.daily_report`: one row per run date (the partition a run replaces): the 7-day count as published, a nowcast (known count divided by the expected share; the mean 28-day level where less than a fifth is expected), a 90% interval from the ratios of matured counts to nowcasts over the latest 56 runs whose dates are 28 days old, the estimated completeness, the week-over-week change against the run seven days earlier, the time of the run and a hash of the SQL (`code_version`).
  - `Pipeline.run(day)` loads what has arrived by 08:00, rebuilds the profile and the report row for that time, and replaces only that day's rows. `Pipeline.reproduce(day)` recomputes a stored run from landing rows loaded by its time and mart rows of earlier runs.
  - `TRAINING_ROWS_SQL`: rows for a model that predicts next week's cases in a state from this week's, with the feature both as published on the run day and as it stood 28 days later.
- `practice_numbers.py` prints every number in the article: the replay, the error of the published count and of the nowcast against the file of 10 November 2026, the coverage of the interval, week-over-week changes, the reproduction of every run, the late file, a rerun of the last day, the training rows, and the files that added no case. `output.txt` is its committed output.
- `figures.py` draws the four figures.

## Part 5

```
cd pipeline-time-correctness/part5
python fixes_numbers.py > output.txt   # about ten minutes on two cores: eight variants x 822 runs, each recomputed
python figures.py                      # figures/*.png (replays four of the variants)
```

The three changes are options of the Part 4 pipeline, in `part4/pipeline.py` (`Pipeline(content_check=..., holidays=..., fallback=...)`). With all of them off the SQL is exactly the SQL of Part 4 and the code version is the same (`9adf5e8d8533`), so Part 4's output does not change.

- `content_check`: the report finds the latest file that changed anything against the file before it (the data cut), counts each reporting date's age from that file, and stores the cut in a new column, `data_cut`. Dates the cut does not cover are treated as unknown, not as zero.
- `holidays`: Germany's nine nationwide public holidays (New Year, Good Friday, Easter Monday, 1 May, Ascension, Whit Monday, 3 October, 25 and 26 December) are treated as Sundays, in the delay profile and in the report.
- `fallback="last_week"`: for a date with less than a fifth of its count expected in hand, the estimate for the latest earlier date of the same weekday that is at least seven days old, instead of the mean 28-day count of the profile's dates (28 to 55 days old).
- `fixes_numbers.py` replays the eight combinations in parallel, recomputes every run of each, and prints the errors against the final counts: overall and weighted by size, on the ten mornings that used a file with nothing new, by weekday and year, run by run against Part 4, the estimated completeness, the interval's coverage, the week-over-week change, and four runs date by date. `output.txt` is its committed output.
- `figures.py` draws the three figures.
