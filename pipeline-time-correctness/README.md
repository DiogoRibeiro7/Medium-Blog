# Correctness Over Time in Data Pipelines

Public companion to a nine-part series on where correct pipelines give wrong numbers because of time. Every record has three dates: when the event happened, when the pipeline received it, and when it was corrected. A pipeline that keeps only one of them cannot reproduce what it reported yesterday, and cannot say how wrong today's number is.

1. **History** (`part1/`): how four fields met the same problem. Insurance run-off triangles and the chain ladder, epidemic reporting delays and nowcasting, valid time and transaction time in temporal databases, slowly changing dimensions in warehouses, and event time, watermarks and triggers in stream processing. Each idea is rebuilt in code, and the published numbers are recomputed where they exist.
2. **Mechanism** (`part2/`): five mechanisms simulated where the right answer is known. How much of the latest periods is missing for a given delay distribution; what a watermark's slack and allowed lateness trade, and what a nowcast does instead of waiting; what an incremental load loses with a high-water mark on event time, and what insert-only, upsert and an append-only version log keep when records are corrected; what training on final values and joining the current version of a dimension do to a model; and what a backfill does to values already sent.
3. **Real data** (`part3/`): three public sources that kept every version. The Robert Koch Institute's daily COVID-19 case files (1,034 files, 15.7 GB), the German COVID-19 Hospitalization Nowcast Hub's reporting triangle and the nowcasts ten models published from November 2021 to April 2022, and quarterly real-GDP vintages for the US, the euro area, Japan and Switzerland. It measures the delay distributions, the shortfall in the latest periods, the errors of real nowcasts and the size of the restatements.
4. **Practice** (`part4/`): a small daily pipeline in DuckDB that lands the RKI files of Part 3 in an append-only table with their publication date and load time, loads them incrementally on arrival, publishes each 7-day count with a nowcast, its estimated completeness and an interval from its own past errors, compares weeks at equal maturity, and builds training rows on both time axes. It is replayed day by day from June 2021 to August 2023, and every one of its 822 runs is then recomputed and compared with what it stored.
5. **Fixes** (`part5/`): three changes to the Part 4 pipeline, tested by replaying all 822 mornings with each change alone and in every combination: a content check that counts each date's age from the latest file that brought anything new, a calendar of Germany's nationwide public holidays, and a fallback that uses the same weekday a week earlier instead of a level four to eight weeks old.
6. **Out of sample** (`part6/`): the Part 5 changes, frozen, and a graded version of the content check, tuned on 2021-2023 only, replayed on the RKI's daily files of 1 January 2024 to 10 November 2026, which no earlier part had read.
7. **Another feed** (`part7/`): the series' two conjectures about files that bring nothing new, checked against data the series had not used: every version of the Johns Hopkins CSSE daily reports for the 50 US states and DC (April 2020 to March 2023) and of its global time series, the state and national holiday calendars, and the RKI files of 2022 to 2026 on days that were holidays in Berlin only.
8. **Scoring against the final version** (`part8/`): Part 7's second conjecture, that evaluations scored against an aggregator's final version understate the errors around holidays, checked on Johns Hopkins' morning counts for Germany and on 60,266 weekly case forecasts of the German and Polish COVID-19 Forecast Hub, each scored against counts dated by publication, by reporting date and by reference date.
9. **A learned calendar** (`part9/`): Part 7's first conjecture, that a producer's calendar learned from the history of its unchanged files predicts its next gaps better than a published holiday calendar and changes at identifiable points, checked on 266 producers in the Johns Hopkins files (the US states, the provinces of Canada, the states of Australia, Hong Kong, and 193 countries), with the changes matched against announcements by the producers.

The manuscripts are kept elsewhere; this repository holds the code, data, outputs and figures.

## Part 1

```
cd pipeline-time-correctness
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 90 tests (11 for Part 1, 19 for Part 2, 15 for Part 3, 9 for Part 4, 6 for Part 5, 5 for Part 6, 7 for Part 7, 6 for Part 8, 12 for Part 9)
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

## Part 6

```
cd pipeline-time-correctness/part6
python fetch_rki_daily.py              # optional: rebuilds data/ from the RKI archive (about an hour, some 23 GB read)
python oos_numbers.py > output.txt     # about forty minutes on two cores: 13 versions, in and out of sample
python figures.py                      # figures/*.png, from the tables in results/
```

- `fetch_rki_daily.py` reads the 1,043 daily files from 1 January 2024 to 10 November 2026 (the archive has none for 14 February 2024 and 26 May 2026) from the same pinned version of the archive as `part3/fetch_rki.py`, with its code, and writes the same extract: `data/rki_state_2024_2026.csv.gz` (cases by state and reporting date, the 28 days before each file) and `data/rki_melde_2024_2026.csv.gz` (national, 120 days, for checks), with the blob id and size of every file in `data/rki_fetch_log_2024_2026.txt`. The pipeline reads Part 3's extract up to 31 December 2023 and this one from 1 January 2024.
- The graded check is another option of the pipeline, `Pipeline(graded=tau)`, with `holidays=True` and `fallback="last_week"`: for each of the 14 latest files, the cases it added against the cases the delay profile expected it to add; a file below the tolerance `tau` is short, its expected share is taken off the dates it covers, and later files' surplus pays the shortfall down. The report gains `missing_cases`.
- The holiday calendar is a parameter, `holiday_years`; Part 5 used 2020-2024, Part 6 uses 2020-2027. With the Part 5 calendar every Part 5 version has the code version it had.
- `oos_numbers.py` picks the tolerance on the 822 mornings of Parts 4 and 5, then replays all thirteen versions from 1 October 2023 and scores 1 January 2024 to 13 October 2026, recomputing every morning of each. It prints every number in the article (`output.txt`) and writes the two tables in `results/` that `figures.py` draws.

## Part 7

```
cd pipeline-time-correctness/part7
pip install holidays==0.108; python fetch_jhu.py   # optional: rebuilds data/ from the Johns Hopkins repository (a few minutes)
python jhu_numbers.py > output.txt                 # every number in the article, in a few seconds
python figures.py                                  # figures/*.png
```

- `fetch_jhu.py` clones the Johns Hopkins CSSE COVID-19 repository without file contents, at its last commit (`4360e50`, 10 March 2023; the repository is archived), and fetches only the file versions it needs. For each of the 1,062 US daily reports (12 April 2020 to 9 March 2023) and each state and DC, `data/jhu_us_states.csv.gz` keeps the confirmed count as first published, the counts of the day before and of seven days before as they stood when the file landed (its commit time), the count in the final version, the number of versions, and the row's `Last_Update` in the new file and the file of the day before. For each date from 24 March 2020, `data/jhu_countries.csv.gz` keeps each country's count in the first version of the global time series that holds the date, with the day before and seven days before from the same version, and the count in the final version. `data/holidays.csv.gz` holds the calendars used, from python-holidays 0.108: US federal holidays, the legal holidays of each state, and the public holidays of 196 countries. Every blob read is listed in `data/jhu_versions.csv.gz`.
- `jhu.py` holds the definitions. A file is unchanged for a state or country when its count equals the day before's as the pipeline had it at landing. Weekdays are compared only in a state's or country's daily regime: at most 10% of its ordinary weekdays unchanged in the 57 days around the date (`jhu_numbers.py` shows 5%, 10% and 20% with windows of 14, 28 and 42 days).
- `jhu_numbers.py` also reads the RKI extracts of Parts 3 and 6 (`rki_melde`, `rki_snapshots`) for the files after Berlin-only holidays and for the comparison of Germany as Johns Hopkins published it with the RKI's count by reporting date. It writes four small tables to `results/`.
- `figures.py` draws the three figures.

### Data

- Johns Hopkins University Center for Systems Science and Engineering (CSSE), COVID-19 Data Repository, [CSSEGISandData/COVID-19@4360e50](https://github.com/CSSEGISandData/COVID-19/tree/4360e50239b4eb6b22f3a1759323748f36752177), CC BY 4.0. Dong, E., Du, H. and Gardner, L. (2020), An interactive web-based dashboard to track COVID-19 in real time, *The Lancet Infectious Diseases* 20(5), 533–534.
- Holiday calendars: [python-holidays](https://github.com/vacanza/holidays) 0.108 (MIT).

## Part 8

```
cd pipeline-time-correctness/part8
pip install holidays==0.108; python fetch_hub.py   # optional: rebuilds data/ from the forecast hub and the RKI archive (a few minutes)
python fetch_hub.py --forecasts-only               # the forecasts, which are not committed (about a minute)
python hub_numbers.py > output.txt                 # every number in the article, in about ten seconds
python figures.py                                  # figures/*.png (after hub_numbers.py)
```

- `fetch_hub.py` clones the German and Polish COVID-19 Forecast Hub without file contents at its last commit (`991c7a0`, 31 May 2023) and reads the 2,496 weekly case forecast files for Germany: `data/hub_forecasts.csv.gz` keeps, for each model, forecast date, location (Germany and the 16 states), horizon (1 to 4 weeks) and target week, the median and six quantiles. `data/hub_truth.csv.gz` is the hub's truth, daily cases by the day the RKI published them. From the RKI archive commit of Parts 3 and 6 it reads the file of 10 November 2026 and writes `data/rki_dates.csv.gz`, cases by state, reporting date, reference date and whether the onset is known. `data/holidays_de.csv.gz` holds the national and state holidays (python-holidays 0.108).
- `hub.py` holds the definitions: the three datings, weeks ending on Saturday as in the hub, national and regional holiday weeks, the absolute log error of the median and 80% interval coverage, the comparison of holiday weeks with the nearest weeks without a holiday, and a bootstrap that resamples whole weeks or holidays.
- `hub_numbers.py` also reads Part 7's Johns Hopkins extract for the morning 7-day counts of Germany. It writes tables to `results/`, which `figures.py` reads; they are not committed.
- The forecasts belong to the teams that submitted them, under licences that differ from team to team, so `data/hub_forecasts.csv.gz` is not committed: run `python fetch_hub.py --forecasts-only` first (about a minute; the reproduce workflow does this before the tests). The hub's truth before April 2020, which came partly from IHME under a non-commercial licence, is left out.

### Data

- KITmetricslab, German and Polish COVID-19 Forecast Hub, [covid19-forecast-hub-de@991c7a0](https://github.com/KITmetricslab/covid19-forecast-hub-de/tree/991c7a054246faee4abace7f5b705f044b0ffd01); forecasts are the work of the participating teams, under the licences stated in the hub. Bracher, J. et al. (2021), A pre-registered short-term forecasting study of COVID-19 in Germany and Poland during the second wave, *Nature Communications* 12, 5173.
- Robert Koch-Institut, SARS-CoV-2 Infektionen in Deutschland (archive), CC BY 4.0, at the commit used in Parts 3 and 6.

## Part 9

```
cd pipeline-time-correctness/part9
pip install holidays==0.108; python fetch_global.py   # optional: rebuilds data/ from the Johns Hopkins repository (a minute)
python calendar_numbers.py > output.txt               # every number in the article, in about a minute
python figures.py                                     # figures/*.png (after calendar_numbers.py)
```

- `fetch_global.py` reads, from the 1,081 first versions of the global time series that Part 7 read (listed in `part7/data/jhu_versions.csv.gz`) and the final version, the rows of the eight countries that the series splits into states, provinces or territories: `data/jhu_global_rows.csv.gz`. It also writes the legal holidays of the Canadian provinces and territories, the Australian states and territories and Hong Kong (`data/calendars.csv.gz`) and every calendar's legal weekend by year (`data/weekends.csv`), from python-holidays 0.108.
- `data/documented_changes.csv` is written by hand: for 15 producers (the ten most populous US states and the ten most populous EU member states in the Johns Hopkins lookup table, and the United Kingdom, where a change of two or more days of the week was detected), the first day of a changed reporting schedule as announced, with the source.
- `learned.py` holds the definitions: the producers (US states from Part 7's extract; countries from Part 7's extract, with the country rows of Denmark, France, the Netherlands, New Zealand and the United Kingdom from this part's; Canadian provinces, Australian states and Hong Kong from this part's), the unchanged flag of Part 7, which producer-days are counted, the kinds of day under the published calendar, the four models (the published calendar; the producer's weekday rates; those plus the group's holiday effect; those plus the producer's own holiday effect), sliding windows, exponential forgetting, change-point detection by day of the week and shared by a producer's seven days, the segmentation used to date changes after the fact, and the matching of announced changes.
- `calendar_numbers.py` prints every number; `output.txt` is its committed output. It writes small tables to `results/` for the figures.

### Data

- Johns Hopkins University Center for Systems Science and Engineering, COVID-19 Data Repository, CC BY 4.0, [CSSEGISandData/COVID-19@4360e50](https://github.com/CSSEGISandData/COVID-19/tree/4360e50239b4eb6b22f3a1759323748f36752177), as read in Part 7.
- Holiday calendars and legal weekends: [python-holidays](https://github.com/vacanza/holidays) 0.108 (MIT).
- Announcements of reporting changes: the sources listed in `data/documented_changes.csv`.
