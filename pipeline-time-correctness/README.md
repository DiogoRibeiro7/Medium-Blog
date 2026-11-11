# Correctness Over Time in Data Pipelines

Public companion to a four-part series on where correct pipelines give wrong numbers because of time. Every record has three dates: when the event happened, when the pipeline received it, and when it was corrected. A pipeline that keeps only one of them cannot reproduce what it reported yesterday, and cannot say how wrong today's number is.

1. **History** (`part1/`): how four fields met the same problem. Insurance run-off triangles and the chain ladder, epidemic reporting delays and nowcasting, valid time and transaction time in temporal databases, slowly changing dimensions in warehouses, and event time, watermarks and triggers in stream processing. Each idea is rebuilt in code, and the published numbers are recomputed where they exist.
2. **Mechanism** (`part2/`): five mechanisms simulated where the right answer is known. How much of the latest periods is missing for a given delay distribution; what a watermark's slack and allowed lateness trade, and what a nowcast does instead of waiting; what an incremental load loses with a high-water mark on event time, and what insert-only, upsert and an append-only version log keep when records are corrected; what training on final values and joining the current version of a dimension do to a model; and what a backfill does to values already sent.
3. **Real data** (`part3/`): three public sources that kept every version. The Robert Koch Institute's daily COVID-19 case files (1,034 files, 15.7 GB), the German COVID-19 Hospitalization Nowcast Hub's reporting triangle and the nowcasts ten models published from November 2021 to April 2022, and quarterly real-GDP vintages for the US, the euro area, Japan and Switzerland. It measures the delay distributions, the shortfall in the latest periods, the errors of real nowcasts and the size of the restatements.
4. Practice (a pipeline that keeps time correct, measured on the Part 3 data) — to come.

The manuscripts are kept elsewhere; this repository holds the code, data, outputs and figures.

## Part 1

```
cd pipeline-time-correctness
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 45 tests (11 for Part 1, 19 for Part 2, 15 for Part 3)
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
