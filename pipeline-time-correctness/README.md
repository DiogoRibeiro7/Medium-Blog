# Correctness Over Time in Data Pipelines

Public companion to a four-part series on where correct pipelines give wrong numbers because of time. Every record has three dates: when the event happened, when the pipeline received it, and when it was corrected. A pipeline that keeps only one of them cannot reproduce what it reported yesterday, and cannot say how wrong today's number is.

1. **History** (`part1/`): how four fields met the same problem. Insurance run-off triangles and the chain ladder, epidemic reporting delays and nowcasting, valid time and transaction time in temporal databases, slowly changing dimensions in warehouses, and event time, watermarks and triggers in stream processing. Each idea is rebuilt in code, and the published numbers are recomputed where they exist.
2. Mechanism (simulation with known truth) — to come.
3. Real data (the Robert Koch Institute's daily COVID-19 case files; the Philadelphia Fed's real-time macroeconomic data) — to come.
4. Practice (a pipeline that keeps time correct, measured on the Part 3 data) — to come.

The manuscripts are kept elsewhere; this repository holds the code, data, outputs and figures.

## Part 1

```
cd pipeline-time-correctness
pip install -r requirements.txt
python -m unittest discover -s tests -v       # 11 tests
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
