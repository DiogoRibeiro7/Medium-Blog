# Code quality for analytics SQL, measured

Companion code for the Medium article *Code Quality for Analytics SQL, Measured: Linters, Tests, Review, Performance and Security on the NYC Flights Data*.

Seven SQL models answer operational questions about the 336,776 flights that left New York in 2013. They exist in two versions. `project_before/` is the project as it often ships: it runs without an error and contains eight defects (text comparison of numbers, a date taken in the session time zone, arithmetic on HHMM clock readings, an inner join to an incomplete dimension, a join on a column that means something else, a join at the wrong grain, NULLs coalesced to zero, and `NOT IN` over a column that becomes nullable once the loader is fixed). `project_after/` is the reviewed project: a typed staging layer, seven marts, and a test suite. Every tool in the article gets the same eight chances to find the defects.

## Data

The `nycflights13` Python package (0.0.3, CC0) ships the CSV files of Hadley Wickham's R package: flights from the Bureau of Transportation Statistics, hourly weather from the Iowa Environmental Mesonet, aircraft from the FAA registry. The package's own import fails on recent Python because it uses the removed `pkg_resources`, so `sqlq/data.py` reads the files from the installed package directly. Nothing is downloaded at run time and no data is committed.

## Layout

| Path | What it holds |
|------|---------------|
| `project_before/models/` | The seven shipped models. |
| `project_after/models/` | Four staging models and seven marts; lint-clean under SQLFluff and sqruff. |
| `project_after/tests/schema.yml` | 31 generic tests (unique, not null, accepted values and ranges, relationships, row counts, expressions). |
| `project_after/tests/singular/` | 8 reconciliation tests: totals that must tie back to staging. |
| `project_after/tests/unit/` | 7 unit tests on fixtures (`cases.yml`) and 5 boundary fixtures added after the first mutation run (`cases_boundaries.yml`). |
| `review/` | The idle-aircraft model as it becomes after the loader fix, with its `NOT IN` kept. |
| `security/demo_queries.py` | Deliberately unsafe query functions for the injection experiment. Do not copy them. |
| `sqlq/` | Loader, model builder (dependency order from sqlglot), test runner, sqlglot mutation operators, data diff. |
| `analysis.py` | Prints every number in the article except timings. |
| `performance.py` | Timings (machine-dependent, not checked in CI). |
| `figures.py` | The three figures. |

## Run

From this folder, with Python 3.13:

```bash
python -m pip install -r requirements.txt
python analysis.py > results/output.txt       # under 7 minutes on two cores
python performance.py > results/performance.txt
python figures.py
```

`results/output.txt` is the run the article quotes. `analysis.py` sets the DuckDB session time zone explicitly on every connection, so its output does not depend on the machine; CI reruns it and fails if the file changes. `results/mutants.json` and `results/injection.json` hold the per-mutant and per-payload outcomes. Timings in `results/performance.txt` come from a two-core cloud machine; rerun them where your queries run.

## Notes

- The 10.1-million-row table in `performance.py` is the 2013 flights repeated 30 times with timestamps shifted by whole years: the values are real, the volume is not.
- The test harness mirrors the ideas of dbt data tests and unit tests but is written in plain Python so that each step can be mutated and timed.
- Mutation scores are reported with Wilson intervals; survivors are classified by whether the mutated build changes any output on the real data.
