# Regression to the mean in four parts

Code for a four-part series on Medium by Diogo Ribeiro. The thesis of the series: regression to the mean is not a force acting on individuals but a consequence of selecting on a noisy measurement; its size can be computed from the correlation between measurements and the selection rule, and many effects measured after selecting extremes are partly or wholly this. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the history, with the data recomputed** ([`part1/`](part1)). Galton's mid-parents and children (1886), both regressions and the correlation; Galton's family records child by child, with standard errors that respect families; Pearson and Lee's fathers and sons (1903), regression in both directions with equal spreads; and Secrist's design (1933) replicated on major league teams, with the spread across teams season by season.
- **Part 2 — the mechanism** ([`part2/`](part2)). The measurement model and Kelley's estimate; three sources of disagreement between occasions (measurement error, transient states, stable traits) and a random walk; the closed-form fall of a selected group, checked by simulation, with Spearman–Brown averaging and the second-baseline design; the correlation between starting value and change, and Oldham's correction; skewed, heavy-tailed and two-group populations; shrinkage with common and unit-specific reliabilities; daily temperature anomalies at JFK and the DAX in levels and returns; and five designs compared on a simulated screening programme, with the anorexia trial (MASS).
- **Part 3 — real data** ([`part3/`](part3)). Three consecutive systolic readings on 10,622 adults in NHANES 2009–2012: the fall of groups selected on the first reading split into population drift, predicted regression and residual, with design-based (jackknife) standard errors; the first reading's extra variance; reclassification at 130 and 140 mmHg; a next-visit scenario and usual-pressure shares; and a held-out check across the two cycles. Batting seasons in the Lahman database 1955–2025: the reliability of a season computed from its multinomial sampling noise, component by component; the sophomore slump of Rookie of the Year winners against the line fitted on other rookies; team wins split into run differential and the Pythagorean residual; and two dozen selections predicted on 1955–1989 and checked from 1990, including Tweedie's formula with Lindsey's density estimate for OPS.
- **Part 4 — in data-science practice.** Forthcoming.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # a few seconds; downloads four files on first run
python figures.py
cd ../part2
python mechanism_numbers.py > output.txt  # about a minute; downloads three files on first run
python figures.py
cd ../part3
python real_data_numbers.py > output.txt  # about ten seconds; downloads five files on first run
python figures.py
```

A GitHub Actions workflow (`.github/workflows/regression-to-the-mean.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 1

Downloaded on first run from a pinned commit of the Rdatasets repository (https://github.com/vincentarelbundock/Rdatasets), checked against SHA-256 digests and cached in `part1/data/` (not committed):

- `HistData/Galton`, `HistData/GaltonFamilies` and `HistData/PearsonLee` (Friendly, R package HistData).
- `Lahman/Teams`: season records of major league teams (Lahman Baseball Database, CC BY-SA 3.0).

## Data for Part 2

Downloaded on first run from the same pinned commit of Rdatasets, checked against SHA-256 digests and cached in `part2/data/` (not committed): `nycflights13/weather` (hourly weather at the New York airports, 2013), `datasets/EuStockMarkets` (daily closes of four European indices, 1991–1998) and `MASS/anorexia` (weights before and after treatment in a trial for anorexia).

## Data for Part 3

Downloaded on first run, checked against SHA-256 digests and cached in `part3/data/` (not committed):

- `NHANESraw` from the R package NHANES 2.1.4 (Pruim), read from a pinned commit of the CRAN mirror on GitHub (https://github.com/cran/NHANES) with `pyreadr`. US National Health and Nutrition Examination Survey, 2009–2010 and 2011–2012. The script keeps only survey cycle, sex, age, the three systolic readings, the averaged systolic reading, the examination weight and the design variables, and prints only aggregates.
- `Lahman/Batting`, `Lahman/People`, `Lahman/AwardsPlayers` and `Lahman/Teams` from the same pinned commit of Rdatasets (Lahman Baseball Database, CC BY-SA 3.0).

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
