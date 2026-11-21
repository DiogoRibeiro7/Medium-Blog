# The denominator decides: EU labour productivity with national-accounts and Labour Force Survey hours, 2009–2026

Companion code for the Medium article on measuring EU labour productivity with two sources of hours worked. The thesis: output per hour is real value added divided by hours, the numerator is shared, so the choice of hours decides the growth rate. Any gap between two productivity growth rates is exactly the gap between the two hours growth rates, with the sign reversed.

## What it finds

All numbers below are from `output.txt`.

**EU-27, real gross value added per hour.**
- **Two denominators.** 2019–2025: 0.64% a year with national-accounts hours, 1.07% with Labour Force Survey (LFS) hours. 2009–2019: 1.24% and 1.38%. Trend 1997–2007 (national accounts): 1.82%.
- **Hours.** 2019–2025: real GVA +8.1%, national-accounts hours +4.1%, LFS hours +1.4%. 2009–2025: hours +8.3% and +4.1%.
- **No recent jump.** Last two years to 2026-Q2: 0.90% and 0.82% a year.
- **Check.** Eurostat's own real GDP per hour: 1.21% (2009–2019) and 0.56% (2019–2025).

**The hours drift apart.** ln(LFS hours / national-accounts hours) falls 0.11% a year in 2009–2019 (Newey–West se 0.04) and 0.27% a year in 2021–2026 (se 0.09). The 2021-Q1 survey break is a step of −0.55 log points against quarterly noise of 0.73; removing it leaves LFS productivity growth at 0.98% a year.

**Countries, 2019–2025.**
- **Gaps.** LFS-based growth is higher in 20 of 27 countries, with a median gap of +0.35 points a year. Largest: Romania +2.21, Luxembourg +1.93, Czechia +1.84, Bulgaria +1.43, Portugal +1.28, Finland +1.13.
- **Classification.** Below 0.5% a year: 9 countries with national-accounts hours, 4 with LFS hours; 7 change group. Spearman rank correlation 0.80; 8 countries move five or more places.
- **Break removed.** With each country's 2021 step removed, LFS growth is still higher in 17 of 27.

**Where the disagreement sits.**
- **Workers or hours per worker.** EU-27 employment grows 4.97 log points in both sources; the whole gap (−2.59 log points) is hours per worker (national accounts −0.99, LFS −3.58).
- **Inside the survey.** The share absent rises from 9.7% to 10.1%, contributing −0.42; hours of those at work contribute −3.16.

**Explanations tested.**
- **Survey population.** LFS population 15–74 versus the demographic estimate: no general relation with the persons gap (correlation 0.06). Czechia fits the story; Portugal does not, because its LFS employment grew faster than national-accounts employment.
- **Cross-border work.** National minus domestic employment: Luxembourg −2.18 log points.
- **Age band.** LFS hours for ages 20–64 widen the EU gap (1.20% a year).

## Data

The committed snapshot is in `data/`, taken on 21 November 2026 from Eurostat:
- `gva.csv`: `namq_10_a10`, gross value added, total economy, chain-linked volumes (2020), seasonally and calendar adjusted, 1995-Q1 to 2026-Q2.
- `hours_na.csv`: `namq_10_a10_e`, hours worked, total employment, domestic concept, SCA.
- `employment_na.csv`: `namq_10_pe`, employment in persons, domestic and national concept, unadjusted.
- `hours_lfs.csv`: `lfsi_ahw_q`, LFS total actual hours worked in the main job, index 2021 = 100, SA, ages 15–74 and 20–64, 2009-Q1 to 2026-Q2.
- `employment_lfs.csv`: `lfsi_emp_q`, LFS employment aged 15–74, SA.
- `absences_lfs.csv`: `lfsi_abt_q`, persons absent from work in the reference week, % of employment, SA.
- `population_lfs.csv`: `lfsq_pganws`, LFS population and employment aged 15–74, total and foreign citizens.
- `population_demo.csv`: `demo_pjangroup`, population on 1 January, five-year age groups 15–19 to 70–74.
- `productivity_eurostat.csv`: `namq_10_lp_ulc`, real labour productivity per hour worked, index 2020 = 100.

## Method notes

- **Productivity growth** between two years uses annual totals (annual output over annual hours), so that the productivity gap equals the hours gap exactly in logs.
- **Index base** 2009 = 100, the first year of the LFS series. **Trend**: exponential fitted to 1997-Q1–2007-Q4.
- **Drift**: OLS slope of ln(LFS hours / national-accounts hours) on time, Newey–West (Bartlett) standard errors with four lags.
- **Break**: the step is the change in that log ratio from the last quarter before 2021-Q1 to 2021-Q1 (from 2019-Q4 for Germany, which has no LFS quarters in 2020). Removing it scales every LFS quarter from 2021-Q1 on; that is an upper bound on the break's effect.
- **Decompositions**: hours = persons × hours per worker; LFS hours per worker = (1 − share absent) × hours of those at work. Both in log points between 2019 and 2025 annual means.
- **Determinism**: no random number anywhere in the analysis.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the definitions on cases with a known answer
python fetch_data.py                       # download from Eurostat and compare with the snapshot
python hours_numbers.py > output.txt       # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/productivity-hours-denominator.yml` runs all of this, and fails if `output.txt` changes.

## Limits

- **No referee.** The analysis compares two published series; it cannot say which one is right.
- **Main job only.** The LFS index covers the main job; national accounts count all jobs.
- **Approximate split.** The absence split uses a percentage series and an index from the same survey; it is approximate.
- **One vintage.** National accounts and LFS back series are revised; this is the November 2026 vintage.

## Licence

Code: Apache-2.0. Data: Eurostat (reuse authorised, CC BY 4.0).
