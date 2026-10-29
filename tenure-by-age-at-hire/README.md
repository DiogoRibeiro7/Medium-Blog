# How long new hires stay, by age at hire

Code for a Medium article by Diogo Ribeiro. The question: does a worker hired at 45 or 55 stay with the employer for less time than one hired at 25 or 30, measured from the start date of the job? The analysis uses the job-duration tables that the U.S. Bureau of Labor Statistics publishes from the National Longitudinal Survey of Youth 1979 (one cohort, born 1957–64, followed week by week since 1978), six releases from 2015 to 2025, and the Current Population Survey tenure tables for the contrast with the stock of current jobs. Every number quoted in the article is printed by `tenure_numbers.py`, and the committed `output.txt` is what the script prints.

What the script does, section by section:

1. Transcribes Table 2 of six NLSY79 releases and documents a break between the 2019 and 2021 releases: the same cohort's first-year exit shares fall by 8–11 points while the text changes from "X percent of those jobs ended" to "the average worker had X percent of jobs end". The two releases measure two different statistics, one pooled over jobs (each hire counts once) and one averaged per worker.
2. Turns the published cells into survival curves from the start date (constant hazard between published cells), interval hazards, medians and restricted mean time with the employer within a horizon H.
3. Bounds the bias from jobs started too close to the survey date to have reached a duration cell.
4. Computes worst cases and break-even exit rates beyond the last published cell: how fast older hires would have to leave for a younger hire to deliver more time within H years.
5. Computes the expected number of hires needed to keep one position filled for H years (renewal function).
6. Builds a career model (piecewise-constant exit hazards by age at start and tenure, a gamma frailty, gaps between jobs, retirement), calibrates it to 23 published numbers (20 duration cells across both statistics, two shares of jobs still ongoing, and the worker-averaged first-year share of 25–34 starters, which fixes the frailty variance), checks it on 14 cells it was not fitted to, and shows that averaging per worker reproduces the 2019–2021 break with identical workers.
7. Compares the stock statistic (median tenure of current jobs, CPS January 2026) with the flow statistic (how long a new hire stays), including a counterfactual in which every age has the same hiring curve.
8. Reports the employer's view from the calibrated model: survival, restricted means and hires per seat to ten years.
9. Repeats the calibration under other assumptions for time between jobs and retirement age.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
python tenure_numbers.py > output.txt   # about two minutes
python figures.py
```

A GitHub Actions workflow (`.github/workflows/tenure-by-age-at-hire.yml`) reruns the script on every change and checks that the output matches the committed `output.txt`.

## Data

Transcribed by hand into `data/` from public BLS releases (U.S. government works); `data/SOURCES.md` lists the URL of every release and how the values were checked. No download is needed.

## Assumptions

The published tables give survival only at 1, 2, 5, 10 and 15 years; between those points the exit rate is taken as constant. The career model adds two assumptions that the tables cannot pin down: the time between jobs (exponential, mean 0.35 years) and the age of leaving the labour force (normal, mean 64, standard deviation 3, kept between 55 and 75). Section 9 of the output repeats the calibration with other values for both.

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
