# Survival Analysis for Data Scientists

Companion code for a six-part series of Medium articles that introduces survival analysis to readers who know basic probability and can read Python.

A great deal of business data is about time until something happens: a customer cancels, a loan defaults, a machine fails, an employee leaves. The table that holds it is a snapshot taken before many of those things have happened, and the records that have not reached the event yet are censored. Survival analysis is the set of methods that use those unfinished records correctly. Each part adds one idea and ends with a data-science application in which the method changes a decision. Every number quoted in an article is printed by a script in this folder.

## The series

| Part | Topic | Application | Folder | Article |
|---|---|---|---|---|
| 1 | Time to event, censoring and the Kaplan-Meier estimator | An acquisition budget from a customer table in which half the customers have not churned yet | [`part1/`](part1) | [Medium](https://medium.com/@diogo-ribeiro-1975/survival-analysis-for-data-scientists-27610dda2e62) |
| 2 | Hazards, the Nelson-Aalen estimator and the log-rank test | A retention A/B test read before most customers have left | [`part2/`](part2) | [Medium](https://medium.com/@diogo-ribeiro-1975/survival-analysis-for-data-scientists-7cce086e5a50) |
| 3 | The Cox model, checking proportional hazards and pricing customer segments | The most to pay for a customer from each lead segment | [`part3/`](part3) | [Medium](https://medium.com/@diogo-ribeiro-1975/survival-analysis-for-data-scientists-ad614e28eda8) |
| 4 | Parametric models, accelerated failure time and the value of customers beyond the data | A five-year acquisition budget from one year of data | [`part4/`](part4) | [Medium](https://medium.com/@diogo-ribeiro-1975/survival-analysis-for-data-scientists-8cb8b9b61bfa) |
| 5 | Competing risks, recurrent events and the price of a service contract | A five-year service contract for a pump fleet with breakdowns, planned replacements and repairs | [`part5/`](part5) | Forthcoming |
| 6 | Machine learning for survival, and how to evaluate it | A benchmark on public data, judged by the decision it supports | | Planned |

## Layout

```
survival-analysis-for-data-scientists/
├── README.md
├── requirements.txt       environment used for the committed outputs
└── partN/
    ├── <subject>.py       computations; prints every number used in the article
    ├── figures.py         draws the figures from the same functions
    ├── output.txt         reference output of the main script
    └── figures/           PNG files used in the article
```

## Running the code

Dependencies are local to this folder, as for every new project in this repository:

```bash
cd survival-analysis-for-data-scientists
pip install -r requirements.txt
```

The committed outputs were produced with Python 3.13 and the versions pinned in `requirements.txt`. Any NumPy 2.x with a recent SciPy and Matplotlib should give the same numbers up to the last printed digit; `scikit-survival` and `statsmodels` are optional.

To reproduce Part 1:

```bash
cd part1
python kaplan_meier.py > output.txt   # about 40 seconds
python figures.py                     # writes four PNG files to figures/, about 2 minutes
```

To reproduce Part 2:

```bash
cd part2
python hazards_logrank.py > output.txt   # about 25 seconds
python figures.py                        # writes four PNG files to figures/, about 30 seconds
```

To reproduce Part 3:

```bash
cd part3
python cox_model.py > output.txt   # about 80 seconds
python figures.py                  # writes four PNG files to figures/, about 75 seconds
```

To reproduce Part 4:

```bash
cd part4
python extrapolation.py > output.txt   # about 5 minutes
python figures.py                      # writes four PNG files to figures/, about 5 minutes
```

To reproduce Part 5:

```bash
cd part5
python competing_risks.py > output.txt   # about 15 seconds
python figures.py                        # writes four PNG files to figures/, about 15 seconds
```

Random seeds are fixed. The workflow `.github/workflows/survival-analysis-for-data-scientists.yml` reruns each part's main script on every change to this folder and fails if the output differs from the committed `output.txt`.

## Part 1 in brief

`part1/kaplan_meier.py` simulates the customer table of a subscription product 36 months after launch: customers sign up at a steady rate, each has a constant monthly churn rate of their own drawn from a gamma distribution, and those still subscribed at the snapshot are censored. It then:

1. Estimates retention with three shortcuts (drop the censored customers, count them as still active, use only customers with enough history) and with a Kaplan-Meier estimator written from scratch, with Greenwood standard errors and log-log intervals, checked against scikit-survival.
2. Turns each estimate into a break-even acquisition cost: a monthly margin of EUR 15 times the expected paid months in the first 24 (the restricted mean survival time).
3. Repeats the comparison on 1,000 simulated tables, measuring bias, spread and interval coverage.
4. Shows what happens when pauses in the subscription are recorded as censoring and the customers who pause are the ones more likely to churn.

The main results:

| Quantity | Value |
|---|---|
| True break-even acquisition cost | EUR 214 |
| Middle 95% of estimates: drop the censored / count them as active | EUR 113 to 120 / EUR 234 to 242 |
| Middle 95% of estimates: Kaplan-Meier / fixed horizon | EUR 210 to 218 / EUR 208 to 221 |
| Customers the fixed-horizon method needs to match Kaplan-Meier | 2.4 times as many |
| Coverage of the Kaplan-Meier 95% interval for 12-month retention | 95.4% |
| Kaplan-Meier break-even when pauses are censored (pause rate half the churn rate) | EUR 223 |

All data in Part 1 are simulated.

## Part 2 in brief

`part2/hazards_logrank.py` simulates a retention A/B test on the same product: 4,500 new customers randomised over three months to the current onboarding or a new one, analysed 3, 6, 9 and 12 months after the start. Two versions of the new onboarding are simulated: a *steady* one that lowers every customer's churn rate by 20%, and a *delay* one that halves churn for two months and raises it by half afterwards. The script:

1. Estimates the cumulative hazard with a Nelson-Aalen estimator and the monthly churn rate per customer-month at risk, both written from scratch and checked against scikit-survival.
2. Computes the true population hazard ratio when every customer's ratio is 0.8, to show how it drifts towards 1 as the arms' survivors become different kinds of customers.
3. Implements the log-rank test with the one-step (Peto) hazard ratio, and the difference in restricted mean survival time between arms with its standard error.
4. Runs 1,000 simulated experiments per scenario (no effect, steady, delay), comparing the log-rank test with a month-1 retention comparison and with the restricted-mean difference at each analysis date.

The main results:

| Quantity | Value |
|---|---|
| True effect on paid months in the first 12: steady / delay | +0.473 (EUR +7.10) / -0.301 (EUR -4.52) |
| Population hazard ratio when every customer's ratio is 0.8, at months 12 and 60 | 0.855 and 0.932 |
| Steady version detected at month 12: log-rank / month-1 comparison | 98.1% / 37.7% |
| No effect, log-rank significant at one or more of the four dates | 13.1% |
| Delay version at month 3: log-rank / month-1 comparison significantly in favour | 98.9% / 96.7% |
| Delay version at month 12: log-rank finds it harmful / restricted-mean interval below zero | 98.5% / 69.9% |
| Coverage of the 95% interval for the restricted-mean difference, all scenarios and dates | 94.2% to 96.1% |

All data in Part 2 are simulated.

## Part 3 in brief

`part3/cox_model.py` simulates the customer table of a software subscription for small businesses 36 months after launch: 8,000 customers from three acquisition channels and three company sizes, on monthly billing (can cancel at any time) or annual billing (can only leave at renewal). Every customer's decision to leave follows a proportional hazards model with a Weibull baseline. The script:

1. Fits the Cox model by Newton-Raphson on the partial likelihood, with Breslow's and Efron's methods for tied churn times and optional strata, checked against scikit-survival and statsmodels.
2. Computes Schoenfeld residuals and the Grambsch-Therneau test of proportional hazards, which flags the billing plan, and refits the model stratified by plan.
3. Converts hazard ratios into expected margin over 24 months with the Breslow estimate of the baseline cumulative hazard.
4. Values a customer from each of nine lead segments with Kaplan-Meier curves per segment and plan, the Cox model with the plan as a covariate, and the stratified Cox model, across 500 simulated tables.

The main results:

| Quantity | Value |
|---|---|
| Value of a referral rather than a search customer over 24 months (true hazard ratio 0.705 throughout) | EUR 7.33 to EUR 30.60, by plan and company size |
| Proportional-hazards test for the billing plan, model with plan as a covariate | significant in 500 of 500 tables |
| Mean error in segment values: Kaplan-Meier per cell / plan as covariate / stratified Cox | -0.47 to +0.33 / +3.10 to +6.34 / -0.21 to +0.25 EUR |
| Root mean squared error, average over segments (worst segment) | 4.97 (10.38) / 5.53 (7.72) / 3.26 (4.65) EUR |
| Mean hazard ratio for 50+ employees, true 0.607: Breslow / Efron ties | 0.620 / 0.609 |

All data in Part 3 are simulated.

## Part 4 in brief

`part4/extrapolation.py` simulates the subscription product of Parts 1 and 2 one year after launch: 6,000 customers, 30% of them arriving by referral, whose lifetimes are stretched by a time ratio of 1.43. It then:

1. Fits six parametric families (exponential, Weibull, Gompertz, log-normal, log-logistic, gamma mixture) as accelerated failure time models by maximum likelihood on censored data, checked against scipy's censored-data fitting.
2. Compares their fit to the observed year and their extrapolated five-year value of a customer, restricted and full means, and the time ratio for referral.
3. Simulates two further businesses whose retention matches the first over its first year (a Weibull process and one with a loyal core of customers who never leave), and evaluates four rules for setting the five-year value (best AIC, AIC-weighted average, observed value plus the lowest plausible tail, observed value only) with 12, 24 and 36 months of data, 100 tables each.

The main results:

| Quantity | Value |
|---|---|
| True five-year value of a paid-channel customer: gamma mixture / Weibull / loyal core | EUR 343.26 / 304.05 / 395.15 |
| Five-year values of the four plausible models in one table at 12 months | EUR 297.78 to 384.39 |
| Best-AIC rule at 12 months, gamma mixture business: middle 95% of errors | EUR -68.62 to +84.42 |
| Observed plus lowest plausible tail: overpays by more than 10% | at most 1% of tables in any business |
| Best-AIC root mean squared error at 12 / 24 / 36 months, gamma mixture business | EUR 36.59 / 10.57 / 4.82 |
| Time ratio for referral, gamma mixture model, mean over 100 tables (true 1.429) | 1.435 |

All data in Part 4 are simulated.

## Part 5 in brief

`part5/competing_risks.py` simulates a fleet of 3,000 industrial pumps installed over ten years. A pump leaves service by breaking down (a hazard that rises with age, higher for high-duty pumps) or by planned replacement (four times as frequent at sites in a modernisation programme); while in service it needs repeated repairs. The script prices a five-year service contract that pays for repairs and breakdowns until the pump leaves service. It:

1. Estimates the cumulative incidence of breakdown with the Aalen-Johansen estimator and compares it with one minus Kaplan-Meier.
2. Fits cause-specific Cox models and a Fine-Gray model for the subdistribution hazard, written from scratch.
3. Estimates the mean cumulative function of repairs and the expected number of repairs per contract, weighting by the probability of still being in service.
4. Prices the contract three ways (dashboard counts, one cause at a time, competing risks) by segment across 500 simulated fleets.

The main results:

| Quantity | Value |
|---|---|
| Breakdown within five years, one fleet: cumulative incidence / one minus Kaplan-Meier | 25.5% / 35.1% |
| Programme sites, mean over 500 fleets: cause-specific / subdistribution hazard ratio for breakdown | 0.990 / 0.457 |
| Repairs within five years, standard pumps outside the programme: per pump in service / per contract | 4.99 / 3.78 |
| Contract price error by segment: dashboard / one cause at a time / competing risks | -24.9% to -28.9% / +27.7% to +114.6% / within 0.6% |

All data in Part 5 are simulated.

## Author

Diogo Ribeiro
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
Contact: dfr@esmad.ipp.pt

Licensed under the repository's [Apache License 2.0](../LICENSE).
