# Survival Analysis for Data Scientists

Companion code for a six-part series of Medium articles that introduces survival analysis to readers who know basic probability and can read Python.

A great deal of business data is about time until something happens: a customer cancels, a loan defaults, a machine fails, an employee leaves. The table that holds it is a snapshot taken before many of those things have happened, and the records that have not reached the event yet are censored. Survival analysis is the set of methods that use those unfinished records correctly. Each part adds one idea and ends with a data-science application in which the method changes a decision. Every number quoted in an article is printed by a script in this folder.

## The series

| Part | Topic | Application | Folder | Article |
|---|---|---|---|---|
| 1 | Time to event, censoring and the Kaplan-Meier estimator | An acquisition budget from a customer table in which half the customers have not churned yet | [`part1/`](part1) | Forthcoming |
| 2 | Hazards, Nelson-Aalen and comparing groups | A retention A/B test read with the log-rank test | | Planned |
| 3 | The Cox proportional hazards model | Drivers of churn, translated into expected months | | Planned |
| 4 | Parametric models, accelerated failure time and extrapolation | Customer lifetime value beyond the observed window | | Planned |
| 5 | Competing risks and recurrent events | Predictive maintenance with failure and planned replacement | | Planned |
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

The committed outputs were produced with Python 3.13 and the versions pinned in `requirements.txt`. Any NumPy 2.x and a recent Matplotlib should give the same numbers up to the last printed digit; `scikit-survival` is optional.

To reproduce Part 1:

```bash
cd part1
python kaplan_meier.py > output.txt   # about 40 seconds
python figures.py                     # writes four PNG files to figures/, about 2 minutes
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

## Author

Diogo Ribeiro
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
Contact: dfr@esmad.ipp.pt

Licensed under the repository's [Apache License 2.0](../LICENSE).
