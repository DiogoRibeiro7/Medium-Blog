# Medium publication reconciliation ledger

*Evidence snapshot: 3 November 2026. Sources: Medium account export supplied by the author and the repository tree. Do not publish or commit the private export itself.*

## Audit summary

- **89** repository manuscripts named `article.md` on `main`.
- **36** already indexed in [`ARTICLES.md`](ARTICLES.md); **53** outside that narrower chronological series.
- The author export contains **348 HTML post records**: **280** with a publication timestamp and canonical URL, plus **68** files explicitly marked as drafts. The 280 published records include responses/comments, so **280 is not a count of full-length articles**.
- **5** repository manuscripts matched export-confirmed published records, including one title truncated in the export. Each matched row below carries its canonical URL and date.
- **27** repository manuscripts have a matching Medium draft entry, but no published story was established for the corresponding manuscript in this pass. A draft can coexist with a separately published edition.
- **57** repository manuscripts have no reliable title match to a published post or exported draft yet. They need content-level reconciliation or author confirmation.
- Several other publicly published Medium articles are identified after the repository table. These are not assumed to be missing code or drafts.

### How status is assigned

| Status | Meaning |
| --- | --- |
| Published YYYY-MM-DD | Matching canonical story and `dt-published` timestamp found in the author's export. |
| Medium draft found; publication not verified | Matching `posts/draft_…html` entry; does not prove that another edition was never published. |
| Unresolved — no exact match | Neither a trustworthy title match nor an export draft match was established; it is not an unpublished verdict. |

Matches use normalized Unicode punctuation, full-title equality, or a distinctive truncated-prefix match. A partial semantic resemblance is **not** enough. Dates are taken from the export timestamps (UTC calendar dates).

## All 89 repository manuscripts

The series-index column is independent of publication state.

| # | Repository manuscript | Series index | Export evidence / publication status |
| ---: | --- | :---: | --- |
| 1 | [A Correlation Is Not a Mechanism: Confounding, Collider Bias, and the Limits of Observational Evidence](a-correlation-is-not-a-mechanism/article.md) | Yes | Unresolved — no exact match |
| 2 | [A Dataset Is Not a Population: Selection, Coverage, Survey Weights, and the Limits of Representativeness](a-dataset-is-not-a-population/article.md) | Yes | Unresolved — no exact match |
| 3 | [A Label Is Not Ground Truth: Label Noise, Annotator Disagreement, Selective Labels, and the Illusion of Benchmark Accuracy](a-label-is-not-ground-truth/article.md) | Yes | Unresolved — no exact match |
| 4 | [Do Callers Who Hang Up Call Back? Testing an Interval-Level Redial Estimator Against the Caller IDs of a Real Call Centre](anonymous-bank-redials/article.md) | No | Medium draft found; publication not verified |
| 5 | [When Regression Gets the Causal Effect Wrong](causal_inference_statistics/part1/article.md) | No | Unresolved — no exact match |
| 6 | [Doubly Robust Estimation: When One Correct Model Is Enough—and When Neither Is](causal_inference_statistics/part10/article.md) | No | Unresolved — no exact match |
| 7 | [The G-Formula: Why Adjusting for a Time-Varying Confounder Can Miss the Total Effect](causal_inference_statistics/part11/article.md) | No | Unresolved — no exact match |
| 8 | [Heterogeneous Treatment Effects: The Best Predictor Is Not Always the Best Treatment Rule](causal_inference_statistics/part12/article.md) | No | Unresolved — no exact match |
| 9 | [Causal Machine Learning and Double Machine Learning: When Flexibility Helps—and When It Does Not](causal_inference_statistics/part13/article.md) | No | Unresolved — no exact match |
| 10 | [Bayesian Causal Inference: When a Precise Posterior Answers the Wrong Question](causal_inference_statistics/part14/article.md) | No | Unresolved — no exact match |
| 11 | [Time-Varying Treatments and Marginal Structural Models: Why Ordinary Adjustment Can Remove the Effect](causal_inference_statistics/part15/article.md) | No | Unresolved — no exact match |
| 12 | [Sensitivity Analysis for Unmeasured Confounding: How Much Hidden Bias Changes the Result](causal_inference_statistics/part16/article.md) | No | Unresolved — no exact match |
| 13 | [Negative Controls, Placebo Tests and Falsification: What a Passing Test Cannot Prove](causal_inference_statistics/part17/article.md) | No | Unresolved — no exact match |
| 14 | [Missing Outcomes and Selection Bias: When Complete Cases Mislead](causal_inference_statistics/part18/article.md) | No | Unresolved — no exact match |
| 15 | [External Validity and Transportability: When a Valid Trial Answers the Wrong Population Question](causal_inference_statistics/part19/article.md) | No | Unresolved — no exact match |
| 16 | [Randomization, Confounding and the Value of Experimental Design](causal_inference_statistics/part2/article.md) | No | Unresolved — no exact match |
| 17 | [From Statistical Evidence to Causal Decisions: When the Best Prediction Makes the Worst Policy](causal_inference_statistics/part20/article.md) | No | Unresolved — no exact match |
| 18 | [Matching and Propensity Scores: When Balanced Data Still Give the Wrong Causal Effect](causal_inference_statistics/part3/article.md) | No | Unresolved — no exact match |
| 19 | [Inverse Probability Weighting: When a Few Observations Control the Result](causal_inference_statistics/part4/article.md) | No | Unresolved — no exact match |
| 20 | [Directed Acyclic Graphs and the Cost of Controlling for Everything](causal_inference_statistics/part5/article.md) | No | Unresolved — no exact match |
| 21 | [Difference-in-Differences: When Parallel Trends Fail](causal_inference_statistics/part6/article.md) | No | Unresolved — no exact match |
| 22 | [Instrumental Variables: When a Strong First Stage Is Not Enough](causal_inference_statistics/part7/article.md) | No | Unresolved — no exact match |
| 23 | [Regression Discontinuity: Why a Sharp Cutoff Is Not Enough](causal_inference_statistics/part8/article.md) | No | Unresolved — no exact match |
| 24 | [Synthetic Controls and Counterfactual Time Series: When a Good Pre-Treatment Fit Is Not Enough](causal_inference_statistics/part9/article.md) | No | Unresolved — no exact match |
| 25 | [A Randomized Trial Is Not a Universal Policy](causal-transportability-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 26 | [The Right to Vote Was Only the Beginning](civil-rights-great-society/article.md) | Yes | Unresolved — no exact match |
| 27 | [When Power Calls for Civility: How Moderation Can Contain Social Change](civility-and-political-change/article.md) | Yes | Unresolved — no exact match |
| 28 | [A Cluster Is Not a Population: K-means, Mixtures, Stability, and the Limits of Unsupervised Labels](clustering-is-not-a-population/article.md) | Yes | Unresolved — no exact match |
| 29 | [A 90% Interval Is Not a Promise](conformal-prediction-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 30 | [The Lowest RMSE Is Not the Best Decision](decision-focused-optimization-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 31 | [A Thousand Observations Are Not a Thousand Replicates](dependent-data-uncertainty-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 32 | [Okun's Law Across the EU, 2009–2026](eu-okun-law/publication/medium/article.md) | No | Unresolved — no exact match |
| 33 | [Heat, Drought and Europe's Wheat](eu-wheat-weather/publication/article.md) | No | [Published 2026-10-28](https://medium.com/@diogo-ribeiro-1975/heat-drought-and-europes-wheat-82f200357074) |
| 34 | [Heat, Drought and Europe's Maize](eu-wheat-weather/publication/maize/article.md) | No | [Published 2026-10-31](https://medium.com/@diogo-ribeiro-1975/heat-drought-and-europes-maize-b239c6a2c30d) |
| 35 | [The Tail Is Not a Rounding Error: Extreme Value Theory for Data Scientists](extreme-value-theory-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 36 | [A Feature Importance Score Is Not a Causal Effect: Permutation, Dependence, and the Limits of Model Explanations](feature-importance-is-not-causality/article.md) | Yes | Unresolved — no exact match |
| 37 | [The Boom Was Not a Balance Sheet](financialization-before-2008/article.md) | Yes | Unresolved — no exact match |
| 38 | [A Missing Edge Is Not a Missing Relationship: Gaussian Graphical Models and Conditional Independence](gaussian-graphical-models-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 39 | [The Gold Was Real. So Was the Gilding: The Political Economy of America's Gilded Age](gilded-age-political-economy/article.md) | Yes | Unresolved — no exact match |
| 40 | [The Recession Ended; the Damage Did Not](great-recession-unequal-recovery/article.md) | Yes | Unresolved — no exact match |
| 41 | [Scientist, Engineer, Architect: A Fourteen-Month Replay of a Hotel Cancellation System](hotel-cancellation-replay/article.md) | No | Medium draft found; publication not verified |
| 42 | [Where to Put the Learned Correction in a Hybrid Model: Outputs, Inputs and Structure on a Simulated Contact Centre](hybrid-discrepancy-generators/article.md) | No | Medium draft found; publication not verified |
| 43 | [Rebuilding the Iberian Day-Ahead Electricity Price from OMIE's Order Books](iberian-electricity-lp/publication/article.md) | No | [Published 2026-10-30](https://medium.com/@diogo-ribeiro-1975/rebuilding-the-iberian-day-ahead-electricity-price-from-omies-order-books-6cebd3f5da00) |
| 44 | [A Factory Is Not a Histogram: Industrial Statistics for Data Scientists](industrial-statistics-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 45 | [A Good Fit Is Not an Identified Model: Inverse Problems for Data Scientists](inverse-problems-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 46 | [Testing MC Dropout as a Posterior: Concentration, Benchmark Scores and Distribution Shift](mc-dropout-posterior/article.md) | No | Medium draft found; publication not verified |
| 47 | [Correcting Regressions for Measurement Error With Replicates, Validation Samples and Instruments](measurement-error/article.md) | No | Medium draft found; publication not verified |
| 48 | [Mechanistic and Data-Driven Synthetic Data Generators: The Trade-Offs, Measured on a Simulated Contact Centre](mechanistic-vs-data-driven-generators/article.md) | No | Medium draft found; publication not verified |
| 49 | [The Forecast Is Not the Fit: Rebuilding an 89% Midterm Election Model](midterm-forecast-fit-vs-prediction/article.md) | Yes | [Published 2026-11-01](https://medium.com/@diogo-ribeiro-1975/the-forecast-is-not-the-fit-rebuilding-an-89-midterm-election-model-8fb36cc269be) |
| 50 | [A Missing Value Is Not a Zero: MCAR, MAR, MNAR, Imputation, and the Limits of Complete Cases](missing-data-is-not-zero/article.md) | Yes | Unresolved — no exact match |
| 51 | [Mixed-Effects Models for Grouped Data: Partial Pooling, Within–Between Confounding and Honest Standard Errors](mixed-effects-models/article.md) | No | Medium draft found; publication not verified |
| 52 | [The Monty Hall Problem, 1889–2026: A History With the Calculations Redone](monty-hall/part1/article.md) | No | Medium draft found; publication not verified |
| 53 | [Which Rules of the Monty Hall Game Change the Decision? A Value-of-Information Analysis of the Host's Protocol](monty-hall/part2/article.md) | No | Medium draft found; publication not verified |
| 54 | [The Host Is a Missingness Mechanism: Monty Hall, Ignorability and the Self-Selected Ratings of the Coat Data](monty-hall/part3/article.md) | No | Medium draft found; publication not verified |
| 55 | [Repeated Play of the Monty Hall Game: Learning Curves, Probability Matching and Models of Why People Stay](monty-hall/part4/article.md) | No | Medium draft found; publication not verified |
| 56 | [Missing Data and Multiple Imputation: Thirteen Methods Compared on Census Wages and NHANES Blood Pressure](multiple-imputation/article.md) | No | Medium draft found; publication not verified |
| 57 | [A Nearest Neighbor Is Not Necessarily Similar: Distance Concentration, Hubness, and the Geometry of High-Dimensional Data](nearest-neighbor-geometry-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 58 | [A Safety Net Is Not a Social Contract: What the New Deal Built—and Whom It Left Out](new-deal-social-contract/article.md) | Yes | Unresolved — no exact match |
| 59 | [An Outlier Is Not an Error: Robust Estimation, Influence, Heavy Tails, and the Limits of Anomaly Scores](outlier-is-not-an-error/article.md) | Yes | Unresolved — no exact match |
| 60 | [Inflation Fell; Prices Did Not: The Pandemic Shock, Inflation, and the Unequal Adjustment, 2020–2025](pandemic-inflation-unequal-recovery/article.md) | Yes | Unresolved — no exact match |
| 61 | [A Principal Component Is Not a Latent Factor: Eigenvectors, Noise, and the Limits of PCA](pca-vs-latent-factors-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 62 | [Basic Needs, Saving and Inflation by Income Group in Portugal, 1988–2025](portugal-household-spending/article.md) | No | Medium draft found; publication not verified |
| 63 | [Pass-Through of Portugal's 2023 Zero VAT on Food, Estimated from Official Price Indices](portugal-iva-zero/article.md) | No | Medium draft found; publication not verified |
| 64 | [Inflation-Plus-Productivity Wage Counterfactuals for Portugal, Computed for Every Base Year from 1975](portugal-wage-counterfactual/article.md) | No | Medium draft found; publication not verified |
| 65 | [Prescribing for a Patient, Not an Age Group: ADHD Medication After 65 in Four Public Datasets](prescribing-individuals/article.md) | No | Unresolved — no exact match |
| 66 | [A Law Is Not a Reform: What the Progressive Era Actually Changed](progressive-era-reform-gap/article.md) | Yes | Unresolved — no exact match |
| 67 | [The Queue Behind the Forecast: Why Good Demand Predictions Still Produce Bad Waiting-Time Decisions](queueing-theory-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 68 | [An Alarm Score Is Not a Probability: Rare Events, Base Rates, and Decision Thresholds](rare-event-detection-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 69 | [Galton's Heights, Pearson's Fathers and Secrist's Firms: The History of Regression to the Mean, With the Data Recomputed](regression-to-the-mean/part1/article.md) | No | Medium draft found; publication not verified |
| 70 | [The Arithmetic of Regression to the Mean: Reliability, Selection Thresholds, Non-Normal Tails and the Designs That Separate It From an Effect](regression-to-the-mean/part2/article.md) | No | Medium draft found; publication not verified |
| 71 | [Predicted Against Observed: Regression to the Mean in NHANES Blood-Pressure Readings and Seven Decades of Major League Seasons](regression-to-the-mean/part3/article.md) | No | Medium draft found; publication not verified |
| 72 | [Where Data Scientists Select on Noise: A/B Test Winners, Programmes for the Worst Performers and Alerts on Daily Metrics](regression-to-the-mean/part4/article.md) | No | Medium draft found; publication not verified |
| 73 | [A Machine Does Not Fail at Its Mean](reliability-statistics-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 74 | [The Economics of the Uneaten Meal](restaurant-portion-economics/publication/medium/article.md) | No | Medium draft found; publication not verified |
| 75 | [Robust Regression Under Data-Entry Errors: Breakdown, Efficiency and the Outliers That Are the Signal](robust-regression/article.md) | No | Medium draft found; publication not verified |
| 76 | [Straight Lines, Real Roads](routing-on-real-roads/publication/article.md) | No | Medium draft found; publication not verified |
| 77 | [The Alarm Is Not the Change: Sequential Change-Point Detection for Data Scientists](sequential-change-detection-for-data-scientists/article.md) | Yes | Unresolved — no exact match |
| 78 | [A Significant Result Is Not a Replicable Effect](significance-selection-winners-curse/article.md) | Yes | Unresolved — no exact match |
| 79 | [What the Classic Examples of Simpson's Paradox Actually Show: Yule, Simpson, Berkeley and the Kidney Stones, Recomputed With Their Uncertainty](simpsons-paradox/part1/article.md) | No | Medium draft found; publication not verified |
| 80 | [Choosing Between the Aggregate and the Stratified Table: Causal Structure, Adjustment and the Limits of Data in Simpson's Paradox](simpsons-paradox/part2/article.md) | No | Medium draft found; publication not verified |
| 81 | [Reversals in Real Data: COVID-19 Fatality by Age, SAT Scores and Test Participation, Penguin Bills by Species, and How Often Chance Produces Them](simpsons-paradox/part3/article.md) | No | Medium draft found; publication not verified |
| 82 | [Simpson's Paradox in Analytics Practice: Ramped A/B Tests, Mix Shifts in KPIs, Conditional Fairness Metrics and Automated Reversal Scans](simpsons-paradox/part4/article.md) | No | Medium draft found; publication not verified |
| 83 | [A High R² Is Not a Long-Run Relationship: Spurious Regression, Unit Roots, and Cointegration](spurious-regression-cointegration/article.md) | Yes | Unresolved — no exact match |
| 84 | [The Price of Stability Was Not Shared Equally](stagflation-market-turn/article.md) | Yes | Unresolved — no exact match |
| 85 | [Evaluating Synthetic Tabular Data: Fidelity, Utility, Privacy and Valid Inference](synthetic-data-evaluation/article.md) | No | Medium draft found; publication not verified |
| 86 | [The Promise Returns: Tax Cuts for the Rich, Historical Recurrence, and the Distribution of Prosperity](tax-cuts-rich-historical-cycles/article.md) | Yes | Unresolved — no exact match |
| 87 | [How Long New Hires Stay, by Age at Hire: Survival From the Start Date in Four Decades of U.S. Job Records](tenure-by-age-at-hire/article.md) | No | [Published 2026-10-30](https://medium.com/@diogo-ribeiro-1975/how-long-new-hires-stay-by-age-at-hire-survival-from-the-start-date-in-four-decades-of-u-s-ed920b91a645) |
| 88 | [The Test Set Is Not an Oracle: Selection Bias, Leakage, Temporal Splits, and Adaptive Model Evaluation](the-test-set-is-not-an-oracle/article.md) | Yes | Unresolved — no exact match |
| 89 | [The Suburb Is Not an Escape: How Housing Unaffordability Can Spread Across a Metropolitan Area](urban-affordability-cascade/article.md) | Yes | Unresolved — no exact match |

## Confirmed Medium posts not matched to the 89 `article.md` paths

These Medium links are genuine author publications, but no corresponding **same-title `article.md` path** was identified in the inventory. They may correspond to other filenames, repositories, or earlier drafts; avoid treating them as missing repository work without a content comparison.

| Published article | Medium URL |
| --- | --- |
| Survival Analysis for Data Scientists — Part 4 | [View publication](https://diogo-ribeiro-1975.medium.com/survival-analysis-for-data-scientists-8cb8b9b61bfa) |
| A Placebo Can Pass and Still Be as Large as the Estimate | [View publication](https://diogo-ribeiro-1975.medium.com/a-placebo-can-pass-and-still-be-as-large-as-the-estimate-5a8747a662bb) |
| Game and Auction Theory for Data Scientists — Part 4 | [View publication](https://diogo-ribeiro-1975.medium.com/game-and-auction-theory-for-data-scientists-92803a10d8a6) |
| An Alert Is Not an Outcome | [View publication](https://diogo-ribeiro-1975.medium.com/an-alert-is-not-an-outcome-459c0367df8c) |
| Survival Analysis for Data Scientists — Part 3 | [View publication](https://diogo-ribeiro-1975.medium.com/survival-analysis-for-data-scientists-ad614e28eda8) |
| Calibrating Wald Tests with Newey-West Standard Errors | [View publication](https://diogo-ribeiro-1975.medium.com/calibrating-wald-tests-with-newey-west-standard-errors-3907ae453459) |
| Portugal’s Housing Market, 2008–2025 | [View publication](https://diogo-ribeiro-1975.medium.com/portugals-housing-market-2008-2025-a-reproducible-analysis-of-prices-incomes-construction-and-18c715332adf) |
| Mastering Data Preprocessing for Machine Learning | [View publication](https://medium.com/data-and-beyond/mastering-data-preprocessing-for-machine-learning-b680ef384bf7) |
| The Dual Identity of Logistic Regression | [View publication](https://medium.com/data-and-beyond/the-dual-identity-of-logistic-regression-3b2c63f3ab46) |
| Understanding Bayes Theorem: The Mathematics and Applications | [View publication](https://medium.com/data-science-as-a-better-idea/understanding-bayes-theorem-the-mathematics-and-applications-a85b68aaf8e9) |


## Next verification stage

1. Compare article body fingerprints against the exported full text, allowing for Medium copy edits and title changes.
2. Separate full articles from short responses in the 280 published records, without deleting or exposing private draft contents.
3. Review duplicate editions sharing subject matter; prefer Medium story IDs over fuzzy titles.
4. Identify which of the **57 unresolved** repository manuscripts are truly unpublished, published under another title, or superseded.
5. Keep `ARTICLES.md` as the series-development chronology. Maintain publication verification here, and only sync verified canonical links into other indexes.
6. Re-run this audit when new Medium export data are supplied. The export has intentionally **not** been added to source control.

## Privacy and evidence notes

This ledger contains repository titles and confirmed canonical public URLs. It does not contain the uploaded export, private draft text, account sessions, IP data, bookmarks, or reading history. The export itself should remain outside the public repository.
