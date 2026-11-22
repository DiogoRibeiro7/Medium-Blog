<p align="center">
  <img src="assets/project-logo.png" alt="Medium-Blog project logo" width="160" height="160">
</p>

# Medium-Blog

[![GitHub issues](https://img.shields.io/github/issues/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/issues)
[![GitHub stars](https://img.shields.io/github/stars/DiogoRibeiro7/Medium-Blog)](https://github.com/DiogoRibeiro7/Medium-Blog/stargazers)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

The code, data, results and figures behind my articles on [Medium](https://medium.com/@diogo-ribeiro-1975). Every article that reports a number has a folder here with the code that produces it. The manuscripts are kept separately, in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium) repository.

## How the articles link here

Recent articles link to their folder at a pinned commit, so their figures and numbers keep pointing at the code that produced them after the folder changes. Two rules follow from that:

- Merge pull requests with a merge commit, not a squash or a rebase, so the commit an article pins stays in the history of `main`.
- Do not rewrite the history of `main` or rename a folder an article links to. Older folders keep their original names, spaces included, for this reason. If a folder has to move, leave a README in its place that says where it went, as [`portugal-housing-generations/`](portugal-housing-generations/) does.

Some older folders still hold an `article.md`. It is a short notice pointing to the private repository, not the manuscript; [`scripts/check_public_editorial.py`](scripts/check_public_editorial.py) fails the build if a full manuscript is committed here. The move of the manuscripts is recorded in [EDITORIAL_MIGRATION.md](EDITORIAL_MIGRATION.md).

## What a project folder contains

Each folder stands on its own: there is no repository-wide environment or package. Most recent folders have:

- a `README.md` with the question, the data sources, how to rerun the analysis, and which file each figure and number comes from;
- a `requirements.txt` with pinned versions;
- the analysis scripts, with their `results/` and `figures/` committed;
- `tests/` that check the methods on cases with a known answer;
- the data, either downloaded and checked against a pinned digest or committed as a snapshot. Where a source cannot be reached from where the analysis was written, a separate `<folder>-data.yml` workflow downloads it on GitHub's runners and commits it;
- its own workflow, `.github/workflows/<folder>.yml`, which runs only when that folder changes. It lints the code, runs the tests and reruns the analysis against the committed results.

To rerun a project:

```bash
cd <folder>
python -m pip install -r requirements.txt
```

then follow the folder's README.

## Projects

The list below is generated from [`projects.yml`](projects.yml) and the first heading of each folder's README. Do not edit it by hand.

<!-- projects:start -->

235 top-level folders: 77 belong to 24 series, 91 are standalone studies in 7 themes, 5 hold repository tooling and 52 hold older material. 10 not yet sorted.

### Series: statistics, machine learning and data engineering

#### Causal Inference in Statistics

Twenty simulation experiments with a known causal effect, one estimator or design per part.

- [`causal_inference_statistics/`](causal_inference_statistics/) — All twenty parts: simulations, result tables and audits

#### Survival Analysis for Data Scientists

A six-part introduction to time-to-event analysis.

- [`survival-analysis-for-data-scientists/`](survival-analysis-for-data-scientists/) — Survival Analysis for Data Scientists

#### The Monty Hall Problem

Four parts on the host protocol and selection.

- [`monty-hall/`](monty-hall/) — The Monty Hall problem, in four parts

#### Simpson's Paradox

Four parts on which table answers the question.

- [`simpsons-paradox/`](simpsons-paradox/) — Simpson's paradox in four parts

#### Regression to the Mean

Four parts on selecting on a noisy measurement.

- [`regression-to-the-mean/`](regression-to-the-mean/) — Regression to the mean in four parts

#### Statistical Process Control

Estimated control limits on production lines and the in-control ARL.

- [`statistical-process-control/`](statistical-process-control/) — Statistical process control in four parts
- [`drift-detectors-as-control-charts/`](drift-detectors-as-control-charts/) — Drift detectors as control charts

#### Backward Evaluation of Forecasting Models

Training on the present and testing on the past, with pre-registered tests.

- [`backward-evaluation/`](backward-evaluation/) — Testing time-series models against the past: the train-period × test-period matrix
- [`backward-evaluation-pooling/`](backward-evaluation-pooling/) — Does stronger evidence of a change mean the change will last? A pre-registered test
- [`backward-evaluation-episode-end/`](backward-evaluation-episode-end/) — An end-of-episode check for the backward training-window rule
- [`backward-evaluation-generality/`](backward-evaluation-generality/) — Backward training windows beyond economics: electricity load and Wikipedia traffic
- [`backward-evaluation-model-choice/`](backward-evaluation-model-choice/) — Choosing a forecasting model by backward evaluation: a pre-registered test
- [`backward-evaluation-guide/`](backward-evaluation-guide/) — Reading the backward-evaluation matrix: a diagnostic, not a rule

#### Overcoming Data Challenges

One method per data problem, each with a real-data experiment.

- [`mixed-effects-models/`](mixed-effects-models/) — Mixed-effects models on three public data sets
- [`multiple-imputation/`](multiple-imputation/) — Multiple imputation on real data
- [`robust-regression/`](robust-regression/) — Robust regression on real data
- [`measurement-error/`](measurement-error/) — Measurement error in regression
- [`heteroscedasticity/`](heteroscedasticity/) — Heteroscedasticity: standard errors, weights and prediction intervals
- [`sample-selection/`](sample-selection/) — Sample selection: what identifies a Heckman correction
- [`excess-zeros/`](excess-zeros/) — Excess zeros: overdispersion versus zero-inflation
- [`rare-events/`](rare-events/) — Rare events: fix the threshold, not the data

#### Synthetic Data for a Contact Centre

Mechanistic, data-driven and hybrid generators, tested on a real call centre.

- [`mechanistic-vs-data-driven-generators/`](mechanistic-vs-data-driven-generators/) — Mechanistic and data-driven synthetic data generators, compared on a simulated contact centre
- [`hybrid-discrepancy-generators/`](hybrid-discrepancy-generators/) — Erlang A plus a learned correction: outputs, inputs or structure
- [`anonymous-bank-redials/`](anonymous-bank-redials/) — Redials in a real call centre: the interval-level estimator against caller IDs

#### Actuarial Data Science: Discrimination, Calibration and the Tariff

Why a good Gini is not a good tariff.

- [`actuarial-tariff-calibration/`](actuarial-tariff-calibration/) — Actuarial data science: discrimination, calibration and the tariff

#### Code Quality for Data Work

SQL quality, pipeline tests and data tests.

- [`analytics-sql-quality/`](analytics-sql-quality/) — Code quality for analytics SQL, measured
- [`ml-pipeline-tests/`](ml-pipeline-tests/) — Testing a machine learning pipeline end to end
- [`data-tests-in-production/`](data-tests-in-production/) — Testing data in production: test the change, not the table
- [`data-tests-limits/`](data-tests-limits/) — What a delivery cannot vouch for: dictionary rules and random row losses

#### Correctness Over Time in Data Pipelines

Where time makes correct pipelines give wrong numbers.

- [`pipeline-time-correctness/`](pipeline-time-correctness/) — Correctness Over Time in Data Pipelines

#### The 2026 Midterm Forecast

Fit against prediction in an election model.

- [`midterm-forecast-fit-vs-prediction/`](midterm-forecast-fit-vs-prediction/) — The Forecast Is Not the Fit: Rebuilding an 89% Midterm Election Model

### Series: economics and economic history

#### U.S. Political Economy, 1870–2025

Nine periods of U.S. economic history measured with public data.

- [`gilded-age-political-economy/`](gilded-age-political-economy/) — The Gilded Age in Data, Power, and Political Economy
- [`progressive-era-reform-gap/`](progressive-era-reform-gap/) — From Statute to Enforcement
- [`new-deal-social-contract/`](new-deal-social-contract/) — The New Deal as Risk Insurance
- [`civil-rights-great-society/`](civil-rights-great-society/) — Enforcing Rights and Measuring Poverty
- [`stagflation-market-turn/`](stagflation-market-turn/) — Stagflation, Disinflation and the Distribution of Adjustment Costs
- [`financialization-before-2008/`](financialization-before-2008/) — Trade, Household Debt and Housing Collateral
- [`great-recession-unequal-recovery/`](great-recession-unequal-recovery/) — Rescue and Recovery After 2008
- [`pandemic-inflation-unequal-recovery/`](pandemic-inflation-unequal-recovery/) — Measuring the Pandemic Recovery
- [`tax-cuts-rich-historical-cycles/`](tax-cuts-rich-historical-cycles/) — Tax Cuts at the Top

#### The Euro-Area Debt Crisis and Its Aftermath

The 2010–2025 ledger, France in 2026, Germany, the ECB and Portugal.

- [`euro-crisis-ledger/`](euro-crisis-ledger/) — A ledger of the euro-area crisis, 2010–2025
- [`france-spread-2026/`](france-spread-2026/) — France's Bond Spread in 2026, Measured Against 2011: Co-Movement, Pass-Through and the Interest Bill to 2036
- [`german-public-investment/`](german-public-investment/) — Germany's Public Capital Stock Under the Debt Brake: Net Investment, Fiscal Room and the 2025 Reform Against Five Low-Debt Peers
- [`ecb-france-backstop/`](ecb-france-backstop/) — The ECB's Backstop and France: Warranted Spreads, the Interest Bill to 2036 and Four Channels to the Rest of the EU
- [`portugal-2010-contagion/`](portugal-2010-contagion/) — Was Portugal Priced on Its Own Fundamentals in 2010-2012? Warranted Spreads, Contagion from Greece and Ireland, and Five European Decisions
- [`portugal-debt-management/`](portugal-debt-management/) — Portugal's Debt Management After the Rescue, Priced Instalment by Instalment: IMF and European Loans, Bond Exchanges and Savings Certificates, 2015-2026

#### EU Dependence on Suppliers Outside the Bloc

Dependence product by product, a redesigned screen and its lead time.

- [`europe-deindustrialisation/`](europe-deindustrialisation/) — The EU's dependence on suppliers outside the bloc, and what deindustrialisation had to do with it
- [`dependency-screen-design/`](dependency-screen-design/) — From concentration to exposure: a supply-based screen for EU import dependence
- [`dependency-early-warning/`](dependency-early-warning/) — Lead time in import-dependence screening

#### Capacity Lock-In, Shocks and Recovery

When a market can rebuild lost capacity.

- [`market-capacity-lock-in/`](market-capacity-lock-in/) — When the market stops correcting: capacity lock-in in food and housing
- [`eu-integration-correlated-shocks/`](eu-integration-correlated-shocks/) — Integration as insurance: correlated supply shocks in the EU single market
- [`recovery-after-busts/`](recovery-after-busts/) — Recovery after capacity busts in EU industries
- [`cure-fraction-identifiability/`](cure-fraction-identifiability/) — How much follow-up does a mixture cure model need?

#### Heat, Drought and Europe's Crops

Weather and yields across EU regions.

- [`eu-wheat-weather/`](eu-wheat-weather/) — Heat, drought and wheat yields in EU regions

#### Portuguese Wages and the Cost of Living since 1975

Counterfactual wages and household spending on basic needs.

- [`portugal-wage-counterfactual/`](portugal-wage-counterfactual/) — Portuguese wages against inflation plus productivity, 1975–2024
- [`portugal-household-spending/`](portugal-household-spending/) — Portuguese household spending on basic needs, saving, and basic-needs inflation

#### Wage Setting in the EU, with Portugal as a Special Case

Labour law as auction and bargaining, and scenario forecasts.

- [`eu-wage-bargaining-power/`](eu-wage-bargaining-power/) — Wage setting in the EU as auction and bargaining, with Portugal as a special case
- [`eu-wage-policy-forecast/`](eu-wage-policy-forecast/) — Forward scenarios for wage-setting rules: Portugal's minimum wage to 2029, pay transparency, and the EU's 60% reference

#### Europe's Low-Wage Path, Measured

Four measurements behind the low-wage argument, and their follow-ups.

- [`eu-us-productivity-gap/`](eu-us-productivity-gap/) — What GDP per hour measures: decomposing the EU–US productivity gap
- [`china-manufacturing-labour-costs/`](china-manufacturing-labour-costs/) — China's manufacturing labour costs against Mexico, Vietnam and the EU, 1995–2022
- [`minimum-wage-in-work-poverty/`](minimum-wage-in-work-poverty/) — Minimum wages, the middle of the distribution and in-work poverty in Portugal and the South
- [`eu-capital-exports/`](eu-capital-exports/) — Saving, investment and cross-border capital flows of the EU27: what "€300 billion a year" measures
- [`eu-us-surplus-by-industry/`](eu-us-surplus-by-industry/) — Where North-West Europe's lower operating surplus per hour sits: an industry decomposition against the US, 2019 and 2021
- [`china-capital-deepening/`](china-capital-deepening/) — Capital deepening in Chinese manufacturing and in the Chinese economy, 1995–2017 (and what came after)
- [`eu-us-ppp-drift/`](eu-us-ppp-drift/) — The EU–US drift between price bases: does ICT deflation explain it?
- [`ppp-drift-pairs/`](ppp-drift-pairs/) — The price drift between every pair of economies, and who compiles their PPPs
- [`ie-lu-fund-intermediation/`](ie-lu-fund-intermediation/) — Who the EU's purchases of US securities are made for: Irish and Luxembourg funds in mirror statistics
- [`ppp-drift-by-category/`](ppp-drift-by-category/) — The PPP drift inside Europe, category by category

#### Billionaire Density and Poverty

The 54-country reanalysis and its follow-ups.

- [`billionaires-poverty-audit/`](billionaires-poverty-audit/) — Billionaire Density and Poverty Across 54 Countries
- [`billionaires-poverty-followups/`](billionaires-poverty-followups/) — Billionaire density and poverty — follow-up series

#### Wealth Concentration and Growth

Wealth against productivity across countries, then inherited and self-made billionaire wealth.

- [`wealth-concentration-misallocation/`](wealth-concentration-misallocation/) — Wealth concentration, the coupling of wealth to productivity, and growth
- [`wealth-origin-growth/`](wealth-origin-growth/) — Inherited and self-made billionaire wealth and growth within countries
- [`wealth-origin-sectors/`](wealth-origin-sectors/) — Inherited billionaire wealth, sector composition and productivity growth
- [`wealth-origin-pandemic/`](wealth-origin-pandemic/) — Inherited billionaire wealth, pandemic exposure and productivity growth
- [`wealth-origin-successions/`](wealth-origin-successions/) — Founder-to-heir successions and productivity growth: a country-level event study
- [`wealth-origin-synthesis/`](wealth-origin-synthesis/) — What five studies established about inherited billionaire wealth and growth

#### Taxes, Efficiency and the Price of Medicines

What a single public buyer obtains for medicines, compared with the United States, and the follow-ups.

- [`taxation-efficiency-drug-prices/`](taxation-efficiency-drug-prices/) — Taxes, efficiency and the price of medicines

### Standalone studies, by theme

#### Statistics for Data Scientists

Standalone 'X is not Y' guides, each with a simulation study.

- [`a-correlation-is-not-a-mechanism/`](a-correlation-is-not-a-mechanism/) — A Correlation Is Not a Mechanism
- [`a-dataset-is-not-a-population/`](a-dataset-is-not-a-population/) — A Dataset Is Not a Population
- [`a-label-is-not-ground-truth/`](a-label-is-not-ground-truth/) — A Label Is Not Ground Truth
- [`causal-identification-before-estimation/`](causal-identification-before-estimation/) — Causal Inference Is Not a Toolbox
- [`causal-transportability-for-data-scientists/`](causal-transportability-for-data-scientists/) — Causal Transportability and External Validity
- [`clustering-is-not-a-population/`](clustering-is-not-a-population/) — A Cluster Is Not a Population
- [`conformal-prediction-for-data-scientists/`](conformal-prediction-for-data-scientists/) — A 90% Interval Is Not a Promise
- [`decision-focused-optimization-for-data-scientists/`](decision-focused-optimization-for-data-scientists/) — The Lowest RMSE Is Not the Best Decision
- [`dependent-data-uncertainty-for-data-scientists/`](dependent-data-uncertainty-for-data-scientists/) — Dependent Data: Autocorrelation, Effective Sample Size, and Uncertainty
- [`extreme-value-theory-for-data-scientists/`](extreme-value-theory-for-data-scientists/) — Extreme Value Theory for Data Scientists: The Tail Is Not a Rounding Error
- [`feature-importance-is-not-causality/`](feature-importance-is-not-causality/) — A Feature Importance Score Is Not a Causal Effect
- [`gaussian-graphical-models-for-data-scientists/`](gaussian-graphical-models-for-data-scientists/) — Gaussian graphical models and conditional independence
- [`industrial-statistics-for-data-scientists/`](industrial-statistics-for-data-scientists/) — A Factory Is Not a Histogram: Industrial Statistics for Data Scientists
- [`inverse-problems-for-data-scientists/`](inverse-problems-for-data-scientists/) — A Good Fit Is Not an Identified Model
- [`missing-data-is-not-zero/`](missing-data-is-not-zero/) — A Missing Value Is Not a Zero
- [`nearest-neighbor-geometry-for-data-scientists/`](nearest-neighbor-geometry-for-data-scientists/) — A Nearest Neighbor Is Not Necessarily Similar
- [`outlier-is-not-an-error/`](outlier-is-not-an-error/) — An Outlier Is Not an Error
- [`pca-vs-latent-factors-for-data-scientists/`](pca-vs-latent-factors-for-data-scientists/) — A Principal Component Is Not a Latent Factor
- [`queueing-theory-for-data-scientists/`](queueing-theory-for-data-scientists/) — Queueing Theory for Data Scientists: Why Accurate Demand Forecasts Still Produce Long Waits
- [`rare-event-detection-for-data-scientists/`](rare-event-detection-for-data-scientists/) — Rare-Event Detection: Prevalence, Calibration, and Decisions
- [`reliability-statistics-for-data-scientists/`](reliability-statistics-for-data-scientists/) — A Machine Does Not Fail at Its Mean
- [`sequential-change-detection-for-data-scientists/`](sequential-change-detection-for-data-scientists/) — The Alarm Is Not the Change: Sequential Change-Point Detection for Data Scientists
- [`significance-selection-winners-curse/`](significance-selection-winners-curse/) — Significance Selection, Power, and the Winner's Curse
- [`spurious-regression-cointegration/`](spurious-regression-cointegration/) — Spurious regression, unit roots, and cointegration
- [`the-test-set-is-not-an-oracle/`](the-test-set-is-not-an-oracle/) — The Test Set Is Not an Oracle

#### Machine Learning and Statistical Methods

Method studies on public and simulated data.

- [`agent-eval-runs/`](agent-eval-runs/) — How many runs does an agent comparison need?
- [`agent-size-evaluation/`](agent-size-evaluation/) — Evaluating over-engineering in LLM coding agents
- [`agreement-regimes/`](agreement-regimes/) — The p95 rule after a change of regime, and forward in time
- [`Blood_Pressure_Missingness/`](Blood_Pressure_Missingness/) — Missing Data Is Not Empty Space: Statistical Analysis of an Irregular Blood-Pressure Tracker
- [`deliberate-overfitting-diagnostics/`](deliberate-overfitting-diagnostics/) — Deliberate overfitting as a diagnostic
- [`edge-of-stability/`](edge-of-stability/) — Gradient descent at the edge of stability
- [`hansard-classifier-drift/`](hansard-classifier-drift/) — Classifier drift in Hansard
- [`hmc-bayesian-pinns/`](hmc-bayesian-pinns/) — HMC and the geometry of Bayesian inverse PDEs
- [`large-data-short-history/`](large-data-short-history/) — Selection, dependence, historical coverage and measurement
- [`mc-dropout-posterior/`](mc-dropout-posterior/) — Testing MC dropout as a posterior
- [`model-collapse-genesis/`](model-collapse-genesis/) — Model collapse measured on six translations of Genesis
- [`model-swap-tails/`](model-swap-tails/) — Swappable models, scored where the risk is: European, US and Chinese models in one day-ahead load pipeline
- [`org-chart-agents/`](org-chart-agents/) — Org charts for language-model agents
- [`org-chart-task-fit/`](org-chart-task-fit/) — What stays fixed when an agent organisation changes tasks
- [`pinn-crossover-dimension/`](pinn-crossover-dimension/) — PINNs against classical ground-state solvers, one to eight dimensions
- [`post-cutoff-forecasting/`](post-cutoff-forecasting/) — Forecasting after the cutoff
- [`prescribing-individuals/`](prescribing-individuals/) — Prescribing for a patient, not an age group
- [`second-count-prediction/`](second-count-prediction/) — Does the agreement of two counts predict the smallest loss they can catch?
- [`statistical-evidence-and-decisions/`](statistical-evidence-and-decisions/) — Statistical evidence and decisions
- [`synthetic-data-evaluation/`](synthetic-data-evaluation/) — Evaluating synthetic tabular data: fidelity, utility, privacy and valid inference
- [`tenure-by-age-at-hire/`](tenure-by-age-at-hire/) — How long new hires stay, by age at hire

#### Operations and Decisions

Decision problems with data: orders, rosters, routing, prices and storms.

- [`coach-change-decision/`](coach-change-decision/) — Changing the football manager mid-season, judged as a decision
- [`coupon-campaign-questions/`](coupon-campaign-questions/) — What a year of grocery coupon campaigns can and cannot say
- [`extreme-weather-warning-decisions/`](extreme-weather-warning-decisions/) — Extreme weather warnings: decisions under uncertainty
- [`hotel-cancellation-replay/`](hotel-cancellation-replay/) — Hotel cancellation replay
- [`iberian-electricity-lp/`](iberian-electricity-lp/) — The Iberian electricity price as a linear programme
- [`island-cable-resilience/`](island-cable-resilience/) — Correlated faults and island isolation on the submarine cable network
- [`learned-branching-variability/`](learned-branching-variability/) — Learned branching against performance variability
- [`operational-research-2026/`](operational-research-2026/) — Operational research in Python, 2026: a two-stage depot example
- [`perishable-order-decisions/`](perishable-order-decisions/) — Algorithms against the decisions around them: daily orders of perishable goods
- [`restaurant-portion-economics/`](restaurant-portion-economics/) — Restaurant portion economics: food waste, demand, and contribution margins
- [`routing-on-real-roads/`](routing-on-real-roads/) — Routing on real roads
- [`staffing-shift-constraints/`](staffing-shift-constraints/) — Shift constraints against forecast quality: staffing NYC 311
- [`storm-clustering-portugal/`](storm-clustering-portugal/) — Storm clustering in Portugal
- [`storm-train-recovery/`](storm-train-recovery/) — Recovery capacity under clustered storms
- [`storm-wind-extremes-portugal/`](storm-wind-extremes-portugal/) — Storm wind extremes in Portugal

#### Housing

Housing supply, prices and affordability.

- [`france-repeat-sales-floor/`](france-repeat-sales-floor/) — Is the purchase price a floor in France too? Repeat sales in the DVF files, 2021–2025
- [`housing-losses-inflation/`](housing-losses-inflation/) — Housing losses and inflation: how much of the loss never shows in the price
- [`housing-supply-counterfactual/`](housing-supply-counterfactual/) — Construction, House Prices and the Missing Counterfactual
- [`housing-vacancy-tax/`](housing-vacancy-tax/) — Taxing empty homes in Portugal: statutory rates, detection and expected cost
- [`portugal-building-ecosystem/`](portugal-building-ecosystem/) — Who builds Portugal's houses? The housebuilding sector measured layer by layer
- [`portugal-housing-generations/`](portugal-housing-generations/) — Portugal's housing market, 2008–2025 (moved to its own repository)
- [`uk-repeat-sales-rigidity/`](uk-repeat-sales-rigidity/) — Do sellers refuse nominal losses? Repeat sales in England and Wales, 1995–2026
- [`uk-sales-under-water/`](uk-sales-under-water/) — Owners under water and the fall in house sales: postcode districts in England and Wales
- [`uk-valuation-purchase-price/`](uk-valuation-purchase-price/) — Does a valuation model need the price the owner paid? Repeat sales in England and Wales
- [`urban-affordability-cascade/`](urban-affordability-cascade/) — The Suburb Is Not an Escape — reproducible Greater Porto housing analysis

#### Economics and Society

Prices, labour, taxes, growth and elections.

- [`civility-and-political-change/`](civility-and-political-change/) — When Power Calls for Civility
- [`eu-okun-law/`](eu-okun-law/) — Okun's law revisited: an EU-27 empirical study
- [`felt-inflation/`](felt-inflation/) — Frequent purchases, official inflation and perceived prices
- [`fixed-or-variable/`](fixed-or-variable/) — Fixed or variable: euro-area mortgages replayed for every start month
- [`gdp-life-outcomes/`](gdp-life-outcomes/) — GDP per Capita, Life Expectancy and Basic Services: What Cross-Country Correlations Can and Cannot Identify
- [`heat-pump-price-ratio/`](heat-pump-price-ratio/) — The price ratio decides heat pumps: household electricity and gas prices in 25 European countries, 2007–2026
- [`house-votes-seats-gerrymandering/`](house-votes-seats-gerrymandering/) — Votes, seats and who drew the map: U.S. House elections, 1972–2026
- [`lump-sum-vs-spreading/`](lump-sum-vs-spreading/) — Lump sum or monthly instalments: investing a windfall in 23 stock markets since 1955
- [`neoliberalism-evidence-synthesis/`](neoliberalism-evidence-synthesis/) — Neoliberalism, work, distribution and housing: an evidence audit
- [`pension-inflation/`](pension-inflation/) — Pension inflation: fixed and indexed pensions replayed on 23 countries' prices
- [`portugal-debt-2035-probability/`](portugal-debt-2035-probability/) — Portugal's 2035 debt target as a probability
- [`portugal-fiscal-reaction/`](portugal-fiscal-reaction/) — How Portugal's Budget Responds to Its Debt: A Fiscal Reaction Function and the 2035 Target
- [`portugal-iva-zero/`](portugal-iva-zero/) — Pass-through of Portugal's zero VAT on 46 food products (IVA Zero) in official prices
- [`portugal-labour-share/`](portugal-labour-share/) — Portugal's labour share under eighteen definitions
- [`portugal-labour-share-industries/`](portugal-labour-share-industries/) — Portugal's labour share by industry, 1995–2024

#### History, Measured

Questions about the past, tested on historical records.

- [`battle-numbers/`](battle-numbers/) — Numbers and victory in 660 battles, 1600 to 1982

#### Science and Mathematics

Questions from science and everyday life, checked with code.

- [`forecast-horizon/`](forecast-horizon/) — How far ahead weather can be forecast: error growth in Lorenz's models and in the real atmosphere
- [`ranking-extremes/`](ranking-extremes/) — Small samples at both ends of a country ranking
- [`rare-flags/`](rare-flags/) — What a flag means when the thing flagged is rare
- [`xx-equals-49/`](xx-equals-49/) — xˣ = 49

### Repository tooling

Repository checks, shared assets and data-source records.

- [`assets/`](assets/) — Logo and shared images
- [`data_sources/`](data_sources/) — Records of shared data sources
- [`docs/`](docs/) — Repository architecture
- [`medium-equations/`](medium-equations/) — Displayed equations for the Medium articles
- [`scripts/`](scripts/) — Repository checks and shared fetch scripts

### Historical archive

Notebooks, scripts and data behind older articles, kept at their original paths. Not guaranteed to run unchanged today.

[`ab_testing/`](ab_testing/), [`anova_analysis/`](anova_analysis/), [`Barabasi-Albert_Network/`](Barabasi-Albert_Network/), [`Bayestheorem/`](Bayestheorem/), [`BhattacharyyaMetrics/`](BhattacharyyaMetrics/), [`causal inference in marketing analytics/`](causal%20inference%20in%20marketing%20analytics/), [`change_point/`](change_point/), [`clima_portugal/`](clima_portugal/), [`compare tests statistics/`](compare%20tests%20statistics/), [`conformal_prediction/`](conformal_prediction/), [`customer_life_value/`](customer_life_value/), [`customer_segmentation/`](customer_segmentation/), [`Data_Is_Not_Decision/`](Data_Is_Not_Decision/), [`data_vs_sampling_dist/`](data_vs_sampling_dist/), [`Deep Learning with Caffe and Python/`](Deep%20Learning%20with%20Caffe%20and%C2%A0Python/), [`DimensionReduction/`](DimensionReduction/), [`Django-Charts/`](Django-Charts/), [`earthmover/`](earthmover/), [`elliptic-curves/`](elliptic-curves/), [`feature_discretization/`](feature_discretization/), [`feature_importance_in_python/`](feature_importance_in_python/), [`Featuring Engineering/`](Featuring%20Engineering/), [`Finance/`](Finance/), [`Graph Algorithms/`](Graph%20Algorithms/), [`income_inequaility/`](income_inequaility/), [`intermittent_time_series-main/`](intermittent_time_series-main/), [`introduction_to_batch_processing_mapreduce/`](introduction_to_batch_processing_mapreduce/), [`just_fun/`](just_fun/), [`Levenshtein Distance/`](Levenshtein%20Distance/), [`manufacturing_data_science/`](manufacturing_data_science/), [`mapreduce/`](mapreduce/), [`Mongo DB/`](Mongo%20DB/), [`Neural Network/`](Neural%20Network/), [`portugal_data/`](portugal_data/), [`Predicting House Prices/`](Predicting%20House%20Prices/), [`preprocessing/`](preprocessing/), [`pseudo_labeling_a_simple_semi_supervised_learning_method/`](pseudo_labeling_a_simple_semi_supervised_learning_method/), [`python-missing-data/`](python-missing-data/), [`python-telegram-tutorial/`](python-telegram-tutorial/), [`R_code/`](R_code/), [`real_data_science/`](real_data_science/), [`short_time_series/`](short_time_series/), [`spectral_methods/`](spectral_methods/), [`stocks_under_10/`](stocks_under_10/), [`survival_analysis/`](survival_analysis/), [`Twitter_Analytics/`](Twitter_Analytics/), [`Walmart-Sales-Prediction/`](Walmart-Sales-Prediction/), [`Web Scrapper/`](Web%20Scrapper/), [`work force model/`](work%20force%20model/), [`Yelp rating in Las Vegas/`](Yelp%20rating%20in%20Las%C2%A0Vegas/), [`Zipf/`](Zipf/), [`zipf_law/`](zipf_law/)

### Not yet sorted

Add these folders to `projects.yml`.

- [`ambulance-tour-length/`](ambulance-tour-length/) — Tour length and the hours that size an ambulance roster
- [`auction-entry-boem/`](auction-entry-boem/) — One more bidder or a higher minimum bid: BOEM's Gulf lease sales, 2013–2026
- [`bid-rigging-screens/`](bid-rigging-screens/) — Screens for bid rigging on firms and countries a model has not seen
- [`europe-industrial-regions/`](europe-industrial-regions/) — Regions, not countries? Industrialisation on the map of Europe, 1900-2021
- [`forecast-success-specification/`](forecast-success-specification/) — The success specification decides which forecast wins: a pre-registered test on M5 food sales
- [`growth-finite-world/`](growth-finite-world/) — Corporate growth on a finite planet: valuation, earnings and energy, 1871–2024
- [`moderate-rent-ceiling/`](moderate-rent-ceiling/) — Where the EUR 2,300 "moderate rent" ceiling sat against real rents, Portugal 2017-2025
- [`productivity-hours-denominator/`](productivity-hours-denominator/) — The denominator decides: EU labour productivity with national-accounts and Labour Force Survey hours, 2009–2026
- [`services-energy-time/`](services-energy-time/) — The energy and the hours behind Europe's service economy, 1995–2024
- [`suffrage-public-health/`](suffrage-public-health/) — Suffrage laws and child mortality: a staggered difference-in-differences reanalysis

<!-- projects:end -->

## Adding a project

1. Create a top-level folder with a URL-friendly name (lowercase, hyphens, no spaces) and a README whose first heading names the study.
2. Add one line for it under `folders` in [`projects.yml`](projects.yml), in alphabetical order, such as `my-new-study: operations-and-decisions`. For a part of a series, add its position: `{group: euro-crisis, part: 7}`.
3. Give it its own workflow, limited to its own paths, and keep its dependencies inside the folder.
4. Open the pull request. The `readme-index` workflow checks `projects.yml` and changes nothing on the branch, so two open pull requests do not conflict over this README.
5. Merge with a merge commit. On `main`, the workflow rebuilds the Projects list and commits it.

A folder missing from `projects.yml` is not an error: the list shows it under "Not yet sorted" until its line is added. To check locally: `python scripts/build_readme.py --check`.

## Older material

The folders in the historical archive, and the notebooks and images at the top level, hold the code behind older articles. They stay at their original paths because published articles link to them, and they are not guaranteed to run unchanged with current libraries. [docs/REPOSITORY_ARCHITECTURE.md](docs/REPOSITORY_ARCHITECTURE.md) sets out how a folder is classified, and [REPOSITORY_ROADMAP.md](REPOSITORY_ROADMAP.md) the plan for auditing the older folders.

The `verify_*.py` scripts at the top level audit the published results of the twenty-part causal inference series in [`causal_inference_statistics/`](causal_inference_statistics/).

## Licence

Unless a folder says otherwise, the content is licensed under the [Apache License 2.0](LICENSE).
