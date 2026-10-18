# Okun's Law in the European Union: What the Data Can and Cannot Say

*Association, prediction and the limits of causal identification in 27 countries, 2009–2026*

Okun's law says that unemployment falls when output grows faster than usual and rises when it grows more slowly. It is one of the most quoted relations in macroeconomics. Its size, though, varies across countries and periods, and what it means depends on what moves output in the first place.

This article summarises a reproducible study of the relation in the 27 member states of the European Union. The full manuscript, with the mathematics, the bibliography and every table, is in the project repository. Every number below is read by code from the same committed result tables as the manuscript.

## Why another look

The study began with a reported hierarchical Bayesian analysis that made three claims:

- Spain has a large association between GDP growth and unemployment.
- The Netherlands has a delayed response.
- Luxembourg has a weak one.

We do not have that analysis's code, data or estimates. So we treated the three claims as hypotheses, wrote down how each would be judged, and only then estimated anything.

## Three questions, kept apart

1. **Association.** How strongly do GDP growth and the change in the unemployment rate move together in each country?
2. **Prediction.** Does that association help forecast unemployment?
3. **Causation.** Is any part of it the effect of an identified shock that moves output?

These can have different answers. A strong association need not forecast well. A useful predictor need not be causal. And an Okun coefficient averages over every kind of shock that hit the economy in the period, so it need not equal the effect of any one of them.

## How the study was run

- **Frozen data.** Official Eurostat and OECD data were downloaded in October 2026 and frozen. Every raw response is hashed with SHA-256, and an audit re-derives each panel from the raw files.
- **Registered before estimation.** Each analysis has a protocol that was committed and pushed before any estimate. Later clarifications are recorded in a change log with their reasons.
- **Replicated.** The Bayesian and predictive analyses were run on Windows and replicated on Linux by GitHub Actions.

## 1. The association is real outside the pandemic

The headline measure is the sum of the coefficients on current GDP growth and two lags, in percentage points of unemployment per percent of GDP growth.

- **Without 2020–2021**, the sum is negative with a 95% interval below zero in 22 of 27 countries. In 17 countries every core specification agrees.
- **With the pandemic quarters**, the sum moves towards zero in 20 countries. Those eight quarters supply a median 78% of the variation in GDP growth, while job-retention schemes kept unemployment from following output. A full-sample coefficient mostly describes the pandemic.

![Classical sums with and without 2020–2021](figures/fig2_pandemic.png)

*Classical two-lag sums: full sample against the sample without 2020–2021. For negative sums, points above the diagonal are closer to zero with the pandemic included.*

**A hierarchical Bayesian model** lets countries borrow strength from one another. It also models what the classical regressions leave in the residuals: heavy tails, persistence and a common European factor.

- The average country sum is −0.198 (90% interval [−0.254, −0.145]).
- The between-country standard deviation is 0.126.
- 22 countries are credibly negative in every accepted specification.

The Bayesian sums are smaller in magnitude than OLS on the same quarters in 25 of 27 countries. Part of that is pooling. Part is that the model's sum excludes the common European cycle, so the two estimate different things.

![Country sums, classical and Bayesian](figures/fig1_countries.png)

*Orange: OLS without 2020–2021 with 95% intervals. Blue: hierarchical posterior medians, 2009–2019, with 90% intervals. Alphabetical; no ranking implied.*

**Did the relation weaken after 2021?** An earlier analysis reported 14 significant post-2021 changes. With about seventeen quarters, the usual normal test is unreliable: in a size study it rejected a true null 22.8% of the time at a nominal 5%. A bootstrap test held its size (4.8%), and under it 0 of the changes survive.

## 2. Rankings are not established

Estimates are uncertain and correlated through common shocks, so a league table of point estimates is misleading.

- The median 90% interval for a country's rank spans 15 of 27 places.
- 99 of 351 country pairs differ credibly in the primary model, but only 59 in every accepted specification.

## 3. The association barely helps forecasting

Two registered comparisons asked whether the association, or Bayesian pooling, improves predictions of the unemployment change. Both scored full predictive distributions with the continuous ranked probability score and refitted every model at every origin. Each difference below is the mean score of the first model relative to the second; a negative value favours the first.

**Given the quarter's own GDP growth**, 1 of 10 comparisons is decided. Before the pandemic, the hierarchical model against a zero-change benchmark: −13.5%, the one decided comparison. After 2022, no model that uses the quarter's GDP clearly beats zero change (−0.3%).

**In real one-quarter-ahead forecasts**, 0 of 5 comparisons are decided.

- The hierarchical model with GDP lags scores −15.3% against its classical counterpart. Most of that gap comes from two pandemic quarters, and the registered test does not decide it.
- Against simple persistence the difference is +4.0%.
- Without the pandemic targets, the hierarchical and richer models win 4 of five comparisons, by small margins (−3.1%, −2.9%, −2.6% and −2.5%). Persistence is still not beaten (−3.5%, not decided).

![Predictive comparisons](figures/fig3_prediction.png)

*Mean score of model A relative to model B; negative means A is better. Left: conditional on the quarter's GDP growth. Right: one-quarter-ahead forecasts.*

The association is real, but it carries little information that the recent path of unemployment does not already contain.

## 4. No defensible causal coefficient

A causal Okun coefficient must refer to a specific shock: the ratio of that shock's effect on unemployment to its effect on output. Reduced-form innovations are not shocks, and ordering variables in a VAR does not make them so. Four strategies were assessed.

**External instruments.** Two shock series identified from high-frequency market data were used as instruments: ECB monetary policy surprises and OPEC oil supply news.

- Both are weak for EU real activity: the largest first-stage t-statistic is 1.27 with quarterly GDP and 1.21 with monthly industrial production, against a conventional threshold of 3.16.
- All weak-instrument-robust confidence sets for the Okun ratio are unbounded.
- Aggregated to quarters, the oil series is predictable from lagged EU data (p = 0.006), which undermines its exogeneity at that frequency.

**Foreign demand.** A shift-share of partner-country growth, weighted by how much of each country's value added foreign demand absorbs, is the best public candidate for a country-specific GDP shock. Its expected first-stage t-statistic is 0.19. A registered feasibility rule stopped the design before estimation.

**Sign restrictions.** A panel VAR identifies demand, technology and labour-supply shocks by the signs of their impact effects.

- The usual uniform prior over rotations gives sharp-looking ratios: −0.43 for demand and −0.55 for technology.
- The identified sets do not depend on that prior. For demand the set is unbounded, and for technology it is unbounded. The sharpness came from the prior, not the data.
- Labour-supply and demand shocks differ at impact only because the signs impose it.

![Instrument strength and sign-restricted responses](figures/fig4_identification.png)

*First-stage strength of every instrument; the dashed line is the conventional threshold.*

![Sign-restricted unemployment responses](figures/fig5_sign_restrictions.png)

*Wide bars: identified sets (robust 90% regions). Thin lines and dots: the uniform-prior posterior.*

Every coefficient in this study is therefore a descriptive association.

## 5. Portugal, Spain, the Netherlands and Luxembourg

The case study asked where output fluctuations are absorbed. GDP growth splits exactly into growth in output per hour, hours per worker and employment, so each country's adjustment can be traced through these margins. Every hypothesis and prediction was registered first, and the same rule was applied to the comparison countries. 1 of 14 predictions is supported and 0 are contradicted; the rest are not determined.

![Margins of adjustment](figures/fig6_margins.png)

*Elasticities to GDP growth, 2020–2021 excluded, with 95% bootstrap intervals. Eurostat publishes no quarterly employment in persons for Portugal.*

**Spain.** Spain's association is large: −0.79 classically and −0.51 in the Bayesian model, below the European mean with probability 0.996, in 8 of 8 accepted fits. Spain absorbs output fluctuations through employment (elasticity +1.11), not productivity (−0.21). Whether temporary contracts carry that adjustment is not decided by the registered test.

**Portugal.** Portugal's association is negative, but its significance depends on the specification (−0.13 on the full sample, −0.49 without 2020–2021). Total hours respond to GDP about as strongly as in Spain (+1.27 against +1.21). Descriptively, and not as a registered prediction, Portugal's participation rises with GDP (+0.18), which offsets part of the fall in unemployment.

**The Netherlands.** The lagged share of the Dutch response is 0.94. The registered rule for a "delayed" response holds in the primary fit but only in 6 of 8 accepted fits. The one supported prediction of the case study is Dutch: output per hour is more responsive to GDP than in Spain, by +0.73 ([+0.31, +1.23]). That pattern fits labour hoarding, although hoarding cannot be separated from unmeasured effort.

**Luxembourg.** The classical sum is sign-unstable (−0.05 full sample, +0.17 without the pandemic). The registered Bayesian rule for a weak association holds in 0 of 8 accepted fits. The proposed explanation was a measurement mismatch: GDP counts cross-border commuters, unemployment counts only residents. If commuters absorbed the fluctuations, domestic employment would move more than resident employment. It does not:

- the gap is +0.01 ([−0.05, +0.05]);
- domestic employment (+0.03) and resident employment (+0.02) barely move with GDP;
- output per hour absorbs the fluctuations almost one for one (+1.06).

**Institutions.** The protocol predicted that stricter employment protection would weaken the unemployment response. In two of the registered variants the cross-country correlation has the opposite sign (−0.60 and −0.67). With at most twenty countries and one year of indicators, this says nothing about what employment protection does.

## The earlier claims, judged

- **Spain: large association.** Supported, in every accepted specification.
- **The Netherlands: delayed response.** Qualified. It holds in the primary model but not in every accepted specification.
- **Luxembourg: weak association.** Qualified. It is weak and sign-unstable classically, but the registered Bayesian rule is met in none of the accepted fits.
- **Luxembourg: commuters explain it.** Not supported. Output per hour, not commuting, absorbs Luxembourg's output fluctuations.
- **Stricter employment protection weakens the response.** Contradicted as a correlation, in two of the registered variants.

No claim about Portugal was documented, so none is judged.

## What remains open

- Do countries differ because their labour markets respond differently to the same shock, or because they face different shocks? Separating the two needs shock-specific identification.
- What does a productivity or labour-supply shock do to unemployment in an EU country? That needs utilisation-adjusted productivity data and quasi-experimental variation in labour supply.
- Why does output per hour absorb almost all of Luxembourg's output fluctuations?
- Did the relation weaken after 2021? Only a longer sample can tell.

## The takeaway

An Okun coefficient estimated from aggregate data summarises how output and unemployment moved together over a particular period. In the EU from 2009 to 2026 that summary is robustly negative outside the pandemic, differs credibly between some countries, adds little to forecasts, and cannot be given a causal reading with the public data available.

## Reproduce it

Everything is in the repository [DiogoRibeiro7/Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-okun-law), directory `eu-okun-law`: frozen data with hashes, protocols, code, result tables, the LaTeX manuscript and this article's source. The file `publication/README.md` lists the reproduction steps. The quarterly data snapshot's manifest hash begins `8a8c8a107d584ba9`.
