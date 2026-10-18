# Okun's Law in the European Union: What the Data Can and Cannot Say

*Association, prediction and the limits of causal identification in {{n_countries}} countries, 2009–2026*

Okun's law says that unemployment falls when output grows faster than usual and rises when it grows more slowly. It is one of the most quoted relations in macroeconomics. Its size, though, varies across countries and periods, and what it means depends on what moves output in the first place.

This article summarises a reproducible study of the relation in the {{n_countries}} member states of the European Union. The full manuscript, with the mathematics, the bibliography and every table, is in the project repository. Every number below is read by code from the same committed result tables as the manuscript.

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

- **Without 2020–2021**, the sum is negative with a 95% interval below zero in {{cl_n_neg_excluded}} of {{n_countries}} countries. In {{cl_n_robust}} countries every core specification agrees.
- **With the pandemic quarters**, the sum moves towards zero in {{cl_n_attenuated}} countries. Those eight quarters supply a median {{cl_pandemic_share_median}} of the variation in GDP growth, while job-retention schemes kept unemployment from following output. A full-sample coefficient mostly describes the pandemic.

![Classical sums with and without 2020–2021](figures/fig2_pandemic.png)

*Classical two-lag sums: full sample against the sample without 2020–2021. For negative sums, points above the diagonal are closer to zero with the pandemic included.*

**A hierarchical Bayesian model** lets countries borrow strength from one another. It also models what the classical regressions leave in the residuals: heavy tails, persistence and a common European factor.

- The average country sum is {{by_mu}} (90% interval {{by_mu_ci}}).
- The between-country standard deviation is {{by_tau}}.
- {{by_n_neg_all}} countries are credibly negative in every accepted specification.

The Bayesian sums are smaller in magnitude than OLS on the same quarters in {{by_n_smaller}} of {{n_countries}} countries. Part of that is pooling. Part is that the model's sum excludes the common European cycle, so the two estimate different things.

![Country sums, classical and Bayesian](figures/fig1_countries.png)

*Orange: OLS without 2020–2021 with 95% intervals. Blue: hierarchical posterior medians, 2009–2019, with 90% intervals. Alphabetical; no ranking implied.*

**Did the relation weaken after 2021?** An earlier analysis reported {{ss_published_holm}} significant post-2021 changes. With about seventeen quarters, the usual normal test is unreliable: in a size study it rejected a true null {{ss_size_normal}} of the time at a nominal 5%. A bootstrap test held its size ({{ss_size_bootstrap}}), and under it {{ss_survivors}} of the changes survive.

## 2. Rankings are not established

Estimates are uncertain and correlated through common shocks, so a league table of point estimates is misleading.

- The median 90% interval for a country's rank spans {{by_rank_width}} of {{n_countries}} places.
- {{by_pairs_primary}} of {{by_pairs_total}} country pairs differ credibly in the primary model, but only {{by_pairs_surviving}} in every accepted specification.

## 3. The association barely helps forecasting

Two registered comparisons asked whether the association, or Bayesian pooling, improves predictions of the unemployment change. Both scored full predictive distributions with the continuous ranked probability score and refitted every model at every origin. Each difference below is the mean score of the first model relative to the second; a negative value favours the first.

**Given the quarter's own GDP growth**, {{pc_decided}} of {{pc_total}} comparisons is decided. Before the pandemic, the hierarchical model against a zero-change benchmark: {{pc_e1_zero}}, the one decided comparison. After 2022, no model that uses the quarter's GDP clearly beats zero change ({{pc_e2_zero}}).

**In real one-quarter-ahead forecasts**, {{fc_decided}} of {{fc_total}} comparisons are decided.

- The hierarchical model with GDP lags scores {{fc_gdp}} against its classical counterpart. Most of that gap comes from two pandemic quarters, and the registered test does not decide it.
- Against simple persistence the difference is {{fc_zero}}.
- Without the pandemic targets, the hierarchical and richer models win {{fcx_decided}} of five comparisons, by small margins ({{fcx_gdp}}, {{fcx_ar1}}, {{fcx_lags}} and {{fcx_factor}}). Persistence is still not beaten ({{fcx_zero}}, not decided).

![Predictive comparisons](figures/fig3_prediction.png)

*Mean score of model A relative to model B; negative means A is better. Left: conditional on the quarter's GDP growth. Right: one-quarter-ahead forecasts.*

The association is real, but it carries little information that the recent path of unemployment does not already contain.

## 4. No defensible causal coefficient

A causal Okun coefficient must refer to a specific shock: the ratio of that shock's effect on unemployment to its effect on output. Reduced-form innovations are not shocks, and ordering variables in a VAR does not make them so. Four strategies were assessed.

**External instruments.** Two shock series identified from high-frequency market data were used as instruments: ECB monetary policy surprises and OPEC oil supply news.

- Both are weak for EU real activity: the largest first-stage t-statistic is {{id_max_t}} with quarterly GDP and {{mo_max_t}} with monthly industrial production, against a conventional threshold of 3.16.
- All weak-instrument-robust confidence sets for the Okun ratio are unbounded.
- Aggregated to quarters, the oil series is predictable from lagged EU data (p = {{id_pred_oil}}), which undermines its exogeneity at that frequency.

**Foreign demand.** A shift-share of partner-country growth, weighted by how much of each country's value added foreign demand absorbs, is the best public candidate for a country-specific GDP shock. Its expected first-stage t-statistic is {{ss_P1_t}}. A registered feasibility rule stopped the design before estimation.

**Sign restrictions.** A panel VAR identifies demand, technology and labour-supply shocks by the signs of their impact effects.

- The usual uniform prior over rotations gives sharp-looking ratios: {{sr_dem_uniform}} for demand and {{sr_tech_uniform}} for technology.
- The identified sets do not depend on that prior. For demand the set is {{sr_dem_bounded}}, and for technology it is {{sr_tech_bounded}}. The sharpness came from the prior, not the data.
- Labour-supply and demand shocks differ at impact only because the signs impose it.

![Instrument strength and sign-restricted responses](figures/fig4_identification.png)

*First-stage strength of every instrument; the dashed line is the conventional threshold.*

![Sign-restricted unemployment responses](figures/fig5_sign_restrictions.png)

*Wide bars: identified sets (robust 90% regions). Thin lines and dots: the uniform-prior posterior.*

Every coefficient in this study is therefore a descriptive association.

## 5. Portugal, Spain, the Netherlands and Luxembourg

The case study asked where output fluctuations are absorbed. GDP growth splits exactly into growth in output per hour, hours per worker and employment, so each country's adjustment can be traced through these margins. Every hypothesis and prediction was registered first, and the same rule was applied to the comparison countries. {{cs_supported}} of {{cs_total}} predictions is supported and {{cs_contradicted}} are contradicted; the rest are not determined.

![Margins of adjustment](figures/fig6_margins.png)

*Elasticities to GDP growth, 2020–2021 excluded, with 95% bootstrap intervals. Eurostat publishes no quarterly employment in persons for Portugal.*

**Spain.** Spain's association is large: {{cl_ES_excl}} classically and {{by_ES}} in the Bayesian model, below the European mean with probability {{claim_es_p}}, in {{claim_es_holds}} of {{by_n_accepted}} accepted fits. Spain absorbs output fluctuations through employment (elasticity {{cs_ES_e_dom}}), not productivity ({{cs_ES_prod}}). Whether temporary contracts carry that adjustment is not decided by the registered test.

**Portugal.** Portugal's association is negative, but its significance depends on the specification ({{cl_PT_full}} on the full sample, {{cl_PT_excl}} without 2020–2021). Total hours respond to GDP about as strongly as in Spain ({{cs_PT_h}} against {{cs_ES_h}}). Descriptively, and not as a registered prediction, Portugal's participation rises with GDP ({{cs_PT_dp}}), which offsets part of the fall in unemployment.

**The Netherlands.** The lagged share of the Dutch response is {{by_NL_lagshare}}. The registered rule for a "delayed" response holds in the primary fit but only in {{claim_nl_holds}} of {{by_n_accepted}} accepted fits. The one supported prediction of the case study is Dutch: output per hour is more responsive to GDP than in Spain, by {{cs_p22}} ({{cs_p22_ci}}). That pattern fits labour hoarding, although hoarding cannot be separated from unmeasured effort.

**Luxembourg.** The classical sum is {{cl_LU_verdict}} ({{cl_LU_full}} full sample, {{cl_LU_excl}} without the pandemic). The registered Bayesian rule for a weak association holds in {{claim_lu_holds}} of {{by_n_accepted}} accepted fits. The proposed explanation was a measurement mismatch: GDP counts cross-border commuters, unemployment counts only residents. If commuters absorbed the fluctuations, domestic employment would move more than resident employment. It does not:

- the gap is {{cs_p11}} ({{cs_p11_ci}});
- domestic employment ({{cs_LU_e_dom_pe}}) and resident employment ({{cs_LU_e_res}}) barely move with GDP;
- output per hour absorbs the fluctuations almost one for one ({{cs_LU_prod}}).

**Institutions.** The protocol predicted that stricter employment protection would weaken the unemployment response. In two of the registered variants the cross-country correlation has the opposite sign ({{cs_eplr_cl_rho}} and {{cs_eplt_noluie_rho}}). With at most twenty countries and one year of indicators, this says nothing about what employment protection does.

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

Everything is in the repository [DiogoRibeiro7/Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-okun-law), directory `eu-okun-law`: frozen data with hashes, protocols, code, result tables, the LaTeX manuscript and this article's source. The file `publication/README.md` lists the reproduction steps. The quarterly data snapshot's manifest hash begins {{hash_quarterly}}.
