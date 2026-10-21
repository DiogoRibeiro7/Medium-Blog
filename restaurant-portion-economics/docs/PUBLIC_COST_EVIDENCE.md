# Public sector cost anchors: evidence and interpretation

Sources are retained in `data/reference/restaurant_cost_benchmarks.csv`
and `data/reference/portion_response_evidence.csv`.

The US National Restaurant Association reported respondent median 2024
food and non-alcoholic beverage expenses at 32.0% of sales for full-service
restaurants and 32.4% for limited-service establishments. In a different
published full-service comparison, the median was 33.7% for businesses
with annual sales under USD 2m and 31.0% for those with at least USD 2m.
The source URLs and original observation year accompany each datum.

A ratio measured at business level includes beverages. It is not an
ingredient-cost estimate for one meal, and comparing US operators with a
hypothetical Portuguese restaurant is a proxy stress exercise, not an
empirical calibration. The industry medians are not population quantiles,
confidence intervals, distribution bounds, or evidence for food-cost
uncertainty support. All EUR values are assumed conversions using a
hypothetical EUR 15 meal price. Do not describe them as Portuguese prices.

Source studies on portions describe quasi-experiments in institutional
dining and smaller dishes sold at a discount. They cannot identify the
customer attrition caused by serving smaller portions at an unchanged
price. Customer loss (0%, 1%, 2%, 5%) therefore remains a sensitivity
parameter and not an estimated demand response.

## Run

`Rscript tests/test-public-cost-anchors.R`

`Rscript scripts/analyse-public-cost-anchors.R`

Outputs: `public_cost_proxy_scenarios.csv` and
`public_cost_source_register.csv`, both ignored by Git and reproducible
without external downloads beyond the source references already recorded.

## Methodological separation

- Observed source evidence: US sector-level cost-to-sales respondent medians.
- Proxy translation: applying a median sales share to an assumed menu price.
- Additional assumptions: unaffected ingredient share, portion reduction,
  implementation expense, procurement pass-through and demand loss.
- Missing identification: actual restaurant-level ingredient spend,
  consumer willingness to pay, repeat visits, and a fixed-price
  portion-size intervention.

No benefit, elasticity or optimal portion size is inferred empirically.
