# wealth-origin-synthesis

Companion code for the closing article of the wealth-origin series: what five studies established about inherited billionaire wealth and productivity growth, what they did not, and what evidence would settle the causal question.

Nothing here fetches data or re-estimates a model. `synthesis.py` reads the committed results of the five companion projects in this repository, puts the headline coefficient from each design on one footing, scores every test the series ran with explicit rules, and works out how much statistical power the succession design has against an effect of the size the panel implies.

The five studies it reads:

- `wealth-concentration-misallocation/` (simulation and cross-country audit): the original claim about the top-1% wealth share, and the first cross-country look at billionaire origin.
- `wealth-origin-growth/` (within-country panel): the headline coefficient, the placebo on past growth, leave-one-country-out and the period split.
- `wealth-origin-sectors/` (sector test): the coefficient with six and with all 18 sector shares held fixed.
- `wealth-origin-pandemic/` (pandemic exposure): four exposure measures, the two-list average and headcount measures, and the 2020 list dropped.
- `wealth-origin-successions/` (succession events): founder-to-heir successions as events, the permutation test and placebo exits.

## Run

```bash
python -m pip install -r requirements.txt
python synthesis.py      # writes results/estimates.csv, ledger.csv, power.json, summary.txt
python figures.py        # writes figures/fig1_estimates.png, fig2_ledger.png
python -m pytest -q tests
```

## Outputs

- `results/estimates.csv`: the inherited-minus-self-made coefficient (TFP growth over three years, the 63 countries with billionaires in at least 7 of 13 lists, country and year fixed effects) under six designs, each with and, where computed, without the 2020 Forbes list; and the succession event-study effect under five variants.
- `results/ledger.csv`: 29 tests grouped into 13 questions. For each: the study that ran it, the test, the statistic, the verdict (`passed`, `open`, `failed`) and the rule that produced it.
- `results/power.json`: the panel-implied effect for the 38 successions with a prediction, the power of the event design against it, the sign-error rate and exaggeration ratio among significant estimates, and the number of events that 80% power would need.
- `results/rule_sensitivity.csv`: the verdict of every test under the default thresholds and under a lenient and a strict set (listed in `RULE_VARIANTS`).
- `results/summary.txt`: every number quoted in the article. CI reruns the script and fails if this file changes.

## Verdict rules

All in `synthesis.py` under `VERDICT RULES`, with their thresholds in `DEFAULT_RULES`, so a reader who disagrees with one can change it in one place and rerun. In short:

- **Re-estimated coefficient** (a new control or sample): passed if negative with a 95% interval below zero; failed if it keeps less than half of the reference estimate (or changes sign) and the interval includes zero; open otherwise.
- **Joint tests and permutation tests**: passed below p = 0.05, open below 0.10, failed above.
- **Reverse timing** (forward minus backward coefficient): passed if the difference's interval is below zero; failed if the backward (placebo) coefficient is itself significantly negative; open otherwise.
- **Event-study lead**: an equivalence test. Passed only if the whole interval lies within half the post-event effect either side of zero, so a wide interval cannot pass by being uninformative.
- **Placebo events**: passed if the placebo is not negative, or is under half the main effect with an interval that includes zero.
- **Leave one country out**: passed if every estimate is negative and at least 90% of intervals exclude zero.
- **Agreement on size**: passed within a factor of 2, open within 3, failed beyond.

## Power calculation

For each matched succession, the within-country panel predicts a change in annual TFP growth. Converting that prediction to the event study's units (three-year cumulative change per log point of the shock) and taking the median gives the effect the event design would have to detect if the panel coefficient were causal. Power, the sign-error rate and the expected exaggeration among significant estimates follow from the normal approximation with the event study's standard error (Gelman and Carlin 2014). The number of events needed assumes the standard error falls with the square root of the number of succession country-years and that new successions resemble the old ones.

## Contact

Diogo Ribeiro, dfr@esmad.ipp.pt
