# Evidence ledger — provisional 2026 forecast sequel

All political and statistical statements in the draft should map to one of
these source classes. No fabricated district observations, poll updates,
candidate probabilities, certified seat totals or unregistered model changes
are permitted. Only the original forecasts were fixed before the outcome;
the interpretation of the new data is necessarily post-election.

| Claim | Status | Evidence and interpretation |
| --- | --- | --- |
| Original OLS point = −32.322888 Republican House seats | Fixed ex ante | `../results/forecast_lock.json` and original article; information cut-off 31 October, recorded 1 November |
| OLS conditional 95% interval = [−48.42838655, −16.21738950] | Fixed ex ante | Same manifest; conditional on original D+8 and 220-seat inputs, classical model assumptions |
| Student-t −33.757549, ridge −31.060791, recency WLS −30.975973 | Fixed ex ante | Same manifest; these are point forecasts without separate registered predictive intervals |
| 0.8934 historical in-sample R², forward RMSE 8.70 | Historical diagnostics | Same manifest and original reproducibility code; neither is a 2026 accuracy percentage |
| Republican 220-seat starting assumption matches the initial 119th Congress party division | Verified historical source alignment, not a new 2026 result | [House Historian](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/) lists 220 Republicans and 215 Democrats for the 119th; [American Presidency Project](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections) lists 220 Republican House seats to defend for 2026 |
| AP on 4 Nov had Dem 228 called and GOP 139 called, with 35/33 leads | Live, not certified | `data/ap_house_live_2026-11-04.json`; AP report 10:51:02 UTC, retrieved 10:51:54 UTC, [publisher](https://apnews.com/hub/elections) |
| Democrats have mathematically secured the House majority | AP called seats | 228 called exceeds 218-seat majority, even if all remaining 68 seats were Republican |
| If all AP leads hold, GOP would obtain 172 seats | Conditional arithmetic, not observed | 139 called GOP + 33 GOP leads; not a final tally, not a probability forecast |
| GOP change would be −48 relative to the registered 220-seat baseline | Conditional arithmetic only | 172 − 220; requires future source-compatible final party divisions before it can be used in the official scorer |
| Registered signed discrepancy would be +15.677112 for OLS if leads hold | Hypothetical, not observed score | −32.322888 − (−48); sign defined as prediction minus observation in registration |
| GOP final seats 172–203 correspond to integer losses covered by 95% OLS interval | Deterministic boundary calculation | ceil(220 − 48.42838655) = 172; floor(220 − 16.21738950) = 203 |
| A 171 GOP-seat official outcome would be outside the model interval | Hypothetical threshold | 171 − 220 = −49, below −48.42838655; **not** a claim that 171 is the result |
| Live scenario lies outside bootstrap sensitivity band | Conditional diagnostic | −48 is below the lower bootstrap sensitivity bound −41.7517; the band is not a calibrated 95% prediction interval |
| Final 2026 House seat change is known | **Not established** | Do not claim it: wait for [House Historian](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/) / [American Presidency Project](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections) aligned initial-election series |
| A single 2026 observation validates or invalidates 95% coverage | **Unsupported** | Coverage is a repeated-sampling property; one hit/miss is insufficient |
| The model's predictors are causal effects | **Unsupported** | Observational predictive associations are not policy-treatment effects |

## Editorial guardrails

1. Use **if all current leads hold** for 172 seats, −48 seat change and
   differences from model points; never write these as observed election data.
2. Report the 31 October information cutoff and the distinct 1 November
   registration date honestly; do not call both an October preregistration.
3. Keep the original OLS forecast and three estimator alternatives fixed. No
   post-hoc retraining, model selection or data redefinition to make a hit.
4. Apply the initial-election party-division target, not today's incumbent
   membership, when an eventual final score is created.
5. Distinguish the model-based 95% prediction interval from the bootstrap
   sensitivity interval, which has no declared frequentist coverage guarantee.
6. State as-of UTC timestamps next to any AP or later reporting claim.
7. Do not publish a final scored manuscript until the final-source requirement
   in `../PRE_ELECTION_REGISTRATION.md` can be met.

## Source bibliography

- [Diogo Ribeiro (31 October 2026), *The Forecast Is Not the Fit*](https://diogo-ribeiro-1975.medium.com/the-forecast-is-not-the-fit-rebuilding-an-89-midterm-election-model-8fb36cc269be).
- [Registered model record and methodology](../PRE_ELECTION_REGISTRATION.md); [`forecast_lock.json`](../results/forecast_lock.json), signed 1 November 2026.
- [Associated Press, 2026 U.S. election results](https://apnews.com/hub/elections), 4 November 2026 live snapshot.
- [Associated Press, Democrats win congressional control](https://apnews.com/article/ef6199e25927a9ac9f4c18eae0e497cd), 4 November 2026 live report.
- [Federal Election Commission, general federal election date](https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/).
- [House Historian, Party Divisions](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/), initial-election result convention.
- [American Presidency Project, Seats Gained/Lost at Mid-Term Elections](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections), aligned historical series.
- [Gneiting, T., and Raftery, A. E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437), *Journal of the American Statistical Association*, 102(477), 359–378; for scoring-rule and calibration distinctions.
