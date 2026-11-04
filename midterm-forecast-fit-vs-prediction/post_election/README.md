# The Votes Are In — provisional 2026 House forecast audit

**Status: research draft, not a final election score.** The original 31 October
forecast and 1 November registration remain frozen. This folder adds an
independent 4 November Associated Press *live* snapshot and calculations that
are explicitly conditional on all currently leading candidates winning.
It does **not** record the final House Historian-aligned initial-election
party division. The eventual, final forecast score must be produced through
[`../score_after_election.py`](../score_after_election.py) when its source
requirements are satisfied. Never alter
[`../results/forecast_lock.json`](../results/forecast_lock.json) or the
historical training data to accommodate the election result.

## Exact inputs

| Evidence | Frozen information |
| --- | --- |
| Original article | [31 October 2026 Medium manuscript](https://diogo-ribeiro-1975.medium.com/the-forecast-is-not-the-fit-rebuilding-an-89-midterm-election-model-8fb36cc269be) |
| Forecast lock | [`../results/forecast_lock.json`](../results/forecast_lock.json), registered 1 November; model data cut-off 31 October |
| Source commit | [`fe98742efb86756fc30967477098482eb7505783`](https://github.com/DiogoRibeiro7/Medium-Blog/tree/fe98742efb86756fc30967477098482eb7505783) |
| Election date | 3 November 2026, [FEC](https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/) |
| AP live snapshot | [AP election results hub](https://apnews.com/hub/elections), report 2026-11-04 10:51:02 UTC, retrieved 10:51:54 UTC |
| Historical outcome definition | [House Historian initial-election party divisions](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/) and [American Presidency Project](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections) |

The AP snapshot is a labeled transcription of one aggregate results feed, not
a claim of final election results. Its values are archived at
[`data/ap_house_live_2026-11-04.json`](data/ap_house_live_2026-11-04.json).
As of the retrieval, AP had called 228 Democratic seats and 139 Republican
seats; Democrats led 35 additional uncalled seats and Republicans led 33. The
435-seat total reconciles, and 228 called seats establish a Democratic
majority. The 172-seat GOP *trend* is a conditional hypothetical, not an
observed initial-election count.

## Reproduce

Python 3.12. Use the existing article's pinned
[`../requirements.txt`](../requirements.txt) for scientific dependencies:

```bash
python -m pip install -r midterm-forecast-fit-vs-prediction/requirements.txt
python midterm-forecast-fit-vs-prediction/post_election/analysis.py
python -m unittest discover -s midterm-forecast-fit-vs-prediction/post_election/tests -v
```

The analysis has no network dependency. It writes:

- [`results/provisional_diagnostics.json`](results/provisional_diagnostics.json):
  labeled, deterministic derived values from the source snapshot and lock;
- eight `figures/fig*.png` scientific illustrations;
- no changes to pre-election model, outputs, locks, historical rows or
  registered estimators.

All unit tests are source-aware. They reject a snapshot mislabeled as final,
an observed target inserted into the live record, invalid seat-accounting
arithmetic and forecast-lock drift. They also verify that the registered
signed-error convention is `prediction - observation`, that the interval
contains 172 but not 171 Republican House seats under the fixed 220-seat
baseline, and that eight PNGs can be rendered from the archived inputs.

## Figure map

| Figure | Evidential role |
| --- | --- |
| `fig1_registered_interval_and_live_scenario.png` | Frozen OLS point/95% conditional interval versus a labeled AP if-leads-hold scenario |
| `fig2_called_seats_vs_uncalled_leads.png` | AP called winners are distinguished from candidates still leading |
| `fig3_frozen_estimators_vs_live_scenario.png` | Four already registered point estimators, no winner selected after seeing returns |
| `fig4_integer_interval_hit_boundary.png` | The one-seat threshold that changes conditional interval inclusion |
| `fig5_error_sensitivity_to_final_totals.png` | Signed-error convention and hypothetical seat-outcome sensitivity |
| `fig6_model_interval_vs_bootstrap_sensitivity.png` | Model-based interval versus *uncalibrated* bootstrap sensitivity band |
| `fig7_ex_ante_to_unverified_and_final_timeline.png` | Information cut-off, registration, election, live snapshot, future source |
| `fig8_history_vs_one_provisional_discrepancy.png` | Historical forward RMSE versus one conditional discrepancy, expressly not a calibration test |

For publishing in Medium, the images should be uploaded and explicitly
captioned; an uncalled lead must never be labeled as a winner. The manuscript
[`article.md`](article.md) remains a **draft** until the final initial-election
outcome can be scored, verified, and independently reviewed. Once the
House Historian or American Presidency Project updates its relevant series,
run the existing verified scorer into a **new** JSON report in this directory,
and then issue a new, dated article revision without rewriting this archived
4 November snapshot.

## Scope and limitations

- The model predicts the president's party *change in House seats*, not a
  Senate majority, gubernatorial races or national vote shares.
- The frozen 220-seat baseline is the original conditional **model input**;
  it need not equal the AP's election-day incumbent membership total.
- AP's current leads have no assigned probabilities here; an if-leads-hold
  scenario is not a national probabilistic forecast.
- A seat within the nominal 95% prediction interval does not validate its
  long-run coverage, and one near a boundary does not falsify it.
- The 18 training elections and 10 forward evaluation elections stay exactly
  as in the original manuscript. There is no model refit or leakage.
- The source checksum, frozen forecast, reference file and dated snapshot
  together distinguish genuinely prospective prediction from post-hoc fitting.

See [`CLAIM_AUDIT.md`](CLAIM_AUDIT.md) for a claim-by-claim evidence ledger.
