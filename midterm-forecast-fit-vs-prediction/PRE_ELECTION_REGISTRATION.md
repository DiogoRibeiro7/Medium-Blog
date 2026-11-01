# A reproducible pre-election lock for the 2026 House forecast

**Original information cut-off:** 31 October 2026.

**Registration record prepared:** 1 November 2026.

**Federal general election:** 3 November 2026.

This registration is a new, append-only audit record for [the existing article](article.md), not a newly fitted or tuned forecast. Its purpose is to make later scoring falsifiable. The model, historical polling transcriptions, original conditional inputs, numerical interval and alternative estimators were fixed before the 2026 election's outcome existed. The 31 October *information* date and 1 November *registration* date are different; the latter must not be retrospectively labeled as an earlier preregistration.

The exact machine-readable record is [`results/forecast_lock.json`](results/forecast_lock.json), created with [`forecast_lock.py`](forecast_lock.py) from the frozen historical source series and computed evidence. It includes:

| Field | Locked value or definition |
| --- | --- |
| Forecast target | Republican president-party change in U.S. House seats; **not** the probability of a majority |
| Generic ballot | Democratic lead of 8 percentage points, expressed as a Republican-oriented margin of −8 |
| Republican seats defended | 220 |
| Historical training observations | 18 elections, 1950–2022, excluding unsourced 1986 |
| Primary estimator | Ordinary least squares, intercept + generic ballot + seats defended |
| 2026 OLS point prediction | **−32.3228880263 seats** |
| Conventional conditional 95% prediction interval | **[−48.4283865506, −16.2173895020]** seats |
| Bootstrap/residual 95% sensitivity interval | **[−41.7517081113, −14.3739743229]** seats; *not* a separate calibrated interval |
| Alternative point estimates | Student-t −33.7575, ridge −31.0608, recency-weighted least squares −30.9760 |
| Forecast evaluation | Signed error = registered forecast minus final observed seat change; absolute error; inclusion in original conditional interval |

The record also contains twelve SHA-256 fingerprints of normalized UTF-8 source files, numerical snapshots, notebook and pinned requirements. Newline normalization avoids representing harmless Git checkout CRLF/LF differences as scientific changes. The referenced frozen source commit is [`fe98742`](https://github.com/DiogoRibeiro7/Medium-Blog/tree/fe98742efb86756fc30967477098482eb7505783). Git's content-addressed history makes the recorded version inspectable, but is **not independent timestamp notarization**, and we do not claim that the earlier historical model specification was chosen blind to later historical data. Only the *2026 outcome* is prospective.

## Source definition and outcome measurement

The [Federal Election Commission](https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/) identifies Tuesday, 3 November 2026 as the regular federal election day. The [U.S. House Historian](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/) publishes party divisions based on the **initial election results** for each Congress. The [American Presidency Project](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections) derives its midterm House gains/losses from that series. Those are the definitions used for the study's historical labels, and they should govern the later observed outcome.

This distinction matters. The election-night projections on 3 November can still include uncertain districts, late counts or unresolved recounts. Vacancies, special elections, mid-Congress party changes and present-day chamber membership are not interchangeable with initial-election party divisions. The observed 2026 seat change must be taken from a **final** source consistent with the initial-election series, with its exact publication or retrieval date and URL recorded. A change in the definition of seats won must not be disguised as a forecasting error.

## Locked scoring protocol

1. Keep the original October 2026 forecast and its 1 November registration artifact untouched. Do not re-estimate the model using 2026 outcomes before scoring it.
2. Wait until after 3 November **and** until a final House Historian-aligned result is available. Preserve the publisher, retrieval date, numerical seat change and link.
3. Compute `signed_error = forecast - observed`, `absolute_error = abs(signed_error)`, and the indicator `lower <= observed <= upper` for the original OLS 95% conditional interval.
4. Evaluate the three already registered alternative point forecasts separately against the **same** observed seat-change definition. Their historical error rankings do not make the alternatives calibrated probabilistic forecasts.
5. Record the outcome and calculations in a **new** `post_election/` report. Never append the observed result to `data/historical_midterms.csv`, overwrite the registered scenario, change the locked interval or replace the 31 October point estimate.
6. Interpret one prospective outcome as **one** forecast score. It cannot validate 95% coverage, prove causation or select a universally optimal model.

The provided [`score_after_election.py`](score_after_election.py) implements steps 2–5 once the election has happened. It is intentionally inoperative before election day. A later analyst must explicitly confirm the final status of an allowed House Historian or American Presidency Project HTTPS source. When the result is available, an example invocation is:

```bash
python score_after_election.py \
  --actual-change SIGNED_INTEGER_FROM_FINAL_SOURCE \
  --official-url https://history.house.gov/Institution/Party-Divisions/Party-Divisions/ \
  --source-retrieved-on YYYY-MM-DD \
  --confirmed-final \
  --output post_election/verified_2026_score.json
```

Replace the placeholders **only after a source-consistent final result exists**. The program refuses dates before 3 November, non-integer or impossible seat changes, nonapproved publishers, unconfirmed partial counts, attempts to overwrite an existing report, and attempts to overwrite the forecast lock. No official 2026 outcome is supplied or evaluated in this repository at registration time. Any unit-test examples that use possible seat changes are clearly artificial and do not record election observations.

## Reproduce or independently audit the registration

Run the following from this directory to regenerate the manifest into a temporary root:

```bash
python analysis.py --output /tmp/midterm-reproduction
python model_extensions.py --output /tmp/midterm-reproduction
python comparison_uncertainty.py --output /tmp/midterm-reproduction
python forecast_lock.py --output /tmp/midterm-reproduction
python -m unittest discover -s tests -v
```

The path-scoped GitHub Actions job checks the numerical evidence and reproduces the manifest with **exact SHA-256 fingerprints, metadata and declared forecasts**. A future study may create a new, explicitly versioned model or updated forecast, but it must be reported as such. It must not silently rewrite this pre-election record. The GitHub PR's public creation timestamp supplies an additional chronological reference for the registration work, subject to the normal limitations of a repository-hosted audit trail.
