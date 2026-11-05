# Final 2026 House outcome: evidence handoff

**State on 5 November 2026: blocked, not scored.** The House Historian's
[party-division series](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/)
still ends with the 119th Congress. The American Presidency Project's
[midterm seat-change series](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections)
shows the 2026 220-seat defended baseline but leaves the House seat-change
cell blank. There is no final source-consistent outcome to submit today.

The original 31 October forecast, 1 November lock and
[`../score_after_election.py`](../score_after_election.py) must not change.
The live AP reports of 4 and 5 November are informative, but neither can
be passed to the registered final scorer.

## Required independent evidence

Only when both official publications have finalized their relevant rows:

1. Inspect the House Historian's initial-election **120th Congress** division,
   not the current membership of the 119th Congress. Record Democratic,
   Republican and Other initial-election seats and retrieval date.
2. Inspect the American Presidency Project's **2026** president-party House
   seat-change column and its House seats-to-defend column. Record both,
   the publisher URL and retrieval date.
3. Confirm the 120th House counts total 435; the 2026 defended baseline is
   220; and the Republican 120th count minus 220 equals the separately
   reported president-party House seat change. Resolve any independent or
   caucusing member attribution *before* the automated handoff.
4. Save a dated, human-reviewed, non-overwriting JSON transcription of the
   two official rows. Keep a captured source copy, retrieval record or
   version reference for independent inspection. The code cannot prove that
   a manually entered number actually appears on a website.
5. Prepare the source handoff, inspect its proposed scoring command, and
   then run the **original** registered scorer separately. A new score must
   be recorded in a new report, never in the forecast lock or historical
   training data.

## Receipt format (template only; intentionally invalid)

The `null` counts and `false` confirmations are intentional. Do not replace
them using AP's provisional results. This is a schema illustration, **not**
an election result.

```json
{
  "record_kind": "final_2026_initial_house_party_divisions",
  "election_date": "2026-11-03",
  "analyst_confirmed_direct_source_inspection": false,
  "house_historian": {
    "source_url": "https://history.house.gov/Institution/Party-Divisions/Party-Divisions/",
    "retrieved_on": null,
    "congress": 120,
    "initial_election_party_division": false,
    "published_final": false,
    "democratic_initial_seats": null,
    "republican_initial_seats": null,
    "other_initial_seats": null
  },
  "american_presidency_project": {
    "source_url": "https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections",
    "retrieved_on": null,
    "election_year": 2026,
    "house_seats_to_defend": 220,
    "republican_house_seat_change": null,
    "published_final": false
  }
}
```

When the *real*, independently checked receipt exists, run these commands
from `midterm-forecast-fit-vs-prediction/`:

```bash
python post_election/final_outcome_handoff.py \
  --evidence PATH_TO_VERIFIED_OFFICIAL_ROWS.json \
  --output post_election/results/official_source_handoff.json

# Review the generated handoff, then execute its proposed command explicitly.
# The handoff itself never computes a forecast error or runs a scorer.
```

The handoff rejects wrong publishers, current-membership substitutions,
missing final-status attestations, impossible party totals, non-integer
seats, mismatched baselines, incompatible source seat changes, future-dated
evidence and attempts to overwrite an existing output. If the two sources
disagree, investigation is required; no automatic source is preferred.

After scoring, report all four registered estimators with the same target,
the OLS signed and absolute errors, the original prediction-interval
inclusion and its assumptions. Reconcile the article's provisional AP
snapshots with the official figure, and prepare a *separately dated final
edition*. An in-interval single observation is not proof of 95% coverage.
