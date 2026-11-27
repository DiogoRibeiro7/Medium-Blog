# Plan of the experiment

Written on 27 November 2026, after the code was complete and before the main run. A pilot of
30 replications on network seed 11 was used to debug the simulator (it found that a 15-minute
buffer at outstations made the routing infeasible, which led to the hub-only buffer rule in
`planning/rules.py`); its numbers are not reported. Everything below was fixed before
`configs/experiment.toml` was run.

## Question

Does the plan that uses aircraft and crews most efficiently also give the most resilient
operation, or does efficiency buy fragility? The article treats this as a hypothesis, not a
conclusion.

## Network and data

- A synthetic Munich-hub airline: 30 outstations, 142 daily flights, one fleet of 180 seats,
  one crew base, maintenance checks at the hub at least every four nights. Outstations with two
  or more daily frequencies get one "night-stop" pair (evening out, early-morning back), a
  common outstation pattern.
- Observed inputs: airport coordinates and time zones (OpenFlights, ODbL, pinned commit and
  checksum), block times calibrated to typical Munich-Porto scheduled times, the mean primary
  delay per flight from EUROCONTROL CODA (2024: 17.5 minutes all causes minus 8.0 reactionary),
  and EASA FTL Table 2 and rest rules.
- Synthetic: the schedule, the passengers, every crew pairing and every cost parameter.

## Planners

1. `greedy`: first-in-first-out aircraft, greedy set cover of flights by legal pairings.
2. `sequential`: aircraft routing first (fewest aircraft, tightest turns, maintenance), then crew
   pairing by column generation with the routing fixed.
3. `integrated`: routing and pairing in one model (short connections linked to aircraft arcs).
4. `robust`: the integrated model with a 15-minute buffer on hub turns and crew aircraft changes,
   plus a penalty pricing each minute of expected propagated delay at the EUROCONTROL reference
   value (€166) times weight 1.

Every plan is checked by an independent validator against the nominal rules.

## Evaluation

- 300 replications per plan and scenario, common random numbers, a two-day horizon.
- Scenarios S0 (baseline) to S6 (delayed inbound aircraft, AOG at an outstation, two sick crews,
  hub capacity at 50% for four hours, a stressed day with doubled primary delays, an evening
  capacity crunch that pushes crews toward their FDP limits).
- The delay distribution's shape (probability of a primary delay and log-normal sigma) is
  calibrated so that the sequential plan's simulated departure punctuality matches CODA's 69.2%
  for 2024; the reactionary share (CODA: 46%) is a validation check, not a target.
- Metrics: planned and operational cost, cancellations, passenger delay, missed connections,
  crew hotel cost, aircraft and crew utilisation, recovery time against the same replication's
  baseline, solve time, gap to the column-generation LP bound.

## Hypotheses (stated before the main run)

- **H1.** The integrated plan's planned cost is at most the sequential plan's (guaranteed by
  construction); the saving is below 2% of planned cost on this network.
- **H2.** Minimising planned cost in the integrated model does not lower expected operational
  cost; it may raise it, because the model uses the extra freedom to tighten connections.
- **H3 (the article's question).** The plan with the lowest planned cost does not have the lowest
  total expected cost: under the baseline delays the robust plan's total (planned plus
  operational) is lower than both the sequential and the integrated plan's, with a paired 95%
  interval that excludes zero.
- **H4.** The value of buffers grows with the delay climate: the buffer that minimises total cost
  is larger at the stressed climate (19.0 minutes of mean primary delay) than at the November
  2025 climate (6.8).
- **H5.** A penalty computed from a point forecast of arrival delay performs no better than no
  penalty, while a penalty computed from the predictive distribution performs better.

## Robustness

- The four planners on four more network seeds (12 to 15), S0, S1 and S5.
- Robust plans over buffers 0-45 minutes and penalty weights 0-2 (Pareto frontier).
- Buffer-only plans under three delay climates.
- Operational cost is itemised by the model; the EUROCONTROL €166 per minute value applied to
  departure delay minutes is reported alongside as a cross-check, not added.

## Changes after this plan was written

Recorded in the order they happened; none changes a parameter, a scenario or a hypothesis.

1. The first launch of the main run stopped with a recursion error in passenger rebooking. When every
   departure on a hub route was cancelled in a replication, the search for a rebooking through the hub
   called itself without end. Such passengers are now stranded (their delay is capped at the rebooking
   horizon and a hotel night is charged), as the simulator already did for spoke-to-spoke itineraries
   with no seat left. The pilot never cancelled a whole route, so it did not meet the bug.
2. The greedy planner's overnight step could pair an aircraft that lands less than one turn before a
   departure with that departure, which a daily routing reads as a too-short same-day turn. It now picks
   the first leftover aircraft that avoids this; when none does, the validator reports the turn. The
   greedy planner ignores maintenance by design: its maintenance violations are reported and the plan is
   still simulated, because the simulator has no maintenance events. Its planned cost therefore
   understates what a legal version would cost.
3. Added before the main run, after this plan: the paired difference between the robust and the
   integrated plan (H3 compares the robust plan with both, and the runner had stored only differences
   against the sequential plan), and the duty and pairing listing for Porto used by one figure.
4. Added after the results, labelled as descriptive in the summary and the article: the break-even share
   of disrupted days at which the robust plan's extra planned cost pays for itself.
5. 27 November 2026, after the results, at the author's request: references to a real airline's
   flights were removed from this plan, the code comments and the article. The opening case and
   Figure 1 now use the synthetic airline's own Porto flights and the crews of the integrated plan,
   read from the results. The model code changed in comments only; the full experiment was rerun
   to confirm that every result is unchanged before the inputs digest was updated.
