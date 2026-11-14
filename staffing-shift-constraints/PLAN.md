# Pre-registration: shift constraints against forecast quality in staffing a 311 call centre

Written and committed on 14 November 2026, after recording only the column schema of the dataset
(`sources/schema.json`, no data values) and before any call count was downloaded. Anything changed after this
commit is listed under "Changes after registration", with its reason and date, and marked as such in the article.
The code was checked only on synthetic data (`tests/`).

## Question

A call centre staffs to a forecast of calls, but it can only staff in shifts. A forecast tells the planner how many
agents each hour needs; the shift rules decide how closely the roster can follow that. This plan asks which of the two
limits a roster more: the quality of the forecast, or the constraints on the shifts. The claim being tested is that
the constraints decide first: under realistic shift rules, the cost forced by the rules is larger than the cost left
by forecast error, and a better forecast is worth less the more rigid the rules are.

## Data

NYC Open Data, *311 Call Center Inquiry* (`wewp-mm3p`): one row per agent-handled call to New York City's 311
information line, with the date and time the call was received (`date_time`). The dataset's own note says that
since March 2020 it does not fully represent agent-handled calls, so the test uses 1 January 2016 to 31 December
2019.

`fetch.py` counts calls per clock hour on the portal (see "Changes after registration" for how), and writes them to
`sources/hourly_calls.csv`, which is committed with its SHA-256 in `sources/FILES` and the dataset's update time in
`sources/REVISION`. Later runs use the committed file.

**Structural checks, before anything else** (`results/structure.txt`): the number of calls per year and per hour of
the day; days with fewer than 20% of the median daily calls of their year; hours missing on the spring
daylight-saving day and doubled on the autumn one. Hours absent from the counts are hours with no calls. A test
week with any day below 20% of its year's median daily calls is excluded from the test and listed.

## Windows

Weeks run from Monday 00:00 to Sunday 23:59, local clock time. The test weeks are the 52 weeks from Monday 31
December 2018 to Sunday 29 December 2019. The roster for test week w is planned at the end of week w - 2, with data
up to then: a lead of 8 to 14 days, as when rosters are published a week ahead. Each week is planned on its own, as
a cyclic week (a shift that runs past Sunday midnight covers the first hours of the same week).

## Requirement per hour

For a forecast of lambda calls in an hour, the requirement is the smallest number of agents s with which an M/M/s
queue (Erlang C) answers at least 80% of calls within 30 seconds, NYC 311's own published target. Average handle
time (AHT) is not published; the main analysis uses 5 minutes, and the whole test is repeated at 3 and 8 minutes as
a sensitivity (reported, not deciding). An hour with a forecast above zero needs at least one agent. No breaks,
absence or other shrinkage are modelled.

## Forecasts

Four forecasts of the calls in each of the 168 hours of week w, from data up to the end of week w - 2:

- **two_weeks:** the same hour of the same weekday in week w - 2.
- **mean4:** the mean of the same hour in weeks w - 2 to w - 5.
- **gbm:** one LightGBM model with a Poisson objective, fitted once on the hours of 2016 to 2018 (from the sixth week
  of 2016, when all lags exist), never refitted. Features: hour of day, weekday, month, day of year, a US federal
  holiday flag for the day and for the day before and after, the same hour in weeks w - 2 to w - 5, and the total
  calls of week w - 2. Settings: 400 trees, learning rate 0.05, 31 leaves, at least 50 rows per leaf, seed 7,
  deterministic mode with 4 threads. No tuning.
- **oracle:** the calls that actually arrived. Not a forecast; the bound on what any forecast could do.

## Shift regimes

Five ways of covering the requirements, from most to least flexible. Each is an integer programme that minimises
paid agent-hours subject to covering every hour's requirement, solved with HiGHS through `scipy.optimize.milp`:

- **R0 hourly:** any number of agents in any hour; staffing equals the requirement. Not a real roster; the bound.
- **R1 8h any start:** 8-hour shifts starting at any hour of any day.
- **R2 8h + 4h:** as R1, plus 4-hour shifts starting at any hour, with 4-hour shifts at most 25% of paid hours.
- **R3 three fixed starts:** 8-hour shifts starting only at 07:00, 15:00 and 23:00.
- **R4 weekly tours:** each agent works the same 8-hour shift, starting at any hour, on 5 consecutive days, with the
  2 following days off.

## Cost

For each test week, regime and forecast, the planned roster gives a staffing level per hour. The requirement
computed from the calls that actually arrived is the yardstick. The cost of a roster is

excess agent-hours + k x shortfall agent-hours,

where excess is staffing above the actual requirement and shortfall is staffing below it, summed over the hours of
the week, with k = 1 and k = 3. Costs are summed over the test weeks and divided by the total actual requirement in
agent-hours, so 0.10 means waste and shortfall worth a tenth of the hours the calls actually needed.

The cost of a cell splits in two:

- **constraint cost** of regime R: cost(oracle, R) - cost(oracle, R0). What the shift rules force even with perfect
  knowledge of the calls.
- **forecast cost** of forecast F under R: cost(F, R) - cost(oracle, R). What the forecast's errors add under those
  rules.

Also reported: paid agent-hours, and the share of calls answered within 30 seconds under the actual calls (Erlang C
with the staffed agents in each hour), for every cell.

## Uncertainty

Test weeks are resampled with replacement, 2,000 times, seed 11; each resample recomputes every normalized cost from
weekly totals. A difference counts as established if its 95% percentile interval excludes zero in the stated
direction.

## Hypotheses

- **H1. The rigid rules bind first.** Under R3 and R4, the constraint cost is larger than the forecast cost of mean4
  and larger than the forecast cost of gbm, at k = 1 and k = 3. Eight comparisons; supported if all are
  established.
- **H2. Rigid rules shrink the value of a better forecast.** The cost saved by moving from two_weeks to gbm is
  smaller under R4 than under R0, at k = 1 and k = 3. Two comparisons; supported if both are established.
- **H3. Flexible rules leave the forecast as the binding limit.** Under R1 and R2, the constraint cost is smaller
  than the forecast cost of gbm, at k = 1 and k = 3. Four comparisons; supported if all are established.

## Also reported

- All 4 forecasts x 5 regimes x 2 values of k, with paid hours and service level.
- The weighted absolute percentage error of each forecast against the calls, over the test weeks.
- The sensitivity to AHT: H1 to H3 at 3 and 8 minutes.
- For R4 with the oracle, in the test week with the median number of calls, the shadow prices of the hourly
  coverage constraints from the linear relaxation: which hours bind, and by how much.

## Changes after registration

All three before any count was received, on 14 November 2026:

- The portal rejected the first form of the query (HTTP 400). The hourly alias was renamed and the grouping written
  as the expression itself.
- The portal then rejected `date_trunc_ymdh(date_time)` as an unknown function, and also its forms with the time
  zone 'America/New_York' or 'UTC' and with a cast to a floating timestamp.
- The counts are now taken from the dataset's own DATE field and the hour of its TIME field ('12:00:16 AM'): for
  each year and each of the 24 clock hours, one query counts the calls per date whose TIME starts with that hour
  (with or without a leading zero) and ends with AM or PM accordingly. A second query per year counts all calls per
  date; the counts by hour must add up to it within 0.5%, or the run stops. Both totals are recorded in
  `sources/REVISION`.
