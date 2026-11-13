# Pre-registration: the p95 rule after a change of regime, and forward in time

Written and committed on 13 November 2026, before any split, segment prediction, segment detection or forward
window was computed. No new data are fetched: the input is the monthly ratio of the eight pairs of counts in
`../second-count-prediction/results/corrected/ratios.csv` (Citi Bike files as counted with the corrected rule;
Chicago and the four NYC TLC pairs as before).

What has been seen: every value of those ratio series (they are drawn and described in the second-count-prediction
article), the results of that study (the p95 rule on the whole history of each pair, and the detection observed on
the whole history), and therefore, roughly, where the large level shifts are (Citi Bike in 2014-2015, the operator
reports from March 2025, TLC FHV from January 2022, the drift of TLC yellow). Not seen: anything this plan defines.

## The conjecture

From the second-count-prediction article:

> For a pair whose agreement changed regime, the p95 rule computed on the months since the last level shift
> predicts the observed smallest catchable loss within a factor of two, for the months it is computed on.

It was read from one pair, Citi Bike files against NYC DOT's table, where the rule on the whole history predicted
10.7% and the detector needed 3.2%. That pair is reported but is not a test. The other seven are the test pairs.

## Settings (unchanged from second-count-prediction)

r_t = log(delivered_t / outside_t). The detector, the 5% matched budget, the definition of a caught month, the
loss grid (38 values from 0.01% to 50.1%) and the observed value (smallest grid loss from which detection stays at
or above 90%) are those of `second-count-prediction`; `feed/detect.py` is a byte-for-byte copy, checked by a unit
test. The p95 rule: D = 95th percentile of |r_t - median of the 12 previous r| over the months considered, and a
predicted loss of 1 - exp(-2D). The factor-of-two comparison, including the clauses for an observed value at the
top or the bottom of the grid, is `detect.within_factor_two`.

One month is dropped: TLC FHV, January 2019, whose file still holds the high-volume trips (Part 4).

## The split

For each pair, the month k that minimises the summed absolute deviation of r from the median of each of the two
segments, with at least 24 months before k and 36 from k on (the first minimum if there are ties). The later
segment is the months from k on. A pair has a **regime change** if the split removes at least half of the summed
absolute deviation of r from its overall median.

On the later segment, with the series cut to that segment (so every z-score history lies inside it):

- the **segment rule**: the p95 rule over the segment's months that have 12 earlier months in the segment;
- the **full-history rule**: the p95 rule over all the pair's months, as in second-count-prediction;
- the **observed** value: detection at each grid loss over the same segment months.

## The forward test

One window per January t, for every t at which the pair has at least 30 months with a full 12-month history in
[t - 36, t) and at least 20 in [t, t + 24), and t + 24 months does not pass the last month of the pair:

- the **trailing rule**: the p95 rule over the months in [t - 36, t);
- the **expanding rule**: the p95 rule over all months before t;
- the **observed** value over [t, t + 24): z-scores against the 12 previous months of the pair, the threshold
  matched to 5% of the window's clean months, detection and the observed value as above.

## Hypotheses

- **H1.** In each of the seven test pairs, the segment rule is within a factor of two of the observed value on
  the later segment. Supported if this holds in all seven.
- **H2.** In every test pair with a regime change, the segment rule is closer to the observed value on the later
  segment than the full-history rule (|log(observed / predicted)|; a tie counts against). Not testable if no test
  pair has a regime change.
- **H3a.** Over all forward windows of the test pairs, the trailing rule is within a factor of two of the observed
  value in at least two thirds of them.
- **H3b.** The trailing rule is closer to the observed value than the expanding rule in more than half of the
  forward windows of the test pairs in which the two are not equally close.

Windows of the same pair overlap by a year, so H3 counts are not independent; they are reported as counts, with
no test of significance.

## Also reported

The split month, the share of deviation it removes and the segment medians for every pair; the motivating pair's
results; every forward window.

## Changes after registration

None yet. Any change will be listed here with its reason and date, and marked in the article.
