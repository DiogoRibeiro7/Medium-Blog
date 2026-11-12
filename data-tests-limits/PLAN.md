# Part 4 plan: what a delivery cannot vouch for

Pre-registration for Part 4 of *Code Quality for Data Work*. Written and committed before any profile, row
count or report value of the feeds below was computed or seen. What had been seen: the text of the yellow,
FHV and high-volume FHV data dictionaries of 18 March 2025; the column names, licence classes and month range
(2019-01 to 2026-09) of the TLC monthly report on NYC Open Data; and the Part 3 results on the yellow feed.
The green dictionary could not be read in this session, so green enters only the second experiment, which
needs no dictionary.

Thesis: **a delivery cannot vouch for itself.** Part 3 ended with two conjectures, both about what checks
built from the delivery cannot see. This part tests them.

## E1. Rules written from a dictionary (Part 3, conjecture 1)

Part 3's conjecture: for a feed whose dictionary is older than the feed's last few changes, most table checks
written strictly from the dictionary fail on every delivery, and the share of rules that always fail grows
with the age of the dictionary.

- **Feeds and deliveries.** Yellow taxi (2019-01 to 2026-09), FHV (2019-01 to 2026-09) and high-volume FHV
  (2019-02 to 2026-09): every monthly file the TLC publishes in that range. A month without a file is
  recorded and left out.
- **Rules.** `feed/contracts.py` holds one contract per feed, taken from the 18 March 2025 dictionary of that
  feed: field names; the kind of type each description implies; which fields every trip must have (every
  field, unless its description says it may be absent: FHV's `SR_Flag`, high-volume FHV's
  `on_scene_datetime`); amounts, distances and durations that cannot be negative; listed codes, and zones from
  the zone lookup (1 to 265); timestamps whose descriptions imply an order; the first month of a field the
  dictionary dates (`cbd_congestion_fee`, January 2025). `feed/rules.py` turns them into up to seven rule
  kinds (schema, not null, ranges, code lists, month window, unique rows, and reconciliation for yellow),
  each strict (any violating row) and tolerant (more than 1% of rows for some field). Yellow is rerun under
  these rules, which differ from Part 3's (zones added; the passenger and 24-hour limits, which are not in
  the dictionary, dropped; every described field required).
- **Outcome.** Per feed, the share of strict rules, and of tolerant rules, that fire on every delivery; the
  number of rules firing per delivery.
- **H1a.** In each of the three feeds, more than half of the strict rules fire on every delivery.
- **H1b.** The number of rules firing on a delivery (both variants) grows with the distance in months between
  the delivery and the dictionary (March 2025): Spearman correlation above zero with a one-sided permutation
  p below 0.05 (10,000 permutations, seed 20261112) in at least two of the three feeds. Months are
  autocorrelated, so the permutation p overstates the evidence; it is a decision rule, not a precise level.

## E2. Random losses of rows (Part 3, conjecture 2)

Part 3's conjecture: for faults that remove rows at random, no metric computed from the delivery alone
approaches the catch rate of the other silent faults (82% in Part 3) at any threshold; only a control total
from outside the delivery does.

- **Feeds and counts.** Yellow, green, FHV and high-volume FHV. The delivered count of a month is the row count
  in the Parquet footer of its file. The outside count is the TLC monthly report's "Trips Per Day", summed
  over the feed's licence classes (Yellow; Green; FHV - Black Car, Livery and Lux Limo; FHV - High Volume),
  times the days of the month.
- **Faults.** A random loss of 1%, 2%, 5%, 10%, 15% or 20% of the rows of one delivery, planted in each
  evaluated delivery in turn; history is always the clean deliveries. A random loss multiplies the count by
  one minus the loss and leaves shares, mixes and quantiles where they were, so only counts can carry it;
  the binomial noise of the loss is ignored (its relative standard deviation, the square root of
  p / ((1 - p) n), is under 0.3% even for 50,000 rows and a 20% loss).
- **Detectors** (`feed/counts.py`), each a robust z-score (median, 1.4826 x median absolute deviation, scale
  floored at 0.005) against the 12 previous clean deliveries, firing at |z| > 5:
  - *volume*: log rows (the Part 3 check, here with the common floor of 0.005 instead of 0.01);
  - *seasonal*: log rows minus a seasonal forecast (log rows twelve months earlier plus the median
    year-on-year change of the three previous months), the best delivery-only metric planned here;
  - *outside count*: log rows minus log of the report's count;
  - and one table check with an outside number and no history: file and report differ by more than 1%.
- **Deliveries evaluated.** 2021-05 to 2026-09 for every detector and feed (the seasonal detector needs 27
  months of history; high-volume FHV starts in 2019-02).
- **Caught.** A faulted delivery is caught if the detector fires on it and not on the clean delivery of the same
  month (as in Part 3). Reported at the fixed line and at a matched budget: the threshold at which the
  detector fires on at most 5% of the clean deliveries of that feed.
- **H2a.** At the matched budget, for losses of 5%, 10% and 20% in every feed, the better of the two
  delivery-only detectors (volume, seasonal) catches fewer than 82% of faulted deliveries.
- **H2b.** At the matched budget, the outside count catches at least as many faulted deliveries as the better
  delivery-only detector at every loss and feed, and at least 90% of them at a 5% loss in every feed.

## Fixed settings

Window 12 deliveries; line |z| > 5; scale floor 0.005 (log units); tolerance 1% for tolerant rules and for the
fixed outside-count check; budget 5% of clean deliveries; losses 1, 2, 5, 10, 15, 20%; evaluation from 2021-05;
dictionary date 2025-03. Anything added after the results are seen is labelled as a follow-up in the article.

## Changes after registration

- 2026-11-12, before any result was computed: the CDN refused the Parquet-footer reads after about 190 files,
  so the delivered count of a month is taken from the profile of its file (count of rows over the whole file,
  the same number the footer holds). Green, which has no contract here, is profiled for its row count and
  schema only. No setting of either experiment changed.
