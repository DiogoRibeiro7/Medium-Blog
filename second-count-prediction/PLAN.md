# Pre-registration: does the agreement of two counts predict the smallest loss they can catch?

Written and committed on 12 November 2026, before any count from the new sources was fetched. The workflow
fetches only names, sizes and HTTP status until this file exists (`sources/METADATA.json`,
`sources/citibike_bucket.json`, committed in the previous commit).

What had been seen before this file was written: web-search snippets that quote a few of the outside values
(some months of NYC DOT's Citi Bike table, the operator's totals for a few months of 2014, 2015 and 2026);
the sizes of the Citi Bike trip archives (among them, the April 2026 archive is 165 MB against 575 MB for March
and 917 MB for May); and Part 4's results on the four NYC TLC feeds. No count of a Citi Bike file, no Chicago
value and no ratio between two counts of the new pairs had been seen.

## The conjecture under test

From Part 4 of *Code Quality for Data Work* (conjecture 1), stated there before these data were chosen:

> For any recurring feed, how well the delivered row count agrees with an independent count on clean months
> predicts the smallest random loss that can be caught. Take the 95th percentile, over clean months, of how
> far the logarithm of their ratio strays from its median over the previous twelve. At a 5% alarm budget, a
> loss is caught nine times in ten once it is about twice that percentile.

The rule was read off Part 4's four pairs, so they cannot test it. Four new pairs do.

## Pairs

Test pairs (new data):

1. **Citi Bike / operator reports.** Delivered: rows of the New York trip files by month of trip start.
   Outside: the month's total trips in the operator's monthly operating report to NYC DOT.
2. **Citi Bike / DOT table.** Delivered: as above. Outside: "Citi Bike Customer Trips by Month" in NYC DOT's
   quarterly Bike Share Usage Data Report.
3. **Chicago / vehicles.** Delivered: trips reported by Transportation Network Providers to the City of
   Chicago, counted by month of trip start on the city's portal. Outside: NUMBER_OF_TRIPS summed by
   MONTH_REPORTED over the vehicle records.
4. **Chicago / drivers.** Delivered: as above. Outside: NUMBER_OF_TRIPS summed by MONTH_REPORTED over the
   driver records.

Calibration pairs (Part 4, already seen, reported but not a test): the NYC TLC yellow, green, FHV and
high-volume FHV files against the TLC monthly report, evaluated from 2021-05 as in Part 4.

## Counts

- **Citi Bike files.** All keys of the public bucket for New York (keys starting with "JC-" are Jersey City and
  are left out). The bucket holds yearly archives for 2013 to 2023 and monthly archives from 2024-01. For a year
  with monthly archives only those are used; otherwise the yearly archive. Every CSV inside an archive, and
  inside zips inside it, is read; a CSV is left out as a copy if its base name was already counted in the
  same archive, or if it has exactly the same number of rows in every month as a CSV already counted. Rows
  are assigned to the month of their start time (`started_at`, `starttime` or `start time`, whichever the
  file has), read as `YYYY-MM-DD...` or `M/D/YYYY ...`. Rows whose start time has neither form are counted
  apart, reported and left out. Months from 2013-06 (the system opened on 27 May 2013).
- **Operator reports.** The PDF links on the operating-reports page, one per month (the first link found for
  a month). The total is the first match, in this order, of: "There were N trips in", "riders took N trips",
  "There were N total trips", "N total trips", in the text of the first six pages. A value under 10,000 is not
  taken as a month's total. Months without a usable total are left out of pair 1 and listed.
- **DOT table.** The first of these documents that downloads and yields at least 24 months: the Q3 2026, Q2
  2026, Q1 2026, Q4 2025 and Q3 2025 xlsx files on nyc.gov (home4, then www), then the Q2 2025 and Q1 2025
  PDFs on the city's publications portal. Months marked as partial (*) are left out.
- **Chicago trips.** count(*) grouped by month of `trip_start_timestamp` in the three trips datasets
  (m6dm-c72p, 2018-11 to 2022-12; n26f-ihde, 2023-2024; 6dvr-xwnh, 2025 on), summed; one count per month if
  the grouped query fails.
- **Chicago vehicles and drivers.** sum(number_of_trips) grouped by month_reported in bc6b-sq4u and j6wf-834c.
  The source sets values above 999 to null, so these sums can only be low on that account.

A pair uses the months in which both of its counts exist.

## Detector (unchanged from Part 4)

r_t = log(delivered_t / outside_t). Robust z of r_t against the 12 previous months of the pair (median,
1.4826 x median absolute deviation, scale floored at 0.005). A random loss of a fraction p of the delivered
rows adds log(1 - p) to r_t. The threshold is the one at which the detector fires on at most 5% of the
pair's evaluated clean months. A faulted month is caught if the detector fires on it and does not fire on the
same month without the loss. Evaluated months: every month of the pair with 12 earlier months in the pair
(from 2021-05 for the calibration pairs). The unit test `tests/test_detect.py` reproduces two of Part 4's
published detection rates with this code.

## Prediction rules

- **p95 rule (under test).** D = 95th percentile, over the evaluated months, of |r_t - median of the 12
  previous r|. Predicted smallest loss caught at least 90% of the time: L = 1 - exp(-2D).
- **Gaussian rule (comparison).** S = median, over the evaluated months, of the robust scale (1.4826 x MAD) of
  the 12 previous r, without the floor. L_g = 1 - exp(-2 x 1.96 x S). It uses the typical spread and ignores
  the tails.

## Observed

Losses on a grid of 28 values from 0.1% to 50.1%, each 10^0.1 times the one before. Detection at each loss as
above. L90 = the smallest grid loss from which detection is at least 90% at it and at every larger grid
loss; if there is none, L90 is "not reached".

## Hypotheses

- **H1 (primary).** In each of the four test pairs, L/2 <= L90 <= 2L. If L90 is not reached, H1 holds for
  that pair only if 2L > 50.1%. H1 is supported if it holds in all four.
- **H2.** In at least three of the four test pairs, |log(L90 / L)| < |log(L90 / L_g)| (a tie counts against
  H2; an L90 not reached is taken as 50.1%).

If a pair cannot be built because a source cannot be fetched, it is dropped and reported; H1 is then judged
on the remaining pairs and H2 needs all but one of them.

## Also reported (descriptive, not tests)

The ratio of the two counts month by month for every pair; the real months on which the detector fires at
the matched threshold, with no loss planted; D, S, L, L_g and L90 for the calibration pairs.

## Changes after registration

None yet. Any change will be listed here with its reason and date, and marked in the article.
