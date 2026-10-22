# Definition breaks and data vintages (protocol)

Registered on 2026-10-22 and committed and pushed before the new data were downloaded or
any estimate below was computed. Changes after the first computation go in the change log
at the end, with reasons.

## 1. Question, and what was already seen

An external assessment (2026-10-22) found two gaps in how the study treats measurement:

- The Bayesian models mask observations affected by the 2021 Labour Force Survey
  redesign, but the classical headline does not.
- The results have not been checked against another Eurostat release of the data.

**Already known before this protocol was written.**

- The classical review's `quality_screened` sample applies the rule of section 2. Its
  two-lag sums are committed in `docs/review/grid.csv` and enter the core robustness
  verdicts.
- The number of headline rows the rule removes was counted from that grid while
  designing this study. It removes 14 rows from ES and 13 from FR, which is all of their
  post-2021 rows, and 1 to 6 rows in ten other countries.
- No headline inference (small-sample bootstraps and their counts) has been computed on
  the screened rows. No new release has been downloaded.

## 2. The study's single definition-break rule

There is one rule, `review.flagged_change`, applied to a change in the unemployment rate
at quarter t:

- **Labour-force breaks.** A break flag (`b`, `bu`) invalidates the change into the break
  quarter.
- **Other labour-force flags.** A low-reliability or definition-differs flag (`u`, `bu`,
  `d`) invalidates every change that touches the flagged quarter. ES and FR carry `d`
  from 2021-Q1 onwards.
- **GDP.** A break flag on any GDP regressor, up to the lag order, invalidates the row.
- **Unflagged changes.** The rule cannot see these. The Netherlands phased the new
  survey in during 2021 without a flag. Every headline row whose window touches
  2020-2021 is already excluded, so the phase-in enters no headline row. In analyses that
  use 2021, such as the full-sample variants and the monthly study, it remains a stated
  caveat.

**Where the rule applies, before this study.**

- The Bayesian likelihood always applies it.
- The classical analysis applies it in the `quality_screened` robustness sample.
- The classical headline does not. Its sample was registered in METHODS.md as
  pandemic-excluded, without the rule, and it stays primary. A sample cannot be
  re-registered after its results are known.

This study adds the rule to the classical headline as a registered sensitivity
(section 3). Section 6 says how the manuscript reports it.

## 3. Sensitivity A: the headline inference under the rule

The headline small-sample study (HEADLINE_INFERENCE_PROTOCOL.md) is repeated on the
screened rows: the `quality_screened` sample, K = 2, no participation control. Nothing
else changes:

- the same statistic and bandwidth;
- the same sieve and block bootstraps, with B = 9999;
- the same seeds, so that any change comes from the rows;
- the same counts.

The implementation must first reproduce the committed `quality_screened` sums and
standard errors of `docs/review/grid.csv`, to 1e-10.

**Reported.**

- Each country's sum, with normal and bootstrap-t intervals, next to its headline value.
- The counts:
  - negative sums;
  - sums significant under the normal reference and under both bootstraps;
  - the same after Holm, and after Benjamini-Hochberg.
- The countries whose classification changes.

## 4. Sensitivity B: the latest Eurostat release

The three headline series are downloaded again with the study's own command, into a new
snapshot `snapshots/eurostat-<date>/`. The command is `eu-okun fetch --start-quarter
2008-Q1 --end-quarter 2026-Q2`. The period is the frozen one, so that only revisions
differ.

**The limit of this check.** Eurostat publishes no historical vintages of these country
series. It does so only for EU and euro-area GDP. This check therefore covers only the
revisions published between 2026-10-15 and the download date. It is not a test against
the full revision history.

**Reported.**

- **Revisions, by series and country:**
  - the share of cells that changed;
  - the mean and largest absolute revision;
  - changes in flags.
- **The classical headline on the new release:** each country's sum next to the frozen
  one, and the same counts as in section 3. The counts come from the normal reference and
  from both bootstraps, with the registered settings and seeds.

## 5. Interpretation, fixed in advance

These are sensitivity analyses. Neither replaces a registered result.

- A count or a country classification that changes is reported, with the countries.
- If no count changes under either sensitivity, the manuscript says so in one sentence.

## 6. Use in the manuscript (fixed in advance)

- **Data section:** the rule of section 2, where it is applied, the Dutch caveat, and
  the vintage check with its limit.
- **Classical section:** one paragraph with the screened counts, next to the headline
  counts.
- **Discussion:** the limitation on the 2021 redesign is updated with the result.

## 7. Outputs

- `eu-okun measurement` writes:
  - `docs/measurement/breaks.csv`;
  - `docs/measurement/revisions.csv`;
  - `docs/measurement/vintage.csv`;
  - `docs/measurement/summary.json`;
  - the report `docs/MEASUREMENT.md`.
- The new snapshot is audited and added to the publication audit.
- A test regenerates the outputs and checks the reproduction of section 3.

## Change log
