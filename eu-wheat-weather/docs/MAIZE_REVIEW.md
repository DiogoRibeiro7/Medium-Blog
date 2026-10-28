# Critical review of the grain maize extension (2026-10-28)

This review was written after the maize results ([MAIZE.md](MAIZE.md)) and the draft maize
article ([../publication/maize/article.md](../publication/maize/article.md)), and before
publication. It follows the wheat study's review ([REVIEW.md](REVIEW.md)).

**How it was done.**

- **Three independent reviewers.** Each read the plan (METHODS.md, section 9), the code,
  the data and the outputs from one angle: econometrics and inference, data and
  measurement, and claims and communication. All three were read-only and asked for
  evidence for every finding.
- **Independent verification.** Every finding listed below as **verified** was recomputed
  independently, with new scripts, from the committed panel, weather, snapshot and
  results. Findings marked **reported** were computed by a reviewer and not recomputed;
  findings marked **plausible** rest on reasoning or outside knowledge.
- **Exploratory analyses.** Leave-one-out checks, data variants and group splits are
  exploratory: none was registered for maize, and none changes a registered verdict.
- **Nothing was changed.** No result, plan or text was changed in the course of this
  review.

## Bottom line

The code does what section 9 says, and both registered verdicts are computed correctly:
the units, the season windows, the order of the sample rules, the Holm family of two, the
bootstrap of the two-way statistic and the decision rule.

- **Drought (E2) is far more robust than heat.**
  - Its two-way p stays at or below 0.013 whichever country, year or suspect data series
    is left out.
  - Under the full registered rule it fails in one deletion. Without Spain, its bootstrap
    p of 0.033 becomes 0.066 after Holm, because heat is no longer rejected first.
- **Heat (E1) is established by the registered rule, but only at the margin, and it is
  not robust.**
  - Its two-way p is 0.0478. It is below 0.05 only because the two-way variance comes out
    slightly smaller than the country-clustered variance; with country clustering alone,
    p = 0.0507.
  - Leaving out any one of nine countries, or the single country-year Romania 2024,
    makes it not determined.
  - Romania supplies half of the estimate, and one small peri-urban region (RO32) a
    fifth.
  - The pooled −4.9% averages about zero in Spain, Portugal, Italy and Greece with about
    −11% elsewhere, so it describes neither group.
- **The placebo** does not threaten either result. Its two-way rejection comes from rain
  after maturity, with a positive sign, and that rain is unrelated to the season's
  weather. But the report's explanation (maize standing in the field) predicts the
  opposite sign.
- **The article** gets every number right, but three statements mislead and must change:
  "well within the usual swing", the selective robustness paragraph, and "established"
  for heat without its fragility.

## What happens to each conclusion

| Claim | Registered verdict | After review |
| --- | --- | --- |
| Placebo, 61 days after maturity | passes (two-way rejects, bootstrap does not) | **Holds** by the rule. The link is positive post-maturity rain, concentrated in France and late regions, unrelated to in-season weather. The two-way joint test over-rejects in this panel (I6). The harvest-timing explanation does not fit the sign. |
| E1, heat (−4.9% per 10 degree days above 30 °C) | established negative | **Holds by the rule, at the margin; not robust.** It turns on the variance estimator (I1), on Romania and Romania 2024 (I2), and on data choices (D1). The number mixes two populations (I3). |
| E2, drought (−3.3% in a dry season) | established negative | **Holds, and far more robust than heat.** Its two-way p stays at or below 0.013 in every deletion and data variant tried. Under the full rule it fails only without Spain (bootstrap p 0.066 after Holm). |
| M3, the bands | descriptive | **Holds as description.** 27-30 °C and 33-36 °C are negative under both methods. The 36 °C-and-above band is mostly Spain. "Where the harm starts" over-reads an irregular pattern (I9). |
| Sensitivity analyses | no sign reverses | **Holds.** But the "without the irrigated south" heat interval is about twice too narrow (I4), and that variant leans on Romania. |
| E4, out of sample | descriptive | **Holds.** The 2012 and 2022 misses are real yield failures in Romania and Hungary (D8). E4 checks M2, not E1 or E2 (I8). |
| Maize beside wheat | descriptive | **The heat difference cannot be told apart:** −2.3 points, ±7.4 (C6). |

## Findings

Severity: **high**, could change a stated conclusion or is likely to mislead; **medium**,
weakens a conclusion or its reading; **low**, a detail.

### Econometrics and inference

**I1. E1's two-way rejection depends on the two-way variance being slightly smaller than
the country one.** High; verified.

| Variance | SE of E1 | p (df 14) |
| --- | ---: | ---: |
| Two-way (registered) | 0.02303 | 0.0478 |
| Country | 0.02337 | 0.0507 |
| Year | 0.01168 | 0.0008 |
| Country-year | 0.01235 | 0.0012 |
| Country plus year | 0.02613 | 0.0766 |

- The two-way variance subtracts the country-year term, which here is slightly larger
  than the year term. No eigenvalue was clipped in any registered fit (reported).
- The registered rule was applied correctly. But E1's two-way leg rests on a small,
  noisy covariance term estimated from 15 years.

**I2. Single countries, and one country-year, decide E1's verdict.** High; verified.

- **Leave one country out, two-way:** E1's p rises above 0.05 without AT, BG, DE, EL, FR,
  HU, PL or RO (0.050-0.078). Without Spain, the two-way p is 0.043 but the bootstrap is
  0.081, so E1 fails too: 9 of 19 deletions flip it.
- **Romania supplies half of the estimate:** −0.0245 of −0.0499 (Frisch-Waugh shares),
  from 18% of the variation. RO32 (Bucuresti-Ilfov, under 0.2% of maize area) alone
  supplies −0.0109.
- **Romania 2024:** this one country-year (8 region-years) supplies −0.0128. Without it,
  E1 is −0.039 with two-way p 0.0501.
- **E2:** its two-way p stays at or below 0.0097 in every country deletion. Under the
  full rule it fails without Spain: its bootstrap p of 0.0332 becomes 0.066 after Holm.
- **Reported:** leaving one year out lifts E1's p above 0.05 for 8 of 15 years. Region
  linear trends with country-year effects strengthen E1 to −0.054; the closest variant
  in the code, region trends with year effects, gives −0.078 (verified).

**I3. The pooled E1 averages about zero in the irrigated south with about −12%
elsewhere.** Medium; verified.

- **Spain, Portugal, Italy and Greece alone:** −0.0070 (two-way p 0.52).
- **Pooled model with EDD × south:** −0.120 elsewhere, and +0.117 for the interaction.
- **Weights:** the four countries hold 58% of the heat variation but 15% of the maize
  area. So −4.9% is set by regression weights, not by where maize grows. The
  area-weighted variant (−8.1%) is a third, different number.

**I4. The narrow "without the irrigated south" interval comes from the two-way
subtraction.** Medium; verified. The two-way SE is 0.0076, against country 0.0115, year
0.0129 and country-year 0.0155. Romania supplies 42% of that variant's heat variation.
The sign is safe; the interval's width is overstated about twofold.

**I5. The placebo: positive rain after maturity, concentrated in France and late
regions, unrelated to the season.** Medium; verified.

- **The two terms:** post-maturity heat +0.008 (p 0.86), zero in 59% of region-years.
  Post-maturity rain +0.0143 per 100 mm (p 0.024).
- **France:** without it, the placebo's two-way p is 0.218.
- **No link to the season:** after the fixed effects, post-maturity rain correlates −0.003
  with season rain. Adding both terms leaves E1 at −0.050.
- **The mechanism.** Maize standing through wet weather loses yield, so the report's
  explanation predicts a negative sign. A maturity date that is too early for late
  regions, or chance, fits better (plausible).
- **Reported:** the two-way joint statistic is 8.0 (p 0.043) but 2.7 to 3.2 under year
  or country-year clustering.

**I6. How often the tests reject a true null: an incomplete simulation.** Medium;
reported, incomplete.

- **What was run:** fake weather (each region's years shuffled the same way) was added to
  the model. The run was stopped early to free the machine: 233 shuffles were saved, 99
  of them with the bootstrap.
- **What it found:**
  - the registered two-way t rejects 7.7% of the time at 5%;
  - the two-way joint test, the placebo's test, rejects 15.5%;
  - the registered rule that needs both methods rejects 3.0%.
- **Reading.** Two-way-only rejections, such as the placebo's, are weak evidence in this
  panel. The registered rule keeps its level.

**I7. Checks that passed.** Low; verified or reported as marked.

- **The bootstrap.** Its Monte Carlo SE at p 0.036 is 0.0019 (verified). Ten other seeds
  give 0.034-0.039 (reported).
- **Singletons.** Keeping the 5 dropped singletons changes nothing (reported).
- **GDD and EDD** correlate 0.29 after the fixed effects (verified). E1 is identified
  separately from GDD.
- **The decision rule.** Holm and the decision rule are correct (verified).
- **A cosmetic label.** A rejected positive E1 would read "rejected but not negative"
  rather than section 5's "contrary to expectation".

**I8. E4 checks M2, not E1 or E2.** Low-medium; reported.

- **The coefficients differ:** M2's EDD coefficient is −0.065 to −0.085, against M1's
  −0.050.
- **2022 extrapolates:** four of its test regions lie beyond the training sample's 99th
  percentile of heat.
- **Carried over from wheat:** the wheat review's E10 (the square of mean precipitation in
  the "normal") applies here too.

**I9. The bands.** Low.

- **Hours per season** (verified): 27-30 °C 208, 30-33 °C 105, 33-36 °C 37, and 9 at
  36 °C and above.
- **Spain** supplies 54% of the variation of time at 36 °C and above (verified).
- **12-15 °C:** its two-way SE (0.0046) is below every one-way SE, and the raw two-way
  matrix has 4 negative eigenvalues (verified).
- **No smooth threshold:** 27-30 and 33-36 differ from zero, but 30-33 between them does
  not.

### Data and measurement

**D1. E1's verdict turns on small data choices.** High for reporting; verified unless
marked.

| Variant (exploratory) | E1 | Two-way p (Holm) |
| --- | ---: | ---: |
| Registered | −0.0499 | 0.0478 |
| Without the 68 Eurostat-estimated Greek rows | −0.0502 | 0.0505 |
| Regions with at least 5 kha in the year | −0.0585 | 0.031 |
| Without six flat or repeated Italian series | −0.0635 | 0.019 |
| Heat over cells below 800 m (an April-September proxy; reported) | −0.0438 | 0.0535 |

Data defects pull both ways. E2 stays at p ≤ 0.005 in every verified variant.

**D2. Regional averages over terrain where no maize grows.** High for heat measurement;
reported.

- **How many regions:** 31 panel regions have less than 80% of their weight below 800 m,
  and their lowland heat is 1.2 to 3.9 times the regional mean.
- **RO32 against RO31:** much of RO32's heat contrast with its neighbour RO31 comes from
  RO31's hill cells. The gap is 12.9 degree days over all cells and 2.5 over cells below
  150 m.
- **Not in the plan:** no registered maize check weights cells by where maize grows.

**D3. Flat and repeated Italian series pass the data rule.** Medium; verified for ITH4
(11.28, 11.29, 11.29 in 2015-2017) and ITG2 (8.97, 9.01, 9.07 in 2019-2021).

- **What Q3 misses:** it catches only area and production that both repeat. ITF6's
  2022-23 repeats were caught; the yield-only repeats were not.
- **Recode joins (reported):** ITH4 also has a level break where its old and new codes
  join in 2014/15.
- **Effect:** dropping six such regions strengthens E1 (D1).

**D4. Eurostat's estimated rows: 68, all Greek.** Medium; verified.

- **The count:** MAIZE_DATA.md's "`e` 136" counts the area and production flags
  separately. It is 68 region-years, every Greek panel row from 2016.
- **Effect:** dropping them takes E1 to not determined under the two-way method (D1).
- **Reported:** Greece also has 13 of the 15 conflicts between old and new codes.

**D5. HR02 in 2013-2014 is not the NUTS-2021 region.** Low; verified. Its area and
production equal HR04's (Continental Croatia) in both years: 286.51 kha and 1,865.96 kt;
251.05 kha and 2,031.52 kt. Dropping HR02 2010-14 leaves E1 at −0.050 (reported).

**D6. Pro-rating the rule cannot reach.** Low; reported.

- **Below Q2's thresholds:**
  - five Czech regions move together while Q2 averages over all seven;
  - in two-region countries (LT, SI, HR, DK), Q2 needs four regions.
- **Possible false positives:** AT 2019-20 and CZ 2014-15.
- **Effect:** removing or keeping them barely changes E1.

**D7. The 1 kha floor admits marginal regions; area responds to failure.** Medium;
verified where stated.

- **Small regions (reported):** 20 regions average under 5 kha and are volatile. A 5 kha
  floor strengthens E1 (D1).
- **Area falls with failure:** yield is measured on the area harvested as grain.
  - HU12: 55 to 32 kha and 6.05 to 1.54 t/ha, 2021-22 (verified).
  - DEE0: 16.4 to 8.7 kha and 9.34 to 4.13 t/ha, 2017-18 (verified).
  - Spain loses irrigated area in dry years (reported).

**D8. The 2012 and 2022 misses are real failures in Romania and Hungary.** Medium;
verified.

- **2012:** RO −6.6 and HU −2.3 of a −9.7-point gap.
- **2022:** HU −4.4 and RO −4.4 of −9.9.
- **2022's variance explained is −0.08:** the regional predictions explain none of the
  differences between regions.
- **Reported:** yields are consistent across neighbouring regions. The failures look
  like rainfed drought that season-total rain cannot see.

**D9. The crop calendar is coarse in the south.** Low; verified for Spain.

- **Spain:** its panel regions plant 29 March to 1 April and mature 10-21 September.
- **Why it matters little for heat:** every season includes July and August, and
  post-maturity heat is small.
- **ES41 (plausible):** its planting is probably too early, since irrigated meseta maize
  is sown later.

### Claims and communication

**C1. "Heat: established, by a narrow margin" will read as "proven".** High.

- **What a reader isn't told:** the placebo's two-way p (0.043) is as small as heat's
  (0.048), and single deletions flip heat (I1, I2, D1).
- **Fix:** define "established" once, as the bar the plan set rather than proof.
- **Fix:** retitle the section, and say heat is "probably harmful, only just past the
  bar".
- **Fix:** carry the hedge into the wheat article's pointer, the README and the ROADMAP.

**C2. "So ten more is well within the usual swing from one season to the next" is
false.** High; verified.

- **What the quoted range measures:** 2.7-18.8 pools all region-seasons, so it is mostly
  differences between regions. The interquartile range of region means is 3.9-20.6.
- **Within a region:**
  - the median change from one season to the next is 5.4 degree days, and only 32% of
    changes reach 10;
  - after the fixed effects, the SD is 4.6 and only 4.6% of seasons sit 10 or more away.
- **Fix:** ten more degree days is a hot season. Change the claim test, which checks the
  pooled range.

**C3. The robustness paragraph is selective.** High; verified.

- **The drought number it leaves out:** without the south, drought is −2.9%, smaller than
  −3.4%. That contradicts "irrigation blunting the damage", which predicts a larger
  drought effect without the south.
- **Romania in both variants:**
  - the no-south heat estimate draws 42% of its variation from Romania (I4);
  - area weights give Romania 30%, France 19% and Hungary 12%.

  *Correction, 2026-10-28:* the 30%, 19% and 12% are shares of the sample's maize area.
  Under area weights, Romania supplies 48% of the identifying variation of EDD
  ([MAIZE.md](MAIZE.md), after review). The point stands, more strongly.

**C4. The placebo paragraph's explanation does not fit the data.** Medium; verified
(I5).

- **Name the driver:** say it is post-maturity rain with a positive sign, unrelated to the
  season.
- **The standing-crop mechanism:** present it only beside the calendar alternative.
- **"Only narrowly":** replace it, since the bootstrap p is 0.158.
- **The same sentence elsewhere:** MAIZE.md states the mechanism as fact, and
  `maize_analysis.render` hard-codes it.

**C5. The article never says which countries supply the heat comparison.** Medium;
verified. Spain supplies 28%, Italy 21% and Romania 18%. The four southern countries
supply 58% while growing 15% of the maize. The wheat article has a "Not every country
equally" bullet; the maize article needs one.

**C6. The comparison with wheat invites a difference that was not tested.** Medium;
verified.

- **The heat gap:** −2.3 points with a margin of about ±7.4 (z = −0.6), assuming
  independence.
- **The unit differs:** ten degree days above 30 °C is a far bigger shock inside wheat's
  harvest window (median 0.8) than in maize's season (median 7.5).
- **The design differs** in season start, GDD base, Holm family and years. Say "a similar
  design".
- **The wheat pointer:** drop "there the answer is clearer" from it.

**C7. The out-of-sample section leaves out where the model misses.** Medium; verified
(D8).

- **Where:** about nine-tenths of each gap is Romania and Hungary.
- **2022:** its regional predictions explain none of the differences between regions.
- **2015:** "predicted well" holds for the total only.

**C8. The bands are given without scale and over-read.** Medium; verified (I9). Give the
hours. Replace "shows where the harm starts" with "hints that harm begins below 30 °C,
though the pattern is irregular", and note Spain's 54% of the hottest band.

**C9. The registration claim is slightly too strong.** Low; verified against the change
log.

- **What the article says:** "Every rule was written down and published before any maize
  yield was looked at."
- **Two entries came later:**
  - the 2026-10-26 maize-data entry (Q5 with a single year; weather) was written after
    the panel was built;
  - the 2026-10-27 entry was recorded after the estimates.
- **Fix:** say so, as the wheat article does for its two revised rules.

**C10. Licences and attribution.** Low-medium; verified.

- **The notices are complete:** Eurostat, GISCO (non-commercial, on the first page) and
  E-OBS (version, acknowledgement, citation).
- **The README is stale:** its GGCMI row still says "Amendment A1 only" and "winter
  wheat", and the snapshot size is still "108 kB".
- **CC BY 4.0 in both articles:** add a link to the licence and a note that the calendar
  was mapped to E-OBS cells, averaged per region and rounded.
- **A reminder:** a paywalled Medium post would conflict with the E-OBS and GISCO
  non-commercial terms.

**C11. Smaller omissions.** Low.

- **"36 °C on the grid is not 36 °C in a field":** the wheat article says it; the maize
  article should too.
- **Greece's 68 estimated rows** (D4).
- **"Removed":** regional figures that are not measurements are removed only where the
  rule's patterns catch them (D3, D6).
- **The calendar** repeats every year.
- **C1500** includes corn-cob-mix.
- **"Probably too small"** could carry its number: 5.2% of the heat variance survives the
  fixed effects.

**C12. Figures.** Low.

- **Figure 4's "Registered model":** this label clashes with Figure 2's "Wheat,
  registered analysis". Use "Main model".
- **Figure 4** has no caption.
- **Figure 1's orange** marks the two-way method only, while the text's test is both
  methods.
- **Orange** means "maize" in Figure 2 and "excludes zero" in Figure 1.

**C13. Consistency.** Low.

- **"Only narrowly"** describes heat in the wheat article but the placebo in the maize
  article.
- **The wheat pointer** has no link to the maize article and omits the split placebo.
- **"2015 is predicted well"** in the README and ROADMAP should say "the total".

**C14. The claim tests.** Low.

- **Wrong or partial checks:** the "usual swing" test checks the wrong quantity (C2). "The
  effects grow" skips the drought estimate without the south (C3).
- **Untested claims:**
  - "shows where the harm starts";
  - "fits irrigation blunting the damage";
  - the placebo mechanism;
  - "there the answer is clearer".
- **A hand-typed number:** "Fifteen harvests" is written in words, which the digit check
  misses.

## Recommended corrections

**A. Without new analysis.** These can be done now.

- **Article:**
  - **Heat:** retitle and re-hedge it: established by the plan's bar, at the margin, not
    robust. Name Romania and the country deletions.
  - **Scale:** fix "well within the usual swing" and give the bands' hours.
  - **Robustness:** rewrite it with the drought number left out and Romania's weight;
    drop the irrigation explanation or qualify it.
  - **Placebo:** rewrite it (I5, C4).
  - **New bullet:** "Not every country equally".
  - **Comparison:** soften the comparison with wheat and the wheat article's pointer.
  - **Prediction:** say where it misses.
  - **Registration:** add the account of the two change-log entries.
  - **Omissions:** add the smaller ones (C11).
- **Figures:** relabel Figure 4, add its caption, and give the colours one meaning.
- **Records:**
  - **MAIZE.md:** an "After review (post hoc)" section with:
    - the one-way standard errors;
    - leave one country out;
    - Romania 2024;
    - the south/rest split;
    - the placebo's terms;
    - the corrected flag count.

    Label each post hoc, beside the registered verdicts.
  - **METHODS.md:** a change-log entry recording this review.
  - **README and ROADMAP:** made consistent (C10, C13).
- **Tests:** fix the claim tests (C14) and add the new claims.

**B. New analyses.** Each needs a registered amendment before it is computed:

1. **Weather over where maize grows,** from a cropland or maize-area map, with cells below
   800 m as a minimum check (D2).
2. **A wider data rule:** repeated yields, level breaks at recode joins, and code-years that
   equal an aggregate's (D3, D5).
3. **Irrigated and rainfed maize as separate estimands,** or an EDD × south term (I3).
4. **Inference for few effective clusters:** a randomization test, or one-way country
   clustering with a stated df, beside the registered methods (I1, I6).

Doing **A** alone is enough to publish an honest maize article. Drought stands, with one
deletion to mention. Heat is reported as what it is: a result that clears the plan's bar
by a hair and rests on Romania. **B** asks whether heat survives once the measurement problems are fixed. It
would be reported beside the registered results, never in place of them.
