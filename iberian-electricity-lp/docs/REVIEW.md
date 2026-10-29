# Critical review of the Iberian electricity study (2026-10-29)

This review was written after the registered results ([RESULTS.md](RESULTS.md)) and the
generated article ([../publication/article.md](../publication/article.md)), and before
publication.

**How it was done.**

- **Three independent reviewers.** Each read the plan, the code, the data and the outputs
  from one angle: the programmes and the market data; the estimands and the inference; and
  the claims and their communication. All three were read-only and had to give evidence
  for every finding.
- **Independent verification.** Every finding marked **verified** below was recomputed
  with new scripts, from the committed tables and results, from the raw OMIE capacity
  files, or against the EUPHEMIA Public Description (18 December 2025). The one finding
  that could not be checked this way is marked **plausible**.
- **Nothing was changed.** No result, plan, code or text was changed in the course of
  this review. The changes it recommends are listed at the end, for a separate step.

## Bottom line

The code does what [METHODS.md](../METHODS.md) says, and the numbers are right:

- **Parsing.** Four raw days, re-parsed independently with exact decimal arithmetic,
  reproduce every summed step, matched volume, flow and price in the derived tables.
- **The programmes.** A separate algorithm reproduces the dual interval of all 37,374
  programmes, and the accepted-volume range of all 5,118 realised programmes, with no
  mismatch. It uses the slopes of the exact integer optimal-value function.
- **The estimands.** About 60 reported quantities were recomputed without the project's
  code, with no discrepancy: every estimate, interval, count and hourly mean.
- **The article.** Every number in it matches the results.
- **The decisions.** The registered decisions survive other interval methods: BCa,
  jackknife-t, and a bootstrap that resamples days within months. They also survive
  leaving out any one day or month.

But four problems change what the study can claim:

1. **E1, the headline, holds by the clearing algorithm's own rules.** EUPHEMIA must
   accept every in-the-money period order in full and reject every out-of-the-money one,
   and it validates its solutions by complementary slackness with the selection of block
   and complex orders fixed. That is exactly what `LP_R` tests. The 100% result confirms
   that OMIE's public files and this reconstruction agree to the cent. It is not an
   empirical discovery about the price. Yet the title, the summary and the conclusions
   present it as one.
2. **The size of E2 is not a clean cost of non-convexity, and its breakdown by order
   type is partly an artefact.**
   - **Exclusive blocks.** They are relaxed as if every block of a group could be
     accepted, when at most one can be.
   - **Fixed terms.** The relaxation drops the fixed terms of the scalable complex orders
     instead of spreading them.
   - **France flows.** They are held fixed even in the 37% of quarter-hours when the line
     to France had room.

   Each of these makes |δ| larger. The direction of E2 holds in every variant checked.
3. **The midday result in E3 holds as registered, but not for the registered reason, and
   the post-hoc account of it is stated too strongly.** Supply added at zero price cannot
   lower any price below zero, so the floor binds at positive prices too. Compared at the
   same price level, the effect is, if anything, larger at midday. The flatter supply
   curve the plan gave as the reason gets no support.
4. **Several sentences overstate or blur what was measured.** These include the umbrella
   term "indivisible", the attribution of E2 to the minimum income condition, hourly
   intervals called simultaneous, and the missing closest literature.

## What happens to each conclusion

| Claim | Registered verdict | After review |
| --- | --- | --- |
| E1, the price as a dual: 100% of 5,118 zone-quarter-hours | reproduces the clearing | **Holds, as a consistency check** of OMIE's files and the reconstruction, which EUPHEMIA's rules require (finding M1). Its bootstrap interval, [100%, 100%], carries no information (I2). |
| E2, mean δ = −11.75 €/MWh | excludes zero | **Holds in direction.** The size is an upper bound for this kind of relaxation: fixed terms are dropped (M4), flows are fixed where the line had room (M3), and exclusive blocks are double-counted (M2, small here: −11.54 with repeats removed). It is −10.93 in quarter-hours when the line to France was full. |
| E2 by type: scalable complex orders only, −12.39, never up | sensitivity | **The numbers hold**, but "never raised the price" is mechanical (M1), and the article's attribution to the minimum income condition is not established (C5). The breakdown is not additive (I5). |
| E2 by type: blocks only, −1.46 | sensitivity | **Not reliable.** Most of it comes from counting alternative exclusive blocks as additive supply: −0.95 with repeated quantities removed, −0.46 with exclusive blocks not relaxed (M2). |
| E3, mean m = −6.35 €/MWh | reported | **Holds**, with flows and the indivisible orders' acceptance fixed (C4). It is −5.92 when the line was full and −7.06 when it had room (M3). |
| E3, G = +2.19, smaller at midday | smaller at midday | **Holds** as registered, also under BCa, jackknife-t and a within-month bootstrap. The mechanism is the zero floor, not a flatter supply curve (I1). |
| Post hoc: among positive prices, "no midday difference" | descriptive | **Not supported as a like-for-like statement** (I1). The floor binds at positive prices too, and at like price levels the effect leans larger at midday. |
| E4, France flows at zero: +15.88 importing, −7.93 exporting | descriptive | **Holds.** The signs are mechanical, and the means are stable when any one day is left out (I6). |

## Findings

Severity: **high**, could change a stated conclusion; **medium**, weakens a conclusion
or its reading; **low**, a detail.

### The programmes and the market data

**M1. E1 holds by EUPHEMIA's rules, not as an empirical finding.** High; verified.

- **The rules.** EUPHEMIA Public Description, §5.1: "Any order submitted at the time
  resolution of the MTU of the bidding zone that is in-the-money must be fully accepted.
  Any order out-of-the money must be rejected. Orders at-the-money can be either accepted
  (fully or partially) or rejected."
- **The validation.** §8.2 checks "complementary slackness": the order acceptance
  criteria, "no paradoxically accepted/ rejected curve orders". The master problem
  selects the block and complex orders (§7.4), and a price determination sub-problem then
  prices that selection (§7.5).
- **What `LP_R` fixes.** It fixes exactly that selection and the flows. Section 1 of the
  plan already notes that clearing the bids and being an optimal dual are the same
  statement. So E1 could fail only through errors in the data, the parsing or the
  reconstruction. It did not: the published price lies inside the exact interval, with
  no tolerance, in all 5,118 zone-quarter-hours, and the 99% gate on E2-E4 was never at
  risk.
- **Smaller results that are mechanical too.**
  - *"Relaxing [the scalable complex orders] alone never raised the price."* The accepted
    volume of these orders never exceeds their offered volume priced at or below π (0 of
    4,608 quarter-hours), so relaxing them can only add supply at or below the price.
  - *E3 and E4.* The sign of m (never positive) and the signs of E4 follow in the same way.
- **What it changes.**
  - *The title.* "The Electricity Price Is a Shadow Price" states the unconditional
    version. That is the textbook programme of all the bids, which E2 rejects.
  - *The framing.* Bullet 1 of the summary, "the Iberian price is exactly the textbook's
    shadow price" and "not a metaphor here" should say that the algorithm is built to
    satisfy this, and that the data confirm it to the cent.

**M2. Exclusive blocks are relaxed as additive supply.** Medium; verified.

- **The rule.** An exclusive group's accepted ratios sum to at most 1 (EUPHEMIA §5.4.2).
  The curve file carries no group identities, and `experiment.programmes` turns every
  offered exclusive-block step into independent divisible supply.
- **An example.** On 1 October 2025, 20:00 to 20:15 (H21Q1), 750 MW of exclusive blocks is
  offered six times, at 103.98, 104.86, 108.84, 114.59, 121.88 and 124.89 €/MWh, and
  1,500 MW is accepted.
- **How common it is.** Removing repeated quantities within a quarter-hour removes 39% of
  the offered exclusive-block volume. This heuristic is imperfect: in the example above,
  two of the six were accepted. The reported accepted share of exclusive blocks, 4.6%,
  is 7.6% once repeats are removed.
- **Variants, mean δ over the 4,608 Spanish or joint quarter-hours:**

  | Variant | Mean δ (€/MWh) | Quarter-hours down |
  | --- | ---: | ---: |
  | Blocks only, as registered | −1.46 | 716 |
  | Blocks only, exclusive blocks with repeated quantities removed | −0.95 | 631 |
  | Blocks only, exclusive blocks not relaxed | −0.46 | 477 |
  | Every order (`LP_X`), as registered | −11.75 | 4,096 |
  | Every order, exclusive blocks with repeated quantities removed | −11.54 | 4,080 |

- **What it changes.** The blocks-only row is mostly an artefact; the E2 decision is
  unaffected. OMIE's unit-level curves, published after a confidentiality period, could
  identify the groups.

**M3. The France flows are fixed even when the line had room.** Medium; verified.

- **How often.** From `capacidad_inter_pbc` (border 3), the free capacity in the direction
  of the flow was zero in 2,891 of 4,608 quarter-hours. The line had room in 1,717
  (37.3%), where the Iberian and French prices are linked and the fixed flow is a strong
  assumption.
- **The effects, by line state:**

  | Mean | Line full | Line with room |
  | --- | ---: | ---: |
  | δ (every order relaxed) | −10.93 | −13.14 |
  | m (1,000 MW at zero price) | −5.92 | −7.06 |

- **What it changes.** The article attaches the overstatement caveat to the merit-order
  effect only. It applies to δ as well, and the summary's "would have lowered the price by
  11.75 €/MWh" carries no flow caveat. The share of quarter-hours with room should be
  stated.

**M4. The relaxation drops the fixed terms; it does not convexify them.** Medium;
verified (numbers), plausible (consequence).

- **The orders' steps.** `LP_X` keeps the step prices of the scalable complex orders and
  drops each order's fixed term. 21.7% of the offered volume of these orders is priced at
  or below zero, which looks like minimum-load tranches whose cost sits in the fixed term.
- **What was left out.** On average, 2,266 MW per quarter-hour of these orders was priced
  below π and not accepted.
- **The consequence.** A relaxation that spread the fixed term over the volume would
  raise the effective offer prices and give a weakly higher price. So |δ| is an upper
  bound for a convex relaxation. Its size also depends on how bidders split their costs
  between the fixed term and the steps, which the public files do not show.

**M5. Joint and split quarter-hours get different counterfactuals.** Low to medium;
verified.

- **Joint quarter-hours.** The relaxation, the extra supply and the France counterfactual
  act on the whole Iberian market, with the line between Spain and Portugal
  unconstrained, because the joint curve has no rows per zone.
- **Split quarter-hours.** Only Spain changes; Portugal's orders and the flow between them
  stay fixed.
- **The difference.** Mean δ is −12.37 €/MWh in joint quarter-hours and −6.76 in Spanish
  split ones. The pooled mean mixes the two, which the article does not say.

**M6. Smaller points.** Low; verified.

- **What the HiGHS check covers.** It solves the same `Steps` object as the exact method
  (`programme.solve`), so it checks the arithmetic, not the model. The independent
  recomputation above covers the rest.
- **Mixed quarter-hours.** METHODS §4 says mixed quarter-hours are reported and left out,
  but `experiment.run` solves every row. There are none, so nothing is affected.
- **Clock changes.** No day with a clock change was drawn, so the code path for 92 or 100
  periods is untested on real data.

### The estimands and the inference

**I1. The zero floor binds at positive prices too, so the post-hoc split does not compare
like with like.** Medium; verified.

- **The floor.** A sell step at zero price cannot lower any price below zero, so |m| ≤ π
  in every quarter-hour: no violation in 4,069 priced above zero.
- **Where it binds.** The floor binds exactly (the new price is 0) in 36 of 894 positive
  midday quarter-hours and 23 of 3,175 others. Positive midday prices are much lower,
  with a median of 34.89 €/MWh against 87.79.
- **Mean |m| by price band:**

  | π (€/MWh) | Midday | Other hours |
  | --- | ---: | ---: |
  | (0, 10] | 0.93 | 1.30 |
  | (10, 20] | 7.99 | 5.46 |
  | (20, 40] | 14.38 | 5.97 |
  | (40, 60] | 11.76 | 8.39 |
  | (60, 80] | 8.60 | 7.99 |
  | (80, 120] | 6.33 | 6.71 |
  | above 120 | 7.67 | 9.54 |

- **Standardised over these bands,** G among positive prices is −1.04 €/MWh, with a 95%
  interval of −2.81 to +0.23 from the first 2,000 registered draws. The reviewer's
  weighting gives −1.68 (−3.14 to −0.30). In relative terms, mean |m|/π is 0.32 at midday
  and 0.14 at other hours. These cuts are exploratory, and their sign depends on the cut.
- **What holds.** The decomposition of the registered G is correct:
  G = p_mid·(a_out − a_mid) + (p_out − p_mid)·a_out = −0.02 + 2.21. At the point estimates,
  G comes from the share of quarter-hours at or below zero, where m = 0 in all 539.
- **What it changes.**
  - *The post-hoc wording.* "Among prices above zero, no midday difference was
    established" should not read as a test of the supply curve's slope; the comparison is
    confounded by the same floor.
  - *The registered reason.* The flatter supply curve gets no support.
  - *RESULTS.md.* "E3, G > 0: borne out" holds for the sign only.

**I2. E1's interval is degenerate.** Medium; verified.

- **The interval.** A percentile bootstrap of an indicator that is 1 in every
  zone-quarter-hour is always [100%, 100%]. The same holds for the tolerance sensitivity
  analysis and the volume check (RESULTS.md).
- **What 48 days can show.** If days were independent, no failing day out of 48 is
  consistent, at 95%, with up to 6.1% of days having some failure.
- **What it changes.** RESULTS.md should say why the interval is degenerate, or drop it.

**I3. The hourly intervals are not simultaneous.** Low; verified. Figure 3's caption
calls them "simultaneous". They are 24 intervals at 97.5% each, Bonferroni over Q2 and
Q3, not over the hours.

**I4. The registered bootstrap ignores the stratification by month.** Low; verified;
conservative. Resampling days within months gives narrower intervals: 97.5% for mean δ
of −12.80 to −10.70 (registered −13.12 to −10.33), and for G of +0.75 to +3.54 (registered
+0.37 to +3.90). No decision changes.

**I5. The breakdown of E2 by order type is not additive.** Low; verified. Scalable complex
orders alone give −12.39, more than every order together (−11.75); blocks alone give
−1.46. "Almost all of that comes from one kind of order" should read "relaxing the
scalable complex orders alone moves the price as much as relaxing every order".

**I6. E4 has moderate tails but is stable.** Low; verified.

- **Importing quarter-hours.** Mean +15.88 €/MWh against a median of +11.60.
- **Exporting quarter-hours.** Mean −7.93 against a median of −4.51.
- **Leaving out one day.** The means move within [15.16, 16.18] and [−8.17, −7.35].
- **Coverage.** No quarter-hour both imports and exports, and none does neither.

### The claims and their communication

**C1. The title and the headline.** High; verified. See M1. In addition, "so the textbook
can be put to a test" (the opening) sets up as a test what the algorithm guarantees.

**C2. The post-hoc midday explanation is stated as settled, and in the summary without a
label.** Medium; verified.

- **The summary.** Its third bullet reads "The effect is smaller around midday because
  33% of midday prices were already at or below zero", with no post-hoc label. The body
  has one.
- **"Not the one I gave."** "But the reason is not the one I gave" is only half right.
  The plan's reason was that solar output keeps the price low and the supply curve flat.
  The low-price half is what the data show; the flat-curve half gets no support (I1).

**C3. "The non-convex part is where the price is made" overclaims.** Medium; verified. The
mean published price over the 4,608 Spanish or joint quarter-hours is 70.33 €/MWh. The
mean δ, −11.75, is about a sixth of it, and an upper bound (M4).

**C4. The held-fixed caveats are incomplete.** Medium; verified.

- **E2.** The flows stay fixed in `LP_X` too, so the overstatement caveat applies to δ
  (M3).
- **E3 and E4.** These use `LP_R`, so they also hold the indivisible orders' accepted
  volume fixed, but the text says only "every bid and flow held fixed". With the price
  6 €/MWh lower, some accepted orders with a minimum income condition would no longer
  qualify. That would push the price back up, so m is overstated. For E4, a higher price
  would let more such orders in, so f is also overstated in size.

**C5. E2 is attributed to the minimum income condition without evidence.** Medium;
verified.

- **What a relaxation removes.** Relaxing the scalable complex orders also removes their
  load gradients, minimum acceptance power and scheduled stops (EUPHEMIA §5.3). Load
  gradients can also keep steps priced below the price out.
- **What the files show.** The curve file is anonymous and aggregated, with no order
  identities, fixed terms or gradients, so the study cannot say which condition rejected
  those steps.
- **The umbrella term.** "Indivisible" (C7) adds to the confusion: these orders are
  scalable, all-or-nothing only in whether the order is activated.

**C6. The closest literature is missing.** Medium; verified, except as noted.

- **O'Neill and coauthors (2005).** E1's construction is theirs: fix the integer
  decisions, then price at the dual of the remaining linear programme. O'Neill, R. P.,
  P. M. Sotkiewicz, B. F. Hobbs, M. H. Rothkopf and W. R. Stewart, "Efficient
  market-clearing prices in markets with nonconvexities", European Journal of Operational
  Research 164(1), 269-285, doi:10.1016/j.ejor.2003.12.011.
- **Convex-hull pricing.** The relaxed price belongs to this literature (Gribik, Hogan and
  Pope, 2007, a working paper; plausible, not checked).
- **Merit-order studies.** The advice to "anyone who estimates merit-order effects from
  bid curves" cites no such study. On OMIE's curves there is Ciarreta, A., M. P. Espinosa
  and C. Pizarro-Irizar, "Is green energy expensive? Empirical evidence from the Spanish
  electricity market", Energy Policy 69 (2014), 205-215, doi:10.1016/j.enpol.2014.02.025.

**C7. "Indivisible orders" is the wrong umbrella term.** Low. The scalable complex orders,
22.8% of offered sell volume and the source of E2, are divisible above a minimum power.
METHODS uses "non-convex".

**C8. Not every split quarter-hour had two prices.** Low; verified. "Then the market splits
into two zones with two prices": the prices differed in 495 of the 510 split
quarter-hours.

**C9. The order types are described loosely.** Low; verified.

- **Exclusive blocks.** "At most one is accepted" holds only for blocks that must be
  accepted in full; in general, the accepted ratios sum to at most 1 (EUPHEMIA §5.4.2).
- **Blocks.** A block's volume can differ between quarter-hours (§5.4).

**C10. Two statements about the interval of shadow prices are imprecise.** Low; verified.

- **Its ends.** "Between the dearest accepted step and the cheapest rejected one"
  describes sell steps only. The ends also depend on buy steps.
- **"No step was partly accepted, so the shadow prices formed an interval."** This
  reverses the implication. What was measured, an interval wider than a point, implies
  that no step was partly accepted, not the other way round.

**C11. Smaller wording points.** Low.

- **The sign of the difference.** "7.21 €/MWh at midday and 7.18 at other hours, a
  difference of −0.03": in that order, the reader computes +0.03. Give the other hours
  first, and say "other hours minus midday".
- **Figure 1's caption.** It should say that the average is the overall mean δ, and that
  only joint quarter-hours were candidates.
- **The failed E2 expectation.** It had already been seen on the development day outside
  the sample (change log, 2026-10-29). The article should say so where it reports the
  failure.
- **"OMIE publishes every bid".** It publishes every bid step, without the orders'
  conditions.
- **The median of 2,951 steps.** It is per Spanish or joint zone-quarter-hour.
- **"The two questions that decide something."** E1 also has a decision rule.

## What holds up

- **Parsing.** It is exact on four raw days re-parsed independently. Every line has ten
  fields, there are no negative quantities, and the price order is Portugal then Spain,
  as the specification says.
- **Flows by side.** Joint rows carry no Spain-Portugal flows, and Portuguese rows no
  France flows. The labels that vary between days are handled correctly.
- **The exact dual intervals.** They are correct in every programme. No programme had an
  unbounded interval or a zero-quantity step, and published prices ranged from −5 to
  300 €/MWh, far from the price limits, so no curtailment applied.
- **The estimands.** They are computed as registered: the point rule, the zone, hours
  11-17, the seed and draws, linear quantiles and ratio-of-sums means.
- **The decisions.** E2 and E3 hold under BCa, jackknife-t and a within-month bootstrap.
  Only 0.26% of resamples have G ≤ 0. Leaving out one month moves G within [1.86, 2.95]
  and mean δ within [−12.24, −11.36].
- **The point rule.** It makes no material difference to E2 or E4.
- **The article's numbers, figures and citations.** Every number matches. The figures
  show what their captions say, apart from C11 and I3. The three cited references and
  the EUPHEMIA document exist as cited, and OMIE's terms are paraphrased correctly.
- **Dates and facts.** The 15-minute day-ahead market began with delivery on 1 October
  2025, and the gate closes at noon, Spanish time. The descriptions of the minimum income
  condition and of how blocks are judged match EUPHEMIA.

## What should change before publication

None of this was done in this review. Each change belongs in a corrections step, with
its own entry in the change log.

- **E1.**
  - Retitle the article around what the study found, not what the algorithm guarantees.
  - Present E1 as a check that the published files and the reconstruction agree to the
    cent, citing EUPHEMIA §5.1 and §8.2 and O'Neill and coauthors (2005).
  - In RESULTS.md, explain the degenerate interval.
- **E2.**
  - State that its size is an upper bound for a relaxation that drops the fixed terms,
    with flows fixed although the line to France had room in 37% of quarter-hours.
  - Report the estimate in quarter-hours when the line was full.
  - Correct or drop the blocks-only row (exclusive groups).
  - Stop attributing the effect to the minimum income condition alone.
  - Describe the breakdown by type as not additive, and say that joint and split
    quarter-hours differ.
- **E3.**
  - Say that the zero floor binds at every price and that the flatter-curve reason gets
    no support.
  - Reword or remove "no midday difference" among positive prices.
  - Label the post-hoc account in the summary.
  - Add the overstatement from holding the indivisible orders' acceptance fixed.
- **Wording.** Fix C7 to C11, I3 and I5, and cite a merit-order study on OMIE's curves.
