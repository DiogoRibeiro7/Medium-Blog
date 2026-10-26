# Critical review of the wheat study (2026-10-26)

This review was written after the registered results ([RESULTS.md](RESULTS.md)) and the
draft article ([../publication/article.md](../publication/article.md)), and before
publication.

**How it was done.**

- **Three independent reviewers.** Each read the plan, the code, the data and the
  outputs from one angle: econometrics and inference, data and measurement, and claims
  and communication. All three were read-only and asked for evidence for every finding.
- **Independent verification.** Every finding listed below as **verified** was
  recomputed independently, with new scripts, from the committed outputs, the crop
  snapshot or the raw E-OBS files. The rest are marked **plausible**.
- **Nothing was changed.** No result, plan or text was changed in the course of this
  review.

## Bottom line

The code does what [METHODS.md](../METHODS.md) says, which a line-by-line audit confirmed:

- the units, signs and E3 indicator;
- the null imposed in the bootstrap, the degrees of freedom and the Holm adjustment;
- the delta-method gradient;
- the degree-day, time-in-bin and potential-evapotranspiration formulas.

Every number in the article matches the results.

But four problems change what the study can claim:

1. **The registered rule for E2 does not test E2.**
   - **What it tests:** it takes significance from the joint test of the two
     precipitation terms. That joint test rejects mostly because of the curvature and
     the wet end.
   - **The contrast tested directly:** the dry-season contrast itself gives two-way
     p = 0.130 and wild bootstrap p = 0.160, or 0.389 after Holm. On the evidence, the
     drought effect is not determined. The "established negative" verdict should not be
     reported as a finding.
2. **The window counts July heat after the harvest.**
   - **Where the heat falls:** 71% of the March–July heat above 30 °C falls in July, as
     do 79% of the time at 36 °C or above.
   - **Who supplies it:** a third of that time is July heat in regions south of 40°N,
     where wheat is normally harvested by late June.
   - **Consequence:** the article's "harmful only at the very top" rests on those days,
     on Spain, and on a two-way standard error smaller than every one-way standard error.
     It is not supported.
3. **Some yield series are not regional measurements, and no rule catches them.** These
   are pro-rated, imputed, carried-forward or likely mis-keyed series (finding D1).
   Dropping them, as an exploration, changes no verdict.
4. **The article's explanation of 2003 is contradicted by the data.** Much of the miss
   comes from Romania and Hungary, where 2003 was not hot by the study's measure. In
   Romania's three worst regions the harvested area collapsed and recovered the next
   year, which points to damage before March.

## What happens to each conclusion

| Claim | Registered verdict | After review |
| --- | --- | --- |
| E5, placebo | passes | **Holds**, as "no detectable link". The window is not entirely after harvest in the north (finding C7). |
| E1, heat (−2.6% per 10 degree days above 30 °C) | not determined | **Holds.** The size is fragile: without Spain it is −1.6% (SE 2.5%). It is probably attenuated by measurement error and by post-harvest July heat. |
| E2, drought | established negative | **Not supported as stated.** The direct test of the contrast is not significant under either method. |
| E3, change since 2012 | not determined | **Holds.** The detectable change is understated: about 7.4%, not 6.2%, at the Holm level. |
| M3, "only days at 36 °C and above" | descriptive | **Not supported** (finding E2). |
| Precipitation optimum, about 350 mm | reported | **Uncertain:** 95% interval [274, 419] mm, which includes the median of 281 mm. |
| E4, 2018 predicted well; 2003 missed | descriptive | **Holds**, but the 2003 explanation is wrong and "EU yield" means the panel regions only. |

## Findings

Severity: **high**, could change a stated conclusion; **medium**, weakens a conclusion
or its reading; **low**, a detail.

### Econometrics and inference

**E1. The E2 rule tests the joint precipitation terms, not the dry-season contrast.**
High; verified.

- **The contrast, tested directly.** Reparametrised so that the contrast is a single
  coefficient, it is −0.0231. Two-way: t = −1.58, p = 0.130. Wild bootstrap: CR1 t = −1.64,
  p = 0.160. Holm with E1 and E3 gives 0.389.
- **What drives the joint test.** The precipitation band excludes zero only at 80 mm
  (the 1st percentile) and from 530 mm upward. With country-only errors (the bootstrap's
  statistic) the joint Wald statistic is 67.6, against 17.2 two-way.

**E2. "Only days at 36 °C and above clearly lower yields" does not hold up.** High for
the article; verified.

- **One country carries it.** Spain supplies 65% of the bin's identifying variation.
  Without Spain the estimate is −0.025 with SE 0.042.
- **The narrow interval comes from the variance estimator.** The two-way SE (0.0090) is
  smaller than every one-way SE: country 0.0114, year 0.0173, country-year 0.0194. The
  raw two-way matrix had 5 negative eigenvalues, which were set to zero.
- **The 27–30 °C bin also rejects** under the wild bootstrap (p = 0.029).
- **No multiplicity adjustment** was applied across the 13 bins.

**E3. Few effective clusters.** Medium; verified.

- Of the identifying variation in degree days above 30 °C, Spain supplies 40%, Italy 22%
  and Romania 15%.
- E1 without Spain is −0.016 (SE 0.025); without Italy it is −0.043 (SE 0.008).
- The t(21) reference treats 22 clusters as equal.

**E4. The wild bootstrap ignores dependence across countries within a year.** Medium;
verified.

- The year-clustered SEs exceed the country-year SEs for every weather term (precipitation:
  0.0285 against 0.0244). Countries are therefore positively correlated within a year,
  which a country-only bootstrap treats as zero.
- The two methods err in different directions, so "holds under both" is weaker than it
  sounds.

**E5. Little identifying variation is left, so measurement error attenuates the
estimates.** Medium; shares verified, size of the bias plausible.

- **How much is left.** After region and country-year effects, 0.8% of the variance of
  growing degree days remains, 7.4% for degree days above 30 °C and 13.5% for
  precipitation.
- **Why that matters.** E-OBS interpolation smooths exactly those within-country
  contrasts, so interpolation noise is a larger share of what is left. The estimates
  are biased toward zero, which also pushes verdicts toward "not determined".

**E6. The precipitation optimum is not established.** Medium; verified. The delta-method
95% interval is [274, 419] mm. It includes the median rainfall, so "the typical region
gets less than the optimum" is not supported.

**E7. The water-balance variant partly measures heat.** Medium-low; reported by the
reviewer.

- Hargreaves evapotranspiration depends on temperature. After the fixed effects, the
  water balance squared correlates +0.34 with heat above 30 °C.
- E1 falls to −0.8% under this variant. The article cites the variant's drought effect as
  support.

**E8. E3's detectable change is understated.** Low; reported by the reviewer. At the Holm
level (α/3) it is 0.071 log points per 10 degree days, not 0.061. The bootstrap is less
powerful still.

**E9. No variant adds region trends while keeping country-year effects.** Low-medium;
verified.

- With region trends added, E1 is −3.4%: two-way p = 0.014, bootstrap p = 0.140.
- The verdict is unchanged, but E1's two-way significance depends on an unregistered
  trend choice.

**E10. E4's "normal" uses the square of mean precipitation, not mean squared
precipitation.** Low; reported by the reviewer. This follows the registered wording
("the effect at its mean weather"). It shifts both the observed and the predicted EU
change by about one point.

**E11. The sensitivity rule was tightened in the report without a change-log entry.**
Low; verified.

- The plan: a variant changes a decision if it reverses the sign. The report also
  requires the interval to exclude zero.
- No point estimate reverses, so nothing depends on this.

### Data and measurement

**D1. Some yield series are not regional measurements.** Medium; verified.

| Region(s) | Years | Problem |
| --- | --- | --- |
| Greece, 6 regions | 2000–2008 | Pro-rated from one series: the cross-region spread of growth is 0.003–0.013, against 0.058–0.255 in 2016–2024. Greece also has 54 rows flagged estimated, and conflicting old- and new-code series for 2009–2014. |
| Netherlands, 11 regions | 2005 | All 8.39–8.40 t/ha: the national yield given to every region. |
| DK04 | 2006–2009 | 10.6–12.5 t/ha, against 6.4 from 2010. The area is understated: regional production sums to the national figure, but area does not. |
| NL22, NL23 | 2004 | 13.73 and 5.13 t/ha; their production looks swapped. |
| ITF2 | 2012–2015 | About 1.0 t/ha, against 3 or more before and after. |
| 12 region-year pairs | various | Identical area and production in consecutive years, all unflagged; 11 of them in Italy. |

Dropping these 86 rows (exploratory, two-way only) moves E1 from −0.0268 (p = 0.130) to
−0.0304 (p = 0.067) and changes no verdict. The reviewer reported these numbers; the
series themselves were verified.

**D2. The 1 March–31 July window counts post-harvest July heat.** High for the heat
narrative; verified.

- **When the heat falls.** By month, recomputed from the raw E-OBS grids:
  - 71.1% of the window's heat above 30 °C falls in July, and 26.9% in June;
  - 78.7% of the time at 36 °C or above falls in July;
  - 33.9% of all that time is July time in the 10 panel regions south of 40°N, where
    wheat is normally harvested by late June.
- **Exploratory evidence from the reviewer.** Splitting the ≥36 °C bin, the
  southern-July part carries the effect (−0.051 per day, p < 0.001), while the rest is
  near zero (−0.011, p = 0.75). Post-harvest heat cannot lower yields, so this points to
  confounding with a hot, dry season, not to heat damage.
- **Not covered by any registered check.** The E5 placebo covers August–September only,
  and no registered sensitivity moves the window's end.

**D3. The 2003 miss is not a heatwave story.** Medium; verified.

- **Romania and Hungary** carry much of the gap. In Romania's three worst regions (RO21,
  RO31, RO32), observed −65% to −71% against predicted about 0, the 2003 heat was below
  each region's own average.
- **Harvested area** collapsed in 2003 and recovered in 2004: RO22 476 → 258 → 307
  thousand ha; RO31 655 → 436 → 594; RO21 221 → 154 → 216. That fits damage before
  1 March, such as winter kill, which the window cannot see. The cause itself is
  plausible, not verified.
- **France** is the largest single contributor (−2.0 points of a −7.4-point gap). It
  was hot in 2003: 7.1 degree days above 30 °C, against 3.9 in other years.

**D4. The 0.5 t/ha floor cuts real failures and passes clear errors.** Low-medium;
verified.

- It excludes RO22 2003 (0.49 t/ha), PT15 2005 and ITG1 2007, which are plausible crop
  failures.
- It passes 13.73, 12.5 and about 1.0 t/ha, which are errors.

**D5. E4's "EU yield" covers the panel regions only, with changing coverage.** Low;
partly verified. The panel holds 39–42% of Germany's wheat area in the test years, and
Denmark is absent in 2003.

**D6. E-OBS extremes and station density.** Medium for interpretation; plausible.

- The ensemble mean smooths maximum temperatures, so "36 °C" on the grid is not 36 °C in
  a field.
- Changing station networks could load onto E3.
- The cell mask is constant over 2000–2024 (verified by the reviewer).

**D7. The below-800 m variant drops much of the Spanish plateau.** Low; reported by the
reviewer.

- Only 28% of ES41's weight lies below 800 m.
- Island regions are missing under the all-cells weighting, because cells E-OBS treats
  as sea count in the coverage denominator. This causes the 49 weather exclusions (FI20,
  ES53, EL42).

### Claims and communication

**C1. "−3.5% per day" at 36 °C and above.** High; verified.

- **The unit.** The bins measure time as fractions of 24 hours, so "a day" means 24 hours
  above 36 °C, which almost never happens in the EU. The article's "0.16 days" is
  3.8 hours per season.
- **How it will be read.** Readers will take "a day at 36 °C" to mean a day reaching
  36 °C. That overstates the effect of such a day by about 4–5 times.
- **Fix.** Use hours (about −0.15% per hour), in both the text and Figure 1.

**C2. "Heat: harmful only at the very top" and "the shape of the response explains
why".** High.

- M3 was registered as deciding nothing, and it had neither Holm nor the bootstrap.
- Findings E2 and D2 undercut it.

**C3. "Counts as an established negative effect" for drought.** High; see finding E1.
The interval includes gains, so the sign is not pinned down either, not just the size.

**C4. The 2003 paragraph.** High; see finding D3.

**C5. The robustness paragraph is selective.**

- "Heat stays harmful in every one" counts the water-balance variant (−0.8%, interval
  −4.5% to +3.1%).
- The example given is the strongest variant.
- The drought effect's weakening (−0.7% with an April start, −1.0% with area weights) is
  not mentioned.

**C6. The causal reading needs its caveats.** Medium.

- "Can be read as a cause" holds only for the weather as a bundle, including sun, dry
  air and the timing of the weather.
- The plan's "the models do not separate the channels" is missing from the article's
  limits.
- So is the attenuation from measurement error (finding E5).

**C7. The placebo is not entirely after harvest.** Medium. In the north, August is
harvest time, and those regions are about 35% of the panel. The result should read "no
detectable link".

**C8. Figure 3 clips 11 regions.** Medium; verified.

- 2003: RO21, RO31 and RO32, at −65% to −71%.
- 2018: ES43 +82%, ES61 +50%, ES52 +46%, RO22 +46%, ITF2 +39% and ES30 +32%.
- 2022: ES62 +45% and ITF2 +32%.

**C9. The precipitation figure invites a comparison between regions.** Medium-low. The
curve shows within-region swings under an assumed quadratic, and the peak has no
interval (finding E6).

**C10. The licence statements understate the conditions.** Medium; verified against the
source pages.

- **GISCO** grants use "on condition that the data will not be used for commercial
  purposes". The copyright notice must also appear on the publication's introductory
  page; the README showed only the attribution.
- **E-OBS** asks that the dataset version be given; the article says only "E-OBS".

**C11. Smaller points.**

- **Scale.** "Ten more degree days" needs a scale: a typical region gets 2–6 in a
  season.
- **Labels.**
  - "EU yield" should say "the study's regions".
  - Figure 4's heat panel title is wrong for the 27 °C and 33 °C rows.
- **Repository link.** The article never links the repository.
- **The registration account.** "Published before I fitted any model" is accurate, but
  it should mention the two data rules revised before any estimate.
- **README, ROADMAP and article.** They differ slightly on the detectable change and the
  placebo wording.
- **ROADMAP overstatements.**
  - It says results regenerate "exactly"; the test uses a 1e-7 tolerance.
  - It says "every qualitative claim" is tested; 11 are.

## Recommended corrections

**A. Without new analysis** (text, figures, licences). These can be done now.

- **Article:**
  - **Drought:** report the dry-season effect as not established, and give the direct
    test as a post-hoc check of the flawed rule.
  - **Heat:** retitle the section; report time at 36 °C or above in hours, with its
    caveats (Spain, one-way errors, the July timing); drop "explains why".
  - **2003:** rewrite the paragraph.
  - **Figures:** fix Figure 3's axes, Figures 1 and 4's units and titles, and the
    precipitation caption.
  - **Wording:** "the study's regions"; the robustness paragraph; "no detectable link";
    the causal and measurement caveats; the repository link; the two revised data rules.
- **Licences:**
  - the GISCO non-commercial condition, in the README and ROADMAP;
  - the copyright notice early in the article;
  - "E-OBS v33.0e" in the article.
- **Records:**
  - **RESULTS.md:** a section "After review" with the direct E2 test, labelled post hoc
    beside the registered verdict.
  - **METHODS.md:** a change-log entry recording this review and the E2 rule's flaw,
    after results.
  - **README and ROADMAP:** made consistent.

**B. New analyses.** Each needs a registered amendment before it is computed, because
each could change E1 or E2:

1. **A window that ends at harvest,** from published regional crop calendars, so that
   post-harvest heat no longer counts as growing-season heat (finding D2).
2. **A data-quality rule,** fixed in advance, for series that are not regional
   measurements: identical regional yields in a year, identical consecutive rows, and
   production that sums to the national total while area does not (finding D1).
3. **Inference that allows for few effective clusters and cross-country dependence
   within a year,** for example a wild bootstrap over both countries and years
   (findings E3 and E4).

Doing **A** alone is enough to publish an honest article: the claims shrink to what the
registered results support. **B** asks whether the heat and drought results change
once the main measurement and data problems are fixed. It would be reported beside the
registered results, never in place of them.
