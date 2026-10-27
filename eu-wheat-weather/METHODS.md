# Heat, drought and wheat yields in EU regions: research plan

Committed and pushed with the data pipeline, before any yield was related to any weather
variable. Every later change goes in the change log at the end, with its timing relative
to the results and its reason.

## 1. Question

How sensitive are common-wheat yields in the regions of the European Union to extreme
heat and to drought during the growing season, and has that sensitivity changed since
2000?

Weather in a given region and year is close to random once the region's normal climate
and the year's national conditions are accounted for. A weather coefficient estimated
that way is therefore interpretable as the effect of a weather shock on yields, given
farmers' choices for that region. That is unlike an Okun coefficient. It is not the
effect of a permanent change in climate, which farmers would adapt to.

## 2. What was seen before this plan was written

- **Eurostat.** The coverage of `apro_cpshr` for common wheat (`C1110`) at NUTS-2:
  - which regions and years have area and harvested production;
  - which NUTS version each region code belongs to.

  Area and production are published at NUTS-2; yield is not. Between 187 and 207 EU-27
  regions have both in each year from 2010, and between 152 and 199 before. 150 of the
  226 regions with NUTS-2021 codes cover at least 20 years. Some regions appear only under codes of
  earlier NUTS versions, and Portugal's NUTS-2024 regions appear only from 2023.
  Germany reports several regions only at NUTS-1.
- **E-OBS.** The availability and size of the v33.0e files (1950-2025).
- **GISCO.** The availability of the NUTS boundaries of 2006 to 2024.
- **Not looked at.** No yield level, trend or anomaly, and no weather value.

## 3. Data

### 3.1 Yields

- **Source.** Eurostat `apro_cpshr`, common wheat and spelt (`C1110`).
  - Area: `AR_THS_HA`, thousand hectares.
  - Harvested production at EU standard humidity: `HPRD_HUMD_EU_THS_T`, thousand
    tonnes.
- **Yield.** Production divided by area, in tonnes per hectare.
- **Snapshot.** Frozen with its SHA-256 hash, flags kept.
- **Harvest years.** 2000-2024.

**Regions.** The EU-27 at NUTS 2021.

- A series published under another code joins a NUTS-2021 region only if the two
  regions' GISCO boundaries (1:3 million) coincide: their symmetric difference is below
  1% of the NUTS-2021 region's area. Other codes are dropped.
- If a year has values under two coinciding codes, the NUTS-2021 code's value is used,
  and differences above 1% are reported.

**Sample rules.**

- A region-year enters if:
  - area is at least 1 thousand hectares;
  - production is present;
  - yield lies between 0.5 and 15 t/ha.

  Values outside that range are reported as implausible.
- A region enters if it has at least 10 such years.
- Eurostat flags (estimated, provisional) are kept and reported. No value is dropped for
  its flag.

### 3.2 Weather

- **Source.** E-OBS v33.0e (Cornes et al. 2018): the ensemble mean on the 0.25° regular
  grid.
  - Variables: daily maximum temperature (`tx`), minimum temperature (`tn`) and
    precipitation (`rr`).
  - The E-OBS elevation grid is used for one sensitivity.
- **From cells to regions.** Each grid cell is weighted by the area of its intersection
  with the region's NUTS-2021 boundary, times the cosine of its latitude.
  - Nonlinear quantities (degree days, bin exposure) are computed per cell and day, and
    only then averaged over the region.
  - Missing cells are dropped and the weights renormalised. A region-day is missing if
    less than half of the region's weight is observed.
  - A region-season is missing if any of its days is missing.
- **Growing season.** 1 March to 31 July of the harvest year. In most of the EU this
  covers stem elongation, flowering and grain filling of winter wheat, and the season
  of spring wheat.
- **Variables.** Degree days use single-sine interpolation between `tn` and `tx`
  (Allen 1976).
  - `GDD`: degree days between 0 °C and 30 °C, in hundreds.
  - `EDD`: degree days above 30 °C (extreme heat), in tens.
  - `P`: precipitation total, in hundreds of mm, and its square.

## 4. Models

**M1 (primary).** For region `i` in country `c` and harvest year `t`:

    log y_it = a_i + d_ct + b1 GDD_it + b2 EDD_it + g1 P_it + g2 P_it^2 + e_it

`a_i` are region effects and `d_ct` are country-by-year effects. The country-year effects
absorb national prices, policies, varieties and reporting changes. Identification
therefore comes from differences between the regions of one country in the same year,
relative to each region's own normal. Countries with a single region in the sample
contribute nothing to M1.

**M2 (prediction).** Region effects and region-specific linear trends, with the same
weather terms and no year effects. M2 can attribute national shocks to weather, so it is
the model for the out-of-sample check (E4).

**M3 (shape, descriptive).** M1 with `GDD` and `EDD` replaced by the days spent in 3 °C
bins of temperature, computed per cell with the sine interpolation. The bins are
`< 0`, `0-3`, ..., `33-36` and `>= 36` °C, with `15-18` °C as the omitted bin. It shows
where yields start to fall with heat; it decides nothing.

**Inference.** Two methods are used for every decision.

- Standard errors clustered two ways, by country and by year, with the small-sample
  correction of Cameron, Gelbach and Miller (2011).
- A wild cluster bootstrap over countries (Rademacher weights, 9999 draws, seed
  20261101), which imposes the null.

## 5. Estimands and decision rules

The three primary tests below form one family, adjusted with Holm at 5%. A result is
**established** only if it holds under both inference methods.

**E1, extreme heat.** `b2` in M1: the change in log yield for 10 more degree days above
30 °C, holding the rest of the season fixed.

- Established negative if `b2 < 0` and the test rejects.
- Otherwise not determined. A positive and rejected `b2` is reported as contrary to the
  agronomic expectation.

**E2, drought.** The effect, in M1, of a dry season on log yield: the region's
10th-percentile precipitation against its median, at the region's own `GDD` and `EDD`.

- Established negative if the joint test of `g1` and `g2` rejects and the effect is
  negative.
- The precipitation that maximises yield, `-g1 / (2 g2)`, is reported when `g2 < 0`.

**E3, change in heat sensitivity.** M1 with `EDD` interacted with the period: 2012-2024
against 2000-2011.

- Established if the interaction test rejects. Its sign says whether sensitivity rose or
  fell.
- Otherwise not determined, and the report gives the smallest change the design could
  detect with 80% power.

**E4, out of sample (descriptive).** For 2003, 2018 and 2022, three years with
widespread heat or drought, M2 is fitted without that year and used to predict it.
Reported:

- the correlation between predicted and observed yield anomalies across regions;
- the share of their variance explained;
- the predicted and observed change of the area-weighted EU yield.

**E5, placebo.** Degree days above 30 °C and precipitation from 1 August to
30 September, after harvest, are added to M1. If they are significant under both
methods, the identification is in doubt, and the report says so before any other
result.

## 6. Sensitivity analyses (fixed in advance)

They are reported for E1 to E3. They do not change a decision unless they reverse its
sign.

- **Thresholds and season.**
  - `EDD` thresholds of 27 °C and 33 °C.
  - A season of 1 April to 31 July.
- **Model.** M2's region trends with year effects in place of country-by-year effects.
- **Weather and weights.**
  - Weather averaged over cells below 800 m only.
  - Regressions weighted by wheat area.
- **Sample.**
  - Regions with at least 20 years only.
  - Without joined series: NUTS-2021 codes only.
- **Drought measure.** The climatic water balance (precipitation minus Hargreaves
  potential evapotranspiration over the season) in place of precipitation.

## 7. What this study does not claim

- **Climate change.** A weather coefficient is a short-run response. Adaptation to a
  changed climate can make long-run effects smaller.
- **Causes.** Yields respond to weather through many channels at once: heat stress,
  water stress, diseases and the timing of operations. The models do not separate them.
- **Measurement.** E-OBS is interpolated from stations, and its accuracy varies with
  station density. A regional average is a coarse proxy for the weather over the wheat
  fields of a region.

## 8. Amendment A1 (2026-10-26, after the results)

Registered after the results and after the critical review
([docs/REVIEW.md](docs/REVIEW.md)), and before anything below was computed. It answers
the review's three main problems:

- heat after the harvest counts as growing-season heat (finding D2);
- some yield series are not regional measurements (D1);
- inference with few effective clusters and dependence across countries within a year
  (E3, E4).

It also tests E2 directly (E1). Its results are reported beside the registered results
of sections 4-6 and never replace them. Where the verdicts differ, both are reported,
the registered one first.

**What was seen before it was written.**

- **Results:** every registered result, and every post-hoc check of the change log of
  2026-10-26.
- **The review's explorations:**
  - dropping 86 suspect rows (two-way only), which moved E1 to -0.0304 (p = 0.067);
  - splitting the 36 °C bin by July in the south.
- **The crop calendar:** its distribution of maturity days over Europe. No yield was
  looked at with it.

### 8.1 A window that ends at harvest

- **Source.** The GGCMI Phase 3 crop calendar, v1.01 (Jägermeyr et al. 2021, Nature
  Food; Zenodo, doi:10.5281/zenodo.5062513; CC BY 4.0).
  - File: `wwh_rf_ggcmi_crop_calendar_phase3_v1.01.nc4`, rainfed winter wheat.
  - Variable: `maturity_day`, a day of the year on a 0.5° grid.
  - Frozen with its SHA-256 hash in `snapshots/`.
- **One calendar for every region.** EU common wheat is mostly winter-sown and rainfed,
  and the calendar does not say where spring wheat is grown. Regions where much of the
  wheat is spring-sown, as in Finland and Sweden, are kept, and this is reported as a
  limitation.
- **A region's maturity day.**
  - Each E-OBS cell takes the value of the 0.5° calendar cell that contains its
    centre.
  - The region's maturity day is the mean over its cells, with the weights of section
    3.2 (all cells), over the cells that have a value, rounded to the nearest day.
  - The day of the year is read in a non-leap year (day 201 is 20 July), and that
    calendar date is used in every harvest year.
- **The harvest window.** From 1 March to the maturity date, both included.
  - `GDD`, `EDD` (30 °C), `P` and the M3 bins are computed over it, exactly as in
    section 3.2.
  - The weather is read up to 31 October so that the placebo window below fits. No
    existing window changes.
- **The amended placebo (E5-A).** `EDD` above 30 °C and precipitation over the 61 days
  after the maturity date (from the day after it). This replaces 1 August to
  30 September, which is not after harvest in the north (finding C7).

### 8.2 A rule for yield series that are not regional measurements

It applies to the panel after the rules of section 3.1. Regions are then checked again
against the 10-year rule, and singletons are dropped as before. The rule uses published
areas, productions and yields only, never weather.

- **Q1, one figure for every region.** In a country-year with at least 3 panel regions,
  if all their yields lie within 0.5% of each other (largest over smallest below
  1.005), those region-years are excluded.
- **Q2, a pro-rated series.**
  - The setting: a country with at least 4 regions present in both years `t-1` and `t`.
  - The test: the standard deviation across those regions of the change in log yield
    from `t-1` to `t`.
  - The rule: if it is below 0.02, both years of those regions are excluded.
- **Q3, carried forward.** If a region's published area and production both equal the
  previous year's, the later year is excluded.
- **Q4, sums that disagree.**
  - The setting: a published NUTS-1 code with at least two published NUTS-2 codes
    beneath it in a year.
  - The test: their productions sum to within 1% of the NUTS-1 production, while their
    areas sum to more than 5% away from the NUTS-1 area.
  - The rule: that year's region-years of those NUTS-2 codes are excluded.
- **Q5, level shifts.** If a region has three or more consecutive panel years in which
  its yield is below half, or above twice, the median of its other panel years, those
  years are excluded.
- **How the thresholds were set.** They were set after the review described these
  series, so they are informed by the data. They are not informed by any estimate of
  this amendment. The report lists every excluded row by rule. It also lists which of
  the review's cases (finding D1) the rule catches and which it misses.

### 8.3 Inference and E2

- **E2 is tested directly.**
  - M1's precipitation terms are reparametrised so that the dry-season contrast of the
    change log of 2026-10-24 is a single coefficient: `P / d1` and `P^2 - (d2 / d1) P`.
  - `d1` and `d2` are the means over regions of `P10 - P50` and `P10^2 - P50^2`.
  - E2 is established negative if that coefficient's test rejects and it is negative.
- **Two-way.** As registered (change log of 2026-10-24).
- **The wild bootstrap.** It now uses the two-way statistic.
  - **The statistic:** the two-way t, or for E5-A the two-way Wald statistic. The
    two-way variance, with its eigenvalue rule, is recomputed in every draw.
  - **The draws:** the null is imposed, with Rademacher weights and 9999 draws, run twice:
    - weights by country (seed 20261101), which keeps each country's dependence over
      the years;
    - weights by year (seed 20261102), which keeps the dependence across countries
      within a year.
  - **The p-value:** the share of draws at least as extreme, in absolute value for a t.
    The amended bootstrap p-value is the larger of the two runs.
  - **Reference:** MacKinnon, Nielsen and Webb (2021) on the wild bootstrap with
    multiway clustering.
- **Decisions.** Holm over E1-E3, separately under the two-way method and the amended
  bootstrap. A result is established only if it is rejected under both, with the sign
  its rule requires (section 5).
- **Leave one country out (descriptive).** E1, E2 and E3 are re-estimated without each
  country in turn, re-applying the singleton rule. The report gives the range and the
  country whose removal moves each estimate most. This decides nothing.

### 8.4 Specification and reporting

- **M1-A.** M1 on the sample of 8.2, with the variables of the harvest window. E1 to
  E3 and E5-A are decided from M1-A alone. E3 and its minimum detectable change are as
  registered, and the change is also given at the Holm level (5% divided by 3).
- **Which change does what (descriptive).** E1 and E2 are reported with two-way
  intervals and both p-values of 8.3 under four combinations:
  - the registered window and sample, so that only the inference changes;
  - the harvest window alone;
  - the data rule alone;
  - both, which is M1-A.
- **Sensitivity (fixed now).** A window that ends 14 days before the maturity date,
  reported for E1-E3. As in section 6, it changes a decision only if it reverses a sign.
- **M3-A (descriptive).** The bins over the harvest window, with two-way intervals and
  the amended bootstrap p-values.
- **Not recomputed.** E4, and the sensitivity analyses of section 6.
- **Report.** `docs/AMENDMENT.md`, generated like `docs/RESULTS.md`. Any change to this
  section after a result is computed goes in the change log.

## 9. A second crop: grain maize (registered 2026-10-26)

Registered before any maize yield was related to any weather variable, and before any
maize yield was looked at. Maize grows through the July and August heat, which wheat
mostly avoids. It is known to be sensitive to heat, so it should give the design far more
heat to work with than wheat. The wheat results, registered and amended, stand as they
are; maize is reported separately.

**What was seen before it was written.**

- **The wheat study:** every wheat result, the review and Amendment A1. Maize uses A1's
  data rule and inference because they fix problems found in wheat, not because of any
  maize result.
- **Eurostat's grain maize, coverage only (`apro_cpshr`, `C1500`).**
  - Which NUTS-2 and NUTS-1 regions and years have both area and harvested production.
    About 51 NUTS-2 regions a year have both in 2000-2006, 88 in 2007-2009, and 199-223
    from 2010.
  - Before 2010 there is no Spain, Italy, Hungary, Poland or Austria, and before 2007 no
    France. Germany is published at NUTS-1 only.
  - No area, production or yield value was looked at.
- **The GGCMI maize calendar (rainfed), distribution only.** Over Europe, planting falls
  mostly between days 91 and 131 (1 April to 11 May). Maturity falls mostly between days
  233 and 283 (21 August to 10 October), and no cell matures after day 290 (17 October).

### 9.1 Data

- **Yields.** Eurostat `apro_cpshr`, grain maize and corn-cob-mix (`C1500`).
  - The same two measures as wheat: area (`AR_THS_HA`) and harvested production at EU
    standard humidity (`HPRD_HUMD_EU_THS_T`).
  - Frozen in a snapshot of its own, with its SHA-256 hash.
- **Harvest years: 2010-2024.** Coverage before 2010 misses most of the EU's maize, as
  above.
- **Regions.** The EU-27 at NUTS 2021, joined by the rules of section 3.1 and its change
  log: the region's own code, a NUTS-1 region with a single NUTS-2 region, and official
  pure recodes.
- **Sample rules, in order.**
  1. Area of at least 1 thousand hectares, and production present.
  2. A yield between 1 and 20 t/ha. The bounds are set from agronomy, not from the
     data: EU grain maize rarely yields below 2 or above 15 t/ha.
  3. The data rule of section 8.2 (Q1-Q5), with the same thresholds.
  4. Complete season weather.
  5. At least 10 years per region.
  6. Singletons dropped, as in M1.
- **The season.** From each region's planting date to its maturity date.
  - Both dates come from the GGCMI Phase 3 crop calendar v1.01, rainfed maize
    (`mai_rf_ggcmi_crop_calendar_phase3_v1.01.nc4`). They are computed for each region
    as in section 8.1: the weighted mean of the calendar cells, rounded half up, read
    in a non-leap year.
  - A season that would start before 1 March starts on 1 March.
  - The calendar is frozen beside the wheat calendar.
- **Variables over the season,** from E-OBS as in section 3.2.
  - `GDD`: degree days between 10 °C and 30 °C, the usual base for maize, in hundreds.
  - `EDD`: degree days above 30 °C, in tens. The threshold is the same as for wheat.
  - `P`: precipitation in hundreds of mm, and its square.
  - The 3 °C time bins of M3.
  - The weather is read up to 31 December, so that the placebo window fits. No existing
    window changes.

### 9.2 Models, estimands and decisions

- **M1-maize.** M1 of section 4: region effects and country-by-year effects, with the
  variables of 9.1.
- **Inference.** As in section 8.3: two-way clustering, and the wild bootstrap of the
  two-way statistic, run with weights by country (seed 20261101) and by year (seed
  20261102), the larger p-value counting. Both use 9999 draws.
- **Decisions.** **E1-maize** and **E2-maize** form one family, adjusted with Holm at 5%
  under each method. A result is established only if it is rejected under both methods
  and is negative.
  - **E1-maize, heat:** the change in log yield for 10 more degree days above 30 °C.
  - **E2-maize, drought:** the dry-season contrast of the change log of 2026-10-24
    (P10 against the median, by region), tested directly as in section 8.3.
- **No E3.** Fifteen harvest years leave too little to compare two periods, so a change
  in sensitivity is not tested.
- **E4-maize, out of sample (descriptive).** As in section 5 and the change log of
  2026-10-24, for 2012, 2015 and 2022, three years of widespread heat or drought in
  maize.
- **E5-maize, placebo.** `EDD` above 30 °C and precipitation over the 61 days after
  each region's maturity date. Their joint test at 5% under both methods is stated
  before any other result.
- **M3-maize (descriptive).** The temperature bins over the season, relative to
  15-18 °C, per 24 hours, with two-way intervals and the bootstrap of 8.3.
- **Sensitivity (fixed now).** Each is reported for E1 and E2. A variant changes a
  decision only if it reverses the sign.
  - `EDD` and `GDD` at 27 °C and 33 °C.
  - A fixed season, 1 May to 30 September.
  - Without Spain, Portugal, Italy and Greece, where much grain maize is irrigated.
  - Regressions weighted by maize area.
- **Beside wheat (descriptive).** E1 and E2 for maize are shown beside wheat's under
  Amendment A1.
- **Reports.** `docs/MAIZE_DATA.md` describes the maize panel before any estimate.
  `docs/MAIZE.md` gives the results, generated like `docs/RESULTS.md`.

### 9.3 Limits fixed in advance

- **Irrigation.** Much of southern Europe's maize is irrigated, which blunts drought and
  some heat. The pooled effect averages over irrigated and rainfed maize, and the
  rainfed calendar is used everywhere.
- **Grain maize only.** Silage maize is not included.
- **Period.** The period is short, which limits E4 and rules out E3.

## Change log

- **2026-10-23, phase 1, before any estimate: NUTS-1 regions.**
  - Eurostat publishes some regions' wheat only at NUTS-1. Among them are Brandenburg
    (`DE4`) and Sachsen-Anhalt (`DEE`), two large wheat regions.
  - A NUTS-1 region that contains a single NUTS-2 region in NUTS 2021 is that region's
    territory. Its series therefore joins the region, by the rule of section 3.1.
  - Where both levels have a value for a year, the NUTS-2 value is used and differences
    above 1% are reported.
  - The crop snapshot of 2026-10-23 adds the NUTS-1 response. The snapshot of
    2026-10-22 is kept unchanged.
- **2026-10-23, phase 1, before any estimate: official recodes replace the geometric
  test.**
  - **The problem.** The 1% geometric test of section 3.1 cannot separate the same
    territory from a changed one. GISCO generalises each NUTS version separately at
    1:3 million, so the boundaries of an unchanged region differ between versions.
    - Pure recodes differ by 1.0-4.6% (Italy's 2010 recodes, such as `ITD3` to `ITH3`)
      and 2.0-3.7% (Greece's 2013 recodes, such as `EL11` to `EL51`). The Ionian Islands
      (`EL22` to `EL62`) differ by 11.6%.
    - Real boundary shifts differ by similar amounts: 3.8-4.7% for Slovenia (`SI01`,
      `SI02`), and 2.3% and 4.6% for Emilia-Romagna and Marche.
  - **The new rule.** It applies the rule's intent, the same territory under another
    code. A NUTS-2 code joins a NUTS-2021 region when Eurostat's official NUTS change
    tables record a chain of pure recodes from it to that region, at every step between
    its version and NUTS 2021.
    - The steps are 2006-2010, 2010-2013, 2013-2016 and 2016-2021.
    - A pure recode is "code change", "code change, label change", "recoded" or
      "recoded and relabelled".
    - Any other change on the way stops the chain. Examples are a boundary shift, a
      split, a merge, or "recoded (boundary change)".
  - **Sources.** The four tables are frozen in `snapshots/nuts-changes-2026-10-23/`.
  - **Reporting.** The geometric shares are still reported for every joined and dropped
    code, as a cross-check.
- **2026-10-24, phase 2, before any estimate: details the plan left open.** They are
  fixed here before any model was fitted to the panel.
  - **Units.** As section 3.2: `GDD` in hundreds of degree days, `EDD` in tens,
    precipitation in hundreds of mm. Log yield is the natural log of t/ha.
  - **Fixed effects.** They are partialled out exactly, by projection on their dummies.
    Observations alone in their region or their country-year carry no information under
    M1, so they are dropped before estimation and counted in the report.
  - **Two-way clustering (Cameron, Gelbach and Miller 2011).**
    - `V = V_country + V_year - V_country-year`.
    - Each term is a sandwich with the factor `G/(G-1) * (N-1)/(N-K)`, where G is that
      term's number of clusters and K counts the weather regressors only. The fixed
      effects are nested within the country clusters.
    - Negative eigenvalues of `V` are set to zero.
    - A single coefficient uses a t test with `min(G_country, G_year) - 1` degrees of
      freedom. A joint test of q coefficients uses `W/q` against `F(q, min(G) - 1)`.
  - **Wild cluster bootstrap.**
    - Clusters are countries, with Rademacher weights, 9999 draws and seed 20261101 for
      every test.
    - The null is imposed: the restricted model's residuals are resampled.
    - The bootstrapped statistic is the t statistic, or for joint tests the Wald
      statistic, with one-way country-clustered errors (CR1).
    - The p-value is the share of draws at least as extreme as the original statistic.
  - **Holm.** The E1-E3 family is adjusted separately under each method. A result is
    established only if it is rejected after Holm under both methods and has the sign
    the rule requires.
  - **E2.** The effect in region `i` is `g1 (P10_i - P50_i) + g2 (P10_i^2 - P50_i^2)`.
    `P10_i` and `P50_i` are the 10th percentile and the median of the region's
    precipitation over its panel years, with linear interpolation. E2's effect is the
    mean over regions, with a delta-method two-way interval. The test is the joint test
    of `g1` and `g2`.
  - **E3.**
    - The interaction is `EDD x 1[year >= 2012]`, alongside `EDD`.
    - The smallest detectable change is `(t_0.975 + t_0.80) x SE`, using the two-way SE
      and the same degrees of freedom.
  - **E4.** M2 is fitted without the year. For each region in the sample that year, the
    "normal" log yield is the region's fitted trend plus the weather effect at its mean
    weather over the fitted years.
    - Predicted anomaly: the weather effect at that year's weather minus the effect at
      the mean weather.
    - Observed anomaly: the log yield minus the normal.
    - Area-weighted EU yields: computed from these, with that year's wheat areas, as a
      percentage of the area-weighted normal.
  - **E5.** The joint test of the two post-harvest terms: `EDD` above 30 °C in tens and
    precipitation in hundreds of mm, from 1 August to 30 September. It is tested at 5%
    under both methods and is not part of the Holm family.
  - **M3.** Coefficients are per day in a bin, relative to `15-18` °C, with two-way 95%
    intervals.
  - **Sensitivity analyses.** They report E1-E3 as estimates with two-way 95% intervals.
    - Region trends with year effects: region effects, region linear trends and year
      effects.
    - Area weights: weighted least squares with weights equal to the wheat area.
    - The water-balance variant: replaces `P` by the water balance in hundreds of mm, in
      the same quadratic form and in E2.
    - Variants that drop observations (cells below 800 m, at least 20 years, NUTS-2021
      codes only) re-apply the singleton rule.
  - **Second route.** M1's coefficients are recomputed with explicit dummy variables and
    must agree with the projection route to 1e-8.
- **2026-10-26, after the results: the review and its post-hoc checks.** A critical
  review ([docs/REVIEW.md](docs/REVIEW.md)) was written after the registered results
  and before publication. Nothing in this entry changes a registered verdict; every
  check below is labelled post hoc wherever it is reported.
  - **The E2 rule was flawed.** It takes significance from the joint test of `g1` and
    `g2`, which asks whether precipitation matters anywhere in its range, not whether
    the dry-season contrast is non-zero. The registered verdict ("established
    negative") stays on record. The report and the article add the direct test of the
    contrast, with both inference methods and Holm, and do not claim an established
    drought effect.
  - **Post-hoc checks added to the outputs**, so that the article can quote them:
    - the direct E2 test;
    - a delta-method interval for the precipitation optimum;
    - each country's share of the identifying variation of `EDD` and of time at 36 °C
      and above;
    - E1 without Spain;
    - the one-way standard errors of the 36 °C bin;
    - a descriptive July window (1-31 July) in the weather table, to show when the
      season's heat falls, and each region's centroid latitude
      (`docs/panel/region_latitude.csv`) for the review's split at 40°N;
    - the wild bootstrap p-value of every temperature bin (M3 was registered with the
      two-way intervals only);
    - E3's minimum detectable change at the Holm level (5% divided by 3);
    - E4's 2003 miss by country, with each country's 2003 heat and the worst regions'
      heat and wheat area.
  - **The sensitivity rule.** Section 6 says a variant changes a decision only if it
    reverses the sign. The report had also required the variant's interval to exclude
    zero. The report now follows section 6. No point estimate reverses its sign, so
    nothing changes.
  - **The temperature bins.** They measure time as fractions of 24 hours. The report and
    the article now give M3 per 24 hours or per hour, never "per day".
- **2026-10-26, after the results, before any of it is computed: Amendment A1
  (section 8).** It adds three things: a window that ends at harvest, a rule for yield
  series that are not regional measurements, and inference that allows for dependence
  across countries within a year. It also tests E2 directly. It is registered before any
  of it is computed, and its results are reported beside the registered ones.
- **2026-10-26, Amendment A1, before any of its estimates: details it left open.**
  - **Q2** uses the sample standard deviation (n - 1).
  - **Q3** compares the harmonised published values for exact equality.
  - **Q4** takes as children the published NUTS-2 codes that begin with the NUTS-1 code.
    They map to NUTS-2021 regions through the region map.
  - **Rounding.** The maturity day is rounded half up.
  - **Missing harvest-window weather.** A region-year without it leaves M1-A, as in
    section 3.2.
  - **E2's weights.** `d1` and `d2` are computed on the sample of each fit: M1-A, each
    combination of 8.4, each country left out and the sensitivity.
  - **The bootstrap.** A draw whose clipped two-way variance is zero has an infinite t
    and counts as extreme.
  - **What was seen.** The data rule, which uses yields only, was run before any
    estimate of the amendment. It flags 249 region-years, against the review's 86,
    because Q2 also flags Czechia in nine pairs of years: the change in log yield there
    differs across 8 regions by an SD of 0.009-0.019. The thresholds were not changed.
- **2026-10-26, after the results: ties in the bootstrap p-value (implementation).**
  - **The problem.** The draw with every weight +1 reproduces the data. Its statistic
    therefore equals the original one in exact arithmetic. Whether it counted as "at
    least as extreme" depended on the last bits of the arithmetic, which differ between
    processors. This made a synthetic size test fail on one CI machine and pass on
    others.
  - **The fix.** A draw within a relative 1e-9 of the original statistic is a tie, and
    ties count as at least as extreme, the convention the plan's wording already implies.
    This applies to both bootstraps.
  - **No result changes.** Every registered and amended p-value regenerates unchanged
    under the fix.
- **2026-10-26, before any maize data beyond coverage: a second crop, grain maize
  (section 9).** It is registered before any maize yield was looked at. Section 9 says
  what was seen first: maize coverage counts and the calendar's distribution.
- **2026-10-26, maize data, before any maize estimate: details section 9 left open.**
  - **Q5 with a single year.** Maize applies the data rule before the 10-year rule
    (9.1), so a region can have a single year. With no other year there is nothing to
    compare, so Q5 does not flag it. In wheat the rule runs on a panel of regions with at
    least 10 years, so nothing there changes.
  - **Weather.** Maize uses all cells, as in the primary wheat model. A region without a
    calendar value has no season weather and leaves the panel at rule 4.
  - **What was seen.** The maize panel and its coverage report (`docs/MAIZE_DATA.md`):
    counts, exclusions and the data rule's flags. No yield was related to any weather
    variable.
- **2026-10-27, maize estimates: details section 9.2 left open, fixed in the code before
  the run and recorded here after it.**
  - **The irrigation sensitivity** drops Spain, Portugal, Italy and Greece before the
    singleton rule.
  - **The area-weighted sensitivity** weights the regressions, and E2's contrast keeps
    the unweighted means over regions of `P10 - P50` and `P10^2 - P50^2`, as in the
    wheat variant.
  - **E4** follows the change log of 2026-10-24 exactly, with maize area.
  - **The placebo's report** says when one method rejects and the other does not. The
    registered rule, which needs both methods to reject, is unchanged.
