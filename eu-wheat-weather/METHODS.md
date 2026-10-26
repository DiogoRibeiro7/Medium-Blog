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
