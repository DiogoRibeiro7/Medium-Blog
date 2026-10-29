# The Iberian electricity price as a linear programme: research plan

Registered 2026-10-28, before any file from the study period was downloaded and before any
programme was solved. Changes after this date go in the change log at the end, dated, with
what had been seen when each was made.

## 1. Question

Every day, the Iberian day-ahead market (MIBEL, run by OMIE) sets a price for each
15-minute period of the next day in Spain and in Portugal. Buyers and sellers submit
stepwise bids: a quantity in MW at a price in €/MWh. If every bid were divisible and
independent of the other periods, clearing the market would be a linear programme, and
the price would be the dual value of its balance constraint. The real clearing is not
that programme. It is solved jointly with the rest of Europe (EUPHEMIA), and some Iberian
sell orders are blocks or scalable complex orders, which are indivisible or tie
periods together. This study asks four questions of one year of OMIE's public order
books.

- **Q1, the price as a dual.** With the realised outcome of the non-convex orders and the
  cross-border flows held fixed, is the published price an optimal dual value of the
  remaining linear programme in each period?
- **Q2, the cost of non-convexity.** Treating every order as divisible and every period as
  separate (the linear relaxation), how far does the price move from the published one?
- **Q3, the merit-order effect.** By how much does 1,000 MW more supply at 0 €/MWh lower
  the price, holding every bid and flow fixed, and is the effect smaller around midday?
- **Q4, the French border.** By how much would the price move if the flows with France
  were zero, holding every bid fixed? (Descriptive only.)

**A fact, not a result.** For one zone, the clearing programme of section 4 has one
equality constraint. By complementary slackness, λ is an optimal dual value exactly when
every sell step priced below λ is fully accepted, every sell step priced above λ is
rejected, and the same holds in reverse for buy steps; a step priced exactly at λ may be
partly accepted. So the optimal dual values form an interval, and "the price clears the
bids" and "the price is an optimal dual" are the same statement. The interval has
positive width only when no step is partly accepted.

## 2. What was seen before this plan was written

- **The format.** OMIE's public file specification, "Modelo de Ficheros para la
  distribución pública de información", version 1.37: the files `marginalpdbc` (prices,
  Portugal then Spain), `curva_pbc` (aggregate supply and demand curves, every offered and
  matched step with its zone, side and order type) and `capacidad_inter_pbc`
  (interconnector capacities; border 2 is Portugal, 3 France).
- **Files outside the study period.** To read the format, three files were downloaded:
  `curva_pbc_20250615.1` (hourly periods, before the 15-minute market), and
  `curva_pbc_20261015.1`, `marginalpdbc_20261015.1` and `capacidad_inter_pbc_20261015.1`
  (after the study period). In them:
  - **Codes.** Zone codes MI (the joint Iberian curve), ES and PT; order types S (simple),
    C01 (simple block), C02 (scalable complex order), C04 (exclusive block); and fixed
    flows Imp/Exp FR, PT and ES.
  - **Flows.** The flows appear among the matched steps only.
  - **Volume.** On 15 October 2026, 527,480 steps in 96 periods.
  - **Prices.** The prices of 15 October 2026 were printed. No curve was compared with a
    price.
- **Terms.** OMIE's legal notice: its public information may be used freely, keeping its
  original content intact and citing the source.
- **Not seen.** No file from the study period was downloaded, and no programme was solved.

## 3. Data

- **Period.** The first year of 15-minute periods: delivery days 1 October 2025 to
  30 September 2026.
- **Days.** Four days per calendar month, 48 in all, drawn without replacement within each
  month by one generator seeded 20261028, months in calendar order. Every period of a drawn
  day is used, including the 92 or 100 periods of the days the clocks change.
- **Files per day.** The highest published version of `curva_pbc`, `marginalpdbc` and
  `capacidad_inter_pbc`, downloaded from omie.es. They stay outside the repository, with
  their names, sizes and SHA-256 hashes in a manifest. OMIE keeps six years of files, so
  the raw data can be fetched again. The derived per-period tables are committed.
- **Source.** Data: OMIE (OMI-Polo Español, S.A.), omie.es, cited wherever the data or
  results appear.

## 4. Zones, periods and the two programmes

- **Zones.** A period whose curve rows are all MI has one zone, the joint Iberian market.
  A period with ES and PT rows has two zones, Spain and Portugal, after the market split.
  A period with MI rows alongside ES or PT rows is reported and left out.
- **Steps.** In a zone-period, the offered steps (O) are the bids. The matched steps (C) of
  types C01, C02 and C04 are the accepted quantities of the non-convex orders. The matched
  Imp and Exp steps are the cross-border flows.
- **The realised programme, `LP_R`.** The simple offered steps are the variables, each
  between 0 and its quantity, and the programme maximises the buyers' bid value minus the
  sellers' offer value. The accepted non-convex quantity and the imports are fixed supply,
  and the exports are fixed demand. One balance constraint makes supply equal demand, and
  its optimal dual values form the interval Λ_R.
- **The relaxed programme, `LP_X`.** As `LP_R`, but every offered C01, C02 and C04 step is a
  divisible variable at its offered price, instead of a fixed accepted quantity. Flows stay
  fixed. Its dual interval is Λ_X.
- **Solving.** HiGHS through SciPy, versions pinned. The dual interval is computed exactly
  from the steps by the fact in section 1. In every solve, HiGHS's own dual value must lie
  in that interval, and its objective must match the exact one; otherwise the run stops.

## 5. A price from a programme

The published price of a zone-period is π (the `marginalpdbc` price of that zone).

- **Point rule.** When a counterfactual programme has dual interval Λ, its price is the
  point of Λ nearest π. This rule attributes to the counterfactual only the change it
  forces. When Λ is a single point, the rule does not matter.
- **The Spanish price.** Q2-Q4 use the Spanish zone, or the joint zone when there is no
  split. Portugal enters Q1 only.

## 6. Estimands and decision rules

All intervals are percentile bootstraps over the 48 days (9,999 resamples, seed 20261029),
with every period of a resampled day kept together. Q2 and Q3 decide something; their
intervals are simultaneous (Bonferroni: 97.5% each). The others are 95%.

- **E1, the price as a dual (Q1).** The share of zone-periods with π in Λ_R, widened by
  0.01 €/MWh at each end.
  - **Decision:** the realised programme reproduces the clearing if the share is at least
    99%.
  - **If it is below 99%,** E2-E4 are reported as descriptive, and the report says where
    the programme fails.
  - **Also reported:** whether `LP_R`'s accepted volume matches the published matched
    volume within 0.1 MW.
- **E2, the cost of non-convexity (Q2).** For each period, δ = (the point of Λ_X nearest
  π) − π, in €/MWh.
  - **Reported:** the mean of δ over periods, the share with |δ| > 0.01 and the mean |δ|.
  - **Decision:** the non-convex orders move the average price if the simultaneous
    interval of the mean of δ excludes zero.
  - **Expectation, stated now:** no direction; δ = 0 in most periods.
- **E3, the merit-order effect (Q3).** For each period, m = the price of `LP_R` with one
  more sell step of 1,000 MW at 0 €/MWh, minus π (point rule).
  - **Reported:** the mean of m, with the mean by hour of the day.
  - **Midday:** the periods of hours 11-17 in OMIE's numbering (10:00 to 17:00, Spanish
    time). The comparison is G = mean |m| outside midday − mean |m| at midday.
  - **Decision:** the effect is smaller at midday if G's simultaneous interval lies above
    zero.
  - **Expectation, stated now:** G > 0. At midday, solar output keeps the price low and the
    supply curve flat.
- **E4, the French border (Q4, descriptive).** For each period, f = the price of `LP_R`
  with the France flows set to zero, minus π (point rule). Reported as the mean of f, over
  periods importing from France and over periods exporting to it.
- **Descriptive.** The share of periods in which ES and PT split; the steps per period; the
  share of offered C01, C02 and C04 volume that was accepted; the share of periods with Λ_R
  wider than a point, and its median width.

## 7. Sensitivity analyses (fixed in advance)

Each reports its estimand beside the main one and changes no decision.

- **E1's tolerance:** 0.1 €/MWh instead of 0.01.
- **E2 by order type:** relaxing the blocks (C01, C04) only, then the scalable complex
  orders (C02) only.
- **E3's size:** 500 MW and 2,000 MW instead of 1,000.
- **E3's point rule:** the midpoint of the counterfactual interval instead of the point
  nearest π.

## 8. What this study does not claim

- **Not a replication of EUPHEMIA.** The flows with France are taken as given, and the
  French bids are not modelled. The non-convex orders enter only through their realised
  outcome (`LP_R`) or as divisible bids (`LP_X`).
- **Not behaviour.** Q3 and Q4 hold every bid fixed. Real bidders would respond, and with
  the French flows fixed, extra Iberian supply cannot be exported, so E3 overstates the
  effect when export capacity is free.
- **Not welfare, and not a forecast.** Prices are the object; no welfare figure or
  forecast is claimed.
- **Not other years.** One year of the 15-minute market, from 48 sampled days.

## 9. Reproducibility and reporting

- **Versions:** pinned in `pyproject.toml`.
- **Seeds and hashes:** every seed, and the file names and hashes, are in the manifests
  and results.
- **Steps, each in its own pull request:**
  - the data step (download, parsing and descriptive checks), with no programme solved;
  - the programmes and estimates, reported in `docs/RESULTS.md` against this plan;
  - the article, generated from the results with no hand-typed numbers;
  - a critical review before publication.

## Change log

- **2026-10-28.** Plan registered.
- **2026-10-28, the data step, before any programme: details the plan left open.**
  - **Downloading.** Each file's versions are tried from 1 upward until OMIE answers 404.
    Every file of the 48 days is at version 1. A dropped connection is retried after 30 s
    and then 60 s; the first run was cut off by one and resumed. A file is written only
    once it has been read in full.
  - **Reading.** Quantities are read as integer tenths of a MW and prices as integer cents,
    as the specification writes them, so that sums and comparisons are exact.
  - **Flows by side.** A matched flow on the sell side is an import (fixed supply); on the
    buy side it is an export (fixed demand). The labels of the flows across the
    Spanish-Portuguese border vary between days: Exp PT and Imp ES on 17 days with a split,
    Exp ES and Imp PT on 6, and both pairs on 2. So the side decides, not the label. The
    check that Spain's exports equal Portugal's imports in every split period confirms the
    reading.
  - **Summed steps.** Offered steps of the same zone-period, side, type and price are
    summed. Both programmes of section 4 depend on the steps only through their prices
    and the total quantity at each price, so the sum changes neither their optimal values
    nor their dual intervals. The summed steps are committed (`data/derived/steps/`, 13 MB),
    so the later steps run offline.
  - **The joint price.** In every joint period the published Portuguese and Spanish prices
    are equal; π is the Spanish price.
  - **What was seen.** The data report, `docs/DATA.md`. It gives the zones of every period
    (510 of 4,608 periods split), the steps per period, the offered and accepted volume of
    each order type, and the flows with France. It also reports three accounting checks on
    the matched quantities, all of which pass. It shows no price levels, and no curve was
    compared with a price. No programme was solved.
- **2026-10-29, the programmes and estimates, before any programme of the study period was
  solved: details the plan left open.**
  - **Developed on a day outside the study.** The code was written and checked on
    15 October 2026, after the study period, whose files were read for this plan
    (section 2). On that day π lay in Λ_R in all 101 zone-periods, and relaxing the
    non-convex orders lowered the price in most periods (mean δ −10.28 EUR/MWh), against the
    expectation in section 6. That expectation, and every rule, stand as registered.
    Earlier in this step, reading the derived table's format printed its first two rows,
    with two published prices of 1 October 2025; no curve was compared with them.
  - **The exact interval.** Λ is the set of λ at which
    [S(<λ) − D(≥λ), S(≤λ) − D(>λ)] contains R, where S and D are the divisible sell and buy
    volume priced in the given range, and R is the fixed demand minus the fixed supply.
    This is the fact in section 1. Its ends are step prices; an unbounded end is recorded
    as infinite.
  - **The HiGHS check.** `scipy.optimize.linprog` with `method="highs"`. HiGHS's dual of
    the balance constraint must lie in Λ within 10⁻⁶ EUR/MWh. Its objective must equal the
    exact optimal value within 10⁻⁹ of the largest the objective could be (the sum of
    |price| × quantity).
  - **Fixed volume.** In each programme, R = exports − imports − the accepted volume of
    the non-convex order types not relaxed. In the partial relaxations of section 7, the
    type not relaxed stays at its accepted volume. E4 takes the France flows out of R. E3's
    extra supply is one more sell step at 0.00 EUR/MWh.
  - **E1's volume check.** When steps are priced exactly at the clearing price, the
    optimal accepted volume can be an interval. The published accepted buy volume must lie
    within 0.1 MW of it.
  - **Prices in cents.** π and the ends of every Λ are whole cents, so E1's tolerance is
    one cent and every comparison with π is exact. E2's share counts |δ| > 0.01 EUR/MWh,
    that is, at least two cents.
  - **Intervals.** One draw of 9,999 × 48 resampled days (`numpy.random.default_rng(20261029)`)
    serves every estimand. In each resample, a mean over periods is the sum over the drawn
    days divided by their number of periods. Percentiles use NumPy's default (linear)
    quantile. Every interval of E2 and E3, including their means by hour and their
    sensitivity analyses, is at 97.5%; E1's and E4's are at 95%. E1's decision uses the
    estimated share. For E4, a resample with no period in the subset is dropped.
  - **E3's midpoint rule.** An unbounded interval has no midpoint; such periods are
    counted and left out of that analysis.
  - **Λ_R's width.** The share is over all zone-periods; the median width is over those
    wider than a point, with both ends finite.
  - **Where E1 fails.** Whatever E1's value, the report gives the zone-periods where π
    lies outside Λ_R, by zone, with the days, the hours and the distance from π to Λ_R.
- **2026-10-29, after the programmes were solved: what was seen and what was added.**
  - **What was seen.** All of `docs/RESULTS.md`: 37,374 programmes, every HiGHS check
    passed, and every estimand of sections 6 and 7.
  - **Added after the results, descriptive and labelled so in the report.**
    - *Where G comes from.* The share of midday and other periods with π ≤ 0, and mean
      |m| over the periods priced above zero, outside and at midday, with their
      difference (95%). Added because a sell step at 0 EUR/MWh cannot lower a price
      already at or below zero, so G could be partly mechanical.
    - *The midpoint rule.* The number of periods in which E3's counterfactual interval is
      wider than a point, added because the midpoint-rule row matched the registered row
      to two decimals.
    - *Expectations.* Two lines comparing the expectations of section 6 with the outcomes.
  - **Not changed.** No rule, tolerance, estimand or decision.
- **2026-10-29, the article: what it adds.**
  - **Figures,** drawn from the committed results: one quarter-hour's bids as the two
    programmes see them; the mean δ of each day for each relaxation; and the mean m by
    hour, beside the share of quarter-hours priced at or below zero by hour. The last is
    descriptive and was computed after the results.
  - **Figure 1's quarter-hour.** First fixed as 20:00 to 20:15 on the first drawn day.
    Drawn, it turned out far from typical: δ = −84.23 EUR/MWh, against a mean of −11.75. So
    it was changed to the first drawn day's joint quarter-hour whose δ is nearest the mean
    δ, a rule the caption states.
  - **The order types** are described from the EUPHEMIA Public Description (18 December
    2025), read for the article.
  - **Not changed.** No result, rule or estimand.
