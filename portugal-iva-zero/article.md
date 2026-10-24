# Pass-Through of Portugal's 2023 Zero VAT on Food, Estimated from Official Price Indices

*How much of the IVA Zero cut reached consumer prices, how much of the reversal followed, what it did to measured inflation and who saved, estimated with Eurostat's monthly price indices for 27 countries and the index at constant tax rates as the measure of treatment.*

From 18 April 2023 to 4 January 2024 Portugal applied a zero VAT rate to a basket of 46 food products, all taxed at 6% before except vegetable oils, which were taxed at 23%. The measure, known as IVA Zero, was a temporary tax cut on a narrow set of goods during a period of high food inflation. Two questions follow for anyone who has to decide whether to use the instrument again. How much of the cut reached the prices consumers paid, and how much of the money went to the households the policy was meant to help?

The first question has already been studied with scraped supermarket prices. Bernardino, Gabriel, Quelhas and Silva-Pereira collected daily online prices from four large Portuguese retailers and found that the cut was passed on in full, stayed passed on while it lasted, and was reversed symmetrically when VAT returned; an earlier working-paper version, as reported in the press, put the reversal at about 70%. As a robustness check they also compared Portugal's monthly HICP subclasses with Spain's from January 2023 and found a fall of 5.48% in the affected categories. Their main data cover the websites of four chains; INE's consumer price indices are built from prices collected in every type of outlet it samples.

This study takes the official indices as its main source and asks what they can establish on their own. Three things are new relative to the earlier work: we measure treatment, subclass by subclass, with the index at constant tax rates instead of a hand-made mapping of the decree, which also gives a dose-response across partly covered subclasses; we compare Portugal with eleven countries screened for tax changes, with placebo-based inference and an explicit calculation of how precise monthly indices can be; and we follow the reversal and the two years after it. We then use the estimate to quantify the effect on headline inflation and the saving of households in each income fifth.

Everything below comes from public tables. The download, the estimates, the tests and the figures are in the [portugal-iva-zero folder of the Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-iva-zero).

## The measure

The government announced the cut on 24 March 2023 and it took effect on 18 April. The basket covered bread, potatoes, pasta and rice; cow's milk, yoghurt and cheese; apples, bananas, oranges, pears and melons; red beans, black-eyed peas and chickpeas; fourteen vegetables, from onions and tomatoes to turnip tops and Portuguese cabbage; pork, chicken, turkey and beef; cod, sardines, hake, horse mackerel, sea bream and mackerel; canned tuna; chicken eggs; olive oil, vegetable oils and butter; plant-based drinks and yoghurts; and gluten-free products for coeliacs. Every item except vegetable oils had been taxed at the reduced rate of 6%. A full pass-through of a 6% cut lowers the consumer price by 1 − 1/1.06 = 5.66%, or 5.83 log points.

The measure was planned to run until October and was later extended to the end of the year. On 10 October 2023, presenting the 2024 budget proposal, it confirmed that the measure would not be renewed and that support would be redirected to vulnerable families. The end date was later moved to 4 January 2024 after retailers argued that a change on 1 January was impractical, and VAT returned on 5 January. Vegetable oils did not return to 23%: the 2024 budget moved them to the intermediate rate of 13%.

The government estimated the forgone revenue at about 410 million euros for April to October and 140 million euros for the extension, about 550 million euros in total. That is an ex-ante figure for VAT not collected. It does not depend on whether prices fell, and it includes purchases by non-residents.

## Data

We use three Eurostat tables, monthly from January 2018 to December 2025, for all 27 member states:

- the HICP index by ECOICOP subclass (`prc_hicp_midx`), the finest level Eurostat publishes, with 48 to 56 food subclasses per country;
- the same indices at constant tax rates (`prc_hicp_cind`), computed by each national statistical institute by holding tax rates at their level in the base period;
- the HICP item weights (`prc_hicp_inw`).

The ECOICOP subclasses end in December 2025, when the HICP moved to the 2018 classification of individual consumption. Two years after the reversal is enough for what follows.

For the distribution of the saving we use the household budget survey tables by income fifth (`hbs_str_t223`, `hbs_exp_t133`) from the wave Eurostat labels 2020. In Portugal that wave is INE's 2022/2023 expenditure survey, collected from February 2022 to February 2023, the year just before the policy. We convert its purchasing-power amounts to euros with INE's published mean expenditure per household for the same survey (24,190 euros a year), and we scale totals with resident household consumption from the national accounts (`nama_10_gdp`) and the number of private households in the 2021 Census (4,149,096).

The observational unit throughout is a country × subclass × month price index. The outcome is its logarithm. Treatment varies across subclasses and over time, but only in Portugal.

Two features of monthly price indices matter for the design. A monthly index is built from prices collected during the month, so a change that takes effect in the middle of a month may be partly or not at all visible in that month. The cut took effect on 18 April; the scraped-price study notes that INE collected the April prices in the first two weeks of the month, and the April index shows no fall. The reversal took effect on 5 January, and January 2024 is fully treated. And a subclass index aggregates many individual products with fixed weights within the year, so a measure that covers some products of a subclass and not others shows up as a partial change in the subclass, proportional to the covered share of its weight. The second feature, which complicates a hand-made mapping of the decree, is what the constant-tax index lets us use.

## Reading the treatment from the constant-tax index

The 46 products of the decree do not map cleanly onto the HICP's subclasses. "Fresh vegetables" contains the fourteen vegetables in the basket and others that were not. "Other meat preparations" contains some products the decree covered and some it did not. Mapping the decree by hand would introduce a classification error that is hard to quantify. Statistics Portugal (INE) has already done the mapping, at the level of the individual prices it collects, when it computed the index at constant tax rates, and we use that instead.

Let `p(c,s,t)` be the log HICP of subclass `s` in country `c` and month `t`, and `p*(c,s,t)` the log of the same index at constant tax rates. We define the tax wedge

```
τ(c,s,t) = p(c,s,t) − p*(c,s,t).
```

The constant-tax index assumes that any tax change is passed on in full and immediately, and removes it. The wedge therefore moves only when a tax rate changes, by the amount a full pass-through would imply, and it does not depend on what prices actually did. That makes it a measure of treatment intensity, not of the outcome.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig01_tax_wedge.png)

*Figure 1. Tax wedge (log HICP minus log HICP at constant taxes) of every Portuguese food subclass whose wedge moved with IVA Zero, rebased to February 2023. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

Between March and May 2023 the wedge fell by between 5.57 and 5.80 log points in thirteen subclasses and returned by the same amount in January 2024. These are the subclasses the measure covered entirely: rice, bread, beef and veal, pork, poultry, dried and salted fish (in Portugal, mostly bacalhau), whole milk, low-fat milk, yoghurt, cheese, eggs, butter and olive oil. Twelve more subclasses moved by between 0.4 and 4.8 log points, the ones the measure covered in part: pasta, other meat preparations, fresh, frozen and preserved fish, other milk products, fresh fruit, fresh and frozen vegetables, potatoes, other tuber products and other bakery products. The wedge of "other edible oils" fell by 20.60 log points, which is log 1.23, and rose in January 2024 by 12.14, which is log 1.13: the intermediate rate.

Twenty-seven food subclasses show no move at all from 2022 to 2025 and form the untreated group: sugar, jams, chocolate and confectionery, flour, breakfast cereals, pizza, lamb, other meats, seafood, preserved milk, dried fruit and nuts, preserved fruit, dried and preserved vegetables, crisps, ice cream, sauces, salt and spices, baby food, ready-made food, other food products, coffee, tea, cocoa, mineral water and juices. Three more subclasses had tax changes of their own and are excluded: soft drinks (tax changes in January 2023 and January 2024, the second consistent with the sugar-content tax), margarine (a change in January 2023), and dried and smoked meat, whose wedge moved by 0.21 log points in January 2024, consistent with the 2024 budget also moving the cured sausage known as alheira to the 13% rate.

In the 2023 HICP the thirteen fully covered subclasses weigh 7.67% of the basket, the twelve partly covered 7.13%, the oils 0.50% and the untreated food subclasses 4.04%. Weighting the partly covered subclasses by their wedge change, the measure was equivalent to a full 6% cut on 11.1% of the HICP, plus the 23% cut on 0.5%.

The April 2023 wedge did not move. The cut took effect on 18 April, but in the constant-tax index it appears only in May, consistent with April prices having been collected before the 18th. We treat April as untreated and report its prices separately. January 2024 is fully treated by the reversal, since VAT returned on the 5th.

## The raw picture

Before any estimator, Figure 2 shows the indices themselves: fixed-weight baskets of the covered and the untreated subclasses, with Portuguese 2023 weights, in Portugal and in the comparison countries described in the next section.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig02_price_baskets.png)

*Figure 2. Log price of the covered and untreated food baskets (Portuguese 2023 weights, geometric), Portugal and the mean of the comparison countries, relative to February 2023. Source: Eurostat prc_hicp_midx and prc_hicp_inw.*

The covered basket in Portugal drops in May 2023 by about five log points relative to the other three series, stays apart until December and rejoins them in January 2024. The figure also shows what the estimators have to deal with. Between January 2021 and December 2022 the four baskets rose by 15 to 26 log points, and the covered staples rose faster than the untreated items, in Portugal more than elsewhere. From December 2022 to April 2023 the four series move together. The relative-price comparisons that follow are only as good as that plateau.

## Estimators

Let `D(s) = 1` for the thirteen fully covered subclasses and `D(s) = 0` for the 27 untreated ones. Write `r(c,s,t) = p(c,s,t) − p(c,s,b)` for the log change since a base month `b`, which we set to February 2023, the last month before the announcement.

**Within Portugal.** The event-study coefficient with subclass and month fixed effects, on a balanced panel, is the difference between the two groups' mean log change:

```
β_W(t) = mean_{s: D=1} r(PT,s,t) − mean_{s: D=0} r(PT,s,t).
```

This estimator assumes that, without the measure, covered and untreated food would have moved together in Portugal. Figure 2 already suggests why that assumption is weak. In 2023 the relative price of staples such as dairy and butter fell across Europe, and the within estimate would attribute that fall to the Portuguese tax.

**Across countries.** We subtract from each Portuguese subclass the mean change of the same subclass in a set of comparison countries `K`:

```
d(s,t) = r(PT,s,t) − (1/|K_s|) Σ_{c ∈ K_s} r(c,s,t),
β_A(t) = mean_{s: D=1} d(s,t) − mean_{s: D=0} d(s,t).
```

where `K_s` is the set of comparison countries that publish subclass `s` (a few publish 34 of the 40). This is a triple difference: covered against untreated, Portugal against other countries, after against before. It removes price movements specific to a product but common across Europe (the olive oil harvest failures, the dairy cycle, the coffee and cocoa markets) and movements specific to Portugal but common across food. The identifying assumption is that, in the absence of the measure, the difference between Portugal's covered and untreated food would have moved like the same difference in the comparison countries.

**Comparison countries.** The tax wedge also screens the comparison set. A country is excluded if the wedge of any of the 40 covered or untreated subclasses moved by half a log point or more in any month from January 2022 on. Spain (its own cut on basic food from January 2023), Poland, Croatia, Cyprus, Czechia, Estonia, Romania and Slovakia had food VAT changes; Bulgaria, France, Ireland, the Netherlands and Slovenia had tax changes in one or two subclasses; Malta publishes no food subclasses at constant taxes. We also exclude Hungary, which capped the prices of several basic foods until 1 August 2023; the wedge cannot show that. Eleven countries remain: Austria, Belgium, Germany, Denmark, Greece, Finland, Italy, Lithuania, Luxembourg, Latvia and Sweden. Two of them ran voluntary price-restraint schemes in this period, a "household basket" in Greece from November 2022 and the "carrello tricolore" in Italy from October to December 2023; we keep them in the baseline and drop them in a robustness check.

Spain is the obvious neighbour and is not in the comparison set. It cut VAT on basic foods from January 2023 and changed the rates again during 2024; its tax wedge moved in ten of the forty subclasses. Between February 2023 and June 2024 the Spanish rates were constant, which is why the scraped-price study could use Spain's monthly HICP as a control for the entry of the Portuguese measure in a robustness check with a sample starting in January 2023. For the reversal, and for a pre-period that starts in 2018, the Spanish changes would enter the comparison, so we exclude it. The screen is deliberately strict: it uses every month from January 2022 to December 2025, and drops a country even if its tax change fell outside the windows we estimate. The robustness section relaxes it.

**Pass-through.** Dividing by the change in the tax wedge of the same contrast,

```
θ(t) = β(t) / [ mean_{s: D=1} Δτ(PT,s,t) − mean_{s: D=0} Δτ(PT,s,t) ],
```

gives the share of the tax change that reached prices: θ = 1 is full pass-through, θ = 0 none. During the policy the denominator is −5.76 log points.

**Uncertainty.** Prices within a subclass are serially correlated, and the treatment is assigned to subclasses, so we cluster by subclass. With a balanced panel the clustered standard error of a difference in group means is the two-sample standard error across subclasses of each subclass's mean over the months considered. There are 13 treated and 27 untreated clusters, which is few. We therefore also report placebo-based inference, which does not rely on asymptotics.

## Event study

Figure 3 shows `β_W(t)` and `β_A(t)` from mid-2021 to December 2025, with 95% intervals, and the tax wedge as a step function.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig03_event_study.png)

*Figure 3. Covered minus untreated food subclasses, log change relative to February 2023, within Portugal and across countries, with 95% intervals clustered by subclass, and the tax wedge of the same contrast. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

The pre-period is not flat over its whole length. Both estimates sit between −5 and −12 log points in 2021 and rise during 2022, which is the faster rise of Portuguese staples visible in Figure 2. From December 2022 to April 2023 they are close to zero: the mean of the across-country estimate over those five months is −0.09 log points, and of the within estimate −0.23.

In March 2023, the month of the announcement, the across-country estimate is +0.30 log points, and in April +0.25; with HICP weights, +0.99 and +0.73. None is distinguishable from zero. The scraped-price study found a short-lived relative increase of 0.8% after the announcement (1.3% in the working paper), an effect of the same size as the weighted estimates, which the monthly index cannot distinguish from noise.

In May the across-country estimate falls to −4.33 log points and in June to −5.41. Over May to December it averages −4.70 log points, against a tax wedge of −5.76:

- across countries: −4.70 log points, pass-through 82% (95% interval 46% to 117%);
- across countries with HICP weights instead of equal weights: −4.19, 73% (36% to 110%);
- within Portugal: −6.37, 111% (62% to 159%).

The within estimate exceeds full pass-through. The difference between the two estimators, −1.67 log points, is exactly the comparison countries' own covered-minus-untreated change over the same months: a Europe-wide fall in the relative price of the staples the measure happened to cover. We do not use the within estimate as a pass-through.

The difference between the equal-weighted and HICP-weighted estimates comes from the covered side. The subclasses with the largest estimated effects (rice, dried and salted fish, olive oil, eggs) have small weights, and most of the large weights belong to poultry, beef, pork, cheese and low-fat milk, whose effects were smaller. Bread, the largest weight of all at 1.74% of the HICP, is the exception, with an effect beyond the tax wedge.

The intervals are wide. A 95% interval from 46% to 117% excludes a pass-through near zero and is consistent with a full one. With 40 clusters and month-to-month noise of a few log points per subclass, that is the precision the monthly HICP allows over an eight-month window.

## Short windows and dose-response

The long-window estimate depends on the eight months after the cut resembling the plateau before it. A tax change that takes effect at a known date permits a sharper comparison: the change in prices from just before to just after, which is exposed to two months of other price movements instead of ten.

From March to May 2023 the across-country estimate fell by 4.63 log points, a pass-through of 80% (95% interval 58% to 103%); with HICP weights 85% (56% to 114%). From March to June it is 99% (74% to 125%). The intervals are narrower than for the long window because there is less time for other price movements to accumulate.

The partly covered subclasses add a second source of variation. If prices respond to the tax, a subclass with 60% of its products covered should fall by about 60% as much as a fully covered one. Figure 4 plots, for 52 Portuguese food subclasses, the price change from March to May relative to the comparison countries against the change in the tax wedge over the same months. In the comparison means we leave out any country-subclass pair with a tax change of its own from 2022 on, which keeps the partly covered subclasses comparable even where a comparison country changed its tax on that item later.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig04_dose_response.png)

*Figure 4. Price change from March to May 2023 relative to the comparison countries against the change in the tax wedge, 52 Portuguese food subclasses (untreated, partly covered, fully covered), with the fitted line and the line of full pass-through. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

The least-squares slope, with heteroskedasticity-robust standard errors, is 0.87 (95% interval 0.65 to 1.09): each log point of tax cut lowered prices by 0.87 log points. Over May to December the slope is 0.84 (0.52 to 1.17). For the twelve partly covered subclasses alone, March to May, the slope is 1.29 with a standard error of 0.29, noisy but consistent with the fully covered ones. The intercept of the March–May fit is −0.54 log points, the fitted change at a zero tax wedge, close to the untreated subclasses' mean change of −0.49.

Twenty-seven untreated subclasses at zero on the horizontal axis spread between −6.2 and +4.8 log points on the vertical one. That spread is the noise against which the tax effect has to be detected: a single untreated subclass can move by five log points in two months. The fully covered subclasses cluster at −5.6 to −5.8 on the horizontal axis and between −2.0 (poultry) and −7.8 (olive oil) on the vertical one.

Figure 5 shows the subclass effects over the whole policy period.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig05_subclass_effects.png)

*Figure 5. Mean price change from May to December 2023, relative to February 2023 and to the comparison countries, for each fully and partly covered subclass, with the change in its tax wedge. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

Six fully covered subclasses fell by more than the tax wedge: rice (−11.20 log points), dried and salted fish (−11.17), olive oil (−9.14), eggs (−8.05), bread (−7.31) and yoghurt (−6.50). Five fell by less than 70% of it: poultry (−3.77), whole milk (−3.76), pork (−3.04), cheese (−2.70) and beef (−2.68), the last two by less than half. The temptation is to read the first group as retailers passing on more than the tax and the second as retailers keeping part of it. The 2024 data argue against both readings. After VAT returned, rice was still 10.65 log points below its February 2023 relative level, olive oil 8.78 and eggs 6.29: their 2023 "effects" include the unwinding of price spikes that had built up in Portugal in late 2022. Cheese and beef were above their February relative level in 2024 (+2.13 and +1.75), so their small 2023 effects may be the cut offset by an unrelated rise. Subclass effects measured over eight months mix the tax with each product's own price cycle; only their average, and the short windows around the two tax changes, isolate the tax.

## Placebo tests

The clustered intervals rest on asymptotics with 40 clusters. Placebo tests ask a more direct question: how often do the same estimators produce an effect this large where no tax changed?

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig06_placebos.png)

*Figure 6. Left and centre: every two-month and one-month change of the across-country covered-minus-untreated gap from 2019 to 2022, against the March–May 2023 and December 2023–January 2024 changes. Right: the across-country estimator applied to each comparison country as if it were Portugal, and to Portugal, May–December 2023. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

**Placebo jumps.** From March 2019 to December 2022 there are 46 two-month changes of the across-country gap. Their standard deviation is 1.16 log points and they range from −2.15 to +3.79. The March–May 2023 change is −4.63: none of the 46 is as large in absolute value. There are 47 one-month changes, with a standard deviation of 0.83 and a range of −1.75 to +2.13; the December 2023–January 2024 change is +3.32, and none of the 47 is as large. The largest placebo changes (+3.79 over two months, +2.13 over one) both ended in December 2022, during the late phase of the pre-period rise. They are the largest movements the gap made without a tax change, and each is smaller than the policy change over the same horizon.

**Placebo dates.** Applying the full eight-month estimator to fake announcement months from January 2019 to June 2022 gives 42 estimates with a standard deviation of 0.98 log points and a range from −1.55 to +2.14, against −4.70 for the real event. For the within estimator the same exercise gives a standard deviation of 2.34 and a range from −2.03 to +7.03; one placebo date is as large in absolute value as the real effect. The within estimator's distribution is wide because it carries the slow relative-price trends that the across-country estimator removes.

**Placebo countries.** Treating each comparison country as if it were Portugal, with Portugal's covered and untreated subclasses and the other ten as its comparison set, gives eleven estimates from −1.84 (Latvia) to +2.14 (Luxembourg), with a standard deviation of 1.52. Portugal's −4.70 is larger in absolute value than all of them.

These tests do not give an exact p-value for a pass-through, but they bound how often noise of this kind produces an effect of this size for the across-country estimator: in almost four years of monthly data and in eleven countries, never.

## Synthetic control

A synthetic control replaces the equal-weighted comparison mean with weights chosen to reproduce Portugal's pre-period path. The outcome is the relative price of the covered basket, `y(c,t) = log P_covered(c,t) − log P_untreated(c,t)`, both baskets with Portuguese 2023 weights. The weights on the eleven comparison countries are non-negative, sum to one, and minimise the squared gap from January 2018 to February 2023.

![Figure 7](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig07_synthetic_control.png)

*Figure 7. Relative price of the covered basket (covered minus untreated food, Portuguese 2023 weights) in Portugal and in its synthetic control, and the gap between them. Source: Eurostat prc_hicp_midx and prc_hicp_inw.*

The weights are Greece 0.46, Germany 0.29, Finland 0.12, Lithuania 0.08 and Austria 0.05, and the pre-period root mean squared gap is 0.74 log points. The gap from May to December 2023 averages −3.29 log points, 57% of the tax wedge. Dropping Greece from the donor pool gives weights of 0.83 on Germany and 0.17 on Finland, a pre-period fit of 1.22, and a gap of −2.84 (49%). Ranking the ratio of post- to pre-period fit across all twelve countries, Portugal comes first, which with twelve units is a permutation p-value of 0.083, the smallest attainable.

We read the synthetic control as the weakest of the estimators here, for two reasons visible in Figure 7. Its counterfactual depends on HICP weights on both sides, so it inherits the low weights on the subclasses with the largest effects. And after the reversal its gap does not return to zero: it averages +1.01 log points in the first half of 2024 and +4.00 in 2025, with no tax change in either period. A counterfactual that drifts by four points within two years cannot resolve a five-point effect to better than a few points.

## Robustness

The across-country estimate for May to December 2023 under each variation, as pass-through with its 95% interval:

- baseline: 82% (46 to 117);
- base month December 2022: 77% (20 to 133); January 2023: 69% (7 to 132); March 2023: 87% (59 to 115); the mean of December 2022 to February 2023 as base: 76% (33 to 118);
- untreated group restricted to food, without beverages: 80% (42 to 117);
- untreated group without preserved milk and dried vegetables, two subclasses that contain products resembling covered ones: 76% (40 to 111);
- covered group without olive oil: 77% (40 to 114);
- comparison set without Greece and Italy: 79% (42 to 117);
- comparison set screened only from December 2022 to March 2024, which keeps countries whose tax changes fell outside the estimation window (17 countries, adding Bulgaria, France, Croatia, Poland, Slovenia and Slovakia): 78% (45 to 112).

Leaving out one covered subclass at a time moves the estimate between 74% (without rice) and 86% (without beef). Leaving out one comparison country moves it between 78% (without Luxembourg) and 84% (without Germany).

A bracketing estimate avoids choosing a base month. If the counterfactual relative price during the policy is taken as the mean of the plateau before it (December 2022 to April 2023, −0.09 log points) and the level after it (February to June 2024, −0.39), the effect is −4.45 log points, 77%.

One assumption deserves separate treatment. The triple difference subtracts the change in Portugal's untreated food relative to the comparison countries. That change was −0.49 log points from March to May and −1.34 on average from May to December. If those falls were unrelated to the measure, subtracting them is right. If they were caused by it, for instance because retailers competing on the salience of "IVA 0" lowered other food prices as well, subtracting them understates the pass-through. Without the subtraction, the covered subclasses fell relative to the same subclasses abroad by 6.04 log points with equal weights (105%) and 5.54 with HICP weights (96%). The data do not separate the two readings. What they do rule out is the opposite concern: there is no sign that retailers recovered margins on untreated food while the covered items were cheaper.

## How precise a monthly index can be

The intervals above are wide enough that it is worth asking what the data could have detected. For an estimator with standard error `σ`, a two-sided test at 5% has 80% power against effects larger than

```
MDE = (z_{0.975} + z_{0.80}) · σ ≈ 2.80 · σ.
```

For the across-country estimate over May to December the clustered standard error is 1.05 log points, so the minimum detectable effect is 2.95 log points, 51% of the tax wedge. For the March–May window the standard error is 0.67 and the minimum detectable effect 1.87 log points, 33% of the wedge. A cut that reached consumers only to the extent of a third would have been detected with probability 0.8 in the short window; one that did not reach them at all would have been obvious.

The harder question is whether the data can distinguish 80% from 100%. The difference is 0.2 × 5.76 = 1.15 log points. Against that alternative, a test of full pass-through has a power of 41% in the short window and 19% in the long one. A pass-through of 80% and one of 100% would produce estimates that overlap most of the time. This is a property of the data, not of the estimator: forty subclass indices, each moving by one or two log points a month for reasons unrelated to tax, cannot resolve a one-point difference in a two-month window.

That is the main difference between this exercise and the scraped-price study. Daily prices of tens of thousands of individual products, with item fixed effects, average away the product-level noise that dominates a monthly subclass index. Their interval at the implementation date is about one percentage point wide in price terms (from −6.97% to −5.85%, about 20 points of pass-through); ours, in the sharpest window, is about 2.6 log points wide, or 45 points of pass-through. The two sources answer the same question at different resolutions, and the official index confirms the direction and the rough size of the effect without being able to confirm that it was complete.

## The reversal

The end of the measure was announced on 10 October 2023, with the 2024 budget proposal, and took effect on 5 January 2024. Figure 8 rebases the comparison to September 2023, the last month before the announcement.

![Figure 8](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig08_reversal.png)

*Figure 8. Covered minus untreated food subclasses, log change relative to September 2023, within Portugal and across countries, and the tax wedge. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

Covered prices rose relative to the comparison before VAT returned: +0.51 log points in October, +1.67 in November, +1.50 in December. No single month is distinguishable from zero, but the direction matches the slight upward drift the scraped-price study found after the end was announced. In January, with VAT back, the across-country estimate reached +4.82 log points, 84% of the wedge, and it averaged +5.04 over January to March, 88% (95% interval 53% to 123%). By March it was +5.56, 97%.

The size of the reversal depends on where it is measured from:

- December 2023 to January 2024, the month of the change alone: 58% (29% to 86%);
- December 2023 to March 2024: 71% (44% to 98%);
- September 2023 to January 2024: 84% (48% to 120%);
- September 2023 to March 2024: 97% (62% to 132%).

Prices of covered items rose by about 1.5 log points between the announcement and the return of VAT, by about 3.3 in the month VAT returned, and by about 0.7 more over February and March. Measured from before the announcement, the reversal is close to full; measured from the month before VAT returned, it is incomplete. A conjecture, which we cannot test with their data: the two versions of the scraped-price study, about 70% at the reversal in the working paper and a symmetric reversal in the published article, may reflect the same pattern seen through different base periods and post-periods. Whether the increases of November and December were retailers anticipating the end of the measure, or the effect of the cut eroding as the end approached, the monthly index cannot say; the scraped-price study found the effect stable until the end was announced and slightly smaller afterwards.

Did prices return to where they would have been without the measure? On the across-country estimator, relative to February 2023, the gap averaged −0.66 log points in 2024 and +0.99 in 2025, with 95% intervals of about ±2.6 and ±5.6 log points. The dose-response slope for the 2024 average is 0.12 (95% interval −0.28 to 0.53): after the reversal, prices bore no detectable relation to the size of the earlier tax cut. Nothing in the official data suggests a lasting effect in either direction.

## Vegetable oils

Vegetable oils are a second natural experiment inside the first: a 23% cut, and a return to 13% instead of 23%.

![Figure 9](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig09_vegetable_oils.png)

*Figure 9. Other edible oils (sunflower, maize and other vegetable oils and their blends): price relative to the comparison countries and tax wedge, relative to February 2023. Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

From March to May 2023 the price fell by 24.84 log points relative to the same subclass abroad, against a wedge of −20.60; from December 2023 to January 2024 it rose by 12.42, against a wedge of +12.14. At both ends the change matches the tax within a few log points, and the return was proportional to the new 13% rate, not to the old 23%. Between the two dates the relative price fell further, to about −30 log points in June, and after January it stayed about ten log points below where the tax alone would put it. The 2022 spike in sunflower oil prices, which put Portugal's relative price 43 log points above its February 2023 level in April 2022, had largely unwound by December 2022, so the further fall after May is not simply its continuation; we do not know its cause. As with rice and olive oil, the clean information is in the jumps, not in the levels.

## Headline inflation

The effect of the measure on the all-items index follows from its effect on food. The tax wedge of the food and non-alcoholic beverages aggregate (CP01) fell by 3.50 log points from March to May 2023, and CP01 weighs 20.6% of the 2023 HICP. At full pass-through the measure lowered the all-items index by 0.72 log points. At the HICP-weighted across-country estimate of 73%, it lowered it by 0.53. We use the food aggregate's own wedge, not the all-items wedge, because the latter also moved with tax changes outside food during 2023.

Figure 10 removes that effect from the published index between April and December 2023 and recomputes year-on-year inflation.

![Figure 10](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig10_headline_inflation.png)

*Figure 10. Portuguese HICP inflation, all items, % change on a year earlier, published and without the estimated effect of the measure (HICP-weighted pass-through of 73%). Source: Eurostat prc_hicp_midx and prc_hicp_cind.*

Year-on-year HICP inflation in May 2023 was 5.4% and would have been 5.9%; in December 2023, 1.9% instead of 2.4%. Annual average HICP inflation in 2023 was 5.26% and would have been 5.63%. The difference reverses in 2024: annual average HICP inflation was 2.67% and would have been 2.31%, and from May to December 2024 the published year-on-year rate is about half a point higher than without the measure. That is a base effect, not an effect of the measure on 2024 prices: from April 2024 on, each month is compared with a month of 2023 in which prices were held down. A temporary tax cut moves measured inflation from the year it is introduced to the year after, and any indexation rule based on annual average inflation inherits the shift. The scraped-price study estimated an effect of about 0.7 points on headline inflation, close to our full-pass-through figure, and INE estimated, as reported in the press, that the end of the measure added 0.7 points. Both refer to the national CPI, whose weights exclude spending by non-residents; the HICP, used here because it has a constant-tax version and is comparable across countries, includes it.

## Who saved, and how much

The saving of a household is its spending on each covered product times the fall in that product's price. The HICP weights give the composition of food spending; the budget survey gives how much each income fifth spends on food. Applying the HICP-weighted pass-through of 73% to the wedge change of every food subclass, the measure lowered the cost of food and non-alcoholic beverages by 2.57%; at full pass-through, by 3.50%.

In the 2022/2023 survey, food and non-alcoholic beverages were 17.1% of consumption in the poorest income fifth and 10.0% in the richest, or about 177 and 349 euros a month per household. Figure 11 converts this into a saving.

![Figure 11](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig11_incidence.png)

*Figure 11. Estimated saving from IVA Zero by income fifth: euros per household and month, and as a share of household consumption. Budget survey 2020 wave (fieldwork 2022/23) in euros through INE's mean expenditure; HICP-weighted pass-through of 73%. Source: Eurostat hbs_str_t223, hbs_exp_t133, prc_hicp_inw, prc_hicp_cind; INE.*

The saving was 4.56 euros a month per household in the poorest fifth, 5.38 in the second, 6.74 in the middle, 7.78 in the fourth and 8.98 in the richest. As a share of consumption it falls with income, from 0.44% in the poorest fifth to 0.26% in the richest. The measure was progressive relative to spending and regressive in euros: the richest fifth received 26.9% of the total saving and the poorest 13.6%. Over the 262 days of the policy, the average household in the poorest fifth saved about 39 euros and in the richest about 77.

Richer households are also larger. Per adult equivalent, the monthly saving ranges from 3.59 euros in the poorest fifth to 4.87 in the richest; the gap per person is smaller than the gap per household.

Two assumptions bear on these figures. The same composition of food spending (the national HICP weights) is applied to every fifth. Staples such as bread, milk, rice and eggs are probably a larger share of food spending for poorer households, in which case the poorest fifth's saving is understated relative to the richest; we state this as a conjecture, because the published survey tables do not split food by product. And the HICP weights cover spending in Portugal by everyone, including non-residents.

**Totals.** Summed over households, the saving was about 239 million euros on the survey basis, or 325 million at full pass-through. Household surveys under-record spending: survey consumption, mean expenditure times the number of households, is 62% of resident household consumption in the 2023 national accounts. Scaled to the national accounts, the saving was about 386 million euros, or 525 million at full pass-through. The government's 550 million is an ex-ante estimate of forgone VAT; it includes purchases by non-residents, which the HICP weights treat as part of domestic consumption, and it does not depend on pass-through. The scaled full-pass-through total and the government estimate are within 5% of each other, and the difference between 386 and 525 million is the 27% of the tax that, on the HICP-weighted estimate, did not reach prices.

## The same money, three ways

For a decision about whether to repeat the measure, the relevant comparison is with other uses of the same money. Figure 12 compares the estimated saving by income fifth with two transfers of the same total: the same amount to every household, and the whole amount to households in the poorest fifth.

![Figure 12](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-iva-zero/figures/fig12_same_money_three_ways.png)

*Figure 12. Euros per household and month: estimated saving from IVA Zero by income fifth, the same total paid equally to every household, and the same total paid only to households in the poorest fifth. Survey basis. Source: as Figure 11.*

On the survey basis, the households' total saving is equivalent to 6.69 euros a month for every household, or 33.43 euros a month for each household in the poorest fifth: 7.3 times what that fifth received through prices. Scaled to the national accounts, the targeted amount would have been 54.06 euros a month. The ratio 7.3 is the inverse of the poorest fifth's share of the saving, 13.6%, and it does not depend on the pass-through or on the scaling.

The comparison is mechanical and leaves out what makes it hard in practice: identifying the poorest fifth requires income data that are incomplete for exactly the households most in need, transfers have administrative costs and take time to set up, and a price cut reaches households that a transfer would miss. On 10 October 2023 the government itself announced that IVA Zero would end and that support would be redirected to vulnerable families. What the figure adds is the size of the trade-off: on these estimates, a universal price cut on food delivered about one euro in seven to the poorest fifth of households.

## Why the cut reached prices: what the data allow

When the measure was introduced, the expectation that retailers would keep part of it was common enough that the Governor of the Banco de Portugal warned publicly, two days after it took effect, that there was a large risk consumers would not feel it. That the cut passed through to a large extent invites an explanation, and the data in this study are not designed to provide one. What follows are conjectures, stated as such.

The measure was salient. It applied to a named list of 46 products, retailers marked the products in stores, and the price of the basket was tracked and reported in the press throughout the period. A retailer that kept the tax would have been identifiable. The Portuguese food retail market is concentrated in a few large chains that compete on visible prices for exactly these staples. Producer prices for several of the covered foods were falling in 2023 after the 2022 peak, which, as the scraped-price study notes, made it easier for retailers to pass on the cut without compressing margins. And the cut came with support for producers and, three days after the announcement, an agreement between the government, producers and retailers that included monitoring of prices; the estimand in this study is the effect of that whole package on the prices of the covered products.

Two observations from our data bear on these conjectures without testing them. The proportionality across partly covered subclasses (Figure 4) is what one would expect if the cut was passed on product by product, at the level of the individual items INE prices, not as a blanket reduction across shelves. And the symmetry of the reversal, measured from before its announcement, is what one would expect if the cut had changed prices because of the tax and not because of a temporary competitive episode that happened to coincide with it.

## The decision-relevant numbers

For someone deciding whether to use the instrument again, the estimates reduce to a few numbers, each with its uncertainty:

- of a cut of 5.83 log points in the tax on covered food, about 80% to 85% reached prices within two months, with 95% intervals that include full pass-through and exclude anything below about 55%;
- the reversal reached prices to about the same extent, but spread from the announcement of the end, nearly three months before VAT returned, to two months after it;
- the measure lowered annual average HICP inflation by about 0.4 points in 2023 and raised it by about the same amount in 2024, through the base effect; in year-on-year rates the effect was about half a point, from May to December of each year;
- it saved households between about 240 and 525 million euros, depending on the coverage basis and the pass-through, against 550 million of estimated forgone VAT, part of which accrued to non-residents;
- the poorest fifth of households received about one euro in seven of the households' saving, 0.44% of their consumption; the richest fifth received about one in four, 0.26% of theirs.

The first two numbers come from the official prices and are what this study adds; the rest combine them with survey and national-accounts data, and inherit those sources' limitations.

## What the official data can and cannot resolve

The monthly HICP shows clearly that the cut and its reversal moved consumer prices, in the expected direction, at the expected dates, and by amounts proportional to the size of the tax change across subclasses. For the across-country estimator, no change of comparable size occurs in almost four years of placebo months or in eleven placebo countries. That is strong evidence that the measure moved prices.

The size of the pass-through is estimated less precisely. The across-country estimate is 82% with a 95% interval from 46% to 117%; the sharpest estimates, around the date of the change and across subclasses with different coverage, put it at 80% to 87% with intervals that reach full pass-through; the HICP-weighted long-window estimate is 73% and the synthetic control 49% to 57%, both less reliable for reasons specific to their construction. The scraped-price estimate of full pass-through lies inside every interval we report for the entry of the measure. A pass-through as low as 50% lies inside the long-window intervals but outside the short-window and dose-response ones. Between 80% and 100%, monthly indices with 40 subclasses cannot discriminate.

The reversal was close to complete when measured from before its announcement, and spread over about five months, from October 2023 to March 2024. The official data show no lasting effect in either direction once VAT returned.

Several limitations are specific to this design. The HICP constant-tax index identifies which subclasses were treated, but within a partly covered subclass it does not say which products were; the dose-response assumes the price response is proportional to coverage. The untreated food subclasses are a different set of products, with different cost structures, and their relative prices move by several log points over months without any tax change; that is the source of most of the noise. The comparison countries had their own food price dynamics, two of them ran voluntary price-restraint schemes, and the results change little without them. April 2023 shows no fall in prices or in the constant-tax index; we treat it as untreated, and it enters the pre-period plateau and the bracketing estimate as such. The incidence calculation assumes the same food composition in every income fifth and uses a survey that records about 62% of household consumption. None of these limitations changes the direction of any result; most of them widen the range within which the pass-through can lie.

---

**Data and code.** [portugal-iva-zero in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-iva-zero), Apache-2.0. Eurostat: [prc_hicp_midx](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_midx/default/table), [prc_hicp_cind](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_cind/default/table), [prc_hicp_inw](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_inw/default/table), [hbs_str_t223](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t223/default/table), [hbs_exp_t133](https://ec.europa.eu/eurostat/databrowser/view/hbs_exp_t133/default/table), [nama_10_gdp](https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp/default/table). Earlier evidence: [Bernardino, Gabriel, Quelhas and Silva-Pereira, The Full, Persistent, and Symmetric Pass-Through of a Temporary VAT Cut](https://www.ricardoduquegabriel.com/files/BGQS_2025.pdf) and the [Banco de Portugal working paper version](https://www.bportugal.pt/sites/default/files/documents/2024-04/WP202404.pdf). The measure: [product list (DGC/APED)](https://www.cascais.pt/sites/default/files/anexos/gerais/new/dgc-aped_publicacao_informativa_cabaz_de_bens_alimentares_iva_zero.pdf), [end announced with the 2024 budget (ECO)](https://eco.sapo.pt/2023/10/10/cabaz-iva-zero-acaba-e-apoios-sao-canalizados-para-familias-vulneraveis/), [extension to 4 January (Jornal de Negócios)](https://www.jornaldenegocios.pt/economia/impostos/iva/detalhe/governo-recua-e-prolonga-iva-zero-ate-4-de-janeiro), [cost estimates (DN)](https://www.dn.pt/economia/a-medida-que-deu-55-euros-em-imposto-a-cada-consumidor-em-2023), [Banco de Portugal Governor's warning (ECO)](https://eco.sapo.pt/2023/04/20/iva-zero-tem-grande-risco-de-nao-ser-sentido-pelos-consumidores-avisa-centeno/), [vegetable oils at 13% in the 2024 budget (Agroportal)](https://bo.agroportal.pt/oe2024-oleo-alimentar-e-alheiras-com-taxa-intermedia-do-iva). Mean household expenditure 2022/23: INE, as summarised by [DREM Madeira](https://estatistica.madeira.gov.pt/en/download-now-3/social-gb/popcondsoc-gb/rendcondsoc-gb/in-focus-income-and-social/send/415-rendimento-e-condicoes-sociais-em-foco/16851-em-foco-inquerito-as-despesas-das-familias-idf-2022-2023.html).
