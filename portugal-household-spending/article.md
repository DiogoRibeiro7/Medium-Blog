# Basic Needs, Saving and Inflation by Income Group in Portugal, 1988–2025

*Three Eurostat sources read together: the share of household spending that goes to food, rent and energy, how that share differs across income fifths, how much households save, and how fast the prices of their baskets rose.*

A previous article compared Portuguese wages with a counterfactual indexed to inflation and productivity. Wages are one side of a household budget. This article looks at the other side: where consumption goes, how much of it is taken by items that are hard to reduce, and how that differs across the income distribution. We do not test a hypothesis. We define the measures, compute them from three public sources, and report what the sources can and cannot resolve.

## Definitions

Both the national accounts and the household budget surveys classify consumption by purpose using COICOP. Let `x(q,t,k)` be the consumption of income fifth `q` in year `t` on COICOP item `k`, and `X(q,t)` its total consumption. We define the set of core basic needs as

```
C = { CP01 food and non-alcoholic beverages,
      CP041 actual rent,
      CP043 maintenance and repair of the dwelling,
      CP044 water supply and other dwelling services,
      CP045 electricity, gas and other fuels }
```

and a wider set `W` that adds health (CP06), the running costs of personal transport (CP072, mostly fuel and maintenance, excluding vehicle purchases) and transport services (CP073). Neither set is canonical. They are fixed in advance so that any result can be recomputed under a different definition.

Imputed rent (CP042) is the rent owner-occupiers are estimated to pay themselves. National accounts include it so that consumption does not depend on the tenure structure of the housing stock, but it is not a cash outlay. Our basic-needs share therefore removes it from both numerator and denominator:

```
s_C(q,t) = Σ_{k ∈ C} x(q,t,k) / ( X(q,t) − x(q,t,CP042) )
```

Unless stated otherwise, every share below is defined this way. Shares that include imputed rent are labelled as such, and the difference between the two is large enough to change conclusions about housing.

The sources are Eurostat tables, downloaded by a script and committed as a snapshot in the [companion folder of the Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), together with the code that produces every number and figure:

- household final consumption by COICOP purpose (`nama_10_co3_p3`), annual, 1995–2022, with 2021 and 2022 provisional for Portugal;
- the household budget survey structure by income quintile (`hbs_str_t223`) and mean consumption per adult equivalent by quintile (`hbs_exp_t133`), seven waves from 1988 to 2020;
- the gross household saving rate (`nasa_10_ki`), 1995–2025;
- an experimental median saving rate by income quintile (`icw_sr_03`), 2010, 2015 and 2020;
- harmonised consumer price indices by COICOP (`prc_hicp_aind`), 1996–2025.

The national accounts give an annual series for all households together. The budget survey gives the distribution, but only every five or six years, and the published tables carry no standard errors, so differences of a point or two between fifths within a single wave should not be read as precise.

## All households, 1995–2022

For all households, `s_C` ranged from 23.7% (2000) to 30.4% (2020). It drifted down in the late 1990s, stayed between 24% and 25% through most of the 2000s, and rose to 28.1% by 2015. The 2020 maximum reflects the denominator as much as the numerator: between 2019 and 2020 spending on restaurants and hotels fell from €20.5 billion to €12.4 billion, on transport from €19.9 billion to €14.7 billion and on clothing from €8.5 billion to €7.3 billion, while food spending rose, so the essentials became a larger share of a smaller total. In 2022 the share was 27.3%.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig01_basic_needs_national_accounts.png)

*Figure 1. Core basic needs as a share of household consumption, imputed rent excluded, 1995–2022, with the wider set as a dashed line. Source: Eurostat nama_10_co3_p3; 2021–2022 provisional.*

Food accounts for most of the core share, between 17.1% and 21.3% of consumption. Actual rent never exceeds 4% in the aggregate and household energy stays between 2.7% and 4.2%. The wider set `s_W` moves between 38.5% and 44.8%. At this level of aggregation the share of consumption absorbed by basic needs changed by a few points over twenty-eight years.

Imputed rent moved more. It rose from 6.9% of total consumption in 1995 to 10.9% in 2015 and 12.3% in 2020, which is the main reason housing appears to grow in importance whenever it is measured with imputed rent included.

## The saving rate

The gross household saving rate fell from 14.3% of disposable income in 1995 to 10.0% in 2005 and reached its minimum, 6.5%, in 2017. Its mean was 12.0% over 1995–2007 and 8.1% over 2008–2019. The three largest annual increases came in 2020 (to 12.1%), 2009 (to 11.5%) and 2024 (to 12.9%, provisional); 2022 was 7.3% and the provisional figure for 2025 is 12.3%.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig02_saving_rate.png)

*Figure 2. Gross household saving rate, households and non-profit institutions serving households, % of disposable income, 1995–2025. Source: Eurostat nasa_10_ki; 2024–2025 provisional.*

This is a ratio of sector totals. It does not say which households saved, and a falling aggregate rate is consistent with many different distributions of saving across households.

## Food shares across seven survey waves

In every wave the food share falls with income, as Engel's law predicts. In 1988 the poorest fifth spent 47.5% of its consumption on food and the richest fifth 20.0%.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig03_food_engel_curves.png)

*Figure 3. Food and non-alcoholic beverages as a share of consumption, by income fifth, in each survey wave from 1988 to 2020 (imputed rent included in the total). Source: Eurostat hbs_str_t223.*

The curve shifted down and flattened between waves. The poorest fifth's food share fell to 29.0% by 1999, 22.2% by 2005 and about 17% from 2010. The distance between the poorest and richest fifth went from 27.5 points in 1988 to 7.1 points in 2020. Food alone would suggest that the gap in the burden of essentials between rich and poor households has almost closed.

## Core basic needs by income fifth

With rent, energy, water and repairs added and imputed rent removed, the poorest fifth's `s_C` was 58.8% in 1988 and 50.3% in 1994. It fell to 44.4% by 1999 and then stopped falling: 44.5% in 2005, 44.6% in 2010 and 48.6% in 2015.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig04_core_share_by_quintile.png)

*Figure 4. Core basic needs (food, actual rent, household energy, water and dwelling repairs) as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

The other fifths follow the same profile at lower levels. The richest fifth went from 27.2% in 1988 to a minimum of 21.1% in 2005 and back to 25.3% in 2015. The ratio of the poorest fifth's share to the richest fifth's was 2.2 in 1988 and 1.9 in 2015. With the wider set the poorest fifth's share was 69.3% in 1988 and 66.4% in 2015, about two thirds of its cash consumption in both years.

Figures 3 and 4 diverge after 1999 because food is no longer the dominant basic need at the bottom of the distribution.

## Rent and energy

Actual rent was 1.6% of the poorest fifth's consumption in 1988, 6.9% in 2005, 8.8% in 2010 and 11.1% in 2015. For the second fifth it rose from 1.7% to 5.5%, for the middle fifth from 1.7% to 3.2%, and for the two richest fifths it stayed between 1% and 2%.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig05_rent_share_by_quintile.png)

*Figure 5. Actual rent paid as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

These are mean shares across all households in each fifth, tenants and owners together. A rise can come from a larger proportion of tenants, from higher rents relative to other spending, or from both. The published tables do not separate these two components; that would require the survey microdata with tenure status. What the shares establish is where rent became a large budget item: in the bottom two fifths.

Electricity, gas and other fuels rose for every fifth and most at the bottom. For the poorest fifth they took 6.3% of consumption in 1988 and in 1999, then 8.1% in 2005, 10.7% in 2010 and 11.4% in 2015. For the richest fifth the share went from 2.8% to 6.2%.

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig06_energy_share_by_quintile.png)

*Figure 6. Electricity, gas and other fuels as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

Figure 7 decomposes `s_C` for each fifth. For the poorest fifth, food was 49.4% of cash consumption in 1988 and the other core items 9.4 points; in 2015 food was 22.0% and the other core items 26.6 points, of which rent and energy were 22.5. For the richest fifth, food fell from 21.5% to 14.4% and rent and energy together gained about three points.

![Figure 7](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig07_composition_by_quintile.png)

*Figure 7. Composition of core basic needs for each income fifth, % of consumption, imputed rent excluded, 1988–2015. Source: Eurostat hbs_str_t223.*

## Imputed rent and the 2020 wave

With imputed rent kept in the total, the housing division (rent actual and imputed, water, energy, repairs) rose from about 12% of every fifth's consumption in 1988 to between 28% and 39% in 2015, highest for the poorest fifth. Most of that increase is imputed rent, which went from 3.9% to 16.9% of the poorest fifth's consumption and from 6.8% to 19.5% of the richest fifth's. A measure of housing burden that includes imputed rent will rise with property valuations whether or not any household pays more.

![Figure 8](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig08_housing_incl_imputed_rent.png)

*Figure 8. Housing, water, electricity, gas and other fuels, imputed rent included, as a share of consumption, by income fifth, in all seven survey waves. The 2020 wave does not separate imputed rent. Source: Eurostat hbs_str_t223.*

The measures that exclude imputed rent stop in 2015 because Eurostat's 2020 tables for Portugal report the housing division without its components. The 2020 housing share also rises for every fifth, by 10 points for the richest (28.2% to 38.2%), where actual rent is about 1% of consumption. Two explanations are consistent with that pattern: a change in how imputed rent was estimated, or the compression of other spending during the pandemic. The published tables do not discriminate between them, so we treat the 2020 housing figures as unresolved.

## Consumption per adult equivalent

The survey also reports mean consumption per adult equivalent by fifth, in purchasing power standards (PPS). PPS levels are in current terms and are not comparable across waves, but ratios between fifths within a wave are. Let `c(q,t)` be consumption per adult equivalent and

```
a(q,t) = c(q,t) · (1 − x(q,t,CP042)/X(q,t)) − c(q,t) · Σ_{k ∈ C} x(q,t,k)/X(q,t)
```

the cash consumption left after core basic needs. Figure 9 plots `c(5,t)/c(1,t)` and `a(5,t)/a(1,t)`.

![Figure 9](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig09_consumption_ratio.png)

*Figure 9. Consumption per adult equivalent of the richest fifth divided by that of the poorest fifth, for all consumption and for cash consumption left after core basic needs. Source: Eurostat hbs_exp_t133 and hbs_str_t223.*

The ratio of total consumption was 4.6 in 1988, 4.9 in 1994, 2.8 in 2010 and 2.3 in 2020: by this measure, the gap between the top and bottom fifth roughly halved. The ratio for consumption left after core needs is larger and fell from 7.8 in 1988 to 5.0 in 1999 and 3.6 in 2015. In 2015, `a` was 42.7% of total consumption for the poorest fifth and 60.1% for the richest.

The two results coexist. Consumption became much less unequal between the top and bottom fifths, and over the same period the share of the bottom fifth's budget taken by core needs stopped falling and then rose.

## Saving by income fifth

Eurostat's experimental statistics on the joint distribution of income, consumption and wealth provide a median saving rate by income fifth, built by statistically matching survey sources. They exist for 2010, 2015 and 2020.

![Figure 10](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig10_saving_by_quintile.png)

*Figure 10. Median saving rate by income fifth, % of disposable income, 2010, 2015 and 2020. Experimental statistics. Source: Eurostat icw_sr_03.*

The median household in the poorest fifth had a saving rate of −12.2% in 2010 and −9.3% in 2015, meaning its estimated consumption exceeded its estimated income. The tables do not say how the difference was financed; drawing down savings, borrowing and transfers not recorded as income are all possible. The second fifth saved 6.5% and 8.1% in those years, the middle fifth 13.3% and 19.2%, and the richest fifth 27.3% and 37.2%. In 2020 every fifth's median rose, and the poorest fifth's became positive at 8.1%, in line with the aggregate increase in Figure 2. These medians are not comparable in level with the national-accounts saving rate, which is a ratio of totals with different definitions of income and consumption.

## Prices of the basic-needs basket

Figure 11 shows the HICP for each core item against all items, with 1996 = 100.

![Figure 11](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig11_basic_needs_prices.png)

*Figure 11. Harmonised consumer price indices for each core basic need and for all items, 1996 = 100. Source: Eurostat prc_hicp_aind.*

From 1996 to 2025, all items rose 86.6% and food 88.0%, with half of food's increase occurring after 2021. The housing-related items rose faster: actual rent 115.4%, electricity, gas and fuels 136.4%, dwelling repairs 153.7%, and water and other dwelling services 179.8%. Weighting the core items by their 2015 budget-survey shares gives a core basic-needs index that rose 33.6% between 2015 and 2025, against 24.8% for all items, and 28.7% against 20.5% between 2020 and 2025.

Because the fifths have different budget shares, they face different inflation rates. For fifth `q` we compute a fixed-weight (Laspeyres) index over the twelve COICOP divisions,

```
π(q; t0, t1) = Σ_d w(q,d) · P(d,t1)/P(d,t0) − 1,
```

with weights `w(q,d)` taken from one survey wave, imputed rent removed from the housing division (the HICP does not cover owner-occupied housing), and normalised to sum to one.

![Figure 12](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig12_inflation_by_quintile.png)

*Figure 12. Cumulative inflation of each income fifth's basket, HICP divisions weighted by budget-survey shares (2005 wave for 2005–2015, 2015 wave afterwards), imputed rent removed. Source: Eurostat prc_hicp_aind and hbs_str_t223.*

Over 2005–2015 the poorest fifth's basket rose 20.2% and the richest fifth's 15.8%, a difference of 4.4 points. Over 2015–2025 the figures are 26.6% and 24.4%, a difference of 2.2 points. In 2021–2023 the five baskets rose between 12.8% and 13.4%: when nearly all divisions rise together, the weights matter little. The differences between fifths accumulate in the years when the essentials rise slightly faster than the rest.

## Consequences for measurement and policy design

The quintile inflation rates bear directly on indexation. Pensions, social benefits and the minimum wage are usually discussed against headline inflation, and the all-items HICP rose 16.8% over 2005–2015 and 24.8% over 2015–2025, close to the survey-weighted average of 17.3% and 25.2%. An adjustment that tracks headline inflation therefore leaves the poorest fifth's basket rising faster than the adjustment: by our estimates, by 3.4 points over 2005–2015 (20.2% against 16.8%) and 1.8 points over 2015–2025 (26.6% against 24.8%). The difference is small in any single year. Over two decades it is not, and a basket-specific index can be built from the same public tables used here.

Imputed rent needs to be declared in any affordability measure. Figures 4 and 8 describe the same households and give different pictures of housing, and the difference grew as imputed rent went from 6.9% to more than 10% of aggregate consumption.

The distributional results rest on six survey waves for the shares and three years of experimental statistics for saving. The direction of the main findings is clear in that evidence: the end of the decline in the poorest fifth's basic-needs share after 1999, the rise of rent and energy at the bottom, and negative median saving in the poorest fifth in 2010 and 2015. Point estimates for individual waves are less secure, and the 2020 wave cannot yet be integrated with the earlier ones.

For low-income households, the composition of basic needs has changed. Food is a much smaller share of their consumption than in 1988, and rent and energy have taken most of that space, while the prices of the housing-related items rose faster than food over most of the period. A measure aimed at the essentials of low-income households is now largely a measure about rent and energy.

## Limitations

The core and wider sets are definitions; other choices would move the levels, though they are unlikely to reverse the ordering of the fifths. Budget-survey shares are means within each fifth and mix tenants and owners, so the rent results cannot be decomposed into tenure and price effects. The 2020 wave lacks the housing components and its housing jump is unexplained. The national-accounts detail by purpose for Portugal ends in 2022, with 2021–2022 provisional. The saving rates by fifth are experimental and available for three years. The inflation rates by fifth hold weights fixed for ten years, ignore substitution between items, and rely on an index that excludes owner-occupied housing. None of the published tables used here report sampling errors.

---

**Data and code.** [Companion folder in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), Apache-2.0. Sources: Eurostat [nama_10_co3_p3](https://ec.europa.eu/eurostat/databrowser/view/nama_10_co3_p3/default/table) (household consumption by purpose), [nasa_10_ki](https://ec.europa.eu/eurostat/databrowser/view/nasa_10_ki/default/table) (household saving rate), [hbs_str_t223](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t223/default/table) and [hbs_exp_t133](https://ec.europa.eu/eurostat/databrowser/view/hbs_exp_t133/default/table) (household budget survey), [icw_sr_03](https://ec.europa.eu/eurostat/databrowser/view/icw_sr_03/default/table) (experimental median saving rate by income quintile) and [prc_hicp_aind](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_aind/default/table) (consumer prices).
