# Basic Needs, Saving and Inflation by Income Group in Portugal, 1988–2025

*What Portuguese households spend on food, rent and energy, how that differs by income, household type and place, how much is left and saved, what it costs in euros, and how fast each group's basket has risen in price, read from Eurostat's national accounts, budget surveys and consumer price indices.*

A previous article compared Portuguese wages with a counterfactual indexed to inflation and productivity. Wages are one side of a household budget. This article looks at the other side: where consumption goes, how much of it is taken by items that are hard to reduce, and how that differs across the income distribution and across types of household. We do not test a hypothesis. We define the measures, compute them from public sources, check them against each other where two sources overlap, and report what the sources can and cannot resolve.

The analysis has three layers. The national accounts give an annual series for all households from 1995. The household budget survey gives the distribution, by income fifth and by household type, in seven waves from 1988. Consumer prices by item, weighted by each group's budget, give an inflation rate for each group. Two further sources, INE's published mean expenditure per household and the national minimum wage, turn some of the survey's purchasing-power amounts into euros.

## Definitions

Both the national accounts and the household budget surveys classify consumption by purpose using COICOP. Let `x(q,t,k)` be the consumption of group `q` in year `t` on COICOP item `k`, and `X(q,t)` its total consumption. A group is usually an income fifth; later sections use household type, age of the reference person and degree of urbanisation. We define the set of core basic needs as

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

We call `X − x_CP042` cash consumption. Unless stated otherwise, every share below is a share of cash consumption. Shares that include imputed rent are labelled as such, and the difference between the two is large enough to change conclusions about housing.

One further definition matters for the annual series. The national-accounts breakdown by purpose is on the domestic concept: it records consumption that takes place in Portugal, including spending by non-residents, and leaves out what residents spend abroad. Household consumption of residents, the national concept, is published only as a total. We return to the consequences below.

## Sources

The sources are Eurostat tables, downloaded by a script and committed as a snapshot in the [companion folder of the Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), together with the code that produces every number and figure:

- household final consumption by COICOP purpose (`nama_10_co3_p3`), annual, 1995–2022, in current prices and chain-linked volumes, with 2022 provisional for Portugal, and resident household consumption (`nama_10_gdp`);
- the household budget survey structure by income quintile (`hbs_str_t223`), by type of household (`hbs_str_t224`), by age of the reference person (`hbs_str_t225`) and by degree of urbanisation (`hbs_str_t226`), and mean consumption by income quintile (`hbs_exp_t133`), seven waves from 1988 to 2020. In Portugal the wave Eurostat labels 2020 was collected in 2022/23 (INE's IDEF 2022/2023); we write it as 2020 (2022/23);
- the same quintile table for Spain, Greece and Italy;
- the budget survey structure for all households (`hbs_str_t211`), used for the all-household weights;
- the gross household saving rate (`nasa_10_ki`), 1995–2025;
- an experimental median saving rate by income quintile (`icw_sr_03`), for three waves labelled around 2010, 2015 and 2020; for Portugal the third uses the 2022 budget survey;
- harmonised consumer price indices by COICOP (`prc_hicp_aind`), 1996–2025, their item weights (`prc_hicp_inw`) and purchasing power parities (`prc_ppp_ind`).

Two non-Eurostat inputs are added by hand, with their sources recorded in the data folder: INE's mean annual expenditure per household for the 2010/2011, 2015/2016 and 2022/2023 surveys, and the dates and amounts of every change in the national minimum wage from DGERT (with the 2000 value from PORDATA, which DGERT's table omits), the same file used in the wage article.

The budget survey gives the distribution only every five or six years, and the published tables carry no standard errors, so differences of a point or two between groups within a single wave should not be read as precise.

## All households, 1995–2022

For all households, `s_C` ranged from 23.7% (2000) to 30.4% (2020). It drifted down from 26.2% in 1995 to 23.7% in 2000, stayed between 24% and 26% until 2010, rose to 29.0% in 2013 and fell back to 25.6% by 2019. The 2020 maximum reflects the denominator as much as the numerator: between 2019 and 2020 spending on restaurants and hotels fell from €20.4 billion to €12.4 billion, on transport from €19.9 billion to €14.7 billion and on clothing from €8.5 billion to €7.3 billion, while food spending rose, so the essentials became a larger share of a smaller total. In 2022 the share was 27.3%.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig01_basic_needs_national_accounts.png)

*Figure 1. Core basic needs as a share of household consumption, imputed rent excluded, 1995–2022 (stacked areas, domestic concept), with core needs over resident households' consumption as a dotted line and the wider set (adding health, the running costs of transport and transport services) as a dashed line. Source: Eurostat nama_10_co3_p3 and nama_10_gdp; 2022 provisional.*

Food accounts for most of the core share, between 17.1% and 21.3% of consumption. Actual rent never exceeds 4% in the aggregate and household energy stays between 2.7% and 4.2%. The wider set `s_W` moves between 38.5% and 44.8%.

The domestic concept puts tourism in the denominator. The ratio of the COICOP total to resident household consumption was 1.035 in 1995, 1.041 in 2010, 1.075 in 2015 and 1.101 in 2019; it fell to 1.040 in 2020 and was back at 1.103 in 2022. The net spending of non-residents in Portugal, their purchases here minus residents' purchases abroad, went from 3.5% of resident consumption to about 10%. Between 2019 and 2020 the domestic total fell 11.5% and the resident total 6.4%.

We cannot split each COICOP item between residents and non-residents from the published tables, but we can bound the resident share. Imputed rent is entirely residents'. If we assume all spending on core needs is also residents', the resident-basis share is

```
s_C^res(t) = Σ_{k ∈ C} x(t,k) / ( X_res(t) − x(t,CP042) )
```

with `X_res` resident household consumption. This is an upper bound when non-residents buy more food and energy in Portugal than residents buy abroad, which is the likely direction for a country with net tourism receipts. The domestic share is a lower bound when the net spending of non-residents is less concentrated on core items than residents' own consumption, which is also likely, since much of it goes to hotels and restaurants. The resident share lies between the two under those two assumptions.

The upper bound is the dotted line in Figure 1. It was 27.2% in 1995, 24.8% in 2000, 25.7% in 2008, 30.9% in 2013, 28.6% in 2019, 31.8% in 2020 and 30.4% in 2022. The two lines are about one point apart until 2010 and three points apart in 2019 and 2022. Part of the decline in the domestic share after 2013 is therefore the growth of tourism in the denominator: on the resident bound the share fell 2.3 points between 2013 and 2019, not 3.4, and in 2022 it was above its 1995 level. What survives on both measures is a low point in 2000, a rise of five to six points to 2013, and a level since 2020 above anything recorded before 2012.

Imputed rent moved more than any core item. It rose from 6.9% of total consumption in 1995 to 10.9% in 2015 and 12.3% in 2020. Part of the movement is the denominator: between 2019 and 2020 the share went from 10.5% to 12.3% while imputed rent itself rose 3.7% and total consumption fell 11.5%.

## Price and volume

A rise in `s_C` can come from core needs becoming dearer relative to everything else, or from households buying relatively more of them. Write the core aggregate and the cash aggregate as `X_C` and `X_A`, and let `P_C` and `P_A` be chained Törnqvist price indices built from the HICP indices of their components, weighted by the national-accounts expenditure shares of adjacent years. Defining volume as nominal spending deflated by its price index, the share factors exactly as

```
log( s_C(t1) / s_C(t0) ) = log( (P_C/P_A)(t1) / (P_C/P_A)(t0) ) + log( (Q_C/Q_A)(t1) / (Q_C/Q_A)(t0) ),
```

with `Q = X / P`. The first term is the relative price of core needs, the second their relative volume. The volume term is a residual, so it also absorbs any mismatch between HICP coverage and national-accounts spending.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig02_core_share_price_volume.png)

*Figure 2. Change in the core basic-needs share of cash consumption, all households, split into the relative price and the relative volume of core needs, log points. Source: Eurostat nama_10_co3_p3 (current prices, domestic concept) and prc_hicp_aind.*

From 1996 to 2008 the share fell 5.0 log points, 2.1 from prices and 3.0 from volume. From 2008 to 2013 it rose 16.2 log points, of which only 2.4 came from relative prices; the remaining 13.8 are relative volume, meaning that spending on everything else fell in real terms much more than spending on food, rent and energy. From 2013 to 2019 the share fell 12.3 log points with no change in relative prices (−0.2). In 2019–2020 relative volume added 15.6 log points as restaurants, hotels and transport collapsed, and in 2020–2022 it removed 15.3 as they recovered, while relative prices added 4.4. Over the whole period, 1996 to 2022, the share rose 5.1 log points: relative prices contributed 6.2 and relative volume −1.0.

The movements within the period are therefore mostly volume: what happened to the rest of the budget during the 2008–2013 contraction, its recovery, and the pandemic. The movement across the whole period is mostly price. Because the denominator is on the domestic concept, part of the volume swings in 2013–2019 and 2019–2022 belongs to non-residents. The decomposition is an accounting identity and identifies contributions, not causes.

One inconvenient finding belongs here. The national accounts also publish chain-linked volumes by purpose, and the implicit deflators they imply differ sharply from the HICP for food. From 1996 to 2020 the implicit food deflator rose 8% while the HICP food index rose 43%, and the implied food volume rose 95%. For all consumption the two price measures are close (60% and 55%). With the national-accounts volumes, the same decomposition attributes −23.3 log points to relative prices over 1995–2008. We do not know the source of the food divergence and do not use the national-accounts volumes in what follows.

## The saving rate

The gross household saving rate fell from 14.3% of disposable income in 1995 to 10.0% in 2005 and reached its minimum, 6.5%, in 2017. Its mean was 12.0% over 1995–2007 and 8.1% over 2008–2019. The three largest annual increases came in 2020 (to 12.1%), 2009 (to 11.5%) and 2024 (to 12.9%); 2022 was 7.3%, 2023 8.9%, and the provisional figure for 2025 is 12.3%.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig03_saving_rate.png)

*Figure 3. Gross household saving rate, households and non-profit institutions serving households, % of disposable income, 1995–2025. Source: Eurostat nasa_10_ki; 2025 provisional.*

This is a ratio of sector totals. It does not say which households saved, and a falling aggregate rate is consistent with many different distributions of saving across households. The 2009 and 2020 increases coincide with the two volume episodes in Figure 2, when households cut spending outside core needs; the saving rate and the core share moved up together in both.

## Food shares across seven survey waves

In every wave the food share falls with income, as Engel's law predicts. In 1988 the poorest fifth spent 47.5% of its consumption on food and the richest fifth 20.0%.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig04_food_engel_curves.png)

*Figure 4. Food and non-alcoholic beverages as a share of consumption, by income fifth, in each survey wave from 1988 to 2020 (2022/23), imputed rent included in the total. Source: Eurostat hbs_str_t223.*

The curve shifted down and flattened between waves. The poorest fifth's food share fell to 29.0% by 1999 and 22.2% by 2005, and was between 17.1% and 18.3% in the three later waves. The distance between the poorest and richest fifth went from 27.5 points in 1988 to 7.1 points in 2020 (2022/23). Food alone would suggest that the gap in the burden of essentials between rich and poor households has almost closed.

## Engel curves item by item

Figure 4 shows one item. To compare items and waves on one scale we fit a Working–Leser Engel curve within each wave, across the five fifths:

```
w_k(q,t) = a_k(t) + b_k(t) · log c(q,t),        q = 1, …, 5,
```

where `w_k` is item `k`'s share of cash consumption and `c` is cash consumption per adult equivalent in PPS. Since spending on the item is `w_k · c`, its budget elasticity is

```
η_k(t) = 1 + b_k(t) / w̄_k(t),
```

evaluated at the mean share `w̄_k`. An item with `η < 1` takes a smaller share as total spending rises (a necessity in the usual terminology) and one with `η > 1` a larger share. Each fit uses five points, the means of the fifths, so the slopes describe the cross-section of group means. They are not estimates with a sampling distribution, and the R² values, above 0.8 in about seven fits in ten, measure how close the five points lie to a line, not precision.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig05_engel_elasticities.png)

*Figure 5. Budget elasticity at the mean share from Working–Leser fits across the five income fifths, one dot per survey wave, 1988–2015, cash shares and cash consumption per adult equivalent. Source: Eurostat hbs_str_t223 and hbs_exp_t133.*

Food is the textbook necessity and stays one. Its slope fell from −19.1 points of budget share per unit of log consumption in 1988 to −8.6 in 2015, which is the flattening in Figure 4, but the mean share fell in proportion and the elasticity stayed between 0.43 and 0.55 in every wave. Energy is a stronger necessity than food in the later waves, with an elasticity of 0.47 in 1988 and 0.33–0.34 in 2010 and 2015. Health stays between 0.36 and 0.65.

Rent behaves differently from everything else. Its elasticity was 0.79 in 1988, 0.57 in 1994 and 0.42 in 1999, then −0.52 in 2005, −0.74 in 2010 and −1.24 in 2015. A negative elasticity means that mean rent per adult equivalent in euros falls as total spending rises across the fifths. This is not a demand elasticity for housing services. Rent is paid only by tenants, the share of tenants falls with income, and the fifths' mean rent mixes the two. The negative sign is consistent with tenancy becoming concentrated in the poorer fifths from 2005, strongly enough to dominate; poorer tenants paying more rent per adult equivalent would produce the same sign, and the published tables do not separate the two. The cash housing aggregate (rent, energy, water and repairs) went from an elasticity of 0.68 in 1988 to 0.19–0.20 in 2005 and 2010 and 0.04 in 2015. In 2015/16 the poorest fifth spent about €141 a month per adult equivalent on these items and the richest fifth about €145; the middle fifth spent €114.

At the other end, education (2.27–2.88) has the highest elasticity in every wave and vehicle purchases (1.75–2.44) the second highest, level with recreation in 1999. Recreation (1.52–1.95) and restaurants and hotels (1.11–1.64) are also luxuries in every wave. Two items changed category. Clothing went from 1.16 in 1988 to 1.81 in 2015, as the poorest fifths cut it most. Communications went from 1.27 in 1988 to 0.68 in 2015: per adult equivalent the richest fifth spent 6.7 times what the poorest spent on telephones and internet access in 1988 and 1.9 times in 2015, about €45 a month against €24.

## Core basic needs by income fifth

With rent, energy, water and repairs added and imputed rent removed, the poorest fifth's `s_C` was 58.8% in 1988 and 50.3% in 1994. It fell to 44.4% by 1999, was flat at 44.5% in 2005 and 44.6% in 2010, and was higher, 48.6%, in 2015. The rise after 2010 rests on a single wave.

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig06_core_share_by_quintile.png)

*Figure 6. Core basic needs (food, actual rent, household energy, water and dwelling repairs) as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

The other fifths follow a similar profile at lower levels, with their minimum in 2005 rather than 1999. The richest fifth went from 27.2% in 1988 to a minimum of 21.1% in 2005 and back to 25.3% in 2015. The ratio of the poorest fifth's share to the richest fifth's was 2.2 in 1988 and 1.9 in 2015. With the wider set the poorest fifth's share was 69.3% in 1988 and 66.4% in 2015, about two thirds of its cash consumption in both years.

Figures 4 and 6 diverge after 1999 because, at the bottom of the distribution, the other core needs grew as food fell.

## Rent and energy

Actual rent was 1.6% of the poorest fifth's consumption in 1988, 6.9% in 2005, 8.8% in 2010 and 11.1% in 2015. For the second fifth it rose from 1.7% to 5.5%, for the middle fifth from 1.7% to 3.2%, and for the two richest fifths it stayed around 1–2%.

![Figure 7](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig07_rent_share_by_quintile.png)

*Figure 7. Actual rent paid as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

These are mean shares across all households in each fifth, tenants and owners together. A rise can come from a larger proportion of tenants, from higher rents relative to other spending, or from both. The published tables do not separate these two components; that would require the survey microdata with tenure status. What the shares establish is where rent became a large budget item: in the bottom two fifths. The negative rent elasticities of Figure 5 are the same fact seen through a regression.

Electricity, gas and other fuels rose for every fifth and most at the bottom. For the poorest fifth they took 6.3% of consumption in 1988 and in 1999, then 8.1% in 2005, 10.7% in 2010 and 11.4% in 2015. For the richest fifth the share went from 2.8% to 6.2%.

![Figure 8](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig08_energy_share_by_quintile.png)

*Figure 8. Electricity, gas and other fuels as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

Figure 9 decomposes `s_C` for each fifth. For the poorest fifth, food was 49.4% of cash consumption in 1988 and the other core items 9.4 points; in 2015 food was 22.0% and the other core items 26.6 points, of which rent and energy were 22.5. For the richest fifth, food fell from 21.5% to 14.4% and rent and energy together gained about three points.

![Figure 9](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig09_composition_by_quintile.png)

*Figure 9. Composition of core basic needs for each income fifth, % of consumption, imputed rent excluded, 1988–2015. Source: Eurostat hbs_str_t223.*

## The rest of the budget

What is not a basic need is, by construction, the room a household has. We group the remaining divisions as discretionary: alcohol and tobacco, clothing, furnishings, vehicle purchases, communications, recreation, education, restaurants and hotels, and miscellaneous goods and services. Health and transport running costs and services sit in the wider set and are excluded here, so for each fifth core needs, health, transport running costs and services, and the discretionary items add up to cash consumption, up to rounding in the published components.

![Figure 10](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig10_discretionary_by_quintile.png)

*Figure 10. Discretionary items as a share of cash consumption, poorest and richest fifth, 1988–2015, with the total above each bar. Source: Eurostat hbs_str_t223, imputed rent excluded.*

For the poorest fifth the discretionary share was 30.7% in 1988, rose to 41.7% in 1999 and fell to 37.1% in 2005, 36.7% in 2010 and 33.7% in 2015. For the richest fifth it was 59.0% in 1988, peaked at 63.9% in 1999 and fell to 55.4% in 2015. Both fifths reached their maximum in the same wave, 1999, the same wave in which the poorest fifth's core share reached its minimum.

The composition changed more than the total. In the poorest fifth, restaurants and hotels went from 4.8% in 1988 to 10.6% in 1999 and 10.9% in 2005, then back to 7.5% in 2015. Clothing fell from 7.5% to 2.4% and communications rose from 0.8% to 4.5%. Vehicle purchases reached 3.0% in 1999 and were 1.6% in 2015. Health, outside the discretionary set, went from 5.6% in 1988 to 12.3% in 2005 and 9.3% in 2015. In the richest fifth vehicle purchases fell from 10.5% to 6.3%, education rose from 1.2% to 4.2% and recreation peaked at 9.4% in 2005.

Between 1999 and 2015 the poorest fifth's core share rose 4.2 points, health and transport running costs and services together rose 4.0, and the discretionary share fell by about eight. The items that gave way were clothing (−4.3 points), restaurants and hotels (−3.1), furnishings (−1.8), vehicle purchases (−1.4) and alcohol and tobacco (−1.2), partly offset by miscellaneous goods and services (+2.7) and communications (+0.8).

## Household type, age and place

The budget survey publishes the same structure by type of household, by age of the reference person and by degree of urbanisation, without income splits. These groups differ in income as well as in type, so their differences are descriptive and do not isolate the effect of age or of living in a city.

![Figure 11](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig11_household_types.png)

*Figure 11. Core basic needs as a share of cash consumption by type of household, age of the reference person and degree of urbanisation, 1988–2015. Source: Eurostat hbs_str_t224, hbs_str_t225 and hbs_str_t226, imputed rent excluded.*

Households whose reference person is 60 or older have the highest core share in every wave: 46.3% in 1988, 37.6% in 1999, 34.1% in 2005 and 37.9% in 2015. Their food share fell from 38.2% to 20.6% while energy went from 5.0% to 9.8% and health from 5.2% to 13.0% in 2005 and 10.1% in 2015. Households under 30 have the lowest food share in 2015 (13.8%) and the highest rent share, 8.5%, up from 4.9% in 2010; in no other age group does rent exceed 3.6%.

By type of household, two adults with children have the lowest core share from 1994 on, 29.2% in 2015, and single adults and two-adult households without children the highest, 36.0% and 35.5%. Single adults with children have the highest rent share, 7.0% in 2010 and 6.7% in 2015. Single adults spend more on energy (8.9%) and on health (8.8%) than any other type, and in the 2020 (2022/23) wave their housing division, imputed rent included, was 47.4% of consumption.

Rural households spend more on core needs than urban ones in every wave: 42.5% against 35.9% in 1988 and 35.8% against 31.5% in 2015. In 2015 they spent less on rent (1.6% against 4.0% in cities) and more on energy (10.3% against 6.6%) and food (20.1% against 16.9%). The housing division including imputed rent tells a different story in the latest wave: from 2015 to 2020 (2022/23) it rose from 31.8% to 41.7% in cities and from 32.8% to 34.8% in rural areas. That pattern is consistent with imputed rents following property values, which rose more in cities, but the published tables do not show imputed rent for that wave and we treat it as a conjecture.

One regularity holds across all thirteen groups. Every household type, every age group and every degree of urbanisation had a higher core share in 2015 than in 2005, as did every income fifth. The rise ranges from 1.1 points (under 30) to 6.3 points (single adults with children). A change that appears in every group at once is consistent with something common to all of them, such as prices, and less so with a change in the mix of households; two waves are not enough to establish which.

## Portugal next to Spain, Greece and Italy

Eurostat publishes the quintile structure for other countries with the same classification. Figure 12 compares Portugal with three southern EU countries that share some features of its housing market and income level. The Italian series ends in 2005, and an EU aggregate for this breakdown was not returned by Eurostat's interface.

![Figure 12](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig12_peer_countries.png)

*Figure 12. Core basic needs of the poorest fifth (solid) and richest fifth (dashed), left, and actual rent of the poorest fifth, right, as a share of cash consumption, Portugal, Spain, Greece and Italy, all waves with the housing components. Source: Eurostat hbs_str_t223, imputed rent excluded.*

In 1988 Portugal had the highest core share in the poorest fifth (58.8%, against 55.9% in Italy, 46.2% in Greece and 46.1% in Spain) and the widest ratio between the poorest and richest fifth, 2.16 against 1.54 in Spain and 1.65 in Greece. By 2015 the poorest fifth's shares were close in the three countries with data, 48.6% in Portugal, 47.7% in Spain and 49.0% in Greece, while Portugal's ratio, 1.92, was still the widest (1.79 in Spain, 1.74 in Greece). In the 2020 wave Spain and Greece reached 52.6% and 52.1% in the poorest fifth and their richest fifths rose more, to 31.9% and 33.0%. Portugal has no comparable 2020 figure. The reference periods of the 2020 wave differ by country, and for Spain and Greece we have not checked how they relate to the pandemic year, so we do not interpret the level of that wave.

The composition differs more than the total. In 2015 Portugal's poorest fifth spent less of its budget on food than its Spanish and Greek counterparts (22.0% against 24.2% and 26.3%) and much more on energy (11.4% against 6.6% and 8.8%). Rent rose at the bottom in all four countries: from 1.3% in 1988 to 12.3% in 2015 and 16.2% in 2020 in Spain, from 2.8% to 12.1% in 2010 and 10.5% in 2015 in Greece, from 1.6% to 11.1% in 2015 in Portugal, and from 3.9% to 8.4% by 2005 in Italy. The rise of rent in the poorest fifth's budget is not specific to Portugal. The energy share was, in 2015, against Spain and Greece; in Italy's last wave, 2005, its poorest fifth spent 9.3% on energy against Portugal's 8.1%.

## Imputed rent and the 2020 wave

With imputed rent kept in the total, the housing division (rent actual and imputed, water, energy, repairs) rose from about 12% of every fifth's consumption in 1988 to between 28% and 39% in 2015, highest for the poorest fifth. Imputed rent accounts for half of the poorest fifth's increase (13.0 of 26.2 points) and for two thirds to four fifths of the other fifths' increases. It went from 3.9% to 16.9% of the poorest fifth's consumption and from 6.8% to 19.5% of the richest fifth's. A housing-burden measure that includes imputed rent can therefore rise without any household paying more.

![Figure 13](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig13_housing_incl_imputed_rent.png)

*Figure 13. Housing, water, electricity, gas and other fuels, imputed rent included, as a share of consumption, by income fifth, in all seven survey waves. The 2020 wave, collected in 2022/23, does not separate imputed rent. Source: Eurostat hbs_str_t223.*

The measures that exclude imputed rent stop in 2015 because Eurostat's tables for the 2020 (2022/23) wave report the housing division without its components. In that wave the housing share rises for every fifth, by 10 points for the richest (28.2% to 38.2%), where actual rent is about 1% of consumption. The comparison spans seven years, the 2022 rise in energy prices visible in Figure 17 and whatever happened to the property valuations that feed imputed rent; a change in how imputed rent was estimated is another possibility. The urban–rural split in the previous section points towards valuations, but the published tables do not discriminate between these explanations, so we treat the 2020 (2022/23) housing figures as unresolved.

## Consumption per adult equivalent

The survey also reports mean consumption per adult equivalent by fifth, in purchasing power standards (PPS). PPS levels are in current terms and are not comparable across waves, but ratios between fifths within a wave are. Let `c(q,t)` be consumption per adult equivalent and

```
a(q,t) = c(q,t) · (1 − x(q,t,CP042)/X(q,t)) − c(q,t) · Σ_{k ∈ C} x(q,t,k)/X(q,t)
```

the cash consumption left after core basic needs. Figure 14 plots `c(5,t)/c(1,t)` and `a(5,t)/a(1,t)`.

![Figure 14](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig14_consumption_ratio.png)

*Figure 14. Consumption per adult equivalent of the richest fifth divided by that of the poorest fifth, for all consumption and for cash consumption left after core basic needs. Source: Eurostat hbs_exp_t133 and hbs_str_t223.*

The ratio of total consumption was 4.6 in 1988, 4.9 in 1994, 2.8 in 2010 and 2.3 in 2020 (2022/23): by this measure, the gap between the top and bottom fifth roughly halved. The ratio for consumption left after core needs is larger and fell from 7.8 in 1988 to 5.0 in 1999 and 3.6 in 2015. In 2015, `a` was 51.4% of cash consumption for the poorest fifth, 59.4% for the second, 65.0% for the middle, 68.7% for the fourth and 74.7% for the richest.

The two results coexist. Consumption became much less unequal between the top and bottom fifths, and over the same period the share of the bottom fifth's budget taken by core needs stopped falling after 1999 and was higher in 2015.

## In euros

The question that started this article was a basket: what households spend on basic needs and what remains. Shares answer it in relative terms. To give amounts, we convert the survey's PPS means into euros. INE publishes the mean annual expenditure per household for each survey: €20,391 in 2010/2011, €20,916 in 2015/2016 and €24,190 in 2022/2023. Dividing by Eurostat's mean in PPS for the same wave gives a conversion factor, which we apply to every fifth:

```
f(t) = E_INE(t) / c_PPS(TOTAL, t),        amount(q,t) = f(t) · c_PPS(q,t).
```

The factors are 0.873 euro per PPS for 2010/2011 and 0.842 for 2015/2016. Eurostat's purchasing power parity for actual individual consumption, averaged over the same fieldwork years, is 0.868 and 0.844, within 0.6% of both, which suggests the survey tables were converted with that parity and that INE's total and Eurostat's mean measure the same thing. For 2022/2023 the factor is 0.987 and matches no parity year we checked (0.881 in 2020, 0.852 in 2022, 0.846 in 2023). We still use INE's total for that wave, but its amounts are less secure. All amounts are monthly, the annual figure divided by twelve.

![Figure 15](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig15_euro_amounts.png)

*Figure 15. Cash consumption per household and month by income fifth, 2015/16, split into core basic needs and everything else, approximate euros, with the national minimum wage spread over twelve months. Source: Eurostat hbs_exp_t133 and hbs_str_t223, INE, DGERT.*

In 2015/16 the average household in the poorest fifth had cash consumption of about €664 a month, of which €323 went to core needs and €341 to everything else. For the middle fifth the figures were €1,235, €432 and €803, and for the richest fifth €2,508, €636 and €1,873. Core spending per household roughly doubles from the poorest to the richest fifth while the rest of cash consumption grows more than five-fold.

Per adult equivalent the core amounts are closer still: €258 a month in the poorest fifth, €261 in the middle fifth and €336 in the richest, against €272, €486 and €990 left after core needs. In 2010/11 the poorest fifth spent €213 per adult equivalent on core needs and the richest €314.

The minimum wage gives a familiar scale. It is paid fourteen times a year, so we spread it over twelve months; averaged over the fieldwork years it was €480 a month (€560 spread) in 2010/11 and €517.50 (€604 spread) in 2015/16. The poorest fifth's household core basket was 47.9% of the spread minimum wage in 2010/11 and 53.5% in 2015/16, while its total consumption was 131% and 132%. These are not statements about any household's income, which in the poorest fifth comes from pensions and transfers as much as from wages. They locate the basket on a scale most readers know.

The basket can be carried forward at fixed quantities. Pricing the poorest fifth's 2015/16 core basket with the HICP indices of its five components gives about €431 a month in 2025, an increase of 33.5%. Over the same years the minimum wage rose 72.3%, to €870, or €1,015 spread over twelve months, so the fixed basket would be 42.5% of it. One assumption understates the basket: the HICP rent index follows the rents paid by sitting tenants, including long-standing contracts, so it understates what a household signing a new lease faced. Holding quantities at their 2015/16 level can err in either direction; it ignores substitution away from items that became dearer, but also any increase in the number of poorer households paying rent. For the 2022/23 wave, where only the housing division is published, the poorest fifth's total consumption was about €1,038 per household and month, 121% of the spread minimum wage, of which about €177 was food.

## Saving by income fifth

Eurostat's experimental statistics on the joint distribution of income, consumption and wealth provide a median saving rate by income fifth. They are built by statistical matching of the budget survey (consumption), EU-SILC (income) and the Household Finance and Consumption Survey (wealth), which produces a synthetic dataset rather than linked records. They exist for three waves, labelled around 2010, 2015 and 2020; for Portugal the third uses the 2022 budget survey.

![Figure 16](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig16_saving_by_quintile.png)

*Figure 16. Median saving rate by income fifth, % of disposable income, waves around 2010, 2015 and 2020 (the last based on the 2022 budget survey for Portugal). Experimental statistics. Source: Eurostat icw_sr_03.*

The median household in the poorest fifth had a saving rate of −12.2% in 2010 and −9.3% in 2015, meaning its estimated consumption exceeded its estimated income. The tables do not say how such a gap would be financed; drawing down savings, borrowing and transfers not recorded as income are all possible. A negative median can also be produced by the matching itself, because income and consumption come from different surveys with different reference periods. The 2010 wave counts non-monetary consumption and the later waves do not, so 2010 is not strictly comparable with 2015. The second fifth saved 6.5% and 8.1% in those waves, the middle fifth 13.3% and 19.2%, and the richest fifth 27.3% and 37.2%. In the third wave every fifth's median is higher and the poorest fifth's is positive at 8.1%. We do not compare these medians with Figure 3: they are medians within fifths from matched survey data, while the national-accounts rate is a ratio of sector totals with different definitions of income and consumption.

## Prices of the basic-needs basket

Figure 17 shows the HICP for each core item against all items, with 1996 = 100.

![Figure 17](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig17_basic_needs_prices.png)

*Figure 17. Harmonised consumer price indices for each core basic need and for all items, 1996 = 100. Source: Eurostat prc_hicp_aind.*

From 1996 to 2025, all items rose 86.6% and food 88.0%, with half of food's increase occurring after 2021. The housing-related items rose faster: actual rent 115.4%, electricity, gas and fuels 136.4%, dwelling repairs 153.7%, and water and other dwelling services 179.8%. The relative movement depends on the period. Weighting the core items by budget-survey shares, the core basic-needs index rose less than all items from 1996 to 2005 (24.4% against 28.0%, 1999 weights) and more afterwards: 23.4% against 16.8% over 2005–2015 and 33.6% against 24.8% over 2015–2025. Within the last decade dwelling repairs rose most (42.8%), followed by food (38.8%) and actual rent (36.4%), with electricity, gas and fuels at 20.7%. This is the relative-price term of Figure 2 seen from the price side: negative before the mid-2000s, positive after.

## Inflation by income fifth

Because the fifths have different budget shares, they face different inflation rates. For fifth `q` we compute a fixed-weight (Laspeyres) index over the twelve COICOP divisions,

```
π(q; t0, t1) = Σ_d w(q,d) · P(d,t1)/P(d,t0) − 1,
```

with weights `w(q,d)` taken from one survey wave, imputed rent removed from the housing division (the HICP does not cover owner-occupied housing), and normalised to sum to one.

![Figure 18](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig18_inflation_by_quintile.png)

*Figure 18. Cumulative inflation of each income fifth's basket, HICP divisions weighted by budget-survey shares (2005 wave for 2005–2015, 2015 wave afterwards), imputed rent removed. Source: Eurostat prc_hicp_aind and hbs_str_t223.*

Over 2005–2015 the poorest fifth's basket rose 20.2% and the richest fifth's 15.8%, a difference of 4.4 points. Most of it opened in 2011–2014: the gap was 1.0 point in 2010 and 4.0 in 2014, a period in which electricity, gas and fuel prices rose 28.9% (2010 to 2013). Over 2015–2025 the baskets rose 26.6% and 24.4%, a difference of 2.2 points, which was slightly negative in 2017–2019 and opened mostly, about four fifths of it, from 2022.

A fixed-weight index over ten years ignores how budgets change. To cover the whole price series and to test the dependence on weights, we also chain the index from 1996, updating each fifth's weights at every survey wave: 1999 weights for 1996–2005, 2005 weights for 2005–2010, 2010 weights for 2010–2015 and 2015 weights after. Within each segment the index is Laspeyres; at the segment boundaries the levels are linked.

![Figure 19](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig19_chained_inflation_gap.png)

*Figure 19. Left: difference between the chained basket indices (1996 = 100) of the poorest and richest fifth, and between the poorest fifth and the headline HICP. Right: the poorest-minus-richest difference in 2025 under the chained weights and with weights fixed at a single wave. Source: Eurostat prc_hicp_aind and hbs_str_t223.*

From 1996 to 2025 the poorest fifth's chained index rose 92.4%, the richest fifth's 86.0% and the headline HICP 86.6%. The difference was not always in the same direction. Until 2001 the two indices were within half a point of each other. From 2002 to 2011 the richest fifth's index was ahead, by up to 2.4 points in 2006; the difference turned positive in 2012 and reached 2.4 points in 2015, 3.1 in 2020, 4.8 in 2022 and 6.3 in 2025. Against the headline index the poorest fifth's basket was 1.8 points behind in 2005 and 5.8 ahead in 2025.

The right panel shows how much the 2025 difference depends on which budgets are used. With weights fixed at the 1999 wave it is 5.7 points, at 2005 7.7, at 2010 4.7 and at 2015 7.7, against 6.3 chained. The sign is the same under all five weightings; the size ranges from 4.7 to 7.7 points. We report the chained value as the central estimate and the range as its sensitivity to the weights, not as a confidence interval.

## The headline index in 2021–2023

Between 2021 and 2023 prices did not move together: food rose 24.3%, restaurants and hotels 26.6%, electricity, gas and fuels 16.9%, transport 10.3%, clothing 1.5% and health 1.0%. The five fifths' baskets, with 2015 weights, rose between 12.8% and 13.4%, all less than the headline HICP (13.8%). They ended close to each other because their weights offset: food is 22.0% of the poorest fifth's cash budget and 14.4% of the richest fifth's, while restaurants and hotels are 7.5% and 13.0%, and transport 10.1% and 19.8%.

The headline index rose more than every fifth's basket because its weights differ from residents' budgets. Applied to the 2021–2023 price changes, the HICP's own 2023 division weights give 13.9%, close to the published 13.8%, and the survey's all-household cash weights give 13.0%. Writing `w^H` and `w^S` for the two weight vectors, `r_d` for the price relative of division `d` and `r^S` for the survey basket's relative, the difference is

```
Σ_d (w^H_d − w^S_d) · r_d = Σ_d (w^H_d − w^S_d) · (r_d − r^S),
```

since both weight vectors sum to one. A division adds to the gap when the HICP gives it more weight than the survey and its price rose faster than the survey basket, or less weight and its price rose more slowly.

![Figure 20](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig20_hicp_weights.png)

*Figure 20. Left: HICP division weights in 2023 against the all-household cash budget shares of the 2015 survey. Right: each division's contribution to the difference between the two baskets' 2021–2023 price change, percentage points, with the division's price change. Source: Eurostat prc_hicp_inw, prc_hicp_aind and hbs_str_t211.*

Of the 0.87-point difference, restaurants and hotels contribute 0.43 (a weight of 14.2% in the HICP against 11.0% in the survey, and a price rise of 26.6%) and food 0.32 (20.6% against 17.9%, price +24.3%). The other ten divisions net to +0.12. Education, transport, communications, health and housing add between 0.07 and 0.10 each, because the HICP weights them less than the survey and their prices rose less than the basket; furnishings and recreation add 0.05 and 0.02; clothing, miscellaneous goods and services and alcohol and tobacco subtract 0.17, 0.12 and 0.09, because the HICP weights them more and their prices rose less.

The HICP is a domestic-concept index: its weights come from consumption in Portugal, including by non-residents. The higher weight of restaurants and hotels is consistent with the tourism share documented in the national accounts above. The comparison also mixes years, 2023 HICP weights against 2015 survey budgets, and the restaurants weight in the HICP itself went from 10.2% in 2022 to 14.2% in 2023. We can attribute the 2021–2023 gap to specific weights; we cannot say how much of each weight difference is tourism and how much is the passage of eight years.

## Consequences for measurement and policy design

The quintile inflation rates bear on indexation. The all-items HICP rose 16.8% over 2005–2015 and 24.8% over 2015–2025, close to the survey-weighted averages of 17.3% and 25.2%. Chaining the two fixed-weight decades, the poorest fifth's basket rose 52.2% over 2005–2025, the richest fifth's 44.1% and the headline index 45.8%. Any adjustment that tracks headline inflation would, on these estimates, have fallen 6.4 points behind the poorest fifth's basket over twenty years; the chained index from 1996 gives 5.8 points over twenty-nine. Whether particular benefits or wages were in fact indexed this way is outside what this analysis examines; the point is that a basket-specific index can be built from the same public tables used here, and that over horizons of a decade or more its difference from the headline index is several points.

The headline index can also diverge from residents' baskets for reasons unrelated to income. In 2021–2023 it rose more than any fifth's basket because its weights differ from residents' budgets, most of all in restaurants and hotels, where non-residents' spending is likely to be concentrated. In a country where that spending is about a tenth of residents' consumption, an index meant to measure the cost of living of residents should be checked against resident weights before it is used for that purpose.

Imputed rent needs to be declared in any affordability measure. In 2015 the poorest fifth's housing costs excluding imputed rent (actual rent, energy, water and repairs) were 26.6% of its cash consumption, while the housing division including imputed rent was 39.1% of its total consumption. A housing-cost ratio that does not say which of the two it uses is ambiguous by more than ten points. The same applies to the aggregate series and the domestic concept: the national-accounts core share in 2022 was 27.3% with non-residents' spending in the denominator and up to 30.4% without it.

For targeting, the Engel elasticities give a compact summary. Food is a necessity with a stable elasticity, so its weight in a budget falls predictably with income; energy is a stronger necessity whose elasticity fell from about 0.5 to 0.34. Rent does not behave as an ordinary good in the cross-section: its euro amount falls with total spending across the fifths. If, as seems likely, this reflects tenants being concentrated at the bottom, its burden depends on tenure as much as on income, and measures aimed at the essentials of low-income households that are defined by income alone will miss that structure. Testing this requires the survey microdata with tenure status.

The distributional results rest on six survey waves for the shares and three years of experimental statistics for saving. The direction of some findings is clear in that evidence: the halt in the decline of the poorest fifth's basic-needs share after 1999, the rise of rent and energy at the bottom, and the rise of the core share between 2005 and 2015 in every group. Others are thinner: the negative median saving of the poorest fifth in the 2010 and 2015 waves may partly be an artefact of the statistical matching, the rise of the poorest fifth's share after 2010 rests on one wave, point estimates for individual waves are less secure, and the 2020 (2022/23) wave cannot be integrated with the earlier ones at the level of detail used here.

For low-income households, the composition of basic needs has changed. Food is a much smaller share of their consumption than in 1988, and rent and energy have taken most of that space. Since 1996 the prices of the housing-related items rose more than food, although over the last decade food rose about as much as rent. Our reading, which is an interpretation rather than a result of this analysis, is that measures aimed at the essentials of low-income households now concern rent and energy as much as food.

## Limitations

The core and wider sets are definitions, and we tested only one alternative. Under the wider set the ordering of the fifths holds, but the ratio between the poorest and richest fifth's shares falls from 1.69 in 1988 to 1.49 in 2015, and the richest fifth's share ends above its 1988 level (44.6% against 40.9%).

Budget-survey shares are means within each group and mix tenants and owners, so the rent results cannot be decomposed into tenure and price effects, and the rent elasticities mix tenure composition with rent levels; they are not demand elasticities. The Engel curves use five group means per wave; they carry no standard errors and say nothing about variation within fifths. The household-type, age and urbanisation splits are not conditioned on income.

The wave labelled 2020 was collected in 2022/23, lacks the housing components, and its housing jump is unexplained. The peer comparison uses waves whose reference periods differ by country.

The national-accounts detail by purpose for Portugal ends in 2022, which is provisional, and it is on the domestic concept. The resident-basis share is a bound that assumes all core spending is residents'. The price-volume split uses HICP prices on national-accounts weights, so its volume term absorbs any coverage mismatch, and the national accounts' own volumes imply a food deflator that we could not reconcile with the HICP.

The euro amounts rest on INE's published mean expenditure per household, applied with one factor to all fifths; for 2010/2011 and 2015/2016 that factor agrees with Eurostat's parity, for 2022/2023 it does not. The fixed-basket projection to 2025 holds quantities constant and uses an HICP rent index that follows sitting tenants.

The saving rates by fifth are experimental, available for three waves, and the 2010 wave defines consumption differently from the other two. The inflation rates by fifth ignore substitution within each weight period and rely on an index that excludes owner-occupied housing; their size depends on the weights by up to three points over 1996–2025. None of the published tables used here report sampling errors.

---

**Data and code.** [Companion folder in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), Apache-2.0. Sources: Eurostat [nama_10_co3_p3](https://ec.europa.eu/eurostat/databrowser/view/nama_10_co3_p3/default/table) (household consumption by purpose), [nama_10_gdp](https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp/default/table) (resident household consumption), [nasa_10_ki](https://ec.europa.eu/eurostat/databrowser/view/nasa_10_ki/default/table) (household saving rate), [hbs_str_t211](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t211/default/table), [hbs_str_t223](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t223/default/table), [hbs_str_t224](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t224/default/table), [hbs_str_t225](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t225/default/table), [hbs_str_t226](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t226/default/table) and [hbs_exp_t133](https://ec.europa.eu/eurostat/databrowser/view/hbs_exp_t133/default/table) (household budget survey), [icw_sr_03](https://ec.europa.eu/eurostat/databrowser/view/icw_sr_03/default/table) (experimental median saving rate by income quintile), [prc_hicp_aind](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_aind/default/table) and [prc_hicp_inw](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_inw/default/table) (consumer prices and weights), [prc_ppp_ind](https://ec.europa.eu/eurostat/databrowser/view/prc_ppp_ind/default/table) (purchasing power parities). INE's mean expenditure per household as reported by [idealista](https://www.idealista.pt/news/node/32320) (2010/2011, 2015/2016) and [DREM Madeira](https://estatistica.madeira.gov.pt/en/download-now-3/social-gb/popcondsoc-gb/rendcondsoc-gb/in-focus-income-and-social/send/415-rendimento-e-condicoes-sociais-em-foco/16851-em-foco-inquerito-as-despesas-das-familias-idf-2022-2023.html) (2022/2023). Minimum wage: DGERT.
