# Basic Needs, Saving and Inflation by Income Group in Portugal, 1988–2025

*Household budget surveys, national accounts and consumer prices, read together: how much of Portuguese spending goes to food, rent and energy, how that differs between the poorest and the richest fifth of households, what is left to save, and whose prices rose fastest.*

A wage is only half of a household's economic position. The other half is what the wage has to pay for. A previous article asked what Portuguese wages would be if they had kept pace with inflation and productivity; this one asks where household spending actually goes, and in particular how much of it is absorbed by the things a household cannot easily do without. There is no thesis to defend here. The aim is to put three public sources side by side, define the measures carefully, and report what they show, including the places where they cannot answer the question.

## What counts as a basic need, and what does not

The analysis uses the COICOP classification of household consumption, which both the national accounts and the household budget surveys follow. I take as core basic needs five items: food and non-alcoholic beverages; actual rent; maintenance and repair of the dwelling; water supply and other services related to the dwelling; and electricity, gas and other fuels. A wider set adds health, the running costs of personal transport (mostly fuel and maintenance, not the purchase of vehicles) and transport services. Neither set is canonical. They are a reasonable reading of what is hard to cut, stated in advance so that every number below can be recomputed under a different definition.

One item needs special treatment. Both sources include imputed rent, an estimate of the rent that owner-occupiers would pay if they rented their own homes. It is a legitimate part of consumption in national accounting, because without it a country's consumption would depend on how many people own their homes, but nobody pays it in cash and nobody can save by not paying it. Throughout, the basic-needs shares exclude imputed rent from both the numerator and the total. When imputed rent is left in, the housing share looks very different, and that difference is part of the story.

The sources are Eurostat's national accounts of household consumption by purpose, annual from 1995 to 2022, with 2022 still provisional; Eurostat's household budget survey tables for Portugal, which cover seven waves from 1988 to 2020 and break consumption down by income fifth; the household saving rate from the sector accounts, 1995 to 2025; an experimental Eurostat series of the median saving rate by income fifth; and the harmonised index of consumer prices by category, 1996 to 2025. All of it is downloaded by a script and committed as a snapshot in the [companion folder of my Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), with the code that produces every figure and number.

## The aggregate: a quarter of consumption, rising since the mid-2000s

Across all households, core basic needs took between 23.7% of consumption (in 2000) and 30.4% (in 2020) over the twenty-eight years the national accounts cover. The share fell gently in the late 1990s, sat at about 24% to 25% through the 2000s, and rose to 28.1% by 2015. The pandemic pushed it to its highest value, 30.4% in 2020, because households could not spend on restaurants, travel or clothing and the essentials became a larger part of a smaller total. By 2022 it was back at 27.3%.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig01_basic_needs_national_accounts.png)

*Figure 1. Core basic needs as a share of household consumption, imputed rent excluded, 1995–2022, with the wider set (adding health and the running costs of transport) as a dashed line. Source: Eurostat nama_10_co3_p3; 2022 provisional.*

Food is most of it, between 17.1% and 21.3% of consumption. Actual rent never exceeds 4% in the aggregate, and household energy stays between 2.7% and 4.2%. The wider set, with health and transport running costs, moves between 38.5% and 44.8%. The aggregate looks stable, and it is: whatever has changed in Portuguese household budgets since 1995 has not changed the share that the average household devotes to the essentials by more than a few points.

The aggregate hides two things. One is imputed rent, which rose from 6.9% of total consumption in 1995 to 10.9% in 2015 and 12.3% in 2020, as home values grew and the estimate followed them. The other is the distribution of the essentials across households, which the national accounts cannot show at all.

## The saving rate

Before turning to the distribution, the other side of the budget. The gross household saving rate, saving as a share of disposable income, fell from 14.3% in 1995 to 10.0% in 2005 and to a low of 6.5% in 2017. It averaged 12.0% between 1995 and 2007 and 8.1% between 2008 and 2019. Two episodes broke the decline: 2009, when it jumped to 11.5% in the financial crisis, and the pandemic, when it reached 12.1% in 2020. After a fall to 7.3% in 2022, the provisional figures put it at 12.9% in 2024 and 12.3% in 2025.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig02_saving_rate.png)

*Figure 2. Gross household saving rate, households and non-profit institutions serving households, % of disposable income, 1995–2025. Source: Eurostat nasa_10_ki; 2024 and 2025 provisional.*

An aggregate saving rate is a ratio of totals. It says how much the household sector as a whole saved, and nothing about which households did the saving. That question needs the budget surveys.

## Engel's law, seven times

The household budget survey ranks households by income and reports what share of their spending went to each category. The oldest regularity in this kind of data is Engel's law: the poorer the household, the larger the share of its spending that goes on food. Portugal follows it in every wave. In 1988 the poorest fifth spent 47.5% of its budget on food and the richest fifth 20.0%.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig03_food_engel_curves.png)

*Figure 3. Food and non-alcoholic beverages as a share of consumption, by income fifth, in each survey wave from 1988 to 2020 (imputed rent included in the total). Source: Eurostat hbs_str_t223.*

What changes between waves is the height and the slope of the curve. The whole curve dropped, fastest at the bottom: the poorest fifth's food share fell to 29.0% by 1999, 22.2% by 2005 and about 17% from 2010 onwards. By 2020 the distance between the poorest and the richest fifth was 7.1 points, against 27.5 in 1988. In the terms of Engel's law, Portuguese households at every income level moved a long way from subsistence over three decades, and the poorest moved furthest.

If food were the only basic need, that would be the end of the story.

## The basic-needs share by income: falling until 1999, rising since

Adding rent, energy, water and dwelling repairs to food, and removing imputed rent, changes the picture. The poorest fifth spent 58.8% of its cash consumption on these core needs in 1988, and 50.3% in 1994. By 1999 the share had fallen to 44.4%. Then it stopped falling. It was 44.5% in 2005 and 44.6% in 2010, and by 2015 it had risen to 48.6%.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig04_core_share_by_quintile.png)

*Figure 4. Food, actual rent, household energy, water and dwelling repairs as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

The other fifths show the same shape at lower levels. The richest fifth went from 27.2% in 1988 to a low of 21.1% in 2005 and back up to 25.3% in 2015. In every wave the poorest fifth spends close to twice the share the richest does on the core needs, and the ratio between them, 2.2 in 1988 and 1.9 in 2015, has hardly moved. With the wider set, which adds health and transport running costs, the poorest fifth's share was 69.3% in 1988 and 66.4% in 2015: two thirds of its cash spending, at both ends of the period.

The food curves of Figure 3 and the core curves of Figure 4 tell different stories because something replaced food in the budgets of the poorest households after 1999.

## Rent and energy

The replacement is in the next two figures. Actual rent, which was 1.6% of the poorest fifth's cash consumption in 1988, reached 6.9% in 2005, 8.8% in 2010 and 11.1% in 2015. For the second fifth it rose from 1.7% to 5.5%. For the middle fifth it went from 1.7% to 3.2%, and for the two richest fifths it stayed at 1% to 2% throughout.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig05_rent_share_by_quintile.png)

*Figure 5. Actual rent paid as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

These are average shares across all households in each fifth, renters and owners together, so a rise can come from more renters, from higher rents, or both. The budget-survey tables used here do not separate the two, and the tenure split would need the survey microdata. What the shares do show is where rent became a large item: almost entirely in the poorest two fifths.

Household energy rose for everyone, and by more at the bottom. Electricity, gas and other fuels took 6.3% of the poorest fifth's cash consumption in 1988 and 1999, then 8.1% in 2005, 10.7% in 2010 and 11.4% in 2015. The richest fifth's share went from 2.8% to 6.2% over the same period.

![Figure 6](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig06_energy_share_by_quintile.png)

*Figure 6. Electricity, gas and other fuels as a share of consumption, imputed rent excluded, by income fifth, 1988–2015. Source: Eurostat hbs_str_t223.*

Put the components together and the change in the poorest households' budgets becomes visible. In 1988, food was 49.4% of their cash consumption and the rest of the core needs about 9 points. In 2015 food was 22.0% and the rest about 27 points, with rent and energy accounting for most of it. For the richest fifth the composition barely changed: food fell from 21.5% to 14.4%, and rent and energy together rose by a few points.

![Figure 7](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig07_composition_by_quintile.png)

*Figure 7. Composition of core basic needs for each income fifth, % of consumption, imputed rent excluded, 1988–2015. Source: Eurostat hbs_str_t223.*

## Imputed rent, and the 2020 survey wave

Leaving imputed rent in the total gives a very different housing picture. The housing division, which includes imputed rent along with actual rent, water and energy, rose from about 12% of every fifth's consumption in 1988 to between 28% and 39% in 2015, with the poorest fifth highest. Most of that rise is imputed rent: it went from 3.9% of the poorest fifth's consumption in 1988 to 16.9% in 2015, and from 6.8% to 19.5% for the richest. A household budget that includes imputed rent will always show housing growing in importance as property values rise, whether or not anyone is paying more.

![Figure 8](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig08_housing_incl_imputed_rent.png)

*Figure 8. Housing, water, electricity, gas and other fuels, imputed rent included, as a share of consumption, by income fifth, in all seven survey waves. The 2020 wave does not separate imputed rent. Source: Eurostat hbs_str_t223.*

The 2020 wave is why the other figures stop in 2015. Eurostat's tables for that wave report the housing division but not its components, so imputed rent cannot be removed. The housing share also jumps in 2020 for every fifth, by 10 points for the richest, from 28.2% to 38.2%. A jump of that size in five years, concentrated where actual rent plays almost no role, is more likely to reflect a change in how imputed rent was measured, or the effect of the pandemic on the rest of the budget, than a change in what households paid. Without the component split it cannot be settled from these tables, and the 2020 housing figures are best read with that caveat.

## What is left after the essentials

Shares describe budgets; levels describe living standards. The budget survey also reports mean consumption per adult equivalent by income fifth, in purchasing power standards, which adjusts for household size and for price levels across the EU. Two ratios between the richest and the poorest fifth come out of it.

![Figure 9](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig09_consumption_ratio.png)

*Figure 9. Consumption per adult equivalent of the richest fifth divided by that of the poorest fifth, for all consumption and for consumption left after core basic needs (imputed rent excluded). Source: Eurostat hbs_exp_t133 and hbs_str_t223.*

For total consumption, the richest fifth spent 4.6 times as much as the poorest in 1988, 4.9 times in 1994, 2.8 times in 2010 and 2.3 times in 2020. Inequality of consumption in Portugal, measured this way, roughly halved. For what is left after the core basic needs, which is the part of the budget a household can actually choose how to spend, the ratio is wider but fell in the same way: from 7.8 in 1988 to 5.0 in 1999 and 3.6 in 2015. In 2015 the poorest fifth had 42.7% of its total consumption left after food, rent, energy, water and repairs; the richest fifth had 60.1%.

The two facts sit together. The gap in consumption between rich and poor households narrowed substantially, and the share of the poorest households' budget taken by the essentials stopped falling around 1999 and then rose.

## Saving by income

The saving rate is where the bottom of the distribution becomes most visible. Eurostat's experimental statistics on the joint distribution of income, consumption and wealth estimate a median saving rate for each income fifth by matching the budget survey with the income survey. They exist for three years, and they are flagged as experimental, but the pattern is clear.

![Figure 10](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig10_saving_by_quintile.png)

*Figure 10. Median saving rate by income fifth, % of disposable income, 2010, 2015 and 2020. Experimental statistics. Source: Eurostat icw_sr_03.*

In 2010 the median household in the poorest fifth had a saving rate of −12.2%, and in 2015 of −9.3%: it spent more than its income, by drawing on savings, borrowing or receiving help not counted as income. The second fifth saved 6.5% and 8.1% in those years, the middle fifth 13.3% and 19.2%, and the richest fifth 27.3% and 37.2%. In 2020 every fifth saved more, and the poorest fifth's median turned positive at 8.1%, consistent with the aggregate jump in Figure 2 and with a year in which spending was constrained. These figures should not be compared directly with the national-accounts saving rate, which is a ratio of totals and uses different definitions of income and consumption; a median by fifth and an aggregate ratio answer different questions.

## Whose prices rose fastest

The last question is whether the essentials have become dearer relative to everything else. Figure 11 shows the price index of each core basic need from 1996 to 2025, against all items.

![Figure 11](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig11_basic_needs_prices.png)

*Figure 11. Harmonised consumer price indices for each core basic need and for all items, 1996 = 100. Source: Eurostat prc_hicp_aind.*

Over the full period, prices of all items rose by 86.6%. Food rose by 88.0%, almost the same, but half of that rise came after 2021. The housing-related items rose much faster: actual rent by 115.4%, electricity, gas and fuels by 136.4%, dwelling repairs by 153.7%, and water and other dwelling services by 179.8%. Weighting the core items by their shares in the 2015 budget survey, the core basic needs rose 33.6% between 2015 and 2025 against 24.8% for all items, and 28.7% against 20.5% between 2020 and 2025.

Because the poorest households spend more of their budgets on exactly these items, they face a different inflation rate. Weighting the twelve consumption divisions of the price index by each fifth's own budget shares, with imputed rent removed because the price index does not cover owner-occupied housing, gives an inflation rate for each income fifth.

![Figure 12](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-household-spending/figures/fig12_inflation_by_quintile.png)

*Figure 12. Cumulative inflation of each income fifth's consumption basket, HICP divisions weighted by household budget survey shares (2005 wave for 2005–2015, 2015 wave afterwards), imputed rent removed. Source: Eurostat prc_hicp_aind and hbs_str_t223.*

Between 2005 and 2015, the poorest fifth's basket rose by 20.2% and the richest fifth's by 15.8%, a difference of 4.4 points. Between 2015 and 2025 the figures were 26.6% and 24.4%, a difference of 2.2 points. In the sharpest episode, 2021 to 2023, the five baskets rose by between 12.8% and 13.4%, almost the same amount: food, energy and nearly everything else moved together, so budget weights mattered little. The differences accumulate in the quieter years, when the essentials keep rising a little faster than the rest.

## What the measurements support

None of this needs a thesis to be useful. It does have direct implications for anyone who uses these statistics to make decisions.

The first is the choice of price index. Benefits, pensions and the minimum wage are discussed against headline inflation. Over 2005–2025 the basket of the poorest fifth rose by several points more than the basket of the richest, and the headline index sits close to the average. An adjustment that tracks headline inflation therefore lets the bottom of the distribution lose a little ground every decade, unless it is calculated on a basket that matches the people it is meant to protect. Computing such a basket takes only public data, as this analysis shows; the cost of not doing it is a bias that is small in any one year and not small over twenty.

The second is imputed rent. A budget share that includes imputed rent mixes a real cash outlay with a valuation of property, and in Portugal the valuation part grew from 6.9% to more than 10% of aggregate consumption. Any measure of affordability, from a housing-cost ratio to a basic-needs share, should state whether it includes it. The difference between Figures 4 and 8 is the size of the error if it does not.

The third is resolution. The national accounts are annual but cannot see the distribution; the budget surveys see the distribution but only every five or six years, and their latest wave cannot separate the components that matter most for housing. The finding that the poorest households' basic-needs share stopped falling after 1999 rests on six survey waves. The saving-rate pattern rests on three years of experimental statistics. Those are thin foundations for precise numbers, and solid enough for the direction.

The fourth is which basic need to watch. Food has become a much smaller part of every household's budget. Rent and energy have taken its place at the bottom of the distribution, and their prices rose faster than food over most of the period. A policy aimed at the essentials of low-income households in Portugal in the 2020s is, to a large degree, a policy about rent and energy.

## Caveats

The core and wider sets of basic needs are definitions, not facts, and different choices would move the levels, though not the patterns across income fifths. The budget-survey quintiles rank households by income, and the share figures are averages within each fifth, mixing renters and owners. The 2020 wave lacks the housing components, so the measures without imputed rent end in 2015, and its housing jump is unexplained. National-accounts detail by purpose for Portugal stops in 2022, and 2021 and 2022 are provisional. The saving rates by income fifth are experimental and exist for three years only. The inflation rates by fifth use fixed weights from a single wave over ten years, ignore substitution between items, and rely on a price index that excludes owner-occupied housing. Consumption in purchasing power standards is in current terms, so the levels are not comparable across waves, although the ratios between fifths are.

---

**Data and code.** [Companion folder in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-household-spending), Apache-2.0. Sources: Eurostat [nama_10_co3_p3](https://ec.europa.eu/eurostat/databrowser/view/nama_10_co3_p3/default/table) (household consumption by purpose), [nasa_10_ki](https://ec.europa.eu/eurostat/databrowser/view/nasa_10_ki/default/table) (household saving rate), [hbs_str_t223](https://ec.europa.eu/eurostat/databrowser/view/hbs_str_t223/default/table) and [hbs_exp_t133](https://ec.europa.eu/eurostat/databrowser/view/hbs_exp_t133/default/table) (household budget survey), [icw_sr_03](https://ec.europa.eu/eurostat/databrowser/view/icw_sr_03/default/table) (experimental median saving rate by income quintile) and [prc_hicp_aind](https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_aind/default/table) (consumer prices).
