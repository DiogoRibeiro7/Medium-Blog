# Inflation-Plus-Productivity Wage Counterfactuals for Portugal, Computed for Every Base Year from 1975

*The same rule puts the 2024 mean monthly wage anywhere between €1,322 and €2,248. A two-term identity shows what the choice of starting year is really choosing.*

A familiar argument in wage debates goes like this. Take the wage of some year in the past. Raise it every year by inflation, so that its purchasing power holds, and by productivity growth, so that workers share in what the economy has learned to produce. Compare the result with today's wage. If today's wage is lower, the difference is presented as what workers have lost. The arithmetic is simple and the conclusion feels mechanical, which is a large part of why it persuades.

The calculation has one parameter that is rarely discussed, the year it starts from, and several measurement choices that are discussed even less. This article runs the rule for Portugal from every base year between 1975 and 2023, translates each result into the 2024 mean and median monthly earnings, and then splits every answer into two components with an accounting identity. The aim is not to defend or attack any particular number. It is to show which choices carry the result, so that anyone who uses this benchmark, in a wage negotiation, a policy note or a newspaper column, can say what they are actually assuming when they use it.

## The rule and the data

The rule, as it is usually stated, adds the two rates:

```
W(t) = W(t-1) × (1 + π(t) + g(t))
```

where π is annual average consumer-price inflation and g is the growth of real GDP per person employed. I also run the multiplicative version, W(t) = W(t-1) × (1 + π(t)) × (1 + g(t)). With inflation at 2% the two are indistinguishable. With the 20 to 30% inflation Portugal had in the late 1970s and early 1980s, the cross term π·g that the additive rule leaves out is worth up to almost two percentage points a year and 0.8 points on average over 1976–85, and over a decade that adds up.

The harder problem is the wage itself. Portugal has no salary statistics that reach back to 1975. The Quadros de Pessoal series of mean monthly earnings published by PORDATA starts in 1985, and a median from the same survey is listed only from 2002 on GEP's DataLABOR portal and from 2021 in INE's database. The one measure that covers the whole period is national-accounts compensation per employee: total compensation of employees, which includes employer social contributions, divided by the number of employees. I take it from AMECO's 2018 vintage for 1975 to 1995 and chain-link it at 1995 to Eurostat's current national accounts for 1995 to 2024. The two sources disagree by 2.2% on the 1995 level, which is why the series are linked by their growth rates rather than stacked. Inflation is the World Bank's series for the national consumer price index. Productivity comes from AMECO up to 1995 and from Eurostat's real labour productivity index afterwards.

The rule therefore produces a ratio: counterfactual compensation per employee in 2024 divided by actual compensation per employee in 2024. To express it in the units people actually argue about, I multiply that ratio by the 2024 Quadros de Pessoal figures, a mean of €1,582.70 a month and a median of €1,186.55, both for full-time employees with complete pay in October. For the mean, this assumes that Quadros de Pessoal earnings and national-accounts compensation grew alike since the base year. For the median it assumes more: that the median moved with the mean, so that the median-to-mean ratio of the base year was the same as in 2024. That cannot be checked for 1975. It can be checked over the one window for which I have the median from the same source at both ends, 2021 to 2024 in INE's data. From a 2021 base, the shortcut (the mean's ratio applied to the 2024 median) gives €1,158, and running the rule on the 2021 median directly gives €1,171. The two differ by 1.1%, because the median-to-mean ratio rose slightly over those three years, from about 0.74 to 0.75.

All the data, the code and the figures are in the [companion folder of my Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-wage-counterfactual). A workflow reruns the analysis on every change and checks that the summary of results is reproduced.

## The answer from 1975

Start in 1975 and apply the additive rule. Compensation per employee in 2024 would be €44,144 a year instead of the actual €31,077, a ratio of 1.420. In salary terms, the mean monthly wage would be €2,248 instead of €1,583, and the median €1,685 instead of €1,187. The multiplicative rule raises the mean to €2,494. Measuring productivity per hour worked instead of per person lowers the additive figure to €2,009.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-wage-counterfactual/figures/fig1_path_from_1975.png)

*Figure 1. Compensation per employee, actual and counterfactual from 1975, in euros per month (annual figure divided by 14), log scale. Sources: AMECO (2018 vintage) to 1995, Eurostat 1995–2024, World Bank CPI; author's calculations.*

Another way to state the same result is in real terms. Between 1975 and 2024 consumer prices rose 34.9-fold and nominal compensation per employee rose 59.8-fold, so real compensation per employee grew by a factor of 1.71. Real output per person employed grew by a factor of 2.70. Pay grew by 71% and productivity by 170%.

The path between the endpoints matters as much as the endpoints. Real compensation per employee fell to 80.3% of its 1975 level by 1985, through the years of the two IMF-supported adjustment programmes [negotiated in 1977–78 and 1983](https://www.bis.org/review/r190715b.htm), and it did not get back above the 1975 level until around 1990 (101.9). From there it rose to 122.3 in 1995, 139.0 in 2000 and 149.0 in 2010, dropped to 135.0 by 2015 and reached 171.3 in 2024. Most of the gap in Figure 1 opens in that first decade: by 1985 the counterfactual is 50.6 log points above the actual series. After that the gap narrows through the 1990s, widens again to about 53 log points in 2012–2016 and ends at 35.1 in 2024.

## Every other base year

Figure 2 repeats the calculation for every base year from 1975 to 2023 and plots the implied 2024 mean wage against the actual one.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-wage-counterfactual/figures/fig2_base_year_curve.png)

*Figure 2. Mean monthly earnings in 2024 implied by the inflation-plus-productivity rule, by base year, against the actual Quadros de Pessoal mean. Solid orange line: additive rule, productivity per person employed. Dashed: multiplicative rule. Dotted: productivity per hour worked. Blue line: actual.*

Under the headline specification the counterfactual is above the actual wage for only seven base years: 1975 to 1978 and 1980 to 1982. From 1983 onwards it is below the actual wage for every single base year, at between 83.5% and 97.6% of it, with a median of 92.8%. In salary terms, the implied 2024 mean ranges from €1,322 (base year 2012) to €1,545 (base year 2021). Across all forty-nine starting years, the same data and the same rule produce a 2024 mean between €1,322 and €2,248, a factor of 1.7 between the lowest and the highest answer.

The two variants move the curve most where it is most dramatic. Under the multiplicative rule the counterfactual stays above the actual wage for every base year up to 1982. With productivity per hour, it is above only for 1975 to 1977. Between the late 1990s and 2010, with inflation low, the three lines nearly coincide; for later base years the per-hour line sits lower, because hours per person employed rose after 2010. The specification choices matter in the high-inflation years, which are also the years that produce the large claims.

## The identity behind the curve

Why does the curve have this shape? The multiplicative version of the rule has a clean decomposition that explains it.

Write w for compensation per employee, y for real GDP per person employed, P for the GDP deflator and C for the consumer price index. Define the adjusted wage share as s = w / (P · y): compensation per employee as a share of nominal output per person employed. The adjustment is needed because pay is measured per employee and output per person employed, so the self-employed are implicitly credited with the average employee's compensation. By construction, actual compensation in the final year T is w(T) = s(T) · P(T) · y(T). The counterfactual from base year b is w(b) · [C(T)/C(b)] · [y(T)/y(b)]. Divide one by the other and productivity cancels:

```
counterfactual / actual = [C(T)/C(b)] / [P(T)/P(b)]  ×  s(b) / s(T)
```

The ratio depends on two things only. The first is whether consumer prices rose faster or slower than GDP prices since the base year. The second is how the adjusted wage share of the base year compares with today's. How fast productivity grew does not enter at all, because it raises the counterfactual and the actual wage by the same amount. The rule is a wage-share target with a price-index choice attached: picking a base year is picking which wage share counts as the norm, and picking the CPI rather than the GDP deflator adds a price wedge on top.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/portugal-wage-counterfactual/figures/fig3_decomposition.png)

*Figure 3. The multiplicative counterfactual-to-actual ratio by base year, in log points, split into the CPI/GDP-deflator wedge and the change in the adjusted wage share. The two solid lines add up to the dashed one.*

Read in log points, the multiplicative 1975 answer is +45.5 (the additive ratio of 1.420 is +35.1), made of +0.4 from the price wedge and +45.1 from the wage share. Over the full forty-nine years consumer prices and GDP prices rose by almost exactly the same amount, so the 1975 result is a statement about the wage share and nothing else. Other base years mix the two terms differently. From 1977 the total is +25.4, a price wedge of −10.9 against a wage-share term of +36.3. From 1985 it is −11.3, made of −14.8 and +3.5. From 2000 it is −2.6, made of −9.3 and +6.7. From 2015 it is −17.5, and here both terms point the same way: −9.6 from prices and −8.0 because the wage share is higher in 2024 than it was in 2015.

Two regularities stand out. The price wedge is negative for every base year from 1977 onwards, by between 2 and 17 log points with a median of about 10: since the late 1970s Portuguese consumer prices have risen less than the prices of everything the economy produces. That makes CPI-plus-productivity an easier bar than a constant wage share, and it is the main reason the counterfactual sits below the actual wage for every base year from 1983. The identity measures the wedge but does not explain it. Differences in the relative prices of consumption, investment goods and exports, and movements in the terms of trade, are plausible candidates; I have not tested them here, and they should be read as conjectures. The wage-share term, for its part, is very large only for base years before the mid-1980s. For later base years it stays within about nine log points either way, but it is not negligible: for base years from 2012 to 2019, when the wage share was below today's, it supplies between a fifth and a half of the total, and for 2016 it is slightly larger than the price wedge.

## Why 1975 sits where it does

In 1975 the adjusted wage share was 88.4% of GDP. In 2024 it was 56.3%. Part of that 1975 figure is the adjustment itself. Self-employment was 23.6% of employment that year, and the adjusted share credits every self-employed person with the average employee's compensation. With nearly a quarter of the workforce self-employed, that assumption carries a lot of weight, and if the self-employed earned less than employees it overstates labour's share. The unadjusted share, compensation of employees over GDP, was 67.6% in 1975 against 47.7% in 2024. Smaller, but still a fall of twenty points.

1975 was also an unusual year in its own right. In the wage increases that followed April 1974, compensation per employee rose by 31.6% in 1975, against consumer-price inflation of 15.3%, while real GDP per person employed fell by 4.0%. A year in which pay had jumped and output had dropped is about as favourable a starting point as the series offers for the claim that wages have fallen behind. Choosing it amounts to saying that the distribution of income in 1975 was the right one. That may be a defensible position, but it is a normative position, and the data cannot settle it.

The same reasoning applies in the other direction. Base years from 2012 to 2017, the austerity years and their aftermath, give the lowest counterfactuals of all, between €1,322 and €1,338, because the wage share in those years was below today's and the price wedge was still negative. Someone who wanted to argue that today's wages are generous would start there. Each side of a wage debate can find a base year that gives it the answer it wants, and the rule will make the result look like arithmetic rather than a choice.

## The measurement choices, in order of how much they move the 1975 answer

The base year moves the counterfactual-to-actual ratio between 0.835 and 1.420. Next to that, the measurement choices are smaller, but they are not negligible, and they all bite hardest in the years the large claims come from.

The additive rule as it is usually stated gives 1.420 for 1975. The multiplicative rule gives 1.576. The difference is the dropped cross term. About 6 of the 10 log points between the two build up in 1976–85, when inflation averaged 22% a year, and the rest afterwards.

Productivity per hour instead of per person gives 1.269. AMECO's hours data before 1995 show hours per person employed rising from about 1,688 a year in 1975 to 1,893 in 1995, which is an unusual pattern for a period of economic development, so I would not put much weight on the per-hour variant before 1995. It is reported because someone will ask for it.

The consumer price index itself is uncertain in the mid-1970s. The World Bank series has inflation of 21.1% in 1976 and 31.0% in 1977. AMECO's national CPI has 19.9% and 26.7%. Using AMECO's index up to 1995 lowers the 1975 ratio from 1.420 to 1.361, about four percent, and the right answer depends on which historical CPI one trusts.

Compensation is not wages. It includes employer social contributions, and those grew as a share of compensation: wages and salaries were 80.3% of compensation in 1995 and 77.8% in 2024. No wages-and-salaries series reaches 1975. If the employer share also rose between 1975 and 1995, then gross wages proper fell further behind the benchmark than compensation did, but that cannot be verified with the data available here.

Finally, there is an independent check on the whole construction. Quadros de Pessoal earnings exist from 1985 and do not need any splicing. Running the rule on them gives a 2024 mean of €1,454 against the actual €1,583, a ratio of 0.919. The national-accounts series from the same base year gives 0.856. The two wage series differ in coverage, since Quadros de Pessoal counts full-time employees with complete pay and leaves out most of general government, and they disagree by six points. They agree on the sign: from 1985, actual wages are above the counterfactual.

## What to report instead of a single number

None of this makes the inflation-plus-productivity benchmark useless. It makes it a function of the base year, and a function should be reported as a function. Three things turn the single-number claim into something a reader can evaluate.

The first is the curve. State the base year, and show where its answer sits among all the others, as in Figure 2. A result that holds for seven base years out of forty-nine is a different finding from one that holds for forty-two.

The second is the decomposition. Since productivity cancels, the multiplicative 1975 result, a mean of €2,494, is exactly the claim that the adjusted wage share should return to its 1975 level, with consumer prices and GDP prices moving as they did. The additive €2,248 is the same claim with the cross term dropped. Written that way, it can be argued on its merits, by people who disagree about what the wage share ought to be, rather than presented as a matter of arithmetic.

The third is a deliberate price index. The CPI answers a question about purchasing power: what can a worker buy with the wage, relative to what output per worker became? The GDP deflator answers a question about distribution: what fraction of the value produced goes to employees? The two give different answers for Portugal by about ten log points for most base years, and an analysis should say which question it is asking.

For Portugal, the data say this. Since 1975, real compensation per employee has grown much more slowly than output per worker, and almost all of that gap is the fall of the wage share from its post-revolution peak, most of it before 1985. From every base year since 1983, wages have cleared the CPI-plus-productivity bar, mainly because consumer prices rose about ten log points less than GDP prices, while the wage share stayed within about nine log points of today's. Whether 1975 or 1985 is the right reference is a question about how income should be distributed. That is the question the benchmark was supposed to answer, and it turns out to be the question the benchmark has to assume.

---

**Data and code.** [Companion folder in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-wage-counterfactual), Apache-2.0. Sources: European Commission [AMECO, spring 2018 vintage](https://github.com/cran/ameco) (as shipped in the CRAN package `ameco` 0.2.9); Eurostat [nama_10_gdp](https://ec.europa.eu/eurostat/databrowser/view/nama_10_gdp/default/table), [nama_10_a10_e](https://ec.europa.eu/eurostat/databrowser/view/nama_10_a10_e/default/table) and [nama_10_lp_ulc](https://ec.europa.eu/eurostat/databrowser/view/nama_10_lp_ulc/default/table); World Bank consumer price inflation for Portugal via [FRED](https://fred.stlouisfed.org/series/FPCPITOTLZGPRT); GEP/MTSSS Quadros de Pessoal mean earnings via [PORDATA](https://www.pordata.pt/en/portugal/average+monthly+earnings+of+employees+total+and+by+sex-375); INE indicator [0013011](https://www.ine.pt/xurl/indx/0013011/PT) for the median.
