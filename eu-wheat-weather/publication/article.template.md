# Heat, Drought and Europe's Wheat

*What {{n_harvests}} harvests in {{n_regions}} regions say about the cost of a hot or dry spring*

Wheat is the most widely grown crop in the European Union, and its harvest depends on a few months of weather. In 2003, 2018 and 2022, heat and drought hit much of the continent in spring and summer, and each time the question came back: how much does a bad season cost, and are farms getting better at riding it out?

This piece reports what a reproducible study of the EU's regions found. The full analysis, with every table, test and line of code, is in the project repository. Every number below is read by code from the same result files, so the article and the analysis cannot drift apart.

## The data

Eurostat publishes, region by region, the area sown with common wheat and the harvest it produced. Dividing one by the other gives the yield. The weather comes from E-OBS, a daily European grid of temperature and rainfall built from thousands of weather stations. I averaged it over each region, counting only the part of each grid cell that lies inside the region.

I fixed the rules for the sample before looking at any result: enough wheat to measure, a plausible yield, complete weather and at least ten years per region. What remains is {{n_obs}} regional harvests in {{n_regions}} regions of {{n_countries}} countries, from {{year_first}} to {{year_last}}. A further {{single_region_countries}} countries have only one wheat region each, so they cannot take part in a comparison that needs two regions of the same country.

## How the comparison works

Yields drift upward with better varieties and farming, and they move with prices and policies that differ from country to country. To separate the weather from all of that, the model compares regions of the same country in the same year, each against its own usual yield. If one region has a dry spring and another region of the same country does not, the difference in how their harvests depart from normal is what the model attributes to the weather.

This is one of the rare economic questions where the answer can be read as a cause. Once a region's normal climate is taken into account, the weather in a given spring is close to random: no farmer and no policy chooses it. So the estimates say what a hot or dry season does to that year's harvest. They do not say what climate change will do, because farmers adapt to a climate that shifts over decades in ways they cannot adapt to a single bad year.

Heat is measured in degree days above 30 °C between 1 March and 31 July: every hour above 30 °C counts, weighted by how far above 30 °C it is. Rainfall is the total over the same months.

Everything, from the measures and models to the tests and what would count as a result, was written down and published before I fitted any model. A result counts only if it survives a correction for asking three questions at once, and holds under two different ways of measuring the uncertainty.

## A check first: weather after the harvest

If the comparison works, the weather in August and September, after the wheat is in, should have nothing to do with that year's yield. And it shows no link: the test gives p = {{e5_p_twoway}} and {{e5_p_boot}} under the two methods. Had it come out the other way, everything below would have been in doubt.

## Drought: rain matters, but the cost of a dry spring is uncertain

Rainfall clearly matters (corrected p = {{e2_holm_twoway}} and {{e2_holm_boot}}). Yields are highest at around {{optimum_mm}} mm between March and July, while the typical region receives about {{median_mm}} mm. The estimated curve falls on both sides: much less rain lowers yields, and so does much more, though the band is wide at the dry end.

![Rainfall over the season and wheat yields](figures/fig2_precipitation.png)

*Change in yield against a region receiving the median rainfall. The band is a 95% interval.*

A dry season, the driest one in ten for that region and on average {{dry_mm}} mm below its median, changes yield by {{dry_pct}}. By the rule I fixed in advance, this counts as an established negative effect: rainfall as a whole is clearly significant, and this effect is negative. But the interval for the dry-season loss on its own runs from {{dry_low}} to {{dry_high}}, so its size is not pinned down. Measuring drought by the water balance instead, rain minus what evaporation would take, the loss is {{wb_pct}} ({{wb_low}} to {{wb_high}}). That was one of the checks I fixed in advance, though not one that decides anything.

## Heat: harmful only at the very top

Ten more degree days above 30 °C go with a change in yield of {{e1_pct}} (interval {{e1_low}} to {{e1_high}}). After the correction for three questions, that is not established (p = {{e1_holm_twoway}} and {{e1_holm_boot}}).

The shape of the response explains why. Counting the days spent in each three-degree band of temperature, only days at 36 °C and above clearly lower yields: {{ge36_pct}} per day (interval {{ge36_low}} to {{ge36_high}}) compared with a day at 15–18 °C. Days that hot are rare before the end of July: the average region and season spends {{ge36_days}} days at 36 °C or above. So an average effect across all hot days dilutes the damage done at the very top.

![Yield change for one more day in each temperature range](figures/fig1_temperature.png)

*Each point compares one more day in that range with a day at 15–18 °C. Bars are 95% intervals; the orange one excludes zero.*

## Has wheat become less sensitive to heat?

Comparing 2012–2024 with 2000–2011, there is no detectable change (p = {{e3_holm_twoway}} and {{e3_holm_boot}}). That is not evidence that nothing changed. The design could only have detected a change of {{e3_mde_pct}} or more per ten degree days, which is larger than the heat effect itself. Whether European wheat has adapted to heat is a question these data cannot answer.

## Can the weather predict a bad year?

I fitted a version of the model without one year and asked it to predict that year from the weather alone.

- **2003:** EU yield {{e4_2003_obs}} observed, {{e4_2003_pred}} predicted.
- **2018:** {{e4_2018_obs}} observed, {{e4_2018_pred}} predicted.
- **2022:** {{e4_2022_obs}} observed, {{e4_2022_pred}} predicted.

Region by region the fit is loose: the correlations between predicted and observed departures from normal are {{e4_2003_corr}}, {{e4_2018_corr}} and {{e4_2022_corr}} for the three years. The 2003 heatwave did far more damage than its season totals of heat and rain imply. The heat may have come at the worst moment for the grain, or done harm that season totals cannot see; these data cannot tell which.

![Predicted and observed yield anomalies by region](figures/fig3_out_of_sample.png)

*Each dot is a region in a year the model never saw. On the diagonal, the prediction is exact.*

## How robust is this?

I fixed {{n_sensitivity}} variations in advance. They include other heat thresholds, a season starting in April, weather measured only below 800 m, regions weighted by their wheat area, and the water balance in place of rainfall. None reverses a sign. Heat stays harmful in every one, and in {{e1_variants_clear}} of them its interval excludes zero; weighting regions by their wheat area, for instance, gives {{e1_area_weighted}}.

![Heat and drought effects under each variation](figures/fig4_sensitivity.png)

## What this does not say

- **Not climate change.** These are effects of one season's weather on that season's harvest. Farmers adapt to a changing climate in ways they cannot adapt to a single bad year.
- **Not nationwide shocks.** By design, the main comparison ignores whatever hits a whole country at once, and a continental heatwave does exactly that. The heat and drought effects above leave out the part of such a year that every region of a country shares; only the prediction test counts it.
- **Not the fields themselves.** The weather is averaged over whole regions, including forests and mountains, and E-OBS is interpolated from stations, which are sparse in places.

## Data, code and acknowledgements

The data, the code, the research plan with every change made to it, and the full results are in the project repository. Wheat statistics: © European Union, Eurostat. Regional boundaries: © EuroGeographics for the administrative boundaries.

We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) and the Copernicus Climate Change Service, and the data providers in the ECA&D project (https://www.ecad.eu).

Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: An Ensemble Version of the E-OBS Temperature and Precipitation Datasets, J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200
