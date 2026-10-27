# Heat, Drought and Europe's Maize

*What {{n_harvests}} harvests in {{n_regions}} regions say about a hot or dry summer*

*Maize statistics © European Union, Eurostat. Weather: E-OBS v33.0e (ECA&D and the Copernicus Climate Change Service), for non-commercial research and education. Regional boundaries © EuroGeographics for the administrative boundaries, from GISCO, for non-commercial use. Crop calendar: GGCMI, under a Creative Commons Attribution licence (details below).*

Grain maize is one of the European Union's main cereals, and unlike wheat it grows through the hottest weeks of the summer. A companion study of wheat, with the same design, found no established effect of heat or drought on wheat yields once its flawed drought test was corrected. This piece asks the same questions of maize. The plan was registered before any maize yield was looked at, and every number below is read by code from the result files in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-wheat-weather).

## The data

Eurostat publishes, region by region, the area of grain maize and the harvest it produced; dividing one by the other gives the yield. Regional figures cover most of the EU's maize only from {{year_first}}, so the study starts there. The weather comes from E-OBS, a daily European grid of temperature and rainfall built from weather stations, averaged over each region.

Each region's season runs from its usual planting date to its usual maturity date, taken from a global crop calendar: planting falls between {{planting_first}} and {{planting_last}}, and maturity between {{maturity_first}} and {{maturity_last}}, depending on the region.

The rules for the sample were fixed in advance: enough maize to measure, a plausible yield, a rule that removes regional figures that are not real measurements (it removed {{n_excluded_rule}} regional harvests, mostly pro-rated or carried forward), complete weather and at least ten years per region. What remains is {{n_obs}} regional harvests in {{n_regions}} regions of {{n_countries}} countries, from {{year_first}} to {{year_last}}.

## How the comparison works

Yields drift with varieties, farming, prices and policies. To separate the weather from all of that, the model compares regions of the same country in the same year, each against its own usual yield. Once a region's normal climate is taken into account, the weather in a given summer is close to random, so the estimates say what a hot or dry season does to that year's harvest. Two cautions apply: the cause is the weather as a bundle, heat with sun, dry air and its timing; and these are effects of one season, not of a changing climate, to which farmers adapt.

Heat is measured in degree days above 30 °C over the season: every hour above 30 °C counts, weighted by how far above it is. A typical region has {{edd_median}} such degree days in a season, and half of all seasons have between {{edd_p25}} and {{edd_p75}}, so ten more is well within the usual swing from one season to the next.

Every rule was written down and published before any maize yield was looked at, after the wheat study and its critical review. Maize uses the fixes that review called for: the drought effect is tested directly, regional figures that are not measurements are removed, and the uncertainty allows for countries that move together within a year. A result counts only if it survives a correction for asking two questions at once, under two ways of measuring the uncertainty.

## A check first, which only just passes

If the comparison works, the weather in the two months after each region's maturity date should have nothing to do with that year's yield. One of the two methods finds a link (p = {{e5_p_twoway}}); the other does not (p = {{e5_p_boot}}). By the rule fixed in advance, both would have to find it for the check to fail, so it passes, but only narrowly. Part of the reason may be agronomic: grain maize often stands in the field for weeks after it matures, so the weather then can affect what is harvested.

## Heat: established, by a narrow margin

Ten more degree days above 30 °C go with a change in yield of {{e1_pct}} (interval {{e1_low}} to {{e1_high}}). After the correction for two questions, that is established, but only just: corrected p = {{e1_holm_twoway}} and {{e1_holm_boot}}.

Splitting temperature into three-degree bands shows where the harm starts. Under both methods, only time at 27–30 °C and at 33–36 °C differs from zero: {{h2730_pct}} and {{h3336_pct}} per hour, against an hour at 15–18 °C. One cool band, 12–15 °C, differs under the first method only. Time at 36 °C and above has the largest estimate, {{ge36_pct}} per hour, but its interval is wide ({{ge36_low}} to {{ge36_high}}) and the second method puts it right at the line (p = {{ge36_boot_p}}). The bands are a descriptive check, with no correction for looking at many at once.

![Yield change for one more hour in each temperature range](figures/fig1_temperature.png)

*Each point compares one more hour in that range with an hour at 15–18 °C. Bars are 95% intervals (two-way clustered); orange ones exclude zero.*

## Drought: established

A dry season, the driest one in ten for that region and on average {{dry_mm}} mm below its median, changes yield by {{e2_pct}} (interval {{e2_low}} to {{e2_high}}). After the correction for two questions, that is established: corrected p = {{e2_holm_twoway}} and {{e2_holm_boot}}. Unlike the first wheat analysis, this tests the dry-season effect itself, not rainfall as a whole.

## Maize beside wheat

Wheat, under the same design, gave {{wheat_a1_e1_pct}} for heat and {{wheat_a1_e2_pct}} for a dry season, and neither is established; its first, registered analysis gave {{wheat_e1_pct}} and {{wheat_e2_pct}}. The drought estimates are close, but only maize's is precise enough to count, and maize's heat estimate is larger. The two crops differ in their seasons, regions and years, so this is a comparison, not a test of a difference.

![Heat and drought for maize and for wheat](figures/fig2_crops.png)

*Lines are 95% intervals (two-way clustered).*

## How robust is this?

I fixed {{n_sensitivity}} variations in advance: other heat thresholds, a fixed season from May to September, leaving out the four southern countries where much maize is irrigated, and weighting regions by their maize area. None reverses a sign, and the effects grow in two of them: without the irrigated south, heat is {{e1_not_irrigated}}; with area weights, heat is {{e1_area_weighted}} and drought {{e2_area_weighted}}. That fits irrigation blunting the damage in the south, though it does not prove it.

![Heat and drought under each variation](figures/fig4_sensitivity.png)

## Can the weather predict a bad year?

A version of the model fitted without one year predicted that year from the weather alone, for the study's regions weighted by their maize area:

- **2012:** {{e4_2012_obs}} observed, {{e4_2012_pred}} predicted.
- **2015:** {{e4_2015_obs}} observed, {{e4_2015_pred}} predicted.
- **2022:** {{e4_2022_obs}} observed, {{e4_2022_pred}} predicted.

2015 is predicted well, but in 2012 and 2022 the losses were far larger than the season's weather predicts. Region by region, the correlations between predicted and observed departures from normal are {{e4_2012_corr}}, {{e4_2015_corr}} and {{e4_2022_corr}}.

![Predicted and observed yield anomalies by region](figures/fig3_out_of_sample.png)

*Each dot is a region in a year the model never saw. On the diagonal, the prediction is exact.*

## What this does not say

- **Not climate change.** These are effects of one season's weather on that season's harvest.
- **Not irrigated and rainfed maize apart.** Much of southern Europe's maize is irrigated; the estimates average over both, and one rainfed crop calendar is used everywhere.
- **Not nationwide shocks.** The comparison ignores whatever hits a whole country at once; only the prediction test counts it.
- **Probably too small.** Little variation in weather is left once regions of a country are compared, and errors in the weather data pull the estimates toward zero.
- **Not a trend.** Fifteen harvests are too few to test whether maize has become more or less sensitive.
- **Not silage maize.** Only grain maize is covered.

## Data, code and acknowledgements

The data, the code, the research plan with every change made to it, and the full results are in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-wheat-weather). Maize statistics: © European Union, Eurostat. Regional boundaries: © EuroGeographics for the administrative boundaries, from GISCO, for non-commercial use. Weather: E-OBS v33.0e, for non-commercial research and education. Crop calendar: GGCMI Phase 3 crop calendar v1.01, CC BY 4.0, from Jägermeyr, J., C. Müller, A. C. Ruane, et al. 2021: Climate impacts on global agriculture emerge earlier in new generation of climate and crop models, Nature Food, 2, 873-885. doi:10.1038/s43016-021-00400-y; data doi:10.5281/zenodo.5062513.

We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) and the Copernicus Climate Change Service, and the data providers in the ECA&D project (https://www.ecad.eu).

Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: An Ensemble Version of the E-OBS Temperature and Precipitation Datasets, J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200
