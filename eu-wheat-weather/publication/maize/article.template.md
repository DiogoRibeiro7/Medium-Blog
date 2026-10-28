# Heat, Drought and Europe's Maize

*What {{n_harvests}} harvests in {{n_regions}} regions say about a hot or dry summer*

*Maize statistics © European Union, Eurostat. Weather: E-OBS v33.0e (ECA&D and the Copernicus Climate Change Service), for non-commercial research and education. Regional boundaries © EuroGeographics for the administrative boundaries, from GISCO, for non-commercial use. Crop calendar: GGCMI, under a Creative Commons Attribution licence (details below).*

Grain maize is one of the European Union's main cereals, and unlike wheat it grows through the hottest weeks of the summer. A companion study of wheat, with a similar design, found no established effect of heat or drought on wheat yields once its flawed drought test was corrected. This piece asks the same questions of maize. The design was registered before any maize yield was looked at, and every number below is read by code from the result files in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-wheat-weather).

In short: by the study's rules, a dry season lowers maize yields, and that result survives leaving out any single country except Spain. Heat probably lowers them too, but it only just clears the bar the study set itself, and much of its estimate comes from one country.

## The data

Eurostat publishes, region by region, the area of grain maize (including corn-cob mix) and the harvest it produced; dividing one by the other gives the yield. Regional figures cover most of the EU's maize only from {{year_first}}, so the study starts there. The weather comes from E-OBS, a daily European grid of temperature and rainfall built from weather stations, averaged over each whole region.

Each region's season runs from its usual planting date to its usual maturity date, taken from a global crop calendar, with the same dates every year: planting falls between {{planting_first}} and {{planting_last}}, and maturity between {{maturity_first}} and {{maturity_last}}, depending on the region.

The rules for the sample were fixed in advance: enough maize to measure, a plausible yield, a rule that removes regional figures that are not real measurements (it removed {{n_excluded_rule}} regional harvests, mostly pro-rated or carried forward, though it catches only the patterns it looks for), complete weather and at least ten years per region. What remains is {{n_obs}} regional harvests in {{n_regions}} regions of {{n_countries}} countries, from {{year_first}} to {{year_last}}.

## How the comparison works

Yields drift with varieties, farming, prices and policies. To separate the weather from all of that, the model compares regions of the same country in the same year, each against its own usual yield. The estimates can be read as causal under one assumption: once a region's normal climate and what its country shares in a given year are accounted for, the weather in a given summer is as good as random. Under it, they say what a hot or dry season does to that year's harvest, with two cautions taken up at the end: the cause is the weather as a bundle, not heat or rain alone, and these are effects of a single season's weather, not of climate change.

Heat is measured in degree days above 30 °C over the season: every hour above 30 °C counts, weighted by how far above it is. A typical region has {{edd_median}} such degree days in a season, and half of all regional seasons have between {{edd_p25}} and {{edd_p75}}. But most of that spread is between regions. Within a region, the typical change from one season to the next is {{swing_median}} degree days, and only {{swing_share10}} of changes reach ten. So ten more degree days make a hot season for that region.

The design was written down and published before any maize yield was looked at, after the wheat study and its critical review. Maize uses the fixes that review called for: the drought effect is tested directly, regional figures that are not measurements are removed, and the uncertainty allows for countries that move together within a year. Two later entries in the plan's change log settle details it had left open: one, written after the regional figures were assembled, says how the data rule treats a region with a single year and which weather is used; the other, written after the estimates, records choices the code had fixed before the run.

A result counts as **established** only if it survives a correction for asking two questions at once, under two ways of measuring the uncertainty. That is the bar the plan set in advance. It is not proof, and a result can clear it by a hair.

## A check first

If the comparison works, the weather in the two months after each region's maturity date should have nothing to do with that year's yield. By the rule fixed in advance, both methods would have to find a link for the check to fail. The second method does not (p = {{e5_p_boot}}), so the check passes.

The first method does find one (p = {{e5_p_twoway}}). What it finds is rain after maturity going with higher yields, {{placebo_rain_pct}} for each 100 mm (p = {{placebo_rain_p}}), mostly in France: without France, its p rises to {{placebo_without_p}}. That rain is unrelated to the season's own rain, and adding it to the model leaves the heat estimate at {{e1_with_placebo_pct}}. A late harvest would not explain it: maize left standing through a wet autumn loses yield, which predicts the opposite sign.<!-- TODO[CITATION]: verify or add source for yield losses of grain maize left standing through wet weather. --> Two explanations remain, and these data cannot tell them apart: a calendar maturity date that is too early for late regions, or chance.

## Heat: probably harmful, only just past the bar

Ten more degree days above 30 °C go with a change in yield of {{e1_pct}} (interval {{e1_low}} to {{e1_high}}). After the correction for two questions, that clears the bar: corrected p = {{e1_holm_twoway}} and {{e1_holm_boot}}. But it clears it by a hair, and it is not robust:

- **The first method** puts it past the bar only because its estimate of the uncertainty comes out slightly smaller than with errors grouped by country alone, which gives p = {{e1_p_country}}.
- **Single countries decide it.** Leaving out any one of {{n_loo_e1_lost}} of the {{n_countries}} countries, Romania among them, makes it not established. So does leaving out only the {{ro2024_n}} Romanian regions in {{ro2024_year}}: heat is then {{ro2024_e1_pct}}, with corrected p = {{ro2024_holm_twoway}} and {{ro2024_holm_boot}}.
- **Romania supplies {{e1_share_ro}} of the estimate**, and one small region, Bucharest and its surroundings, {{e1_share_ro32}}.
- **The pooled number describes neither group.** A model that lets heat differ in Spain, Portugal, Italy and Greece, where much maize is irrigated, gives {{e1_south_pct}} there and {{e1_rest_pct}} elsewhere.

Heat probably lowers maize yields, but the evidence here is marginal.

An average season has {{hours_2730}} hours at 27–30 °C, {{hours_3033}} at 30–33 °C, {{hours_3336}} at 33–36 °C and {{hours_ge36}} at 36 °C and above. Splitting temperature into these three-degree bands hints that the harm begins below 30 °C, though the pattern is irregular. Under both methods, only time at 27–30 °C and at 33–36 °C differs from zero: {{h2730_pct}} and {{h3336_pct}} per hour, against an hour at 15–18 °C. The band between them, 30–33 °C, does not. One cool band, 12–15 °C, differs under the first method only. Time at 36 °C and above has the largest estimate, {{ge36_pct}} per hour, but it is rare, Spain supplies {{ge36_share_es}} of its variation, its interval is wide ({{ge36_low}} to {{ge36_high}}) and the second method puts it right at the line (p = {{ge36_boot_p}}). The bands are a descriptive check, with no correction for looking at many at once.

![Yield change for one more hour in each temperature range](figures/fig1_temperature.png)

*Each point compares one more hour in that range with an hour at 15–18 °C. Bars are 95% intervals (two-way clustered); orange marks the bands that differ from zero under both methods.*

## Drought: established

A dry season, the driest one in ten for that region and on average {{dry_mm}} mm below its median, changes yield by {{e2_pct}} (interval {{e2_low}} to {{e2_high}}). After the correction for two questions, that is established: corrected p = {{e2_holm_twoway}} and {{e2_holm_boot}}. Unlike the first wheat analysis, this tests the dry-season effect itself, not rainfall as a whole.

This result is much sturdier than heat's. Under the first method it stays clear whichever single country is left out. Under the full rule it fails in one case: without Spain, the second method no longer confirms it once the correction for two questions is applied.

## Maize beside wheat

Wheat, under a similar design, gave {{wheat_a1_e1_pct}} for heat and {{wheat_a1_e2_pct}} for a dry season, and neither is established; its first, registered analysis gave {{wheat_e1_pct}} and {{wheat_e2_pct}}. The drought estimates are close, but only maize's is precise enough to count. Maize's heat estimate is larger, but the two cannot be told apart: the difference is {{heat_gap}} percentage points, with a margin of error of about ±{{heat_gap_margin}}. Ten more degree days are also a far bigger shock for wheat, whose harvest window has a median of {{wheat_edd_median}} such degree days, than for maize, with {{edd_median}}. The two crops differ in their seasons, regions, years and some details of the design, so this is a comparison, not a test of a difference.

![Heat and drought for maize and for wheat](figures/fig2_crops.png)

*Lines are 95% intervals (two-way clustered).*

## How robust is this?

I fixed {{n_sensitivity}} variations in advance: other heat thresholds, a fixed season from May to September, leaving out the four southern countries where much maize is irrigated, and weighting regions by their maize area. None reverses a sign.

Heat is larger without the irrigated south ({{e1_not_irrigated}}) and with area weights ({{e1_area_weighted}}). Drought is larger with area weights ({{e2_area_weighted}}) but smaller without the south ({{e2_not_irrigated}}), the opposite of what irrigation blunting the damage would predict. Both heat variations lean on Romania, which supplies {{ro_share_not_irrigated}} of the variation in heat without the south and {{ro_share_area_weighted}} with area weights. And the narrow interval without the south comes from the first method's arithmetic: with errors grouped by country alone, it would be {{not_irrigated_se_ratio}} times as wide.

![Heat and drought under each variation](figures/fig4_sensitivity.png)

*The main model, then each variation fixed in advance. Lines are 95% intervals (two-way clustered).*

## Can the weather predict a bad year?

A version of the model fitted without one year predicted that year from the weather alone, for the study's regions weighted by their maize area:

- **2012:** {{e4_2012_obs}} observed, {{e4_2012_pred}} predicted.
- **2015:** {{e4_2015_obs}} observed, {{e4_2015_pred}} predicted.
- **2022:** {{e4_2022_obs}} observed, {{e4_2022_pred}} predicted.

The 2015 total is predicted well, but in 2012 and 2022 the losses were far larger than the season's weather predicts. Romania and Hungary account for {{e4_2012_top2}} of the shortfall in 2012 and {{e4_2022_top2}} in 2022. The season's weather, as the model measures it, does not account for the harvest failures there. Drought concentrated in a few weeks, which a season total smooths over, is one possibility; these data do not identify the cause. Region by region, the correlations between predicted and observed departures from normal are {{e4_2012_corr}}, {{e4_2015_corr}} and {{e4_2022_corr}}; in 2022 the predictions explain none of the differences between regions.

![Predicted and observed yield anomalies by region](figures/fig3_out_of_sample.png)

*Each dot is a region in a year the model never saw. On the diagonal, the prediction is exact.*

## What this does not say

- **Not climate change.** These are effects of one season's weather on that season's harvest. Farmers adapt to a changing climate in ways they cannot adapt to a single bad year.
- **Not the mechanism.** A hot spell comes with sun, dry air and its own timing, and the model does not separate them.
- **Not irrigated and rainfed maize apart.** Much of southern Europe's maize is irrigated;<!-- TODO[CITATION]: verify or add source for the share of irrigated grain maize in Spain, Portugal, Italy and Greece. --> the estimates average over both, and one rainfed crop calendar is used everywhere.
- **Not every country equally.** Spain supplies {{edd_share_es}} of the variation in heat that the comparison uses, Italy {{edd_share_it}} and Romania {{edd_share_ro}}. Spain, Portugal, Italy and Greece together supply {{south_var_share}} of it but grow {{south_area_share}} of the maize.
- **Not nationwide shocks.** The comparison ignores whatever hits a whole country at once; only the prediction test counts it.
- **Probably too small.** Only {{heat_variance_left}} of the variation in heat is left once regions of a country are compared, and errors in the weather data pull the estimates toward zero.
- **Not the fields themselves.** The weather is averaged over whole regions, including hills and mountains where no maize grows, and E-OBS is interpolated from stations, which smooth out the hottest days: 36 °C on the grid is not 36 °C in a field.
- **Not perfect yields.** The data rule misses some flat or repeated regional series, and all {{n_estimated}} Greek regional harvests from {{estimated_first}} on are Eurostat estimates rather than measurements.
- **Not a trend.** With {{n_harvests}} harvests, there are too few to test whether maize has become more or less sensitive.
- **Not silage maize.** Only grain maize is covered.

## Data, code and acknowledgements

The data, the code, the research plan with every change made to it, the full results and the critical review are in the [project repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/eu-wheat-weather). Maize statistics: © European Union, Eurostat. Regional boundaries: © EuroGeographics for the administrative boundaries, from GISCO, for non-commercial use. Weather: E-OBS v33.0e, for non-commercial research and education. Crop calendar: GGCMI Phase 3 crop calendar v1.01, CC BY 4.0 ([licence](https://creativecommons.org/licenses/by/4.0/)), from Jägermeyr, J., C. Müller, A. C. Ruane, et al. 2021: Climate impacts on global agriculture emerge earlier in new generation of climate and crop models, Nature Food, 2, 873-885. doi:10.1038/s43016-021-00400-y; data doi:10.5281/zenodo.5062513. Changes: the calendar's planting and maturity days were mapped to the E-OBS grid, averaged over each region and rounded to whole days.

We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) and the Copernicus Climate Change Service, and the data providers in the ECA&D project (https://www.ecad.eu).

Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: An Ensemble Version of the E-OBS Temperature and Precipitation Datasets, J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200
