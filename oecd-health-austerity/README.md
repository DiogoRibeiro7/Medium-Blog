# OECD health spending cuts and mortality: an event study

Reproduction code for the article on the health spending cuts after 2009 and what happened to treatable mortality and life expectancy in the countries that made them.

## Run

```
pip install -r requirements.txt
./run_all.sh
```

Everything is downloaded from the OECD SDMX API at run time (health expenditure, avoidable mortality, life expectancy, risk factors, national accounts, population). No data files are stored in this repository. The run takes about a minute.

## Scripts

- `fetch_data.py` downloads the OECD extracts into `raw/`.
- `build_panel.py` builds the country-year panel for the 38 OECD members, 2000-2023.
- `panel_models.py` fits the cross-section and two-way fixed-effects models used as a benchmark in Figure 5.
- `austerity_event.py` runs the event study: the series-break screen, dose-response long differences, the cut-versus-other permutation test, synthetic controls with in-space placebos, the robustness checks and the placebo-in-time test.
- `austerity_figures.py` draws the six figures in `figures/`.

## Notes

- Dose: change in real health spending per person (constant-price USD PPP) between 2009 and 2013, in log points x 100. A cut is a fall of at least 5%.
- Outcomes: treatable, preventable and avoidable mortality (age-standardised, per 100,000, logged) and life expectancy at birth, changes measured from 2009.
- Six countries are set aside by the series-break screen (a one-year gap of more than 12 points between spending growth and GDP growth in 2008-2014); the robustness table keeps them in.
- Contact: dfr@esmad.ipp.pt
