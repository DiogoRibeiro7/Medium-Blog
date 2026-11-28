# OECD health spending and outcomes

Reproduction code for the article on health spending, treatable mortality, efficiency rankings and pandemic excess deaths in OECD countries.

## Run

```
pip install -r requirements.txt
./run_all.sh
```

Everything is downloaded from the OECD SDMX API at run time (health expenditure, avoidable mortality, life expectancy, risk factors, national accounts, population, weekly excess mortality). No data files are stored in this repository.

## Scripts

- `fetch_data.py` downloads the OECD extracts into `raw/`.
- `build_panel.py` builds the country-year panel for the 38 OECD members, 2000-2023.
- `panel_models.py` fits the cross-section, two-way fixed-effects, dose-response and change-on-change models for treatable mortality.
- `panel_robustness.py` checks why the within-country effect vanishes (year effects, country trends, long differences, power).
- `efficiency_speccurve.py` computes efficiency rankings over 72 specifications (outcome, input, window, method).
- `pandemic.py` and `pandemic_all.py` relate pre-pandemic outcomes to excess deaths in March 2020 to December 2022.
- `thesis_checks.py` runs the horse race between treatable mortality, preventable mortality, spending and income, with leave-one-out and bootstrap checks.
- `figures.py` draws the five figures in `figures/`.

## Notes

- Main models use 2000-2019; the pandemic years enter only the excess-mortality analysis.
- Spending is in constant-price USD PPP per capita; mortality is age-standardised per 100,000.
- Contact: dfr@esmad.ipp.pt
