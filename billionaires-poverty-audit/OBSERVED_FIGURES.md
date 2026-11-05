# Eight figures: citizenship-based exploratory comparison

This package is an **observational sensitivity audit**, not a replication of
the original 64-country billionaire-*residence* figure and not an estimate of
the causal effect of wealth concentration.

The input `data/observed_citizenship_54_2026.csv` is the exact 54-country
matched panel recovered from [GitHub Actions run 41866282683](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/41866282683).
Its SHA-256 is
`db18001307fc89c77275f9d6147c23ce5f33b9e35e8da640bef82ce725b6fbda`.
See [CITIZENSHIP_EXPLORATORY.md](CITIZENSHIP_EXPLORATORY.md) for the incomplete
76-country secondary Forbes transcription (total 3,359), official OWID/WDI
retrieval provenance and 22 unmatched country records.

## Numerical results: n=54

| Model | Billionaire coefficient (percentage points per tenfold) | HC3 p | R² |
| --- | ---: | ---: | ---: |
| Billionaire density, no controls | −47.11 | <0.001 | 0.5471 |
| GDP per capita only | — | — | 0.8049 |
| GDP + billionaire density | −7.91 | 0.119 | 0.8113 |
| GDP + billionaire density, population-weighted | −2.76 | 0.904 | 0.7771 |
| GDP + billionaire density + GDP squared | −6.63 | 0.228 | 0.8324 |
| GDP + billionaire density + log population | −8.72 | 0.101 | 0.8130 |
| GDP adjusted, poverty observation year ≥2022 | −7.41 | 0.111 | 0.8490 (n=43) |
| GDP adjusted, poverty observation year ≥2023 | −8.65 | 0.145 | 0.8605 (n=34) |

The R² in the weighted specification measures a **different estimand**
(population-weighted rather than equal-country fit). R² in differently
restricted samples should not be compared as if nested on identical data.

GDP-only vs GDP-plus-billionaires R² differs by approximately 0.00634
*on the same 54 countries*. The billionaire coefficient in the adjusted
main model has an HC3 95% interval approximately [−17.87, +2.04] percentage
points per tenfold difference. A lack of conventional significance does
not demonstrate the causal effect is zero.

Removing each of the 54 countries one at a time produces adjusted
coefficients between approximately −10.10 and −5.96. Those negative
coefficients remain confounded observational associations.

OWID observation-year coverage:
**2020: 5, 2021: 6, 2022: 9, 2023: 25, 2024: 8, 2025: 1**.
Observation year is not necessarily household survey interview year.
Forbes citizenship counts are from 2026 and GDP comes from the most
recent available values up to 2025.

## Figures and captions

| File stem | Subject | Suggested Medium placement |
| --- | --- | --- |
| `01_raw_citizenship` | Raw billionaire-density association | Reconstructing the association |
| `02_gdp_benchmark` | GDP per capita association | Development as common driver |
| `03_model_fit` | Equal-sample R² benchmark | Interpreting R² |
| `04_gdp_partial` | GDP-adjusted partial regression | What GDP adjustment can show |
| `05_specification_curve` | HC3 coefficient and 95% CI under seven specifications | Statistical versus economic significance |
| `06_leave_one_out` | All 54 country-omission fits | Robustness and influence |
| `07_poverty_years` | OWID observation-year distribution | Survey timing and comparability |
| `08_predictor_relationship` | Correlation of GDP with billionaire density | Historical starting points |

**Alt-text template**: each chart is a country-level exploratory figure
based on a secondary Forbes citizenship transcription, matched to OWID/WDI
poverty and GDP indicators. Axes, sample and model definitions appear
within each figure. Do not substitute these plots for the original image.

## Reproduce offline

```bash
cd billionaires-poverty-audit
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_observed_figures.py' -v
python observed_figures.py \
  --input data/observed_citizenship_54_2026.csv \
  --output figures/citizenship_exploratory
```

This creates eight PNG/SVG pairs (PNG width at least 2,200 px), plus
`figure_results.json` with model results, 54 leave-one-out records,
image hashes and source-panel SHA-256. The GitHub Actions workflow archives
these outputs as a temporary artifact. Existing figure output folders are
never overwritten.

**Publication gate**: Do not publish the eight figures as proof of a
billionaire-driven poverty reduction mechanism, wealth concentration or an
exact reconstruction of the original plot. Forbes primary-list validation
and residence mapping remain unresolved.
