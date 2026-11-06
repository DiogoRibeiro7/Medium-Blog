# Poverty survey-vintage sensitivity: exploratory citizenship panel

The existing 54-country study combines Forbes 2026 **citizenship** counts with World Bank/OWID poverty observations from different years, mainly paired with 2025 GDP/population. This creates temporal misalignment even before the separate Forbes March-versus-November count problem.

The audit script `survey_vintage_audit.py` makes the mismatch testable. It:

1. Loads the SHA-256-locked, 54-country exploratory panel through the existing validator.
2. Reports the number of countries and combined population in every observed poverty year.
3. Calculates per-country lag as **GDP observation year minus poverty survey year**, including the 2024 GDP exception for Lebanon.
4. Fits equal-country GDP-adjusted OLS/HC3 models on nested samples restricted by maximum source-year lag, recording the excluded ISO3 codes and sample size.
5. Fits an additional full-sample descriptive model with the centered source-year lag as a diagnostic covariate.
6. Writes JSON and CSV offline, refusing to overwrite existing outputs.

```bash
cd billionaires-poverty-audit
python -m unittest discover -s tests -p 'test_survey_vintage_audit.py' -v
python survey_vintage_audit.py --output results/survey_vintage_sensitivity
```

Interpretation rules:

- The lag-restricted estimates use **different country samples**. Their coefficient changes combine date-mismatch sensitivity with sample-composition changes; they do not isolate a pure time effect.
- Including source-year lag as a covariate does not turn cross-sectional association into causal identification.
- All billionaires are attributed by citizenship. This panel still uses a secondary count transcription, not the verified March 2026 full country counts and not residence counts.
- The original 64-country sample selection remains unresolved. Neither a high R² nor HC3 significance can establish that greater billionaire concentration reduces poverty.
- GDP per capita is a measure of economic development, not a clean causal intervention; survey comparability and poverty-threshold interpretation also matter.

This is additional stress testing of an **exploratory** panel. Publication of a reconstructed historical causal or residence-based claim remains blocked by the Forbes vintage and original-sample evidence gates.
