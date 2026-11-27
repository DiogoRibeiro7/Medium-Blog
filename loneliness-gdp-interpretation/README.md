# Loneliness, GDP and the limits of interpretation

Public research companion for a private editorial manuscript. This is a **methodological audit and exact synthetic demonstration**, not a replication of Gallup or an estimate of causal effects.

## Reproduce

From this folder, run:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python analysis.py
```

Three generated PNGs and their underlying CSVs appear in outputs/. The GitHub Actions workflow runs the tests and uploads the outputs. All data are constructed by the code, not collected from survey respondents.

- composition: two countries with identical financial-group rates within each country but pooled rates of 32% and 15%.
- measurement: equal hypothetical latent prevalence of 20% and different response mechanisms producing 28% and 16%. Parameters are illustrative, not calibrated.
- daily: exactly 23% each day in both scenarios, but 23 versus 100 distinct people across a week.

These examples prove non-identification from particular summaries. They do not show that the observed Gallup association is spurious. See SOURCES.md for the evidence and limits.
