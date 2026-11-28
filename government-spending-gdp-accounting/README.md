# When 60% of GDP Is Not 60% of the Economy

Reproducible empirical companion to the Medium manuscript. The question is whether total government spending as a percentage of GDP measures government production, and what can be learned from the observed gap.

## Data

EU27 member countries (fixed membership), 1995–2025. Eurostat ESA 2010:

- [gov_10a_main](https://doi.org/10.2908/gov_10a_main): general government (S13), percentage of GDP, total expenditure (TE), gross value added (B1G), final consumption (P3), cash benefits (D62PAY), purchased social transfers in kind (D632PAY), compensation (D1PAY), intermediate consumption (P2), gross fixed capital formation (P51G), interest (D41PAY), total revenue (TR) and balance (B9).
- [nama_10_gdp](https://doi.org/10.2908/nama_10_gdp): GDP at market prices (B1GQ), nominal EUR millions and annual real growth.

Source acquisition: [data PR #295](https://github.com/DiogoRibeiro7/data/pull/295). Download the reviewed candidate artifact and unpack it to a separate directory. Do not mix retrieval vintages. The candidate is not yet a registered canonical dataset; pin the manifest SHA-256 independently after audit. Attribution: Eurostat, dataset codes and access timestamps in the manifest. [Reuse policy](https://ec.europa.eu/eurostat/help/copyright-notice).

## Reproduce

Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analyse.py --input /path/to/candidate --output outputs --manifest-sha256 <reviewed_manifest_sha256>
```

The program verifies all three raw SHA-256 hashes, lengths, filters and manifest identity, parses sparse or dense JSON-stat arrays, preserves flags and missing values, then produces:

- `panel.csv`: country-year observations, flags, and derived measures;
- `coverage.csv`: per-series observation counts (no imputation);
- `comparison.csv`: latest year with complete TE/B1G for all 27 countries;
- `france.csv`: time series of distinct fiscal and production concepts;
- `denominator.csv`: exact symmetric numerator/denominator decomposition using reconstructed nominal spending (approximate at source precision);
- `growth_associations.csv`: equal-country-year pooled and within-country descriptive correlations, with common complete cases for each estimand and sample sizes;
- `figures/comparison.png`, `figures/france.png`, `figures/denominator.png`, `figures/associations.png`: publication candidates;
- `run_metadata.json`: source manifest digest, selected year and methodological limitations.

## Interpretation and limitations

TE and B1G share the GDP denominator, but are different quantities. S13 gross value added is not all public-sector production: public corporations can be outside S13. The complement of S13 GVA includes other sectors and net product taxes, not an ownership-based private share. Final consumption is not total expenditure. Cash transfers redistribute income; transfers in kind may purchase market production. Categories overlap (e.g. final consumption with wages and intermediate consumption), so do not sum them as mutually exclusive expenditure components.

No causal inference is claimed. Fiscal ratios and growth can share a mechanical denominator effect; reverse causation, automatic stabilisers and omitted variables remain. Nominal GDP changes include inflation; the denominator decomposition is accounting, not an estimate of fiscal multipliers. Flags are preserved, not silently discarded; review flagged observations and sensitivity to exclusion before publication. Results and figures remain pending until the candidate is acquired and audited.
