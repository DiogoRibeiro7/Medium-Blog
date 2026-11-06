# More Homes, Higher Prices? The Housing Supply Counterfactual

This is the public, executable companion to the Medium article **“More Homes, Higher Prices: The Housing Chart That Forgot Its Counterfactual.”** The manuscript is maintained separately in the private `research-articles-medium` editorial repository.

The project examines a widely circulated 2021–2025 Bulgarian chart that plots the number of completed homes against annual house-price growth in the second quarter. It checks the stated figures, explains why their coexistence is not evidence that construction fails to moderate price growth, and adds audited Portuguese and four-country Eurostat context.

## Evidence tiers

| Evidence | Source | What it can establish | What it cannot establish |
| --- | --- | --- | --- |
| Bulgaria annual completions and Q2 price growth | NSI Bulgaria, Eurostat | Descriptive change and reference-period mismatch | Effect of the new homes on prices |
| Portugal 1995–2025 newly completed dwellings | INE via Pordata | Changing annual new-build flows | Total stock change, available supply, causal price effects |
| Portugal 2021 housing use | INE/LNEC Census | Habitual, secondary and vacant stock counts | Homes immediately affordable, habitable or geographically suitable |
| Four-country annual 2015–2025 panel | Eurostat `prc_hpi_q`, `tipsho30`, `tipsho60`, `sts_cobp_a`, `lfst_hhnhtych` | Permits/households, price, real HPI and standardised price/income descriptive paths | Construction causal effect, comparable national completion flow, investment-demand effects |
| Inverse-demand model | Assumed formula with uncalibrated sensitivity | Logical possibility of rising prices despite added supply | Empirical country scenario or policy forecast |

No causal effect is estimated. The 2022 Bulgarian HPI source break, the whole-year versus Q2 timing mismatch, provisional Portuguese completion values, differences between permits and completions, and Eurostat household survey breaks are all retained rather than silently hidden.

## Figures

1. [The original Bulgaria comparison, with periods separated](figures/fig01_bulgaria_observed.png)
2. [Construction flow relative to Bulgaria's 2025 stock](figures/fig02_bulgaria_stock_flow.png)
3. [Portugal's long new-build cycle](figures/fig03_portugal_new_completions.png)
4. [Portugal's 2021 occupancy decomposition](figures/fig04_portugal_occupancy.png)
5. [Housing permits per 1,000 households in four countries](figures/fig05_european_permit_intensity.png)
6. [Real housing-price indices, 2015–2025](figures/fig06_european_real_hpi.png)
7. [Portugal permits and real HPI: an association, not an effect](figures/fig07_portugal_permits_prices.png)
8. [Contemporaneous and two-year-lag Pearson correlations](figures/fig08_correlations_by_lag.png)
9. [Price/income ratio change relative to each country's 2015 value](figures/fig09_price_income_relative.png)
10. [Hypothetical demand shocks and supply counterfactuals](figures/fig10_synthetic_counterfactual.png)

The first nine are *observational descriptions* and the tenth is explicitly **synthetic**. The second uses the **same fixed 2025 housing-stock denominator for all years**, not reconstructed historical stocks. Observed country comparisons are limited to Portugal, Spain, Germany and France; EU27 is an overlapping aggregate reference, not a fifth independent national experiment.

## Reproduce

Requires Python 3.12 (the analysis also uses only syntax supported on Python 3.10+).

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py --output figures
```

The script validates each input, renders ten PNG figures and writes `figures/results.json` with the calculated anchors, explicit limitations, figure file list and input/output SHA-256 checksums. It does not contact any network endpoint.

The four cross-country CSVs were derived earlier from archived Eurostat JSON-stat responses in the `european-housing-affordability` research project; the exact original dataset identifiers and transformation choices are recorded in `data/source_manifest.json` and in [SOURCES.md](SOURCES.md). This public companion includes the actual derived observations needed to recompute all ten figures, without requiring access to the separate private research project. Future raw-vintage refreshes are new reviews, not silent replacements.

## Read before interpreting

- Building **permits** can be followed by completions after considerable delays, or not at all.
- An increase in the national **construction flow** says little about the fraction of the existing market stock becoming accessible in a given location.
- A vacancy in a Census is not a listing for rent or sale and does not establish acceptable repair status.
- The 29% category for Portuguese 2024 foreign-born buyers measures **birthplace**, not non-resident investors or speculative transactions.
- Synthetic inverse-demand plots have no calibrated country coefficient and are not numerical predictions.
- Pearson associations from at most ten annual observations cannot identify a causal housing-supply effect, regardless of statistical sign.

The proposed identification strategy and falsification checks are recorded in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). The documented data provenance and academic sources are in [SOURCES.md](SOURCES.md).

## Editorial and reuse

Code and original figure designs: Diogo Ribeiro's research companion, 2026. Official source data remain under the publishers' reuse conditions, including attribution to Eurostat, INE/Pordata and NSI Bulgaria. Editorial prose is maintained in a separate private repository and is not copied here. Figures may be embedded by commit-pinned public links once this PR merges.
