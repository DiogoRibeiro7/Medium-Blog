# Publication figures and evidence

The figure pipeline reads only the CSV tables generated from immutable World
Bank and Eurostat data contracts by the preceding empirical phases. It does
not fetch data independently or copy raw files into the article repository.

## Reproduce

Python 3.13 is the CI environment. From this directory:

```bash
python -m pip install -r requirements-figures.txt
python -m unittest discover -s figure_tests -v
python pipeline.py --fetch
python eurostat_sectors.py --fetch
python historical_shocks.py
python technology_substitution.py
python robustness.py
python figures.py
```

The publication workflow fetches and verifies source bytes using the existing
contracts, then generates all analytical inputs in a single environment.

## Figure plan

| ID | Narrative | Source table |
| --- | --- | --- |
| fig01_nominal_ppp | Economic size depends on the unit | gdp_rankings_2025.csv |
| fig02_sector_composition | Services are heterogeneous | eurostat/eurostat_sections.csv |
| fig03_volume_deflator | Nominal share changes may reflect prices | eurostat/eurostat_endpoint_changes.csv |
| fig04_gdp_wealth | GDP flow and wealth stock can diverge | gdp_wealth_endpoint_growth.csv |
| fig05_shock_benchmarks | Estimated shock losses depend on benchmark | shocks/wdi_shock_metrics.csv |
| fig06_substitution | Nominal value loss versus baseline-price output | substitution/named_scenarios.csv |
| fig07_structure_robustness | Sensitivity of pre-shock structure associations | robustness/structure_correlation_sensitivity.csv |
| fig08_classification_robustness | Concentration rank stability across classifications | robustness/classification_sensitivity.csv |

The GDP chart uses seven nonoverlapping entities, not an EU aggregate plus
member states. The wealth chart shows annualized real growth, excludes the
flagged Bulgarian baseline, retains all comparison rows in its evidence, and
reports excluded countries. The shock chart uses matched-country medians
and interquartile ranges (not confidence intervals). The substitution figure
shows hypothetical outcomes, not AI forecasts. The correlation charts
preserve sample sizes and undefined values. All choices were fixed in code;
there is no manual chart selection based on desired results.

## Publication package

Each figure is rendered as a 200-dpi PNG and an SVG; both are accompanied by
a CSV of selected original analytical records. The JSON manifest records
captions, source table paths and SHA-256 hashes, selected row counts, registry
commits and asset hashes. A chart may omit unavailable or flagged rows
under the documented rule while retaining the original selected records.

The workflow compares repeat renders byte for byte in the same environment,
checks images and SVGs, validates evidence membership in the original CSVs,
and uploads two artifacts:

- **gdp-resilience-publication-figures**: eight PNGs, eight SVGs, eight
  figure-evidence CSVs, and figure_manifest.json.
- **gdp-resilience-manuscript-evidence**: complete analytical CSV tables and
  phase summaries for the numerical audit before drafting the manuscript.

These generated files are ignored by Git in this code repository; the final
selected publication assets belong in research-articles-medium after review.
Visual inspection and the manuscript claim-to-evidence audit remain separate
steps. Determinism is checked within one environment; different versions of
fonts, dependencies, or operating systems may produce different image bytes.
