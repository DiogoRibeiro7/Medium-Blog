# Real-data quality review (20 October 2026)

Evidence: successful [GitHub Actions run 39450536083](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/39450536083), artifact `restaurant-waste-source-audit`. Original Zenodo files passed publisher MD5 verification.

In the Finnish 155-day source, day 136 (intervention; menu 11; 340 diners) contains discrepancies:

| Measure | Recorded kg | Published g/diner | Derived from kg and diners |
| --- | ---: | ---: | ---: |
| Plate waste | 11.633 | 8.2500 | 34.2147 |
| Kitchen and serving waste | 2.410 | 21.7684 | 7.0882 |

Neither measure is automatically corrected. The R audit compares the supplied rates, rates recomputed from mass and diner totals, and exclusion of discordant days.

From the original CI artifact, the published-rate menu-matched comparisons are:

| Measure | Baseline matched g/diner | Intervention g/diner | Descriptive change |
| --- | ---: | ---: | ---: |
| Plate | 23.2875 | 18.2279 | -21.73% |
| Kitchen/serving | 22.6022 | 10.7209 | -52.57% |

A separately calculated mass-derived sensitivity (recomputing per-diner values from kilograms and diners for all days) yields approximately **-16.34%** for plate waste and **-55.81%** for kitchen/serving waste. These are *not* corrected or causal estimates. Reproduce them using `Rscript scripts/audit-empirical-results.R`, compare with the exclusion sensitivity, and audit any additional missing mass measurements.

This is a school food-waste *art intervention*, not an experiment where commercial restaurants reduced meal portions while retaining prices. It supplies evidence about observed waste, **not** demand, restaurant margin or causal portion-size effects. The discrepancy must be transparently reported in a Medium article.

The real commercial-sector URBAN-WASTE source also passed checksum validation. Its CSV schema will now be inspected and stored, but no unknown numeric field will be assumed to mean kilograms or euros until validated against a codebook.
