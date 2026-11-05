# Causal Inference in Statistics — complete archived-results inventory

Inventory date: 5 November 2026. Scope: `causal_inference_statistics/part1` through `part20` on `main`.

## Coverage

| Artifact class | Tables | Data rows | Verification level |
| --- | ---: | ---: | --- |
| Primary experiment `results.csv` | 20 | 215 | Previously reproduced by the consolidated numerical audit |
| Supplemental experiment/validation CSVs | 12 | 94 | Present and structurally inspected; **not all independently rerun** |
| **All archived numerical CSVs** | **32** | **309** | Structural inventory only for this full set |

This resolves a scope ambiguity in `REPRODUCIBILITY_AUDIT.md`: its **21 tables and 221 rows** refer to the 20 primary archives plus the six-row Part 13 nuisance validation, **not** to the full set of supplemental CSVs now committed. The earlier reproduction claim remains valid for those 21 tables; it must not be expanded implicitly to the 32-table inventory.

The 12 supplemental tables are:

| Part | Additional archived CSVs |
| --- | --- |
| 1 | `coverage_targets.csv` |
| 2 | `randomization_uncertainty.csv` |
| 3 | `matching_diagnostics.csv` |
| 4 | `weight_sensitivity.csv` |
| 5 | `population_coefficients.csv` |
| 7 | `weak_iv_inference_results.csv` |
| 8 | `rd_bias_corrected_results.csv`, `rd_falsification_results.csv` |
| 9 | `sc_preperiod_holdout_results.csv` |
| 13 | `validation_results.csv` |
| 15 | `msm_validation_results.csv` |
| 20 | `validation_results.csv` |

Parts 6, 10–12, 14, 16–19 currently have only their primary CSVs; some contain additional analytical scripts or tests that do not archive CSVs.

## Automated checks

Run from repository root:

```bash
python causal_inference_statistics/check_causal_series_inventory.py
python -m unittest discover -s causal_inference_statistics -p 'test_check_causal_series_inventory.py'
python verify_causal_series.py
```

The first command checks that every part has an article and primary experiment, that the expected **32** CSVs have valid unique headers and nonempty, structurally complete rows, and that newly added supplemental CSVs are registered. It does not verify numeric identities, finite moments, causal identification, source provenance, or whether a companion script can reproduce its CSV. The second command tests failure detection. The third command is the existing primary-result consistency verifier.

For full primary-result reproduction, use `python verify_causal_series.py --recompute` with the declared numerical dependencies. The additional 11 supplemental CSVs outside the prior Part 13 validation audit merit their own fresh-execution comparison before claiming complete series-wide computational reproduction.

## Review boundary

The migration of manuscripts into the separate private article repository is not part of this audit. The inventory enumerates the existing scientific-code repository only and does not alter article manuscripts, generated values, or CI configuration.
