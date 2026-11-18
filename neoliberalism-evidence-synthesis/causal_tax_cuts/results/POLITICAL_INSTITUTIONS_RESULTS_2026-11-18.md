# Observed DPI2023 political-institutional coverage — 18 November 2026

This is a **descriptive completeness and source provenance audit** of the
same 18-country Hope–Limberg/WID/WDI tax-cut cohort. There is no new ATT,
political matching, standard error, or causal claim. The input is the
[IDB Database of Political Institutions DPI2023](https://data.iadb.org/dataset/the-database-of-political-institutions-dpi-2023),
available under CC BY 4.0, DOI
[10.60966/8tx1xs4i](https://doi.org/10.60966/8tx1xs4i).

## Exact source vintage

| Source or join | SHA-256 / verified observation |
| --- | --- |
| Original IDB DPI2023 CSV | `5b5b047b88bdb659fefc73a36764de0fc8ca9f74cbeb20a8d06d74445881d1a2` |
| Original CSV size | 5,626,170 bytes; 8,731 country-year rows, all countries |
| WID/WDI joined outcome/macro panel | `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321` |
| Original pre-censor risk sets | **20 eligible reforms; 239 donor–event pairs** |
| Donors still at risk at year five | **174 donor–event pairs** |
| Frozen risk-set reconciliation | `same_source_validated` |

Original DPI coverage begins in **1975**, whereas the paper cohort starts
in **1965**. The 18 countries × 51 study years contain 918 possible cells,
of which 180 belong to 1965–1974 and are **structurally out of source**.
The remaining 738 country-years (18 × 41) have source rows for 1975–2015;
the importer does not impute any political variables.

| Individual political field, 1975–2015 | Observed country-years / 738 |
| --- | ---: |
| Executive party economic orientation, known left/center/right | 684 |
| Executive orientation not applicable | 45 |
| Executive orientation unknown | 9 |
| Political checks / veto-player index (`checks`) | 726 |
| Executive electoral competitiveness (`eiec`) | 738 |

The 45 not-applicable observations include the Swiss collegial-executive
case. Neither these values nor the nine unknown classifications are
invented ideological categories. The three known orientations are **nominal
labels**, never a 1–3 ideology scale for regression.

## Pre-reform coverage (five years, `t-5..t-1`)

| Completeness criterion | Observed count |
| --- | ---: |
| Baseline reforms with all five orientation values | 17 / 20 |
| Baseline reforms with all five checks-index values | 17 / 20 |
| Baseline reforms with all five electoral-competitiveness values | 17 / 20 |
| Baseline reforms complete on all three controls | **17 / 20** |
| Initial donor–event pairs complete on all three controls | **206 / 239** |
| Year-five donor–event pairs complete on all three controls | **143 / 174** |
| Reforms with both treated and ≥1 year-five donor complete | **17 / 20** |

The three ineligible-to-complete histories (without altering baseline
eligibility) are Sweden 1971, Canada 1972 and the United Kingdom 1979.
Sweden and Canada precede the data source entirely; the UK has only
four pre-reform years in the official source. All other 17 baseline
reforms have complete three-variable five-year treated histories.

These counts are country–event **pairs**, not 239 independent countries
or observations suitable for naive inference. In particular, completeness
for 17 treated reform histories and at least one donor does **not** mean
that their politics were comparable, that exchangeability holds, or that
joint WDI support improved. The independent
[joint macro support audit](JOINT_SUPPORT_RESULTS_2026-11-18.md) still
finds just 2 of 20 reforms with three-dimensional donor-hull overlap at
year five. No political effect is fitted or implied here.

## Reproducibility and future work

The exact machine-readable snapshot is
[`political_institutions_2026-11-18.json`](political_institutions_2026-11-18.json).
See [`../POLITICAL_INSTITUTIONS.md`](../POLITICAL_INSTITUTIONS.md) for the
January-1 snapshot rule, exact country mapping, codebook and command-line
instructions. The source-probe workflow downloads DPI2023 on every PR,
checks the original cohort against the frozen joined-panel hash, and emits
four seven-day audit files under `tax-cut-political-institutions`.

Next, specify an identification strategy sensitive to real political
confounding, exposure timing and sparse multidimensional donor support
*before* introducing an adjusted ATT or causal interpretation.
