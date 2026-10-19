# Value added and hours by industry, frozen 2026-10-19

Inputs of [GDP_MEASUREMENT_PROTOCOL.md](../../GDP_MEASUREMENT_PROTOCOL.md). Do not modify
these files. The audit (`eu_okun.gdp_measurement_data.audit_gdp_measurement`) checks every
raw response against the SHA-256 recorded in `manifest.json`, and re-derives
`processed/quarterly.csv` from them.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-19T19:54:34Z |
| Downloaded by | `eu-okun fetch-gdp-measurement --output-dir snapshots/gdp-measurement-2026-10-19` |
| Period | 2008-Q1 to 2026-Q2 |
| Manifest SHA-256 | `7cc75f218c9e190d5a2846b1bb637510066cd784323db60a4c466f21ae025de7` |

| File | Source | Updated |
| --- | --- | --- |
| `raw/va_clv.json` | Eurostat `namq_10_a10`, `B1G`, `TOTAL`, `CLV20_MEUR`, `SCA` | 2026-10-19 |
| `raw/va_clv_k.json` | Eurostat `namq_10_a10`, `B1G`, `K`, `CLV20_MEUR`, `SCA` | 2026-10-19 |
| `raw/va_cp.json` | Eurostat `namq_10_a10`, `B1G`, `TOTAL`, `CP_MEUR`, `SCA` | 2026-10-19 |
| `raw/va_cp_k.json` | Eurostat `namq_10_a10`, `B1G`, `K`, `CP_MEUR`, `SCA` | 2026-10-19 |
| `raw/hours.json` | Eurostat `namq_10_a10_e`, `EMP_DC`, `TOTAL`, `THS_HW`, `SCA` | 2026-10-15 |
| `raw/hours_k.json` | Eurostat `namq_10_a10_e`, `EMP_DC`, `K`, `THS_HW`, `SCA` | 2026-10-15 |

Value added is published for all 27 countries; hours for 26 (not Belgium). The value-added
vintage (2026-10-19) is later than the frozen GDP series (2026-10-15); revisions between
them are not reconciled. Eurostat data are reused under its copyright and reuse policy.
