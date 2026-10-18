# Case-study inputs, frozen 2026-10-18

Inputs of [HETEROGENEITY_PROTOCOL.md](../../HETEROGENEITY_PROTOCOL.md). Do not modify these
files. The audit (`eu_okun.heterogeneity_data.audit_heterogeneity`) checks every raw
response against the SHA-256 recorded in `manifest.json`, and re-derives
`processed/quarterly.csv` and `processed/epl.csv` from them.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-18T12:48:38Z |
| Downloaded by | `eu-okun fetch-heterogeneity --output-dir snapshots/heterogeneity-2026-10-18` |
| Period | 2008-Q1 to 2026-Q2 |
| Manifest SHA-256 | `4a6a6d1c82af1e3cd3c2740246f8ff23198c9fd898bb859f15bc16c129e07a4e` |

| File | Source | Updated |
| --- | --- | --- |
| `raw/emp_dc.json` | Eurostat `namq_10_pe`, `EMP_DC`, `THS_PER`, `SCA` | 2026-10-15 |
| `raw/emp_nc.json` | Eurostat `namq_10_pe`, `EMP_NC`, `THS_PER`, `SCA` | 2026-10-15 |
| `raw/emp_a10.json` | Eurostat `namq_10_a10_e`, `TOTAL`, `EMP_DC`, `THS_PER`, `SCA` | 2026-10-15 |
| `raw/hours.json` | Eurostat `namq_10_a10_e`, `TOTAL`, `EMP_DC`, `THS_HW`, `SCA` | 2026-10-15 |
| `raw/temp.json` | Eurostat `lfsq_etgaed`, `T`, `Y15-74`, `TOTAL`, `THS_PER` (not seasonally adjusted) | 2026-09-11 |
| `raw/employees.json` | Eurostat `lfsq_eegaed`, `T`, `Y15-74`, `TOTAL`, `THS_PER` (not seasonally adjusted) | 2026-09-11 |
| `raw/temp_share.json` | Eurostat `lfsq_etpga`, `T`, `Y15-74`, `PC` | 2026-09-11 |
| `raw/pt_share.json` | Eurostat `lfsq_eppga`, `T`, `Y15-74`, `PC` | 2026-09-11 |
| `raw/oecd_epl.csv` | OECD `DSD_EPL@DF_EPL`, 2008, all versions (`EPL_R` and `EPL_T`, version 3, are used) | — |

**Concepts.**

- Domestic employment counts everyone working in resident production units, including
  non-resident commuters.
- Resident employment counts residents, wherever they work.

**Coverage.**

- `namq_10_pe` publishes no seasonally adjusted employment for BE, CZ, FR, MT, PL or PT,
  and no domestic employment for EL.
- `namq_10_a10_e` has no BE.
- The OECD EPL indices cover 20 EU countries.

**Flags.** Labour-force series carry break flags (2014-Q1 and 2021-Q1 everywhere, and
other years in some countries). Year-on-year changes that span a flag are dropped in the
analysis. Nothing is interpolated or imputed.

Sources: Eurostat (reuse with acknowledgement); OECD Indicators of Employment
Protection (CC BY 4.0).
