# Unemployed persons, unemployment rates and GDP growth, with the EU aggregate, frozen 2026-10-21

Inputs of [AGGREGATE_PROTOCOL.md](../../AGGREGATE_PROTOCOL.md). Do not modify these
files. The audit (`eu_okun.aggregate_data.audit_eu_aggregate`) checks every raw response
against the SHA-256 recorded in `manifest.json`, and re-derives `processed/quarterly.csv`
from them.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-21T15:38:06Z |
| Downloaded by | `eu-okun fetch-eu-aggregate --output-dir snapshots/eu-aggregate-2026-10-21` |
| Period | 2008-Q1 to 2026-Q2 |
| Geographies | the 27 member states and `EU27_2020` |
| Manifest SHA-256 | `a5ccb971a68a20e3512a83b7f620be5f65fd467b09dd8adabfdf942b9dfaa1db` |

| File | Source | Updated |
| --- | --- | --- |
| `raw/unemployed_ths.json` | Eurostat `une_rt_q`, `Y15-74`, `T`, `THS_PER`, `SA` | 2026-09-10 |
| `raw/rate.json` | Eurostat `une_rt_q`, `Y15-74`, `T`, `PC_ACT`, `SA` | 2026-09-10 |
| `raw/gdp.json` | Eurostat `namq_10_gdp`, `B1GQ`, `CLV_PCH_PRE`, `SCA` | 2026-10-20 |

- **Labour force.** The labour force of a country is its unemployed persons divided by its
  unemployment rate, both from the same seasonally adjusted release.
- **Vintages.**
  - The unemployment release is the same as in the frozen study snapshot.
  - The GDP release (2026-10-20) is later than the frozen GDP series (2026-10-15). Revisions
    between them are not reconciled.
  - Only the `EU27_2020` growth rate is taken from this file.
- **Reuse.** Eurostat data are reused under its copyright and reuse policy.
