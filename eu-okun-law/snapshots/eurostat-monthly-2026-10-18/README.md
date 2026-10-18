# Frozen monthly Eurostat snapshot, 2026-10-18

Input to [`docs/MONTHLY_IDENTIFICATION_RESULTS.md`](../../docs/MONTHLY_IDENTIFICATION_RESULTS.md)
([MONTHLY_IDENTIFICATION_PROTOCOL.md](../../MONTHLY_IDENTIFICATION_PROTOCOL.md)). Do not
modify these files: the audit that precedes the analysis checks them against the SHA-256
hashes below and re-derives the processed CSV from the raw responses
(`eu_okun.monthly_data.audit_monthly_snapshot`).

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-18T09:31:41Z |
| Downloaded by | `eu-okun fetch-monthly --start-month 2000-01 --output-dir snapshots/eurostat-monthly-2026-10-18` |
| Source | Eurostat Statistics API, JSON-stat 2.0: `une_rt_m` (updated 2026-10-16), `sts_inpr_m` (updated 2026-10-15) |
| Filters | `une_rt_m`: `SA`, age `TOTAL`, sex `T`, `PC_ACT`. `sts_inpr_m`: `PRD`, NACE `B-D`, `SCA`, `I21` |
| Period | 2000-01 to 2026-08, the last month with both series for all 27 countries |
| Query URLs | recorded in `snapshot_manifest.json` |

| File | SHA-256 |
| --- | --- |
| `raw/une_rt_m.json` | `127942bd921a8d1365a31f7872592f7a52f09eb0ed5a479047cefd18f382054f` |
| `raw/sts_inpr_m.json` | `4be88b1a6316a909f61dc439965a0c53b10ee85a236724711b2a5aaf5d28f814` |
| `processed/eu27_monthly.csv` | `7884d2f2a7a89df2cbefa159d074e8b91729b64e277d517b068291e09e462dbd` |
| `snapshot_manifest.json` | `ec2e65b6cbe8831fb99fe791702d37913c142c6e6e2cea0704faab8cc70f2e5c` |

The raw files are Eurostat's responses, byte for byte. Missing cells stay empty (one:
Estonia's unemployment rate in 2000-01, before its series starts) and Eurostat's flags
are kept in the CSV: 'b' break in series, 'd' definition differs, 'p' provisional. The
manifest also records how each country produces its monthly unemployment figure,
from Eurostat's reference metadata for `une_rt_m` (read 2026-10-18), which decides the
primary sample.

Source: Eurostat. Eurostat data may be reused with acknowledgement of the source;
see Eurostat's copyright and reuse policy.
