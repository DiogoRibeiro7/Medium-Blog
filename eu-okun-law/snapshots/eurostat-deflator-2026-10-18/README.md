# Frozen GDP-deflator snapshot, 2026-10-18

Input to [SIGN_RESTRICTIONS_PROTOCOL.md](../../SIGN_RESTRICTIONS_PROTOCOL.md). Do not modify
these files. The audit (`eu_okun.sign_restrictions.audit_deflator`) does two things:
- it checks them against the SHA-256 hashes below;
- it re-derives the processed CSV from the raw response.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-18T11:13:42Z |
| Downloaded by | `eu-okun fetch-deflator --output-dir snapshots/eurostat-deflator-2026-10-18` |
| Source | Eurostat `namq_10_gdp`, `B1GQ`, `PD_PCH_PRE_NAC` (implicit deflator, % change on the previous quarter, national currency), `SCA`; dataset updated 2026-10-16 |
| Period | 2008-Q1 to 2026-Q2, all 27 countries (1998 values, none missing) |

| File | SHA-256 |
| --- | --- |
| `raw/namq_10_gdp_deflator.json` | `8a3ab55cad92a85037333c0928e9e31e76de06be66875fff751ee0a929524e2a` |
| `processed/eu27_deflator.csv` | `153d4dda988cbe1d64d42fbff26ec1b317aa4dbc03e146026a6f1389d7992db4` |
| `manifest.json` | `30900e56883b8a6e59518534db92a108ad185374ba7f8db08a2f4c1c068869c8` |

**Flags.** Eurostat's flags are kept and are not used in estimation: 102 values are 'p'
(provisional) and 2 are 'b' (break in series).

**Vintage.** This release is one day later than that of the GDP growth in
`snapshots/eurostat-2026-10-15` (`namq_10_gdp` updated 2026-10-15). Any revisions between
the two releases are not reconciled.

Source: Eurostat. Eurostat data may be reused with acknowledgement of the source.
