# Eurostat release of 2026-10-22, for the vintage check

Input to the vintage check of [MEASUREMENT_PROTOCOL.md](../../MEASUREMENT_PROTOCOL.md),
section 4. The study's estimates use the frozen snapshot of 2026-10-15; this one is used
only to measure revisions since then. Do not modify these files: the audit
(`eu_okun.audit.audit_snapshot`) checks them against the hashes in
`snapshot_manifest.json`.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-22T08:39:42Z |
| Downloaded by | `eu-okun fetch --start-quarter 2008-Q1 --end-quarter 2026-Q2 --output-dir snapshots/eurostat-2026-10-22` |
| Source | Eurostat Statistics API, JSON-stat 2.0: `namq_10_gdp` (updated 2026-10-20), `une_rt_q` and `lfsi_emp_q` (updated 2026-09-10) |

| File | SHA-256 |
| --- | --- |
| `raw/namq_10_gdp.json` | `f5a386554ce80c00b92a6c60cd5c20d46fe1d2065022d076b17c93dc962a5615` |
| `raw/une_rt_q.json` | `78bde702827a54c6c2760e8d3dae8ce8b3db9995084cf5a39610f7da19bd18c5` |
| `raw/lfsi_emp_q.json` | `b6dccbcd94215b3473bdd66c39c726b5495d14114f2a621282e83b6a7abb9c8f` |
| `processed/eu27_quarterly.csv` | `d82140f3ef41c3ce366ec1a75959969b1e7a90363d5374e9b58e1882e0ca8324` |
| `snapshot_manifest.json` | `58feb8e65a3aee374fee55af9c3eaa421f0e3a4781004cb0b014659b4c2f6244` |

- **What changed since the freeze.**
  - The `une_rt_q` and `lfsi_emp_q` responses are byte-identical to those of the frozen
    snapshot. Eurostat has published no labour-force release since 2026-09-10.
  - Only GDP differs: the 2026-10-20 release replaces the 2026-10-15 one.
- **Historical vintages.** Eurostat publishes none for these country series.
- **Reuse.** Eurostat data are reused under its copyright and reuse policy.
