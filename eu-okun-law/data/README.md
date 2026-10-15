# Eurostat data contract

The fetch command creates `raw/` JSON-stat responses,
`processed/eu27_quarterly.csv` and `snapshot_manifest.json`.
Those outputs are not committed by default. Snapshots that published results
depend on are frozen under `../snapshots/` instead (for example
`snapshots/eurostat-2026-10-15`, used by the classical results review).

The manifest records the exact query URLs, retrieval date, source
update dates, unmodified response SHA-256 hashes, country coverage,
observation and missingness counts and quality-flag counts. Archive
an immutable snapshot alongside any published numerical estimates,
because Eurostat revises previously released observations.

GDP: `namq_10_gdp` (`B1GQ`, `CLV_PCH_PRE`, `SCA`).
Unemployment: `une_rt_q` (`Y15-74`, `T`, `PC_ACT`, `SA`).
Participation: `lfsi_emp_q` (`ACT`, `Y15-74`, `T`, `PC_POP`, `SA`).

The reference start is 2009 Q1. Downloaded 2008 quarters provide
lag and difference history for GDP; the labour-force series mostly start in
2009 Q1 (Eurostat's break-corrected back series), so the first usable
unemployment change is 2009 Q2 for 25 countries. Never silently interpolate missing
measurements, discard Eurostat status flags or treat 2021 survey
definition changes as confirmed comparable observations.

Sources: https://doi.org/10.2908/namq_10_gdp ;
https://doi.org/10.2908/une_rt_q ;
https://doi.org/10.2908/lfsi_emp_q .
