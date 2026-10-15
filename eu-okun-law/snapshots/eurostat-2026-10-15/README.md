# Frozen Eurostat snapshot, 2026-10-15

Input to [`docs/CLASSICAL_RESULTS_REVIEW.md`](../../docs/CLASSICAL_RESULTS_REVIEW.md).
Do not modify these files: the review, and the audit that precedes every command,
check them against the SHA-256 hashes below.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-15T14:49:07Z |
| Downloaded by | `eu-okun fetch --start-quarter 2008-Q1 --end-quarter 2026-Q2` in the *EU Okun empirical estimates* workflow, run 38754743850 (commit `e73ee68`) |
| Source | Eurostat Statistics API, JSON-stat 2.0: `namq_10_gdp` (updated 2026-10-15), `une_rt_q` and `lfsi_emp_q` (updated 2026-09-10) |
| Query URLs | recorded in `snapshot_manifest.json` |

| File | SHA-256 |
| --- | --- |
| `raw/namq_10_gdp.json` | `1558316778ce81b89c9d6d030c42c4a328c6d3383435cc824cea6297be397f8f` |
| `raw/une_rt_q.json` | `78bde702827a54c6c2760e8d3dae8ce8b3db9995084cf5a39610f7da19bd18c5` |
| `raw/lfsi_emp_q.json` | `b6dccbcd94215b3473bdd66c39c726b5495d14114f2a621282e83b6a7abb9c8f` |
| `processed/eu27_quarterly.csv` | `250a7dd94d3678ca9ee4f29567e771fb53aeadf89e1f2fd0155d2f31d6e0ec0d` |
| `snapshot_manifest.json` | `8a8c8a107d584ba91b61584fd29397e68ff83d82982371a35db253c0803a0464` |

The raw files are Eurostat's responses, byte for byte. The analysis commands accept
this directory as `--data-dir` (they read it and write elsewhere), for example:

```bash
eu-okun review --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/review \
  --report docs/CLASSICAL_RESULTS_REVIEW.md
```

`eu-okun audit --output-dir <dir>` writes `coverage_audit.json` into `<dir>`, so run it on
a copy rather than on this directory.

Source: Eurostat. Eurostat data may be reused with acknowledgement of the source;
see Eurostat's copyright and reuse policy.
