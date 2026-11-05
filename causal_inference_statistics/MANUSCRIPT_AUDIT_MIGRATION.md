# Causal-series manuscript audit after private migration

The 20 full manuscripts have been copied into private `DiogoRibeiro7/research-articles-medium` (the Part 18–20 synchronization was merged in private PR #7). The public `Medium-Blog` repository retains the executable science, archive CSVs and reference-ledger evidence.

The following historical tools read complete manuscript text; they must not treat a future public compatibility notice as evidence:

- `audit_references.py`
- `build_reference_ledger.py`
- `check_manuscript_tables.py`

Each now accepts `--manuscript-root` pointing to the `causal_inference_statistics` directory in a **locally checked-out private manuscript repository**. For example, from the public repository root:

```bash
python causal_inference_statistics/check_manuscript_tables.py --parts 1 2 3 --manuscript-root /path/to/private/causal_inference_statistics
python causal_inference_statistics/audit_references.py --manuscript-root /path/to/private/causal_inference_statistics
python causal_inference_statistics/build_reference_ledger.py --manuscript-root /path/to/private/causal_inference_statistics
```

The current default remains the public source directory while full manuscripts remain there. Once they are replaced with historical-path notices, use the explicit private manuscript root. A notice produces an actionable error instead of a misleading successful audit or an empty bibliography. No credentials, tokens, or private manuscript contents are copied into the public repository.

Do not run the ledger generator solely to test migration: it writes `reference_ledger.csv` and `reference_occurrences.csv`, which are existing verification artifacts. Use the read-only table checker or unit tests when validating access first.

This dependency change precedes the separate public cleanup PR. The public repository will continue to retain historical Git commits, which the migration does not erase.
