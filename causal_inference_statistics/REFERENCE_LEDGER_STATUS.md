# Reference ledger — materialized inventory

Generated from the current reference sections of the 20 articles.

- **300 citation occurrences**, mapped to article number and bibliography position in `reference_occurrences.csv`.
- **210 distinct citation strings**, with deterministic IDs, DOI fields and verification status in `reference_ledger.csv`.
- **3 records explicitly verified** in this PR from publisher-backed prior audit batches; **207 pending**. Previous audit documents describe further checks, but those entries are not automatically promoted without recording their individual evidence.

## Regeneration

```bash
python causal_inference_statistics/build_reference_ledger.py
```

The generator now preserves verification only when the reference ID, exact citation text and a nonblank verification source match. If the citation is edited, it reverts to `pending` until rechecked. It creates both CSV files, avoiding a separate spreadsheet workflow.

## Remaining work

Reconcile prior batch audit documents against individual ledger entries, verify all remaining unique references against publisher/catalogue records, and record URLs and notes per verified entry. The 210 entries are **distinct citation strings**; variant citations for a single work may still need manual harmonization.

Do not confuse creation of an inventory with a completed bibliography audit. Continue scientific notation and figure reviews independently once the citation process is stable.
