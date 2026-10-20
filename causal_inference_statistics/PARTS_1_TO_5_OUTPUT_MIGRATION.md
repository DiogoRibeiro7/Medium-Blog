# Parts 1–5: archived-results migration

Parts 1–2 now archive their existing CSV console output. Part 3 archives its grouped summary as a tidy, scenario-labelled CSV. Part 4 archives its existing CSV table. Part 5 already writes results.csv.

No numerical CSVs are fabricated or copied from article tables. To complete this audit, rerun experiments 1–5, commit generated files, then run:

```bash
python causal_inference_statistics/audit_reproducibility.py --parts 1 2 3 4 5
```

Python 3.12+, NumPy, SciPy and scikit-learn are required. Computationally intensive simulations may take time.

The missing archived outputs mean this PR is not yet a full reproducibility pass.
