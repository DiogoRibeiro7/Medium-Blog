# Causal Inference series: reproducibility audit

This directory contains twenty manuscripts and twenty `partN/experiment.py` scripts.

## Inventory (audited 2026-10-19)

- Parts 1–5: `experiment.py` exists; no committed `results.csv` was found. These are **not** yet comparable with archived output.
- Parts 6–20: both `experiment.py` and `results.csv` exist.
- The inventory is a file-presence check only. It is **not** evidence that scripts have been rerun successfully or that numerical claims in the manuscripts have been checked.

## Run the audit

From the repository root, install the relevant scientific dependencies in an isolated environment. Scripts collectively require NumPy, SciPy and scikit-learn.

```sh
python causal_inference_statistics/audit_reproducibility.py --parts 6 7
python causal_inference_statistics/audit_reproducibility.py --all --timeout 600
```

The checker copies each script into a temporary directory, runs it with the current Python interpreter, then compares generated and archived CSV files using exact text/header comparison and floating-point tolerance (default `1e-8`). It reports missing files, process failures, timeouts, and numerical differences with a nonzero exit status. It does not modify the checked-in results or install dependencies. Execution times vary substantially; the `--all` option may be computationally expensive.

## Next steps

1. Generate and archive the missing outputs for Parts 1–5, adjusting those scripts to write CSVs when needed.
2. Rerun Parts 6–20 in an environment with recorded package versions. Investigate mismatches rather than rewriting CSVs to force a pass.
3. Check that each manuscript table is generated from its matching CSV.
4. Introduce deterministic assertions for known estimands, expected result schemas and analysis-specific assumptions.
5. Extend to CI in smaller batches after confirming runtime and dependency constraints.

The reproducibility audit is not complete until all twenty parts pass in a clean environment and manuscript claims are checked against the generated outputs.
