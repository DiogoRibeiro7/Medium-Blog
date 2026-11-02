# Causal Inference in Statistics — consolidated reproducibility audit

Audit updated: 2026-11-02.

## Current numerical reproduction status

The 20-part series has **20 primary experiment/result pairs** and **one additional Part 13 nuisance-model validation pair**. Every currently catalogued table has been independently recalculated and matched against its committed CSV output:

- **21 tables, 221 result rows** in total.
- Fixed-seed computations were reproduced in the audit environment (Python 3.12, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0).
- Reported numerical values matched the committed files within the verifiers' tolerance of `1e-8`. Committed result files were not rewritten.
- The earlier inventory dated 2026-10-19 is superseded by this audit. Its claim that Parts 1–5 lacked results or had not been rerun is no longer current.

These findings establish computational reproducibility for the covered deterministic experiments. They do **not** by themselves certify the correctness of causal assumptions, identification arguments, or every statement in the manuscripts.

## One-command verification

Run from the repository root:

```bash
# Fast consistency checks of the committed numerical outputs
python verify_causal_series.py

# Independently rerun all 20 experiments and Part 13 validation
python verify_causal_series.py --recompute
```

The consolidated verifier invokes the six specialised verification modules. Its complete recomputation mode passed on 2026-11-02.

The existing independent audit runner remains available:

```bash
python causal_inference_statistics/audit_reproducibility.py --all --timeout 600
```

## Detailed audit record

| Coverage | Reproduced rows | Evidence |
| --- | ---: | --- |
| Parts 1–5: foundations | 82 | [Foundations audit](FOUNDATIONS_REPRODUCIBILITY_AUDIT.md) |
| Parts 6–10: identification | 48 | [Identification audit](IDENTIFICATION_REPRODUCIBILITY_AUDIT.md) |
| Parts 11 and 15: longitudinal | 8 | [Longitudinal audit](LONGITUDINAL_REPRODUCIBILITY_AUDIT.md) |
| Parts 12–14: advanced | 37 | [Advanced audit](ADVANCED_REPRODUCIBILITY_AUDIT.md) |
| Parts 16–20: sensitivity | 40 | [Sensitivity audit](SENSITIVITY_REPRODUCIBILITY_AUDIT.md) |
| Part 13: additional nuisance validation | 6 | [Part 13 validation audit](part13/NUISANCE_VALIDATION_AUDIT.md) |
| **Total** | **221** | **21 numerical tables** |

The main series has one `experiment.py` and one `results.csv` per part. Part 13 also includes a separately audited validation experiment and output. The table counts describe currently checked artefacts, not every possible additional robustness experiment.

## Interpretation and remaining work

Computational reproduction and scientific validity are separate questions. The following remain appropriate before treating the series as fully scientifically reviewed:

1. **Identification and design:** review exchangeability, consistency, positivity/overlap, selection, interference, treatment-version assumptions, and design-specific conditions such as exclusion restrictions and parallel trends where relevant.
2. **Robustness:** test alternative random seeds, sample sizes, data-generating processes, nuisance-model specifications, overlap regimes, and Monte Carlo error. Fixed-seed agreement does not replace these checks.
3. **Manuscript-to-evidence alignment:** check equations, figure sources, rounding, captions, and textual claims against the experiments. In particular, verify article and Medium rendering separately from CSV reproduction.
4. **References and provenance:** complete outstanding source-ledger and bibliography metadata checks, without assuming that a reproduced numerical result independently validates its cited source.
5. **Automation:** consider a CI job for the fast checker and an optional scheduled full recomputation with a measured runtime budget; the present audit was run manually.

Additional manuscript/result and claim-evidence audits elsewhere in this directory remain complementary to this numerical verification record.
