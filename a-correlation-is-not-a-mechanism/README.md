# Article #35 — A Correlation Is Not a Mechanism

This directory develops a causal-identification article with nine planned synthetic experiments. **Currently implemented: studies 1–3 only.** The manuscript, figures, remaining studies and scientific audit are not complete.

## Reproduce current studies

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py --output-dir /tmp/causal-identification-study
```

The analysis writes six CSVs (replicate-level and summary for each study) to the explicit output directory; it does not modify tracked evidence files. Default seed is `20261102` with 300 replications per study, each using a stable study-specific NumPy SeedSequence and independent child streams. Results are synthetic and are not empirical causal estimates.

Current checks compare Monte Carlo means with theoretical omitted-variable bias, an exact Simpson reversal, and the conditional regression slope induced by a Gaussian collider. A summary `monte_carlo_se` measures uncertainty in the simulation mean; `sampling_sd` measures dispersion of individual estimates. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the remaining scope and publication acceptance gates.
