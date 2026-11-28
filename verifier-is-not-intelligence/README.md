# A verifier is not an intelligence

Research companion to an independent Medium manuscript on RLVR, evaluation, deployment and AI-risk claims. This is a **methodological analysis**, not a training run, benchmark of existing models, jobs forecast or estimate of existential risk.

## Reproduce

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python analysis.py
```

The three PNGs and CSVs in outputs/ are generated deterministically from stated assumptions. The workflow uploads them as an artifact.

1. Verifier selection: a perfect verifier improves selection, whereas a deliberately misleading binary verifier increases acceptance from 60% toward 100% while true quality falls from 55% toward 41.67%. Both start from 55% true quality. This is first-accepted IID search, not a simulation of RL training. It is an existence counterexample, not an estimate of verifier error in current LLMs.
2. Serial reliability: under independence and all-steps-required assumptions, 99% per-step success gives about 36.6% over 100 steps. Recovery, correlation and parallel paths can invalidate this product.
3. Review cost: 15 minutes generation plus 20 review plus a 90-minute rework episode with probability q; versus 60 minutes manual. Break-even q = 25/90, about 27.8%. Time is not full economic cost or safety.

No external dataset is necessary: all rows are generated mathematical scenarios. SOURCES.md distinguishes empirical studies from the illustrative companion.
