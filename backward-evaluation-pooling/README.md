# Does stronger evidence of a change mean the change will last? A pre-registered test

Computational companion to a Medium article. It tests a conjecture left open by the companion
project [`backward-evaluation`](../backward-evaluation/): that pooling the backward-window forecast
with the expanding-window forecast, with a weight that grows with the strength of the evidence
that the past differs from the present, keeps the backward rule's gains after lasting changes and
removes its losses after temporary episodes.

The rule, the comparators, the pass/fail criteria and the eight test series were fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) and committed before the test data were downloaded;
the analysis code was committed before the data were read. The git history shows the order.

## Result

**Not supported.** On the eight new US series (primary scoring, March-December 2020 not scored):

- C1 (gains kept) holds: where the plain rule gained, the pool gained 0.132 nats in total against
  0.099 for the rule;
- C2 (losses removed) fails: where the rule lost (five series, −0.309 in total) the pool still lost
  −0.248, a reduction of a fifth, not a half;
- C3 (the weighting matters) fails: the pool's mean gain (−0.0145 nats per month) is below every
  fixed pool (0.25: +0.0434; 0.5: +0.0341; 0.75: +0.0149), although its average weight when the walk
  stopped (0.76) is close to the 0.75 pool.

The premise fails as well: across 88 triples of consecutive five-year blocks, the Spearman
correlation between rejection strength and the persistence of the change is +0.04 (p = 0.72).
Equal-weight pooling over window lengths had the largest mean gain (+0.0635).

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # run by the data workflow; snapshot committed
python sims.py --reps 300    # results/sims*.  (~10 min on 2 cores)
python premise.py            # results/premise_*
python forward.py            # results/forward_*  (Student-t check takes most of the ~10 min)
python figures.py
python -m pytest -q tests
python check_results.py results/sims_summary.txt results/premise_summary.txt results/forward_summary.txt
```

The matrix, the losses, the models and the backward walk are imported from
`../backward-evaluation/` (`backward_eval.py`, `schemes.py`).

## Files

- `PREREGISTRATION.md` - hypothesis, rule, comparators, criteria, series, settings
- `pooling.py` - evidence-weighted and fixed pools, the criteria
- `sims.py` - lasting regimes versus temporary episodes
- `premise.py` - rejection strength against persistence, block triples
- `forward.py` - the pre-registered forward test, with a Student-t(5) check
- `common.py` - series and transformations
- `figures.py`, `tests/`, `check_results.py`
