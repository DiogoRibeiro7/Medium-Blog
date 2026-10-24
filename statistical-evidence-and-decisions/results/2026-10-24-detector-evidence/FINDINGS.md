# Preregistered detector study: completed results

Run date: 24 October 2026. This is a report of a completed, reproducible
synthetic Monte Carlo experiment, not a claim of universal detector superiority.

- GitHub Actions run: [#40089779139](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/40089779139)
- Run commit: `8107000be634ba6a02f6e2dbd219e9a932082638`
- Artifact: `detector-evidence-results-40089779139` (GitHub artifact ID `12558587152`)
- Primary protocol: `PrimaryProtocol` in `evidence_synthesis.py`, unchanged
- Generated data: 100 independent streams per regime, 199 independent
  null-calibration streams, 120 deployment observations per stream
- Response scenario: `low_variance_signal`
- Evaluation regimes: `gradual_drift` and `stable`
- Decision parameters: action cost 0.6, refit cost 0.05, immediate labels
- Meaningful gain threshold: mean regret reduction of at least 0.01
  per deployment observation, compared with CUSUM

The artifact manifest's SHA-256 digests for all five CSV tables were
independently checked against the downloaded files and matched.

## Primary economic result

Mean per-observation net regret in the gradual-drift regime
(lower is better):

| Policy | Mean regret | Difference from CUSUM |
| --- | ---: | ---: |
| Frozen | 0.249472 | +0.031824 |
| Scheduled | 0.218668 | +0.001019 |
| CUSUM | 0.217649 | reference |
| GLR | 0.217060 | −0.000589 |
| BOCPD | 0.226625 | +0.008977 |

Paired GLR minus CUSUM regret was −0.000589, with adjusted
bootstrap uncertainty limits [−0.004824, +0.003344].
This interval includes zero and does not reach the preregistered
economically meaningful improvement of −0.01.

Paired BOCPD minus CUSUM regret was +0.008977, with adjusted
bootstrap uncertainty limits [+0.003687, +0.013818].
Within this synthetic scenario, the evidence points to worse economic
regret for BOCPD than CUSUM.

The adjusted intervals are approximate stream-level Monte Carlo
bootstrap intervals, conditional on this design and the calibrated
thresholds. They must not be interpreted as universal or causal effects.

## Stable-regime false alarms

| Detector | Any-alarm frequency in 100 stable streams | Simultaneous upper bound |
| --- | ---: | ---: |
| CUSUM | 11% | Not part of candidate eligibility check |
| GLR | 6% | 13.71% |
| BOCPD | 10% | 18.75% |

Neither candidate's simultaneous upper bound established compliance
with the preregistered 10% false-alarm ceiling. This is a lack of
verification, not proof that either population false-alarm rate is
above 10%.

## Preregistered verdict

- GLR: `material_gain_not_supported`
- BOCPD: `material_gain_not_supported`

Consequently, the experiment does not justify replacing CUSUM with GLR or
BOCPD in the declared synthetic setting. Complexity alone is not evidence
of better decisions.

## Scope and limitations

The study is restricted to a one-dimensional residual disagreement
score, one response mechanism, a gradual change, one intervention cost,
one retraining cost, a fixed deployment horizon, one set of calibrated
hyperparameters, and generated observations. It cannot support claims
about every drift regime, all costs, all parameter settings or real-world
causal interventions.

The null-stream alarm bound assumes exchangeability of complete
simulation-stream maxima. The regret intervals do not include
uncertainty from rerunning the threshold calibration with other random
seeds. Selection of a detector after inspecting these results would
require an independent confirmatory evaluation.

## Provenance and future directions

See `summary.json` for a machine-readable summary, original experiment
identifiers, and the verified SHA-256 digests. The full artifact includes
`models.csv`, `economics.csv`, `calibration.csv`, `context.csv`,
`verdict.csv`, `manifest.json` and `REPORT.md`.

The original GitHub Actions artifact is time-limited. This checked-in
report preserves the central numerical findings after that artifact
expires. Complete numerical reproduction remains possible via
`run_full_evidence.py`.

Any follow-up should use independent new scenarios or genuine real-world
data with a defensible outcome and utility definition, rather than
retuning detectors on the same evaluation streams.
