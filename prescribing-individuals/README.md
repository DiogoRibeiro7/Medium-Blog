# Prescribing for a patient, not an age group

Companion code for the Medium article *Prescribing for a Patient, Not an Age Group: The Statistics of ADHD Medication in Later Life*.

One script prints every number in the article and draws both figures. Nothing in it uses patient data: it is arithmetic on published summary figures and on assumptions stated below.

## What the script covers

1. **Same average, different people.** Two hypothetical distributions of the change in systolic blood pressure on stimulant medication, both with the average rise of 2.0 mmHg reported for adults by Mick, McManus and Goldberg (2013). One is homogeneous; the other is a two-component mixture calibrated so that 10 of 64 people (the share in Michielsen et al., 2020) rise by 10 mmHg or more.
2. **Relative versus absolute risk.** The odds ratio of 1.04 per year of medication use for cardiovascular disease (Zhang et al., 2023), compounded over five years and applied to two illustrative baseline risks, 2% and 20%.
3. **The drug holiday as an experiment.** The probability that comparing weeks on and off medication points the wrong way, Φ(−d√k), for k on/off pairs and a true benefit d in units of the standard deviation of one on-minus-off difference; and the power of a one-sided paired t-test.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python prescribing_individuals.py > results/output.txt
```

Python 3.13, a few seconds. The figures go to `figures/`; `results/output.txt` is the run the article quotes. The calculation is deterministic.

## Assumptions

The distributions in section 1 are illustrative shapes consistent with two published summaries, not estimates from individual data. The baselines in section 2 are illustrative, not estimates for any age group, and the odds ratio comes from an observational study of people aged 6 to 64. Section 3 assumes independent, normally distributed weekly differences; real symptom scores are autocorrelated and expectation effects add bias when weeks are not blinded, both of which make a single unblinded comparison less reliable than these numbers suggest.
