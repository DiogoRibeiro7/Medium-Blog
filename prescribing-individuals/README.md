# Prescribing for a patient, not an age group

Companion code and data for the Medium article on age rules for ADHD medication in older adults ([`article.md`](article.md)). It asks what public data say about the three statistical arguments against prescribing by age: that an age cutoff is a proxy for measurable risks, that average drug effects hide individual responses, and that relative risks mean little without the patient's own baseline.

## Sources

| Source | What is used | Module |
| --- | --- | --- |
| NHANES 2011–2023 (CDC) | Blood pressure, treatment and cardiovascular history by age (oscillometric cycles 2017–2023), replicate readings, prescription stimulant use (all cycles) | `prescribing/nhanes.py` |
| ClinicalTrials.gov (API v2) | Amphetamine and methylphenidate trials with posted results, all indications and ages: age eligibility, participants aged 65+, blood-pressure and heart-rate outcomes by arm, adverse events by arm | `prescribing/trials.py` |
| FDA Adverse Event Reporting System (openFDA) | Report counts by age band for methylphenidate-class and amphetamine-class drugs and seven event groups | `prescribing/faers.py` |
| Medicare Part D Prescribers by Geography and Drug (CMS) | National claims and beneficiaries aged 65+, 2013 onwards | `prescribing/medicare.py` |

## Analyses

- **Baselines by age.** Survey-weighted systolic pressure, hypertension (130/80 or treatment) and cardiovascular history, with Taylor-linearised standard errors.
- **Who a stimulant would push over the line.** For adults not yet hypertensive, the share of each age group moved to 130/80 or 140/90 or above, under a uniform response and under a normal distribution of individual responses whose mean and between-person SD come from the trials (with the upper 95% limit of the SD as a sensitivity check).
- **Absolute cardiovascular risk.** For adults 40–79 without cardiovascular disease, 10-year risk from the Pooled Cohort Equations (Goff et al., 2014) at the measured systolic pressure and after a sustained rise of 2, 5 or 10 mmHg; a cross-check applies the trial-based relative risk per 10 mmHg (Ettehad et al., 2016) to each person's baseline.
- **Age as a proxy.** The spread of 10-year risk inside each age band, and an age cutoff (65 or 70) judged as a classifier of high risk (20% or more): the share of people it excludes who are not at high risk, and the share of high-risk people it lets through.
- **Variance ratios.** For each randomised parallel placebo-controlled trial posting arithmetic means with an SD or SE (change from baseline preferred, final values otherwise), in all trials and in adult-only trials, the log ratio of the SD on drug to the SD on placebo (Nakagawa et al., 2015), pooled by random effects. A ratio above 1 is evidence that people respond differently; the between-person SD of the response is SD_placebo × sqrt(VR² − 1). From the change-score trials, the expected share with a rise of 10 mmHg or more on each arm under a normal approximation.
- **Adverse events in trials.** Pooled risk ratios against placebo for insomnia, blood pressure, heart rate, appetite and anxiety terms.
- **Adverse-event reports by age.** Reporting odds ratios against all other drugs in the same age band, and the ratio of the ROR in older bands to the ROR at 18–44.
- **Medicare.** Claims for beneficiaries aged 65+ by drug class and year.
- **Measuring one patient.** Within-visit SD of replicate readings by age, the readings needed to detect a given rise, and the chance that an on/off comparison points the wrong way.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m prescribing analyse      # from the committed extracts in data/
python -m prescribing fetch        # download everything again (needs internet access)
```

The download step runs on GitHub Actions (`.github/workflows/prescribing-individuals-run.yml`): a manual run fetches the data, recomputes the results and opens a pull request. `.github/workflows/prescribing-individuals.yml` reruns the tests and the analysis on every change and checks that `results/summary.txt` matches the committed run.

## Outputs

- `data/`: tidy extracts (gzipped CSV and JSON); raw downloads are not committed
- `results/summary.txt`: every number quoted in the article
- `results/results.json`: the same results in machine-readable form
- `figures/`: the article's figures

## Limitations

NHANES is cross-sectional and its stimulant users are few, especially after 60. Trial outcome tables vary in what they post; the variance-ratio analysis uses only arithmetic mean changes with observed dispersion, which excludes model-based least-squares means. Spontaneous adverse-event reports have no denominator of people exposed and are shaped by who reports and why; the reporting odds ratio compares a drug with all other drugs within the same age band, which removes the overall level of reporting by age but not age-specific reporting habits. Medicare Part D covers people aged 65 and over and some younger disabled beneficiaries; beneficiary counts summed across brand rows can count the same person twice, so claims are the primary measure.
