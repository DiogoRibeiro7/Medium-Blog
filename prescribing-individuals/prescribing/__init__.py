"""Data and analysis for the article on age rules for ADHD medication.

Four public sources, each with a fetch step and an analysis step:

- NHANES (CDC): blood pressure, hypertension treatment, cardiovascular history,
  replicate readings and prescription stimulant use by age, 2011 to 2023.
- ClinicalTrials.gov: adult stimulant trials with posted results, used for the
  age eligibility of the evidence base, a variance-ratio test of response
  heterogeneity and adverse-event risk ratios against placebo.
- FDA Adverse Event Reporting System (openFDA): reports for methylphenidate-class
  and amphetamine-class drugs by age band and event group.
- Medicare Part D Prescribers by Geography and Drug (CMS): national stimulant
  claims and beneficiaries aged 65 and over, 2013 onwards.
"""

__version__ = "0.2.0"
