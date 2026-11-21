# Taxing empty homes in Portugal: statutory rates, detection and expected cost

Reproducible companion for the standalone Medium article **Taxing Empty Homes in Portugal: Statutory Rates, Detection and the Expected Cost of Vacancy** (November 2026).

## Question

Portugal's IMI code already triples the property-tax rate on dwellings vacant for more than a year and multiplies it by ten in designated housing-pressure zones. In 2026 the government instead cut the income tax on rents from 25% to 10% for almost every residential lease. This companion puts both levers into the same owner decision and prices them in euros, then asks what the surcharge is worth once the probability of being found is counted, and what the rate cut costs per lease it adds.

Nothing here estimates a causal effect of either measure in Portugal. Four layers are kept apart:

1. **Observed counts.** Census 2021 vacant stock (IHRU Table 11); vacant properties flagged by municipalities for tax year 2023; Vancouver's Empty Homes Tax monitoring series; UTAO's annual cost of the 2026 rate cut. Sources and caveats in [`data/SOURCES.md`](data/SOURCES.md).
2. **Statutory arithmetic.** IMI at 1×, 3×, 10× and 20× for one dwelling with an assumed taxable value of €150,000 and an assumed rate of 0.35%.
3. **A decision identity.** The owner lets when the private value of keeping the home empty, *U*, satisfies *U* ≤ *F* + *π*·*ρ*·(*m* − 1)·*B*, where *F* is the net rental flow, *π* the probability of being classified vacant and assessed, *ρ* the probability that the surcharge survives legal challenge, *m* the multiplier and *B* ordinary IMI. The rate cut raises *F* by at most 15% of the rent received.
4. **Scenario grids.** Expected surcharge against *π*; cost per additional lease against the unknown number of additional leases; audit sample sizes; the old eligibility × response grid. None is a forecast.

## Reproduce

Python 3.10 or later, from this folder:

```bash
python -m pip install -r requirements.txt
python run.py          # standard library only; writes results/
python figures.py      # matplotlib; writes figures/*.png
python -m unittest discover -s tests -v
```

Everything runs offline from the transcribed files in `data/`. Each loader checks its file against the cited values and stops if anything differs.

## Outputs

- `results/summary.txt`: every number the article quotes, in plain text; `results/summary.json`: the same, machine-readable.
- `results/tax_scenarios.csv`: IMI, surcharge and the largest *U* consistent with letting, with and without the rate cut.
- `results/expected_surcharge.csv`: expected surcharge for *π* from 0 to 1 at 3×, 10× and 20×.
- `results/cost_per_additional_lease.csv`: UTAO's annual cost divided by a range of additional leases.
- `results/audit_sample_sizes.csv`: random-audit sizes needed to estimate the legally vacant share.
- `results/sensitivity.csv`: the 248,534-dwelling eligibility × response grid (arithmetic only).
- `figures/fig01_census_to_flagged.png` to `figures/fig05_vancouver_declared_vacancy.png`.

## What the numbers say, and do not

- 12,776 vacant properties were flagged in 64 municipalities for tax year 2023, against 723,215 vacant dwellings in the 2021 Census and 248,534 vacant for reasons other than sale or rent in buildings in good condition. The ratios (1.8% and 5.1%) divide counts with different units (property vs dwelling) and years, and are given for scale.
- For the illustrative flat the 3×, 10× and 20× surcharges are €1,050, €4,725 and €9,975 a year; the rate cut adds at most €1,458. At *π* = 5.1% the expected 10× surcharge is €243. The 10× surcharge matches the cut's bound at *π* ≈ 31%; the 3× surcharge never does.
- UTAO puts the cut at €279.1 million a year in IRS (€308.9 million with the IRC measure). It costs more per additional lease than the €10,800 annual rent of the illustrative flat unless it brings about 25,843 leases (28,602 with IRC) that would not otherwise exist.
- A simple random audit of about 2,400 addresses estimates the legally vacant share of the 248,534 group to ±2 points at 95%.
- Vancouver's declared vacancy rate fell from 1.18% (2017) to 0.49% (2024). These are the tax's own monitoring data, not an estimate of its effect.

## Licence

Apache-2.0, matching the surrounding repository.
