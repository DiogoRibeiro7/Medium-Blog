# Technological substitution and the durability of economic value

This phase uses observed baseline composition but hypothetical stress parameters.
It examines the distinction between economic value generated today, activity
that may be displaced tomorrow, and replacement value added that may arise
domestically. It does not estimate how likely any scenario is.

## Model

Let s be the selected activity's fraction of baseline GDP (WDI) or GVA
(Eurostat), d the displaced fraction, c the reduction in its implicit
value-added deflator, eta the demand-response elasticity, r the fraction
of displaced activity replaced domestically, and k the replacement GVA
intensity relative to the displaced baseline GVA.

Normalize baseline portfolio price and volume to one:

- Remaining deflator: P = 1 - c.
- Remaining value-added volume proxy: Q = (1 - d) P^(-eta).
- Incremental domestic replacement value added: R = d r k.
- Nominal aggregate change (% of baseline): 100 s (P Q + R - 1).
- Baseline-price aggregate change: 100 s (Q + R - 1).
- Price contribution: 100 s (P - 1) Q.

The final two contributions add exactly to the nominal change. The
baseline-price measure is a static accounting proxy, not an observed
chain-linked volume index. Replacement is assumed to be incremental
domestic GVA outside the original target, with an unchanged deflator;
it is not the gross revenue of the replacement provider. No separate
income multiplier is applied. Other domestic activity and net product
taxes (for GDP comparisons) are held constant. For the all-services
portfolio, replacement denotes new incremental domestic value added
outside the displaced baseline activity, not necessarily a different
industry.

For example, a hypothetical activity representing 20% of GVA that has
a 25% price reduction with unchanged volume reduces nominal aggregate
GVA by 5% of baseline while leaving the volume proxy unchanged. If
demand is unit elastic, the same price reduction leaves its nominal
value unchanged in this simplified model. Neither scenario establishes
a corresponding welfare change.

## Exposure weights

Eurostat A64 current-price national accounts, 2019 and 2024, EU27 as
published and 27 members. Eight separate portfolios: J, K, M, N, the
combined J/K/M/N portfolio, G/H/I, manufacturing C, and O/P/Q.
These are separate experiments, not additive impacts. Selected
business services is an analytical label, not an official knowledge
intensity or AI exposure classification. O/P/Q refers to industries,
not public ownership. Manufacturing and O/P/Q are comparison portfolios,
not presumed immune to technology. Missing, negative or inconsistent
shares are unavailable; no partial groups are used. Source flags persist.

WDI GDP shares, 2019 and 2021, 33 countries and the fixed EU27 aggregate.
Compare all services and manufacturing separately. The all-services
shock is deliberately broad and is not a realistic estimate of the
activities susceptible to AI. EU shares are GDP-weighted across all
27 members, never partial aggregates. These dates precede the identified
WDI coverage gap for the US after 2021. Do not compare WDI GDP-share
and Eurostat GVA-share impacts without acknowledging the denominator
and year differences. Country and EU rows overlap geographically and
must not be pooled as independent observations.

## Scenarios and sensitivity

Seven named scenarios include no change, pure repricing (with fixed
and unit-elastic demand), pure displacement, combined shocks without
replacement, combined shocks with partial domestic replacement,
and full domestic volume replacement at unchanged prices.

The unweighted sensitivity grid varies displacement and price cuts
(0, 20%, 40%), elasticity (0, 0.5, 1), and domestic replacement
(0, 50%, 100%) with replacement intensity fixed at 50%. The 81 grid
points do not represent probabilities, confidence intervals or empirical
estimates. The named full replacement scenario uses 100% intensity.
Reverse stress tests calculate the isolated price or volume cut necessary
to reduce aggregate nominal value by 1%, 5% and 10%, assuming no
demand response or replacement; unattainable losses are explicitly
flagged.

## Reproduce

From this directory, Python 3.12+ (standard library only):

```bash
python -m unittest discover -s tests -v
python technology_substitution.py --fetch
python technology_substitution.py
```

The existing source contracts verify the original bytes from pinned
registry commits. The workflow tests identities and deterministic replay.

Outputs under results/substitution/:

- exposures.csv: observed activity weights, denominator and coverage.
- named_scenarios.csv: interpretable scenarios for each exposure.
- sensitivity_grid.csv: 81 deterministic parameter combinations per exposure.
- reverse_stress.csv: implied cuts for predefined aggregate loss targets.
- summary.json: source revisions, counts and study scope.

## Interpretation limits

This is a direct static stress test, not a forecast of GDP, measured
occupational exposure, an input-output model, or an estimate of the causal
effect of AI. Deflator and volume factors are assumptions about value
added, not measured service prices or physical units. There is no
model of intermediate-input substitution, wage adjustment, investment,
price spillovers, profit allocation, international ownership, consumer
surplus, or downstream demand. A reduction in nominal value does not
imply that the activity becomes useless or that welfare falls. Technology
can also create new goods and services, raise productivity or improve
quality. Empirical exposure and production-network analysis require
separate evidence; do not silently infer them from the scenario grid.
