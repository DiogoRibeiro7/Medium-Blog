# Causal identification: disaggregate the political hypothesis

**Question:** Can we establish that specific market-oriented reforms affected labour income, inequality, financialisation, or the affordability of housing?

**Answer:** Potentially, for sharply defined policies and outcomes. Not by reading the national time series as a single, controlled 45-year experiment. "Neoliberalism" is a historically useful umbrella term, not a binary treatment with a unique adoption date, dosage and counterfactual.

## Estimand-first design

The two articles contain at least four distinct policy hypotheses. They cannot share one identification strategy.

| Hypothesis | Treatment definition | Outcome | Candidate design | Central validity problem |
| --- | --- | --- | --- | --- |
| Cutting high-income taxes redistributes national income toward the top. | Legally defined major top-income or capital-tax reform, in country and year. | Pre-tax top 1% national-income share; GDP growth as separate outcome. | Matched country-year event study; staggered difference-in-differences with modern group-time effects. | Reforms are endogenous to economic and political conditions; anticipatory income shifting. |
| Labour institutions alter workers' share of the value produced. | A documented collective-bargaining, wage-setting or union-rights reform. | Consistent labour-share estimand and earnings percentiles. | Sector-region policy exposure, event study, or a valid narrow border design. | Simultaneous industrial change, employment composition and self-employment measurement. |
| Privatisation and financial liberalisation change housing's function as an asset. | Dated eligibility or ownership-rule changes, not a generic ideology score. | Local transaction prices, ownership mix, rents and vacancy. | Spatial discontinuity, transaction-level DiD or policy-timing synthetic control. | Selection into targeted places, neighbourhood spillovers, supply responses and investor anticipation. |
| Public and social housing protects affordability. | Staggered building or eligibility programme with measurable exposure. | Rent burden and displacement for eligible households. | Eligibility-threshold RDD or staggered municipal rollout with proper controls. | Non-random local allocation and population sorting. |

## Required empirical workflow

1. **Pre-register a unit, intervention and outcome.** Define whose welfare and what time horizon are being studied. Avoid changing from median wages to national average FTE salary after seeing the result.
2. **Document reform timing and intensity from primary legislation.** Verify dates, eligibility, geographic scope, compliance and anticipation windows. A change of government is not enough.
3. **Build an untreated or less-exposed comparison.** Use institutional heterogeneity or eligibility rules, not a presumed global "before 1980" control.
4. **State the identifying assumption.** Show pre-treatment outcomes and, where possible, event-study leads; check anticipation and spillovers. Failure to reject a pre-trend is not proof of parallel trends.
5. **Estimate an explicit estimand.** For staggered adoption, use group-time ATT or comparable robust methods rather than a potentially misleading naïve two-way fixed-effect coefficient with heterogeneous effects.
6. **Probe the specification.** Alternative labour-share definitions, household-income denominators, region-specific supply constraints, credit conditions, interest rates, migration and employment composition.
7. **Report uncertainty and negative findings.** Cluster at the treatment assignment level; show confidence intervals, sensitivity to treatment windows, placebo dates and plausible omitted confounders.
8. **Keep national extrapolation separate.** A local accommodation-policy estimate may be internally credible yet fail to quantify nationwide long-run housing changes.

## What has already been researched

Hope and Limberg (2022), [*The economic consequences of major tax cuts for the rich*](https://doi.org/10.1093/ser/mwab061), employ a cross-country matching approach for 18 OECD countries over 1965–2015. Their reported distributional results provide **narrower quasi-experimental evidence** on a defined fiscal reform family than any tax-rate/Gini line chart can. The source and limitations are audited in [`../tax-cuts-rich-historical-cycles/data/research_estimate_audit.md`](../tax-cuts-rich-historical-cycles/data/research_estimate_audit.md). Their result is neither a causal estimate for all neoliberalism nor a proof that every individual reform had an identical effect.

The 18-definition Portuguese labour-share project and the four-country housing-supply project already demonstrate why choosing an income denominator, omitting the self-employed or treating permits as homes built can materially alter a conclusion. Those are necessary **measurement checks**, not substitute causal identification.

## Decision for this companion

This PR delivers a verified **descriptive layer**, which is a prerequisite for responsible causal work. It intentionally does **not** output a regression named `neoliberalism_effect`, invent treated/control countries, manufacture standard errors for government-issued aggregates, or interpret a 1980-versus-2025 comparison as an experiment. The next research increment should choose **one policy family** with documented treatment timing and a defensible comparison group; only then should the project contain a causal estimator.
