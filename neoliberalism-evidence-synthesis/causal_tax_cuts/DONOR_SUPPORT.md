# Future donor reforms and empirical macro support

This is a **diagnostic of donor availability and selection**, not a new
matched treatment effect. The prior [macro matching audit](MACRO_MATCHING.md)
requires donor countries to experience no major tax cut from the treated
reform year `t` through evaluation horizon `t+h`. That rule is coherent with
a counterfactual of *no new major cut*, but it selects controls using policy
decisions that happen **after** the treated country's reform. Such decisions
can respond to the same economic and political conditions as the outcome.

This audit measures how much donor support changes solely because of that
future-reform restriction. It does not silently reclassify later-treated
donors as valid untreated controls, and it does not estimate an ATT.

## Fixed eligibility and horizon-specific censoring

For each reform and candidate donor, keep the previous analysis's baseline
conditions fixed:

1. The treated country has no major cut at `t-1`, and the donor has the
   identical binary tax-cut history at `t-1` through `t-5`.
2. Both countries have an **observed WID top-1% income share** in every year
   `t-5` through `t+5`, without interpolation. This full window is required
   even when evaluating horizon zero.
3. The three **World Bank WDI** variables—GDP-per-capita growth, inflation,
   trade openness—are observed for both countries in every year `t-5`
   through `t-1`. Only those five pre-reform observations enter profiles.

The pre-censor pool includes every donor satisfying these restrictions,
regardless of whether it later reforms. The horizons have precise meanings:

| Horizon | Donor rule |
| --- | --- |
| `-1` | Full pre-censor pool; a donor may reform at `t` or later |
| `0` | Remove donors with a tax cut in `t` |
| `1` through `5` | Cumulatively remove donors with a cut during `t` through `t+h` |

The complete observed outcome window and pre-reform macro data stay fixed
between horizons. This isolates the **additional attrition caused by the
future-reform rule** from horizon-specific WID availability, although the
fixed complete-outcome-window requirement itself is another source of
possible selection.

## What the macro support metrics mean

For each reform, calculate the mean of each of the three WDI series in the
five years before reform. Measure the Euclidean distance between treated
and donor profiles after standardising each variable by the population
standard deviation of the **treated profile plus the entire pre-censor donor
pool**. Hold those three scales fixed at every horizon. The nearest-donor
distance therefore changes only if the donor pool changes, not because the
denominator is recalculated. It is not the ridge-Mahalanobis score from the
earlier nearest-neighbour matcher.

For each dimension, check whether the treated pre-five-year mean lies within
the surviving donors' minimum–maximum interval. `within_all_three_marginal_ranges`
means that **all three separate conditions** hold. This is only a necessary,
not sufficient, condition for joint support: a point can fall within each
one-dimensional range yet lie outside the donors' three-dimensional convex
hull. The check does **not** identify a positivity probability or establish
exchangeability; even perfect measured overlap would leave political,
institutional, and time-varying confounding unresolved.

## Reproduce the source-backed audit

From `neoliberalism-evidence-synthesis/`, after the
[World Bank source importer](MACRO_CONTROLS.md) has produced the joined panel:

```bash
python -m causal_tax_cuts.donor_support \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --provenance data/wdi-source/world_bank_provenance.json \
  --output-dir data/tax-study/donor-support
```

Outputs:

- `support_reform_eligibility.csv`: every observed reform, exclusion reason,
  and donor counts before censoring and at horizons zero and five.
- `future_donor_exposure.csv`: each baseline donor–reform pair, the first
  possible donor reform within `t..t+5`, its censoring horizon, and its
  **fixed-scale pre-reform** distance.
- `support_by_horizon.csv`: the seven risk sets per baseline-eligible reform,
  donor losses, three separate marginal range flags, their conjunction,
  nearest distance, and range violations in fixed standard-deviation units.
- `donor_support_audit.json`: aggregate counts, horizon summaries,
  methodological caveats, joined-panel SHA-256, original WDI archive hashes
  when supplied, and reconciliation to the **previous five/five matchable
  reform identifiers**. The program refuses to save on a mismatch.

The GitHub source-probe workflow regenerates these outputs from the same
live WID/WDI join as the earlier diagnostics, and publishes them for seven
days under `tax-cut-donor-support`. Always compare source checksums before
contrasting audit runs from different dates.

## Interpretation and next gate

The [first real-data donor-support report](results/DONOR_SUPPORT_RESULTS_2026-11-15.md)
freezes the initial risk-set attrition and marginal support measurements.
Donor–reform pairs are **not independent**: a country can appear in several
reforms' candidate pools. Neither pair counts nor nearest distances are
inferential sample sizes or standard errors.

The subsequent [dynamic-exposure feasibility audit](DYNAMIC_EXPOSURE.md)
now explicitly counts additional major cuts **in the treated paths** as
well as the censored donor paths, and describes two different candidate
policy strategies. It does **not** justify causal treatment weighting.
Joint covariate overlap, political and institutional confounders, policy
timing and a credible identification/inference strategy remain unresolved.
