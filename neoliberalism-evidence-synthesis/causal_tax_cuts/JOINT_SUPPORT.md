# Joint three-variable donor support: an empirical geometry audit

This is a **finite-sample pre-treatment support diagnostic** for the tax-cut
study. The [previous donor-support audit](DONOR_SUPPORT.md) asks whether
the treated country's five-year pre-reform mean falls between the minimum
and maximum of *each* surviving donor variable. Meeting all three tests
does **not** imply that a donor mixture can reproduce the treated profile.

This step tests the stronger joint condition in three dimensions. It does
not estimate a treatment effect, causal propensity, policy effect weight,
standard error or probability of treatment assignment.

## Precisely which observations are compared?

Keep every previous eligibility and censoring restriction unchanged:

- The initiating reform is a Hope–Limberg `taxcut_2SD` event, with the same
  five-year history of coded tax cuts as each potential comparator.
- The initiator and donor have observed WID top-1% income shares for the
  *entire* `t-5..t+5` window, even for the earlier evaluation horizons.
- The three WDI indicators are observed during each of the five **prior**
  years, `t-5..t-1`: real GDP-per-capita growth (`NY.GDP.PCAP.KD.ZG`),
  CPI inflation (`FP.CPI.TOTL.ZG`), and trade openness (`NE.TRD.GNFS.ZS`).
  The profile is the three-component vector of five-year means.
- A donor with a coded cut between `t` and horizon `t+h` is removed at
  that horizon, as in the prior audit. `h=-1` denotes the full pre-censor
  pool; `h=0..5` removes same-year and later donor reforms cumulatively.

The analysis uses the previous 29 observed reform years, 20 supported
baseline reform episodes and 239 pre-censor donor–event pairs. No later
WDI measurement is used to define any profile, and horizons differ **only**
through the donor risk set. Like all earlier diagnostics, the complete
future WID outcome requirement may create selection.

## Joint feasibility versus three separate ranges

Let \(\boldsymbol{x}_t\in\mathbb{R}^3\) be the treated country's
pre-reform macro mean, and \(\boldsymbol{x}_j\) the corresponding mean
for eligible donor \(j\) at the same event year. We ask if there is a
nonnegative combination of surviving donors such that

\[
\boldsymbol{x}_t = \sum_{j=1}^{m_h} w_j\boldsymbol{x}_j,
\qquad w_j\ge 0,\quad \sum_{j=1}^{m_h}w_j=1.
\]

This is **convex-hull membership**. It is stricter than merely checking
that each coordinate of \(\boldsymbol{x}_t\) lies within the donors'
coordinate-wise minima and maxima. For example, donors at
\((0,0,0),(1,0,0),(0,1,0),(0,0,1)\) span all three marginal intervals
`[0,1]`, but the target `(0.8,0.8,0.8)` lies outside their tetrahedron.

By [Carathéodory's theorem](https://library.slmath.org/books/Book51/files/07teissier.pdf),
any point in the convex hull of vectors in \(\mathbb{R}^3\) can be
represented by at most **four donors**. The code therefore enumerates
donor subsets of size one to four, solves a small barycentric linear
system and accepts only nonnegative weights that sum to one and reconstruct
the profile within the declared tolerance. This is a certificate of
**geometry, not a fitted or estimated synthetic-control weight vector**.

To keep units comparable, coordinates are centered at the treated vector
and divided by the population standard deviation across the treated and
*full pre-censor donor* profiles, separately for each event. A constant
dimension receives scale 1.0. The scale never changes as future reforms
remove donors. Convex-hull membership is invariant to invertible positive
rescaling, although numerical tolerances naturally depend on the scale.

Two thresholds make boundary sensitivity visible:

| Status | Certificate requirement |
| --- | --- |
| `inside_strict` | Maximum absolute standardized coordinate error at most `1e-9`; all weights nonnegative within numerical tolerance |
| `tolerance_sensitive` | Fails strict but passes `1e-6`; **not robust joint overlap** |
| `outside` | Fails both tolerances with at least one surviving donor |
| `no_donors` | No surviving donor; support fields are null |

The algorithm's limited weights and donor IDs are recorded solely so that
another researcher can reconstruct each geometric certificate. They must
not be used for counterfactual outcomes without a separate identification
and estimation design. This is three-dimensional support of **averages**,
not support of the full 15-dimensional yearly prehistory.

## Reproduce from the official joined source

From `neoliberalism-evidence-synthesis/`:

```bash
python -m causal_tax_cuts.joint_support \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --provenance data/wdi-source/world_bank_provenance.json \
  --frozen-reference causal_tax_cuts/results/donor_support_2026-11-15.json \
  --output-dir data/tax-study/joint-support
```

The command writes:

- `joint_support_event_horizons.csv`: seven rows per baseline-eligible
  reform; horizon risk set, earlier marginal flag, strict and relaxed
  convex-hull flags, status, witness donor IDs, barycentric weights and
  maximum reconstruction residual.
- `joint_support_audit.json`: horizon totals, eligibility, source and joined
  panel SHA-256 hashes, official WDI archive dates, declared thresholds,
  methodological caveats and reconciliation to the previous donor audit.

The event counts, donor–event risk sets, and marginal-support totals must
**exactly** match those from `donor_support.py` on the same joined input.
If a previously frozen reference has an identical panel hash, those
counts must match its archived values too; otherwise the report explicitly
marks `source_vintage_changed` instead of asserting equivalence.

The source-probe GitHub Action regenerates the CSV and JSON from the live
WID/WDI join, and uploads the seven-day `tax-cut-joint-macro-support`
artifact. The initial verified counts are frozen in
[`results/JOINT_SUPPORT_RESULTS_2026-11-18.md`](results/JOINT_SUPPORT_RESULTS_2026-11-18.md).

## Interpretation: necessary checks, not identified effects

Joint donor-hull membership is still **not causal positivity**. It does
not establish that all potential treatment assignments have positive
probability, that treated and donors are exchangeable conditional on these
variables, or that a weighted counterfactual is identified. The three WDI
means omit political, institutional, fiscal, distributional, capital-flow
and international-shock covariates. Donor–event pairs are repeatedly drawn
from a small set of countries; they are not independent units.

Additional treatment-history and policy-timing limitations remain as
documented in [DYNAMIC_EXPOSURE.md](DYNAMIC_EXPOSURE.md): treated paths can
undergo repeat cuts while comparator paths are censored for future cuts.
If empirical joint support is sparse, simply fitting more aggressive
weights or extrapolating from distant donors would not remove those
identification problems. A defensible next phase first needs an explicit
intervention, contemporaneous institutional variables, correct sequencing
and an inference design that respects country-level dependence.
