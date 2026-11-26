# Minimum convex approximation error

Following PR #770, retain the original Finland 1986 and Germany 2008
donor pools at horizons zero and five. Use only t-5..t-1 outcomes and
three WDI macro means, with the same pre-censor standardization.

Compare two prespecified loss functions: mean squared standardized
outcome residual and the sum of mean squared standardized outcome and
macro residuals (equal block weights). Enumerate simplex faces and
verify first-order optimality independently. Report minimum objective
norm, separate block RMSE, raw outcome RMSE in percentage points,
feature-level residuals, donor weights, effective donor count, maximum
weight, and maximum leave-one-donor-out objective norm.

Weight vectors need not be unique; concentration and out-of-objective
balance for outcome-only fits describe the selected certificate, not
all minimizers. This is an exploratory pre-fit comparison, not an
acceptance test: no universally valid RMSE threshold is assumed.
Before estimating effects, independently justify any fit threshold,
test held-out pre-period prediction and resolve the policy definition,
anticipation, confounding and future donor selection issues.
No post-treatment outcome contrasts or causal effects are computed.

Run from neoliberalism-evidence-synthesis:

    python -m causal_tax_cuts.convex_approximation --panel PATH --output-dir OUTPUT

The live workflow produces eight configurations and a source hash.
Empirical values are not asserted until the workflow runs.
