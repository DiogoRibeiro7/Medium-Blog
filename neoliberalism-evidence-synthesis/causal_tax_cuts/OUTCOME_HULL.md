# Outcome-augmented support for Finland 1986 and Germany 2008

Following PR #768, test whether the original three-WDI-mean geometric
support survives adding pre-reform top-1% income-share information.

The fixed donor pool, future-cut censoring and original annual onset
are unchanged. Compare macro-only, macro plus outcome mean and OLS
slope, and macro plus the complete annual outcome trajectory.
Use t-5..t-1 and t-5..t-2 outcome windows, with identical macro windows.
The shorter outcome window is a limited anticipation sensitivity,
not an alternative statutory treatment date or a fully anticipation-free
design. The latter would also require shifting macro windows and rechecking
eligibility. At horizons zero and five, use the same event-specific
pre-censor scales for every single-donor deletion.

Enumerate up to p+1 donors for p-dimensional convex feasibility.
Report support and number of deletions preserving support, not fitted
counterfactual weights. No outcome after onset enters features.
The balanced future outcome requirement and future policy conditioning
remain limitations. No policy-specific ATT, p-value or causal identification
is implied by a feasible hull. Failure means exact interpolation under
the chosen features is unavailable, not proof that all causal designs fail.

Run from neoliberalism-evidence-synthesis:

    python -m causal_tax_cuts.outcome_hull --panel PATH --output-dir OUTPUT

The source workflow uploads the 24 diagnostic rows and SHA-256 manifest.
Historical numerical results are pending the first successful live run.
