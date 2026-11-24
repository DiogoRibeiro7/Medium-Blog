# Outcome-augmented donor support: 24 November 2026

Source: [successful workflow 45293924362](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/45293924362),
artifact 14875866039 (tax-cut-outcome-hull), PR #769.
The frozen JSON records all 24 configurations from the official WID/WDI
panel with SHA-256 `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.

| Case | Horizon | Donors | Macro hull | Macro + level/slope | Macro + trajectory | Macro hull deletion successes |
|---|---:|---:|---|---|---|---:|
| FIN 1986 | 0 | 16 | Yes | No | No | 16/16 |
| FIN 1986 | 5 | 8 | Yes | No | No | 5/8 |
| DEU 2008 | 0 | 16 | Yes | No | No | 16/16 |
| DEU 2008 | 5 | 16 | Yes | No | No | 16/16 |

Every row has the same classification with or without the last
pre-onset outcome observation. Eight macro-only configurations pass;
all sixteen outcome-augmented configurations fail. Exact geometric
interpolation fails under these specified features, not necessarily
under every possible causal design. No causal effect, inference or
policy attribution is established. The original donor selection,
future-outcome availability and policy-censoring limitations remain.

## Next design gate

Quantify the minimum standardized convex approximation error and its
feature-specific residuals before considering an approximate synthetic
control. Predefine relative feature scaling and an acceptable pre-fit
criterion independently of post-treatment outcomes. Compare outcome-only
fit against joint macro/outcome fit; report the trade-off, donor
concentration and leave-one-out sensitivity. If no defensible pre-fit
exists, retain the negative identification finding rather than report
an effect from an unsupported counterfactual.
