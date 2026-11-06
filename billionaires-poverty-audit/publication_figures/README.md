# Publication-ready figures: billionaire density and poverty

These PNG images are preserved copies of the reproducible `citizenship_exploratory` output from the GitHub Actions run [42203993729](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/42203993729). They are stored in the repository so editorial work does not depend on an expiring Actions artifact.

## Scope and interpretation

- The panel comprises **54 countries** using **Forbes billionaire counts by citizenship**, with available poverty and GDP data. It is an independently assembled, exploratory sensitivity analysis.
- **These figures do not reproduce the original social-media chart**, which reported 64 countries and a billionaire-residence measure. Do not describe them as a replication of that chart or as evidence that billionaire concentration reduces poverty.
- The analysis reports a raw billionaire-density model R² of 0.5471 versus GDP-only R² of 0.8049 and joint-model R² of 0.8113. With GDP controlled, the billionaire-density coefficient is -7.912, HC3 p = 0.119. The descriptive relationship is not causal.
- Fixed panel SHA-256: `db18001307fc89c77275f9d6147c23ce5f33b9e35e8da640bef82ce725b6fbda`.
- The original downloaded artifact ZIP SHA-256 was `10e8bd507f927dad710318fbf2bf881292e6c54fba949679673b894f65a508ef`.

## Suggested editorial order

| Figure | Suggested use |
| --- | --- |
| [01 — Raw citizenship association](01_raw_citizenship.png) | Introduce the exploratory cross-country association and its limitations. |
| [02 — GDP benchmark](02_gdp_benchmark.png) | Compare the billionaire-only association with development. |
| [03 — Model fit](03_model_fit.png) | Illustrate R² across models fitted to identical countries. |
| [04 — GDP-partial association](04_gdp_partial.png) | Show the adjusted relationship after controlling for GDP. |
| [05 — Specification curve](05_specification_curve.png) | Discuss model and sample sensitivity. |
| [06 — Leave-one-out](06_leave_one_out.png) | Document influence of individual countries. |
| [07 — Survey years](07_poverty_years.png) | Explain poverty-survey vintage limitations. |
| [08 — Predictor relationship](08_predictor_relationship.png) | Discuss how development and billionaire density covary. |

The main Medium article should prioritize figures 01–05, with figures 06–08 as supplementary evidence if space permits. Keep captions explicit about **citizenship** versus **residence**, sample size, and observational identification. The private article draft lives in `DiogoRibeiro7/research-articles-medium`; only code, data, and reproducible figures belong in this public repository.

## Reproduction

Run the repository's billionaire-poverty audit workflow or use the scripts under `billionaires-poverty-audit` to regenerate the exploratory outputs. The source artifact also includes SVG versions, a figure manifest with dimensions and image hashes, and the fixed country panel. These PNGs are the portable, Medium-ready copies.
