# Publication companion: the economics of the uneaten meal

The article manuscript and its seven editorial sections are maintained in
[the private research-articles-medium repository](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/themes/operations-and-decisions/restaurant-portion-economics).
Access requires permission from the repository owner.

This public project retains the reproducible scientific analysis and publication
assets, without copying the private editorial text back into the public repository.

## Public files

- `medium/article.md`: historical-path notice, **not** the article text.
- `medium/figures/*.png`: ten figures generated in base R by
  `R/figures.R` through `scripts/make-figures.R`.
- `tests/test-article.R`: independent regression checks for economic
  calculations and presence and PNG signatures of ten research figures.
- `build-article.R`: optional assembler that reads and writes only inside
  an explicitly supplied private repository checkout.

## Run public tests

From `restaurant-portion-economics/`:

```sh
Rscript publication/tests/test-article.R
Rscript tests/test-economics.R
Rscript tests/test-profit-bounds.R
Rscript tests/test-menu-designs.R
```

To assemble the editorial manuscript privately, after checking out the
private repository with appropriate access:

```sh
Rscript publication/build-article.R /path/to/research-articles-medium
```

This command reads the seven private sections in
`themes/operations-and-decisions/restaurant-portion-economics/sections/` and
writes `article.md` in the same private folder. It does **not** modify
the public compatibility notice or publish to Medium.

## Figures and evidence

Six figures (waste stages, contribution, tornado, feasibility map,
break-even elasticity, menu designs) need no external data. Four empirical
figures (URBAN-WASTE selection and ratio fragility, Finnish daily
series and sensitivity) need Zenodo files. The
`restaurant-portion-empirical.yml` workflow downloads and checksums the
source data, runs the analyses and regenerates the research PNGs.

Locally, after downloading original sources and running the analyses:

```sh
Rscript scripts/make-figures.R
```

Measured data, published cost benchmarks and hypothetical financial
parameters remain explicitly distinguished in the public computations.
The source's commercial waste-weight units and measurement-window matching
remain unresolved. No physical units or financial effects are inferred
from provisional ratios.
