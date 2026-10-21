# Publication: the economics of the uneaten meal

A long-form Medium article drafted from the restaurant food-waste,
commercial-data-audit, and restaurant-contribution-margin research.

## Files

- \`sections/*.md\`: maintained narrative source sections.
- \`medium/article.md\`: assembled, copy-ready Medium article (not published).
- \`medium/figures/*.png\`: the ten article figures, drawn in base R by
  \`R/figures.R\` through \`scripts/make-figures.R\`. Upload them to Medium
  in the order they appear in \`medium/article.md\`.
- \`tests/test-article.R\`: verifies the assembled text, minimum 5,000-word
  length, the most important hypothetical financial calculations, and that
  every figure the article references exists.

## Run

From \`restaurant-portion-economics/\`:

\`\`\`sh
Rscript publication/tests/test-article.R
Rscript tests/test-economics.R
Rscript tests/test-profit-bounds.R
Rscript tests/test-menu-designs.R
\`\`\`

To reassemble locally if a section is edited, use
\`Rscript publication/build-article.R\`, then commit the edited sections
and regenerated \`medium/article.md\` in the same PR. CI will reject drift.

Publication is manual. No workflow posts text to Medium.

## Figures

Six figures (waste stages, contribution, tornado, feasibility map,
break-even elasticity, menu designs) need no external data. The four
empirical figures (URBAN-WASTE selection and ratio fragility, Finnish daily
series and sensitivity) need the Zenodo files, so the
\`restaurant-portion-empirical.yml\` workflow downloads and checksum-verifies
the sources, runs the analyses, redraws all ten figures and commits any
changed PNG back to the pull-request branch. Locally, after
\`scripts/download-data.R\` and the analysis scripts:

\`\`\`sh
Rscript scripts/make-figures.R
\`\`\`

## Evidence labels

- Actual measurements: Finnish school waste; URBAN-WASTE commercial CSV
  subset eligibility, sample composition and sensitivity.
- Real published industry benchmark: US respondent medians of food and
  non-alcoholic beverage spending as a share of sales.
- Hypothetical parameters: restaurant meal price, per-meal costs, avoidable
  cost fractions, customer losses, discounts, refill uptake, and
  implementation expense.
- Unidentified effects: response of demand to smaller same-price portions,
  corresponding real restaurant profit effects, and food composition/calories.

The commercial source's waste-weight units and source-to-portion
measurement-window matching remain unresolved. No physical unit or
financial benefit is inferred from its provisional ratios.

The article is research commentary and a conditional business analysis,
not a claim that restaurants should reduce portions unconditionally.
