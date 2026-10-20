# Real-data validation protocol

Sources are downloaded directly from Zenodo, not bundled in Git:

- Finland school waste (Kuusisto, Latva & Wilén, 2026), record
  [20825233](https://zenodo.org/records/20825233), CC BY 4.0.
  Publisher file: `Food waste data school Finland 2024-2025.xlsx`,
  MD5 `3f7e610fedc1810f7e8bc1bd47103ee7`.
- URBAN-WASTE food tracking device (Eriksson & Lindgren, 2019),
  record [3355300](https://zenodo.org/records/3355300), CC BY 4.0.
  Publisher file: `Food Tracking Device dataset.csv`,
  MD5 `a88a1d7b09abd17064087e62d1704797`.

The `restaurant-portion-empirical.yml` workflow installs R and only
`jsonlite`/`readxl`, downloads publisher data, checks checksums, runs
the Finnish importer and menu-matched descriptive analysis, and archives
CSV results as workflow artifacts. It is path-scoped and manually runnable.

Run from the project root:

```sh
Rscript -e 'install.packages(c("jsonlite","readxl"))'
Rscript scripts/download-data.R
Rscript scripts/validate-sources.R
Rscript scripts/analyse-finland.R
```

Inspect the run's `restaurant-waste-source-audit` artifact and logs.
If the parser rejects the workbook, examine its actual headers and rows
before changing the aliases. A successful download alone is not a
successful empirical validation. Treat the whole run as failed until
ingestion completes and the archived tables are present.

The Finnish baseline/intervention comparisons are descriptive. Neither
a school initiative nor a restaurant kitchen-waste tracking dataset
establishes that smaller portions sold at unchanged prices increase
profitability. No demand response, restaurant menu price or cost variables
are identified here. Research conclusions must be conditional on those
missing quantities.

The raw publisher files remain excluded from Git, and the numerical
outputs have not yet been independently audited or endorsed.
