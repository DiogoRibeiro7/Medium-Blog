# UK CPI and disposable income — historical analysis

`Untitled.ipynb` is a historical exploration of UK consumer prices, alcohol prices,
and household disposable income. It builds two figures using Office for National
Statistics (ONS) series. The figure captions refer to data through September 2022.

## Data and figures

The notebook retrieves live data from ONS:

- Consumer price indices: `mm23.csv` (selected CPI series).
- Household income: the ONS `NRJR` time series.

The last two plotting cells produce:

| Figure | Notebook code | What it shows |
| --- | --- | --- |
| Disposable income versus CPI | `plt.savefig("DisposableIncomes.tiff")` | Relative UK CPI and disposable-income indices, rebased to 1988 Q1 |
| Alcohol versus overall CPI | `plt.savefig("AlcoholCPI.tiff")` | Alcohol and all-items price indices, rebased to 1988 Q1 |

The notebook already stores rendered PNG copies of both plots in its cell
outputs. The two standalone TIFF files were additional 600-dpi exports,
not source datasets or inputs to any notebook cell.

## Storage decision (2026-10-24)

The original TIFF exports were removed from the current repository tree
because each occupied approximately 77.8 MB, or about 155.5 MB together.
The authored notebook, calculations, and its embedded figures are preserved.
These export names are excluded in the root `.gitignore` to avoid accidental
recommitment.

This is an archival notebook, not a verified reproducible software project.
The ONS URLs reference live datasets, and running the notebook again may
produce different data, figures, or errors depending on source revisions and
the historical Python environment.

For access to the exact removed TIFF assets, retrieve them from the repository
history at commit `b65c8b90541817f74da45548d3c61068c93e4f6b`:

```bash
git show b65c8b90541817f74da45548d3c61068c93e4f6b:just_fun/AlcoholCPI.tiff > AlcoholCPI.tiff
git show b65c8b90541817f74da45548d3c61068c93e4f6b:just_fun/DisposableIncomes.tiff > DisposableIncomes.tiff
```

Publication links to exact historical paths should be checked before assuming
the exported TIFF URLs are no longer needed.
