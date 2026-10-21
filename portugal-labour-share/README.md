# Portugal's labour share under eighteen definitions

Companion code for the Medium article *Portugal's Labour Share Under Eighteen Definitions*.

The labour share of Portuguese GDP is computed under six treatments of self-employed income and three denominators, from Eurostat national accounts for 1995–2024. The script tests each version for a trend and for a break whose date is chosen by the data. It splits the fall around 2012 between the government sector and the rest of the economy, and repeats the break search for Spain, Italy and Greece.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python labour_share.py
python figures.py
python -m pytest -q tests
```

Python 3.13. The analysis takes a few seconds. It writes:

- `results/summary.txt`: every number quoted in the article
- `results/variants.csv`: level, change, trend and p-values for each of the eighteen shares
- `results/shares_long.csv`: the eighteen annual series
- `results/breaks_portugal.csv`, `results/break_selection.csv`, `results/break_profile_raw_market.csv`: break tests
- `results/cross_country.csv`: Portugal, Spain, Italy and Greece
- `figures/fig1_…png` to `figures/fig4_…png`: the article's figures

All randomness is seeded per test, so a rerun with the pinned versions reproduces the committed `summary.txt`. The workflow in `.github/workflows/portugal-labour-share.yml` checks this on every change to the folder.

## Definitions

Codes are ESA 2010 items. All values are current prices.

**Numerators.** Compensation of employees (D.1) plus the part of gross mixed income (B.3g, the income of the self-employed) that each rule counts as labour:

| Rule | Labour income of the self-employed |
|---|---|
| `raw` | none |
| `employment` | self-employed persons × compensation per employee |
| `hours` | self-employed hours × compensation per employee hour |
| `mixed_all` | all of B.3g |
| `mixed_two_thirds` | two thirds of B.3g |
| `mixed_rest` | B.3g × D.1 / (D.1 + B.2g), the split of the rest of the economy |

**Denominators.**

| Denominator | Definition |
|---|---|
| `market_prices` | GDP (B.1gq) |
| `factor_cost` | D.1 + B.2g/B.3g, i.e. GDP less taxes net of subsidies on production and imports (D.2 − D.3) |
| `factor_cost_ex_government` | factor cost less general government factor income (government B.1g − D.29 + D.39, which equals government D.1 plus its operating surplus); government compensation is also removed from the numerator |

For the government-excluded `employment` and `hours` variants, the imputed wage of the self-employed uses economy-wide compensation per employee, because government employment is not in the snapshot.

## Inference

- **Trend.** OLS slope, with a Newey–West p-value (3 lags) and an AR(1) parametric bootstrap p-value. The bootstrap simulates the no-trend null with the AR(1) fitted to the trend model's residuals and compares the OLS t-statistic.
- **Break.** Supremum F over every candidate year with 15% trimming, under three specifications: a shift in the mean, a level shift on a trend, and a level shift plus a slope change on a trend (the interrupted time-series form). The naive p-value is the HAC p-value of the shift at the selected year, as if the year had been fixed in advance. The bootstrap p-values compare the supremum F with its distribution under the no-break trend model. The AR(1) errors in that null take their persistence either from the no-break model's residuals or from the selected break model's residuals.
- **Government.** The change in the factor-cost share between two years is split exactly into a change inside government, a change in the rest of the economy, and a change in the government's weight.

## Data

The snapshot in `data/` was downloaded from the Eurostat SDMX 2.1 API on 21 October 2026. `fetch.py` holds the exact keys:

- `nama_10_gdp`: GDP and its income components, Portugal, Spain, Italy, Greece
- `nama_10_a10_e`: total employment and employees, persons and hours, Portugal
- `gov_10a_main`: general government value added, compensation, other taxes and subsidies on production, Portugal
- `nasa_10_nf_tr`: gross operating surplus and gross mixed income, total economy and households, Portugal

2025 is in the snapshot but provisional and is not used. Missing values of government D.29 and D.39 are read as zero.

To check the snapshot against the current Eurostat vintage, run `python fetch.py --out live --compare`, or trigger the workflow manually. Its `live-eurostat` job downloads the current data, reports revisions, reruns the analysis and keeps the outputs as an artifact.

## License

Apache-2.0, as the rest of the repository.
