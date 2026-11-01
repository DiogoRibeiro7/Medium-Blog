# The Recession Ended; the Damage Did Not

Reproducible companion for **[Article #25: The Financial Crisis of 2008, the Great Recession, and America's Unequal Recovery, 2007–2019](article.md)** in the [Medium-Blog article series](../ARTICLES.md). The companion follows [the pre-2008 globalization and credit history](../financialization-before-2008/article.md).

The analytical interval ends **31 December 2019**, before COVID-19. The first **seven publication figures** use original public institutional observations. The **eighth** is a clearly labeled hypothetical bank-capital identity. None estimates a causal policy effect, default probability, or bank-specific solvency.

## Reproduce entirely offline

Use Python 3.12 or another interpreter compatible with the pinned packages. From this directory:

```bash
python -m pip install -r requirements.txt
python prepare_data.py --check
python -m unittest discover -s tests -v
python analysis.py
```

No API key, BLS connection, FRED access, or other live download is needed. Original BLS responses and FRED CSVs are checked in, the Federal Reserve SCF table is visibly transcribed, and exact SHA-256 digests are recorded in [data/sources.json](data/sources.json). The unit tests also verify the source-byte hashes. To generate a new annual panel and BLS monthly table from originals, use `python prepare_data.py`; `--check` is a non-writing byte-for-byte audit.

For independent figure regeneration without replacing the checked-in images:

```bash
python analysis.py --output /tmp/great-recession-review
```

The analysis creates **exactly eight PNGs** and **three CSV result tables**. The [Jupyter notebook](model_walkthrough.ipynb) checks the same frozen historical values, recovery dates, racial wealth ratios, and fictional capital identity without introducing an alternative dataset.

## Files

| File | Role |
| --- | --- |
| [article.md](article.md) | Long-form Medium-ready source-based historical manuscript |
| [prepare_data.py](prepare_data.py) | Typed BLS API parsing, completeness validation, FRED annualization and deterministic table generation |
| [analysis.py](analysis.py) | Figures, recovery benchmarks, SCF comparisons, hypothetical sensitivity grid |
| [model_walkthrough.ipynb](model_walkthrough.ipynb) | Executable tour of analytical choices |
| [data/sources.json](data/sources.json) | Original publishers, FRED IDs, snapshot vintages, hashes, frequencies and caveats |
| [data/raw/](data/raw/) | Nine frozen source files actually used by the scripts |
| [data/scf_median_wealth_2022_dollars.csv](data/scf_median_wealth_2022_dollars.csv) | Direct, transparent transcription of the Fed's historical SCF median net-worth table |
| [data/annual_panel_2007_2019.csv](data/annual_panel_2007_2019.csv) | Derived 13-year descriptive panel, 17 columns |
| [data/race_monthly_2007_2019.csv](data/race_monthly_2007_2019.csv) | Derived 156-month White/Black CPS unemployment data and difference |
| [results/recovery_milestones.csv](results/recovery_milestones.csv) | Four benchmark dates and their first post-recession crossings |
| [results/racial_outcomes.csv](results/racial_outcomes.csv) | Triennial SCF medians, group ratios and independently measured BLS gap |
| [results/capital_scenarios.csv](results/capital_scenarios.csv) | Fictional bank sales after markdowns under three initial capital buffers |
| [tests/test_analysis.py](tests/test_analysis.py) | Source, arithmetic, plotting and publication checks |

## Eight publication figures

| # | Figure | Measurement and boundary |
| --- | --- | --- |
| 1 | [Unemployment and recession](figures/figure_01_unemployment_and_recession.png) | BLS UNRATE, shaded NBER recession December 2007–June 2009. NBER dating is independent of plotted unemployment. |
| 2 | [Output versus jobs](figures/figure_02_output_and_jobs_recovery.png) | BEA GDPC1 Q4 2007 index and BLS PAYEMS December 2007 index, different frequencies and units; not output per worker. |
| 3 | [Black/White unemployment](figures/figure_03_black_white_unemployment.png) | BLS official seasonally adjusted CPS series and percentage-point gap; no causal attribution. |
| 4 | [Housing and debt service](figures/figure_04_housing_and_debt_service.png) | FHFA all-transactions NSA house-price index and Fed aggregate household debt-service payments ratio; not median equity. |
| 5 | [Median income and Gini](figures/figure_05_income_and_inequality.png) | Census real median household money income, now 2025 dollars, and household pre-tax money-income Gini; survey breaks matter. |
| 6 | [Rates and Fed assets](figures/figure_06_rates_and_fed_balance_sheet.png) | Effective federal funds rate and weekly Federal Reserve gross assets in trillions; assets are not net fiscal transfers. |
| 7 | [SCF racial wealth](figures/figure_07_scf_racial_wealth.png) | Federal Reserve repeated-cross-section median net worth in thousands of 2022 dollars and group ratios; uncertainty not supplied in frozen table. |
| 8 | [Hypothetical bank deleveraging](figures/figure_08_hypothetical_bank_deleveraging.png) | Invented 100-asset bank, 8% target and 8/10/12% starting equity, no new capital or price feedback. **Not historical data.** |

## Why these benchmarks?

- **NBER dates** are December 2007 peak and June 2009 trough. We do not compute recession dates from unemployment alone.
- **Real GDP:** first post-trough quarterly GDPC1 observation equal to or greater than **2007 Q4** is **2010 Q4**. This is a revised series and neither a counterfactual output trend nor GDP per person.
- **Payroll employment:** first post-trough PAYEMS observation equal to or above **December 2007** is **May 2014**. A job count is not working-age employment per person or job quality.
- **House prices:** first FHFA USSTHPI observation equal to or above **2007 Q4** is **2016 Q3**. Q4 2007 is an explicit benchmark, not a presumed nationwide peak. Seasonality and alternative indexes matter.
- **Median income:** first annual Census median equal to or above **2007** after the trough is **2016**. Income survey definitions and household composition changed.

Changing the benchmark, frequency, data vintage, or population changes the question. The dates are descriptive, not evaluations of particular administrations or interventions.

## Input provenance

Original BLS race-series data were obtained through the [BLS Public Data API](https://www.bls.gov/developers/) in two calls (2007–2016 and 2017–2019), each within the API's public request window. The aggregate unemployment, payroll, real GDP, policy rate and Fed-asset series were downloaded from the [St. Louis Fed's FRED CSV exporter](https://fred.stlouisfed.org/). The quarterly FHFA price and Federal Reserve debt-service source and annual Census income/Gini source were **copied byte-for-byte** from the already archived 31 October 2026 FRED exports in the [preceding article](../financialization-before-2008/data/raw/). All data used by Article #25 are nevertheless present locally in this companion; reproduction does not depend on that predecessor directory.

The SCF CSV transcribes fifteen values from the **median wealth by race** panel of the Fed's [2023 accessible historical data](https://www.federalreserve.gov/econres/notes/feds-notes/greater-wealth-greater-uncertainty-changes-in-racial-inequality-in-the-survey-of-consumer-finances-accessible-20231018.htm). It retains the published two-decimal precision and the 2022-dollar basis; no individual-level microdata or confidence intervals are invented.

## Measurement safeguards

1. Source snapshots are immutable and checksummed. The two generated research tables are compared byte-for-byte under `prepare_data.py --check`.
2. The script requires complete monthly/quarterly coverage without interpolation. The weekly Fed balance-sheet series is checked for expected observation counts by calendar year.
3. The main annual data panel uses arithmetic means of monthly and quarterly levels. The figures often use the original observations to retain their frequency. Stocks, flows and rates are not mixed in a fabricated common accounting identity.
4. Real household income is in 2025 dollars, SCF median wealth in 2022 dollars, GDP in chained dollars. The two household measures are not combined into the same dollar series.
5. The bank-capital illustration begins with A=100, E=k0×100, markdown ℓ×100, and required simple E/A=k*. Assuming no new equity, sales at book value and debt repayment, additional sale fraction is max(0, 1−ℓ−(k0−ℓ)/k*). This is not a risk-weighted regulatory calculation.
6. The period ends in 2019 deliberately; references to later GAO audits are retrospective institutional evidence, not extensions of the measured time series.

### What the charts cannot determine

They do not estimate TARP's counterfactual employment effect, the loss prevented by an AIG or Bear Stearns intervention, the number of foreclosures a mortgage program avoided, the benefits and costs of QE, or the cause of racial disparities. Those require identifiable treatments, comparisons, survey uncertainty, matched firm or household data, and explicit assumptions. The citations in the manuscript lead to government reports and relevant original research where such mechanisms are discussed.

This is a publication-ready manuscript in a repository. It is **not** an automatic posting to Medium.
