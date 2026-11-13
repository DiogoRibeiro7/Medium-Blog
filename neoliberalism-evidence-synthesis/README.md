# Neoliberalism, work, distribution and housing: an evidence audit

A reproducible empirical companion to two Medium essays by Diogo Ribeiro:

1. [The Forty-Five-Year Experiment: What Neoliberalism Did to Work, Wealth and Everyday Life](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/economics-and-society/neoliberalism-forty-five-year-experiment/article.md)
2. [The Housing Crisis Is a Wage Crisis—and a Crisis of Neoliberalism](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/housing/housing-wage-crisis-neoliberalism/article.md)

The manuscripts live in a separate private editorial repository. This **public** folder holds the underlying measurements, assumptions, code, tests, and publication-ready figures. It integrates existing, carefully attributed analyses rather than silently fetching new data or treating political ideology as a single randomised experiment.

## What the data show

### 1. National housing prices versus average salaries, 2015–2025

Eurostat's nominal national house-price index (`prc_hpi_q`) and annual average full-time-adjusted employee salary (`nama_10_fte`) both start from 100 in 2015. Their ratio is a *relative change in the price-to-salary index*, **not** an estimate of how many salaries buy a house.

| Country | House-price change | Nominal average salary change | Change in house-price-to-salary index | Eurostat price-to-disposable-income index change |
| --- | ---: | ---: | ---: | ---: |
| Portugal | +163.8% | +58.5% | **+66.5%** | +60.9% |
| Spain | +80.6% | +29.3% | +39.7% | +24.5% |
| EU27 aggregate | +61.9% | +38.7% | +16.7% | +6.8% |
| Germany | +52.7% | +41.0% | +8.3% | +1.6% |
| France | +27.3% | +24.8% | +1.9% | −5.6% |

*Sources and caveats:* [Eurostat housing and salary definitions](SOURCES.md); values are reproduced from the [generated summary](results/summary.json). EU27 is a weighted aggregate that **overlaps** the four countries. `nama_10_fte` is an **average gross employee salary per full-time equivalent**, not a median worker's wage, a household's disposable income, or a regional earnings distribution. The official Eurostat price/income measure has a different denominator: adjusted household disposable income per person. The two indicators must not be interchanged.

### 2. Portugal's labour share is definition-sensitive, 1995–2024

Using Eurostat income-side national accounts and GDP at market prices:

| Treatment of self-employment income | 1995 | 2024 | Change |
| --- | ---: | ---: | ---: |
| Employees' compensation only | 46.8% | 47.7% | **+0.9 percentage points** |
| Impute self-employed labour income from employee headcount | 59.0% | 56.3% | **−2.7 pp** |
| Impute it from hours worked | 59.4% | 55.2% | **−4.2 pp** |
| Treat all gross mixed income as labour | 62.1% | 58.0% | **−4.2 pp** |

The complete underlying study tests **18 definitions**, distinguishes HAC from autoregressive-bootstrap inference, and is available in [`../portugal-labour-share`](../portugal-labour-share). A one-line claim that "Portugal's labour share has steadily fallen" would be unjustified by these endpoints.

### 3. The Reagan-era U.S. tax changes and inequality

The highest U.S. statutory marginal individual income-tax rate was **70% in 1980** and **28% in 1988**. The U.S. Census household money-income Gini rose over the longer 1967–2018 sample, but the **1993 survey redesign introduces a comparability break**. The historical tax, Gini, GDP-growth and private-investment series appear in separate figures, without a regression line or causal coefficient. See [`../tax-cuts-rich-historical-cycles`](../tax-cuts-rich-historical-cycles) for the original sourced analysis and the audit of published tax-reform research.

## Figures and Medium captions

The five high-resolution PNGs are built from the frozen datasets by [`figures.py`](figures.py):

1. [House prices and average salaries, 2015–2025](figures/fig01_nominal_house_prices_vs_salaries.png)
2. [Sensitivity to the definition of income](figures/fig02_housing_income_denominator_sensitivity.png)
3. [Portuguese labour share under four defensible definitions](figures/fig03_portugal_labour_share_definitions.png)
4. [U.S. top statutory rates and survey inequality, 1964–2018](figures/fig04_us_tax_and_inequality_context.png)
5. [U.S. GDP growth and private investment, 1964–2018](figures/fig05_us_growth_and_investment_context.png)

Suggested precise captions, placement in the essays and alternative text are in [FIGURE_GUIDE.md](FIGURE_GUIDE.md).

## Reproduce entirely offline

Requires Python 3.10 or newer. From this folder:

```bash
python -m pip install -r requirements.txt
python analysis.py --check
python -m unittest discover -s tests -v
python figures.py --output /tmp/neoliberalism-evidence-figures
```

To regenerate the committed analytical outputs intentionally:

```bash
python analysis.py
python figures.py
```

`analysis.py --check` verifies the **SHA-256 of every frozen input** from `data/manifest.json` and compares its derived CSV and JSON with the committed results byte for byte. The figures are generated offline and validated as PNGs in unit tests; rendered pixels are not compared across operating systems and font-library versions.

The [source manifest](data/manifest.json) pins the source-repository revision and records the single substantive extraction: a balanced subset of the existing Eurostat FTE salary snapshot (five geographies, 2015–2025). All other CSV files are unmodified copies of public, already documented research outputs. No invented data and no external API calls are involved.

## Identification and what **cannot** be concluded

The code estimates **no causal effect** of neoliberalism, Reagan, Thatcher, taxation, financial liberalisation, bargaining institutions or wages on housing. National prices can rise relative to wages because of supply, credit, land constraints, migration, non-wage demand, public provision, monetary policy and institutional changes, often interacting with one another.

The story can make an empirically supported **distributional argument** without claiming one historical ideology is a single identifiable treatment. See the [claim-by-claim evidence audit](CLAIM_AUDIT.md) and [identification protocol](CAUSAL_PROTOCOL.md) for the distinction between measured descriptive changes, published quasi-experimental studies and causal claims that remain open.

The original source snapshots retain their publishers' usage rights; they are not relicensed as original datasets by the repository's source-code licence.
