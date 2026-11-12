# What GDP per hour measures: decomposing the EU–US productivity gap

Computational companion to Part 1 of a Medium series on Europe's low-wage path. Question: when Europe is said to have "lower productivity" than the United States, which number is meant, and how much of the gap survives once the definition, the price basis, the aggregate, the hours worked and the income accounts are made explicit?

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig6_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/eu-us-productivity-gap.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

Three requests to the OECD Data Explorer SDMX API, made on 12 November 2026; `data/fetch_log.txt` records the URL, HTTP status, size and SHA-256 of each response, and the raw responses are kept gzipped in `data/raw/`.

- `OECD.SDD.TPS DSD_PDB@DF_PDB` (Productivity Database), total economy, 1990–2025: GDP and gross value added in national currency and in US dollars at current and constant (2020) PPPs, hours worked, employment, population, labour compensation including an imputation for the self-employed (`LCTOT`), and the growth-accounting contributions of ICT and non-ICT capital deepening and multifactor productivity to growth in value added per hour.
- `OECD.SDD.NAD DSD_NAMAIN10@DF_TABLE2`: gross national income (B5G), net national income (B5N), consumption of fixed capital (P51C) and GDP, national currency, current prices.
- `OECD.SDD.NAD DSD_NAMAIN10@DF_TABLE1_INCOME`: compensation of employees (D1), gross operating surplus and mixed income (B2A3G), taxes less subsidies on production and imports (D2X3) and GDP.

OECD data are reused under the OECD terms and conditions; they are not relicensed by the repository's licence.

## Definitions

- **Groups.** North-West: Austria, Belgium, Denmark, Finland, France, Germany, the Netherlands, Sweden. South: Greece, Italy, Portugal, Spain. Centre-East: Bulgaria, Croatia, Czechia, Estonia, Hungary, Latvia, Lithuania, Poland, Romania, Slovakia, Slovenia. Ireland and Luxembourg are kept apart because their GDP includes large profits of foreign-owned firms. EU25 is the EU27 without Cyprus and Malta, which are not in the Productivity Database.
- **Aggregates** are sums: GDP of the members divided by their hours, employment or population. They are hours-weighted (or population-weighted) by construction. The OECD's own EU27 aggregate is reported next to them; it differs from the EU25 sum by about two points of the US level because its hours total is not the sum of the members' hours.
- **Current PPPs** compare levels at each year's prices. **Constant 2020 PPPs** take the 2020 comparison and move each country with its own volume growth; the two coincide in 2020 by construction.
- **GNI and NNI per hour** scale GDP per hour by each country's GNI/GDP or NNI/GDP ratio (same PPP). Bulgaria's ratios stop in 2022 and are carried forward.
- **Per-capita decomposition.** ln(GDP/pop) = ln(GDP/hours) + ln(hours/employment) + ln(employment/pop), each relative to the United States; the identity is exact.
- **Group shares of the gap.** EU25 GDP per hour is the hours-weighted mean of the groups, so the gap to the US is Σ_g s_g (y_US − y_g); each group's share is its term divided by the total.
- **Income side.** GDP per hour = labour compensation (`LCTOT`, employees plus imputed self-employed) + net taxes on production and imports (D2X3) + the rest; the rest is split into consumption of fixed capital and a net remainder (net operating surplus and mixed income after the self-employed imputation; for the US it also contains the statistical discrepancy, 1.0% of GDP in 2024). GDP is used rather than the Productivity Database's gross value added because the database records US value added at market prices (GDP/GVA = 1.000) and European value added at basic prices (GDP/GVA 1.05–1.21), so a GVA comparison overstates the European gap.
- **2019–2024.** Growth of GDP volume (constant PPPs), total hours and employment for the same members, and the implied growth of GDP per hour and hours per worker (`results/growth_2019_2024.csv`).
- **Hours sensitivity.** Per-hour levels move one for one with measured hours; the summary reports how far North-West hours per worker would have to be understated to reach the US level. The OECD warns that its average-hours series are meant for comparisons over time rather than of levels.
- **Growth accounting.** Average annual contributions over 1998–2007, 2008–2019 and 2020–2023 (US data start in 1998; Italy and Spain end in 2023). The cumulative comparison sums the contributions over the 26 years.

## Caveats

- Current-PPP and constant-PPP comparisons disagree by up to 22.6 points of the US level over 1995–2024 (North-West, 1997). This is the known space–time inconsistency between price surveys and national deflators (Feenstra, Inklaar and Timmer 2015; Oulton 2015); the repository does not resolve it, it measures it.
- Payroll levies are classified differently across countries: since 2012 Statistics Sweden records most statutory employer contributions other than pensions as taxes on production, which lowers Sweden's measured labour compensation and raises its net taxes.
- In the Productivity Database, measured growth of value added per hour and the sum of capital deepening and MFP differ by up to 0.20 percentage points a year for Austria and 0.50 for Estonia (under 0.1 for the countries in Figure 5 except Austria); the cumulative comparison uses the growth-accounting sum.
- The non-labour part of value added mixes returns to capital, rents, markups and profits booked in a country by multinational groups. The split says where the gap sits in the income accounts, not why.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
