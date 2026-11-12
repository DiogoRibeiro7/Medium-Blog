# Capital deepening in Chinese manufacturing and in the Chinese economy, 1995–2017 (and what came after)

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. Part 2 of the series stated a conjecture: that the growth of value added per worker in Chinese manufacturing after 2013 came with faster growth of capital per worker, so that value added per unit of capital fell while value added per worker rose, and China's edge moved from cheap labour to cheap capital. This repository tests it.

It measures output per worker, capital per worker and output per unit of capital for Chinese manufacturing (19 industries) and for the rest of the Chinese economy from the China Industrial Productivity (CIP) database; compares Chinese manufacturing with manufacturing in eleven European countries and Japan; repeats the economy-wide comparison with the OECD's productive capital stocks and the investment-to-growth ratio; shows how China's investment was allocated across sectors; and computes how much the manufacturing result depends on that allocation, with a break-even share of national investment. For the years after the CIP database ends it chains the efficiency indicators that China's National Bureau of Statistics publishes for industrial enterprises.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD snapshot and CIP workbooks (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig3_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/china-capital-deepening.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

`data/fetch_log.txt` records the URL, size and SHA-256 of every file.

- **CIP 4.0** (China Industrial Productivity database, 2023 release, RIETI; Harry X. Wu and co-authors): 37 industries, 1987–2017. Files 1.2 (value added), 1.4 (capital and labour compensation), 1.6 (producer prices), 2.1 (investment, current yuan), 2.2 (net capital stock, 2000 yuan), 2.3 (capital input index), 3.1 (employment), 3.2 (hours), 4 (growth accounts, including real value-added growth by industry) and 5 (classification), read as published from `data/raw/cip/`. RIETI holds the copyright of the database, so the workbooks are not redistributed here: `python fetch.py --out data --only cip --verify` downloads them from RIETI and checks each against the SHA-256 recorded in `data/fetch_log.txt` (the snapshot used for the article was downloaded on 12 November 2026), and the workflow does the same before it reruns the analysis. Everything in `results/` is derived from them.
- **OECD Data Explorer** (SDMX API, 12 November 2026):
  - `DSD_EO_LTB@DF_EO_LTB` (Economic Outlook long-term baseline): productive capital stock in volume (`KTPV_AV`), GDP in volume and potential employment for China and 13 comparison economies, 1990–2025;
  - `DSD_NAMAIN10@DF_TABLE1_EXPENDITURE`: GDP at current prices and in volume and gross fixed capital formation at current prices;
  - `DSD_NAMAIN10@DF_TABLE9A`, `@DF_TABLE6` and `@DF_TABLE7`: net fixed assets, value added and employment of manufacturing (ISIC Rev. 4 section C) in eleven European countries, Japan and the US.
- **NBS published figures** (`data/nbs_published.csv`), entered by hand with the URL of each release: revenue per 100 yuan of assets, revenue per employee, total assets and the profit rate of industrial enterprises above designated size, 2018–2025, each with the NBS's change on a comparable basis; and fixed-asset investment in manufacturing and in total for 2013 and 2017.

OECD data are reused under the OECD terms and conditions; CIP data belong to RIETI and the database's authors. Neither is relicensed by the repository's licence.

## Definitions

- **Manufacturing** is CIP industries 6 to 24. **Real value added** is a Törnqvist aggregate of the CIP's real value-added growth by industry, weighted by nominal value added and expressed in 2000 yuan; the **capital stock** is the sum of the CIP's real net capital stocks in 2000 yuan; employment and hours are sums. The CIP's total investment is within 1% of the OECD's gross fixed capital formation for China in 2007, 2013 and 2017, and its total value added within 3.1% of GDP.
- **Output per unit of capital** = real value added ÷ real net capital stock; its growth is the growth of output per worker minus the growth of capital per worker. Growth rates are average annual log changes in percent.
- **Capital deepening contribution** = capital compensation share of value added (average of the two end years) × growth of capital per hour; the remainder of output-per-hour growth contains total factor productivity and changes in labour and capital quality.
- **Investment-to-growth ratio (ICOR)** = gross fixed capital formation ÷ GDP, averaged over a period, divided by average real GDP growth. Not computed when growth is below 0.5% a year.
- **Allocation scenarios.** With depreciation δ, real manufacturing investment is K(t) − (1 − δ)K(t − 1) and the implied deflator is nominal investment ÷ real investment; at δ = 8% the CIP manufacturing stock implies a deflator that stays close to 1 (0.96 in 1995, 1.08 in 2017; 2000 = 1), at 6% one that drifts upwards (1.04 to 1.31). Each scenario gives manufacturing a different share of China's total investment in 2014–2017, deflates it with the implied deflator and accumulates it from the 2013 stock. The **break-even share** is the constant share at which output per unit of capital in manufacturing would have grown by zero over 2013–2017. Scenarios for δ = 6%, 8% and 10%.
- **NBS chained indices**: for each year, level ÷ (level − change on a comparable basis) gives the change on unchanged coverage; chained from 2017 = 100. Assets per employee = revenue per employee ÷ revenue per 100 yuan of assets × 100.

## Caveats

- The CIP allocates China's gross fixed capital formation to 37 industries; its 4.0 release distributes the gap between investment reported by industry and capital formation in the input-output tables across industries, which the authors say changed the capital stocks substantially. The manufacturing result is sensitive to that allocation; the break-even analysis says how sensitive.
- NBS fixed-asset investment is not gross fixed capital formation: it includes land purchases and second-hand assets and has exceeded capital formation by a wide margin since the 2010s. Its industry shares are used only as a reference for the scenarios, never as a level.
- The CIP's real value added uses the CIP's own deflators, which give lower growth than the official accounts; capital stocks are at 2000 prices, so ratios of capital to output are not comparable in level with ratios at current prices for other countries. Comparisons with Europe are of growth rates.
- The NBS indicators cover industrial enterprises above designated size (mining, manufacturing and utilities), use revenue rather than value added and total assets (including receivables and inventories) rather than fixed capital. They show the direction after 2017, not a test of the conjecture.
- The OECD's capital stock for China is the OECD's own estimate for its long-term model, not an official Chinese statistic.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
