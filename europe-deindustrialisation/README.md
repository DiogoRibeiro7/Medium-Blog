# Europe's deindustrialisation: what it was, what it cost, and whether the turn back is visible

Code and data for the article *What Deindustrialisation Cost the European Union, and What a 20% Manufacturing Target Would Require*.

The analysis has five parts.

1. **What deindustrialisation was** (`decompose.py`). Manufacturing's share of employment, its share of value added at current prices and its share at constant (2020) prices, for each member state, the EU-27, the EU-15, the United Kingdom and the United States, from 1960 where the data reach back. Two identities split the fall in the employment share into a change in the real output share and a rise in relative productivity, and the fall in the nominal share into the real share and the relative price of manufactures.
2. **External balances** (`external.py`). Goods balance and current account as a share of GDP and the net international investment position, set against each member state's change in its real manufacturing share over 2000–2024; dispersion of current accounts within the first twelve euro members.
3. **Regions** (`regions.py`). 232 NUTS 2 regions in 26 member states. Exposure is the change in manufacturing employment over 2000–2008 as a share of 2000 employment; outcomes over 2008–2023 are total and manufacturing jobs, GDP per head relative to the national mean, population, the employment rate and unemployment. Within-country least squares with standard errors clustered by country.
4. **Dependencies** (`dependencies.py`). Extra-EU imports by HS chapter and fifteen strategic HS 4-digit headings, 1999–2025: Herfindahl–Hirschman index of supplier shares, China's share, the largest supplier. Natural-gas imports by partner, 1990–2024, and energy import dependency.
5. **The turn** (`turn.py`). Industrial production, manufacturing investment and employment since 2015–2019; manufacturing's share of GDP; the growth that the target of 20% of GDP by 2035 requires, against the fastest 11-year increases in the real manufacturing share that any member state has achieved since 1995.

## Data

All inputs are public and are fetched by GitHub Actions (`.github/workflows/europe-deindustrialisation-data.yml`), which commits the snapshot to `data/`.

- `fetch.py`: Eurostat dissemination API (JSON-stat). National accounts by industry (`nama_10_a64`, `nama_10_a64_e`, `nama_10_a64_p5`), GDP and trade (`nama_10_gdp`), international investment position (`tipsii10`), current account (`tipsbp10`), industrial production (`sts_inpr_a`), energy dependency (`nrg_ind_id`), gas imports by partner (`nrg_ti_gas`), and the regional tables `nama_10r_3empers`, `nama_10r_2gdp`, `demo_r_d2jan`, `lfst_r_lfu3rt`, `lfst_r_lfe2emprt`.
- `fetch_long.py`: AMECO (European Commission) through DBnomics, for the long series from 1960: employment, gross value added at current and constant prices, current account and net exports of goods.
- `fetch_comext.py`: Eurostat Comext `DS-045409`, EU-27 imports by partner in euros, every HS chapter and fifteen HS 4-digit headings, 1999–2025.

`data/fetch_log.txt` records every request and its status.

## Reproducing

```
python -m pip install -r requirements.txt
python -m pytest -q tests
python decompose.py && python external.py && python regions.py && python dependencies.py && python turn.py
python figures.py
```

Each script writes CSV files and a `*_summary.txt` to `results/`. `check_results.py` compares regenerated summaries with the committed ones within a small tolerance; the workflow `.github/workflows/europe-deindustrialisation.yml` runs all of the above on every pull request that touches this folder.

## Points to keep in mind

- The Commission's figures (17.4% in 2000, 14.3% in 2024) are manufacturing value added as a share of GDP. `turn.py` reproduces that measure (17.4% and 14.2%). The decomposition in `decompose.py` uses shares of gross value added, which are higher because GDP includes taxes less subsidies on products.
- Constant-price shares are rescaled to equal the current-price share in 2020, so their level before 2020 is in 2020 prices and their changes are what matters.
- Germany before 1991 is West Germany. Bulgaria lacks a complete AMECO series for the 2000–2024 decomposition.
- Ireland's manufacturing value added is inflated by the accounting of multinational groups since 2015; Ireland, Luxembourg, Malta and Cyprus are shown but excluded from the cross-country fits in Part 2.
- Regional manufacturing employment for 2024 covers too few regions, so the regional outcome window ends in 2023.
- In Comext, the United Kingdom is partner `GB` and counts as extra-EU in every year, because the reporter is the EU-27 in its 2020 composition.
