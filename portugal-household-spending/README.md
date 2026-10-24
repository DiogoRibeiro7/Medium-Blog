# Portuguese household spending on basic needs, saving, and basic-needs inflation

Analysis code for a follow-up to *portugal-wage-counterfactual*: what share of household spending goes to basic needs, how that differs by income and by type of household, what it amounts to in euros, how much households save, and how fast each group's basket has risen in price.

- National accounts, 1995–2022: household consumption by COICOP purpose. Core basic needs are food (CP01), actual rent (CP041), dwelling repairs (CP043), water and other dwelling services (CP044) and electricity, gas and fuels (CP045); the wider set adds health (CP06), running personal transport (CP072) and transport services (CP073). Imputed rent (CP042) is removed from numerator and total, since it is not a cash outlay. The COICOP breakdown is on the domestic concept; resident household consumption (`nama_10_gdp`) gives an upper bound for the residents' share.
- The aggregate core share split into relative price (HICP, Törnqvist) and relative volume.
- Household budget survey waves, 1988–2020, by income quintile, type of household, age of the reference person and degree of urbanisation, and for Spain, Greece and Italy. The 2020 wave has no split of housing into actual and imputed rent, so the measures that exclude imputed rent stop in 2015.
- Working–Leser Engel curves across the five quintiles in each wave, and budget elasticities by item.
- Approximate euro amounts per household and month, using INE's published mean expenditure per household, against the national minimum wage (DGERT).
- Household saving rate, 1995–2025, and Eurostat's experimental median saving rate by income quintile (2010, 2015, 2020).
- HICP by COICOP division, 1996–2025, weighted by each quintile's budget shares (imputed rent removed): fixed-weight windows and an index chained at every survey wave from 1996, with its sensitivity to the weights; HICP weights against survey budgets for 2021–2023.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python spending.py
python deeper.py
python figures.py
python -m pytest -q tests
```

`spending.py` writes `results/summary.txt`, `deeper.py` writes `results/deeper_summary.txt`, both with the CSVs behind them, and `figures.py` draws twenty figures in `figures/`, numbered in the order they appear in the article:

- `fig01` basic needs as a share of consumption, national accounts, 1995–2022, with the resident-basis bound
- `fig02` change in the core share split into relative price and relative volume
- `fig03` household saving rate, 1995–2025
- `fig04` food share by income fifth in each survey year (Engel curves)
- `fig05` budget elasticities by item and wave (Working–Leser)
- `fig06` food, rent, energy, water and repairs by income fifth, 1988–2015
- `fig07` actual rent by income fifth
- `fig08` electricity, gas and fuels by income fifth
- `fig09` composition of basic needs for each income fifth
- `fig10` discretionary items, poorest and richest fifth
- `fig11` core basic needs by household type, age and degree of urbanisation
- `fig12` Portugal, Spain, Greece and Italy: core needs and rent by fifth
- `fig13` housing including imputed rent, all seven survey waves (the 2020 break)
- `fig14` richest over poorest fifth, consumption per adult equivalent, total and after basic needs
- `fig15` euro amounts per household and month, 2015/16, against the minimum wage
- `fig16` median saving rate by income fifth (experimental)
- `fig17` prices of each basic need against all items since 1996
- `fig18` cumulative inflation of each income fifth's basket over fixed windows
- `fig19` chained basket indices since 1996, poorest minus richest fifth, and sensitivity to the weights
- `fig20` HICP weights against survey budgets and their contribution to the 2021–2023 gap

## Data

`fetch.py` downloads the Eurostat datasets in `data/` from the statistics API (JSON-stat); `.github/workflows/portugal-household-spending-data.yml` runs it and commits the snapshot, and `data/fetch_log.txt` records what each request returned. Datasets: `nama_10_co3_p3` (current prices and chain-linked volumes), `nama_10_gdp`, `nasa_10_ki`, `hbs_str_t211`, `hbs_str_t223` (Portugal and peers), `hbs_str_t224`, `hbs_str_t225`, `hbs_str_t226`, `hbs_exp_t133`, `icw_sr_03`, `prc_hicp_aind`, `prc_hicp_inw`, `prc_ppp_ind`. 2022 is provisional in `nama_10_co3_p3` for Portugal. In Portugal the budget-survey wave Eurostat labels 2020 was collected in 2022/23 (IDEF 2022/2023).

Two inputs are entered by hand, each with its source in the file: `data/ine_idef_totals.csv` (INE's mean expenditure per household for 2010/2011, 2015/2016 and 2022/2023) and `data/minimum_wage_changes.csv` (DGERT's dates and amounts of every minimum-wage change, the same file as in *portugal-wage-counterfactual*).

The workflow in `.github/workflows/portugal-household-spending.yml` runs the tests, reruns the analysis on every change, checks that both summaries are unchanged and that all twenty figures are drawn.

## License

Apache-2.0, as the rest of the repository.
