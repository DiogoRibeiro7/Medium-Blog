# Portuguese household spending on basic needs, saving, and basic-needs inflation

Analysis code for a follow-up to *portugal-wage-counterfactual*: what share of household spending goes to basic needs, how that differs by income, how much households save, and whether basic needs have become dearer than the average basket.

- National accounts, 1995–2022: household consumption by COICOP purpose. Core basic needs are food (CP01), actual rent (CP041), dwelling repairs (CP043), water and other dwelling services (CP044) and electricity, gas and fuels (CP045); the wider set adds health (CP06), running personal transport (CP072) and transport services (CP073). Imputed rent (CP042) is removed from numerator and total, since it is not a cash outlay.
- Household budget survey waves, 1988–2020, by income quintile. The 2020 wave has no split of housing into actual and imputed rent, so the measures that exclude imputed rent stop in 2015.
- Household saving rate, 1995–2025, and Eurostat's experimental median saving rate by income quintile (2010, 2015, 2020).
- HICP by COICOP division, 1996–2025, weighted by each quintile's budget shares (imputed rent removed) to give inflation by quintile, and a fixed-weight index of the core basic needs against all items.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python spending.py
python figures.py
python -m pytest -q tests
```

It writes `results/summary.txt` (every number), the CSVs behind it, and twelve figures in `figures/`:

- `fig01` basic needs as a share of consumption, national accounts, 1995–2022
- `fig02` household saving rate, 1995–2025
- `fig03` food share by income fifth in each survey year (Engel curves)
- `fig04` food, rent and household energy by income fifth, 1988–2015
- `fig05` actual rent by income fifth
- `fig06` electricity, gas and fuels by income fifth
- `fig07` composition of basic needs for each income fifth
- `fig08` housing including imputed rent, all seven survey waves (the 2020 break)
- `fig09` richest over poorest fifth, consumption per adult equivalent, total and after basic needs
- `fig10` median saving rate by income fifth (experimental)
- `fig11` prices of each basic need against all items since 1996
- `fig12` cumulative inflation of each income fifth's basket

## Data

`fetch.py` downloads the Eurostat datasets in `data/` from the statistics API (JSON-stat); `.github/workflows/portugal-household-spending-data.yml` runs it and commits the snapshot, and `data/fetch_log.txt` records what each request returned. Datasets: `nama_10_co3_p3`, `nasa_10_ki`, `hbs_str_t211`, `hbs_str_t223`, `hbs_exp_t133`, `icw_sr_03`, `prc_hicp_aind`. 2022 is provisional in `nama_10_co3_p3` for Portugal. In Portugal the budget-survey wave Eurostat labels 2020 was collected in 2022/23 (IDEF 2022/2023).

The workflow in `.github/workflows/portugal-household-spending.yml` reruns the analysis on every change and checks that `results/summary.txt` is unchanged.

## License

Apache-2.0, as the rest of the repository.
