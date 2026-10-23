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

It writes `results/summary.txt` (every number), the CSVs behind it, and `figures/fig1_core_share_by_quintile.png`, `figures/fig2_poorest_composition.png`, `figures/fig3_basic_needs_prices.png`.

## Data

`fetch.py` downloads the Eurostat datasets in `data/` from the statistics API (JSON-stat); `.github/workflows/portugal-household-spending-data.yml` runs it and commits the snapshot, and `data/fetch_log.txt` records what each request returned. Datasets: `nama_10_co3_p3`, `nasa_10_ki`, `hbs_str_t211`, `hbs_str_t223`, `hbs_exp_t133`, `icw_sr_03`, `prc_hicp_aind`. 2021 and 2022 are provisional in `nama_10_co3_p3` for Portugal.

The workflow in `.github/workflows/portugal-household-spending.yml` reruns the analysis on every change and checks that `results/summary.txt` is unchanged.

## License

Apache-2.0, as the rest of the repository.
