# Pass-through of Portugal's zero VAT on 46 food products (IVA Zero) in official prices

From 18 April 2023 to 4 January 2024 Portugal applied a 0% VAT rate to 46 food products previously taxed at 6% (vegetable oils at 23%). This folder estimates how much of the cut reached consumer prices, and how much of the reversal, using only official statistics: Eurostat's monthly HICP by ECOICOP subclass for every EU country, the same indices at constant tax rates, and the HICP weights.

- **Treatment from the data.** INE's constant-tax index assumes full pass-through, so the gap between the HICP and the HICP at constant taxes (the tax wedge) moves only when a tax rate changes. Its change between March and May 2023 identifies, subclass by subclass, how much of each Portuguese food subclass the measure covered (13 fully, 12 partly, other edible oils at 23%, 28 untreated). The same wedge screens the comparison countries.
- **Estimators.** Treated minus untreated food subclasses within Portugal; the same after subtracting each subclass's mean change in eleven comparison countries with no food tax change since 2022 (a triple difference); short windows around each change; a dose-response across all 53 subclasses; and a synthetic control on the relative price of the covered basket.
- **Inference.** Placebo jumps (every two-month and one-month change in 2019–2022), placebo countries, placebo dates, synthetic-control ranks, leave-one-out over subclasses and countries.
- **Consequences.** Effect on headline HICP inflation in 2023 and 2024, and the saving by income fifth from the budget survey.

## Run

```bash
python -m pip install -r requirements.txt
python ivazero.py
python figures.py
python -m pytest -q tests
```

`results/summary.txt` has every number; the CSVs behind it are in `results/`. Ten figures in `figures/`:

- `fig01` tax wedge of each Portuguese food subclass (the treatment)
- `fig02` covered and untreated food baskets, Portugal and comparison countries
- `fig03` event study, within Portugal and across countries, with 95% intervals
- `fig04` dose-response: price change against the tax-wedge change, every subclass
- `fig05` effect by subclass
- `fig06` placebo jumps and placebo countries
- `fig07` synthetic control
- `fig08` the reversal in January 2024
- `fig09` headline HICP inflation with and without the measure
- `fig10` saving by income fifth

## Data

`fetch.py` downloads `prc_hicp_midx`, `prc_hicp_cind` and `prc_hicp_inw` for the 27 member states (food codes and the all-items index, from 2018) through `.github/workflows/portugal-iva-zero-data.yml`, which commits the snapshot; `data/fetch_log.txt` records each request. The old ECOICOP subclasses end in December 2025, when the HICP moved to a new classification. The budget-survey tables (`hbs_str_t223`, `hbs_exp_t133`), resident household consumption (`nama_10_gdp`) and INE's mean expenditure per household are copied from *portugal-household-spending*.

Policy dates and figures used in the code: announcement 24 March 2023; in force 18 April 2023 to 4 January 2024; end announced 27 October 2023; 4,149,096 private households (Census 2021); government cost estimate of about 550 million euros (410 million to October plus 140 million for the extension, as reported in the press).

The workflow `.github/workflows/portugal-iva-zero.yml` runs the tests, reruns the analysis and checks `results/summary.txt` against the committed run (`check_summary.py` allows one unit in the last printed digit, for floating-point differences between machines).

## License

Apache-2.0, as the rest of the repository.
