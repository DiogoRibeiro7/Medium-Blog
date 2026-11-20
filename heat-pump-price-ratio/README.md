# The price ratio decides heat pumps: household electricity and gas prices in 25 European countries, 2007–2026

Companion code for the Medium article on when a heat pump is cheaper to run than a gas boiler. The thesis: the ratio of the electricity price to the gas price decides it. A heat pump with seasonal performance factor SPF beats a condensing boiler of seasonal efficiency η exactly when electricity costs less than SPF / η times as much as gas per kWh. The ratio decides the sign of the saving; the gas price decides its size.

## What it finds

All numbers below are from `output.txt`.

**The panel.** 921 country half-years, 2007-S2 to 2026-S1, in 24 EU countries and the United Kingdom (to 2020-S1). Household electricity in band DD (5,000–14,999 kWh a year) and gas in band D2 (20–199 GJ a year), all taxes and levies included.

**The ratio.**
- **Range.** Median 2.70 over all country half-years, from 0.80 (Sweden, 2022-S1) to 5.38 (Belgium, 2021-S1).
- **Latest year** (2025-S2 and 2026-S1, 23 countries). From 1.08 (Sweden) to 4.62 (Romania), median 2.62.
- **EU average.** 2.52 in 2007-S2, highest 3.12 in 2017-S1, 2.31 in 2026-S1.

**How good a heat pump has to be.** With the boiler at 90%:
- **Break-even.** The break-even SPF had a median of 2.43.
- **Field efficiencies.** A heat pump at the UK field median (SPF 2.8) was cheaper to run in 73.1% of country half-years; at the German field mean (3.4), in 90.0%.
- **Latest year.** Cheaper at 2.8 in 19 of 23 countries and at 3.4 in 21. It was not cheaper even at 3.4 in Hungary and Romania.

**What it saves.** For a home that needs 10,000 kWh of heat a year, at the latest year's prices and SPF 3.4, savings ranged from −143 euros (Romania) to 1,706 euros (Sweden). Across the latest year's countries, the saving correlates with the gas price (0.93) as strongly as with the ratio (−0.87).

**Taxes.** Taxes and levies raise the ratio in 13 of 25 countries and lower it in the rest.
- **Largest rises.** Poland +1.04 and the United Kingdom +0.95.
- **Largest falls.** The Netherlands −0.93 and Denmark −0.78.

**Deciding on today's ratio.** The ratio at the start gave the right answer for the next ten years' running costs in 81.7% of 438 starts at SPF 2.8, and in 91.6% at SPF 3.4.

## Data

The committed snapshot is in `data/`, taken on 20 November 2026, from Eurostat:
- `electricity.csv`: `nrg_pc_204`, household electricity prices, half-yearly, bands DC and DD, euros per kWh.
- `gas.csv`: `nrg_pc_202`, household gas prices, half-yearly, band D2, euros per kWh (gross calorific value).
- `hdd.csv`: `nrg_chdd_a`, heating degree days, 2015–2024.

Each price file holds all taxes and levies (`I_TAX`), prices without VAT and other recoverable taxes (`X_VAT`), and prices without any taxes or levies (`X_TAX`).

## Method notes

- **Running cost per kWh of heat.** p_e / SPF for the heat pump, p_g / η for the boiler. The heat pump is cheaper exactly when p_e / p_g < SPF / η.
- **Efficiencies.** η = 0.90 for a condensing boiler on a gross-calorific-value basis, with 0.85 and 0.80 as checks. SPF values: 2.80 (median of air-source heat pumps in the UK Electrification of Heat project) and 3.40 (mean of air-to-water heat pumps in Fraunhofer ISE's field test in existing German homes).
- **Prices are averages per kWh.** They include standing charges spread over the band's consumption. For gas that matches a household that disconnects from the grid. For electricity, the extra kWh of a heat pump cost less than the band average, so the comparison leans slightly towards the boiler.
- **Latest year.** The mean of the last two half-years with both prices.
- **Heat demand.** 10,000 kWh a year; the degree-day variant scales it by each country's heating degree days relative to the EU (27) mean for 2015–2024.
- **Determinism.** Sums use `math.fsum`, and there is no random number anywhere.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the definitions on cases with a known answer
python fetch_data.py                       # download from Eurostat and compare with the snapshot
python heat_numbers.py > output.txt        # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/heat-pump-price-ratio.yml` runs all of this, and fails if `output.txt` changes.

## Limits

- **Running costs only.** Installation, maintenance, subsidies and the lifetime of the equipment are left out.
- **Bands.** Prices are band averages, not the tariff any household is on. Time-of-use and heat-pump tariffs are not in the data.
- **Missing countries.** Cyprus, Finland and Malta report no household gas prices; Poland stops in 2023-S2 and the United Kingdom in 2020-S1.
- **One SPF per heat pump.** Real efficiency varies with the climate, the flow temperature of the heating system and the installation.

## Licence

Code: Apache-2.0. Data: Eurostat (reuse authorised, CC BY 4.0).
