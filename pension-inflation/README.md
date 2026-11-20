# Pension inflation: fixed and indexed pensions replayed on 23 countries' prices

Companion code for the Medium article on inflation and pensions. The thesis: inflation is the risk a pension hides. A pension statement shows a number of euros, and that number never falls, while what it buys falls every year.

## What it finds

All numbers below are from `output.txt`.

**The replay.** A pension of fixed amount was started in every year from 1955 in 23 countries: the OECD members of 1973, without Turkey. Each pension is valued at its own country's consumer prices.
- **At 85.** In the median history it buys 42.1% of what the first payment bought. In the worst tenth it buys 13.1% or less. Steady inflation of 2% would leave 67.3%.
- **Over a lifetime.** Count each year by the chance of being alive to receive it (Eurostat life table, EU 2019, ages 65 to 94). The median pension keeps 63.5% of its real value, and the worst tenth keeps 40.3% or less.
- **When the loss arrives.** Half of the lifetime loss falls after 80, when a pensioner can no longer work to make it up.
- **The euro era.** Pensions started in 1999 had lost a median 43.3% by 2025, close to the steady-2% path.

**Partial indexation rules**, replayed on the same 918 starts (figure 3):
- **Caps.** Prices capped at 2.5% keep a median 91.2% of lifetime income in the calmer half of histories, but only 38.8% in the worst tenth. A fixed pension keeps 33.4% there. A cap insures against small inflation, not large.
- **Fixed rises.** A rise of 3% a year over-protects in calm histories (101.2%) and fails in rough ones (39.9%).
- **Prices minus 0.5 point.** This rule loses about 5% of lifetime income in every history.

**What protection costs.** At a 3.5% nominal and a 1.5% real yield, with the EU life table, an indexed annuity starts 17.4% lower than a level one. It overtakes it at 75, and has paid more in total from 84. A UK market quote from 16 November 2026 starts 30.7% lower.

**Was it worth it?** We replayed 473 purchases in 22 countries from 1955 to 1996, each with that year's 10-year yield and a 2% real rate.
- **1960s purchases.** The indexed annuity did better in most of them: the level one did better in only 12.7%.
- **1980s and 1990s purchases.** The level annuity did better in almost all of them, because high nominal yields priced in inflation that never came.

**Portugal.** A general-regime pension of 1,000 euros in 2008 was rebuilt from the annual update ordinances.
- **The freeze.** It stayed at 1,034.24 euros from 2010 to 2017 while prices rose 9.2%.
- **The low point.** Its real value fell to 88.8% of the 2008 value in 2022, and was 95.8% in 2025.
- **Check.** The transcribed rates match the official coefficient table in Portaria 372-B/2024/1 to within 0.0002.

## Data

The committed snapshot is in `data/`, taken on 20 November 2026:
- **Consumer prices.** OECD `DF_PRICES_ALL`: consumer price index, all items, annual average.
- **Long-term interest rates.** OECD `DF_FINMARK`: 10-year government bonds, annual average.
- **Wages.** OECD `AV_AN_WAGE`: average annual wages, constant prices and PPPs.
- **Life tables.** Eurostat `demo_mlifetable`: probability of dying by age, and life expectancy at 65, for the EU and Portugal, 2019 and 2024.
- **HICP.** Eurostat `prc_hicp_aind`: annual average index.
- **Portuguese pension updates.** `portugal_updates.csv` is transcribed by hand from the ordinances in the Diário da República. Each row names its source. `portugal_first_bracket_coefficients.csv` is the official cumulative table used to check the transcription.

`fetch_data.py` downloads the OECD and Eurostat series again and reports whether the snapshot still matches.

## Method notes

- **Real value.** A payment fixed in the start year, divided by the price index of each later year (annual averages).
- **Lifetime real income kept.** The average real value of the payments at ages 65 to 94, each year weighted by the probability of being alive at that age. The weights add up to 20.0 years, the same as Eurostat's life expectancy at 65.
- **Rules.** Each rule raises the payment once a year by a function of that year's inflation. Capped rules never cut the payment when prices fall.
- **Annuities.**
  - **Pricing.** Annual payments while alive, priced with the same life table.
  - **The replay.** The level annuity uses the country's 10-year yield in the purchase year. The indexed annuity uses a real rate of 2% (1% and 3% are in `output.txt`).
  - **What is compared.** The value of the level annuity's real payments, discounted at the real rate, per unit of premium. The indexed annuity is worth exactly one on that scale.
- **Portugal.**
  - **Each update.** The percentage of the bracket the pension is in, with the minimum rise in euros where the ordinance sets one. The interim rise of July 2023 is on the December 2022 amount.
  - **Left out.** Fixed-euro extraordinary rises, one-off payments and temporary cuts.
- **Determinism.** Sums use `math.fsum`, and there is no random number anywhere.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python fetch_data.py                       # snapshot compared with the OECD and Eurostat
python pension_numbers.py > output.txt     # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/pension-inflation.yml` runs all of this, and fails if `output.txt` changes.

## Limits

- **Prices.** A national consumer price index is not a pensioner's own basket.
- **Life table.** One EU table is used for every country and decade. People who retired in 1960 lived shorter lives than this table assumes, so the lifetime measures apply to today's pensioners facing past price histories.
- **Real rate.** The annuity replay assumes a constant real rate. Real yields moved a lot over the decades, so that comparison is an illustration of the bet, not a record of prices people were offered.
- **Wages.** A price-indexed pension keeps its purchasing power, not its place next to wages. That comparison uses OECD average wages from 2000.

## Licence

Code: Apache-2.0. Data: OECD (CC BY 4.0) and Eurostat, reused under their terms.
