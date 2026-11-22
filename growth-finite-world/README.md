# Corporate growth on a finite planet: valuation, earnings and energy, 1871–2024

Companion code for the Medium article on why companies are expected to keep growing, what stock prices assume about that growth, and whether the energy data allow it.

## What it finds

All numbers below are from `output.txt`.

**A company does not need to grow; its price may.**
- $3B a year for ever, discounted at 8%, is worth $37.5B. At a price of $75B the same cash flow implies 4% growth for ever, and if that expectation goes to zero the price halves while the business is unchanged.
- With a flat cash flow, half of the value arrives within 9 years. With 4% growth at 8%, half arrives after 18 years and 15% after 50.

**A firm cannot outgrow its market, or the economy, for ever.** A firm at 5% of a market, growing 10% a year in a market growing 3%, would hold all of it after 46 years. Along a logistic path its growth converges to the market's.

**S&P earnings per share have tracked income per person, not the economy.** From 1871–1880 to 2013–2022, real EPS grew 1.83% a year, real US GDP 3.08% and real GDP per person 1.64%. The ratio of EPS to GDP per person has no trend (−0.04% a year, se 0.06).

**What the price requires.** With the 1871–2023 real total return of 6.90% as the required return, the lowest perpetual real growth consistent with the CAPE is 0.83% at the median CAPE (16.5), 3.56% at the last CAPE in the data (29.9, June 2023) and 4.40% at the CAPE of about 40 reported for autumn 2026. At that level half of the index's value arrives after year 29 and 31% after year 50.

**The world economy and its energy.** From 1970 to 2022 world GDP grew 3.48% a year and primary energy 2.11%, so GDP per unit of energy rose 1.35% a year; no decade did better than 1.89%. World primary energy fell in only five years between 1966 and 2024.

**Countries.** 31 of 152 countries grew between 2000 and 2022 while using less primary energy. They hold 33% of the sample's GDP and 13% of its population, they cut 5,355 TWh while the rest added 63,200 TWh, and in 21 of the 24 with consumption-based data their consumption CO2 fell less than their territorial CO2.

**Services are not weightless.** Across countries in 2022 energy per person rises at least in proportion to GDP per person (elasticity 1.18; 1.01 among countries above $15,000 that are not fuel exporters).

## Data

Three public sources, each pinned to a commit and checked by SHA-256 in `fetch_data.py`:

- [datasets/s-and-p-500](https://github.com/datasets/s-and-p-500), commit `87a2284`, a mirror of Robert Shiller's monthly S&P Composite data (licence PDDL). Earnings and dividends end in June 2023.
- [owid/co2-data](https://github.com/owid/co2-data), commit `22d9a01` (CC BY 4.0): GDP from the Maddison Project Database 2023, population, territorial and consumption-based CO2 from the Global Carbon Budget.
- [owid/energy-data](https://github.com/owid/energy-data), commit `f8de88a` (CC BY 4.0): primary energy from the Energy Institute Statistical Review and the US EIA.

`build_data.py` turns them into the compact tables in `data/`, which are committed. The analysis reads only `data/`.

One number is not in the pinned data: the CAPE of about 40 in autumn 2026, as reported by the CFA Institute's Enterprising Investor blog on 18 November 2026. It is set as `CAPE_NOW` in `growth_numbers.py` and labelled as external in `output.txt`.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against closed-form and brute-force answers
python growth_numbers.py > output.txt      # every number in the article
python figures.py                          # figures/
python fetch_data.py --check               # .cache/ matches the pinned sources
```

The workflow `.github/workflows/growth-finite-world.yml` runs all of this and fails if `output.txt` changes.

## Files

- **`growth_core.py`**: perpetuities, implied growth, the share of value beyond a horizon, saturation of a market, logistic share paths, growth decomposition, decoupling classes, OLS with HC1 errors, and the real total-return index.
- **`growth_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the five figures.
- **`fetch_data.py`**, **`build_data.py`**: download, verify and compact the sources.
- **`tests/test_core.py`**: 19 checks of the core functions.

## Licence

Code: Apache-2.0. Data: as licensed by the sources above.
