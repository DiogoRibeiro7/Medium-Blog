# Rent or buy in sixteen US metropolitan areas, 1987-2025: does forced saving decide?

Code for an article that tests one claim: buying a home beats renting mainly because the mortgage forces the owner to save, so that a renter who invests the difference does as well or better. Households buy or rent in each year from 1987 and leave after 5, 10 or 20 years, in 16 metropolitan areas with both an S&P CoreLogic Case-Shiller index and a BLS rent index, and in the United States as a whole.

## Data (`fetch.py`, `data/`, not tracked)

Five contracts in [`../data_sources/`](../data_sources/) pin every input to a commit of the data registry [`DiogoRibeiro7/data`](https://github.com/DiogoRibeiro7/data), with size and SHA-256:

- `bls-cpi-rent-us-metros` (canonical): CPI-U rent of primary residence for the U.S. and the 16 areas, and all-items CPI-U.
- `census-acs-housing-costs-msa` (canonical): ACS 1-year 2022 and 2023 median value, contract and gross rent, real estate taxes, rooms by tenure.
- `sp-corelogic-case-shiller-datahub` (external): Case-Shiller indices from a pinned commit of the Datahub mirror.
- `freddie-mac-pmms-fred` (external): weekly 30-year fixed rate, FRED query with fixed dates.
- `damodaran-historical-returns` (external): annual returns on stocks, bonds and bills, an Internet Archive capture of the January 2026 workbook.

`fetch.py` downloads them all and stops if any byte differs. External files are downloaded from their upstream URLs because their bytes may not be stored.

## Run

```
pip install -r requirements.txt
python fetch.py
python -m unittest discover -s tests
python rentbuy_numbers.py > output.txt   # every number in the article; about a minute
python figures.py                        # figures/fig1-fig3
```

CI reruns everything and fails if `output.txt` changes.

## Model (`rentbuy_core.py`)

- **Owner.** Buys in the middle of a year at that year's average price, with 20% down and 2% buying costs; 30-year fixed mortgage at that year's average Freddie Mac rate; refinances (cost 1% of the balance, same payoff date) whenever a year's average rate is at least a point below the loan's; pays property tax (the area's ACS ratio of taxes to value, or 1.2% of the purchase price growing 2% a year under California's Proposition 13), insurance 0.4% and maintenance 1% of value; sells after H years paying 6%.
- **Renter.** Rents a home like the owner's and invests the down payment and buying costs in the S&P 500 with dividends (or a 60/40 mix, or Treasury bills).
- **Saving.** Each year, whoever pays less for housing may invest the difference. The share invested, s, is the same for both: s = 1 is the disciplined saver, s = 0 spends it. The owner's principal repayments are saved whatever s is. Wealth is linear in s, so s = 0 and s = 1 give the whole picture and the share at which the two tie.
- **Levels.** Prices and rents are index series turned into dollars with ACS 2023: the median value of owner-occupied homes, and the median contract rent scaled by (rooms in owned homes / rooms in rented homes) raised to e = 0.5, so that the rent is for a home like the owner's; e = 0 and e = 1 bracket it.
- **Categories.** For each area, purchase year and hold: renting wins either way, saving decides (the owner beats a renter who spends the difference but not one who invests it), owning wins either way, or saving favours the owner.

Income taxes are left out on both sides (no mortgage interest deduction, no tax on the renter's returns); a sensitivity run takes 0.75 points a year off the renter's returns.

Contact: dfr@esmad.ipp.pt
