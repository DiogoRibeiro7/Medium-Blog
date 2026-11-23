# Holding stocks for 20 or 30 years: losses after inflation got rarer, not smaller

Companion code for a Medium article that tests one claim: **a long horizon lowers how often stocks lose after inflation, but not how much they lose when they do.** Real stock and bond wealth over holding periods of 1 to 30 years, every start year, in 23 OECD markets from the 1950s to 2025 and in the United States from 1871 to 2022.

## What it finds

All numbers below are from `output.txt`. OECD figures use share prices plus an assumed 3% dividend yield.

**How often against how much.** Pooled over the 23 markets:
- **How often.** The share of start years ending in a real loss fell from 35.8% over 1 year to 20.4% over 10, 7.9% over 20 and 0.9% over 30.
- **How much.** The median loss among the start years that lost rose from 11.9% over 1 year to 26.8% over 10 and 30.3% over 20. Over 30 years it was 35.1%, from only 6 start years in 2 markets.
- **Same start years.** Holding the start years at 1955-1995 for every horizon gives the same pattern: 40.9%, 21.7%, 7.5% and 0.9% of starts lost; the median loss was 11.4%, 25.8%, 29.8% and 35.1%.

**The dividend assumption** moves how often, not how much. Over 20 years, with yields of 0%, 2%, 3% and 4%:
- the real-loss share is 27.6%, 12.1%, 7.9% and 5.9%;
- the median loss when losing is 29.3%, 31.5%, 30.3% and 28.6%.

**The United States, 1871-2022 (Shiller, actual dividends).**
- **Real losses.** In 31.1% of start years over 1 year, 9.9% over 10 and none over 20 or 30.
- **Worst starts.** The worst 20-year start (1900) ended at 1.11 times the money in real terms; the worst 30-year start (1890) at 2.65.
- **Independent windows.** The 151 years of data hold only 7 non-overlapping 20-year windows.

**Markets.**
- **20 years.** 13 of 23 markets had at least one start year with a real loss. The worst were Greece from 1999 (0.20), Italy from 1959 (0.23), Iceland from 2005 (0.29) and France from 1962 (0.44).
- **30 years.** Two markets, Greece and Italy.
- **When.** The 20-year losses started mostly in the 1960s (29 of 68 start years) and from 1990 to 2005 (26).

**The shuffle test.** Each market's own yearly real returns were put in random order 2,000 times.
- **How often.** The historical loss share matches the shuffles up to 15 years. Over 20 to 30 years it is below them: 7.9% against a shuffled median of 10.8% over 20 years, and 0.9% against 4.4% over 30.
- **How much.** The median loss when losing stays inside the shuffled band at every horizon.
- **Variance ratios.** They are below the shuffled ones from 3 to 15 years (0.57 against 0.79 over 10 years). That is mean reversion, which made losses rarer without making them smaller.

**Stocks against bonds** (a ten-year par bond on the OECD long-term yield). Stocks lagged bonds in:
- 38.2% of start years over 1 year;
- 34.4% over 10;
- 23.6% over 20;
- 11.2% over 30.

**An equal-weight mix of the markets.** Local real returns, rebalanced yearly, exchange rates left out:
- real losses in 8.3% of start years over 10 years, 1.8% over 15 and none over 20 or more, in 1956-2025;
- the worst 20-year start (1961) ended at 1.19 times the money.

## Data (`fetch.py`, `data/`, not tracked)

Two contracts in [`../data_sources/`](../data_sources/) pin every input to a commit of the data registry [`DiogoRibeiro7/data`](https://github.com/DiogoRibeiro7/data), with size and SHA-256:

- **`oecd-share-prices-bond-yields-cpi`** (canonical, CC BY 4.0). Three OECD SDMX responses retrieved on 23 November 2026:
  - share price indices (`DF_FINMARK`, `SHARE`), monthly averages, **without dividends**;
  - ten-year government bond yields (`DF_FINMARK`, `IRLT`);
  - consumer prices (`DF_PRICES_ALL`, all items).
- **`shiller-sp-composite-monthly`** (external). Robert Shiller's monthly S&P composite price, dividend, CPI and long rate, from commit `87a2284` of the Datahub mirror `datasets/s-and-p-500`. Downloaded at each run, not stored.

`fetch.py` downloads them and stops if any byte differs.

## Run

```
pip install -r requirements.txt
python fetch.py
python -m unittest discover -s tests -v
python horizon_numbers.py > output.txt   # every number in the article; about ten seconds
python figures.py                         # figures/fig1-fig5
```

The workflow `.github/workflows/long-horizon-stock-risk.yml` runs all of this and fails if `output.txt` changes.

## Method (`horizon_core.py`)

- **Year ends.** December monthly averages for share prices and yields.
  - Consumer prices are read for December too; Australia and New Zealand use the fourth-quarter average, and Ireland before 1975 uses November.
  - Japan's consumer prices in the OECD file end in 2021, so its windows end in 2020.
- **Stocks.**
  - **OECD markets.** The share price return plus a constant dividend yield *d*, reinvested once a year: (*P*ₜ / *P*ₜ₋₁)(1 + *d*) − 1.
  - **United States.** Shiller's monthly price plus a twelfth of the annual dividend each month.
- **Bonds.** A ten-year par bond bought at the year-end yield, held for one coupon and sold as a nine-year bond at the next year-end yield. It approximates a constant-maturity fund; neither source publishes bond returns.
- **Windows.** Every start year *s* and horizon *H* with data at *s* and *s* + *H*, so windows overlap.
  - "Independent" counts the non-overlapping windows a market's span holds.
  - An "episode" is a run of consecutive losing start years in one market.
- **Shuffle test.**
  - For each market, the yearly real log returns are permuted 2,000 times (seed 20261123), keeping the same years in a random order.
  - Pooled loss shares and median losses are recomputed for each shuffle.
  - Variance ratios use overlapping *H*-year sums, in markets with at least 3*H* years.
- **Start conditions.** The real share price against its average over the ten year-ends up to the start, and, for the United States, Shiller's CAPE in December.

## Limits

- **Dividends are assumed** for the OECD markets.
  - Over 1957-2022 Shiller's average US dividend yield was 2.91%.
  - The OECD US index plus 3% ran 0.35 to 1.05 points a year below Shiller's total return (median gap by horizon), so 3% is, if anything, low for the United States.
  - The true yield also moved over time: 3.80% in 1957-1989 and 2.01% in 1990-2022 for the United States.
- **Overlap.** Shares of start years describe the record; they are not probabilities. The longest horizons rest on a handful of independent windows.
- **Bonds are approximated** from yields, and the 1981-2021 fall in yields flatters bonds in the windows it covers.
- **No costs, taxes or currencies.** The equal-weight mix averages local real returns; it is not a portfolio one saver could have held.

## Licence

Code: Apache-2.0. Data: OECD (CC BY 4.0), from the data registry. Shiller's data is downloaded, not redistributed.

Contact: dfr@esmad.ipp.pt
