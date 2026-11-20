# Lump sum or monthly instalments: investing a windfall in 23 stock markets since 1955

Companion code for the Medium article on investing a windfall at once against spreading it over monthly instalments. The thesis: a lump sum usually beats spreading it out. Spreading costs return in most years, and what it buys is a smaller chance of a bad first year.

## What it finds

All numbers below are from `output.txt`.

**The replay.**
- **Setup.** A windfall invested in a country's stock index in one month, or in 12 equal monthly instalments. Cash waiting to be invested earns the country's 3-month interbank rate (or the overnight rate where the 3-month rate is missing).
- **Coverage.** 13,241 start months, from January 1955 to October 2025, in 23 markets (the OECD members of 1973, without Turkey).
- **How it is judged.** Both are compared one month after the last instalment.

**Share prices only, as the OECD publishes them (no dividends):**
- **How often.** The lump sum was ahead in 56.7% of start months, and in more than half of start months in 22 of 23 markets. By market, from 47.8% (New Zealand) to 64.2% (Norway).
- **By how much.** The median lead was 1.7% of the windfall. When the lump sum won, it won by a median 7.2%; when it lost, it lost by a median 7.0%.

**Dividends.** Leaving them out favours spreading.
- **Assumed yields.** With an assumed yield of 2%, 3% or 4% a year, the lump sum was ahead in 60.3%, 62.1% and 63.8% of start months.
- **The United States, exactly.** Shiller's S&P composite, 816 start months: 61.6% without dividends, 67.9% with them reinvested.

**What spreading buys.** After a year, the windfall was worth less than 80% of its starting value:
- in 9.2% of start months for the lump sum;
- in 2.5% for 12 instalments.

The 5th percentile rose from 0.73 to 0.85.

**The loss aversion that makes spreading the better bet.** Losses weighted 1.79 times gains without dividends, 2.69 times with 3% dividends, and 3.48 times for the US with actual dividends.

**Conditions at the start.**
- **Starting at an all-time high.** The lump sum was ahead in 61.8% of such start months, more often than elsewhere (54.6%).
- **High cash rates.** With cash at 8% or more, spreading won more often than not without dividends, and about as often as the lump sum with them.

**Length of spreading.** From 3 to 36 months, the lump sum was ahead in 56%–57% of start months for every length. Its median lead grows from 0.5% to 2.7%. The protection stops growing after about 24 months.

## Data

The committed snapshot is in `data/`, taken on 20 November 2026, from OECD `DF_FINMARK`:
- `share_prices.csv`: share price indices, monthly averages;
- `rate_3m.csv`: 3-month interbank rates;
- `rate_overnight.csv`: overnight interbank rates.

Shiller's monthly S&P composite with dividends comes from the `datasets/s-and-p-500` repository at commit `87a2284`.
- It is downloaded into `data/raw/` by `fetch_data.py` and checked against a pinned SHA-256 digest.
- It is not committed here.

## Method notes

- **The lump sum.** It buys at the index of the start month.
- **The instalments.** Each of the first *n* − 1 instalments is 1/*n* of the windfall. The waiting cash earns the month's annual rate divided by 12, and the last instalment invests everything left, interest included.
- **The comparison.** Both are valued one month after the last instalment, when both are fully invested. From then on they move together, so the gap in per cent stays as long as both are held.
- **Index values.** OECD share price indices are monthly averages and leave out dividends. The dividend variants add a constant yield, reinvested monthly.
- **Starting points.**
  - An all-time high means the index is at its highest level so far, with at least 60 months of history.
  - A fall means the index is 20% or more below its highest level so far.
- **Loss-aversion break-even.** The outcome is valued as *x* − 1 when the windfall gained and as λ(*x* − 1) when it lost. The break-even is the λ at which both strategies have the same expected value.
- **Determinism.** Sums use `math.fsum`, and there is no random number anywhere.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the replay on cases with a known answer
python fetch_data.py                       # Shiller's data (digest-checked) and the OECD comparison
python lump_numbers.py > output.txt        # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/lump-sum-vs-spreading.yml` runs all of this, and fails if `output.txt` changes.

## Limits

- **Overlap.** Start months overlap, so the shares describe the record rather than estimate a probability. With one start every 13 months, the lump sum was ahead in 55.0% of 1,025 windows.
- **Costs and taxes.** Transaction costs and taxes are left out. Twelve purchases cost more than one.
- **Index funds.** A national index is not a fund someone could buy in 1960. The replay assumes the index could be bought at its monthly average.

## Licence

Code: Apache-2.0. Data: OECD (CC BY 4.0). Shiller's data is downloaded, not redistributed.
