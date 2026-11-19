# Fixed or variable: euro-area mortgages replayed for every start month

Companion code for the Medium article on choosing between a fixed and a variable mortgage rate. The thesis: variable was cheaper on average for most start years, but fixed is insurance, so judge it by the worst years (the 2022–23 payment shock), not by the average cost.

## What it finds

All numbers below are from `output.txt`. They are for a 30-year loan of 100,000 euros started in every month from January 1999 to October 2026.
- **The variable loan** pays 12-month Euribor plus 1 point, reset once a year.
- **The fixed loan** pays the 10-year AAA euro-area government par yield plus 1 point, fixed for ten years.

Both reference rates are floored at zero.

**The ten-year bill** covers the 215 start months with ten years of data (1999-01 to 2016-11).
- **Who won.** Variable was cheaper in 90.2% of them.
- **The premium.** Fixing cost 15,911 euros more interest on average, about 1,600 euros a year.
- **Why.** The 10-year yield at the start was above the next ten years' average Euribor in 194 of 215 months, by 1.82 points on average. Rates kept falling below what the yield curve expected.

**The shock.** A variable loan started in October 2021 went from 322 to 536 euros a month by October 2023, a rise of 66.7%. If the first payment was 30% of income, it reached 45.4%.
- **Who was hit.** Loans started from 2019 to 2021 saw rises of 59.0% on average.
- **What fixing saved.** For loans started from January 2015 to June 2022, fixing saved 7,195 euros on average up to October 2026.
- **The fixed side.** A fixed payment started in 2019–2021 fell to 78.5% of its first value in real terms.

**Premium and payout.** Fixing cost money in the start months where the variable payment barely moved. It saved money in the months where the payment rose the most (figure 5).

**A loan started in October 2026.** Variable starts at 4.24%, fixed at 4.42%. Each scenario lays one past start month's yield-curve errors onto today's curve.
- **Errors as they were.** Variable is cheaper in 88% of scenarios, and the payment never rises by more than 43%.
- **Errors without their downward average.** Variable is cheaper in only 33% of scenarios. The payment rises by more than 25% in 62% of them, and by up to 87%.

## Data

- **Euribor.** Eurostat's monthly averages (`irt_st_m`), downloaded by `fetch_data.py` and checked against a pinned SHA-256. Euribor is a benchmark of the European Money Markets Institute, so it is not committed.
- **Committed snapshot** in `data/`, taken from Eurostat on 19 November 2026:
  - euro yield curves (`irt_h_euryld_m` for 1999–2005, `irt_euryld_m` from September 2004), par and spot rates;
  - the euro-area HICP (`prc_hicp_minr`);
  - compensation per employee in the euro area (`nama_10_lp_ulc`).
- **Checks.** The replay's rates are compared in `output.txt` with four ECB published averages for new housing loans. The two fixed rates (September 2021, June 2023) and the two floating rates (October and December 2023) are within 0.35 points.

## Method notes

- **Rates.** Variable rates use the Euribor average of the month of each reset. Fixed rates use the par yield of the month the loan starts. The yield curve is Eurostat's historical series before September 2004 and the AAA curve after.
- **What a loan costs.** The interest paid over the horizon, so that the principal repaid does not count.
- **Expectations.** These come from one-year forward rates of the spot curve (continuous compounding), plus the gap between 12-month Euribor and the one-year spot rate at the start.
- **Checks in `output.txt`.**
  - fixed loans 0.25 or 0.5 points dearer;
  - margins of 1.5 points;
  - 6-month and 3-month Euribor with matching resets;
  - all euro-area government bonds;
  - no floor;
  - a 30-year fixed rate.
- **Determinism.** Sums are `math.fsum` or fixed-order Python sums, so the output does not depend on the processor's vector kernels.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python fetch_data.py                       # Euribor, checked against its digest; snapshot compared with Eurostat
python mortgage_numbers.py > output.txt    # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/fixed-or-variable.yml` runs all of this and fails if `output.txt` changes.

## Limits

- **Proxies for bank rates.** Bank rates are not in Eurostat's database, so loan rates are reference rates plus a margin. Real margins moved over time and differ by country and bank.
- **One euro-area path.** Euribor is common to all euro countries, but margins, fixed-rate periods and floors differ by country.
- **A falling-rate era.** From 1999 to 2021 rates mostly fell, often more than markets expected. The average saving of variable loans is partly that surprise and may not repeat.
- **Not advice.** The scenarios show the range of past surprises, not a forecast.

## Licence

Code: Apache-2.0.
