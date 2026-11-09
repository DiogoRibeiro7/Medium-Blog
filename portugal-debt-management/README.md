# Portugal's Debt Management After the Rescue, Priced Instalment by Instalment: IMF and European Loans, Bond Exchanges and Savings Certificates, 2015-2026

Code, data and figures for the article. The question: when Portugal traded expensive debt for
cheaper debt after its 2011-2014 programme (repaying the IMF and part of the European loans early,
exchanging and buying back bonds, and funding itself through savings certificates), how much did
it really save, and what risk did it take on in exchange?

## Run

```bash
pip install -r requirements.txt
python fetch.py --out data   # or let the data workflow do it
python -m pytest -q tests
python imf.py                # early repayment of the IMF loan, 2015-2018
python european.py           # EFSF 2019 and EFSM 2023-2025 repayments
python exchanges.py          # bond exchanges and buybacks
python retail.py             # savings certificates, series E and F
python figures.py            # six figures in figures/
```

## Data

`fetch.py` downloads, and `.github/workflows/portugal-debt-management-data.yml` refreshes and commits:

- Banco de Portugal BPstat API: direct State debt by instrument (Treasury bonds, bills, savings and
  Treasury certificates, loans in euro and in other currencies, from IGCP data), Portuguese
  Treasury bond yields by residual maturity, ECB key rates, and other general-government series
  (raw JSON-stat pages in `data/bpstat`).
- ECB Data Portal: three- and twelve-month Euribor, the euro-area AAA three-month spot rate,
  exchange rates of the dollar, yen, pound and renminbi.
- FRED: three-month rates for the dollar, yen, pound and renminbi.
- Eurostat: ten-year convergence yields for Portugal and Germany, quarterly government interest
  and debt.

The IMF's own pages reject automated requests, so the SDR interest rate and the euro value of the
SDR are rebuilt from the SDR basket (`common.sdr`) and checked against values the IMF published
(`documented.csv`). Every request is logged in `data/fetch_log.txt`. `documented.csv` holds the
facts taken from official documents and dated reports (IMF terms, the costs and savings published
by IGCP and the government, the European repayments, each documented bond exchange, the rules of
the savings certificates), each with its source.

## Method

- **IMF.** Two paths of IMF credit in SDR, the original twelve semi-annual instalments of each
  purchase and the observed stock; charges at the SDR rate plus the margin and the level-based
  surcharge. Each instalment repaid early is refinanced with a euro bond maturing on its original
  due date at the Portuguese yield for that maturity. The saving splits into interest and the
  exchange rate on the principal; ex ante (rates frozen at each repayment) and ex post. A second
  variant refinances everything at ten years.
- **European loans.** Saving at par against the matched Portuguese yield, at the published average
  loan rates, with the break-even rate for each repayment.
- **Exchanges.** Bought bonds priced at the Portuguese yield for their residual maturity; the
  premium paid against the interest no longer paid; redemption years pushed back.
- **Savings certificates.** 2022-2026 vintages from monthly net inflows, paid the series E or F
  formula, against twelve-month Euribor and the five-year yield at subscription.

## Results

`results/imf_summary.txt`, `results/european_summary.txt`, `results/exchanges_summary.txt` and
`results/retail_summary.txt` hold every number used in the article; the CSV files hold the series
behind the figures. The reproduce workflow, `.github/workflows/portugal-debt-management.yml`,
reruns tests, analysis and figures on every change and fails if a summary number moves beyond a
small tolerance (`check_results.py`).
