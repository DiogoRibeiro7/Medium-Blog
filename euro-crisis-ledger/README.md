# A ledger of the euro-area crisis, 2010–2025

Code and data for the article *Who Gained and Who Paid in the Euro-Area Crisis: A Counterfactual Ledger of Interest, Official Loans and Bank Exposures, 2010–2025*.

The ledger covers five programme countries (Greece, Ireland, Portugal, Cyprus, Spain) and five core countries (Germany, France, the Netherlands, Finland, Austria). It keeps three kinds of entry apart: transfers measured from data (interest saved or paid because of the crisis, against two counterfactuals), transfers documented by those who paid or received them (central-bank income on Greek bonds, the 2012 private-sector haircut), and positions that are not yet costs (official loans outstanding).

## Scripts

- `fetch.py`: downloads the data (run by GitHub Actions).
- `yields.py`: two counterfactual 10-year yields. *No-fragmentation*: each country keeps its 2003–2007 spread to the GDP-weighted euro-area average. *Safe-asset*: Germany follows a synthetic control of non-euro advanced-economy yields fitted on 1999–2009, and the others keep their 2003–2007 spreads to Germany. Placebo runs for every donor. Crisis window: 2010–2014.
- `interest.py`: the pass-through of market yields to the implicit interest rate on government debt (estimated on the core countries), and the interest effect in euros, 2010–2025, by market repricing and by actual interest against a simulated counterfactual; sensitivity to the pass-through.
- `ledger.py`: the ledger: interest account, documented central-bank income, official loans outstanding at each creditor's contribution key, foreign bank claims on programme countries' public sectors (BIS), and the shift of programme-country debt from bonds to official loans.
- `figures.py`: the eight figures.
- `documented.csv`: every figure taken from an official statement or a study, with its source and date.
- `check_results.py`: compares regenerated summaries with the committed ones within a small tolerance.

## Data

Fetched by `.github/workflows/euro-crisis-ledger-data.yml` into `data/`:

- Eurostat: `irt_lt_mcby_m` (monthly long-term government bond yields), `gov_10dd_edpt1` (debt), `gov_10a_main` (interest payable and receivable), `nama_10_gdp` (GDP), `gov_10q_ggdebt` (quarterly debt by instrument), `gov_10dd_ggd` (debt by holder, recent years).
- FRED (OECD Main Economic Indicators): 10-year government bond yields of non-euro advanced economies (`IRLTLT01..M156N`) and the euro-dollar rate (`EXUSEU`).
- BIS consolidated banking statistics (`WS_CBS_PUB`): foreign claims of banks by nationality on Greece, Ireland, Portugal, Cyprus, Spain and Italy. The analysis uses basis U (ultimate risk), position C (foreign claims) and sector O (official sector).

## Reproducing

```
python -m pip install -r requirements.txt
python -m pytest -q tests
python yields.py && python interest.py && python ledger.py && python figures.py
```

## Points to keep in mind

- The counterfactuals bracket the answer rather than pin it down. The safe-asset donors (Switzerland, the United Kingdom, Norway, Sweden in particular) also received safe-haven flows during the crisis, which biases the German saving towards zero; the euro-area average in the no-fragmentation counterfactual was itself pushed up by the stressed countries, which biases it upwards.
- Yield gaps after 2014 are reported but not counted: from 2015 the ECB's asset purchases moved yields for reasons other than the crisis. Gaps in 2010–2014 keep acting until 2025 through debt issued in those years.
- Programme countries stopped borrowing in the market and paid official rates instead; for them the comparison of actual interest with a counterfactual is the relevant one, and it includes the effect of the 2012 Greek restructuring and of concessional official lending.
- Loans outstanding are positions, not costs: no losses have been realised on EFSF, ESM, EFSM or GLF loans. Each creditor's share is approximated with the EFSF and ESM contribution keys.
- Bank claims are in book value converted from US dollars; their decline includes repayments, sales (some to the Eurosystem) and impairments.
- Output and employment losses in the programme countries are not transfers and are outside the ledger.
