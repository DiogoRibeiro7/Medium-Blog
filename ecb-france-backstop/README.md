# The ECB's Backstop and France: Warranted Spreads, the Interest Bill to 2036 and Four Channels to the Rest of the EU

Code, data and figures for the article. The question: can the European Central Bank control what
happens to France's borrowing costs, and if it cannot, how would a French repricing reach the rest
of the EU?

## Run

```bash
pip install -r requirements.txt
python fetch.py --out data   # or let the data workflow do it
python -m pytest -q tests
python warranted.py          # warranted spreads from 120 models, flags, the OMT episode
python bill.py               # France's interest bill to 2036 with and without a spread cap
python channels.py           # other sovereigns, banks, the Eurosystem, the EU budget
python tools.py              # spreads around past ECB announcements
python figures.py            # nine figures in figures/
```

## Data

`fetch.py` downloads, and `.github/workflows/ecb-france-backstop-data.yml` refreshes and commits:

- Eurostat: monthly 10-year convergence-criterion yields (`irt_lt_mcby_m`); quarterly government
  debt, net lending and interest (`gov_10q_ggdebt`, `gov_10q_ggnfa`); quarterly GDP and real growth
  (`namq_10_gdp`); the quarterly current account (`bop_gdp6_q`); annual government accounts, debt
  and GDP (`gov_10a_main`, `gov_10dd_edpt1`, `nama_10_gdp`); EU-27 gross national income
  (`nasa_10_nf_tr`).
- ECB Data Portal: the same yields (`IRS`), TARGET balances (`TGB`).
- ECB website: PSPP and PEPP public-sector holdings by jurisdiction (CSV files linked from the
  programme pages).
- EBA: the 2025 EU-wide transparency exercise (`tr_sov.csv`, `tr_oth.csv`, metadata), June 2025.

Every request is logged in `data/fetch_log.txt`. `documented.csv` holds the facts taken from
official documents and dated reports (the TPI's terms, France's excessive deficit procedure, the
October 2026 spread peak, ratings, loss-sharing rules, the capital key, NGEU repayment), each with
its source.

## Method

- **Warranted spread.** For nine euro-area sovereigns, the quarterly 10-year spread to Germany is
  regressed on fundamentals relative to Germany, lagged one quarter. 120 models: six feature sets,
  with and without the German yield, with and without country fixed effects, five estimation
  windows (2000-2007; 2000-2019 without the crisis; 2013-2019; 2000Q1-2024Q2; real time). A model
  flags a quarter when the residual exceeds the 95th percentile of its own residuals. The headline
  uses the 96 models outside the 2000-2007 window, which would call almost any spread unwarranted.
- **Bill.** France's implicit interest rate moves towards the 10-year yield at an estimated
  pass-through; the yield is split into German yield, warranted spread and residual, and scenarios
  cap the spread at the warranted level, at the level where half the models flag, or at zero.
- **Channels.** Spillover of France-specific spread moves to eight other sovereigns (with a
  calibration on simulated data); bank exposures to French general government against CET1 and a
  mark-to-market loss on fair-valued holdings; Eurosystem holdings of French bonds and their
  loss-sharing split; France's share of EU GNI applied to NGEU repayment.

## Results

`results/warranted_summary.txt`, `results/bill_summary.txt`, `results/channels_summary.txt` and
`results/tools_summary.txt` hold every number used in the article; the CSV files hold the series
behind the figures. The reproduce workflow, `.github/workflows/ecb-france-backstop.yml`, reruns
tests, analysis and figures on every change and fails if a summary number moves beyond a small
tolerance (`check_results.py`).
