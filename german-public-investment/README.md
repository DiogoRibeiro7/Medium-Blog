# Germany's Public Capital Stock Under the Debt Brake: Net Investment, Fiscal Room and the 2025 Reform Against Five Low-Debt Peers

Code, data and figures for the article. The question: did Germany's debt brake cause it to
under-invest, judged against the Netherlands, Austria, Finland, Denmark and Sweden, and what has
changed since the reform of March 2025?

## Run

```bash
pip install -r requirements.txt
python fetch.py --out data   # or let the data workflow do it
python -m pytest -q tests
python investment.py         # net investment, capital stock, subsectors, functions
python space.py              # balances against the brake, fiscal room, counterfactual debt
python reform.py             # quarterly investment to mid-2026, documented reform figures
python figures.py            # six figures in figures/
```

## Data

`fetch.py` downloads from Eurostat, and `.github/workflows/german-public-investment-data.yml`
refreshes and commits: government accounts by subsector (`gov_10a_main`: gross fixed capital
formation, consumption of fixed capital, investment grants paid, interest, net lending), debt
(`gov_10dd_edpt1`), GDP (`nama_10_gdp`), population (`nama_10_pe`), net fixed assets of general
government (`nama_10_nfa_bs`), investment by function (`gov_10a_exp`) and quarterly investment and
GDP (`gov_10q_ggnfa`, `namq_10_gdp`). Every request is logged in `data/fetch_log.txt`.

`documented.csv` holds the figures taken from official statements and studies (the reform's
parameters, the special fund's outflows, the KfW municipal panel, the IMK and IW estimate of needs),
each with its source and date.

## Method

- **Net investment** is gross fixed capital formation minus consumption of fixed capital of general
  government. Because Germany finances rail and hospitals largely through grants to bodies outside
  general government, a second measure adds investment grants paid; it also includes subsidies to
  private investment, so the two measures bracket public capital spending.
- **Periods**: 1995-2008 (before the debt brake), 2011-2019 (brake in force, before the pandemic),
  2020-2024 (escape clause and after). The change between the first two periods is compared with
  each peer's change.
- **Fiscal room**: headline balances by subsector against the brake's limits (the brake applies to
  structural balances in budget accounting; a cyclical allowance is stated); the primary balance
  against the debt-stabilising primary balance; and a counterfactual debt path in which Germany
  invests at its peers' average net rate, financed by debt at its own implicit interest rate, with no
  growth dividend.
- **Reform**: rolling four-quarter gross public investment to 2026-Q2, and first-half comparisons.

## Results

`results/investment_summary.txt`, `results/space_summary.txt` and `results/reform_summary.txt` hold
every number used in the article; the CSV files hold the series behind the figures. The reproduce
workflow, `.github/workflows/german-public-investment.yml`, reruns tests, analysis and figures on
every change and fails if a summary number moves beyond a small tolerance (`check_results.py`).
