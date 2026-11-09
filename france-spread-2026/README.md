# France's Bond Spread in 2026, Measured Against 2011: Co-Movement, Pass-Through and the Interest Bill to 2036

Code, data and figures for the article. The question: is France's 2026 spread widening a 2011-style
euro-area crisis, and how much of the coming rise in its interest bill is already decided? France
2025-2026 is compared with Italy and Spain in 2011.

## Run

```bash
pip install -r requirements.txt
python fetch.py --out data   # or let the data workflow do it
python -m pytest -q tests
python fiscal.py             # levels, pass-through, backtests
python contagion.py          # co-movement of spreads
python stress.py             # scenarios and Monte Carlo to 2036
python figures.py            # seven figures in figures/
```

## Data

`fetch.py` downloads, and `.github/workflows/france-spread-2026-data.yml` refreshes and commits:

- Eurostat: monthly 10-year convergence yields (`irt_lt_mcby_m`); annual interest payable (D.41),
  net lending (B.9), Maastricht debt and GDP (`gov_10a_main`, `gov_10dd_edpt1`, `nama_10_gdp`);
  quarterly interest, debt and GDP from 2019 (`gov_10q_ggnfa`, `gov_10q_ggdebt`, `namq_10_gdp`).
- ECB Data Portal: the same convergence yields (dataset `IRS`), which fill months Eurostat has not
  yet published.

Every request is logged with its URL, HTTP status and SHA-256 in `data/fetch_log.txt`.

The analysis is monthly. On 9 November 2026 no open daily source of euro-area 10-year yields
answered a scripted request: Eurostat's daily table is not disseminated through the API, the ECB's
daily benchmark yields return no series, Stooq answers 404, and the Banque de France's daily
constant-maturity rate needs a Webstat API key. The fetch log of that day records the attempts. The
monthly series ended in September 2026 at that date; rerunning the data workflow picks up later
months as they are published.

## Method

- **Co-movement** (`contagion.py`). Spreads to Germany, monthly changes. For each country, the
  common factor of the other eight (Italy, Spain, Belgium, Austria, the Netherlands, Finland,
  Portugal, Ireland, France; Greece left out) is the first principal component of their
  standardised changes. R² of the country's changes on that factor measures shared movement;
  the cumulative change splits into a part loaded on the factor and a country-specific rest.
  Windows of 24 months; moving-block bootstrap intervals (blocks of 3, 2,000 draws).
- **Pass-through** (`fiscal.py`). The implicit rate (interest payable over average debt) moves
  towards the 10-year yield: i(t) - i(t-1) = rho [x(t) - i(t-1)] + a. Estimated by country and
  pooled, 1999-2025; checked against 1 / average maturity + deficit / debt for France; backtested
  out of sample on Italy and Spain 2011-2019 and France 2020-2025.
- **Stress test** (`stress.py`). From 2025, the implicit rate follows the pass-through and the debt
  ratio follows d(t) = (A (1 + i/200) - pb) / (1 - i/200), A = d(t-1) / (1 + g/100). Scenarios
  for yields, the primary balance and growth, a decomposition of the rise in interest, and a
  Monte Carlo (20,000 paths). The 2026 projection is checked against the first half of 2026.

## Results

`results/fiscal_summary.txt`, `results/contagion_summary.txt` and `results/stress_summary.txt`
hold every number used in the article; the CSV files hold the series behind the figures. A second
workflow, `.github/workflows/france-spread-2026.yml`, reruns tests, analysis and figures on every
change and fails if a summary number moves beyond a small tolerance (`check_results.py`).

The scenarios are assumptions, stated in `stress.py` and in the summary: they are not forecasts.
