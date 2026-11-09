# Was Portugal Priced on Its Own Fundamentals in 2010-2012? Warranted Spreads, Contagion from Greece and Ireland, and Five European Decisions

Code, data and figures for the article. The question: in the euro crisis, did Portugal's bond
spread reflect its own fundamentals, which on the public accounts were better than Greece's, or
was it driven beyond them by contagion from Greece and Ireland and by European policy choices?

## Run

```bash
pip install -r requirements.txt
python fetch.py --out data   # or let the data workflow do it
python -m pytest -q tests
python warranted.py          # 160 warranted-spread models; Portugal, Greece and Ireland compared
python contagion.py          # Greece- and Ireland-specific shocks and Portugal's response
python events.py             # the months of the European decisions
python programme.py          # programme forecasts, lending terms, the 2011 ECB hikes
python figures.py            # five figures in figures/
```

## Data

`fetch.py` downloads, and `.github/workflows/portugal-2010-contagion-data.yml` refreshes and commits:

- Eurostat: monthly 10-year convergence-criterion yields (`irt_lt_mcby_m`); quarterly government
  debt, net lending and interest (`gov_10q_ggdebt`, `gov_10q_ggnfa`); quarterly GDP and real growth
  (`namq_10_gdp`); the quarterly current account (`bop_gdp6_q`); annual net international
  investment position (`tipsii10`) and private debt (`tipspd10`); annual deficit and debt
  (`gov_10dd_edpt1`), real growth (`nama_10_gdp`) and unemployment (`une_rt_a`).
- ECB Data Portal: the same yields (`IRS`), TARGET balances (`TGB`), three-month Euribor and the
  main refinancing rate (`FM`), rates on new loans for house purchase and to firms (`MIR`), and the
  stock of Portuguese housing loans (`BSI`).

Every request is logged in `data/fetch_log.txt` (the quarterly investment-position request is
refused by Eurostat and the annual one is used). `documented.csv` holds the facts taken from
official documents and dated reports (the debt and deficit figures published in April 2010, the
programme projections and lending terms, the July 2011 summit terms, event dates, the Deauville
event study), each with its source.

## Method

- **Warranted spread.** For nine euro-area sovereigns (Greece excluded from estimation and
  predicted out of sample), the quarterly 10-year spread to Germany is regressed on fundamentals
  relative to Germany, lagged one quarter. 160 models: eight feature sets (including external debt,
  measured by the net international investment position), with and without the German yield, with
  and without country fixed effects, five estimation windows. The headline uses the 128 models
  outside the 2000-2007 window. A real-time check prices the 2009 debt and deficit figures that
  markets saw in April 2010 instead of today's revised ones.
- **Contagion.** Monthly spread changes. The Greece-specific move is the part of Greece's change
  not shared with the average of seven other euro-area sovereigns; the Ireland-specific move is the
  part of Ireland's change shared neither with them nor with Greece. Portugal's change is regressed
  on the three (through the origin), giving responses and a cumulative decomposition. Block
  bootstrap intervals and a calibration on simulated data.
- **Events.** The monthly decomposition in the months of the decisions (Deauville, the EFSF and its
  size, the 2011 hikes, the July 2011 summit, the LTRO, OMT), with confounded months marked.
- **Programme.** Growth projected when the Greek and Portuguese programmes were designed against
  outturns; the April 2011 and July 2011 lending terms on the European share of Portugal's
  programme; Euribor, mortgage rates and the stock of Portuguese housing loans around the hikes.

## Results

`results/warranted_summary.txt`, `results/contagion_summary.txt`, `results/events_summary.txt` and
`results/programme_summary.txt` hold every number used in the article; the CSV files hold the
series behind the figures. The reproduce workflow, `.github/workflows/portugal-2010-contagion.yml`,
reruns tests, analysis and figures on every change and fails if a summary number moves beyond a
small tolerance (`check_results.py`).
