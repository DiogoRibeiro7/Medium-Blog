# Who the EU's purchases of US securities are made for: Irish and Luxembourg funds in mirror statistics

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. Part 4 of the series stated a conjecture: that much of the EU's gross buying of US securities is done by investment funds resident in Ireland and Luxembourg on behalf of investors outside the EU, and is matched by those investors' purchases of the funds' shares, which the balance of payments records as non-EU purchases of EU securities. This repository tests it with the geographical breakdowns of the 27 member states' balance of payments and international investment positions.

It measures how much of the EU's buying and holding of US securities is done by residents of the two hubs; estimates who holds the hubs' equity and fund shares as the difference between what non-residents hold and what the other 26 members report holding (mirror data); attributes the hubs' US purchases to EU and non-EU owners in proportion to that ownership; splits the EU's gross portfolio inflow into purchases of hub securities and the rest; and checks whether the hubs' quarterly US purchases move with the money flowing into their funds.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Eurostat snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig6_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/ie-lu-fund-intermediation.yml` reruns the tests and the analysis, checks that the summary is unchanged and draws the figures.

## Data (`fetch.py`, `data/`)

`data/fetch_log.txt` records the URL, size and SHA-256 of every response; raw responses are kept gzipped in `data/raw/`, and `python fetch.py --tidy-only` rebuilds the CSV files from them.

- **Eurostat** (JSON-stat dissemination API, 13 November 2026), million euro:
  - `bop_eu6_q`: EU27 portfolio investment against non-EU residents, the United States and the United Kingdom, quarterly; assets by resident sector, liabilities in total;
  - `bop_c6_q`: portfolio investment of each member state by partner (rest of the world, other EU members, non-EU, United States, United Kingdom, Ireland, Luxembourg), quarterly, for equity and investment fund shares (and each separately) and debt securities;
  - `bop_iip6_q`: the matching end-of-year positions for each member and for the EU27;
  - `nasa_10_f_bs`: financial balance sheets of Ireland, Luxembourg and the EU27 by sector, used for the fund shares held by residents of each hub.

Eurostat data are reused under Eurostat's terms; they are not relicensed by the repository's licence. The ECB securities holdings statistics, the IMF Coordinated Portfolio Investment Survey and the Central Bank of Ireland's statistics on fund holders by geography would sharpen the estimates; their servers were not reachable from the environment used to build this snapshot, so they are not used.

## Definitions

- **Hubs**: Ireland (IE) and Luxembourg (LU). **Hub securities**: portfolio equity and investment fund shares (BPM6 item F5) issued by residents of a hub.
- **Mirror split**: non-resident holdings of a hub's securities (its portfolio liabilities) minus the holdings of those securities reported by each other member (its assets with the hub as partner). The remainder is held outside the EU. Flows are split the same way each quarter.
- **Missing cells**: France and Austria do not publish their holdings in either hub, Luxembourg does not publish its holdings in Ireland, Germany's flows by partner stop after 2020-Q3, and a few small members start late. A missing holding is imputed as the member's intra-EU holdings of equity and fund shares times the hub's share in the intra-EU holdings of the members that publish both. A missing flow is imputed as the member's intra-EU net purchases in the quarter times the hub's share of its intra-EU holdings at the end of the previous year (its own share where published, as for Germany, otherwise the reporters' share). **Bounds** put the missing members' whole intra-EU holdings (or the positive or negative part of their flows) in the hubs, or none of them.
- **Luxembourg's foreign holdings by partner**: Luxembourg publishes only its total portfolio assets in the position data, so its non-EU holdings are the EU27 total minus the other 26 members and its intra-EU holdings are its total minus that. Where a small member does not publish its non-EU holdings or purchases (positions: Greece to 2018, Malta to 2016, Belgium and Bulgaria in 2014; flows: Greece to 2019 and 24 quarters for Malta), they are filled with its total times its first published non-EU share of holdings. Luxembourg's US equity is its total US securities minus US debt securities.
- **Non-EU ownership of a hub's funds** (ω): fund shares held outside the EU divided by fund shares held by anyone, residents of the hub included (financial balance sheets). Fund shares held outside the EU are the mirror residual for equity and fund shares minus the hub's company shares held outside the EU, which are its company-share liabilities minus the part EU members hold (estimated with the ratio of company shares in the holdings of the members that publish both).
- **Attribution**: each hub's purchases (holdings) of US securities times its ω, averaged over the start and end of the year for flows. The joint lower bound assigns each missing member's intra-EU holdings to the hub where they lower the attribution most.
- **Co-movement**: quarterly OLS of the hubs' purchases of US securities on non-EU and EU purchases of hub securities, with Newey-West standard errors (four lags), 2015Q1–2025Q4; the same for the hubs' purchases of all non-EU securities and, as a placebo, for the other 25 members' purchases of US securities.

## Caveats

- The residual includes every error in the hubs' liabilities and the other members' assets; it is "not reported by an EU member", which is not proof of a non-EU owner.
- The attribution assumes that a hub's US securities are owned in the same proportions as its fund shares. Funds sold mainly outside the EU may hold more or less US paper than funds sold inside it, and not all of a hub's portfolio assets belong to funds.
- Fund shares that EU investors keep with custodians outside the EU can be missing from their own country's reported assets; they then fall in the non-EU residual.
- The breakdown says nothing about which countries the non-EU owners are in.
- Data are one vintage (13 November 2026); balance-of-payments series are revised.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
