# The EU–US drift between price bases: does ICT deflation explain it?

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. Part 1 of the series found that Europe's GDP per hour relative to the United States has two official histories since 1995: one at constant 2020 PPPs, which falls, and one at each year's PPPs, which does not. The two run parallel until 2005 and diverge afterwards. Part 1 stated a conjecture: that the divergence comes mainly from how ICT goods, software and digital services are deflated in national accounts, not from changes in the methods of the PPP benchmarks. This repository tests it.

It measures the drift between the two price bases for GDP, actual individual consumption and household consumption; compares it across 43 economies (41 for household consumption); measures how fast ICT prices fell in the United States and in Europe in household consumption, in the value added of the ICT-producing industries and in investment; and recomputes the drift with ICT left out of the deflators or deflated the same way on both sides of the Atlantic.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD and Eurostat snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/eu-us-ppp-drift.yml` reruns the tests and the analysis, checks that the summary is unchanged and draws the figures.

## Data (`fetch.py`, `data/`)

`data/fetch_log.txt` records the URL, size and SHA-256 of every response; raw responses are kept gzipped in `data/raw/`.

- **OECD Data Explorer** (SDMX API, national accounts `DSD_NAMAIN10`, 12 November 2026):
  - `DF_TABLE1_EXPENDITURE`: GDP, actual individual consumption and household final consumption expenditure, national currency, current prices and chain-linked volumes, every economy published;
  - `DF_TABLE4`: PPPs for GDP, household consumption and actual individual consumption (national currency per US dollar) and exchange rates;
  - `DF_TABLE5_T501`: household consumption by purpose (COICOP), current prices and volumes, domestic concept;
  - `DF_TABLE6`: gross value added by industry (ISIC Rev. 4), current prices and volumes;
  - `DF_TABLE1_EXPENDITURE_GFCF_ASSET`: gross fixed capital formation in ICT equipment, software and databases and in total.
- **Eurostat** `prc_ppp_ind`: PPPs relative to the EU27 for GDP, actual individual consumption and household consumption, used only to check that the OECD and Eurostat agree on the PPP of the United States.

OECD and Eurostat data are reused under their terms; neither is relicensed by the repository's licence. Values for 2025 are not used.

## Definitions

- **Drift.** For an economy against the United States, relative to 2020: ln D = ln(P / P_US) − ln(PPP), with P the implicit deflator (current prices ÷ chain-linked volume) and PPP in national currency per US dollar, both 2020 = 1. It is the change in relative prices recorded by the national deflators minus the change recorded by the PPPs, and equals the log of the current-PPP level divided by the constant-PPP level. A rising drift means that the deflators recorded more European inflation, relative to the United States, than the PPPs did. Changes are in log points × 100.
- **EU group.** The 24 members used in Part 1: the EU27 without Cyprus, Malta (not in the OECD productivity database) and Estonia (no 1995 data there). Group values are sums at current PPPs divided by sums at 2020 PPPs, so the group drift is exact, not an average of country drifts. It splits exactly into a group deflator and a group PPP index (`drift.group_parts`).
- **ICT.** In household consumption, COICOP 08 (communication) and 09.1 (audio-visual, photographic and information-processing equipment). In value added, ISIC C26 (computer, electronic and optical products) and section J (information and communication), with J58, J59–60, J61 and J62–63 reported separately. In investment, ICT equipment (N1132) and computer software and databases (N1173).
- **Relative price of an item** = its deflator ÷ the deflator of its aggregate (all household consumption, all value added, or GDP for investment). Europe's path is the mean annual change over the EU members reporting the item, weighted by the size of the aggregate in US dollars at PPPs.
- **Counterfactuals.** All change the deflators and leave current-price values and PPPs as published, so only the constant-PPP history moves. *ICT left out*: the deflator of the aggregate without the ICT items, as a Törnqvist index (annual log change of the total minus the share-weighted changes of the ICT items, divided by one minus their share), on both sides. *US priced like Europe*: the US relative price change of each ICT item is replaced by Europe's. *Europe priced like the US*: each member's relative price change of each ICT item is replaced by the US one. On the production side the adjustment to the GVA deflator is applied to the GDP deflator.
- **Investment upper bound** = US share of GDP of each ICT asset × (Europe's relative price change − the US one), summed. It assumes all ICT investment is produced at home, so that a different deflator passes fully into GDP, which overstates the effect because much ICT equipment is imported.

## Caveats

- The PPPs for the United States are published only for GDP, actual individual consumption and household consumption, not for ICT items. The counterfactuals therefore ask how much of the drift would disappear if ICT were deflated differently, holding the PPPs as published; they cannot say whether the ICT PPPs themselves moved like the deflators.
- On the production side, leaving out the ICT industries' value added does not remove ICT prices from the double deflation of the industries that buy ICT services, so it is a partial measure of the deflation channel. The investment upper bound covers the main part of the rest.
- Table 5 is on the domestic concept and Table 1 household consumption on the national concept; only relative price changes within Table 5 are used. Four EU members (Czechia, Ireland, the Netherlands, Romania) do not report the ICT consumption categories (10% of the group's consumption); their adjustment is set to zero. Ireland and Luxembourg do not publish C26 volumes; only 10 members publish software investment.
- The Törnqvist aggregation of sections reproduces the published GVA deflator to within 0.5 points over 2005–2019 for the United States and 3 points for every EU member; COICOP divisions reproduce the consumption deflator to within 0.5 points for the United States and 1.5 points for the five largest EU members (tests).

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
