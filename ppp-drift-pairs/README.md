# The price drift between every pair of economies, and who compiles their PPPs

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. The previous follow-up found that the drift between constant- and current-PPP histories (deflators minus PPPs) separates European economies from non-European ones, and stated a conjecture: that the drift arises mainly where the European PPP comparison is connected to the non-European one, not within either group. This repository tests it on every pair of economies in the OECD's national accounts and PPP tables.

It measures the drift for GDP, actual individual consumption and household consumption for 43–46 economies; turns them into 903–1,035 pairs; compares pairs within the Eurostat-compiled group, within the OECD-compiled group and across the two; fits a block model with a permutation test and a price-level control; reads the annual steps for the signature of how each group's PPP time series were built; and follows the gap between the groups from 1995 to 2024.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD and Eurostat snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig4_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/ppp-drift-pairs.yml` reruns the tests and the analysis, checks that the summary is unchanged and draws the figures.

## Data (`fetch.py`, `data/`)

`data/fetch_log.txt` records the URL, size and SHA-256 of every response; raw responses are kept gzipped in `data/raw/`.

- **OECD Data Explorer** (SDMX API, national accounts `DSD_NAMAIN10`, 13 November 2026):
  - `DF_TABLE1_EXPENDITURE`: GDP, actual individual consumption and household final consumption expenditure, national currency, current prices and chain-linked volumes, every economy published;
  - `DF_TABLE4`: PPPs for the same three aggregates (national currency per US dollar) and exchange rates.
- **Eurostat** `prc_ppp_ind`: PPPs relative to the EU27 for the same aggregates, used only to check that the OECD and Eurostat series are the same numbers for the European comparison (they are, to 1e-4 log points).

OECD and Eurostat data are reused under their terms; neither is relicensed by the repository's licence. Values for 2025 are not used.

## Definitions

- **Drift** of economy i against j, relative to 2020: ln(P_i / P_j) − ln(PPP_i / PPP_j), with P the implicit deflator (current prices ÷ chain-linked volume) and PPP in national currency per US dollar, both 2020 = 1, in log points × 100. Because both terms are transitive, the drift of i against j equals the drift of i against the US minus the drift of j against the US, so n economies give n(n−1)/2 pairs built from n−1 independent numbers (`tests`).
- **Groups** follow who compiles the PPPs in the Eurostat-OECD programme. *Eurostat group*: the EU27, Switzerland, Norway, Iceland, Türkiye, Serbia, North Macedonia, Albania, Montenegro, Bosnia and Herzegovina, and the United Kingdom (compiled by Eurostat until it moved to the OECD side after leaving the EU). *OECD group*: the OECD's non-European members (United States, Japan, Korea, Canada, Australia, New Zealand, Israel, Mexico, Chile, Colombia, Costa Rica). Other economies in the OECD tables are left out.
- **Block model**: OLS of each economy's drift against the US over 2005–2019 on a Eurostat-group indicator, optionally with the log of the 2005 price level (PPP ÷ exchange rate, US = 100) as a control. The gap is the difference between group means; the permutation p-value is the share of 10,000 random relabellings with the same group sizes whose gap is at least as large (seed fixed).
- **Steps**: annual changes in an economy's drift against the US. A *quiet* step is one below 0.3 points in absolute value.
- **Group gap over time**: mean drift against the US of the Eurostat-group economies with every year 1995–2024 minus the mean of the OECD-group economies with every year.
- **Benchmark years**: the Eurostat-OECD benchmark rounds 1999, 2002, 2005, 2008, 2011, 2014 and 2017; the OECD's own calculation for its non-European members was triennial until 2016.

## Caveats

- Pairs are not independent: they carry the same information as the country drifts, and the pair statistics are summaries of the country drifts, not extra evidence.
- Group membership is almost the same as geography. The only economy that changed side, the United Kingdom, did so after 2019, which leaves five years to observe it.
- Chile and Costa Rica lack household-consumption PPPs for both 2005 and 2019; the group sizes differ slightly by aggregate.
- The published series are one vintage (13 November 2026). PPPs back to 1970 may be revised every December, and the European series for 1995–2012 were revised exceptionally in December 2016.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
