# Integration as insurance: correlated supply shocks in the EU single market

Computational companion to a Medium article. Claim: a single market insures its members against national
supply shocks only to the extent that those shocks are independent. Weather-type harvest shocks across the
EU-27 are weakly correlated and pool well; common shocks (energy and input prices, the ageing of farm
managers, interest rates for housing) cannot be pooled. Food is the traded case; housing, which cannot be
traded, is the contrast.

## Run

```bash
python -m pip install -r requirements.txt
python evidence.py     # results/evidence_summary.txt, country_table.csv, evidence_*.csv
python simulate.py     # results/summary.txt, model_*.csv (about 15 seconds)
python figures.py      # figures/fig1_*.png ... fig7_*.png
python -m pytest -q tests
```

## What is computed

- **Shocks**: log national cereal production (Eurostat `apro_cpsh1`, cereals for grain), 26 member states
  (Malta grows none), 2000–2024, minus a Hodrick–Prescott trend (lambda 100). Correlations, diversification
  ratio, effective number of independent producers, lower- and upper-tail co-exceedance against the
  Gaussian benchmark, a 3-year moving-block bootstrap, sub-period correlations, production concentration.
- **Common shocks**: first principal component of national food inflation (HICP CP011), cereal shocks and
  house price growth.
- **Housing**: construction employment by member state (`nama_10_a10_e`), the share of member states in
  which it fell each year, and peak-to-trough versus trough-to-2025 rates for the deepest busts.
- **Farm managers**: share of holdings and of standard output run by managers aged 65+ (`ef_m_farmang`).
- **Model** (`simulate.py`): 27 member states with population shares, a self-sufficiency proxy
  (production share over population share), shocks drawn with the estimated correlation (shrunk 10% to the
  identity), income-limited demand calibrated so that the share priced out at the reference price equals the
  national share unable to afford a meal with meat, fish or a vegetarian equivalent every second day
  (`ilc_mdes03`), and the national Gini (`ilc_di12`). Pooled: one EU price, with a world buffer of 5% of
  consumption at the world price. Unpooled: normal trade kept, each country's own harvest deviation stays at
  home. Sweeps over the mean correlation and over the share of each shock that is common (with the world price
  moving with it); results by member state; an ageing scenario in which half of the output of holdings run by
  managers 65+ is not replaced.

## Data

`fetch.py` downloads every Eurostat dataset used (all member states in one request each) and is run by
`.github/workflows/eu-integration-correlated-shocks-data.yml`, which commits the snapshot and
`data/fetch_log.txt` (URL, status, filter set, SHA-256 of each response). Source: Eurostat, reused under
Commission Decision 2011/833/EU; the data are not relicensed by the repository's Apache-2.0 licence.
