# When the market stops correcting: capacity lock-in in food and housing

Computational companion to a Medium article on when price signals fail to bring supply back. The article's
claim: a market corrects a shortage only if capacity is rebuilt faster than it is lost and people can still pay.
When either condition fails, it can lock into scarcity instead of correcting it.

The folder has two parts:

- **Evidence** (`evidence.py`): Portuguese series on farm labour, farm managers' ages, agricultural output and real
  producer prices, food prices, construction employment, dwelling completions, house prices relative to pay, and
  housing-cost overburden.
- **Model** (`model.py`, `scenarios.py`): a market for an essential good with income-limited demand, anchored
  domestic capacity that is lost fast and rebuilt slowly, and borrowed supply (imports, non-resident labour) at a
  fixed cost up to a cap. A food version (flow good, borrowing possible) and a housing version (stock good, no
  imports, fitted to Portugal 2008–2025).

## Run

```bash
python -m pip install -r requirements.txt
python evidence.py      # results/evidence_summary.txt and evidence_*.csv
python scenarios.py     # results/summary.txt and the CSVs behind each figure (about 6 minutes)
python figures.py       # figures/fig1_*.png ... fig8_*.png
python -m pytest -q tests
```

`MC_N=200 python scenarios.py` runs a smaller Monte Carlo for a quick check.

## Figures

- `fig1_farms`: farm labour by type, output volume against labour and real farm-gate prices, farm managers by age
- `fig2_housing`: completions since 1975; construction employment, completions and house prices relative to pay since 2008
- `fig3_exclusion`: share of households who cannot afford their need, as a function of price
- `fig4_fold`: steady states of anchored food capacity against the effort premium, open and capped borrowing
- `fig5_erosion`: a slow rise in the effort premium, then a disruption to borrowed supply
- `fig6_wave`: capacity after a retirement wave, by its speed and by how eroded the sector already was
- `fig7_housing_model`: the housing fit and four counterfactual histories
- `fig8_exposure`: priced-out share during a disruption, as a function of anchored capacity

## The model in one paragraph

Incomes are lognormal with Portugal's 2025 Gini (30.9). Each household needs one unit of the good and spends at
most 40% of its income on it; a reference price is set so that 3% (food) or 5% (housing) cannot afford their
need. Food demand also has a discretionary layer, so the aggregate price elasticity at the reference price is
−0.22. Anchored capacity *K* follows the margin between price and unit cost. The cost is high when the sector is
small (no suppliers, buyers or people to learn from), falls as it grows and rises again at large scale. It
also includes a premium *φ* for drawing people into physical work. Capacity grows at most 7% a year (scaled by
*K/(K+K_g)*, since newcomers learn from incumbents) and shrinks at up to 9% a year when production loses money,
plus 2.5% attrition. With these rates, the fastest net rebuild is about 3.7% a year and the fastest loss 11.5%,
which matches Portuguese construction employment in 2014–2025 and 2008–2013. Borrowed supply is available at
the reference price up to a cap *M*.

All model parameters are in `model.py` and `scenarios.py`. The Monte Carlo redraws the shape of the cost curve
and the speeds over wide ranges (2,000 draws) to check which conclusions survive.

## Data

- `fetch.py` downloads the Eurostat datasets not already pinned. It is run by
  `.github/workflows/market-capacity-lock-in-data.yml`, which commits the snapshot and `data/fetch_log.txt`
  (URL, status, filter set and SHA-256 of every response). Datasets: `ef_m_farmang`, `aact_ali01`, `aact_eaa05`,
  `sts_copr_a`, `ilc_di12`, `ilc_di03`.
- `data/archived/` holds unmodified snapshots copied from other folders of this repository, with checksums in
  `data/archived/SHA256SUMS`:
  - from `portugal-housing-generations` at commit `7883bad` (the project now lives in its own repository): INE
    indicators 0000076 and 0011725 (dwellings completed), 0013235 (dwellings licensed); Eurostat `prc_hpi_a`,
    `sts_cobp_a`, `ilc_lvho07a`, `ilc_lvps08`, `lfst_hhnhtych`, `prc_hicp_aind` (rents);
  - from `portugal-labour-share-industries`: Eurostat `nama_10_a10`, `nama_10_a10_e`;
  - from `portugal-household-spending`: Eurostat `prc_hicp_aind`.

Terms: INE data are reused under CC BY 4.0 ("Fonte: Instituto Nacional de Estatística, IP – Portugal"); Eurostat
data under the Commission's reuse decision 2011/833/EU ("Source: Eurostat"). Neither is relicensed by this
repository's Apache-2.0 licence, which covers the code, figures and results.
