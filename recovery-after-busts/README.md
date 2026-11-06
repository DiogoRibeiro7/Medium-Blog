# Recovery after capacity busts in EU industries

Computational companion to a Medium article in preparation. Question: when an industry loses a large share
of its capacity, does the loss come back, how fast, and does it depend on whether the industry's output is
traded across the single market?

## Run

```bash
python -m pip install -r requirements.txt
python episodes.py          # A*64 industries, 1995-2025: results/episodes.csv, episodes_all.csv, episodes_summary.txt
python long_episodes.py     # AMECO sectors, 1960-2025: results/long_episodes.csv, long_episodes_summary.txt
python recovery_models.py   # mixture cure models: results/cure_*.csv, km_not_recovered.csv, cure_summary.txt (about 5 minutes)
python -m pytest -q tests
```

## Definitions

- **Episode (bust)**: a fall of at least θ log points below the running maximum of the log series
  (θ = 0.20 for A*64 industries, 0.15 for the coarser AMECO sectors). The peak is the year of that maximum,
  the trough the lowest year before the series next exceeds the peak. Depth = log(peak / trough).
- **Recovery**: the series exceeds its pre-bust peak (full) or regains half of the lost log points (half),
  counted in years from the trough and right-censored at the last observed year.
- **Capacity measures**: employment (persons) and gross value added in volume.
- **Versions**: *absolute* (own log level) and *relative* (own log level minus the cumulated leave-one-out
  mean log change of the same industry in the other countries). The relative version removes trends and
  shocks shared by all countries, so a bust is a loss of the country's place among its peers.
- **Tradable**: agriculture, mining and manufacturing (NACE A, B, C). Sensitivity: the most traded services
  (H50, H51, J, K, M69–M75) counted as tradable.
- **Integration stage** at the trough year (`stages.py`): pre-accession; customs union (member, before the
  single market; the founding six from July 1968); single market (from 1993 or accession; 1994 for AT, FI
  and SE through the EEA); euro (from adoption); outside (US, Japan, Canada, Australia, New Zealand,
  Switzerland, Norway, Iceland, and the UK from 2021).
- **Main sample**: excludes episodes whose peak is the first year of the series (the true peak may be
  earlier) and central and eastern European episodes with troughs before accession (transition).

## Model

A mixture cure model (`cure.py`): each bust either never regains its peak, with logit probability depending
on tradability, stage and log depth, or regains it after a lognormal time depending on tradability and log
depth. Fitted by maximum likelihood (BFGS, light ridge penalty); 90% intervals from a bootstrap over
countries. The never-recover share is identified by the plateau of the recovery curve, so it is fragile when
follow-up is short: the A*64 robustness check restricted to troughs up to 2015 shows how much.

## Data

- Eurostat `nama_10_a64_e` (employment), `nama_10_a64` (value added, chain-linked volumes), `nama_10_gdp`,
  `prc_hpi_a`, downloaded by `fetch.py`.
- AMECO (European Commission), sectoral employment and value added at 2020 prices, downloaded from DBnomics
  by `fetch_long.py`. West Germany to 1991 is chained to unified Germany at 1991.

Both are run by `.github/workflows/recovery-after-busts-data.yml`, which commits the snapshot and
`data/fetch_log.txt`. Eurostat and AMECO data are reused under Commission Decision 2011/833/EU; they are not
relicensed by the repository's licence.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt
