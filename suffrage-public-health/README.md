# Suffrage laws and child mortality: a staggered difference-in-differences reanalysis

Code for a reanalysis of Grant Miller, "Women's Suffrage, Political Responsiveness, and Child
Survival in American History", *Quarterly Journal of Economics* 123(3), 2008, with the data
Miller published on his Stanford page.

Miller relates state suffrage laws (1869-1920) to state mortality (1900-1936) and city spending
(1905-1930), with state (or city) and year fixed effects and state-specific linear trends, and
finds that child deaths fell by 8-15%. Every state has suffrage from 1920, so two-thirds of the
mortality panel has no untreated state. This folder asks how much of the estimate survives
estimators that use only clean comparisons (states that had not yet enfranchised women,
1900-1919), and a comparison of child deaths with adult deaths in the same state and year.

## Run

```
pip install -r requirements.txt
python fetch.py                 # downloads the two Stata files and checks their SHA-256
python -m pytest -q tests
python suffrage.py              # results/summary.txt and CSVs; about 8 minutes (bootstrap)
python figures.py               # figures/fig1-fig4
```

`python suffrage.py 49` runs with 49 bootstrap draws instead of 499, for a quick look.

## Files

- `fetch.py` downloads the two harmonized Stata files linked from the paper's page into
  `data/raw/` (not tracked) and stops if a checksum differs. The URLs, sizes and SHA-256 are
  registered in the data registry as
  [`external/miller-2008-suffrage-mortality-finance`](https://github.com/DiogoRibeiro7/data/tree/2cd5ea3859afac626ce5900797cba468a079104f/external/miller-2008-suffrage-mortality-finance)
  and copied in
  [`data_sources/miller-2008-suffrage-mortality-finance.json`](../data_sources/miller-2008-suffrage-mortality-finance.json)
  at registry commit `2cd5ea3859af`. Every table is rebuilt from them at each run.
- `dates.py` codes the year of the first suffrage law (full, presidential or primary) by state,
  from CAWP (2014). The coding reproduces Miller's Online Appendix Table 2 exactly
  (`tests/test_data.py`).
- `did.py` has the estimators: two-way fixed effects with optional group trends and
  state-clustered standard errors; the weights two-way fixed effects puts on each treated
  state-year; Callaway and Sant'Anna (2021) group-time effects with not-yet-treated controls;
  the Borusyak, Jaravel and Spiess (2024) imputation estimator, optionally with unit trends; and a
  state-cluster bootstrap.
- `suffrage.py` runs every analysis and writes `results/summary.txt`.
- `figures.py` draws the four figures from the CSVs in `results/`.

## What the run shows

- Miller's specification reproduces: deaths at ages 1-4 to 15-19 fall by 7.6% to 12.3%.
- Stopped in 1919, the same specification gives 1.9% for ages 1-19; clean-comparison
  estimators give 0% to 1.4% for child deaths and about 3% for child deaths relative to adult
  deaths, with bootstrap intervals that include zero.
- Most of the published estimate enters with 1920 and 1921, when the remaining states switch and
  their only comparison is states that had suffrage for years.
- With cities identified correctly, Miller's health-and-sanitation spending estimate reproduces
  (+0.059 against his +0.063).

## Data notes

- In the municipal file, `id` is the city's population rank in each year (New York is 1 every
  year), not a city identifier. Used as a city fixed effect it flips the sign of the health and
  sanitation coefficient. Cities are identified here by state and place FIPS codes (319 cities).
- Total spending for 1905 is not comparable with later years (its median is about the size of
  health and sanitation spending); total-spending results drop 1905.
- Quincy, Massachusetts appears twice in 1923; the first row is kept.
- The two Voteview files on the page need a Google sign-in, so the roll-call analysis is not
  reproduced.
- The data carry no licence statement, so they are not stored here: each run downloads them from
  the author's links and checks them against the registry, with credit to Miller (2008).

Contact: dfr@esmad.ipp.pt
