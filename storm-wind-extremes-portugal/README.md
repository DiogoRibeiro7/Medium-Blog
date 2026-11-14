# Storm wind extremes in Portugal

Companion code for an extreme value analysis of storm Kristin's winds (28 January 2026)
and of Portugal's storm-wind tail since 1950. Three questions:

1. How rare were Kristin's winds, place by place, and how sure can 76 winters make us?
2. Is the tail of the season-maximum wind getting heavier?
3. Do Portugal's wind design values (buildings and overhead lines) sit where the tail says?

## What it does

`wind_extremes.py` reads the ERA5 daily maximum 10 m wind at 38 nodes (1950/51 to
2025/26) and the GHCNh station records that the sibling project `storm-clustering-portugal`
already committed, plus the data that `fetch.py` adds here, and then:

- fits a generalized extreme value (GEV) distribution to the October-March maxima of each
  node, separately and with one shape parameter shared by all nodes, and gives Kristin's
  return period with season-block bootstrap and profile-likelihood intervals, fitted with
  and without the 2025/26 season;
- simulates what including or excluding the event that prompted the study does to its
  estimated return period;
- checks the shape with peaks over a threshold (declustered storms, generalized Pareto),
  shows how the return period depends on the shape, and repeats the fit with ERA5 back
  to 1940 so the cyclone of 15 February 1941 is in the record;
- tests for trends in the location and scale of the GEV with a field test (the season
  order permuted for all nodes at once), uses the winter NAO as a covariate of known
  physical relevance, and measures the power of the test for trends of given size;
- compares the Portuguese National Annex of Eurocode 1 (basic wind velocity, its
  Gumbel-based conversion between return periods, and the implied peak velocity) and
  the overhead-line design gusts with the fitted tail, with station 50-year mean winds,
  and with the gusts IPMA reported during Kristin;
- measures how far a season maximum is shared between nodes (F-madogram extremal
  coefficients) and what that means for a network of many sites.

`evtlib.py` holds the extreme value tools (GEV and GPD likelihoods with analytic gradients,
profile likelihood, pooled-shape fits, declustering, extremal index, madogram). The tests
in `tests/` check them against `scipy.stats` and simulated data.

## Data

From `../storm-clustering-portugal/data` (see that project's README for sources):
ERA5 daily maximum 10 m wind at 38 nodes (Open-Meteo archive API, CC BY 4.0), the GHCNh
daily station table (NOAA NCEI), and the monthly NAO index (NOAA CPC).

Added here by `fetch.py`, run by the workflow `storm-wind-extremes-portugal-data.yml`:

- `data/era5_gust/`: ERA5 daily maximum 10 m gust at the 38 nodes;
- `data/era5_sites/`: ERA5 daily maximum 10 m wind at the grid cells of nine stations;
- `data/era5_kristin/`: ERA5 daily maximum wind and gust on 26-30 January 2026 at the
  cells of the stations whose Kristin gusts IPMA reported;
- `data/era5_early/`: ERA5 daily maximum 10 m wind at the 38 nodes, 1940-1950;
- `data/kristin_reports.csv`: every GHCNh report from five airports, 26-30 January 2026;
- `data/era5_cells.csv`, `data/fetch_log.txt`: returned grid cells, and a log of every
  request with its SHA-256. (The first run committed the two CSV files through Git LFS
  before the folder was exempted from it; the cell table for its 57 requests was restored
  from that run's printed log.)

Written by hand, with sources in the file:

- `data/kristin_gusts.csv`: gusts reported on 28 January 2026 (IPMA and the Coimbra
  intermunicipal network, via Lusa), approximate coordinates and Eurocode wind zone;
- `data/portugal_outline.csv`: Natural Earth 1:50m outline (public domain), cut by
  `make_outline.py`.

The design values are in the constants at the top of `wind_extremes.py`: NP EN 1991-1-4
National Annex (v_b,0 = 27 m/s in zone A and 30 m/s in zone B, with
v_b(T)/v_b,0 = 0.7 - 0.077 ln(-ln(1 - 1/T))), the overhead-line values of EN 50341-3-17 as
reproduced in two Porto engineering theses, and E-REDES specification D00-C10-001.

## Run

```
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python wind_extremes.py > results/output.txt
```

`WIND_QUICK=1` runs a smoke test with few permutations and bootstrap samples. The
workflow `storm-wind-extremes-portugal.yml` reruns everything; on the working branch it
commits `results/output.txt` and the figures, and on `main` and pull requests it checks
that the output is reproduced exactly. The numbers quoted in the article come from that
run.
