# Storm clustering in Portugal

Public reproducibility companion for Diogo Ribeiro's article on how often storms reach mainland Portugal in trains, how rare the 2025/26 winter was, and whether clustering has changed since 1950.

The manuscript is kept separately in the private [`research-articles-medium`](https://github.com/DiogoRibeiro7/research-articles-medium) repository. This folder holds the data, the code, the printed results and the figures.

## Run

Requirements: Python 3.11 or newer, NumPy, pandas, SciPy and Matplotlib (the versions CI uses are pinned in `requirements.txt`).

```bash
python -m pip install -r requirements.txt
python storm_clustering.py > results/output.txt
```

The analysis is deterministic and downloads nothing; it takes about half an hour, most of it in the simulation checks and the power analysis (`STORM_QUICK=1` runs a shortened version in a few minutes, for testing only). It prints every number quoted in the article to `results/output.txt`, writes seven figures to `figures/`, and renders the ten displayed equations to `figures/equations/` (Medium does not render LaTeX, so the article shows them as images). The workflow `.github/workflows/storm-clustering-portugal.yml` reruns it: on the working branch it commits the output, and on `main` and pull requests it fails if the printed numbers change.

## Data

Everything in `data/` is downloaded by two GitHub Actions workflows and committed with a log of every request (time, HTTP status, size, SHA-256 and URL). The analysis never touches the network.

**ERA5 daily maximum wind** (`data/era5/`, `fetch_era5.py`, workflow `storm-clustering-portugal-era5.yml`). The daily maximum of the hourly 10 m wind speed from ERA5 (Hersbach et al. 2020, Copernicus Climate Change Service), at 0.25°, served by the Open-Meteo historical weather API (CC BY 4.0), for 38 grid nodes: the nodes of a half-degree lattice inside mainland Portugal plus two on the Algarve coast, 1 October 1950 to 31 March 2026, days in UTC. Open-Meteo returned the nearest land cell for two coastal requests (37.0°N 8.5°W became 37.25°N; 38.5°N 9.0°W became 38.75°N); `data/era5_points.csv` lists requested and returned coordinates. No value is missing. The download is resumable and pauses on the free API's rate limits; it took about an hour and a half.

**Station records** (`data/daily_wind.csv.gz`, `fetch.py`, workflow `storm-clustering-portugal-data.yml`). NOAA's Global Historical Climatology Network hourly (GHCNh, the successor of the Integrated Surface Database), every mainland Portuguese station with wind data. Reports whose wind-speed quality code marks them suspect or erroneous are dropped (`data/qc_codes.csv`). Because the number of reports a day rose from one or two in the 1950s to eight in the 1970s and 24–48 at airports after 2000, the daily maximum over all reports is not comparable over time; the analysis uses only the report closest to each of the eight synoptic hours (00, 03, …, 21 UTC, within 15 minutes) and only days with at least six of the eight. Most of IPMA's synoptic stations stop appearing in GHCNh after September 2022; Porto, Faro and Beja are the three stations with usable records from the 1970s to 2026, and they are the independent check on ERA5.

**Circulation indices** (`data/cpc_tele_index_nh.txt`, `data/cpc_nao_daily.txt`). NOAA Climate Prediction Center monthly teleconnection indices (NAO, East Atlantic, Scandinavian, East Atlantic/West Russia, Polar/Eurasia and others; rotated principal components of 500 hPa height, 1981–2010 climatology) and the daily NAO, saved verbatim.

**Documented storms** (`data/known_storms.csv`). Fourteen storms with the UTC day of their main impact on mainland Portugal and a source for each, used only to check the catalogue: Klaus, the Oeste storm of 23 December 2009, Xynthia, Gong, Emma, Leslie, Daniel, Elsa, Fabien, Ingrid, Joseph, Kristin, Leonardo and Marta.

## Method

1. **Severity index.** For each node, the 98th percentile of the daily maximum wind over 1981–2010; each day's index is 1000 times the mean over nodes of the cubed relative excess over that percentile (Klawa and Ulbrich 2003, with area as the weight).
2. **Episodes.** A storm day is an October–March day whose index exceeds the 96th percentile of the index over all winter days (about seven storm days a winter). Consecutive storm days form one episode; its severity is the summed index. Seasons are labelled by the October they start in.
3. **Dispersion and the circulation.** Variance-to-mean ratio of counts by winter and by blocks of 7 to 182 days; Poisson regression of monthly counts on the teleconnection indices, chosen by forward selection on AIC (NAO, East Atlantic, Scandinavian, Polar/Eurasia).
4. **Self-exciting model on the daily grid.** On each day on which a new episode can start, it starts with probability 1 − exp(−λ). λ is a background, set by the season and the indices, times an optional gamma-distributed multiplier per winter (the "stormy winter" frailty, integrated out by Gauss–Laguerre quadrature), plus an excitation term: each episode adds a kernel of total mass n (the branching ratio) and time scale τ, spread over the following days. Days inside an episode and the calm day after it are dead time: a new episode cannot start there by construction, and they are excluded from the likelihood. Without this, a kernel that peaks at lag zero predicts follow-on storms exactly where the catalogue cannot contain them, and the excitation is understated.
5. **Checks.** Likelihood-ratio tests; histories simulated from the model without excitation (indices and stormy winters) and refitted, to see whether winter-to-winter storminess can mimic self-excitation; pairs of episodes by separation, dispersion by block length and follow-on shares compared with simulated histories; the daily NAO averaged over the previous week added to the background.
6. **Rarity.** Each 2025/26 metric is ranked among the 76 winters and compared with 20,000 winters simulated from each model, with the indices of a random past winter or with those of 2025/26. Each simulated episode draws its length and severity jointly from the catalogue.
7. **Trend.** Poisson and Mann–Kendall tests on counts; the self-exciting model with a linear trend in the background and in the logit of the branching ratio; the two halves of the record; 31-winter moving windows; the same on the station record; and a power analysis that simulates records as long as this one with a known change in the branching ratio.
8. **Robustness.** Thresholds at the 94th and 98th percentiles, episodes merged across one or two calm days, northern and southern nodes separately, 1979 onwards, and the three-station record.

## Figures

- `figures/01_winter_2025_26.png`: daily severity index over the 2025/26 winter, ERA5 and stations, with the named storms.
- `figures/02_episodes_per_winter.png`: storm episodes per winter, 1950/51 to 2025/26, and the winter-mean NAO.
- `figures/03_pair_excess.png`: pairs of episodes by separation relative to the model without excitation.
- `figures/04_dispersion_by_block.png`: dispersion of counts by block length, observed and simulated.
- `figures/05_intensity_2025_26.png`: fitted daily chance of a new episode through 2025/26.
- `figures/06_return_periods.png`: return periods of the 2025/26 metrics under both models, with the empirical record.
- `figures/07_trend_and_power.png`: branching ratio in moving 31-winter windows, and the power of the trend test.

## Caveats

- ERA5 smooths extremes: its winds are lower than station gusts. The index uses each node's own percentile, so what matters is the ranking of days, which ERA5 gets broadly right (Spearman correlation with the station index and agreement on the strongest episodes are printed in the output). Before 1979 fewer observations constrain ERA5.
- On daily data two storms on consecutive days are one episode: Joseph and Kristin, 26–28 January 2026, are one. The catalogue sees clustering at time scales of two days and more.
- The teleconnection indices are monthly; the daily NAO check addresses part of the sub-monthly circulation, not all of it.
- The station records reflect changes of instruments and sites as well as weather; their downward trend in counts is not seen in ERA5.
