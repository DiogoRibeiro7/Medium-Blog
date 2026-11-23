# Users and residents of Lisbon and Porto: commuters, employees, hospital and tourist beds against dwellings, with a centre–ring comparison across 248 European urban areas

Companion code for a Medium article. The author's thesis is that the centres of Lisbon and Porto attract users (commuters, workers, patients, visitors) and lose residents, and that these are one mechanism seen from two sides. The code counts both sides for the 35 municipalities of the two metro areas, measures what was added for each side since 2021, and places the two centres among the European functional urban areas (FUAs) in Eurostat's city statistics, asking what goes with a centre keeping its residents.

## What it finds

All numbers are from `output.txt`.

- **What counts as the centre.** Lisboa municipality held 19.0% of its metro area's residents in the 2021 census; Eurostat's "Lisboa (greater city)" held 62.2% of its FUA. Porto: 13.4% and 75.1%. The municipal comparisons use the municipalities; the European comparison has to use Eurostat's units.
- **Users.**
  - People entering to work or study were 74.98% of Lisboa's residents in the 2021 census and people leaving 10.00%: 1.650 users per resident, a daytime population of about 900,000 for 545,796 residents. Porto: 1.601. 29 of the 33 ring municipalities are below 1.
  - Full-time employees by place of work (Quadros de Pessoal) grew 22.9% in Lisboa and 26.4% in Porto from 2021 to 2024. Lisboa took 52.5% of the employees added in its metro area and Porto 48.4%, with 20.1% and 13.8% of the residents.
  - Lisboa holds 53.3% of its metro area's public hospital beds and 71.5% of its tourist beds; Porto 58.2% and 61.6%.
- **Residents and homes.**
  - Lisboa's share of its metro area's residents went from 19.2% to 19.0% between the end of 2011 and the 2021 census, and from 20.1% to 19.6% between the end of 2021 and the end of 2025 in INE's revised series. Its residents fell 1.04% in 2025 while its ring grew 0.48%. Porto's share was flat to 2021 and rose to 14.7% by 2025.
  - Dwellings fell 1.89% in Lisboa and 2.92% in Porto between the 2011 and 2021 censuses; they rose 1.58% and 2.04% in the rings.
  - Lisboa completed 1,294 dwellings in new residential construction in 2022–2025: 0.53 a year per 1,000 residents, 6.5% of its metro area's completions, 1.50 for every 100 full-time employees added in 2021–2024. Porto completed 4,834: 4.95 per 1,000 residents a year, 23.3% of its metro area's completions, 13.81 per 100 employees added.
  - Lisboa added 3,814 tourist beds between 2023 and 2025 and completed 759 dwellings in 2023–2025. Porto added 3,997 beds and completed 3,908 dwellings.
- **Europe (248 FUAs in 13 countries, about 2015 to 2021).**
  - In the median FUA the centre's residents grew 0.055% a year and the ring's 0.388%; 76.2% of centres lost share of their FUA's residents. Only 31.5% lost residents' share while gaining jobs' share.
  - Within countries, the centre-minus-ring gap in residents' growth follows the gap in dwellings' growth (Spearman 0.540) much more than the gap in jobs' growth (0.245). With country fixed effects, residents' gap = 0.699 × dwellings' gap (s.e. 0.063) + 0.049 × jobs' gap (0.020), within R² 0.409. Leaving out one country at a time, the dwellings coefficient stays between 0.649 and 0.797.
  - Centres that gained jobs relative to their ring also tended to gain dwellings (0.084, s.e. 0.026) and residents (0.107, s.e. 0.029). Attracting jobs does not, across Europe, go with losing residents.
  - Lisbon's greater city: jobs +3.37% a year, dwellings +0.15% (9.7th percentile); jobs minus dwellings 3.22 points, 96.0th percentile. Porto: 2.88 points, 94.0th. All ten Portuguese centres had jobs growing faster than dwellings, with a median gap of 2.99 points against −0.07 elsewhere.

## Data

Two external sources in the data registry [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), plus one shared with `portugal-infrastructure-saturation`:

- `external/eurostat-urban-audit-cities-fuas`: six Eurostat responses (`urb_cpop1`, `urb_lpop1`, `urb_clma`, `urb_llma`, `urb_clivcon`, `urb_llivcon`). Eurostat serves identical bytes for identical data, so the pin is the SHA-256 of each response. Contract: [`../data_sources/eurostat-urban-audit-cities-fuas.json`](../data_sources/eurostat-urban-audit-cities-fuas.json).
- `external/ine-portugal-centre-users-homes`: nine INE responses (`0011654` in-commuting, `0012650` employees, `0013366` tourist beds, `0013249` dwellings completed). Contract: [`../data_sources/ine-portugal-centre-users-homes.json`](../data_sources/ine-portugal-centre-users-homes.json).
- `external/ine-portugal-municipal-infrastructure`: 18 of its INE responses (population, census housing and commuting, prices, rents, enrolment, hospital beds), checked against [`../data_sources/ine-portugal-municipal-infrastructure.json`](../data_sources/ine-portugal-municipal-infrastructure.json).

INE stamps every response with its extraction time, so INE pins are content hashes (SHA-256 of the indicator code, INE's last-update date and the data block). `fetch.py` stops if any hash differs; `--accept-revision` lets a run continue on a new vintage. All 33 responses were retrieved on 23 November 2026. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v   # the tools against answers known by construction
python fetch.py                           # 6 Eurostat and 27 INE responses
python panel.py                           # data/europe.csv and data/metro.csv
python run_analysis.py > output.txt       # every number in the article; also writes results/
python figures.py                         # figures/
```

The workflow `.github/workflows/centres-users-residents.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Users per resident** is 1 + (people entering − people leaving) / 100, both as % of residents, from the 2021 census commuting of people who work or study. It is a daytime count; it leaves out visitors and patients.
- **Two population sources.** Census counts and INE's revised estimates (June 2026) are never mixed in one ratio. 2011–2021 changes use the census against INE's estimate for the end of 2011; changes from 2021 use the revised series at both ends. The revised end-2021 estimate is 10.9% above the census count in Lisboa and 3.7% in its ring.
- **Employees.** Quadros de Pessoal counts full-time employees on full pay, by the location of the establishment, in October; public administration and the self-employed are outside it. Four size bands are withheld in 2021 (Arouca, Espinho); none in 2024.
- **Dwellings completed** are new construction for family housing. Rehabilitation, extensions and conversions are not included, and 2024–2025 are INE estimates.
- **Tourist beds** are the capacity of hotels, local accommodation above INE's survey threshold, and rural tourism. Small short-term rentals are not counted.
- **Centre and ring in Europe.** The centre is Eurostat's city (the "greater city" where one exists), the ring is the FUA minus the city. Each FUA uses the year nearest 2015 within 2014–2016 and nearest 2021 within 2020–2022 in which all six series exist. Growth is 100 × log change / years.
- **Boundary changes.** FUAs are dropped when the city is within 2% of the FUA in any series (27), or when a series moves implausibly fast: people more than 3.5% a year, jobs more than 8%, dwellings outside −0.5% to 3.5% (81, of which 65 German, where many rings lose 1–3% of their dwellings a year in the series). Robustness: without Germany, with all series within 2% a year (jobs 5%), leaving out each country.
- **Accounting.** Residents = dwellings × residents per dwelling, so the residents' gap is the dwellings' gap plus the gap in residents per dwelling. Within countries, the dwellings' gap carries 53.2% of the variance of the residents' gap. The regressions describe how the three gaps move together; they do not identify what causes what.

## Files

- **`sources.py`**: the 33 queries.
- **`fetch.py`**: download and verify; `--record` writes the two contracts owned by this folder.
- **`panel.py`**: the European centre–ring panel and the 35-municipality metro panel, with consistency checks.
- **`core.py`**: log growth, rings, within-group demeaning, OLS with HC1 errors, fixed effects, Spearman, percentiles.
- **`run_analysis.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the three figures.
- **`tests/test_core.py`**: 11 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU); INE, Statistics Portugal, reused under CC BY 4.0. *Fonte: Instituto Nacional de Estatística, I.P. – Portugal*; employees: MTSSS/GEP, Quadros de Pessoal.
