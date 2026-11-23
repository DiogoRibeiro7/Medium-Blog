# Saturated rings and the polycentric alternative: population growth against housing, schools, transport and hospitals in Portugal's 308 municipalities, 2011–2025

Companion code for a Medium article. The author's thesis is that Lisbon and Porto are saturated, short of housing and of the infrastructure that serves their residents, and that the answer is to build that infrastructure (transport, streets, sanitation, schools, hospitals) in a polycentric network of medium cities and metro peripheries. The code measures where people went, whether the infrastructure followed, where the pressure is now, what predicted growth after 2021, and which places could take growth and what each of them lacks.

## What it finds

All numbers are from `output.txt`.

- **Where people went.**
  - The revised INE estimates put 11,424,031 residents in Portugal at the end of 2025, 7.78% more than at the end of 2021. Births minus deaths were −141,095 in 2022–2025, so net migration was about 966,000.
  - The Lisbon metro area held 28.5% of residents in 2021 and took 40.9% of the growth. Its ring alone took 34.5%. The two metro areas together took 52.8%.
  - Growth rises with size. Mean growth by size decile ran from −1.51% a year (smallest tenth) to 0.01% in 2011–2021, and from 0.23% to 2.04% in 2021–2025. The slope on log size is 0.41 and 0.56 points a year. Gibrat's law fails; the urban system is concentrating.
  - The rank-size (Zipf) exponent of the 50 largest municipalities moved from 1.733 to 1.712 (standard error about 0.34). Municipalities are administrative units, so these are not city-size exponents.
  - The revised estimate for 31 December 2021 is 256,051 residents above the 2021 census. Lisboa alone holds 23.3% of that gap (its estimate is 10.9% above its census count) and the two metro areas 67.3%.
- **Whether the infrastructure followed (2021 census unless stated).**
  - Housing: in the Lisbon ring households grew 5.47% in 2011–2021 and dwellings 1.58%. Only 4.10% of its dwellings were vacant for sale or rent, and 15.31% of households were crowded. A 90 m² dwelling cost 13.8 years of median household income in 2024, against 8.6 in the medium cities.
  - Schools: the Lisbon ring enrols 89.2 children in pre-school per 100 resident children aged 3–5 (2024/25), up from 75.9 in 2011/12. 12 of its 17 municipalities are below 100.
  - Transport: the mean public-transport commute is 46.0 minutes in the Lisbon ring, 31.0 in the medium cities. 28.3% of ring residents commute out of their municipality.
  - Hospitals: 2.03 public (SNS and PPP) beds per 1,000 residents in the Lisbon metro area taken whole, 2.48 in Porto's, 3.58 in the medium cities.
- **A saturation index** (housing, schools, transport, hospitals; equal weights; 278 mainland municipalities).
  - The Lisbon ring scores 1.03, the Porto ring 0.42, the medium cities 0.16, Lisboa −0.25 and Porto −0.43. The cores are saturated in housing but hold the region's schools and hospitals.
  - Moita, Seixal, Alenquer, Montijo, Sesimbra, Palmela and Ílhavo are in the top 30 under at least 90% of 5,000 random weightings. Most other ranks move a lot with the weights.
  - Pressure has crossed the metro boundary. Alenquer, Arruda dos Vinhos and Sobral de Monte Agraço, outside the official metro area, rank 3, 8 and 10.
- **What predicted growth after 2021.**
  - The most saturated fifth of mainland municipalities took 51.6% of mainland growth in 2021–2025; the least saturated fifth took 4.5%. The Spearman correlation between saturation and later growth is 0.52.
  - In a regression, housing pressure predicts growth until the 2021 sale price enters. Price then carries it: 0.71 points a year per standard deviation, with an out-of-sample R² of 0.35 when each NUTS III region is predicted from the others.
  - Spare capacity alone (vacant dwellings) has a weak negative correlation with growth (−0.14).
- **Candidate nodes.**
  - The binding constraint is transport in 21 of the 33 ring municipalities. In the medium cities it is housing in 13 and hospitals in 11 of 25.
  - The medium cities had 66,641 dwellings vacant for sale or rent in 2021, 0.38 for each household the two metro areas added in 2021–2025.

## Data

46 responses of the public INE API, retrieved on 23 November 2026 (indicators and filters in `sources.py`). They are an external source in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `external/ine-portugal-municipal-infrastructure`. The registry commit, URLs and hashes are in [`../data_sources/ine-portugal-municipal-infrastructure.json`](../data_sources/ine-portugal-municipal-infrastructure.json).

INE stamps every response with its extraction time, so two downloads of the same data differ byte for byte. The pin is therefore a content hash: the SHA-256 of the indicator code, INE's last-update date and the data block. `fetch.py` downloads the responses into `data/raw/` and stops if any content hash differs; `--accept-revision` lets a run continue on a new vintage. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v   # the tools against answers known by construction
python fetch.py                           # the 46 INE responses
python panel.py                           # data/panel.csv, one row per municipality
python run_analysis.py > output.txt       # every number in the article; also writes results/
python figures.py                         # figures/
```

The workflow `.github/workflows/portugal-infrastructure-saturation.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Groups.** Lisboa and Porto are the two core municipalities. The rings are the rest of each metro area: Grande Lisboa and Península de Setúbal (NUTS 2024) and the Área Metropolitana do Porto. Medium cities are the 25 mainland municipalities outside both metro areas with at least 50,000 residents in the 2021 census.
- **Two population sources.** The 2011–2021 change uses the census of 19 April 2021 against INE's estimate for 31 December 2011, over 9.3 years. The 2021–2025 change uses INE's revised annual estimates at both ends. Census counts and revised estimates are never mixed in one ratio.
- **Indicators of pressure.** Each is oriented so that higher means more pressure:
  - Housing: share of crowded main residences; minus the share of dwellings vacant for sale or rent; log of the 2024 price-to-income ratio. That ratio is the median sale price per m² times 90, over the median declared income net of IRS of a tax household.
  - Schools: minus the 2024/25 gross pre-school and basic-education enrolment rates.
  - Transport: mean public-transport commute and the out-commuting share (2021 census).
  - Hospitals: minus SNS and PPP beds per 1,000 residents in the NUTS III region.
  - Each indicator is standardised after clipping at the 2.5th and 97.5th percentiles. A dimension is the mean of its indicators, and the index is the mean of the four dimensions.
- **Weights.** Ranks are recomputed under 5,000 draws of Dirichlet(1,1,1,1) weights on the four dimensions. The tables give the 5th–95th percentile rank and the share of draws in the top 30.
- **Hospital beds.** INE counts beds where it locates each hospital entity. A hospital centre with several sites can appear under one municipality: Torres Vedras and Peniche have hospitals but no SNS beds recorded, while Caldas da Rainha, head of the Oeste centre, has 303. That is why the index uses the NUTS III region and the group comparison takes each metro area whole. Alto Tâmega e Barroso records no SNS beds at all, apparently because the Chaves hospital is recorded with its centre in Vila Real.
- **Gross enrolment rates.** A rate is pupils enrolled in the municipality per 100 resident children of the age. Below 100, the municipality enrols fewer children than live there: they study elsewhere or, for pre-school, are not enrolled.
- **Growth model.** OLS with HC1 standard errors on mainland municipalities with prices and incomes (267). Out-of-sample R² leaves out one NUTS III region at a time. The models are predictive. Prices are themselves shaped by expected demand, so no coefficient is a causal effect.

## Files

- **`sources.py`**: the 46 queries.
- **`fetch.py`**: download and verify; `--record` writes the contract.
- **`panel.py`**: the 308-municipality panel and its consistency checks.
- **`core.py`**: Gabaix-Ibragimov exponent, OLS with HC1 errors, standardisation, Dirichlet ranks, leave-one-region-out R², weighted median.
- **`run_analysis.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the five figures.
- **`tests/test_core.py`**: 10 checks.

## Licence

Code: Apache-2.0. Data: INE, Statistics Portugal, reused under CC BY 4.0. *Fonte: Instituto Nacional de Estatística, I.P. – Portugal*.
