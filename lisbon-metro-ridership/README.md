# The Lisbon metro's falling ridership, 2025–2026: calendar, closures, fleet and the rest of the network in the official data

Companion code for a Medium article. A post circulating in November 2026 said that the Metropolitano de Lisboa had lost passengers for two years running, against the trend of European capitals and of Porto and Coimbra, and that the riders had moved to cars. The article has no thesis of its own: it takes each part of that claim as a question and follows the official data. The code reads INE's monthly metro survey from January 2004 to August 2026 for the three Portuguese systems, adjusts Lisbon's change for the calendar and for the days the network was shut, compares it with Porto and with Metro Sul do Tejo, and sets it against what the metro ran, who rode, the other modes of the Lisbon metro area, residents, jobs and motorway traffic.

## What it finds

All numbers are from `output.txt`.

- **The fall.** INE counts 176.648 million passengers in 2024 and 173.451 million in 2025 (−1.81%), and 109.076 million in January–August 2026 against 114.245 million a year earlier (−4.52%). By quarter, 2025 went +1.7, −2.1, −1.3, −5.2%; 2026 −4.0, −4.4 and −5.6% (July–August).
- **The long view.** Lisbon peaked at 185.4 million in 2005, lost 24.1% between 2011 and 2013 and regained 35.6% between 2014 and 2019. In 2025 it was 5.3% below 2019 with 7.8% more place-km; Porto was 33.3% above 2019 (place-km +29.3%) and Metro Sul do Tejo 28.8% above (place-km −0.9%). Passengers per 1,000 place-km: Lisbon 51.8 in 2019, 45.5 in 2025; Porto 41.7 and 43.0; Metro Sul do Tejo 49.5 and 64.3.
- **Calendar and closures.** Day weights estimated from 84 normal months: a Saturday carries 0.60 of a weekday and a Sunday or holiday 0.42. Per day the network was open, 2025 is −0.79% (calendar −0.50 points, closures −0.52 points); January–August 2026 is −4.43% (calendar +0.56, closures −0.65). The 2025 remainder lies between −0.72% and −0.85% and the 2026 one between −4.30% and −4.57% under every alternative tried; without the months with works, 2026 is −3.91%.
- **Three systems.** Per day open, 2025: Lisbon −0.8%, Porto +6.5%, Metro Sul do Tejo +0.1%; January–August 2026: −4.4%, +2.4%, −0.4%. Lisbon's quarterly gap with Porto averages −6.7 points in 2024, −7.4 in 2025 and −6.9 in 2026; with Metro Sul do Tejo −6.0, −1.0 and −4.0. All three fell in Q4 2025.
- **Supply.** In 2025 train-km fell 1.85% and circulations 3.20% while place-km rose 1.12% (six-car trains all day: 746 → 768 places per train-km). Passengers per train-km: 34.97 in 2024, 34.98 in 2025. Regularity index 97.95 → 95.95. Fleet availability in the concession contract: 85.54, 90.75, 83.32, 81.05 in the quarters of 2025, 77.99 and 78.59 in the first two quarters of 2026 (reference 85.89); trains within the scheduled interval below the 80% reference from Q2 2025. Over the six quarters, fleet availability and Lisbon's change correlate at 0.72 (descriptive; six points).
- **Riders.** Occasional tickets 31.6 → 28.5 million (−9.8%); passes paid by the customer 105.2 → about 96 million; everything else, mostly free passes, 35.1 → 43.8–44.7 million. Tariff revenue −5.9%. Lisbon airport +2.9% in 2025, +0.8% in January–August 2026, −1.1% in June–August 2026.
- **The rest of the network.** Regular bus passengers in the metro area (INE) +8.9% in 2025 (+30.1 million): Grande Lisboa +5.6%, Península de Setúbal +24.4%. Tagus ferries −1.9%, CP Lisbon suburban trains −2.1% (press), Metro Sul do Tejo −0.7%.
- **People, jobs and roads.** Lisbon municipality's residents −1.04% in 2025; the four municipalities with metro stations −0.56%, the other 14 of the metro area +0.65%. Employment +3.4% in Grande Lisboa. Average daily traffic on the A1, A2, A5 and A12 in April–June 2026: +0.67% on a year earlier, while the metro lost about 21,000 trips a day.

## Data

Two new external sources and one shared:

- `ine-portugal-metro-ridership` (contract [`../data_sources/ine-portugal-metro-ridership.json`](../data_sources/ine-portugal-metro-ridership.json)): seven INE responses — metro passengers (`0000898`), passenger-km (`0000899`) and place-km (`0000900`), January 2004 to August 2026; Lisbon airport passengers embarked (`0000861`) and disembarked (`0000862`), January 2015 to August 2026; regular bus passengers by NUTS II region (`0013815`), 2023–2025; river-line passengers (`0000896`). Each query names its periods. INE's last-update date moves every month for the monthly series, so these pins are the SHA-256 of the indicator code and the data block.
- `eurostat-regional-employment-pt` (contract [`../data_sources/eurostat-regional-employment-pt.json`](../data_sources/eurostat-regional-employment-pt.json)): employed persons 15–74 by NUTS 2 region (`lfst_r_lfe2emp`), pinned by SHA-256.
- `ine-portugal-municipal-infrastructure`: the population responses `pop_est_2024` and `pop_est_2025`, checked against that contract.

Figures no API publishes are in [`transcribed.py`](transcribed.py), each with its document: the metro's reports to the regulator (AMT) for 2019, 2024 and 2025; the concession-contract indicators for 2025 (by quarter) and January–June 2026 (by month); the company's counts by ticket type (press); the other operators' totals; the calendar of closures; motorway traffic from IMT's report for Q2 2026.

All responses were retrieved on 26 November 2026. No data are committed: `data/` and `results/` are rebuilt on every run and ignored by git. Both are recorded in the data registry (`DiogoRibeiro7/data`, `external/ine-portugal-metro-ridership` and `external/eurostat-regional-employment-pt`), and the shared population record lists this folder as a consumer.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v   # the tools against answers known by construction
python fetch.py                           # 7 INE, 2 shared INE and 1 Eurostat responses
python panel.py                           # data/monthly.csv, bus.csv, river.csv, population.csv, employment.csv
python run_analysis.py > output.txt       # every number in the article; also writes results/
python figures.py                         # figures/
```

The workflow `.github/workflows/lisbon-metro-ridership.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **Day weights.** For each system, log monthly passengers = year effect + calendar-month effect + log(*D* + *s*·*S* + *u*·*H*), with *D* weekdays, *S* Saturdays and *H* Sundays and public holidays (national holidays, the 2013–2015 suspensions, and the municipal holiday of Lisboa, Porto or Almada). *s* and *u* are found by grid search on 2015–2019 and 2023–2024. The month effects take out seasonality; the weights are identified by the changing mix of days within the same calendar month across years.
- **Closures.** A full-day closure removes one weekday; a strike that shut the network until 10:30 removes 0.35 of one (tested at 0.25 and 0.45); the blackout of 28 April 2025 removes 0.6 (tested at 0.4 and 0.8). Porto and Metro Sul do Tejo lose 0.6 of a day on the blackout and the two general strikes. Works that closed parts of the Lisbon network (August 2024, May 2026, August 2026) are not adjusted; the output repeats the 2026 result without those months.
- **Passengers.** INE's count for Lisbon is the company's total, valid tickets plus an estimate of fare evasion (4.8 million in 2024). Bus passengers are counted by INE by the region where the trip starts, CP's suburban figure is validations, and the operators' press figures use their own definitions; the six changes in Figure 4 are indicative, not a sum of like with like.
- **Contract indicators** (fleet availability, circulations run, trains within the scheduled interval) are defined by the concession contract and published from 2025, so there is no 2024 baseline. The AMT regularity index is a different indicator, available for 2024 and 2025.
- **Correlations** over six or eleven quarters are descriptions of co-movement, not estimates of effects.

## Files

- **`sources.py`**: the ten queries.
- **`fetch.py`**: download and verify; `--record` writes the two contracts owned by this folder.
- **`transcribed.py`**: figures read from reports and press, with their sources.
- **`panel.py`**: the monthly panel and the annual tables, with consistency checks.
- **`core.py`**: Easter and the Portuguese holiday calendar, day-type counts, day weights, closure adjustment, period changes.
- **`run_analysis.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the four figures.
- **`tests/test_core.py`**: 9 checks.

## Licence

Code: Apache-2.0. Data: INE, Statistics Portugal, reused under CC BY 4.0. *Fonte: Instituto Nacional de Estatística, I.P. – Portugal*; Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU). Transcribed figures belong to their publishers (Metropolitano de Lisboa, AMT, IMT, CRTM and the newspapers named in `transcribed.py`).
