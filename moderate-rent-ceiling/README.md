# Where the EUR 2,300 "moderate rent" ceiling sat against real rents, Portugal 2017-2025

From 2026 Portugal taxes residential rental income at 10% instead of 25% when the monthly rent does not exceed a "moderate" ceiling of 2.5 times the minimum wage, EUR 2,300 in 2026 (Decreto-Lei 97/2026). This folder measures, municipality by municipality and year by year before the ceiling existed, how far EUR 2,300 was from the rents of new leases and from the incomes of the households that pay them. Nothing after December 2025 is used.

- **Floor area to reach the ceiling.** INE publishes rents of new leases per m² (1st quartile, median, 3rd quartile) by municipality. Dividing EUR 2,300 by each gives the floor area a lease at that price would need to reach the ceiling, compared with the floor area of rented homes in the Census 2021.
- **Modelled share of new leases above the ceiling.** Rent per m² is log-normal with the published quartiles; floor area follows the municipality's rented homes in the Census; rent per m² falls with floor area at an elasticity of −0.43 (s.e. 0.03), estimated from INE's medians by dwelling type in the 24 municipalities over 100,000 inhabitants. Sensitivity: elasticity ±1.96 s.e. and zero, floor areas ±10%, dispersion +20%. Checks: a log-normal tail test on the Census rent distribution, and the UTAO's estimate that 0.9% of leases in the 2024 tax base had rents above EUR 2,300.
- **Incomes.** Twelve months at EUR 2,300 against the median and the 80th percentile of gross declared income of fiscal households by municipality (INE, from tax data), and against the government's own reference household of about EUR 5,750 a month.
- **Over time.** The same rule (2.5 x each year's minimum wage) applied from 2017, with INE's 2021-methodology medians spliced onto the 2026 methodology at 2020, municipality by municipality.

## Run

```bash
python -m pip install -r requirements.txt
python ceiling.py
python figures.py
python -m pytest -q tests
```

`results/summary.txt` has every number; the CSVs behind it are in `results/`. Six figures in `figures/`, numbered as in the article:

- `fig01_area_needed.png` floor area needed to reach EUR 2,300, three municipalities over time and every municipality in 2025
- `fig02_by_type.png` monthly rent of a typical new lease by dwelling type, large municipalities, 2025
- `fig03_share_above.png` modelled share of new leases above EUR 2,300, with the sensitivity band and the UTAO figure
- `fig04_distribution.png` modelled national distribution of monthly rents of new leases, 2020 and 2025
- `fig05_rule_over_time.png` square metres the ceiling rule buys at the median rent per m², 2017-2025
- `fig06_income.png` median and 80th-percentile household income by municipality, 2024, against the ceiling

## Data

`fetch.py` downloads every series from the INE indicator API (no token) through `.github/workflows/moderate-rent-ceiling-data.yml`, which commits the snapshot; `data/fetch_log.txt` records each request with its SHA-256. Indicators: `0014711` (quartiles of rent per m² of new leases, Metodologia 2026, 2020-2025), `0013041` (the same, Metodologia 2021), `0009631`/`0009632` (medians and counts, Metodologia 2021, 2017-2023), `0014697` (counts, Metodologia 2026), `0014706`/`0014705` (medians and counts by dwelling type, quarterly, large municipalities), `0012712`, `0012714`, `0012673` (income of fiscal households), `0012522`, `0012526`, `0014625` (Census 2021). The raw responses of the Census cross-tab (about 12 MB) are not kept; its tidy table is.

Constants in the code: the minimum wage from 2017 to 2026 (EUR 557, 580, 600, 635, 665, 705, 760, 820, 870, 920); the ceiling rule (2.5 times the minimum wage); the government's reference household income of about EUR 5,750 a month, as reported when the ceiling was announced; a 35% housing-cost burden as a benchmark.

Limits: INE publishes rents per m², not per month, so monthly rents are modelled; floor areas come from the 2021 stock of rented homes, not from new leases; the elasticity is estimated in large municipalities and applied everywhere; 61 small municipalities have suppressed quartiles (the 247 with quartiles hold 99% of new leases); fiscal households are tax units, not households; the two INE methodologies differ by about 3-6% at the national median.

The workflow `.github/workflows/moderate-rent-ceiling.yml` reruns the analysis, checks `results/summary.txt` against the committed run (`check_summary.py`), runs the tests and draws the figures.

## License

Apache-2.0, as the rest of the repository.
