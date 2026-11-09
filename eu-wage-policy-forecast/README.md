# Forward scenarios for wage-setting rules: Portugal's minimum wage to 2029, pay transparency, and the EU's 60% reference

Companion code for a Medium article that projects, with uncertainty bands, what the labour-law changes now on the table would do to each part of the wage distribution, to employment and to job mobility. It is decision support: four paths for Portugal's minimum wage in 2027–2029, the Portuguese transposition of the pay transparency directive, and each EU country's distance from 60% of the median wage. No path is recommended.

It builds on the backward-looking companion in `../eu-wage-bargaining-power`, whose auction-and-bargaining model it extends with an equal-pay rule that can apply to part of the firms.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python empirics.py       # estimates and their bootstrap draws (about 20 s)
python experiments.py    # model floor grid, calibration and coverage scenarios (about 5 min on two cores)
python forecast.py       # Monte Carlo scenarios (a second)
python figures.py
python -m pytest -q tests
```

`python fetch.py --out data` refreshes the data snapshot; the workflow `.github/workflows/eu-wage-policy-forecast-data.yml` does that on GitHub and commits the result, because the Eurostat and OECD APIs are not reachable from every machine. `.github/workflows/eu-wage-policy-forecast.yml` reruns everything on each change and checks that the three summaries are unchanged.

## What it does

**Empirics (`empirics.py`).** From the Structure of Earnings Survey (SES) waves 2006–2022 (full-time employees, October monthly earnings, 21 countries with a statutory minimum wage):

- the elasticity of P10/median with respect to the Kaitz index (minimum wage/median), with country and wave fixed effects, linear and quadratic in log Kaitz, and Portugal's own slope over its five waves;
- the elasticity of the low-pay share;
- the spillover of the minimum wage to the median: growth of the SES median between waves against growth of national-accounts compensation per employee and of the minimum wage (theta), plus Portugal's own drift;
- two backtests of that median model: the SES Kaitz index of 2022 forecast from 2018, and the Banco de Portugal base-wage Kaitz index of 2025 forecast from 2019;
- an AR(1) for Portuguese compensation per employee (1996–2025) as the noise around published wage projections;
- the share of employees in enterprises with at least 250, 150 and 50 persons employed (Eurostat business statistics by size class, 2023), which is who the transparency reporting duties reach;
- each EU country's Kaitz index against the 60% reference (OECD, median, full-time, 2025).

Uncertainty is a bootstrap over countries (2,000 draws) or, for Portugal's own estimates, a t distribution around the point estimate.

**Model (`model.py`, `experiments.py`).** Workers differ in ability, firms in productivity; wages are set by sequential auctions with bargaining (Postel-Vinay and Robin 2002; Cahuc, Postel-Vinay and Robin 2006), with a minimum wage that ends matches producing less than it. Covered firms are bound by an equal-pay rule: a raise to one worker in a firm × ability-band cell must, with probability *visibility*, be extended to every peer paid less, and every bid accounts for that expected cost. Coverage goes to the largest firms until a target share of employment is reached. The engine is compiled with numba.

- Floor grid: steady states for minimum wages 0.40–1.10, eight seeds.
- Calibration: at Portugal's floor (m = 0.65, 22% at the minimum, Kaitz 0.64), visibility is set so that covering every firm lowers the mean wage by 1%, 2% or 3% three to five years later (2% is the US estimate of Cullen and Pakzad-Hurson 2023).
- Coverage scenarios: the rule switched on for 34.8% (firms with 150+, from 2027), 49.8% (50+, from 2031) and 100% of employees, eight years, ten seeds; sensitivity to visibility and to a low floor.

**Forecast (`forecast.py`).** 4,000 draws combine wage-growth projections with AR(1) noise, theta and the drift, the three elasticities, and a randomly drawn model seed. The paths share the draws, so path differences are measured with common random numbers.

## Scenario inputs (`inputs.py`)

Every number that does not come from the snapshot is in `inputs.py` with its source:

- Minimum wage (euro a month, 14 payments): 920 in 2026; 970 (2027) and 1,020 (2028) in the 2025–2028 tripartite agreement and the 2027 budget report; 1,100 in 2029 as the government programme's target; UGT's 1,000 and CGTP's 1,100 for 2027.
- Compensation per employee projections: Ministry of Finance (OE2027, Oct 2026), Banco de Portugal (June 2026), Conselho das Finanças Públicas (Sep 2026).
- Banco de Portugal base-wage Kaitz index: 87% (2019), 91% (2025).
- Own-wage elasticities of employment: Dube and Zipperer (2024).
- Pay transparency: the Portuguese draft of 5 Aug 2026 (reporting from 7 June 2027 for 150+, from 7 June 2031 for 50–149); published effects from Denmark (Bennedsen et al. 2022), the UK (Blundell et al. 2025) and the US (Cullen and Pakzad-Hurson 2023); the Portuguese gender gap of 2023 (15.4% total, 12.5% base pay).

## Outputs

- `results/empirics_summary.txt`, `results/model_summary.txt`, `results/forecast_summary.txt`: every number used in the article
- `results/elasticity_draws.csv`, `results/spillover_draws.csv`: bootstrap draws
- `results/forecast_kaitz_paths.csv`, `forecast_risk_curve.csv`, `forecast_bottom.csv`, `forecast_model_outcomes.csv`, `forecast_employment.csv`, `forecast_transparency_evidence.csv`, `forecast_eu.csv`
- `results/model_floor_grid*.csv`, `model_calibration.csv`, `model_reform*.csv`, `coverage.csv`, `eu_reference.csv`, `ses_panel*.csv`
- `figures/fig1_bite_paths.png` … `fig5_eu_reference.png`

## Definitions and caveats

- Three Kaitz indices are used and they are not interchangeable: SES October monthly earnings of full-time employees (Portugal 0.64 in 2022), the Banco de Portugal base wage in the private sector (0.91 in 2025), and the OECD annual measure (0.59 in 2025). Forecasts are made on the first two.
- The median model is linear. As the minimum approaches the median base wage, more of the distribution is lifted directly and the median itself becomes the minimum; the projections near 1 show that the boundary is reached, not what happens beyond it.
- Portugal's SES Kaitz index already sits at the top of the EU sample (0.65); every path but the indexed one goes beyond it, where the EU elasticities are extrapolations.
- The model's only employment margin is match destruction below the floor: no labour-supply response, no monopsony gain, no price pass-through. Its employment loss is an upper bound in its own logic; published elasticities give the other end.
- The equal-pay rule in the model is a commitment channel calibrated to published average-wage effects. The model has no gender; gender-gap forecasts transfer published effects to the covered share of employees.
- The coverage shares refer to the business economy (B–S excluding public administration and households); employees in enterprises with fewer than 10 persons are the total minus the larger classes.

Contact: dfr@esmad.ipp.pt. Licence: Apache-2.0.
