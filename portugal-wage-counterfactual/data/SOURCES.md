# Data sources

| File | Content | Source | Retrieved |
|---|---|---|---|
| `ameco_prt.csv` | Compensation per employee, employees, employment, hours, real GDP, real GDP per person employed, GDP deflator, national CPI, adjusted wage share; Portugal 1960–2018 | European Commission AMECO, spring 2018 vintage, as shipped in the CRAN package `ameco` 0.2.9 (https://github.com/cran/ameco); extracted with `../extract_ameco.py` | 2026-10-23 |
| `cpi_inflation.csv` | Consumer price inflation, annual average, % | World Bank `FP.CPI.TOTL.ZG` via FRED `FPCPITOTLZGPRT` (https://fred.stlouisfed.org/data/FPCPITOTLZGPRT.txt) | 2026-10-23 |
| `eurostat_productivity_d11.csv` | Real labour productivity per person and per hour worked (index 2015 = 100); wages and salaries D11, million euro | Eurostat `nama_10_lp_ulc` (RLPR_PER, RLPR_HW, unit I15) and `nama_10_gdp` (D11, CP_MEUR); 2025 provisional and not used | 2026-10-23 |
| `qp_earnings.csv` | Mean monthly earnings (ganho), full-time employees with complete pay, October; median for 2021 and 2024 | Mean: GEP/MTSSS Quadros de Pessoal via PORDATA (https://www.pordata.pt/en/portugal/average+monthly+earnings+of+employees+total+and+by+sex-375, updated 2026-01-05). Median: INE indicator 0013011, 2nd quartile, Portugal (https://www.ine.pt/xurl/indx/0013011/PT) | 2026-10-23 |
| `minimum_wage.csv` | National minimum wage (salário mínimo nacional, general), monthly, euros (escudo values converted at 200.482) | DGERT/MTSSS via PORDATA (https://www.pordata.pt/pt/estatisticas/salarios-e-pensoes/salarios/salario-minimo-nacional, updated 2026-01-05); one value per year as published; kept for comparison only, since it mixes end-of-year and start-of-year rates | 2026-10-23 |
| `minimum_wage_changes.csv` | Every change in the general national minimum wage since 27 May 1974: effective date, value (escudos or euros), legal instrument; the 2000 value (63,800 escudos) is missing from DGERT's table and taken from PORDATA | DGERT, Evolução da RMMG (https://www.dgert.gov.pt/evolucao-da-remuneracao-minima-mensal-garantida-rmmg) | 2026-10-23 |


Compensation of employees, GDP, employment and employees for 1995–2024 are read from the Eurostat snapshot in `../../portugal-labour-share/data/` (downloaded 2026-10-21).
