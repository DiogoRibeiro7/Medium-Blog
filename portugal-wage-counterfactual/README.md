# Portuguese wages against inflation plus productivity, 1975–2024

Takes the wage of a base year, raises it every year by consumer-price inflation plus real labour-productivity growth, and compares the result in 2024 with what wages actually reached:

```
W_t = W_{t-1} · (1 + π_t + g_t)          additive rule (headline)
W_t = W_{t-1} · (1 + π_t) · (1 + g_t)    multiplicative rule
```

The only wage series that reaches back to 1975 is national-accounts compensation per employee (D1 / employees, including employer contributions): AMECO for 1975–1995, chain-linked at 1995 to the Eurostat snapshot in `../portugal-labour-share/data` for 1995–2024. The counterfactual-to-actual ratio is applied to the 2024 Quadros de Pessoal mean (€1,582.7) and median (€1,186.55) monthly earnings. No Portuguese median wage exists for 1975, so the median result assumes the median/mean ratio of the base year equals that of 2024.

The script also runs every base year from 1975 to 2023, splits the multiplicative gap into a consumer-price/GDP-deflator wedge and the change in the adjusted wage share, and repeats the exercise on the Quadros de Pessoal mean alone from 1985 and the median alone from 2021.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python wage_counterfactual.py
python figures.py
python -m pytest -q tests
```

It writes:

- `results/summary.txt`: every number
- `results/base_year_sensitivity.csv`: counterfactual/actual for each base year, three variants
- `results/path_from_1975.csv`: actual and counterfactual monthly compensation, and the adjusted wage share
- `results/decomposition_by_base_year.csv`: the multiplicative gap split into the CPI/GDP-deflator wedge and the change in the adjusted wage share, in log points
- `results/inputs_linked.csv`: the linked annual inputs
- `figures/fig1_base_year_curve.png`, `figures/fig2_decomposition.png`, `figures/fig3_path_from_1975.png`

The workflow in `.github/workflows/portugal-wage-counterfactual.yml` reruns the analysis on every change and checks that `summary.txt` is unchanged.

## Data

See `data/SOURCES.md`. AMECO is the spring 2018 vintage shipped in the CRAN package `ameco` 0.2.9; `extract_ameco.py` rebuilds `data/ameco_prt.csv` from it. 2025 is provisional in the Eurostat series and is not used.

## Caveats

- Compensation per employee includes employer social contributions. Wages and salaries (D11) are 80.3% of compensation in 1995 and 77.8% in 2024; no D11 series reaches 1975.
- CPI inflation for 1976 and 1977 is 21.1% and 31.0% in the World Bank series and 19.9% and 26.7% in AMECO's national CPI. With AMECO's CPI up to 1995 the 1975 counterfactual is about 4% lower (`summary.txt`).
- AMECO hours before 1995 show hours per person employed rising, so productivity per person is the headline measure and productivity per hour a variant.

## License

Apache-2.0, as the rest of the repository.
