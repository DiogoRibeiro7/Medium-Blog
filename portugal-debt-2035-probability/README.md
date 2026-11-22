# Portugal's 2035 debt target as a probability

Reproduction code for the Medium article on whether Portugal's public debt can reach 60% of GDP by 2035, and how much one recession changes the answer.

The question is turned from a date into a probability: given the 2027 budget as a starting point and Portugal's own 1998–2025 history of shocks, how often does debt end up at or below 60% of GDP in 2035?

## Run

```bash
pip install -r requirements.txt
python fetch_data.py      # downloads 4 Eurostat API responses into raw/ and checks SHA-256
python build_data.py      # rebuilds data/panel.csv
python run_analysis.py    # results/results.json, results/summary.txt, figures/*.png (about 45 seconds)
python -m unittest discover -s tests -v
```

CI (`.github/workflows/portugal-debt-2035-probability.yml`) lints, fetches and checks the digests, runs the tests, reruns the analysis and fails if `results/summary.txt` changes.

`raw/` and `data/` are rebuilt at every run and are not committed.

## Data

Registered in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data) as the external source `eurostat-portugal-government-finance`. The pinned contract is `data_sources/eurostat-portugal-government-finance.json` at the repository root: query URL, bytes, SHA-256 and Eurostat's own `updated` stamp for each response, retrieved on 22 November 2026.

| File | Eurostat dataset | Content |
| --- | --- | --- |
| `raw/edp_pc.json` | `gov_10dd_edpt1` ([10.2908/GOV_10DD_EDPT1](https://doi.org/10.2908/GOV_10DD_EDPT1)) | Debt, net lending and interest, % of GDP |
| `raw/edp_mio.json` | `gov_10dd_edpt1` | The same, EUR million |
| `raw/gdp.json` | `nama_10_gdp` ([10.2908/NAMA_10_GDP](https://doi.org/10.2908/NAMA_10_GDP)) | Real growth, deflator growth, nominal GDP |
| `raw/yld.json` | `irt_lt_mcby_a` ([10.2908/IRT_LT_MCBY_A](https://doi.org/10.2908/IRT_LT_MCBY_A)) | 10-year convergence-criterion bond yield |

Eurostat serves only its latest vintage. After the April or October EDP notification the hashes change; `fetch_data.py` then stops and prints the new hashes. `python fetch_data.py --accept-revision` continues on the new vintage. Reuse follows Eurostat's free re-use policy (Commission Decision 2011/833/EU) with attribution.

Figures from the 2027 State Budget, the Commission's autumn 2026 forecast and the CFP are inputs to the scenarios, not downloaded data; they are cited in the article and set in `debt_model.py`.

## What the code does

- **History** (`run_analysis.py`): the exact debt-dynamics identity, year by year, splitting each change in the debt ratio into nominal growth, interest, primary balance and stock-flow adjustment (the largest residual is below 1e-13 pp). It also regresses the change in the headline balance on the change in an HP-filtered output gap.
- **Model** (`debt_model.py`): an annual debt-ratio model with output gap, scarring, deflator, an effective interest rate that reprices towards the marginal cost of new debt, automatic stabilisers, and the excessive-deficit-procedure response when the deficit exceeds 3%.
- **Scenarios**: the 2027 budget stance held (primary surplus 2.5%), a drift to the Commission's 1.8% structural primary surplus, and three recessions starting in 2027 (mild, 2009-type, 2009–12-type with bond-market stress). For each, the constant primary surplus from 2028 that reaches 60% in 2035.
- **Monte Carlo**: 10,000 paths per policy. Each draws Portugal's own 1998–2025 shocks (output-gap innovations, deflator residuals, bond-yield changes, stock-flow surprises) as joint year vectors in 3-year blocks. Shocks are demeaned. A robustness run drops 2020 from the pool.
- **Policy dial**: the probability of being at or below 60% in 2035 for primary surpluses from 1% to 4% of GDP.

## Outputs

- `results/results.json`: every number quoted in the article.
- `results/summary.txt`: the rounded headline numbers, compared in CI.
- `figures/fig1_what_moved_debt.png`: yearly decomposition, 2009–2025.
- `figures/fig2_scenarios.png`: deterministic paths.
- `figures/fig3_fan_charts.png`: Monte Carlo fan charts for the two policies.
- `figures/fig4_probability_by_year.png`: probability of being at or below 60%, year by year.
- `figures/fig5_policy_dial.png`: probability in 2035 against the primary surplus held.

## Licence

Code: same licence as the repository. Data: Eurostat, re-used under Commission Decision 2011/833/EU.
