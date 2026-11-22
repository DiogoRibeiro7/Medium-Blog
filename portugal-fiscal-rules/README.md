# A Decision Analysis of Fiscal Rules for Portugal

Reproduction code for the Medium article that follows [`portugal-debt-2035-probability`](../portugal-debt-2035-probability/) and [`portugal-fiscal-reaction`](../portugal-fiscal-reaction/). The first held the primary surplus fixed. The second estimated how Portugal's budget has responded to its debt. This one treats the fiscal rule itself as the decision. It compares six ways of setting the primary balance from 2028, all calibrated to the same average effort, on three risks:

- a bad debt outcome in 2035;
- a headline deficit above 3% of GDP;
- a tightening of a point of GDP or more while the economy is below potential.

## Run

```bash
pip install -r requirements.txt
python fetch_data.py      # downloads AMECO (ameco0.zip) into raw/ and checks SHA-256
python build_data.py      # rebuilds data/portugal.csv: Portugal, 1994-2025
python run_analysis.py    # results/results.json, results/summary.txt, figures/*.png (about 45 seconds)
python -m unittest discover -s tests -v
```

`raw/` and `data/` are rebuilt at every run and are not committed. CI (`.github/workflows/portugal-fiscal-rules.yml`) lints, fetches and checks the digest, runs the tests, reruns the analysis and fails if `results/summary.txt` changes.

## Data

Everything comes from AMECO, the European Commission's annual macro-economic database, in its autumn 2026 release, registered in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data) as `ec-ameco-autumn-2026` and pinned in `data_sources/ec-ameco-autumn-2026.json` at the repository root. The series are Portugal's gross debt, primary balance, headline balance, interest, total revenue and primary expenditure (EDP basis, % of GDP), the output gap, nominal and real GDP, the GDP deflator and the long-term interest rate. AMECO's forecasts for 2026–2028 are dropped.

## What the code does

- **Macro block** (`rules.py`): the same annual debt model as the two earlier articles, from the 2027 budget's starting point. It has an output gap with scarring, a deflator, gradual repricing of the debt stock, interest, stock-flow adjustments and the corrective arm. Two additions:
  - a fiscal multiplier of 0.5 on changes in the structural primary balance, with a one-year lag;
  - real-time output-gap error: rules that use the gap miss half of each year's cyclical surprise.
- **Rules** (from 2028; 2027 is the budget's 2.5% under all of them):
  - fixed primary surplus, with room for the automatic stabilisers;
  - fixed primary surplus with no room for the cycle;
  - a debt rule at Portugal's estimated long-run response (0.078);
  - a strong debt rule (0.2);
  - a nominal spending path in four-year plans, re-anchored to a structural primary balance at each plan;
  - a cyclically adjusted headline-balance rule with a Swiss-style amortisation of shortfalls over six years.
- **Fiscal surprises:** the residuals of Portugal's fiscal reaction function, 1996–2025, applied to every rule as one-off additions to the primary balance.
- **Shocks:** 10,000 futures built by drawing three-year blocks of Portugal's 1998–2025 joint yearly shocks (output gap innovation, deflator, yield, stock-flow, fiscal surprise). The same futures are used for every rule (common random numbers).
- **Equal effort:** each rule's level is set by bisection so that its average primary balance over 2028–2035, across futures, is 2.5% of GDP.
- **Comparison:** the three risks; Pareto dominance; the best rule for every weighting of the three risks; a sweep of the debt rule's strength.
- **Sensitivity:** five variant rules, and the sensitivity of the comparison to:
  - the effort level (2.0, 2.5, 3.0);
  - the multiplier (0, 0.5, 1);
  - the real-time gap error (0, 0.5, 1);
  - the block length (1, 3, 5);
  - the seed.

## Outputs

- `results/results.json`: every number quoted in the article.
- `results/summary.txt`: rounded headline numbers, compared in CI.
- `figures/fig1_one_bad_future.png`: one resampled future with the 2008–2010 shocks, under the six rules.
- `figures/fig2_scorecard.png`: the six rules on four measures.
- `figures/fig3_tradeoff.png`: debt outcomes against tightening in downturns, with the debt-rule sweep.
- `figures/fig4_weights.png`: the best rule for each weighting of the three risks.
- `figures/fig5_sensitivity.png`: real-time gap error and effort level.

## Licence

Code: same licence as the repository. Data: European Commission, AMECO, re-used under Commission Decision 2011/833/EU with attribution.
