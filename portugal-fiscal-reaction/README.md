# How Portugal's Budget Responds to Its Debt: A Fiscal Reaction Function and the 2035 Target

Reproduction code for the Medium article that follows the probabilistic stress test in [`portugal-debt-2035-probability`](../portugal-debt-2035-probability/). That piece held the primary surplus fixed. This one estimates how Portugal's primary balance has actually responded to its debt and to the cycle since 1996, sets it against the other 26 EU countries, and reruns the 2035 debt simulation with that behaviour in place of a fixed surplus.

## Run

```bash
pip install -r requirements.txt
python fetch_data.py      # downloads AMECO (ameco0.zip) into raw/ and checks SHA-256
python build_data.py      # rebuilds data/panel.csv: EU-27, 1994-2025
python run_analysis.py    # results/results.json, results/summary.txt, figures/*.png (about 45 seconds)
python -m unittest discover -s tests -v
```

`raw/` and `data/` are rebuilt at every run and are not committed. CI (`.github/workflows/portugal-fiscal-reaction.yml`) lints, fetches and checks the digest, runs the tests, reruns the analysis and fails if `results/summary.txt` changes.

## Data

Everything comes from AMECO, the European Commission's annual macro-economic database, in its autumn 2026 release. The data source is registered in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data) as `ec-ameco-autumn-2026`. The pinned contract is `data_sources/ec-ameco-autumn-2026.json` at the repository root: URL, bytes and SHA-256 of `ameco0.zip`, retrieved on 22 November 2026.

| Variable | AMECO code | Unit |
| --- | --- | --- |
| Gross debt | `UDGG` (EDP) | % of GDP |
| Primary balance | `UBLGIE` (EDP) | % of GDP |
| Interest | `UYIGE` (EDP) | % of GDP |
| Output gap | `AVGDGP` | % of potential GDP |
| Cyclically adjusted primary balance | `UBLGBP` | % of potential GDP |
| GDP, current and 2020 prices; deflator | `UVGD`, `OVGD`, `PVGD` | |
| Long-term interest rate | `ILN` | % |

The general government series start in 1995, and AMECO's forecasts for 2026–2028 are dropped. For Portugal, the 2025 values match Eurostat's October 2026 EDP notification (debt 89.15%, primary balance 2.62%). AMECO is replaced at every Commission forecast. On a new release, `fetch_data.py` stops; `--accept-revision` continues.

## What the code does

- **Rule** (`frf.py`): the primary balance is regressed on last year's debt, the output gap, last year's primary balance, and dummies for 2020 and 2021. OLS with Newey–West errors, 1996–2025.
- **Robustness:**
  - Huber M-estimation;
  - the cyclically adjusted primary balance as the outcome;
  - 2020–2021 dropped instead of dummied;
  - level shifts from 2012, 2015 or 2016;
  - a pre/post-2008 split in the debt response;
  - 15-year rolling windows.
- **EU context:** the same regression for each of the 27 member states. DerSimonian–Laird random-effects pooling gives the EU average and spread, and empirical-Bayes shrinkage pulls each country towards it. A pooled fixed-effects estimate has errors clustered by country.
- **Implications:** the long-run primary balance the rule settles at for each debt level, with 90% intervals from the sampling distribution. The anchor is the debt ratio at which that balance just holds debt steady.
- **Simulation** (`debt_model.py`): the same debt model and starting point as the 2035 article, with three policies from 2028:
  - a fixed 2.5% primary surplus;
  - Portugal's estimated rule;
  - the same response to debt, anchored on today's 2.5% at 84.5% debt.

  There are 10,000 block-bootstrap paths per policy. Each draws Portugal's 1998–2025 shocks (output gap, deflator, yield, stock-flow, rule residual) as joint year vectors, and draws the rule's coefficients from their estimated uncertainty.
- **Dial:** the probability of being at or below 60% in 2035, as a function of the primary balance the rule still asks for once debt reaches 60%.

## Outputs

- `results/results.json`: every number quoted in the article.
- `results/summary.txt`: rounded headline numbers, compared in CI.
- `figures/fig1_rule_and_anchor.png`: primary balance against lagged debt, the rule's long-run line and its anchor.
- `figures/fig2_eu_responses.png`: the debt response in each EU country, raw and shrunk.
- `figures/fig3_rolling_response.png`: Portugal's response on 15-year windows.
- `figures/fig4_debt_paths.png`: debt to 2045 under the three policies, no shocks.
- `figures/fig5_probability_by_year.png`: probability of being at or below 60%, year by year.
- `figures/fig6_rule_level_dial.png`: probability in 2035 against the rule's level at 60% debt.

## Licence

Code: same licence as the repository. Data: European Commission, AMECO, re-used under Commission Decision 2011/833/EU with attribution.
