# Billionaire density and poverty — follow-up series

Three follow-ups to the reanalysis of the "Where billionaires are common, poverty is rare" chart (companion: [`../billionaires-poverty-audit`](../billionaires-poverty-audit)). Each part runs a check the first article could not.

| Part | Question | Status |
| --- | --- | --- |
| 1 | Does the association depend on the poverty line and on income versus consumption surveys? Is it about the level of welfare or its distribution? | Article drafted (manuscript #132) |
| 2 | Does billionaire density measure wealth concentration? Billionaire counts against WID top wealth shares, and citizenship against residence counts | Article drafted (manuscript #133) |
| 3 | Within countries over time: when billionaire density rises, does poverty fall faster than the country's own trend? | Planned |

## Data

- **Panel:** the SHA-256-locked 54-country citizenship panel from the first article.
- **PIP:** World Bank Poverty and Inequality Platform, release `20260922`, 2021 PPP, survey years only, no gap filling, fetched at 20 poverty lines from $2 to $100 a day by [`fetch_pip.py`](fetch_pip.py) in the GitHub Actions workflow [`billionaires-poverty-followups-fetch.yml`](../.github/workflows/billionaires-poverty-followups-fetch.yml). The tables in [`data/`](data/) are hash-checked against [`data/pip_manifest.json`](data/pip_manifest.json), which also records the SHA-256 of every raw API response.
- Each panel country is matched to the PIP survey that reproduces its $30 headcount (all 54 match to within 0.000003 points), so every line describes the same survey.
- **WID (Part 2):** World Inequality Database bulk file of 6 November 2026 (903 MB, SHA-256 in [`data/wid_manifest.json`](data/wid_manifest.json)), read by [`fetch_wid.py`](fetch_wid.py) in [`billionaires-poverty-followups-wid.yml`](../.github/workflows/billionaires-poverty-followups-wid.yml). Kept: net personal wealth shares, pre-tax income shares, Ginis and averages for adults with equal splitting within couples (`992j`), from 1980, with WID's per-observation `data_quality` grade.
- **Forbes primary counts (Part 2):** the frozen aggregate output of the 5 November 2026 Forbes person-list audit, [`../billionaires-poverty-audit/evidence/forbes-2026-run-42099896845`](../billionaires-poverty-audit/evidence/forbes-2026-run-42099896845), hash-checked against its `SHA256SUMS.txt`: `countryOfCitizenship` (citizenship) and `country` (used as residence; see the article for the evidence).

## Part 1 — reproduce

```bash
cd billionaires-poverty-followups
python -m pip install -r requirements.txt
python part1_poverty_lines.py            # print every number, write results/part1 and figures/part1
python part1_poverty_lines.py --check    # fail if any number differs from results/part1/numbers.json
python -m unittest discover -s tests -v
```

Outputs: [`results/part1/summary.txt`](results/part1/summary.txt) (readable), [`results/part1/numbers.json`](results/part1/numbers.json) (full precision) and six figures in [`figures/part1/`](figures/part1/).

Methods: OLS with HC3 errors; Papke–Wooldridge fractional logit with delta-method average marginal effects; leave-one-country-out prediction error; decile welfare levels computed as 10 × decile share × survey mean. All results are descriptive, not causal.

## Part 2 — reproduce

```bash
python part2_wealth_shares.py            # print every number, write results/part2 and figures/part2
python part2_wealth_shares.py --check    # fail if any number differs from results/part2/numbers.json
```

Outputs: [`results/part2/summary.txt`](results/part2/summary.txt), [`results/part2/numbers.json`](results/part2/numbers.json) and five figures in [`figures/part2/`](figures/part2/).

Methods: partial correlations with Fisher-z intervals; OLS with HC3 errors; fractional logit with average marginal effects per standard deviation at all 20 poverty lines; leave-one-country-out prediction error. WID shares are for 2024; a country counts as "observed" when any year since 1995 carries a WID quality grade above zero, and as imputed otherwise. Residence results use the 51 countries with at least one resident billionaire (a +0.5 version keeps all 54).
