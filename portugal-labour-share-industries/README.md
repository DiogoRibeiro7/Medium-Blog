# Portugal's labour share by industry, 1995–2024

Companion code for the Medium article *Portugal's Labour Share by Industry: A Shift-Share Decomposition, 1995–2024*, the follow-up to *Portugal's Labour Share Under Eighteen Definitions* ([`../portugal-labour-share`](../portugal-labour-share)).

The labour share of Portuguese factor-cost value added is rebuilt from Eurostat's 64-industry accounts. The script:

- compares crediting the self-employed with the economy-wide pay per employee against crediting them with their own industry's pay;
- splits the changes over 1995–2024, 2010–2013 and 2016–2024 into a part inside industries and a part due to industries changing weight, at three levels of industry detail;
- relates the within-industry change after 2016 to each industry's initial pay;
- repeats the decomposition for Spain, Italy and Greece.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python industries.py
python figures.py
python -m pytest -q tests
```

Python 3.13. The analysis takes a few seconds. It writes:

- `results/summary.txt`: every number quoted in the article
- `results/imputation_levels.csv`: the adjusted share by country, year and imputation rule
- `results/shift_share_grid.csv`: within and between for every window, scope, numerator and level of detail, from endpoints and chained year by year
- `results/contributions_*.csv`: contributions by section and by industry
- `results/pay_and_share_2016_2024.csv`: pay, value added and share change by industry, four countries
- `results/peers.csv`: Portugal, Spain, Italy and Greece
- `figures/fig1_…png` to `figures/fig4_…png`: the article's figures

The permutation test is seeded, so a rerun with the pinned versions reproduces the committed `summary.txt`. The workflow in `.github/workflows/portugal-labour-share-industries.yml` checks this on every change.

## Definitions

- **Value added.** Gross value added at basic prices (B.1g) less other taxes net of subsidies on production (D.29 − D.39), i.e. factor cost. Summed over industries, it equals compensation plus operating surplus and mixed income in the national accounts used by the first article.
- **Industries.** The 64 industries of the A64 breakdown. U (extraterritorial organisations) has no value added and is dropped. C19 (coke and refined petroleum) is merged with C20 (chemicals), and H51 (air transport) with H52 (warehousing), because each has negative value added in some year and a share on a near-zero denominator makes the decomposition terms explode. That leaves 61 industries, which are also grouped into the 21 NACE sections (20 after dropping U) and the 10 A10 branches.
- **Self-employed labour income.** Self-employed persons (total employment less employees) times the group's compensation per employee. The imputation is done economy-wide, within each A10 branch or within each A64 industry, and per hour within each A10 branch. Industry hours are incomplete for Portugal before the 2000s, which is why the per-hour version stops at A10. The imputation is capped at the group's value added less its compensation. The adjusted shares in the decomposition use the A64 per-person imputation.
- **Scope.** The whole economy, or the market economy without real estate (whose value added is dominated by the imputed rent of owner-occupied housing), public administration, education, health and households as employers.
- **Shift-share.** With weights w = Y_i / ΣY and shares s = L_i / Y_i, Δ(Σ w s) = Σ w̄ Δs (within) + Σ Δw (s̄ − S̄) (between), exactly, with bars for two-year averages and S̄ the average aggregate share. Subtracting S̄ leaves the total between term unchanged, because the weight changes sum to zero. It makes each industry's between term interpretable: a growing industry raises the aggregate only if its share is above average. Chained versions sum the year-on-year decompositions.
- **Pay and share.** The weighted rank correlation between 2016 compensation per employee and the 2016–2024 change in each industry's share, weighted by 2016 value added, with a permutation p-value (9,999 permutations).

## Data

`fetch.py` downloads every industry of `nama_10_a10`, `nama_10_a10_e`, `nama_10_a64` and `nama_10_a64_e` for Portugal, Spain, Italy and Greece from the Eurostat SDMX API, keeping each value's flag. The snapshot in `data/` was downloaded by the workflow `.github/workflows/portugal-labour-share-industries-data.yml` on 21 October 2026, and `data/fetch_log.txt` records what each request returned. Rerunning that workflow refreshes the snapshot and commits it if Eurostat has revised anything.

2025 is in the snapshot but provisional and is not used. Section 2 of the analysis also reads Portugal's mixed income (B.3g) and the national aggregates from the first article's snapshot in `../portugal-labour-share/data/`.

## License

Apache-2.0, as the rest of the repository.
