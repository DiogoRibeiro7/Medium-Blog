# The PPP drift inside Europe, category by category

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. The previous follow-up found that, inside the group of economies whose PPPs Eurostat compiles, the drift between national deflators and PPPs is larger for economies that were cheap in 2005, and stated a conjecture: that this price-level slope comes from the services whose prices converged fastest, not from goods. This repository tests it with Eurostat's PPPs by analytical category and the matching household-consumption deflators by purpose.

It computes, for 33 European economies and each COICOP division, selected groups and Eurostat's goods and services split, the drift against the EU27 over 2005–2019; regresses each on the log of the 2005 household-consumption price level; splits the slope of the household drift exactly into share-weighted divisions and three aggregation terms; relates the aggregation term to shifts in spending towards relatively cheap categories; compares the drift with two measures of price convergence; and checks windows and samples.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Eurostat snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig3_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/ppp-drift-by-category.yml` reruns the tests and the analysis, checks that the summary is unchanged and draws the figures.

## Data (`fetch.py`, `data/`)

`data/fetch_log.txt` records the URL, size and SHA-256 of every response; raw responses are kept gzipped in `data/raw/`, and `python fetch.py --tidy-only` rebuilds the CSV files from them.

- **Eurostat** (JSON-stat dissemination API, 13 November 2026):
  - `prc_ppp_ind`: PPPs (EU27 = 1), price level indices (EU27 = 100) and nominal and real expenditure for household consumption, its twelve COICOP divisions, eleven groups, and consumer goods (non-durable, semi-durable, durable) and consumer services, 2003–2024 (household consumption from 1995);
  - `nama_10_co3_p3`: household consumption by purpose, implicit deflators (2020 = 100) and current prices, EU27 members, Switzerland, Norway, Iceland, Serbia, Albania, Bosnia and Herzegovina and the EU27 aggregate;
  - `nama_10_fcs`: household consumption by durability, current prices and chain-linked volumes.

Eurostat data are reused under Eurostat's terms; they are not relicensed by the repository's licence. The United Kingdom, Türkiye, North Macedonia and Montenegro have category PPPs but no deflators by purpose in these tables, and are left out.

## Definitions

- **Drift** of economy i in category c over [a, b], against the EU27, log points × 100: ln(P_ic(b)/P_ic(a)) − ln(PPP_ic(b)/PPP_ic(a)) − ln(P_EU,c(b)/P_EU,c(a)), with P the implicit deflator (national currency; the EU27 in euro) and PPP the Eurostat parity for the matching analytical category. Against another reference the drift shifts by a constant across economies, so slopes do not depend on the reference (`tests`).
- **Slope**: OLS of the drift on the log of the 2005 household-consumption price level (EU27 = 100), with HC1 standard errors.
- **Kinds of division**: goods (food, alcohol and tobacco, clothing and footwear, furnishings); services (housing, water, electricity and gas, which is mostly rents; communication; restaurants and hotels; miscellaneous goods and services); mixed (transport, recreation); and health and education, whose PPPs cover government provision while the deflators cover household purchases only.
- **Goods and services split**: Eurostat's consumer goods and consumer services PPPs, matched with Törnqvist deflators built from durability items (durable, semi-durable and non-durable goods; services).
- **Decomposition**: household drift = Σ_c s_c × drift_c + A − B − C, with s_c the economy's average share of division c in 2005 and 2019; A the gap between the household deflator and the share-weighted division deflators; B the same gap for PPPs; C the gap for the EU27 reference deflators weighted with the economy's shares. The slope of each term on the price level adds up to the slope of the household drift.
- **Composition term**: −Σ_c Δs_c × ln(PPP_c/PPP_household), with Δs_c the change in the division's share 2005–2019 and the log relative price averaged over the two years: positive when spending moves towards divisions that are cheap relative to the economy's average.
- **Convergence**: slope on the 2005 price level of the change in each division's price level measured by PPPs (Eurostat's price level indices) or by deflators (relative deflator change minus exchange-rate change against the euro).
- **Housing without energy**: deflator and PPP backed out of the division and its energy group with the economy's average energy share, as if the division were a geometric mean of the two; approximate.

## Caveats

- Category drifts are noisy (standard deviations of 10–50 points across economies), so single-category slopes are imprecise; the decomposition is exact but its terms carry the same noise.
- Health and education PPPs include government provision; their drifts mix a PPP and a deflator of different scope.
- The national accounts give household consumption on the domestic concept by purpose, while the household PPP refers to the national concept.
- Deflators by purpose are published only to 2022 for most economies, so the longest window is 2005–2022.
- Data are one vintage (13 November 2026); PPPs back to 1995 may be revised every December.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
