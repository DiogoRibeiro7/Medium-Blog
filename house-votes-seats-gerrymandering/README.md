# Votes, seats and who drew the map: U.S. House elections, 1972–2026

Companion to the Medium article *Votes, Seats and Who Drew the Map: Partisan Bias in U.S. House Elections, 1972–2026*. It checks a widely shared post about the 2026 midterms ("Texas: 53% of the vote, 37% of the seats"), measures the votes-to-seats relationship in every House election since 1972, and estimates whether the party that controls redistricting causes the bias.

Every number in the article is written by `summarize.py` to [`results/summary.txt`](results/summary.txt).

## What is here

| Step | Script | Output |
| --- | --- | --- |
| Download raw returns | `fetch.py` | `data/raw/` (not committed, about 120 MB) |
| District table, 1972–2024 | `build_returns.py` | `data/derived/house_districts_raw_1972_2024.csv` |
| State-by-election panel | `state_panel.py` | `data/derived/state_panel.csv`, imputed district table |
| Votes–seats curves, bias | `votes_seats.py` | `results/curves_by_year.csv`, `national_bias.csv`, `post_check_2026.csv`, `national_symmetry.csv` |
| Who drew the map: causal estimates | `causal.py` | `results/causal_summary.json`, event-study files |
| 2026: old maps at the 2026 swing | `counterfactual_2026.py` | `results/old_map_projection_2026.csv`, `counterfactual_2026.json` |
| Sensitivity | `sensitivity.py` | `results/sensitivity.json` |
| Every quoted number | `summarize.py` | `results/summary.txt` |
| Figures | `figures.py` | `figures/` |

## Data

* **1976–2022:** MIT Election Data and Science Lab, *U.S. House 1976–2022* (CC0), via the TidyTuesday mirror of 7 November 2023.
* **1972 and 1974:** ICPSR 6311 as distributed in Gary King's `JudgeIt` R package (two-party Democratic share and turnout by district).
* **2024:** MIT Election Data and Science Lab, 2024 precinct returns (`MEDSL/2024-elections-official`), aggregated to districts. Fixes for the precinct files are documented in `build_returns.py` (bookkeeping rows, TOTAL rows next to by-mode rows, Bergen County municipal totals, spelling variants, NJ-3 party labels, TX-18 flagged as special, two unopposed seats not on the ballot).
* **2026:** unofficial statewide House vote totals and seat counts for 37 states, transcribed from the Wikipedia article on each state's 2026 House elections on 17 November 2026 (`data/state_results_2026_unofficial.csv`). District-level 2026 returns were not available in machine-readable form two weeks after the election; 2026 enters only through statewide totals. Two Louisiana seats go to runoffs on 12 December and are counted as Republican, as in the post being checked.
* **Who drew each map:** `data/map_drawers.csv`, coded by hand for every plan in states with five or more seats, 1972–2026, plus Utah's 2026 plan. *R* or *D* means one party could enact the plan without the other (trifecta, or a legislature able to override or not subject to a veto); courts, commissions and split governments are neutral. Each code carries a confidence level; the 1970s and 1980s codes are the least certain, and every estimate is also reported without the low-confidence codes.

The derived district table reproduces the official party division of the House at every election from 1972 to 2024 (checked in `tests/`).

## Measures

* **Uniform-swing partisan bias** (district data, 1972–2024): shift every district in a state by the same amount until the statewide vote is 50/50; bias is the Democratic seat share at that point minus one half. Unopposed winners are given 75% of the two-party vote (70% and 80% in the sensitivity file).
* **Seat bonus against the cube law** (statewide totals, 1972–2026): the Democratic seat share minus the share a symmetric curve with responsiveness ρ gives at the state's vote; ρ is estimated from all states and years at 2.997.
* **Old-map projection** (2026): the 2024 district results moved by each state's 2024-to-2026 statewide swing.

## Reproduce

Python 3.12 or later with the versions in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python fetch.py            # only needed to rebuild the district table
python build_returns.py
python state_panel.py
python votes_seats.py
python causal.py
python counterfactual_2026.py
python sensitivity.py
python summarize.py
python figures.py
python -m unittest discover -s tests -v
```

The workflow `.github/workflows/house-votes-seats-gerrymandering.yml` reruns everything after `build_returns.py` from the committed district table and fails if a committed number changes.

## Limits

The 2026 figures are unofficial and partial (California had about 83% of its vote counted when transcribed). The map-drawer codes are judgements, documented row by row. The uniform-swing measures assume that a statewide shift moves every district alike; the 2016–2018 and 2022–2024 projections in `results/old_map_projection_calibration.csv` show how far that assumption missed in a wave year and in a normal one.
