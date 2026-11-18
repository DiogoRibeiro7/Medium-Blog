# Changing the football manager mid-season, judged as a decision

Companion code for the Medium article on in-season manager changes in the big five European leagues
(England, Spain, Italy, Germany, France), 2012/13 to 2025/26.

**The question.** A board that changes its manager after a bad run sees results improve. That
proves little, because results after a bad run improve anyway. The fair yardstick is a club in the
same spot at the same stage of a season that kept its manager that week. That is the alternative a
board actually has, and it includes clubs that change their manager a few weeks later.

**The design.**
- **Decision points.** Every club after every league match is a decision point, provided the
  incumbent has led at least five league matches in a row.
- **Changes.** A change is a decision point after which a different manager leads the club in the
  next league match.
- **Comparison clubs.** Each change is compared with the five kept-manager decision points nearest
  to it:
  - exactly the same latest result, points from the last five matches and side of the relegation
    line;
  - within about two matches of the same stage of the season, and within a caliper on the estimated
    chance of a change;
  - nearest by Mahalanobis distance on season points per match, points per match over the last ten,
    goal difference, league position, shots-on-target share, squad value relative to the league
    median, and the strength of the next ten opponents.
- **Outcome.** Points over the next ten league matches.

**The result** (`output.txt`):
- **What a change appears to do.** Clubs that changed manager took 5.2 more points over the next ten
  matches than their last five predicted.
- **What would have happened anyway.** Clubs in the same spot that kept their manager took 4.6
  more.
- **What the change itself added.** About +0.57 points over ten matches, 95% interval
  [+0.12, +1.07], or about +1.2 points over the rest of a season.
  - Nothing measurable in the first three matches.
  - No measurable change in the chance of finishing in the bottom three.
- **Where the gain sits.** Mostly at clubs whose squad is worth more than their league's median:
  +1.7 points over ten matches.

## Data

| Source | What it gives | Licence |
|---|---|---|
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (final state, 6 July 2026) | every league match with both managers; player market valuations | CC0 |
| [datasets/football-datasets](https://github.com/datasets/football-datasets) at commit `436a51f` (mirror of football-data.co.uk) | shots and shots on target per match | PDDL |

`fetch_data.py` downloads the source files into `.cache/` and checks each against a pinned SHA-256
digest. `build_panel.py` turns them into the three compact tables in `data/`, which are committed.
The analysis reads only `data/`:

- `data/matches.csv`: one row per club and league match.
- `data/squad_values.csv`: squad market value per club and season, on 15 September.
- `data/next_jobs.csv`: for each manager spell, the first later match leading a different club.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # the design on synthetic leagues with a known effect
python coach_numbers.py > output.txt         # every number in the article
python figures.py                            # figures/
python fetch_data.py && python build_panel.py  # optional: rebuild data/ from the pinned sources
```

The workflow `.github/workflows/coach-change-decision.yml` runs all of this and fails if
`output.txt` changes.

## Files

- **`coach_core.py`:** decision points, standings, the model of a change, matching, bootstrap,
  placebo, event paths.
- **`coach_numbers.py`:** all numbers, printed in order, and the tables in `results/`.
- **`simulate.py`:** synthetic leagues in which boards change managers after bad runs and the true
  effect is chosen.
- **`figures.py`:** the six figures.
- **`tests/test_core.py`:** standings, change detection, and the two checks that matter:
  - with no true effect the method returns about zero, while the naive before/after gain is large;
  - with a real effect the method finds it.

## Checks and their limits

- **Placebo.** Treating the nearest kept club of each change as if it had changed gives −0.12
  points, interval [−0.57, +0.34].
- **Unmatched earlier window.** The two groups differ by −0.03 points over matches 14 to 10 before
  the decision, a window no matching variable uses.
- **Null simulation.** On ten simulated leagues in which a change does nothing, the method returns
  −0.03 on average.
- **Distance matching alone** (without the exact match on recent results) drifts upwards by about
  0.2 points on the same simulations, which is why the exact match is used.

The records show that a manager changed, not why. A dismissal, a resignation and a move to another
club look the same. The 13 changes in which the outgoing manager led another club within 30 days
are kept in, and dropping them changes nothing material. A board also knows things these data do
not. `output.txt` reports how strong that hidden information would have to be for the true gain to
reach one, two or three points.

## Licence

Code: Apache-2.0. Data: as listed above.
