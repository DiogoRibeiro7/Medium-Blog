# xˣ = 49

Companion code for the Medium article *The Equation xˣ = 49: Convergence Proofs, a Period-Doubling Bifurcation, and the Complex Solutions Behind the Lambert W Formula*.

The three scripts reproduce every number and every figure in the article.

## Files

| File | What it does |
|------|--------------|
| `solve_xx.py` | Solves xˣ = 49 by bisection, Newton's method (raw and log form), two fixed-point iterations, a hand-written Lambert W (Halley) and SciPy, against a 60-digit reference. Also runs the million-value timing comparison. |
| `xx_theory.py` | Checks the article's theorems numerically: the Newton error bound, the step count for the raw form, convergence orders in 4,000-digit arithmetic, the two-cycle and its multiplier, the fold at 1/e, the complex solutions with argument-principle counts, and the interval-Newton enclosure. |
| `figures.py` | Draws the five figures (`fig1_…` to `fig5_….png`) into the current directory. Imports the other two scripts, so keep all three together. |

## Run

```bash
pip install numpy scipy mpmath matplotlib
python solve_xx.py
python xx_theory.py
python figures.py
```

Python 3.9 or later. The three scripts take about half a minute in total, most of it in `xx_theory.py`. Results are printed to the terminal.

## Notes

- The timings printed by `solve_xx.py` depend on the machine; the article quotes one run and relies only on the ordering.
- `xx_theory.py` verifies statements; it does not prove them. The proofs are in the article.
- The interval enclosure relies on mpmath's interval arithmetic rounding outward correctly.
