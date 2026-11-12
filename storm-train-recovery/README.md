# Recovery capacity under clustered storms

Public reproducibility companion for Diogo Ribeiro's article *Recovery Capacity under Clustered Storms: A Queueing View of Portugal's 2026 Storm Train*.

The manuscript is kept separately in the private [`research-articles-medium`](https://github.com/DiogoRibeiro7/research-articles-medium) repository. This folder holds the data, the code, the printed results and the figures.

## Run

Requirements: Python 3.11 or newer, NumPy, SciPy and Matplotlib (the versions CI uses are pinned in `requirements.txt`).

```bash
python -m pip install -r requirements.txt
python storm_train.py > results/output.txt
```

The script is deterministic and downloads nothing; it takes about ten seconds. It prints every number quoted in the article to `results/output.txt`, writes seven figures to `figures/`, and renders the nine displayed equations to `figures/equations/` (Medium does not render LaTeX, so the article shows them as images). The CI workflow reruns it and fails if the printed numbers change.

## What is in it

**1. The Portuguese storm train.** `data/eredes_customers_out_2026.csv` holds 53 E-Redes outage counts published between 27 January and 22 February 2026, each with the time it refers to and the article or press release that reported it. E-Redes reported two kinds of figure: customers without power in the whole of mainland Portugal (`total`) and customers without power in the areas hit by Storm Kristin (`kristin`; the company's wording varies between "zona da depressão Kristin", "zonas mais críticas" and "localidades afetadas"). A two-phase curve

N(t) = N₀ [a·exp(−t/τ₁) + (1 − a)·exp(−t/τ₂)]

is fitted in logs to the 28 reports that measure the Kristin backlog alone: whole-mainland counts before the first new faults on 2 February, and Kristin-area counts afterwards, excluding any taken within 36 hours of a new storm or a new wave of faults (the `fit_baseline` column marks which, and the `note` column says why). N₀ is the reported peak of about one million customers at 06:00 on 28 January. Uncertainty comes from a residual bootstrap. The script also measures the drain rate of the Kristin backlog between consecutive anchor reports, so the windows with and without a new storm can be compared, and integrates customer-days without power against the fitted Kristin-only curve.

**2. Florida and Louisiana.** `data/us_restoration.csv` holds restoration points for Florida Power & Light after Charley, Frances and Jeanne (2004), Wilma (2005), Ian (2022) and Milton (2024), and for Entergy after Katrina (2005), from U.S. Department of Energy situation reports, Florida Public Service Commission filings and utility statements. Rows marked `statement` are a utility's own summary ("75% restored by day 3", "restored in 13 days") rather than a timed count. Times for the US are days since landfall; for Kristin, days since the outage peak. `data/capacity.csv` puts the restoration workforce next to the number of customers affected and, where it was published, the number of damaged poles.

**3. Storm seasons.** A season of 120 days with a mean of two damaging storms is simulated 20,000 times with storms arriving either independently (Poisson) or in trains (a stationary Hawkes process with an exponential kernel of mean 3 days, built from its cluster representation with a 60-day burn-in). The branching ratio n is set from the dispersion of seasonal counts, ψ = 1/(1 − n)² − 1, for ψ = 0.5 and ψ = 1. Storm damage is lognormal with mean 1 ("one storm-unit of repair work") and log-sd 1.

**4. The repair queue.** Each season's backlog Q is drained by own crews at μ₀ = 0.1 storm-units a day plus an outside surge S of up to 0.1 more. The surge is requested to clear the backlog in 3 days, arrives with a 4-day time constant and is sent home with a 1-day time constant as the backlog shrinks. A storm that arrives while repairs are under way does extra damage: its damage is multiplied by 1 + κ·min(1, Q), with κ = 0.5 in the base case. The script compares five levers against this base system under both arrival processes: 30% less damage per storm (hardening), 50% more own crews, a surge that arrives in 1 day, keeping surge crews 10 days after each storm, and halving the compounding.

The parameters of parts 3 and 4 are **illustrative**. They are chosen so that the mechanics are visible and the scale is loosely Kristin-like (own crews about doubled by the surge, as E-Redes went from about 1,200 to about 2,400 field staff in a week). They are not estimates for E-Redes, FPL or Entergy, and the simulated "backlog-days" are in storm-units, not customers.

## Figures

| File | What it shows | Status |
| --- | --- | --- |
| `figures/01_storm_train.png` | E-Redes customers without power, 27 January to 22 February 2026, with the five named storms and the field staff | Published counts |
| `figures/02_kristin_backlog.png` | Two-phase fit to the Kristin backlog; drain rate window by window | Published counts, fitted |
| `figures/03_restoration_comparison.png` | Share of affected customers still without power: Kristin, FPL 2004–05 and 2022–24, Entergy after Katrina | Published counts and utility statements |
| `figures/04_capacity.png` | Restoration workers per 1,000 customers affected and per damaged pole | Published figures |
| `figures/05_next_storm.png` | Chance of another damaging storm within h days of one, for independent storms and for trains | Simulation, illustrative parameters |
| `figures/06_backlog_tails.png` | Distribution of seasonal backlog-days, with and without trains and compounding | Simulation, illustrative parameters |
| `figures/07_levers.png` | Reduction in mean and 95th-percentile backlog-days from five levers | Simulation, illustrative parameters |

## Caveats on the data

- An E-Redes "customer" is a delivery point (a home, a shop, a business), not a person; FPL and Entergy count customer accounts. The two are close but not identical.
- E-Redes did not publish counts at regular times. Between reports the script interpolates linearly in logs, which smooths over any peak that fell between two reports; the customer-days it computes for the later storms are therefore on the low side.
- Two of the early counts are lower bounds ("mais de 550 mil", "mais de 290 mil") and are used as values.
- The Kristin-area counts after 2 February can include new faults inside the Kristin area. The drain-rate windows are labelled accordingly.
- The Wilma, Ian and Milton points include utility statements of the form "half restored after five days"; the Katrina curve is three statements and is drawn dotted.
- Workforce figures are what each utility reported at the time; they mix line workers, tree crews and support staff in different proportions.
