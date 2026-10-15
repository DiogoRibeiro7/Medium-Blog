# Evaluating over-engineering in LLM coding agents

Companion code for the Medium article *Measuring Over-Engineering in LLM Coding Agents: Metrics, Variance and Experimental Design*.

One script prints every number in the article and draws every figure. Nothing in it runs a language model: agent behaviour is simulated under stated assumptions, so the results describe what each evaluation design can and cannot detect, not any particular agent.

## What the script covers

| Section | Question |
|---------|----------|
| 1 | Code size in tokens rather than lines, and behavioural footprint measured with characterisation probes (a semicolon-delimiter ticket with two patches). |
| 2 | Safety at the ceiling: exact bounds after zero failures, Newcombe's interval, and how often a percentile bootstrap over tickets claims non-inferiority for a harmful arm. |
| 3 | Variance components, tickets needed for 80% power, and the cost-optimal number of runs per ticket, R* = (CV / τ) · √(c_T / c_E), checked against simulation. |
| 4 | Three estimators of the size ratio (geometric mean, median, ratio of totals): power and coverage of their own targets. |
| 5 | One-factor-at-a-time ablation against a 2^(8−4) resolution IV fractional factorial with two substitute components, including equivalence (TOST) rates. |
| 6 | Best-of-k selection by size: the reduction, and the share of unsafe patches delivered with and without safety probes. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python agent_size_evaluation.py > results/output.txt
```

Python 3.13, about a minute on a laptop. The script writes the five figures to `figures/` and prints the numbers; `results/output.txt` is the run the article quotes. All randomness comes from one seeded generator, so a rerun with the pinned versions reproduces the output.

## Assumptions

The simulation parameters are assumptions chosen to be plausible, not measurements: a true size ratio of 0.85, effect heterogeneity across tickets τ = 0.3 on the log scale, run-to-run log standard deviation 0.5 plus Poisson counting noise, ticket sizes log-normal with median 60 tokens. Section 6 rests on a stated conjecture, that candidates which omit a guard are smaller (10% of candidates, 30% smaller on average). Replace these with estimates from a pilot and rerun.
