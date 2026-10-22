# How many runs does an agent comparison need?

Companion code for the Medium article *How Many Runs Does an Agent Comparison Need? Variance Components from 6,428 Released τ-bench Runs*.

One script prints every number in the article and draws every figure. It calls no model: it reads 6,428 agent runs that Sierra Research released with τ-bench and τ²-bench, in which every task was run four or eight times by the same model under the same setup.

## Data

| Source | Models | Domains | Runs per task | Runs |
|--------|--------|---------|---------------|------|
| τ-bench, `historical_trajectories/` | GPT-4o, Claude 3.5 Sonnet (2024-10-22) | airline (50 tasks), retail (115) | 4 and 8 | 1,980 |
| τ²-bench, `data/tau2/results/final/` (paper runs, June 2025 task sets) | Claude 3.7 Sonnet, GPT-4.1, GPT-4.1-mini, o4-mini | airline (50), retail (114), telecom (114) | 4 | 4,448 |

Both upstream repositories are MIT licensed. `fetch_data.py` reads the source files pinned to commits `59a200c` (τ-bench) and `57e5414` (τ²-bench) and writes `data/runs.csv`, one row per run: benchmark, domain, model, task, repeat, pass or fail, and for τ²-bench the dollar cost of the agent and of the simulated user. The table is committed, so the analysis does not need the 440 MB of downloads; `python fetch_data.py --check` rebuilds it from the pinned sources and confirms it is identical.

## What the script covers

| Section | Question |
|---------|----------|
| 1 | The data, and a check that it reproduces the τ-bench pass^k leaderboard. |
| 2 | Where run-to-run noise sits: within- and between-task variance, and passes out of four against the independent-runs model. |
| 3 | Twenty paired model comparisons under two questions (these exact tasks, tasks like these) and three standard errors. |
| 4 | Whether runs are independent given the task (chi-square on whole-run pass rates), and a simulator fitted on runs 1–2 predicting runs 3–4. |
| 5 | Coverage of five 95% interval methods with three runs per task, including the percentile bootstrap over whole runs. |
| 6 | Runs per task, dollars, and tasks needed to detect 2, 5 and 10 point differences, and the floor that runs cannot lower. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python agent_eval_runs.py > results/output.txt
```

Python 3.13, about forty seconds on a laptop. The script writes four figures to `figures/` and prints the numbers; `results/output.txt` is the run the article quotes. All randomness comes from one seeded generator, so a rerun with the pinned versions reproduces the output, and CI checks that it does.

## Assumptions

Sections 1 to 4 and the standard errors in section 3 use the released runs directly. Sections 5 and 6 need runs that were never made, so they draw each task's true pass rate from its posterior under a Beta prior fitted by moments to the model and domain, then simulate runs from it. Fitted on runs 1–2 of every τ²-bench cell, this simulator predicts how runs 3–4 split (both fail, one passes, both pass) within one percentage point. Population statements treat the released tasks as a sample from the kind of tasks the benchmark stands for; that is an assumption about how the tasks were written, not something the runs can check.

Costs are the τ²-bench recorded costs per run (agent plus simulated user) at mid-2025 prices.
