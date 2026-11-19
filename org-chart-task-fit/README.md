# What stays fixed when an agent organisation changes tasks

Companion code for the Medium article *What Stays Fixed When an Agent Organisation Changes Tasks: ChatDev, MetaGPT and AG2 in the MAST Traces*, a follow-up to *Copying the Org Chart Into Multi-Agent LLM Systems* (folder `org-chart-agents`).

One script prints every number in the article and draws every figure. It calls no model.

## Data

All traces come from the MAST repository of Cemri et al. (2025), *Why Do Multi-Agent LLM Systems Fail?*, at commit `a70542e`:

- **ChatDev**, configured as a software company, on 30 MMLU multiple-choice questions (two runs each; the unpaired `mmlu_0.log` is set aside) and on the 30 ProgramDev software tasks, one run each. The configuration logs GPT-4o in both. The MMLU runs ran on a Linux machine with no display and the ProgramDev runs on macOS, as file paths and system messages in the logs show.
- **MetaGPT**, configured with a coder, a tester and a reviewer, on the same 30 MMLU questions and 30 ProgramDev tasks.
- **AG2** on 200 GSM-Plus problems in three configurations, the ones MAST's case study compared: the original two-agent chat (`org_traces`), an improved prompt (`prompt_traces`), and a new topology that adds a verifying agent (`topology_traces`). One run per configuration; the traces do not record the model.

The AG2 runs are scored against the answer key of GSM-Plus (Li et al., 2024) at commit `3474129` of github.com/qtli/GSM-Plus. Each trace is matched to its GSM-Plus row by question text; all 600 match, and the three configurations hold the same 200 problems in the same order.

`fetch_traces.py` fetches both pinned commits (sparse, shallow) and writes `data/chatdev_runs.csv`, `data/metagpt_runs.csv` and `data/ag2_runs.csv`. The rows hold counts, token totals per stage, flags and extracted answers, never trace or question text. Neither source repository carries a licence file, so nothing from them is redistributed here. `python fetch_traces.py --check` rebuilds the tables from the pinned commits and confirms they are identical to the committed ones.

### What is extracted, and how

- **ChatDev.** Tokens, calls, duration and artefact counts come from ChatDev's end-of-run summary. Tokens per stage add up each call's logged usage under the phase in which it was made and sum exactly to the run total. A "test report" is each logged execution of the program by the test engineer; it counts as clean when ChatDev records "The software run successfully without errors". The platform is read from file paths (`/home/` or `/Users/`) and macOS input-method messages.
- **MetaGPT.** The sequence of speakers and whether any execution output (an exit code, a traceback or a test-runner summary) appears anywhere in the trace.
- **AG2.** The final answer is the last `\boxed{}` in the original and improved-prompt runs, and the content of the `SOLUTION_FOUND` line in the verifier runs. A verifier run without such a line hit the conversation's message cap and has no final answer. Answers are compared numerically (relative tolerance 1e-4, or equal to two decimals). GSM-Plus "critical thinking" problems lack information needed for an answer, and their key is `None`. `gold_in_transcript` records whether the key's number appears anywhere in the agents' messages.

## What the script covers

| Section | Question |
|---------|----------|
| 1 | ChatDev on two kinds of work: roles, phases, budget, where the tokens went (with a bootstrap over tasks), review and testing. |
| 2 | MetaGPT on the same two task sets: the message sequence and whether tests are ever run. |
| 3 | AG2 with an improved prompt or an added verifier: accuracy with Wilson intervals, by problem type, and paired against the original chat with exact McNemar tests; the verifier runs that never declared an answer. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python org_task_fit.py > results/output.txt
```

Python 3.13, a few seconds. The bootstrap uses one seeded generator, so a rerun with the pinned versions reproduces the output. CI checks that it does, and that the tables rebuild from the pinned commits.

## Limits

One released run per AG2 configuration and one per ProgramDev task, so these are single draws of stochastic systems. MAST's paper reports six repetitions per AG2 configuration with its own scoring, and its numbers differ from these. The ChatDev comparison crosses task type with machine: the two task sets ran on different operating systems, which matters most for testing.

## Licence

Code: Apache-2.0. The traces and GSM-Plus belong to their authors; cite Cemri et al. (2025) and Li et al. (2024) if you use the derived tables.
