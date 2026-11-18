# Org charts for language-model agents

Companion code for the Medium article *Copying the Org Chart Into Multi-Agent LLM Systems: Released Traces and Three Models From Organisation Theory*.

One script prints every number in the article and draws every figure. It calls no model. Section 1 measures execution traces that other researchers released; sections 2 to 4 are small models whose assumptions are stated in the output.

## Data

The MAST repository of Cemri et al. (2025), *Why Do Multi-Agent LLM Systems Fail?*, released the traces of several multi-agent systems. Two of them were run on the same 30 MMLU multiple-choice questions (19 subjects) under `traces/mmlu`:

- **ChatDev**, configured as a software company with a chief executive, chief product officer, chief technology officer, programmer, code reviewer, test engineer and counsellor. Two runs per question; the logged configuration names GPT-4o. The folder holds one further, unpaired run of question 0 (`mmlu_0.log`), which is set aside.
- **MetaGPT**, configured with a coder, a tester and a reviewer. One run per question; the trace does not record the model.

`fetch_traces.py` reads the traces at commit `a70542e` and writes one row per run to `data/chatdev_runs.csv` and `data/metagpt_runs.csv`. The rows hold counts, token totals per stage and extracted answer indices, never trace text. The MAST repository carries no licence file, so the traces are not redistributed here; `python fetch_traces.py --check` fetches the pinned commit and confirms that the committed tables rebuild identically.

No answer key is used. Nothing in the analysis depends on which answers were right.

### What is extracted, and how

- Tokens, cost, duration and artefact counts come from ChatDev's own end-of-run summary. Tokens per stage add up each call's logged usage under the phase in which it was made; they sum exactly to the run total.
- The answer a ChatDev run commits to is read from the final version of its code: a literal index in a return statement, an assignment such as `correct_index = 3`, a constructor argument after the options list, or the string `The answer is (3)`. A run is `single` when every match gives the same index, `branching` when the code can return several, and `none` when it holds no index. A random sample of eight `single` runs was checked by hand.
- The MetaGPT coder's answer is a literal in 26 runs. In three it is resolved from `options.index("...")` with `ast.literal_eval`, which reads literals and executes nothing, and in one (question 7, an arithmetic case) it is resolved by hand, as documented in the code. The tester's expected answer is the first `assert ... == "The answer is (d)"` outside a `pytest.raises` block.
- A MetaGPT reviewer "disputes the answer" when it says the answer or premise is incorrect. All 60 reviewer messages were searched for wording of this kind, and the one match (question 22) was read in full.

## What the script covers

| Section | Question |
|---------|----------|
| 1 | What two agent "software companies" did with 30 multiple-choice questions: product meetings, GUIs, tests, reviews, tokens, and who could change the answer. |
| 2 | Separation of duties with reviewers whose errors are correlated with the producer's (one-factor probit model, exact by quadrature, checked by Monte Carlo), with and without anchoring on the producer's answer; a weaker independent reviewer against a copy of the producer. |
| 3 | Five agents deliberating over three rounds with conformity, deference to a lead, and evidence-driven correction; final group accuracy, unanimity and abandoned right answers. |
| 4 | One human with a fixed daily reading budget overseeing a growing team: net value against team size and the cost of an uncaught error. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python org_chart_agents.py > results/output.txt
```

Python 3.13, about ten seconds. The script writes five figures to `figures/` and prints the numbers; `results/output.txt` is the run the article quotes. All randomness comes from one seeded generator, so a rerun with the pinned versions reproduces the output. CI checks that it does, and that the trace tables rebuild from the pinned commit.

## Assumptions

The models in sections 2 to 4 are illustrations of mechanisms, not estimates of any particular system. Their parameters (error rates, correlations, conformity, reading budget, cost of an uncaught error) are stated in the output and varied over ranges. Two inputs come from published work: two erring models choose the same wrong answer 60% of the time (Kim et al. 2025, used to set how often wrong agents agree in section 3), and adults read English non-fiction silently at about 238 words per minute (Brysbaert 2019, used in sections 1 and 4).

## Licence

Code: Apache-2.0. The traces belong to their authors; cite Cemri et al. (2025) if you use the derived tables.
