# Blind testers, anchored testers and testers from another model family

Companion code for a follow-up to *Copying the Org Chart Into Multi-Agent LLM Systems* (folder `org-chart-agents`). That article's Model 1 had two parameters for a reviewer that checks a producer's work: anchoring *a*, the probability that the reviewer defers to the answer it has read, and error correlation *ρ* between the reviewer's errors and the producer's. In the released MetaGPT traces the tester's expected answer was the coder's in 30 of 30 runs, so the traces showed anchoring directly. They could not show *ρ*, because the runs had no answer key, so the article assumed it.

This experiment measures both parameters, on questions with a known answer.

The project is repository-first. The article is written from the numbers this code produces.

## Design

The same multiple-choice question goes to a model in up to four roles, each a separate call with no shared history:

- **coder**: decides which option the team's function returns;
- **blind tester**: decides what the unit test should expect, before seeing the coder's work;
- **anchored tester**: decides what the test should expect after reading the coder's full response (MetaGPT's order);
- **warned tester**: as anchored, but told that the coder may be wrong and to work the answer out itself.

The prompts are in `common.py` (`SYSTEM`, `INSTRUCTION`), version `v1`. Every response ends with a line `ANSWER: X`, which is the only part scored.

The coder and the blind tester answer every sampled question. The anchored and warned testers answer every question on which the blind tester disagreed with the coder, plus a fixed random 20% of all questions, the **check set**. When the blind tester agrees with the coder, reading the coder's answer can only confirm it, unless reading it makes the tester change its mind. The check set tests that assumption. Asking only where it matters saves most of the compute.

What is measured, per model and per pair of models:

- error rates of the coder and the blind tester;
- **error correlation**: the tetrachoric correlation of the two error indicators, which is the *ρ* of Model 1, for a blind tester from the same model and for blind testers from every other model;
- **anchoring**: among the questions where the blind tester disagreed, the share where the anchored (or warned) tester adopted the coder's answer;
- **what passes**: the share of questions on which the coder was wrong and the test expected the coder's answer, for each design. Model 1's residual counts every case where both are wrong. On multiple choice, a tester that is wrong in a different way still flags the answer, and the analysis reports both.

Intervals: Wilson for proportions, and a bootstrap over questions (2,000 replicates, seeded) for correlations and paired differences.

## Questions

MMLU (Hendrycks et al., 2021), test split, from `cais/mmlu` on Hugging Face (`all/test-00000-of-00001.parquet`), downloaded on every run. Pinned in `config.json` at revision `c30699e` with the file's SHA-256, which every run checks: 14,042 questions in 57 subjects. 400 questions drawn at random (seed 20261123), the same for every model. MMLU is MIT-licensed. Neither the questions nor the key are redistributed here.

## Models

Open-weight models served by [Ollama](https://ollama.com) on GitHub-hosted CPU runners, in their default quantised builds, at temperature 0 with a fixed seed, up to 300 output tokens and a 4,096-token context. The reasoning trace is switched off (`think: false`) for models that have one.

| Key | Ollama model | Developer | Family |
|---|---|---|---|
| `llama3.1-8b` | `llama3.1:8b` | Meta | Llama |
| `qwen3-8b` | `qwen3:8b` | Alibaba (Qwen team) | Qwen |
| `qwen3.5-9b` | `qwen3.5:9b` | Alibaba (Qwen team) | Qwen |
| `ministral-3-8b` | `ministral-3:8b` | Mistral AI | Ministral |
| `gemma4-12b` | `gemma4:12b` | Google | Gemma |
| `eurollm-9b` | `hf.co/bartowski/EuroLLM-9B-Instruct-GGUF:Q4_K_M` | EuroLLM consortium | EuroLLM |

The two Qwen models give one pair from the same family, so the correlation can be compared at three distances: the same model in two roles, two models of one family, and models of different families. Every shard records the Ollama version and the weights' digest (`runs/*.provenance.json`), and the analysis warns if the shards of a model ran different weights.

The plan was the free GitHub Models API, with GPT-4o, the model in the MAST traces. GitHub retired that service on 30 July 2026, so the experiment runs on open models that fit a CPU runner instead. Its results are about these models, not about GPT-4o.

## Running

`.github/workflows/org-chart-blind-tester-run.yml` starts when `run_request.json` changes on this branch, or by hand from the Actions tab once it is on main. A plan job makes one job per model and shard (positions modulo the number of shards, from 4 to 9 per model, sized so that each job takes about an hour). Each job installs Ollama, pulls the model and resumes from any responses already on the runs branch. It then asks its questions and uploads its file. A last job pushes every file to the **runs branch**, `org-chart-blind-tester/runs/<key>.shard<i>of<k>.jsonl`, one JSON line per call (prompt version, model, role, question id, response text, parsed answer, finish reason, token counts and time). The outputs never land on main. A job that hits the time limit stops cleanly, and the next run carries on from where it stopped.

The responses used in the article are pinned in `config.json` (`runs_commit`, commit `510cd3e` of the runs branch): 6,406 calls, 400 questions for each of the six models, about 56 hours of model time on free runners. `results/output.txt` and `results/summary.json` hold every number and `figures/` the four figures. CI fetches the pinned responses, reruns `analyse.py` and fails if a number changes.

## What the analysis covers

1. Coverage: questions completed, unparsed answers, answers read from "the answer is X", replies cut at the token limit.
2. Error rates of the coder and the blind tester.
3. Error correlation (tetrachoric) between every coder and every blind tester: same model, same family, other family.
4. Anchoring: how often the anchored and warned testers adopt the coder's answer where the blind tester disagreed; the check set.
5. Misses and false alarms of each design, with paired bootstrap intervals; the cost ratio at which a tester from another family beats the same-model blind tester.
6. The first article's Model 1 fed the measured rates.
7. What reordering buys, by question difficulty (how many other models' coders got the question wrong).
8. Sensitivity: the correlations without the questions on which nearly every coder chose the same wrong option, a sign of a wrongly keyed question.

`run_request.json` names the runs branch, the number of questions (0 for all) and the shards. The smoke run used `org-chart-blind-tester-smoke` with 6 questions. The experiment uses `org-chart-blind-tester-runs`. Once complete, the runs branch is registered as an external, pinned reference in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data).

Locally, with Ollama running:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v                          # stand-in Ollama, about 30 s
ollama pull llama3.1:8b
python run.py --runs-dir runs --models llama3.1-8b --shard 0 --shards 1
python analyse.py --runs-dir runs > results/output.txt
```

## Tests

`tests/fake_models.py` stands in for an Ollama server, with simulated agents whose error rates, error correlations and anchoring rates are known: 0.70 to 0.80 within a model, 0.60 within a family and 0.40 across families. It also simulates a server error and unparseable replies. `tests/test_pipeline.py` runs every shard as a sequence of interrupted jobs that resume. It checks that nothing is duplicated or lost and that the shards cover every question once. It also checks that follow-up roles are asked exactly where the design says and that the analysis recovers the simulation's correlations and anchoring rates.

## Limits

- Small open models, quantised to about 4 bits, on CPU. Their error rates on MMLU are higher than GPT-4o's, and their anchoring may differ.
- One prompt version, at temperature 0. Anchoring may depend on the wording, so the warned tester is a second wording, not a general answer.
- The task is a multiple-choice question, not code. That is MetaGPT's MMLU set-up with the code taken out: the test's expected value is the only thing a tester contributes there.
- The coder and the tester are the same model with different instructions. They are not two copies of one agent with a shared history.
- MMLU is old and probably in every model's training data. That inflates accuracy, but the experiment is about which errors two testers share, not about accuracy.

## Licence

Code: Apache-2.0. Cite Hendrycks et al. (2021) for MMLU.
