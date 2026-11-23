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

The coder and the blind tester answer every sampled question. The anchored and warned testers answer every question on which the blind tester disagreed with the coder, plus a fixed random 20% of all questions, the **check set**. When the blind tester agrees with the coder, reading the coder's answer can only confirm it, unless reading it makes the tester change its mind. The check set tests that assumption. Asking only where it matters saves most of the quota.

What is measured, per model and per pair of models:

- error rates of the coder and the blind tester;
- **error correlation**: the tetrachoric correlation of the two error indicators, which is the *ρ* of Model 1, for a blind tester from the same model and for blind testers from every other model;
- **anchoring**: among the questions where the blind tester disagreed, the share where the anchored (or warned) tester adopted the coder's answer;
- **what passes**: the share of questions on which the coder was wrong and the test expected the coder's answer, for each design. Model 1's residual counts every case where both are wrong. On multiple choice, a tester that is wrong in a different way still flags the answer, and the analysis reports both.

Intervals: Wilson for proportions, and a bootstrap over questions (2,000 replicates, seeded) for correlations and paired differences.

## Questions

MMLU (Hendrycks et al., 2021), test split, from `cais/mmlu` on Hugging Face (`all/test-00000-of-00001.parquet`), downloaded on every run. The resolved revision and the file's SHA-256 are recorded by `probe.py` and pinned in `config.json` after the first probe. 300 questions drawn at random (seed 20261123). Models with a smaller quota answer the first 100 of the same order, so their questions are a random subset of everyone else's. MMLU is MIT-licensed. Neither the questions nor the key are redistributed here.

## Models and quota

Served by [GitHub Models](https://docs.github.com/en/github-models) through the workflow's own token (`permissions: models: read`), at temperature 0 with up to 400 output tokens:

| Key | Candidates (first one in the catalog is used) | Family |
|---|---|---|
| `gpt-4o-mini` | `openai/gpt-4o-mini` | OpenAI |
| `gpt-4o` | `openai/gpt-4o` (the model in the MAST traces) | OpenAI |
| `llama` | `meta/Llama-3.3-70B-Instruct`, `meta/Meta-Llama-3.1-8B-Instruct` | Meta |
| `mistral-small` | `mistral-ai/mistral-small-2503` | Mistral AI |
| `phi-4` | `microsoft/Phi-4` | Microsoft |

The free quota is a number of requests per model per day: 150 for models in the catalog's `low` tier and 50 for `high`. A `low` model gets 300 questions, a `high` one 100. At about 2.7 calls per question that is roughly six days of scheduled runs.

## Running

`.github/workflows/org-chart-blind-tester-probe.yml` runs on pushes to this branch. It runs the tests, probes the catalog and MMLU, makes one call per model and commits `results/probe.json`.

`.github/workflows/org-chart-blind-tester-run.yml` runs daily once on main, or by hand from the Actions tab. Each run works until every model's quota is used up or its questions are done. It resumes from earlier runs and commits the responses to the **`org-chart-blind-tester-runs` branch**, `org-chart-blind-tester/runs/<key>.jsonl`, one JSON line per call (prompt version, model id and the id the service reported, role, question id, response text, parsed answer, finish reason, token usage). The outputs never land on main. Once complete, the runs branch is registered as an external, pinned reference in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data).

Locally:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v          # simulated models, about 30 s
GITHUB_TOKEN=... python run.py --runs-dir runs    # a token with models:read
python analyse.py --runs-dir runs > results/output.txt
```

## Tests

`tests/fake_models.py` stands in for the API with simulated agents whose error rates, error correlations (0.7 to 0.8 within a model, 0.4 across models) and anchoring rates are known. It also simulates daily quotas, a short rate limit, content-filter refusals and unparseable replies. `tests/test_pipeline.py` runs the scheduled loop over several simulated days. It checks that nothing is duplicated or lost and that follow-up roles are asked exactly where the design says. It also checks that the analysis recovers the simulation's correlations and anchoring rates.

## Limits

- One prompt version, at temperature 0. Anchoring may depend on the wording, so the warned tester is a second wording, not a general answer.
- The task is a multiple-choice question, not code. That is MetaGPT's MMLU set-up with the code taken out: the test's expected value is the only thing a tester contributes there.
- The coder and the tester are the same model with different instructions. They are not two copies of one agent with a shared history.
- MMLU is old and probably in every model's training data. That inflates accuracy, but the experiment is about which errors two testers share, not about accuracy.
- GitHub Models serves the models through Azure, and its content filter can refuse a question. Refusals are recorded and excluded from the correlation estimates.

## Licence

Code: Apache-2.0. Cite Hendrycks et al. (2021) for MMLU.
