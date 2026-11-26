# Repetition loops in eight open language models: model size against the decoding rule

Companion code for a Medium article about the loops a language model falls into when it always writes its most likely next token. The same 144 prompts, taken from 18 public-domain books, are continued for 256 tokens by eight open models, under six decoding rules. Thesis: **the rule makes the loop.** Greedy decoding loops at every model size in the experiment; sampling removes the loop, and a bigger model does not.

- **The models.** Two families, each trained on one dataset at several sizes:
  - Pythia 70M, 160M, 410M, 1B and 1.4B (EleutherAI), final checkpoint `step143000`. All five saw the same data in the same order.
  - SmolLM2 135M, 360M and 1.7B (Hugging Face), revision `main`.
- **The rules.** Greedy (always the most likely token); temperature sampling at *T* = 0.25, 0.5, 0.75 and 1; nucleus sampling with *p* = 0.9.
- **A loop.** A continuation ends in a block of at most 64 tokens that repeats back to back at least three times and covers at least 24 tokens.
- **The human baseline.** The 256 tokens that follow each prompt in the book, scored by each model token by token.

## What it finds

Filled in from `output.txt` when the full run is analysed.

## Data

- **Prompts.** The 18 books of the NLTK copy of Project Gutenberg (`corpora/gutenberg.zip` in [nltk/nltk_data](https://github.com/nltk/nltk_data), commit `550b662`).
  - `fetch_data.py` downloads the zip at that commit, checks its SHA-256 against `digests.txt` and extracts it to `data/gutenberg/` (not committed).
  - Eight prompts per book: 40 words that start at a sentence start, spread over each book after its first 5%. The 260 words that follow are the human continuation.
  - The texts are in the public domain in the United States. The data registry records the source as the external reference `nltk-gutenberg-selections` in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data).
- **Models.** Downloaded from the Hugging Face Hub by `generate_runs.py` at the revisions in its `MODELS` table. Each output file records the commit the revision resolved to. Pythia is Apache-2.0; SmolLM2 is Apache-2.0.

## Reproduce

The models run in GitHub Actions on CPU, one job per model (from about 15 minutes for Pythia 70M to several hours for the largest two).

```
python -m pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python fetch_data.py                            # the Gutenberg zip, checked against digests.txt
python -m unittest discover -s tests -v         # the methods on cases with a known answer
python generate_runs.py --model pythia-70m      # one model; writes results/gen_pythia-70m.json.gz
python loops_numbers.py > output.txt            # every number in the article, from results/gen_*.json.gz
python figures.py                               # figures/fig1 to fig6
```

`python generate_runs.py --smoke` runs the whole path with a tiny random model and no download.

The workflow `.github/workflows/decoding-loops.yml` lints, tests and runs the smoke test on every push. The full run (all eight models) starts on manual dispatch or on a push whose commit message contains `[full-run]`; its generations are kept as artifacts for 90 days. A push whose message contains `[analyse]` reruns only the analysis on the generations of the run named in `full_run_id.txt`.

## Files

- **`loops_core.py`**: prompts, the decoding rules, the decoding loop, loop detection, repetition and probability measures, word n-gram chains. numpy only.
- **`lm_backend.py`**: the Hugging Face models behind the same interface, with batched left-padded decoding and teacher-forced scoring.
- **`generate_runs.py`**: one model, 144 prompts, six rules, plus the scores of the human continuations.
- **`loops_numbers.py`**: all numbers, printed in order, and `results/summary.json` for the figures.
- **`figures.py`**: the six figures.
- **`fetch_data.py`**, **`digests.txt`**: the books at a pinned commit.
- **`tests/test_core.py`**: 19 tests. Among them:
  - loop detection on sequences with a known period and start;
  - greedy decoding of a hand-made Markov chain, which must end in its known cycle;
  - the sampling rules against their definitions;
  - word n-gram chains with backoff on a toy text;
  - a counting loop (verse numbers that change while the words repeat), found only once digits are masked.
