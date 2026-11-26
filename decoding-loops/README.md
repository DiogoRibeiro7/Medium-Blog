# Repetition loops in eight open language models: model size against the decoding rule

Companion code for a Medium article about the loops a language model falls into when it always writes its most likely next token. The same 144 prompts, taken from 18 public-domain books, are continued for 256 tokens by eight open models, under six decoding rules. Thesis: **the rule makes the loop.** Greedy decoding loops at every model size in the experiment; sampling removes the loop, and a bigger model does not.

- **The models.** Two families, each trained on one dataset at several sizes:
  - Pythia 70M, 160M, 410M, 1B and 1.4B (EleutherAI), final checkpoint `step143000`. All five saw the same data in the same order.
  - SmolLM2 135M, 360M and 1.7B (Hugging Face), revision `main`.
- **The rules.** Greedy (always the most likely token); temperature sampling at *T* = 0.25, 0.5, 0.75 and 1; nucleus sampling with *p* = 0.9.
- **A loop.** A continuation ends in a block of at most 64 tokens that repeats back to back at least three times and covers at least 24 tokens.
- **The human baseline.** The 256 tokens that follow each prompt in the book, scored by each model token by token.

## What it finds

All numbers below are from `output.txt`.

**The rule against the size.** Share of the 144 continuations that end in a loop:
- Greedy: 70% to 96% across the eight models (1,012 of 1,152 continuations in all). The smallest share is SmolLM2 1.7B's.
- Temperature 0.25: 65% to 90%. Temperature 0.5: 23% to 65%. Temperature 0.75: 2% to 10%.
- Temperature 1: 0 of 1,152. Nucleus sampling (*p* = 0.9): 8 of 1,152. The books themselves: 0.
- Within Pythia, where only the size changes, greedy goes from 96% (70M) to 86% (1B and 1.4B). Within SmolLM2 it goes from 88% (135M and 360M) to 70% (1.7B).
- Every prompt loops under at least one model; 70 of 144 loop under all eight.

**Where the loops start.** Greedy loops start at a median of 18 tokens (quartiles 6 and 38); 83% start within 50 tokens.
- The median moves from 11 tokens (Pythia 70M) to 26 (Pythia 1.4B) and 40 (SmolLM2 1.7B).
- The repeating block grows from a median of 10 tokens to 21 and 23. Blocks of four tokens or fewer fall from 18 (Pythia 70M) to 3 (Pythia 1.4B).
- Inside a loop the probability of the chosen token climbs with every repetition: 0.30 to 0.42 before the loop, 0.37 to 0.51 in the first cycle, 0.87 to 0.95 in the sixth.

**What greedy writes when it does not loop.**
- The Bible is the exception among the books: 21 of 64 greedy continuations end in a token loop. With every digit masked, 39 do: the model repeats a verse while its number counts on (31:14, 31:15, ...).
- SmolLM2 1.7B recites the King James Bible for 32 tokens or more in 6 of 8 Bible prompts, four of them for (nearly) all 256 tokens. No other model reproduces 32 tokens of any book.
- Of the 1,152 greedy continuations, 62 neither loop, count, recite nor end in a longer block repeated twice. Those still repeat 30% to 81% of their 4-grams; the books repeat 1%.

**Probability.** Greedy text averages −0.25 to −0.37 nats per token under the model that wrote it. The books score −2.33 to −4.28, close to sampling at temperature 1 (−3.18 to −4.47).

**Which words.** Function words (the 100 commonest words of the 18 books) make up 64% to 69% of the loop blocks and 61% to 69% of the greedy text before the loop, against 51% of the books and 48% to 51% of samples at temperature 1. The words most over-represented in loop blocks come from stock dialogue: glad (×75), afraid, going, am, sorry.

**Finite memory.** Greedy word n-gram chains on the same books (memory of 0 to 3 words) loop on all 144 prompts; the loop starts at a median of 0, 0, 4 and 31 words.

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
