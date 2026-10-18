# Model collapse measured on six translations of Genesis

Companion code for the Medium article *Model Collapse Measured on Six Translations of Genesis*.

A bigram language model is fitted to the Book of Genesis in six languages (the NLTK Genesis corpus: English KJV, Finnish, French, German, Portuguese, Swedish), then retrained for fifty generations on text it wrote itself. The experiment varies what the next model is trained on and how the previous model samples, and every closed-form result quoted in the article is checked against Monte Carlo.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Python 3.13. The first run downloads the corpus (about 0.5 MB) into `data/`. The whole pipeline takes roughly ten minutes on a laptop with two cores. It writes:

- `results/summary.txt`: every number quoted in the article
- `results/theory_checks.json`: closed forms against simulation
- `results/samples.txt`: text written by the models at selected generations
- `figures/fig1_…png` to `figures/fig5_…png`: the article's figures
- `results/bigram_loop.json`: per-generation metrics for every run (not committed, about 3 MB)

All randomness is seeded, so a rerun with the pinned versions reproduces the committed outputs; the workflow in `.github/workflows/model-collapse-genesis.yml` checks this on every change to the folder.

## Design

- **Tokens.** Lower-cased runs of letters plus the punctuation marks `. , ; : ! ?`. Words are tagged with their language; punctuation is shared, so a punctuation mark is the only point where the model can switch language and the language of every generated word is known exactly.
- **Model.** Maximum-likelihood bigram model stored as sparse pair counts (`bigram.py`). Sampling runs all documents in parallel with one `searchsorted` per position.
- **Loop.** Each generation writes 1,088 documents of 200 tokens (the size of the training set) and the next model is fitted to them. Ten seeds, fifty generations.
- **Data policies.** `replace` (generated text only), `accumulate` (original corpus plus all generated text), `anchor` (90% generated text plus a fresh 10% slice of the original corpus).
- **Decoding.** Temperature 1, temperature 0.8, nucleus sampling with p = 0.95.
- **Evaluation.** A random 10% of 200-token blocks is held out. Metrics: word types the model can produce (overall and by language), share of held-out word tokens it can no longer produce, Gini–Simpson and Shannon entropy of the word distribution, per-context Gini–Simpson, entropy rate, held-out cross-entropy, language shares of the generated words.

## Closed forms checked in `theory.py`

- Replace: E[D(p_t)] = D(p_0)(1 − 1/N)^t for the Gini–Simpson index D (Wright–Fisher heterozygosity decay).
- One round: expected surviving types Σ[1 − (1 − p_i)^N]; expected mass lost Σ p_i(1 − p_i)^N, which matches the expected singleton share of the new sample (Good–Turing).
- Accumulate: E[D(p_∞)]/D(p_0) = ∏_{k≥2}(1 − 1/(Nk²)) = sin(π/√N) / ((π/√N)(1 − 1/N)).
- Anchor with fresh share α: stationary E[D]/D(p_0) = 1 − 1/(Nα(2 − α) + 1 − α).
- Temperature without sampling noise: p_t ∝ p_0^(τ^(−t)).
- Iterated nucleus sampling stops with at most ⌈1/(1 − p)⌉ − 1 surviving words.
- At a context shared by several languages, temperature τ moves the mass of a language to m^β exp((1 − β) H_β(q)), β = 1/τ, with H_β the Rényi entropy of its continuation distribution.

## Data

The corpus is downloaded from the NLTK data repository (`corpora/genesis.zip`). The World English Bible and Lolcat files in the package are not used. The Finnish file contains the translators' explanatory notes, which are kept as distributed.
