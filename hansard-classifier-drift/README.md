# Classifier drift in Hansard

**How much of a 42-year trend in a text-derived rate is the measuring instrument changing?**

A common way to measure behaviour in parliamentary text is a dictionary: a list of
phrases ("Order. Order.", "resume his seat", "withdraw that remark") counted across the
record and divided by the number of speeches. The resulting rate is then compared across
decades. That comparison assumes the dictionary makes the same mistakes in every decade.
This project tests that assumption for the House of Commons, 1979–2021, and corrects
the rates for the mistakes it finds.

The project is repository-first: the article is written from the numbers this code
produces, and nothing in it comes from elsewhere.

## Question and estimands

The unit is a speech by the Chair (the Speaker, a Deputy Speaker or a committee chair).
The construct is a **disciplinary intervention**: the Chair acting to restrain the
conduct of a Member or of the House (`labels/CODEBOOK.md`). Eras are Speakerships.

For each era *e*:

- **naive rate**: dictionary-tagged Chair speeches per 1,000 Member speeches;
- **corrected rate**: estimated true disciplinary interventions per 1,000 Member speeches;
- **precision** and **recall** of the dictionary in that era.

Because the tagged count equals the true count times recall over precision, the change in
the naive rate between two eras splits exactly into three terms:

log(naive_e / naive_r) = log(rate_e / rate_r) + log(recall_e / recall_r) − log(precision_e / precision_r)

The decomposition against the Weatherill era (the most policed by the naive count) is the
main result. The same correction is applied to the gap between interventions when the
Opposition and when the Government had the floor.

## Design

1. **Roles and order.** Chair speeches are identified by printed title, then by
   parliamentary post. Speeches are put in sitting order from the TheyWorkForYou id
   (sitting, column, paragraph). "Who had the floor" is the last Member to speak that day
   before the Chair; their side (Government or Opposition) comes from the party and the
   date. The 2010–15 coalition counts both partners as Government.
2. **Dictionary.** `src/hansard_drift/dictionary.py`, frozen before any label was seen:
   eight discipline categories and one time-keeping category. Every call of "Order" is
   counted, so "Order. Order." is two.
3. **Validation sample.** Chair speeches stratified by era × dictionary tag × side of the
   floor (36 strata, about 780 speeches), drawn once with a fixed seed and shown to the
   annotator without the tag, the stratum or the era.
4. **Correction.** Each stratum's share of true interventions gets a Beta posterior with a
   Jeffreys prior; stratum totals are known exactly from the full corpus, so every
   quantity is a weighted sum of stratum shares, propagated by Monte Carlo (20,000 draws).
5. **A learned classifier.** A TF-IDF logistic regression trained on the labels is
   evaluated within eras (repeated K-fold) and across eras (leave one era out), to see
   whether learning from data removes the drift or only moves it.
6. **Lexical drift.** Label-free: the terms that most distinguish each era's Chair
   (log-odds with an informative Dirichlet prior, Monroe, Colaresi and Quinn 2008) and the
   Jensen–Shannon divergence between eras.

Two side results come free with stage 1: what the dictionary tag is made of era by era,
and how much of the Chair's discipline a "drop speeches under *k* words" filter removes
(the filter many text pipelines apply before embedding).

## Data

[Hansard Speeches 1979–2021, version 3.1.0](https://doi.org/10.5281/zenodo.4843485)
(Evan Odell), CC BY 4.0, read from its parquet conversion on the Hugging Face hub,
[`biglam/hansard_speech`](https://huggingface.co/datasets/biglam/hansard_speech): 2.69
million rows, 9 May 1979 to 29 April 2021. The commit and the SHA-256 of every file are
recorded in `results/stage1/manifest.json`. The corpus itself is not committed.

## Running it

The whole thing runs on GitHub Actions (`.github/workflows/hansard-classifier-drift-run.yml`):

| Stage | What it does | Trigger |
|---|---|---|
| 1 | download (~1.4 GB), roles, tags, naive rates, lexical drift, validation sample | label a PR `hansard-stage1`, or run the workflow with `stage1` |
| labels | hand-label `labels/sample.csv` | `hansard-drift label --annotator NAME`, locally |
| 2 | corrected estimates, classifier comparison, figures, `results/RESULTS.md` | label a PR `hansard-stage2`, or run the workflow with `stage2` |

Locally:

```bash
pip install -e ".[fetch,dev]"
hansard-drift fetch           # stage 1
hansard-drift stage1
hansard-drift label --annotator diogo
hansard-drift stage2          # stage 2
```

`python -m pytest` runs the tests, including an end-to-end run on a synthetic corpus
whose drift is known (`hansard-drift synth`).

## Outputs

- `results/stage1/`: population counts by stratum, denominators, naive rates by era,
  side and year, dictionary categories, length filter, lexical drift, role audit,
  manifest.
- `labels/`: codebook, blind sample, design, annotator files.
- `results/stage2/`: strata, corrected rates with precision and recall by era,
  decomposition, sides, classifier comparison, summary.
- `results/RESULTS.md`: every number the article uses.
- `figures/`: the article's figures.

## What this does not cover

- Role detection is a classifier too. The sample is drawn from speeches classified as the
  Chair's, so Chair speeches printed under a personal name with no post are missed and
  not measured.
- Hansard is an edited record. A heckle the reporters did not write down cannot be
  recovered by any classifier; the correction is for the dictionary's errors against the
  record, not for the record's errors against the Chamber.
- "Who had the floor" is a proxy: the Chair often addresses the benches, not the Member
  speaking.
- The Hoyle era is 18 months, most of them in the hybrid Covid Parliament.
- Time-keeping is tagged and counted but not validated.

## Licence

Code: Apache-2.0 (the repository's licence). Data: CC BY 4.0, Evan Odell; Hansard is
Parliamentary copyright under the Open Parliament Licence.

Motivated by Trevor Dwyer's [*Order! Order!*](https://medium.com/@trevor.dwyer/order-order-a3cfa542dbaa),
which counts the Chair's interventions in the same corpus with a SQL Server cube.
