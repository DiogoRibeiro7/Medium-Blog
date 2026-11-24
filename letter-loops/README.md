# Greedy letter chains in 36 languages: where "follow the likeliest next letter" ends, and why

Companion code for a Medium article about a popular visualisation: start at any letter, move to the letter most likely to follow it, repeat. With Peter Norvig's English letter pairs, every letter ends in the loop e → r → e. This folder repeats the experiment with a second corpus and 35 more languages, then changes the choices made before any counting. Thesis: **the loop is decided by the rule and by what counts as a letter, a word and a corpus, not by the language.**

- **The rule.** Taking the single most likely successor turns a table of letter pairs into a map from each letter to one letter. Any such map on a finite set ends in loops.
- **The choices.**
  - *Alphabet:* the standard letters of each language (CLDR), or the same letters with accents and final forms folded.
  - *Word boundaries:* pairs inside words only, as in Norvig's table, or the end of a word as a symbol.
  - *Weighting:* running text, or a dictionary in which each word counts once.

## What it finds

All numbers below are from `output.txt`.

**English, two corpora.**
- Norvig's table settles the arrow of 15 letters; wordfreq agrees on all 15.
- The 50 commonest pairs differ by a typical factor of 1.15 between the two corpora.
- In wordfreq, too, every letter ends in e → r. Q → u has 97.6% (99.1% in the post); x → p has 26.3% (26.4%).

**How unusual one loop is.**
- With pairs inside words, 24 of 36 languages have a single loop.
- Uniformly random arrows would give about 7.6 such languages.
- Each language's own arrows shuffled (incoming arrows per letter kept) give 14.8; shuffled with no letter pointing at itself, 28.0.
- The 12 exceptions:
  - 6 have two competing loops (Spanish ent and ar, Russian ов and ан);
  - 4 have a doubled letter that points at itself (Swedish tt and ll, Dutch oo, Finnish ll and ää, Icelandic nn and pp);
  - 2 have a letter that never has a successor.

**Add the end of a word.**
- English loops on "the", French on "de", Italian on "di", Ukrainian on "в", Urdu on "کے".
- The word the chain writes is its language's most frequent word in 10 of 36 languages and in the top 10 in 16.
- In 10 it ranks below 1,000. Finnish gets "ka" (rank 10,075), because the greedy rule picks one letter at a time, not the likeliest word.
- 31 of 36 languages have a single loop.

**Change the alphabet or the weighting.**
- Folding accents and final forms changes the loops in 16 of 36 languages inside words and in 9 with boundaries.
- Counting a dictionary changes them in 26 and in 32. English with boundaries then has two loops, "s" and e → r.
- Across the eight combinations, every language has at least 3 different sets of loops; Dutch has 8.

**A second corpus.** Every count is multiplied by noise of the size seen between Norvig and wordfreq.
- The loops survive in a median 54% of draws inside words and 52% with boundaries.
- English: e → r survives in 81% (the usual change is an extra loop ll), "the" in 71% ("t" 18%, "an" 7%).

**Sampling instead of the maximum.** Successors are drawn with probabilities proportional to count^(1/T).
- At T = 1 the English chain spends 40% of its time in "the"; at T = 0.1, 96%.
- The extra loops of Dutch, Swedish, Icelandic, Finnish and Hungarian drain away as T falls: the word loop takes all the time.

## Data

- **Word frequencies.** wordfreq 3.1.1 by Robyn Speer (Speer 2022, doi:10.5281/zenodo.7199437), installed from PyPI; its word lists ship inside the package.
  - `fetch_data.py` checks the version and the SHA-256 of all 36 lists against `digests.txt`.
  - 18 languages have the large list and 18 the small one.
  - wordfreq data are CC BY-SA 4.0. They are built from Wikipedia, OpenSubtitles 2018, Google Books Ngrams, NewsCrawl and GlobalVoices, OSCAR web text, the Leeds Internet Corpus, ParaCrawl, Twitter, Reddit and the SUBTLEX lists of Marc Brysbaert and colleagues, which are freely available (http://crr.ugent.be/programs-data/subtitle-frequencies).
- **Alphabets.** The main exemplar characters of each locale in the Unicode CLDR 49, from the npm package `cldr-misc-full` 49.0.0.
  - `fetch_data.py` downloads the tarball, checks its SHA-256 and writes one `characters.json` per locale to `data/cldr/` (not committed).
  - Serbo-Croatian uses the Croatian locale.
- **Norvig's table.** The top 50 English bigrams of Norvig (2013), counted in the Google Books 1-grams, are entered in `letters_core.py` from https://norvig.com/mayzner.html.

wordfreq case-folds its words, which turns German ß into ss and Greek final ς into σ. Those letters never occur and leave the alphabet.

The data registry records both sources as external references: `wordfreq-3-1-1-word-frequencies` and `unicode-cldr-49-exemplar-characters` in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data).

## Reproduce

```
python -m pip install -r requirements.txt
python fetch_data.py                       # wordfreq lists and the CLDR tarball, checked against digests.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python letters_numbers.py > output.txt     # every number in the article; about two minutes
python figures.py                          # figures/fig1 to fig5
```

`figures.py` needs fonts with Hebrew and Arabic letters: on Ubuntu, `sudo apt-get install fonts-noto-core`. The workflow `.github/workflows/letter-loops.yml` runs all of this and fails if `output.txt` changes.

## Files

- **`letters_core.py`**: alphabets, folding, counting, the greedy map and its loops, the baselines, noise and temperature.
- **`letters_numbers.py`**: all numbers, printed in order, and the arrays the figures use (in `results/`, not committed).
- **`figures.py`**: the five figures.
- **`fetch_data.py`**, **`digests.txt`**: the inputs at pinned versions.
- **`tests/test_core.py`**: 21 tests. Among them:
  - the exact one-loop probability of a random map against enumeration of all maps up to n = 6;
  - loops, dead ends and depths on a hand-made map;
  - counting and boundaries on a toy word list;
  - the Norvig comparison on Norvig's own counts;
  - the stationary distribution of a two-state chain.
