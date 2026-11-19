# What a flag means when the thing flagged is rare

Companion code for the Medium article on base rates. The thesis is that rare things make weak flags. When few people are guilty, even an accurate detector flags mostly innocent people, so a flag means little until you know the base rate.

## What it finds

All numbers below are from `output.txt`.

**Worked cases.**
- **Mammography** (1% base rate, 90% sensitivity, 9% false positives): 9.2% of positives are right.
- **Face matching at the 2017 Champions League final in Cardiff:** 7.0% of 2,470 alerts were right.
- **OpenAI's withdrawn text classifier:** 8.2% of flags would be right if 3% of essays were AI-written.
- **Card alerts:** at the EEA card-fraud rate (0.015% of payments), a detector that catches 90% with 0.1% false alarms is right on 11.9% of its alerts.

**AI-text detectors on real essays** (Liang et al. 2023 data, seven detectors).
- **False alarms.** On average they flag 5.2% of essays by US eighth-graders and 61.2% of TOEFL essays by non-native writers.
- **Detection.** They flag 48.4% of plain ChatGPT essays, and 11.2% after a self-edit prompt.
- **What a flag is worth.** If 3% of essays are AI-written, 2% to 3% of flags on non-native writers are right, for every detector. On average, a non-native student's essay is flagged more often than an AI-written one.

**A real fraud model** (ULB / Worldline card transactions, logistic regression trained on day 1, tested on day 2).
- **The model.** AUC 0.958.
- **At the data's own fraud rate (0.151%).** Catching 80% of frauds, 17.7% of alerts are fraud.
- **At the EEA rate, ten times rarer.** The same scores give 2.1%, or 47 false alerts per fraud.

**What a flag needs.** If 3% of cases are positive, 90% of flags are right only when the false-alarm rate is below 0.309%. At the EEA card-fraud rate the bound is 0.002%.

## Data

Nothing is committed. `fetch_data.py` downloads both sources at pinned commits and checks every file against `digests.txt`:

- **Detector scores.** The scores of seven detectors (Crossplag, GPTZero, the Hugging Face OpenAI detector, Originality.ai, Quil, Sapling, ZeroGPT) are from [Weixin-Liang/ChatGPT-Detector-Bias](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias), commit `1d05ed8`, MIT licence. A score above 0.5 is a flag, as in the paper.
- **Card transactions.** Two days of transactions by European cardholders, September 2013, released by the ULB Machine Learning Group and Worldline (Dal Pozzolo et al. 2015). The copy is from [stat432/credit-analysis](https://github.com/stat432/credit-analysis), commit `125090b`. Its `creditcard.csv` matches the SHA-256 published for the original (`76274b69…`).

Published figures (Gigerenzer et al. 2007; Eddy 1982; South Wales Police via Wired and The Guardian; OpenAI 2023; Turnitin 2024; EBA and ECB 2024) are entered in `flag_numbers.py` next to their source.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python fetch_data.py                       # both sources, checked against their digests
python flag_numbers.py > output.txt        # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/rare-flags.yml` runs all of this and fails if `output.txt` changes. Model scores are rounded to eight decimals and ranked with a stable sort, so the output does not depend on the processor's arithmetic kernels.

## Limits

- **Small, old detector samples.** The detector results come from 2023 versions of the tools, on 88 to 176 essays per group. Detectors change, and the TOEFL essays are short.
- **The base rate is assumed.** No one knows the share of essays that are AI-written in a given class. The 3% and 11% scenarios come from Turnitin's own aggregate figures, which are themselves detector output.
- **Ten times rarer is a construction.** The fraud data are one processor, two days, in 2013, with anonymised features. The "ten times rarer" figure reweights legitimate transactions and assumes the scores of frauds and of legitimate payments keep their distributions.

## Licence

Code: Apache-2.0.
