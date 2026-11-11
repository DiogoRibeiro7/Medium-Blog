# Actuarial data science: discrimination, calibration and the tariff

Public reproducibility companion for a four-part series, now complete, on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/actuarial-tariff-calibration) repository; this folder holds the code, the printed numbers and the figures. The thesis of the series: **a good Gini is not a good tariff.** The Gini index of an insurance score depends only on the order in which the score ranks policyholders. A tariff is a price, and a price has to collect, from every group of policyholders it treats alike, about what that group costs. Two tariffs with the same ranking, and so the same Gini, can charge the same people very different amounts. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — how motor tariffs were judged, from minimum bias to the Gini index** ([`part1/`](part1)). The Canadian experience of 1957–58 that Bailey and Simon (1960) used to state their criteria for a set of rates, transcribed from their Tables A and D: their four methods recomputed and tested for balance, average departure and chi-square against their Table E; Bailey's (1963) minimum-bias iteration against Berquist's published calculations, and its fixed point, which is the Poisson maximum-likelihood fit (Jung 1968); the same tariffs scored by an exposure-weighted Gini index. The Swedish third-party motor experience of 1977 (2,182 tariff cells): the marginal-totals tariff, which is the Poisson GLM; nested tariffs by Gini and deviance; the customary one-way tariff; and a family of tariffs with exactly the same ranking (powers of the GLM frequency, and level errors) whose Gini is identical while their deviance and their balance by rating factor and by premium level are not.
- **Part 2 — the mechanism, in a simulation with a known true premium** ([`part2/`](part2)). A synthetic motor portfolio with six rating factors, unobserved heterogeneity and exposure that depends on age; population quantities computed without sampling noise from the known truth. Powers and levels of the true frequency (one Gini, many prices; the premium's own Lorenz curve against the concentration curve of the claims); the Murphy decomposition of the Poisson deviance into miscalibration, discrimination and uncertainty by isotonic regression; in-sample isotonic recalibration as the greatest convex minorant of the ordered Lorenz curve, and discrimination as a Kullback–Leibler divergence over its steps; four fitted models (a banded Poisson GLM, Poisson boosting, a claim classifier, boosting without the exposure offset) on a test sample and in the population; isotonic recalibration on a separate sample, with and without a minimum exposure per step; and model choice by Gini and by deviance on 400 test samples of 170,000 policies.
- **Part 3 — a calibration audit of GLM and boosted frequency models on freMTPL2** ([`part3/`](part3)). The French motor third-party liability portfolio (677,991 policies): a data audit (exposure above one year, a cluster of policies with implausible claim counts, the fixed amounts of the IRSA-IDA convention, the concentration of payments in a few large claims); two Poisson GLMs (bonus-malus linear, bonus-malus in bands) and Poisson boosting on a 60/20/20 split; the test-sample Gini, Murphy decomposition, balance and the premium's own Gini; balance by segment with Poisson intervals and by premium decile; a paired bootstrap and ten repeated splits; the same rankings scored on claim counts, capped losses and total losses; and recalibration on the calibration sample (global factor, spline on the log score, isotonic steps with a cross-validated minimum exposure) judged on the test sample.
- **Part 4 — two insurers pricing the same portfolio** ([`part4/`](part4)). The models of Part 3 quoting the freMTPL2 test sample against each other, with a logit choice rule whose price sensitivity runs from none to cheapest-quote-wins: double lift charts; ordered Lorenz curves with relativities and the relativity Gini matrix of Frees, Meyers and Cummings (2014), with the mini-max choice; one-period shares and loss ratios for pairs of tariffs, including powers and level shifts of the boosted rate that keep its Gini; where the business goes by bonus-malus band; twin models fitted on two halves of the training sample; repeated rate reviews in which each insurer rescales its level by its own loss ratio; a paired bootstrap, ten refitted splits and losses capped at 100,000 instead of claim counts.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # a few seconds after the first download
python figures.py
cd ../part2
python mechanism_numbers.py > output.txt  # about three minutes, single-threaded; writes figure_data.json
python figures.py
cd ../part3
python audit_numbers.py > output.txt      # about four minutes, single-threaded; writes figure_data.json
python figures.py
cd ../part4
python market_numbers.py > output.txt     # about four minutes; imports the data and models of part3
python figures.py
cd ..
python -m unittest discover -s tests      # tests of the shared tools in tariffcal.py
```

`tariffcal.py` holds the tools the parts share: the exposure-weighted Gini and ordered Lorenz curve, the Gini of the premium itself, the Poisson deviance, its Murphy decomposition by isotonic regression, a step-function isotonic recalibration with a minimum exposure per step, and the greatest convex minorant of a curve.

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to reproduce the committed output exactly; with more threads the boosting and the linear algebra can differ in the last printed digit.

A GitHub Actions workflow (`.github/workflows/actuarial-tariff-calibration.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data

- **Canada, policy years 1957 and 1958** (Part 1). Earned car-years by five classes and four merit-rating groups (Bailey and Simon 1960, Table A, in thousands) and the relative loss ratios at 1B rates (Table D, "actual"), transcribed into `part1/history_numbers.py` together with the published relativities (Table C), tests (Table E) and Berquist's minimum-bias calculations (discussion of Bailey 1963, Tables 1 and 2). The scan of the 1960 paper is partly illegible in places; the values read for Method 4 and the one illegible loss ratio (class 5, merit B, recovered from Table D) are flagged in the script.
- **Simulated portfolio** (Part 2). Generated by `part2/mechanism_numbers.py` from a fixed seed; nothing is downloaded.
- **Sweden, 1977** (Part 1). Third-party motor insurance compiled by the Swedish Committee on the Analysis of Risk Premium in Motor Insurance (Hallin and Ingenbleek 1983), `swautoins` in [CASdatasets](https://github.com/dutangc/CASdatasets), downloaded by `data.py` from a pinned commit, checked against its SHA-256 and kept in `.cache/` (not committed). CASdatasets is distributed under the GPL (>= 2).
- **France, freMTPL2** (Parts 3 and 4). Motor third-party liability policies of a French insurer observed over one year, `freMTPL2freq` and `freMTPL2sev` in CASdatasets, from the same pinned commit and checked the same way. Exposure is capped at one year and claim counts at four, the usual cleaning for this data (Noll, Salzmann and Wüthrich 2020); claim amounts are joined to policies by `IDpol`. In the pinned version the number of claim amounts equals the claim count on every policy.

## References

Bailey, R. A. (1963). Insurance rates with minimum bias. *Proceedings of the Casualty Actuarial Society*, 50, 4–11; discussion by J. R. Berquist, 11–13.

Bailey, R. A., & Simon, L. J. (1960). Two studies in automobile insurance ratemaking. *Proceedings of the Casualty Actuarial Society*, 47, 1–19; also *ASTIN Bulletin*, 1(4), 192–217.

Denuit, M., Charpentier, A., & Trufin, J. (2021). Autocalibration and Tweedie-dominance for insurance pricing with machine learning. *Insurance: Mathematics and Economics*, 101, 485–497.

Dimitriadis, T., Gneiting, T., & Jordan, A. I. (2021). Stable reliability diagrams for probabilistic classifiers. *Proceedings of the National Academy of Sciences*, 118(8), e2016191118.

Frees, E. W., Meyers, G., & Cummings, A. D. (2011). Summarizing insurance scores using a Gini index. *Journal of the American Statistical Association*, 106(495), 1085–1098.

Frees, E. W., Meyers, G., & Cummings, A. D. (2014). Insurance ratemaking and a Gini index. *Journal of Risk and Insurance*, 81(2), 335–366.

Goldburd, M., Khare, A., Tevet, D., & Guller, D. (2020). *Generalized Linear Models for Insurance Rating* (2nd ed.). CAS Monograph Series No. 5. Arlington, VA: Casualty Actuarial Society.

Hallin, M., & Ingenbleek, J.-F. (1983). The Swedish automobile portfolio in 1977: a statistical study. *Scandinavian Actuarial Journal*, 1983, 49–64.

Jung, J. (1968). On automobile insurance ratemaking. *ASTIN Bulletin*, 5(1), 41–48.

Noll, A., Salzmann, R., & Wüthrich, M. V. (2020). Case study: French motor third-party liability claims. SSRN 3164764.

Wüthrich, M. V. (2023). Model selection with Gini indices under auto-calibration. *European Actuarial Journal*, 13(1), 469–477.
