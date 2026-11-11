# Actuarial data science: discrimination, calibration and the tariff

Public reproducibility companion for a four-part series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/actuarial-tariff-calibration) repository; this folder holds the code, the printed numbers and the figures. The thesis of the series: **a good Gini is not a good tariff.** The Gini index of an insurance score depends only on the order in which the score ranks policyholders. A tariff is a price, and a price has to collect, from every group of policyholders it treats alike, about what that group costs. Two tariffs with the same ranking, and so the same Gini, can charge the same people very different amounts. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — how motor tariffs were judged, from minimum bias to the Gini index** ([`part1/`](part1)). The Canadian experience of 1957–58 that Bailey and Simon (1960) used to state their criteria for a set of rates, transcribed from their Tables A and D: their four methods recomputed and tested for balance, average departure and chi-square against their Table E; Bailey's (1963) minimum-bias iteration against Berquist's published calculations, and its fixed point, which is the Poisson maximum-likelihood fit (Jung 1968); the same tariffs scored by an exposure-weighted Gini index. The Swedish third-party motor experience of 1977 (2,182 tariff cells): the marginal-totals tariff, which is the Poisson GLM; nested tariffs by Gini and deviance; the customary one-way tariff; and a family of tariffs with exactly the same ranking (powers of the GLM frequency, and level errors) whose Gini is identical while their deviance and their balance by rating factor and by premium level are not.
- Parts 2–4 (mechanism; an audit of GLM and boosted models on freMTPL2; pricing against a competitor) will be added here.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # a few seconds after the first download
python figures.py
```

A GitHub Actions workflow (`.github/workflows/actuarial-tariff-calibration.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data

- **Canada, policy years 1957 and 1958** (Part 1). Earned car-years by five classes and four merit-rating groups (Bailey and Simon 1960, Table A, in thousands) and the relative loss ratios at 1B rates (Table D, "actual"), transcribed into `part1/history_numbers.py` together with the published relativities (Table C), tests (Table E) and Berquist's minimum-bias calculations (discussion of Bailey 1963, Tables 1 and 2). The scan of the 1960 paper is partly illegible in places; the values read for Method 4 and the one illegible loss ratio (class 5, merit B, recovered from Table D) are flagged in the script.
- **Sweden, 1977** (Part 1). Third-party motor insurance compiled by the Swedish Committee on the Analysis of Risk Premium in Motor Insurance (Hallin and Ingenbleek 1983), `swautoins` in [CASdatasets](https://github.com/dutangc/CASdatasets), downloaded by `data.py` from a pinned commit, checked against its SHA-256 and kept in `.cache/` (not committed). CASdatasets is distributed under the GPL (>= 2).

## References

Bailey, R. A. (1963). Insurance rates with minimum bias. *Proceedings of the Casualty Actuarial Society*, 50, 4–11; discussion by J. R. Berquist, 11–13.

Bailey, R. A., & Simon, L. J. (1960). Two studies in automobile insurance ratemaking. *Proceedings of the Casualty Actuarial Society*, 47, 1–19; also *ASTIN Bulletin*, 1(4), 192–217.

Frees, E. W., Meyers, G., & Cummings, A. D. (2011). Summarizing insurance scores using a Gini index. *Journal of the American Statistical Association*, 106(495), 1085–1098.

Hallin, M., & Ingenbleek, J.-F. (1983). The Swedish automobile portfolio in 1977: a statistical study. *Scandinavian Actuarial Journal*, 1983, 49–64.

Jung, J. (1968). On automobile insurance ratemaking. *ASTIN Bulletin*, 5(1), 41–48.

Wüthrich, M. V. (2023). Model selection with Gini indices under auto-calibration. *European Actuarial Journal*, 13(1), 469–477.
