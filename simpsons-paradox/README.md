# Simpson's paradox in four parts

Code for a four-part series on Medium by Diogo Ribeiro. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the history, with the tables recomputed** ([`part1/`](part1)). Yule (1903), Simpson (1951), the Berkeley admissions of 1973 (Bickel, Hammel and O'Connell, 1975), the kidney-stone treatments (Charig et al., 1986), and Jeter and Justice. For each table: stratum and aggregate rates, exact tests, the Mantel–Haenszel common odds ratio, the Breslow–Day test, a decomposition of the aggregate gap into allocation and within-stratum rates, and the posterior probability that the reversal holds in the population and not only in the sample.
- **Part 2 — which table is right** ([`part2/`](part2)). The kidney-stone counts under all 25 directed acyclic graphs on three variables: which ones the data reject, and the effect each surviving graph implies. The adjustment formula and inverse probability weighting; total, controlled direct, natural direct and natural indirect effects; the Berkeley departments read as a direct-effect question, with a stylised model of an unmeasured qualification; Berkson's hospital, M-bias and butterfly bias computed exactly in linear Gaussian models; the E-value; and Lord's paradox.
- **Part 3 — real data where it bites** ([`part3/`](part3)). COVID-19 case fatality by age in China and Italy (counts compiled by von Kügelgen, Gresele and Schölkopf, 2021): total, direct and indirect effects, their posterior intervals, the Italian time series, every pair of eleven countries and a stylised testing model. SAT scores and teacher salaries across US states (Guber, 1999). Bill length and depth in the Palmer penguins, by species and by sex. How often reversals arise by chance: random tables, sampling noise, and every pair of regular baseball players over consecutive seasons, season pairs 1990–91 to 2018–19, against three null models.
- **Part 4 — in data-science practice** ([`part4/`](part4)). A/B tests on Hillstrom's e-mail experiment (64,000 customers): what randomisation guarantees about aggregate and segment effects, subgroup heterogeneity, and a ramp-up re-enacted on the real outcomes, pooled against Mantel–Haenszel by period. Decomposition of a business metric into mix and rate on every flight that left New York in 2013 (nycflights13), with Shapley attribution of the mix to airport and departure hour. Fairness metrics on ProPublica's COMPAS file, in aggregate and within successively finer strata, with Chouldechova's identity checked. A scanner that searches a data set for reversals, with an evidence filter, posterior probabilities and a split-sample replication.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # a few seconds; downloads two files on first run
python figures.py
cd ../part2
python causal_numbers.py > output.txt     # a few seconds
python figures.py
cd ../part3
python real_data_numbers.py > output.txt  # under a minute; downloads four files on first run
python figures.py
cd ../part4
python practice_numbers.py > output.txt   # under a minute; downloads three files on first run
python figures.py
```

A GitHub Actions workflow (`.github/workflows/simpsons-paradox.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 1

Downloaded on first run from a pinned commit of the Rdatasets repository (https://github.com/vincentarelbundock/Rdatasets), checked against SHA-256 digests and cached in `part1/data/` (not committed):

- `UCBAdmissions`: the six largest departments of the Berkeley graduate admissions of fall 1973, as distributed with R.
- `Lahman/Batting`: season batting lines from the Lahman Baseball Database (CC BY-SA 3.0), used for the Jeter–Justice example.

The other historical tables are short and are written into the script with the source of every number.

## Data for Part 4

Downloaded on first run from pinned commits on GitHub, checked against SHA-256 digests and cached in `part4/data/` (not committed):

- Kevin Hillstrom's MineThatData E-Mail Analytics and Data Mining Challenge file (2008), from a public copy of the original CSV.
- `nycflights13/flights` (Wickham), via Rdatasets: every flight that departed JFK, LaGuardia or Newark in 2013.
- ProPublica's `compas-scores-two-years.csv` (Larson, Mattu, Kirchner and Angwin, 2016), from the `propublica/compas-analysis` repository. The file contains defendants' names; the script reads only the columns it needs and prints only aggregates.

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
