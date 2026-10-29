# Regression to the mean in four parts

Code for a four-part series on Medium by Diogo Ribeiro. The thesis of the series: regression to the mean is not a force acting on individuals but a consequence of selecting on a noisy measurement; its size can be computed from the correlation between measurements and the selection rule, and many effects measured after selecting extremes are partly or wholly this. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the history, with the data recomputed** ([`part1/`](part1)). Galton's mid-parents and children (1886), both regressions and the correlation; Galton's family records child by child, with standard errors that respect families; Pearson and Lee's fathers and sons (1903), regression in both directions with equal spreads; and Secrist's design (1933) replicated on major league teams, with the spread across teams season by season.
- **Part 2 — the mechanism.** Forthcoming.
- **Part 3 — real data.** Forthcoming.
- **Part 4 — in data-science practice.** Forthcoming.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # a few seconds; downloads four files on first run
python figures.py
```

A GitHub Actions workflow (`.github/workflows/regression-to-the-mean.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 1

Downloaded on first run from a pinned commit of the Rdatasets repository (https://github.com/vincentarelbundock/Rdatasets), checked against SHA-256 digests and cached in `part1/data/` (not committed):

- `HistData/Galton`, `HistData/GaltonFamilies` and `HistData/PearsonLee` (Friendly, R package HistData).
- `Lahman/Teams`: season records of major league teams (Lahman Baseball Database, CC BY-SA 3.0).

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
