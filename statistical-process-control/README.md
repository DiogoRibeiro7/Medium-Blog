# Statistical process control in four parts

Public reproducibility companion for a four-part series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/statistical-process-control) repository (Part 1: [`part1/article.md`](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/statistical-process-control/part1/article.md)); this folder holds the code, the printed numbers and the figures. The thesis of the series: the in-control average run length of 370 belongs to a three-sigma Shewhart chart whose mean and standard deviation are *known*. Every chart in use has limits estimated from a finite Phase I sample, so its actual in-control ARL (the conditional ARL) is a random variable fixed by those estimates; its distribution is skewed, a majority of charts built the textbook way false-alarm more often than advertised, and the 370 is at best an average over a population of charts nobody owns. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — where the 370 comes from, with the numbers recomputed** ([`part1/`](part1)). The normal tail against the Chebyshev and unimodal bounds Shewhart could appeal to; the geometric run length of a chart with known parameters; exact average run lengths of the Western Electric zone tests by Markov chain (checked by simulation) and Nelson's eight tests by simulation; the British action and warning limits; three-sigma limits under eight distributions for individual values and means of five; and a first look at estimated limits — the conditional in-control ARL of X-bar and individuals charts built from 20 to 1000 Phase I subgroups, the widening that restores the average false-alarm probability, and the correlation that shared estimates put between alarms.
- **Part 2 — the mechanism, from one chart to a production line** ([`part2/`](part2)). The conditional ARL of one chart derived exactly by conditioning, its sensitivity to the estimate of sigma, three estimators compared, and the condition under which its mean is infinite; machines charted in parallel, with Sidak limits and each machine's own Phase I or a pooled one; short runs after a changeover, fixed limits against a self-starting (Q) chart and the shift it absorbs, and sigma pooled across products; stations in series, with a cause-selecting chart whose regression is estimated in Phase I; and Hotelling's T^2 with an estimated mean vector and covariance matrix, by Imhof's method, for 2 to 50 sensors.
- **Part 3 — real data.** Forthcoming: a public multistage continuous-flow production line and the UCI SECOM semiconductor line.
- **Part 4 — practice.** Forthcoming: a line-level false-alarm budget, limits with a guaranteed in-control performance, and alert thresholds in ML.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # about a minute; no downloads
python figures.py
cd ../part2
python mechanism_numbers.py > output.txt  # about four minutes; no downloads; writes figure_data.json
python figures.py
```

A GitHub Actions workflow (`.github/workflows/statistical-process-control.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 1

None. Part 1 recomputes published numbers from their definitions (normal and other distribution functions, Markov chains, and seeded simulation of Phase I samples).

## Data for Part 2

None. Every chart is built from a simulated in-control Phase I sample and its conditional false-alarm probability is computed exactly given its estimates.

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
