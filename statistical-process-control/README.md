# Statistical process control in four parts

Code for a four-part series on Medium by Diogo Ribeiro. The thesis of the series: the in-control average run length of 370 belongs to a three-sigma Shewhart chart whose mean and standard deviation are *known*. Every chart in use has limits estimated from a finite Phase I sample, so its actual in-control ARL (the conditional ARL) is a random variable fixed by those estimates; its distribution is skewed, a majority of charts built the textbook way false-alarm more often than advertised, and the 370 is at best an average over a population of charts nobody owns. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — where the 370 comes from, with the numbers recomputed** ([`part1/`](part1)). The normal tail against the Chebyshev and unimodal bounds Shewhart could appeal to; the geometric run length of a chart with known parameters; exact average run lengths of the Western Electric zone tests by Markov chain (checked by simulation) and Nelson's eight tests by simulation; the British action and warning limits; three-sigma limits under eight distributions for individual values and means of five; and a first look at estimated limits — the conditional in-control ARL of X-bar and individuals charts built from 20 to 1000 Phase I subgroups, the widening that restores the average false-alarm probability, and the correlation that shared estimates put between alarms.
- **Part 2 — the mechanism.** Forthcoming.
- **Part 3 — real data.** Forthcoming.
- **Part 4 — in data-science practice.** Forthcoming.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # about a minute; no downloads
python figures.py
```

A GitHub Actions workflow (`.github/workflows/statistical-process-control.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 1

None. Part 1 recomputes published numbers from their definitions (normal and other distribution functions, Markov chains, and seeded simulation of Phase I samples).

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
