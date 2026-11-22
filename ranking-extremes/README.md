# Small samples at both ends of a country ranking

Companion code for the Medium article on Euro Top Tech's country table of software engineers' happiness and satisfaction with tax-paid services: why the countries with the fewest reports take the top and bottom places, and what the table looks like once each country's sample size is taken into account.

## What it finds

All numbers below are from `output.txt`.

**Who is at the ends.** 12 of the 29 countries rest on 5 to 9 rated reports; together they hold 84 of 744 reports (11%). They take 8 of the 12 places at the two ends of the two rankings: Luxembourg (5 reports) is first on services and Greece (8) last; Kosovo and Lithuania (5 each) are first and second on happiness, and Slovakia, Finland and Slovenia (6 to 9) are the bottom three. If size did not matter, 2.5 of every six end places would go to them.

**How much is noise.** A hierarchical normal model with unknown within-country and between-country spreads, fitted on a grid, gives a within-country SD of single ratings of 2.50 for services (90% interval 0.80 to 3.82; the page's share of 8+ ratings implies about 2.8) and 1.80 for happiness, and a between-country SD of true means of 0.83 for services and 0.25 for happiness. Sampling noise is about 44% of the variance of the published services means and 83% of the happiness means.

**Funnel plots.** Of the 12 small countries, one (Greece, services) lies outside the 95% limits of what noise alone would give; none on happiness. Seven countries are outside on services and three on happiness; seven of these ten have 30 or more reports. Luxembourg's lead is 1.41 standard errors; Switzerland's is outside the 99.8% limits.

**Shrinkage and rank intervals.** Luxembourg moves from 1st to 5th on services (90% rank interval 1 to 23) and Kosovo from 1st to 4th on happiness (1 to 26). Switzerland is the only country whose services interval lies inside the top 10; on happiness no interval lies inside the top or the bottom 10, and the most likely last country is Germany, the one with the most reports (112).

**Simulation.** In 20,000 tables drawn from the fitted model with the same sample sizes, the published first place goes to a small country 62% of the time on services and 73% on happiness, against 42% for the truly best country. The published first is the true first in 37% of tables on services and 16% on happiness.

**Validity check.** For the 20 countries also in the OECD Trust Survey 2023, the engineers' services ratings correlate 0.83 with the share of the population satisfied with healthcare, education and administrative services (0.89 with healthcare alone). In the hierarchical model the composite tracks most of the real between-country variation.

**On ratings like these.** On 300 simulated tables of skewed, whole-number 0-10 ratings, the 90% intervals for country means cover 86.5% and 88.3% of true means and the 90% rank intervals 93.0% and 92.4% of true ranks; shrinkage picks the true first more often than the raw table (34% against 27%, 20% against 12%).

## Data

- **Euro Top Tech, "Happiness and taxes in Europe"** (https://www.eurotoptech.com/research/happiness-taxes-europe), read on 22 November 2026. The page recomputes from a live database and offers no download, so its 29-row country table is transcribed in `snapshot.py`; the digest of its canonical text is recorded in the data registry (`DiogoRibeiro7/data`, `external/eurotoptech-happiness-taxes-europe`) and checked by the tests.
- **OECD, Government at a Glance 2025**, dataflow `OECD.GOV.GIP,DSD_GOV_INT@DF_GOV_SPS_2025`: percent satisfied with healthcare, education and administrative services, 2023. `fetch_data.py` downloads it into `.cache/` and checks it against a SHA-256 digest (registry record `external/oecd-government-at-a-glance-satisfaction-2025`).

No data files are committed: the OECD table is rebuilt at every run, and the analysis tables for the figures go to `.cache/results/`.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # methods against known answers, snapshot against its digest
python fetch_data.py                         # OECD table, pinned by digest
python ranking_numbers.py > output.txt       # every number in the article; about a minute
python figures.py                            # figures/
```

The workflow `.github/workflows/ranking-extremes.yml` runs all of this and fails if `output.txt` changes.

## Files

- **`snapshot.py`**: the transcribed country table, the page's totals and quoted correlations, and the canonical digest.
- **`ranking_core.py`**: the model (closed-form evidence on a grid, joint posterior draws), shrinkage weights, ranks and rank intervals, funnel limits, counts of small groups at the ends, simulation of tables, and a generator of skewed whole-number ratings.
- **`ranking_numbers.py`**: all numbers, printed in order.
- **`figures.py`**: the five figures.
- **`fetch_data.py`**: download and check of the OECD table.
- **`tests/test_core.py`**: 20 checks.

## Licence

Code: Apache-2.0. Data: as published by Euro Top Tech and the OECD; cite both.

Contact: dfr@esmad.ipp.pt
