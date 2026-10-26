# Redials in a real call centre: the interval-level estimator against caller IDs

Companion code for a Medium article that takes the redial estimator of the two earlier articles in this repository to real data. The data are the call-by-call records of the "Anonymous Bank" call centre for 1999, collected by Ilan Guedj and Avi Mandelbaum and published by the Technion SEE Center. The question: when the only data are a half-hourly table of offered and abandoned calls, can the probability that a caller who hangs up calls again be estimated from it? The customer IDs in the records give an answer to check against.

The queue simulator is the one in [`../mechanistic-vs-data-driven-generators`](../mechanistic-vs-data-driven-generators) (package `gentradeoffs`), which the scripts put on the import path.

## Data

The raw data are not stored here. Download the archive (about 11 MB) and check it:

```bash
python fetch.py        # writes data/AnonymousBank.rar and checks its SHA-256
```

Reading the archive needs `bsdtar` (package `libarchive-tools` on Debian and Ubuntu). A folder of extracted monthly files also works: put it in `data/raw/` or point `ANONBANK_DATA` at it.

The data are free to use. The documentation asks users to notify Avi Mandelbaum (avim@ie.technion.ac.il) and to acknowledge the source in any work that uses them.

The half-hourly table built from the records, `data/halfhour.csv.gz`, is committed, so everything except R1 and R2 runs without the raw data.

## What is in `src/anonbank`

- `data.py`: reads the twelve monthly files, parses the clock times, drops 12 malformed rows and flags queued, abandoned and served calls.
- `table.py`: the half-hour table a contact-centre platform would export (working days Sunday to Thursday, 07:00 to 24:00).
- `ids.py`: what the customer IDs say: calling again after hanging up and after service, by the ID's annual volume, the delay to the next call, and a weighting for the callbacks the IDs cannot see (the system recorded the ID only for calls that queued).
- `stats.py`: the interval statistics: lag and lead slopes of surplus calls on abandonments, the controlled slope, autocorrelation and dispersion of the surplus, with a day-cluster bootstrap.
- `sim.py`: a simulator of this centre, with the real days' staffing replayed, an optional AR(1) clustering of demand within the day, and redials.
- `indirect.py`: indirect inference for the redial probability from one statistic, and jointly with the clustering from five.

## Experiments

| Script | Question |
|--------|----------|
| `experiments/r1_data.py` | The records, the cleaning and the half-hour table. |
| `experiments/r2_ids.py` | Calling again after hanging up versus after service, from the IDs. |
| `experiments/r3_slopes.py` | Lag and lead slopes for all offered calls, for identified redials and for the rest. |
| `experiments/r4_simcheck.py` | Each estimator on simulated histories with a known redial probability, with and without clustered demand. |
| `experiments/r5_estimate.py` | Each estimator on the real table, under four assumptions about the redial delay. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python fetch.py
cd experiments
python run_all.py          # about six minutes on two cores
python make_figures.py
```

`run_all.py --quick` runs everything at small sizes; CI does that on every change, without the raw data. Every number quoted in the article is in `results/summary.txt`; raw results are in `results/*.json`. All randomness flows from one master seed through `numpy.random.SeedSequence`, and a full run reproduces the committed results exactly.

## Caveats

- The customer ID was recorded only for calls that waited in the queue, so a callback answered at once is invisible. The weighting in `ids.py` corrects for this under the assumption that, within a half hour, identified customers queue as often as everyone else.
- IDs that place hundreds of calls a year behave like shared lines; the clean comparisons use IDs with at most 10 or 50 calls a year.
- The agents per half hour are counted from the services in the records, a lower bound on the agents on shift. The simulator's handle time is scaled so that it abandons as many calls as the data.

## License

Code: Apache-2.0, as the rest of the repository. Data: see the Technion SEE Center's terms above.
