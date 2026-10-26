# Hotel cancellation replay

Companion code for the Medium article on the data scientist as scientist, engineer and architect. It replays a hotel cancellation system night by night on the real outcomes of two Portuguese hotels and measures how much of the value depends on the model and how much on the decisions around it.

## Data

Antonio, N., de Almeida, A. and Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126). 119,390 bookings due to arrive between 1 July 2015 and 31 August 2017 at a resort hotel in the Algarve and a city hotel in Lisbon, as redistributed by TidyTuesday. The file is committed compressed in `data/hotels.csv.gz`; its SHA-256 is checked on every load. The city hotel's soft opening (July–August 2015) is dropped, as the data authors did in their own production system (Antonio et al., 2019, *Data Science Journal* 18, 32).

## The decision

Every night a batch runs *h* days before an arrival date (30 by default). It sees the bookings entered before that day and not yet cancelled, scores each one, and decides how many of their rooms to put back on sale. A resold room earns one night at the median rate of the visible bookings; a guest who arrives to a resold room is walked at three nights' rate. The decision that maximises expected profit is a quantile of the distribution of the number of rooms released, so it needs that distribution and not only its mean. Value is reported as the share of the perfect-foresight profit (reselling exactly the rooms that will be released) captured over 854 live arrival nights, July 2016 to August 2017.

## What is compared

One reference system and seventeen variants, each changing one thing:

- **The model and its inputs**: no model (historical rate), logistic regression, gradient boosting (reference), boosting tuned on a time-ordered split, adding the deposit column whose timing is unclear.
- **How the data reach the model**: training on all bookings with the label "ever cancelled"; training on "everything with a known status" (the construction the data authors report abandoning in production); reading columns that are only set at check-in, served as recorded and as an unarrived booking would show them; fitting once; refitting weekly.
- **Where the parts meet**: a decision that treats bookings as independent; a decision that receives only the expected count; a decision that receives yes/no labels; a class-weighted model swapped in without changing the decision; batches 7, 14 and 60 days before arrival.

Differences from the reference come with 95% intervals from a paired bootstrap over weeks.

## Layout

```text
cancelreplay/
  data.py         load, checksum and data contract
  features.py     feature registry: when each column becomes known
  snapshots.py    point-in-time snapshots and the three training sets
  models.py       base rate, logistic regression, gradient boosting
  decision.py     distribution of released rooms, release rule, profit
  replay.py       system specification, walk-forward refits, nightly log
  experiments.py  the variants, bootstrap, descriptive analyses
  report.py       writes results/summary.txt and results/results.json
  figures.py      the article's figures
tests/            snapshot edges, decision rule, data contract
```

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m cancelreplay check   # data contract
python -m cancelreplay run     # every variant, results and figures (about 15 minutes)
python -m cancelreplay fetch   # download the CSV again and verify its checksum
```

`.github/workflows/hotel-cancellation-replay.yml` runs the tests and the full replay on every change and fails if `results/summary.txt` differs from the committed run.

## Limitations

The replay holds the resale price fixed and assumes every released room sells, so it measures how well the system anticipates releases, not how marketable a room is at each horizon. It ignores bookings made after the batch, which a real overbooking policy would also forecast. The published file records some columns after check-in although the documentation describes a snapshot taken the day before arrival; the replay can only bracket what those columns would have shown at decision time. The hotels' own interventions during 2017 (contacting guests the model flagged) may have changed some outcomes, and the file does not record which bookings were contacted.
