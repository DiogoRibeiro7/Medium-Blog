# Routing on real roads

Do delivery routes planned on straight-line distances cost more on real roads than routes
planned on road distances, by more than the solver's own noise, and does water make it
worse? A reproducible study on the OpenStreetMap road networks of Lisbon and Porto.

**Status: plan registered ([METHODS.md](METHODS.md)); data built
([docs/DATA.md](docs/DATA.md)); every registered route solved and estimated
([docs/RESULTS.md](docs/RESULTS.md)); the article written from the results
([publication/article.md](publication/article.md)); a critical review
([docs/REVIEW.md](docs/REVIEW.md)) and the corrections it recommended, logged in the plan's
change log.**

## Running it

```bash
pip install -e ".[dev]"
python -m pytest -q          # offline, from the committed tables in data/derived/
```

Rebuilding the tables needs the pinned extract in `data/raw/`. Geofabrik offers a dated
extract for a few days only; after that, the committed tables and the hashes in their
manifest are the record. The build checks the extract's hashes first:

```bash
routing-roads data           # about 3 minutes; writes data/derived/ and docs/DATA.md
```

The routes and estimates run offline from the committed tables:

```bash
routing-roads route          # 2,560 solves on 4 processes, about an hour; resumes if stopped
routing-roads estimate       # results/estimates.json and docs/RESULTS.md, in seconds
```

The article is generated from the committed results; a test checks that it is current, that
it types no result by hand and that its claims in words match the results:

```bash
routing-roads publication    # publication/article.md, values.json and figures/
```

- **Customers:** OpenStreetMap pharmacies in Lisboa, Almada, Porto and Vila Nova de Gaia,
  with simulated daily orders.
- **Distances:** shortest directed paths on the drivable network (one-way streets
  respected), against great-circle distances.
- **Solver:** PyVRP 0.14.1, with an iteration budget and fixed seeds, so the same routes
  come out on any machine.

## Data and licence

Map data © OpenStreetMap contributors, available under the Open Database Licence (ODbL),
from Geofabrik's dated extract `portugal-261027.osm.pbf`. The raw extract is not
committed. The derived tables in `data/derived/` and the solves computed from them are
published under the ODbL ([data/derived/NOTICE.md](data/derived/NOTICE.md)); the code is
under the repository's licence.
