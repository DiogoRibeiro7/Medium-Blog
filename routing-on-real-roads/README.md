# Routing on real roads

Do delivery routes planned on straight-line distances cost more on real roads than routes
planned on road distances, by more than the solver's own noise, and does water make it
worse? A reproducible study on the OpenStreetMap road networks of Lisbon and Porto.

**Status: plan registered ([METHODS.md](METHODS.md)); data built
([docs/DATA.md](docs/DATA.md)); every registered route solved and estimated
([docs/RESULTS.md](docs/RESULTS.md)).** The article and a critical review follow, each in its
own pull request.

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

- **Customers:** OpenStreetMap pharmacies in Lisboa, Almada, Porto and Vila Nova de Gaia,
  with simulated daily orders.
- **Distances:** shortest directed paths on the drivable network (one-way streets
  respected), against great-circle distances.
- **Solver:** PyVRP 0.14.1, with an iteration budget and fixed seeds, so the same routes
  come out on any machine.

## Data and licence

Map data © OpenStreetMap contributors, available under the Open Database Licence (ODbL),
from Geofabrik's dated extract `portugal-261027.osm.pbf`. The raw extract is not
committed. Derived tables, once committed, are published under the ODbL.
