# Routing on real roads

Do delivery routes planned on straight-line distances cost more on real roads than routes
planned on road distances, by more than the solver's own noise, and does water make it
worse? A reproducible study on the OpenStreetMap road networks of Lisbon and Porto.

**Status: plan registered ([METHODS.md](METHODS.md)), before any distance was computed or
any route was solved.** The data, the estimates, the article and a critical review follow,
each in its own pull request.

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
