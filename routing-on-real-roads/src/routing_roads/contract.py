"""Constants fixed by the registered plan (METHODS.md) or by its change log."""

from __future__ import annotations

from dataclasses import dataclass

# Section 2: the pinned extract (Geofabrik, OpenStreetMap data under the ODbL).
EXTRACT = "portugal-261027.osm.pbf"
EXTRACT_URL = "https://download.geofabrik.de/europe/" + EXTRACT
EXTRACT_BYTES = 426_796_277
EXTRACT_MD5 = "bc612795c7b549382d86abfc169b4b20"
EXTRACT_SHA256 = "73ae017f091747109dbfe0417896090d204f6bed73dd603170646df791e8961a"

# Section 3: the drivable network, as OSMnx 2.1.1's "drive" filter defines it, in Overpass
# semantics: a way is kept if it has a highway tag and no listed tag's value matches its
# pattern anywhere (unanchored, case-sensitive). tests/test_osm.py checks this against
# OSMnx's own filter string.
DRIVE_EXCLUDE = {
    "area": "yes",
    "access": "private",
    "highway": ("abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|"
                "escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|"
                "razed|rest_area|service|services|steps|track"),
    "motor_vehicle": "no",
    "motorcar": "no",
    "service": "alley|driveway|emergency_access|parking|parking_aisle|private",
}
BUFFER_M = 3_000                   # section 3: the area buffered by 3 km
PROJECTED_CRS = "EPSG:3763"        # ETRS89 / Portugal TM06, for the buffer (change log)
EARTH_RADIUS_M = 6_371_008.8       # section 4: great-circle distances


@dataclass(frozen=True)
class Pair:
    """Section 4: a city alone, and the city with the municipality across the water."""

    key: str
    city: str
    other: str
    order_probability: float


PAIRS = (Pair("lisbon", "Lisboa", "Almada", 0.2), Pair("porto", "Porto", "Vila Nova de Gaia", 0.5))
MUNICIPALITIES = tuple(m for p in PAIRS for m in (p.city, p.other))

DAYS = 40                          # section 4
DAYS_SEED = 20261028

# Section 5: the plans.
CAPACITY = 12
ITERATIONS = 5_000
SEEDS = (1, 2, 3, 4)
# Section 7: the sensitivity analyses run on the first 20 days with seeds 1 and 2.
SENSITIVITY_DAYS = 20
SENSITIVITY_SEEDS = (1, 2)
# Section 6: the bootstrap.
BOOTSTRAP_DRAWS = 9_999
BOOTSTRAP_SEED = 20261029
LEVEL = 0.95                       # reported intervals
SIMULTANEOUS_LEVEL = 1 - 0.05 / 4  # deciding intervals: Bonferroni over the four areas
