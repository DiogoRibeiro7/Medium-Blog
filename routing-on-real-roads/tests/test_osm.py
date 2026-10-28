"""The drive filter is OSMnx's, read from OSMnx itself."""

from __future__ import annotations

import re

from routing_roads.contract import DRIVE_EXCLUDE
from routing_roads.osm import drivable


def test_the_drive_filter_matches_osmnx() -> None:
    from osmnx import _overpass

    text = _overpass._get_network_filter("drive")
    assert text.startswith('["highway"]')
    parsed = dict(re.findall(r'\["([a-z_]+)"!~"([^"]+)"\]', text))
    assert parsed == DRIVE_EXCLUDE


def test_overpass_semantics() -> None:
    assert drivable({"highway": "residential"})
    assert drivable({"highway": "primary", "oneway": "yes"})
    assert not drivable({"name": "no highway tag"})
    assert not drivable({"highway": "footway"})
    assert not drivable({"highway": "service"})
    assert not drivable({"highway": "residential", "access": "private"})
    assert not drivable({"highway": "residential", "motor_vehicle": "no"})
    # unanchored: any value containing a listed pattern is dropped, as in Overpass
    assert not drivable({"highway": "residential", "area": "yes;no"})
    assert drivable({"highway": "residential", "service": "drive-through"})
