"""Cut the mainland Portugal outline (and the nearby Spanish border) out of Natural Earth.

Used once to write data/portugal_outline.csv, which the figures draw under the ERA5 nodes.
Source: Natural Earth 1:50m admin-0 countries (public domain), from
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson

    python make_outline.py path/to/ne_50m_admin_0_countries.geojson
"""

import csv
import json
import sys
from pathlib import Path

BOX = (-10.0, -5.5, 36.7, 42.4)  # lon min, lon max, lat min, lat max


def main(path: str) -> None:
    gj = json.loads(Path(path).read_text())
    rows = []
    for feat in gj["features"]:
        code = feat["properties"].get("ADM0_A3")
        if code not in ("PRT", "ESP"):
            continue
        for k, poly in enumerate(feat["geometry"]["coordinates"]):
            ring = poly[0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            # keep the mainland rings that reach into the box (drops islands elsewhere)
            if max(lons) < BOX[0] or min(lons) > BOX[1] or max(lats) < BOX[2] or min(lats) > BOX[3]:
                continue
            if len(ring) < 50:
                continue
            rows += [(code, k, round(x, 4), round(y, 4)) for x, y in ring]
    out = Path(__file__).with_name("data") / "portugal_outline.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["country", "ring", "lon", "lat"])
        w.writerows(rows)
    print(f"{len(rows)} vertices written to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
