#!/usr/bin/env bash
# Fetch the pinned input data into ./data (or into $CABLE_DATA if set).
#
# Cable routes and landing points: TeleGeography Submarine Cable Map, licensed CC BY-NC-SA 3.0,
# as fetched on 2026-09-24 and committed to github.com/ixpavi/subsea-siting. They are fetched
# here, not redistributed in this repository.
# Coastlines: Natural Earth 1:10m land and minor islands, public domain.
set -euo pipefail
DEST="${CABLE_DATA:-$(cd "$(dirname "$0")" && pwd)/data}"
mkdir -p "$DEST"

fetch() {  # fetch <repo url> <dir> <commit> <path>...
  local url=$1 dir=$2 commit=$3; shift 3
  if [ ! -d "$DEST/$dir/.git" ]; then
    git clone --quiet --filter=blob:none --no-checkout "$url" "$DEST/$dir"
  fi
  git -C "$DEST/$dir" sparse-checkout set --no-cone "$@"
  git -C "$DEST/$dir" checkout --quiet "$commit"
}

fetch https://github.com/ixpavi/subsea-siting ixpavi_subsea-siting 98d377f0b628d174a9d4630825e34cf09fcd7f6e \
  /public/data/cables.json /public/data/cable-details.json /public/data/landing-points.json
fetch https://github.com/nvkelso/natural-earth-vector ne ca96624a56bd078437bca8184e78163e5039ad19 \
  /geojson/ne_10m_land.geojson /geojson/ne_10m_minor_islands.geojson
echo "data in $DEST"
