#!/usr/bin/env bash
# Reproduce every number and figure. Data are downloaded from the OECD SDMX API into ./raw (not tracked).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw data results figures
python3 -I src/fetch_data.py raw
python3 -I src/build_panel.py raw data/panel.csv
python3 -I src/panel_models.py data/panel.csv results
python3 -I src/austerity_event.py data/panel.csv results
python3 -I src/austerity_figures.py results results figures data/panel.csv
