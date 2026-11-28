#!/usr/bin/env bash
# Reproduce every number and figure. Data are downloaded from the OECD SDMX API into ./raw (not tracked).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw data results figures
python3 -I src/fetch_data.py raw
python3 -I src/build_panel.py raw data/panel.csv
python3 -I src/panel_models.py data/panel.csv results
python3 -I src/panel_robustness.py data/panel.csv results
python3 -I src/efficiency_speccurve.py data/panel.csv results
python3 -I src/pandemic.py raw data/panel.csv results
python3 -I src/pandemic_all.py raw data/panel.csv results figures
python3 -I src/thesis_checks.py results
python3 -I src/figures.py results figures data/panel.csv
