#!/usr/bin/env bash
# Reproduce every number and figure. No download step: the aggregate series are transcribed in src/series.py
# from public web pages (URLs listed in that file), because the sources are press releases and a web table.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures
python3 -I src/timing_check.py results/timing_check.json > /dev/null
python3 -I src/robustness.py results/robustness.json > /dev/null
python3 -I src/transfer_model.py results/transfer_model.json > /dev/null
python3 -I src/figures.py results figures
