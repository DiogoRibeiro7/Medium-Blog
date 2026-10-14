#!/usr/bin/env bash
# Full run, in the order the results were produced for the article (about 4 hours on a
# two-core CPU). Each step writes to results/ and can be re-run; the grid and PINN steps
# skip work already saved.
set -euo pipefail
cd "$(dirname "$0")"
python e1_convergence.py
python e2_grids_by_dimension.py
python e4_dmc.py
python e3_pinn.py --seeds 0
python e3_pinn.py --seeds 1 2 --dims 3 5 8
python summarize.py
python make_figures.py
