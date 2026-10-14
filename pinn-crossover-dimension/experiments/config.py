"""Settings shared by the experiment scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

L = 5.0                      # box half-width of the PINN (and of experiment 1)
# Grid boxes. Experiment 1 shows the box-truncation error in the energy is below 1e-8
# (relative) at L = 5.0 for the oscillator and L = 4.5 for the coupled problem, and
# about 1e-6 at L = 4.0 for both. A smaller box means a coarser grid for the same N,
# so experiment 2 runs every grid method at both sizes, and the comparison at a given
# accuracy uses whichever is cheaper while still reaching it.
GRID_L = {"oscillator": [4.0, 5.0], "coupled": [4.0, 4.5]}
G = 1.0                      # coupling strength of the non-separable problem
DIMS = {"oscillator": list(range(1, 9)), "coupled": list(range(2, 9))}

# memory available to one grid solve on the machine used for the article (7.7 GB RAM)
MEMORY_BUDGET = 5.0e9
LAPTOP_MEMORY = 16e9          # reference lines in the figures
WORKSTATION_MEMORY = 256e9

# points per axis tried for each grid method (ascending)
GRID_N = {
    "fd2": [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
    "fd4": [8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128],
    "sinc": [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40],
}

# diffusion Monte Carlo
DMC_TAU = 0.04
DMC_WALKERS = 4000
DMC_STEPS = [1000, 4000, 16000]
DMC_TAU_CHECK = [0.08, 0.04, 0.02]


def quick_suffix(quick: bool) -> str:
    return "_quick" if quick else ""
