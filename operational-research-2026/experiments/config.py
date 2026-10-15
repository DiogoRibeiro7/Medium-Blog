"""Settings shared by the experiment scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_HISTORY = 104                      # two years of weekly demand
N_TEST = 50_000                      # fresh weeks used to score every design
QUANTILES = [50, 75, 90, 95]         # zone-wise quantile plans
SIZES = [10, 26, 52, 104, 208]       # history lengths in the stability study
REPLICATIONS = 10                    # independent histories per length

# quick mode (CI smoke run)
QUICK_N_TEST = 2_000
QUICK_SIZES = [10, 26]
QUICK_REPLICATIONS = 3
