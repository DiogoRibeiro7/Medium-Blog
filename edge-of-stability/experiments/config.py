"""Settings shared by the experiment scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_TRAIN = 1000          # digits: 1000 training images, 797 test images
SIZES = [64, 64, 64, 64, 10]  # three hidden layers of width 64
INIT_GAIN = 1.0         # weight variance gain / fan_in
