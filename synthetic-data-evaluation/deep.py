"""Train CTGAN and TVAE on the Adult training third and save one synthetic table each.

Needs the packages in requirements-deep.txt (PyTorch and ctgan). The output
tables are committed to data/synthetic/, so the rest of the pipeline can be
rerun without PyTorch; rerunning this script retrains the models, and neural
network training on a different machine or library build will not reproduce
the committed tables bit for bit.

Usage: python deep.py [--epochs 300]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ctgan import CTGAN, TVAE

from data import CATEGORICAL, NUMERIC, load_adult, three_way_split
from evaluate_axes import SPLIT_SEED

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    train, _, _ = three_way_split(load_adult(), SPLIT_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cls, seed in (("ctgan", CTGAN, 21), ("tvae", TVAE, 22)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = cls(epochs=args.epochs, enable_gpu=False)
        model.set_random_state(seed)
        start = time.time()
        model.fit(train, discrete_columns=CATEGORICAL)
        fitted = time.time() - start
        syn = model.sample(len(train))
        for col in NUMERIC:
            syn[col] = np.round(syn[col].astype(float)).astype(np.int64)
        syn = syn[train.columns]
        syn.to_csv(OUT / f"{name}.csv.gz", index=False)
        print(f"{name}: {args.epochs} epochs, fitted in {fitted / 60:.1f} min")


if __name__ == "__main__":
    main()
