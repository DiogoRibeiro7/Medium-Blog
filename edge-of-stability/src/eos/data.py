"""The UCI handwritten digits data shipped with scikit-learn (no download needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits


@dataclass(frozen=True)
class Split:
    x_train: np.ndarray
    y_train: np.ndarray  # one-hot targets
    labels_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels_test: np.ndarray


def digits_split(n_train: int = 1000, split_seed: int = 0) -> Split:
    """Fixed random train/test split, standardised with training statistics.

    The split seed is kept separate from the model seed so that every run in a
    sweep sees exactly the same data.
    """
    x, labels = load_digits(return_X_y=True)
    order = np.random.default_rng(split_seed).permutation(len(x))
    x, labels = x[order].astype(np.float64), labels[order]
    mean = x[:n_train].mean(axis=0)
    std = x[:n_train].std(axis=0) + 1e-8
    x = (x - mean) / std
    onehot = np.eye(10)[labels]
    return Split(
        x_train=x[:n_train],
        y_train=onehot[:n_train],
        labels_train=labels[:n_train],
        x_test=x[n_train:],
        y_test=onehot[n_train:],
        labels_test=labels[n_train:],
    )
