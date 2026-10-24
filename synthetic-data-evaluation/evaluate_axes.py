"""Fidelity, utility and privacy of one synthetic copy of the Adult training set.

The Adult file is split into three disjoint thirds (data.three_way_split):
train (the confidential data every generator learns from), holdout (the test
set for train-synthetic-test-real and the non-members for the privacy
tests) and reference (an independent real sample, the score an ideal
generator would get). Each generator writes one synthetic table the size of
train. CTGAN and TVAE tables are produced by deep.py (which needs PyTorch)
and read from data/synthetic/ when present.

Writes results/axes.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as mt
from data import load_adult, three_way_split
from generators import Encoder, TrainingCopy, factories, synthesize

HERE = Path(__file__).resolve().parent
SPLIT_SEED = 20261024
OUT = HERE / "results" / "axes.json"
DEEP = HERE / "data" / "synthetic"
GEN_SEEDS = {"independent": 11, "copula": 12, "cart": 13, "smote": 14, "training copy": 15}


def load_split():
    df = load_adult()
    train, holdout, reference = three_way_split(df, SPLIT_SEED)
    enc = Encoder.from_frame(df)
    return df, train, holdout, reference, enc


def synthetic_tables(train, reference, enc):
    T = enc.to_codes(train)
    target = enc.columns.index("income")
    tables = {"reference (real)": enc.to_codes(reference)}
    makers = {"training copy": lambda: TrainingCopy(enc)}
    makers.update(factories(enc, class_col=target))
    for name, make in makers.items():
        rng = np.random.default_rng(GEN_SEEDS[name])
        tables[name] = synthesize(make, T, len(T), 1, rng, proper=False)[0]
    for name in ("ctgan", "tvae"):
        path = DEEP / f"{name}.csv.gz"
        if path.exists():
            tables[name] = enc.to_codes(pd.read_csv(path)[enc.columns])
    return tables


def evaluate(train, holdout, enc, tables) -> dict:
    T, H = enc.to_codes(train), enc.to_codes(holdout)
    target = enc.columns.index("income")
    emap = train.groupby("education")["education-num"].first().to_dict()
    ranges = {
        c: (float(train[c].min()), float(train[c].max()))
        for c in enc.columns
        if not enc.categorical[c]
    }
    b_real, v_real = mt.analysis_fit(train)
    z = 1.959963984540054
    lo_r, hi_r = b_real - z * np.sqrt(v_real), b_real + z * np.sqrt(v_real)

    out = {
        "n_train": len(T),
        "n_holdout": len(H),
        "columns": enc.columns,
        "train_consistency": mt.consistency(train, emap, ranges),
        "train_point_masses": {
            "capital-gain == 0": float(np.mean(train["capital-gain"] == 0)),
            "capital-loss == 0": float(np.mean(train["capital-loss"] == 0)),
            "hours-per-week == 40": float(np.mean(train["hours-per-week"] == 40)),
        },
        "trtr": mt.tstr(T, H, enc, target, seed=0),
        "real_coef": b_real.tolist(),
        "real_se": np.sqrt(v_real).tolist(),
        "terms": mt.ANALYSIS_TERMS,
        "generators": {},
    }
    for i, (name, S) in enumerate(tables.items()):
        rng = np.random.default_rng(100 + i)
        sdf = enc.from_codes(S)
        marg = mt.marginal_distances(T, S, enc)
        pair = mt.pairwise_tvd(T, S, enc)
        priv = mt.privacy(T, H, S, enc, rng)
        disc = mt.discriminator(T, S, enc, seed=0)
        disc_pair = mt.discriminator(T, S, enc, seed=0, pair_with=priv["nearest_train"])
        b, v = mt.analysis_fit(sdf)
        lo_s, hi_s = b - z * np.sqrt(v), b + z * np.sqrt(v)
        d = np.sort(priv["d_train"])
        out["generators"][name] = {
            "marginal": marg,
            "marginal_mean": float(np.mean(list(marg.values()))),
            "pairwise_tvd": pair.tolist(),
            "pairwise_tvd_mean": float(pair[np.triu_indices_from(pair, 1)].mean()),
            "discriminator_auc": disc["auc"],
            "pmse": disc["pmse"],
            "discriminator_auc_paired": disc_pair["auc"],
            "pmse_paired": disc_pair["pmse"],
            "consistency": mt.consistency(sdf, emap, ranges),
            "point_masses": {
                "capital-gain == 0": float(np.mean(sdf["capital-gain"] == 0)),
                "capital-loss == 0": float(np.mean(sdf["capital-loss"] == 0)),
                "capital-gain < 0": float(np.mean(sdf["capital-gain"] < 0)),
                "capital-loss < 0": float(np.mean(sdf["capital-loss"] < 0)),
                "hours-per-week == 40": float(np.mean(sdf["hours-per-week"] == 40)),
            },
            "tstr": mt.tstr(S, H, enc, target, seed=0),
            "coef": b.tolist(),
            "se": np.sqrt(v).tolist(),
            "coef_diff_in_real_se": ((b - b_real) / np.sqrt(v_real)).tolist(),
            "ci_overlap": mt.ci_overlap(lo_r, hi_r, lo_s, hi_s).tolist(),
            "exact_copies": priv["exact_copies"],
            "median_dcr": priv["median_dcr"],
            "dcr_share": priv["dcr_share"],
            "mia_auc": priv["mia_auc"],
            "mia_tpr_at_1pct": priv["mia_tpr_at_1pct"],
            "dcr_quantiles": d[np.linspace(0, len(d) - 1, 401).astype(int)].tolist(),
        }
    return out


def main() -> None:
    _, train, holdout, reference, enc = load_split()
    tables = synthetic_tables(train, reference, enc)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(evaluate(train, holdout, enc, tables), indent=1))


if __name__ == "__main__":
    main()
