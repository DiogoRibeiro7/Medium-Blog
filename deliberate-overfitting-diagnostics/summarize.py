"""Collect every number quoted in the article into results/summary.txt."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RES


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main():
    out: list[str] = []
    w = out.append

    # ---------------------------------------------------------------- exp1a
    a = pd.read_csv(RES / "exp1a_memorisation_grid.csv")
    acc = a[a["train_acc"].notna()]
    w("== Exp 1A: memorisation grid (simulation, n=2000, 3 replications) ==")
    w(f"lowest training accuracy over every model, noise level and label version: {acc['train_acc'].min():.4f}")
    t = acc[acc["labels"] == "real"].groupby(["model", "eta"])["test_acc"].mean().unstack(1)
    w("mean test accuracy, real labels (columns = eta):")
    w(t.round(3).to_string())
    s = acc[acc["labels"] == "shuffled"]["test_acc"]
    w(f"test accuracy with shuffled labels: min {s.min():.3f}, mean {s.mean():.3f}, max {s.max():.3f}")
    eff = a[a["effort"].notna() | (a["effort_unit"] != "")].copy()
    eff = eff[eff["effort_unit"].fillna("") != ""]
    e = eff.groupby(["effort_unit", "eta", "labels"])["effort"].mean().unstack(2)
    e["ratio_real_to_shuffled"] = e["real"] / e["shuffled"]
    w("memorisation effort, mean over replications:")
    w(e.round(3).to_string())
    w("")

    # ---------------------------------------------------------------- exp1b
    b = pd.read_csv(RES / "exp1b_detection_power.csv")
    g = b.groupby("eta").agg(reps=("rep", "size"), memorised=("memorised", "mean"),
                             power_leaves=("p_leaves", lambda p: (p <= 0.05).mean()),
                             power_holdout=("p_holdout", lambda p: (p <= 0.05).mean()),
                             leaves_real=("leaves_real", "mean"), leaves_real_sd=("leaves_real", "std"),
                             leaves_null=("leaves_null_mean", "mean"),
                             holdout_acc=("holdout_acc", "mean"), holdout_acc_sd=("holdout_acc", "std"))
    w("== Exp 1B: share of data sets declared to carry signal (alpha = 0.05) ==")
    w(g.round(3).to_string())
    w("")

    # ---------------------------------------------------------------- exp1c
    c = pd.read_csv(RES / "exp1c_feature_set_ranking.csv")
    w("== Exp 1C: feature-set ranking, UCI Adult (70/30 split) ==")
    w(c.round(4).to_string(index=False))
    w(f"rank by tree training accuracy: {list(c.sort_values('tree_train_acc', ascending=False)['feature_set'])}")
    w(f"rank by held-out accuracy:      {list(c.sort_values('reg_xgb_test_acc', ascending=False)['feature_set'])}")
    w("")

    # ---------------------------------------------------------------- exp2a
    d = pd.read_csv(RES / "exp2a_adult_duplicate_floor.csv")
    w("== Exp 2A: duplicate floor, UCI Adult, columns added in order ==")
    w(d.round(4).to_string(index=False))
    w(f"max |tree training error - plug-in floor|: {np.abs(d['overfit_tree_train_error'] - d['plug_in_floor']).max():.2e}")
    w("")

    # ---------------------------------------------------------------- exp2b
    ch = pd.read_csv(RES / "exp2b_cover_hart.csv")
    m = ch.groupby(["eta", "noise_dims", "n"])[["r_1nn", "cover_hart_lower", "xgb_test_error"]].mean()
    w("== Exp 2B: Cover-Hart lower bound from leave-one-out 1-NN error (mean of 4 reps) ==")
    w(m.round(3).to_string())
    w("")

    # ---------------------------------------------------------------- exp2c
    lc = pd.read_csv(RES / "exp2c_learning_curves.csv")
    fits = pd.read_csv(RES / "exp2c_asymptotes.csv")
    w("== Exp 2C: learning curves (means of 3 reps) ==")
    w(lc.groupby(["data", "n"])[["test_error", "train_error", "overfit_train_error"]].mean().round(4).to_string())
    w(fits.round(4).to_string(index=False))
    w("")

    # ---------------------------------------------------------------- exp2d
    rs = pd.read_csv(RES / "exp2d_residual_variance_sim.csv")
    w("== Exp 2D: nearest-neighbour residual variance, simulation (mean, sd over 4 reps) ==")
    w(rs.groupby(["sigma2", "noise_dims", "n"])[["product_estimate", "difference_estimate"]]
      .agg(["mean", "std"]).round(3).to_string())
    h = pd.read_csv(RES / "exp2d_residual_variance_housing.csv")
    h["r2_ceiling_product"] = 1 - h["product_estimate"] / h["var_y"]
    h["r2_xgb"] = 1 - h["xgb_cv_mse"] / h["var_y"]
    h["xgb_mse_over_floor"] = h["xgb_cv_mse"] / h["product_estimate"]
    w("California housing, log median house value:")
    w(h.round(4).to_string(index=False))
    w("")

    # ---------------------------------------------------------------- exp3
    mtx = pd.read_csv(RES / "exp3_detection_matrix.csv")
    info = pd.read_csv(RES / "exp3_check_details.csv")
    w("== Exp 3: planted bugs x checks (True = check fails) ==")
    w(mtx.pivot(index="bug", columns="check", values="flagged").to_string())
    w(info.round(4).T.to_string())

    text = "\n".join(out) + "\n"
    (RES / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
