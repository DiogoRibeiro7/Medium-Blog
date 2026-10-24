"""Collect the numbers of every experiment into results/summary.txt."""

from __future__ import annotations

import json
from io import StringIO

import numpy as np
import pandas as pd

from common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def main() -> None:
    buf = StringIO()

    def out(text: str = "") -> None:
        buf.write(text + "\n")

    def table(frame: pd.DataFrame) -> None:
        out(frame.to_string())
        out()

    pd.set_option("display.width", 200)

    e1 = load("e1_fidelity")
    f = pd.DataFrame(e1["rows"])
    out("E1  Fidelity in the observed regime")
    out(f"    {e1['reps']} replicate histories; holdout and synthetic "
        "samples of 960 rows (40 days)")
    out("    discriminator ROC AUC (0.5 = indistinguishable), mean and sd:")
    s = f.dropna(subset=["auc"]).copy()
    s["n_days"] = s["n_days"].fillna(0).astype(int)
    table(s.groupby(["generator", "n_days"])["auc"]
          .agg(["mean", "std"]).round(3).unstack())
    out("    pooled service level and abandonment of the samples, 90 days "
        "of training (holdout = real):")
    p = f[(f.n_days == 90) | f.n_days.isna()]
    table(p.groupby("generator")[["sl", "abandon"]].mean().round(4))
    out(f"    mixture components chosen by BIC (replicate 0, 90 days): "
        f"{e1['gmm_components_rep0_n90']}")
    out()

    e2 = load("e2_surge")
    f = pd.DataFrame(e2["rows"])
    q = e2["training_day_ratio_quantiles"]
    out("E2  Fresh demand scaled by m (unannounced)")
    out(f"    {e2['reps']} replicates, 60 evaluation days per m")
    out("    day-level offered / expected in training: min {:.3f}, "
        "p5 {:.3f}, p95 {:.3f}, max {:.3f}".format(*q))
    out("    pooled service level:")
    table(f.groupby(["m", "generator"])["sl"].mean().unstack().round(3))
    out("    pooled abandonment:")
    table(f.groupby(["m", "generator"])["abandon"].mean().unstack()
          .round(3))
    red = pd.DataFrame(e2["redials"])
    red["m"] = red["m"].astype(str)
    out("    Erlang A with redials (probability p), in-sample and under m:")
    table(red.groupby(["m", "p"])[["sl", "abandon", "auc"]].mean()
          .round(3).unstack())
    abl = pd.DataFrame(e2["ablation"])
    out("    ablation: error of the analyst's generator (estimate - reality)"
        " with true components switched in")
    table(abl.groupby(["m", "arrivals", "service", "patience", "redials"])
          [["sl_error", "abandon_error"]].mean().round(3))

    e3 = load("e3_intervention")
    f = pd.DataFrame(e3["rows"])
    out("E3  Adding k agents to every interval")
    out(f"    {e3['reps']} replicates, 40 evaluation days")
    out("    effect on pooled service level (vs k = 0, same method):")
    table(f.groupby(["method", "k"])["sl_effect"].mean().unstack()
          .round(3))
    out("    effect on pooled abandonment:")
    table(f.groupby(["method", "k"])["abandon_effect"].mean().unstack()
          .round(3))
    out("    level of pooled service level:")
    table(f.groupby(["method", "k"])["sl"].mean().unstack().round(3))
    sup = pd.DataFrame(e3["support"])
    out(f"    training agents range (mean over replicates): "
        f"{sup.train_agents_min.mean():.1f} to {sup.train_agents_max.mean():.1f}")
    out()

    e4 = load("e4_error_budget")
    out("E4  Error budget for the long-run service level")
    out(f"    truth {e4['truth']['sl']:.4f} (from {e4['truth_days']} days), "
        f"{e4['reps']} replicates")
    a = pd.DataFrame(e4["panel_a"])
    a["error"] = a["estimate"] - e4["truth"]["sl"]
    out("    a) 90 real days, growing synthetic sample: bias | RMSE")
    table(a.groupby(["generator", "synthetic_days"])["error"]
          .agg(["mean", rmse]).round(4).unstack(0))
    b = pd.DataFrame(e4["panel_b"])
    b["error"] = b["estimate"] - e4["truth"]["sl"]
    out("    b) growing real data, large synthetic sample: bias | RMSE")
    table(b.groupby(["generator", "real_days"])["error"]
          .agg(["mean", rmse]).round(4).unstack(0))

    e5 = load("e5_calibration")
    f = pd.DataFrame(e5["rows"])
    out("E5  Calibrating Erlang A's patience rate theta")
    out(f"    {e5['reps']} replicates per sample size")
    s = f.groupby(["estimator", "n_days"])["theta"].agg(["mean", "std"])
    s["implied_mean_patience_s"] = 1 / s["mean"]
    table(s.round(6))
    out(f"    long-run limits ({e5['limit_days']} days):")
    for k, v in e5["limits"].items():
        out(f"      {k:24s} theta {v:.6f}  (1/theta = {1 / v:.1f} s)")
    out("    references:")
    for k, v in e5["references"].items():
        out(f"      {k:24s} theta {v:.6f}  (1/theta = {1 / v:.1f} s)")
    out(f"    joint forecast at m = {e5['surge']} with each theta:")
    for k, v in e5["surge_forecasts"].items():
        out(f"      {k:40s} SL {v['sl']:.3f}  abandon {v['abandon']:.3f}")
    out()

    e6 = load("e6_privacy_cost")
    f = pd.DataFrame(e6["rows"])
    out("E6  Distance to closest record, and cost")
    out(f"    {e6['reps']} replicates, 1,000 synthetic rows each; "
        "reference = held-out real rows")
    table(f.groupby("generator")[["median_dcr_ratio",
                                  "share_closer_than_p5",
                                  "share_exact_copies"]].mean().round(3))
    out("    timing on the machine that made this file (seconds):")
    table(pd.DataFrame(e6["timing"]).set_index("generator").round(3))

    e7 = load("e7_decision")
    out("E7  Staffing for an uncertain surge")
    out(f"    target: day service level >= {e7['target']} on at least "
        f"{e7['coverage']:.0%} of days; surge uniform on "
        f"{e7['surge'][0]}-{e7['surge'][1]}; {e7['days']} launch days")
    curves = pd.DataFrame({n: pd.Series(c) for n, c in e7["curves"].items()})
    curves.index = curves.index.astype(int)
    out("    share of days meeting the target, by extra agents k:")
    table(curves.sort_index().round(3))
    out("    chosen k and how it does in reality:")
    table(pd.DataFrame(e7["chosen"]).T)
    per = np.array(e7["per_scenario"])
    out(f"    robust variant: {per.shape[0]} scenarios; share of scenarios "
        "meeting the requirement at each k:")
    out("      " + ", ".join(f"k={k}: {v:.2f}" for k, v in
                             zip(e7["ks"], np.mean(per >= e7["coverage"],
                                                   axis=0))))

    (RESULTS / "summary.txt").write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
