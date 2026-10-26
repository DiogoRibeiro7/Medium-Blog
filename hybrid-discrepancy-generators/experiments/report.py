"""Collect the numbers of every experiment into results/summary.txt."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from common import RESULTS, load


def main() -> None:
    buf = StringIO()
    pd.set_option("display.width", 220)

    def out(text: str = "") -> None:
        buf.write(text + "\n")

    def table(frame) -> None:
        out(frame.to_string())
        out()

    h1 = load("h1_fit")
    f = pd.DataFrame(h1["prediction"])
    out("H1  Fit in the observed regime")
    out(f"    {h1['reps']} replicate histories, 40 held-out days")
    out("    prediction given each half hour's inputs: day-level RMSE of "
        "the service level (mean over replicates)")
    s = f[~f.model.str.endswith("(impossible)")]
    table(s.groupby(["model", "n_days"])["day_rmse_sl"].mean().unstack()
          .round(4))
    out("    same, abandonment share:")
    table(s.groupby(["model", "n_days"])["day_rmse_ab"].mean().unstack()
          .round(4))
    out("    same, interval-level RMSE of the service level:")
    table(s.groupby(["model", "n_days"])["interval_rmse_sl"].mean()
          .unstack().round(4))
    out("    share of predicted rows that were impossible before clipping:")
    imp = f[f.model.str.endswith("(impossible)")]
    table(imp.groupby(["model", "n_days"])["share_impossible"].mean()
          .unstack().round(3))
    g = pd.DataFrame(h1["generation"])
    out("    generation: discriminator AUC against held-out days (real vs "
        "real is about 0.54 at this sample size), mean and sd:")
    a = g.dropna(subset=["auc"])
    table(a.groupby(["model", "n_days"])["auc"].agg(["mean", "std"])
          .round(3).unstack())
    out("    generation: share of generated rows that needed clipping:")
    gi = g.dropna(subset=["share_impossible"])
    table(gi.groupby(["model", "n_days"])["share_impossible"].mean()
          .unstack().round(3))
    ph = g.dropna(subset=["p_hat"])
    out("    redial probability estimated inside H1 (mean, sd): "
        + ", ".join(f"{int(n)} days {v['mean']:.3f} ({v['std']:.3f})"
                    for n, v in ph.groupby("n_days")["p_hat"]
                    .agg(["mean", "std"]).iterrows()))
    rng_ = h1["train_load_range"]
    if rng_:
        out(f"    load per agent in the 90 training days of replicate 0: "
            f"p1 {rng_['p1']:.3f}, p99 {rng_['p99']:.3f}, "
            f"max {rng_['max']:.3f}")
        rb = pd.DataFrame(h1["residual_bins"])
        out("    mean residual (reality minus Erlang A, service level) by "
            "load bin, replicate 0, 90 days:")
        table(rb.set_index("load").round(4))
        c = pd.DataFrame(h1["curve"])
        out("    learned correction to the service level along load per "
            "agent (mid-morning, 30 agents):")
        table(c.iloc[::3].set_index("load").round(4))

    h2 = load("h2_surge")
    f = pd.DataFrame(h2["rows"])
    out("H2  Fresh demand scaled by m (unannounced)")
    out(f"    {h2['reps']} replicates, {h2['eval_days']} days per m")
    out("    pooled service level:")
    table(f.groupby(["model", "m"])["sl"].mean().unstack().round(3))
    out("    pooled abandonment:")
    table(f.groupby(["model", "m"])["abandon"].mean().unstack().round(3))
    ph = f.dropna(subset=["p_hat"]).groupby("rep")["p_hat"].first()
    out("    redial probability estimated per replicate: "
        + ", ".join(f"{v:.3f}" for v in ph))
    out()

    h3 = load("h3_redials")
    f = pd.DataFrame(h3["estimates"])
    out("H3  Sizing the redial slot by indirect inference")
    out(f"    truth: p = {h3['true_p']}, mean delay {h3['true_delay']:.0f} s;"
        f" {h3['reps']} replicates")
    out("    estimated p (mean, sd) by days of data, assumed delay 600 s:")
    e = f[(f.world == "redials 0.6") & (f.delay == 600.0)]
    table(e.groupby("n_days")["p_hat"].agg(["mean", "std", "min", "max"])
          .round(3))
    out("    placebo world with no redials:")
    table(f[f.world == "no redials"].groupby("n_days")["p_hat"]
          .agg(["mean", "std", "min", "max"]).round(3))
    out("    estimated p by assumed delay:")
    table(f[f.world == "redials 0.6"].groupby(["delay", "n_days"])["p_hat"]
          .agg(["mean", "std"]).round(3).unstack())
    out(f"    30% surge forecast (resampled day levels + redials) by assumed"
        f" delay, and reality:")
    sf = f[f.world == "surge forecast"]
    table(sf.groupby(["delay", "n_days"])[["sl", "sl_reality"]].mean()
          .round(3).unstack())
    aux = pd.DataFrame(h3["auxiliary"])
    out("    auxiliary coefficient (mean, sd):")
    table(aux.groupby(["world", "n_days"])["beta"].agg(["mean", "std"])
          .round(3).unstack(0))
    b = h3["binding"]
    out("    binding function, replicate 0: "
        + ", ".join(f"p={p:.1f}: {v:.3f}" for p, v in
                    zip(b["grid"], b["beta_simulated"]))
        + f"; observed {b['beta_observed']:.3f}")
    out()

    h4 = load("h4_confounding")
    f = pd.DataFrame(h4["rows"])
    out("H4  Patience rate and discrepancy")
    out(f"    reality's mean patience {h4['true_mean_patience']:.0f} s; "
        f"{h4['reps']} replicates; ratio estimate of 1/theta: "
        + ", ".join(f"{v:.0f} s" for v in
                    f[f.model == "ratio estimate"]["patience"]))
    g = f[f.model != "ratio estimate"]
    out("    held-out day-level RMSE of the service level:")
    table(g.groupby(["model", "patience"])["day_rmse_sl"].mean().unstack()
          .round(4))
    out(f"    service level forecast for a {h4['surge']:.0%} surge "
        f"(reality: {g['surge_sl_reality'].mean():.3f}):")
    table(g.groupby(["model", "patience"])["surge_sl"].mean().unstack()
          .round(3))

    h5 = load("h5_intervention")
    f = pd.DataFrame(h5["rows"])
    out("H5  Adding k agents to every half hour")
    out(f"    {h5['reps']} replicates, 40 days; effect on pooled service "
        "level:")
    table(f.groupby(["model", "k"])["sl_effect"].mean().unstack().round(3))
    out("    effect on pooled abandonment:")
    table(f.groupby(["model", "k"])["abandon_effect"].mean().unstack()
          .round(3))

    h6 = load("h6_decision")
    out("H6  Staffing for a surge uniform on "
        f"{h6['surge'][0]}-{h6['surge'][1]}")
    out(f"    target: day service level >= {h6['target']} on at least "
        f"{h6['coverage']:.0%} of {h6['days']} launch days; estimated "
        f"redial probability {h6['p_hat']:.3f}")
    curves = pd.DataFrame({n: pd.Series(c) for n, c in h6["curves"].items()})
    curves.index = curves.index.astype(int)
    table(curves.sort_index().round(3))
    table(pd.DataFrame(h6["chosen"]).T)

    (RESULTS / "summary.txt").write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
