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

    # ------------------------------------------------------------------ R1
    if (RESULTS / "r1_data.json").exists():
        r1 = load("r1_data")
        out("R1  The data")
        for k in ("rows_read", "malformed_rows", "calls", "days",
                  "working_days_kept", "distinct_ids", "ids_over_200_calls"):
            out(f"    {k}: {r1[k]}")
        out("    outcomes: " + ", ".join(f"{k} {v}" for k, v in
                                        r1["outcomes"].items()))
        for k in ("share_with_id", "share_with_id_queued",
                  "share_with_id_abandoned",
                  "share_of_id_calls_from_ids_over_200", "offered_per_day",
                  "abandon_share", "service_level_20s", "mean_agents",
                  "mean_offered_per_interval", "recorded_handle_time",
                  "mean_wait_per_offered",
                  "identified_redial_share_of_offered"):
            out(f"    {k}: {r1[k]:.4f}")
        out("    calls per ID, quantiles: " + ", ".join(
            f"{k} {v:.0f}" for k, v in r1["id_calls_quantiles"].items()))
        bi = pd.DataFrame(r1["by_interval"])
        bi.index = [f"{7 + i // 2:02d}:{30 * (i % 2):02d}"
                    for i in range(len(bi))]
        out("    mean per half hour across kept days:")
        table(bi.round(2))

    # ------------------------------------------------------------------ R2
    if (RESULTS / "r2_ids.json").exists():
        r2 = load("r2_ids")
        out("R2  Calling again, from customer IDs")
        for vol, name in (("10", "IDs with at most 10 calls"),
                          ("50", "IDs with at most 50 calls"),
                          ("None", "all IDs")):
            c = pd.DataFrame(r2["curves"][vol])
            w = c.pivot(index="window_min", columns="group",
                        values=["again", "prior", "again_weighted",
                                "prior_weighted"])
            w.columns = [f"{a}_{b}" for a, b in w.columns]
            w["excess"] = w["again_abandoned"] - w["again_served"]
            w["excess_weighted"] = (w["again_weighted_abandoned"]
                                    - w["again_weighted_served"])
            n = c.groupby("group")["n"].first().to_dict()
            out(f"    {name} (queued calls: {n['abandoned']} abandoned, "
                f"{n['served']} served):")
            table(w.round(3))
        for win, rows in r2["by_volume"].items():
            out(f"    by annual calls of the ID, window {win} min:")
            table(pd.DataFrame(rows).set_index("volume_class").round(3))
        d = r2["detection"]
        out("    visibility: share of offered calls that queued "
            f"{d['queued_share_of_offered']:.3f}; ID recorded for "
            f"{d['id_share_queued']:.3f} of queued and "
            f"{d['id_share_not_queued']:.3f} of not-queued offered calls; "
            "queue share by half hour, quartiles "
            + ", ".join(f"{v:.3f}" for v in d["queue_share_quartiles"]))
        out("    queue share a callback 20 minutes later would meet: after "
            f"abandonment {d['queue_share_20min_after_abandoned']:.3f}, "
            f"after service {d['queue_share_20min_after_served']:.3f}")
        out("    queue share at the visible callbacks within an hour "
            "(IDs with at most 50 calls): after abandonment "
            f"{d['queue_share_at_visible_callbacks_abandoned']:.3f}, after "
            f"service {d['queue_share_at_visible_callbacks_served']:.3f}")
        out("    (weighted columns: each visible callback weighted by one "
            "over its half hour's queue share; meant for windows up to an "
            "hour)")
        out()
        s = r2["delay_summary"]
        out("    delay from hanging up to the next call, IDs with at most "
            f"50 calls, next call within 3 hours: n {s['n']}, median "
            f"{s['median']:.1f} min, mean {s['mean']:.1f} min, within 10 "
            f"min {s['share_within_10']:.3f}, within 30 min "
            f"{s['share_within_30']:.3f}, within 60 min "
            f"{s['share_within_60']:.3f}")
        q = np.array(r2["delay_quantiles_min"])
        out("    deciles (min): " + ", ".join(
            f"{v:.1f}" for v in q[::20]))
        out()

    # ------------------------------------------------------------------ R3
    r3 = load("r3_slopes")
    out("R3  Slopes of the surplus on abandonments, real table")
    out(f"    day-cluster bootstrap with {r3['reps']} replicates; value "
        "and sd")
    rows = []
    for series, stats in r3["series"].items():
        for name, v in stats.items():
            rows.append({"series": series, "statistic": name,
                         "value": v["value"], "sd": v["sd"]})
    f = pd.DataFrame(rows)
    table(f.pivot(index="statistic", columns="series",
                  values=["value", "sd"]).round(4))

    # ------------------------------------------------------------------ R4
    r4 = load("r4_simcheck")
    out("R4  Simulation check with a known redial probability")
    out(f"    {r4['histories']} histories of {r4['days']} days per world; "
        f"handle-time multiplier {r4['service_multiplier']:.3f}; "
        f"clustering in the clustered world rho {r4['clustering_rho']:.3f}, "
        f"sigma {r4['clustering_sigma']:.3f}")
    f = pd.DataFrame(r4["rows"])
    f["err"] = f["p_hat"] - f["p_true"]
    s = f.groupby(["world", "estimator", "p_true"]).agg(
        mean=("p_hat", "mean"), sd=("p_hat", "std"),
        rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))))
    table(s.round(3).unstack("p_true"))
    st = pd.DataFrame(r4["statistics"])
    out("    mean statistics on the simulated histories (sd in brackets "
        "is across histories):")
    names = ["lag", "lead", "lag_minus_lead", "controlled", "acf1",
             "dispersion"]
    m = st.groupby(["world", "p_true"])[names].mean().round(4)
    sd = st.groupby(["world", "p_true"])[names].std().round(4)
    table(m)
    table(sd.add_suffix(" sd"))
    out("    binding functions without clustering (statistic at each p):")
    b = pd.DataFrame({n: v["values"] for n, v in r4["bindings"].items()},
                     index=r4["bindings"]["lag"]["grid"])
    table(b.round(4))
    out("    joint response surfaces, R^2: " + ", ".join(
        f"{k} {v:.3f}" for k, v in r4["joint_r2"].items()))
    j = f[f.estimator == "joint"]
    out("    joint estimate of rho in each world (mean, sd): " + ", ".join(
        f"{w} p={p}: {g['rho_hat'].mean():.3f} ({g['rho_hat'].std():.3f})"
        for (w, p), g in j.groupby(["world", "p_true"])))
    out()

    # ------------------------------------------------------------------ R5
    if (RESULTS / "r5_estimate.json").exists():
        r5 = load("r5_estimate")
        out("R5  Redial probability estimated from the real table")
        out(f"    handle-time multiplier {r5['service_multiplier']:.3f}; "
            "90% day-cluster bootstrap intervals")
        f = pd.DataFrame(r5["rows"])
        cols = ["statistic", "statistic_sd", "p_hat", "p_lo", "p_hi",
                "rho_hat", "rho_lo", "rho_hi", "sigma_hat"]
        cols = [c for c in cols if c in f.columns]
        table(f.set_index(["delay", "estimator"])[cols].round(3))
        for _, r in f[f.estimator == "joint"].iterrows():
            out(f"    joint, {r['delay']}: observed vs fitted statistics")
            fr = pd.DataFrame({"observed": r["observed"],
                               "fitted": r["fitted"], "r2": r["r2"]})
            table(fr.round(4))
        if r5.get("id_excess"):
            out("    excess calling again after abandonment from IDs "
                "(abandoned minus served), by window in minutes, raw and "
                "weighted for visibility; columns are the ID volume cap:")
            for kind in ("raw", "weighted"):
                out(f"    {kind}")
                table(pd.DataFrame({v: e[kind] for v, e in
                                    r5["id_excess"].items()}).round(3))
        out("    binding functions without clustering, by delay:")
        for d, bs in r5["bindings"].items():
            b = pd.DataFrame({n: v["values"] for n, v in bs.items()},
                             index=bs["lag"]["grid"])
            out(f"    {d}")
            table(b.round(4))

    text = buf.getvalue()
    (RESULTS / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
