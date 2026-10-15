"""Experiment 4: compare policies against the noise that seeds alone produce.

Reads results/e3_runs.csv and e2_metrics.json and reports, per test set:

* shifted geometric means (CPU seconds, shift 1; nodes, shift 100) and solved counts;
* paired ratios against SCIP's default, averaging log-time over seeds within each
  instance, with a bootstrap 95% interval over instances;
* the same ratio computed one seed at a time, as a single-seed study would report it;
* a placebo: the default compared with itself under a different seed;
* the learned rule against plain pseudocost branching, with and without the
  CPU time spent in its Python callback;
* the spread of each instance's solve time across seeds.

Outputs
  results/e4_log.txt, results/e4_summary.json
  figures/fig1_imitation.png, figures/fig2_placebo.png, figures/fig3_nodes_time.png,
  figures/fig4_seed_spread.png
"""

from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SHIFT = {"cpu": 1.0, "cpu_net": 1.0, "nodes": 100.0}
REF = "relpscost"
NAMES = {"relpscost": "SCIP default (reliability pseudocost)", "pscost": "Pseudocost",
         "learned": "Learned (imitates strong branching)", "fullstrong": "Full strong branching"}
SPLIT_NAMES = {"test_small": "500 x 1000 (training size)", "test_medium": "750 x 1000 (larger)"}

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR = {"learned": BLUE, "pscost": ORANGE, "fullstrong": AQUA}


def sgm(x, shift):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x + shift))) - shift)


def lg(x, shift):
    return np.log(np.asarray(x, float) + shift)


def paired(df, pol, metric, seeds, rng, ref=REF, ref_metric=None, B=10_000):
    """Instance-level paired log-ratio of `pol` against `ref` (seeds matched)."""
    sub = df[df.seed.isin(seeds)]
    a = sub[sub.policy == pol].pivot(index="instance", columns="seed", values=metric)
    b = sub[sub.policy == ref].pivot(index="instance", columns="seed",
                                     values=ref_metric or metric)
    a, b = a.loc[b.index, seeds], b[seeds]
    d = (lg(a.values, SHIFT[metric]) - lg(b.values, SHIFT[metric])).mean(1)
    boot = rng.choice(d, (B, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    single = {int(s): float(np.exp((lg(a[s], SHIFT[metric]) - lg(b[s], SHIFT[metric])).mean()))
              for s in seeds}
    wins = float(np.mean(a.values < b.values))
    return dict(ratio=float(np.exp(d.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                single_seed=single, share_runs_better=wins,
                instances_better=int(np.sum(d < 0)), instances=int(len(d)))


def placebo(df, metric, seeds):
    p = df[df.policy == REF].pivot(index="instance", columns="seed", values=metric)
    out = []
    for s, t in itertools.permutations(seeds, 2):
        out.append(float(np.exp((lg(p[s], SHIFT[metric]) - lg(p[t], SHIFT[metric])).mean())))
    spread = (p[seeds].max(axis=1) / p[seeds].min(axis=1)).values
    sd = float(np.mean(lg(p[seeds].values, SHIFT[metric]).std(1, ddof=1)))
    return dict(ratios=out, min=min(out), max=max(out),
                per_instance_max_over_min_median=float(np.median(spread)),
                per_instance_max_over_min_max=float(np.max(spread)),
                mean_within_instance_sd_log=sd)


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_imitation(metrics, path):
    rows = [("learned_" + metrics["chosen_target"], "Learned scorer"),
            ("pseudocost", "Pseudocost score"), ("most_fractional", "Most fractional"),
            ("random", "Random candidate")]
    fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for yy, (k, lab) in zip(range(len(rows))[::-1], rows):
        m = metrics[k]
        ax.plot([0, m["mean_score_ratio"]], [yy, yy], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(m["mean_score_ratio"], yy, s=70, color=BLUE if k.startswith("learned") else INK2,
                   edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(m["mean_score_ratio"] + 0.02, yy, f"{m['mean_score_ratio']:.2f}  (picks the best "
                f"candidate at {m['top1']:.0%} of nodes)", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows))[::-1], [r[1] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("strong-branching score of the chosen candidate / best score at the node "
                  f"(mean over {metrics['random']['nodes']:,} validation nodes)", fontsize=8.5, color=INK2)
    ax.set_title("Imitation quality on held-out instances", loc="left", fontsize=10.5, color=INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_placebo(summary, path):
    splits = [s for s in ["test_small", "test_medium"] if s in summary]
    rows = [("placebo", "Default vs default,\nanother seed"), ("learned", "Learned"),
            ("pscost", "Pseudocost"), ("fullstrong", "Full strong branching")]
    ypos = {k: len(rows) - 1 - i for i, (k, _) in enumerate(rows)}
    fig, axes = plt.subplots(1, len(splits), figsize=(12, 4.8), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, split in zip(axes, splits):
        S = summary[split]
        style(ax)
        pl = S["placebo_cpu"]
        ax.axvspan(pl["min"], pl["max"], color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.axvline(1.0, color=MUTED, lw=1, zorder=1)
        ax.scatter(pl["ratios"], ypos["placebo"] + np.linspace(-0.15, 0.15, len(pl["ratios"])),
                   s=18, color=MUTED, zorder=2)
        vals = list(pl["ratios"])
        for pol in ["learned", "pscost", "fullstrong"]:
            if pol not in S["paired_cpu"]:
                ax.text(1.0, ypos[pol], "  not run at this size", fontsize=8, color=MUTED,
                        va="center")
                continue
            r = S["paired_cpu"][pol]
            sv = list(r["single_seed"].values())
            vals += sv + [r["lo"], r["hi"]]
            ax.scatter(sv, ypos[pol] + 0.18 + np.linspace(-0.08, 0.08, len(sv)), s=22,
                       color=COLOR[pol], alpha=0.55, edgecolor="none", zorder=3)
            ax.plot([r["lo"], r["hi"]], [ypos[pol] - 0.12] * 2, color=COLOR[pol], lw=2, zorder=3)
            ax.scatter(r["ratio"], ypos[pol] - 0.12, s=70, marker="D", color=COLOR[pol],
                       edgecolor=SURFACE, lw=2, zorder=4)
        ax.set_xscale("log")
        lo, hi = min(vals) * 0.85, max(vals) * 1.15
        ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5]
        ax.set_xlim(lo, hi)
        ax.set_xticks([t for t in ticks if lo <= t <= hi], [f"{t:g}" for t in ticks if lo <= t <= hi])
        ax.minorticks_off()
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([ypos[k] for k, _ in rows], [lab for _, lab in rows], fontsize=9, color=INK)
        ax.set_xlabel("CPU time relative to SCIP default (shifted geometric mean; < 1 is faster)",
                      fontsize=8.5, color=INK2)
        ax.set_title(f"Set covering {SPLIT_NAMES[split]}", loc="left", fontsize=10.5, color=INK, pad=8)
    axes[0].scatter([], [], s=18, color=MUTED, label="seed pairs (placebo)")
    axes[0].scatter([], [], s=22, color=INK2, alpha=0.55, label="one seed at a time")
    axes[0].scatter([], [], s=70, marker="D", color=INK2, label="all five seeds, 95% interval over instances")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, frameon=False,
                   fontsize=8.5, labelcolor=INK2)
    fig.text(0.01, -0.06, "Grey band: range of the default solver's ratio against itself across "
             "seed pairs. Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_nodes_time(summary, path):
    splits = [s for s in ["test_small", "test_medium"] if s in summary]
    fig, axes = plt.subplots(1, len(splits), figsize=(10, 4.2), dpi=200)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, split in zip(axes, splits):
        S = summary[split]
        style(ax, grid="both")
        ax.axvline(1, color=MUTED, lw=1)
        ax.axhline(1, color=MUTED, lw=1)
        for pol in ["learned", "pscost", "fullstrong"]:
            if pol not in S["paired_cpu"]:
                continue
            n, t = S["paired_nodes"][pol], S["paired_cpu"][pol]
            ax.plot([n["lo"], n["hi"]], [t["ratio"]] * 2, color=COLOR[pol], lw=2)
            ax.plot([n["ratio"]] * 2, [t["lo"], t["hi"]], color=COLOR[pol], lw=2)
            ax.scatter(n["ratio"], t["ratio"], s=70, color=COLOR[pol], edgecolor=SURFACE, lw=2,
                       zorder=3)
            ax.text(n["ratio"] * 1.08, t["ratio"] * 1.06, NAMES[pol].split(" (")[0], fontsize=8.5,
                    color=INK2)
            if pol == "learned" and "learned_net_vs_default" in S:
                tn = S["learned_net_vs_default"]
                ax.plot([n["ratio"]] * 2, [tn["lo"], tn["hi"]], color=BLUE, lw=1.2, alpha=0.6)
                ax.scatter(n["ratio"], tn["ratio"], s=70, facecolor=SURFACE, edgecolor=BLUE, lw=2,
                           zorder=3)
                ax.text(n["ratio"] * 0.93, tn["ratio"], "Learned, without the\nPython callback",
                        fontsize=8, color=INK2, va="center", ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("nodes relative to SCIP default", fontsize=8.5, color=INK2)
        ax.set_ylabel("CPU time relative to SCIP default", fontsize=8.5, color=INK2)
        ax.set_title(f"Set covering {SPLIT_NAMES[split]}", loc="left", fontsize=10.5, color=INK, pad=8)
        ax.minorticks_off()
        for setter, lim in [(ax.set_xticks, ax.get_xlim()), (ax.set_yticks, ax.get_ylim())]:
            ticks = [t for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3]
                     if lim[0] <= t <= lim[1]]
            setter(ticks, [f"{t:g}" for t in ticks])
    fig.text(0.01, -0.03, "Points: paired shifted-geometric-mean ratios over all seeds; bars: "
             "95% bootstrap intervals over instances. Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_spread(summary, path):
    splits = [s for s in ["test_small", "test_medium"] if "default_cpu_by_instance" in summary[s]]
    fig, axes = plt.subplots(1, len(splits), figsize=(11, 4.4), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, split in zip(axes, splits):
        S = summary[split]
        style(ax)
        data = S["default_cpu_by_instance"]
        order = sorted(data, key=lambda i: np.median(data[i]))
        for y, i in enumerate(order):
            v = np.array(data[i]) / np.median(data[i])
            ax.plot([v.min(), v.max()], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
            ax.scatter(v, [y] * len(v), s=16, color=INK2, zorder=2)
        r = S["paired_cpu"]["learned"]["ratio"]
        ax.axvline(1.0, color=MUTED, lw=1)
        ax.axvline(r, color=BLUE, lw=2)
        ax.text(r * 0.98, -0.9, f"learned rule's average\neffect against the default, x{r:.2f}",
                color=INK2, fontsize=8, ha="right", va="top")
        ax.set_ylim(-2.6, len(order) - 0.4)
        ax.set_xscale("log")
        ticks = [0.5, 0.75, 1, 1.25, 1.5, 2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlim(0.45, 2.1)
        ax.minorticks_off()
        ax.set_yticks([])
        ax.set_xlabel("CPU time of each seed / median over the 5 seeds", fontsize=8.5, color=INK2)
        ax.set_title(f"SCIP default, set covering {SPLIT_NAMES[split]}", loc="left", fontsize=10.5,
                     color=INK, pad=8)
    axes[0].set_ylabel("instances, sorted by median CPU time", fontsize=8.5, color=INK2)
    fig.text(0.01, -0.03, "Each row is one instance solved with five seeds; the bar spans the fastest "
             "and slowest seed. Figure: Diogo Ribeiro.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(quick: bool = False):
    sfx = "_quick" if quick else ""
    df = pd.read_csv(config.RESULTS / f"e3_runs{sfx}.csv")
    # CPU time without the Python branching callback: what the same decisions would
    # cost if feature computation and inference were free (a lower bound)
    df["cpu_net"] = df.cpu - df.py_seconds
    metrics = json.loads((config.RESULTS / f"e2_metrics{sfx}.json").read_text())
    rng = np.random.default_rng(0)
    lines, summary = [], {}
    for split, g in df.groupby("split"):
        lines.append(f"\n== {split}: set covering {SPLIT_NAMES[split]}, {g.instance.nunique()} instances")
        S = dict(policies={}, paired_cpu={}, paired_nodes={})
        for pol, p in g.groupby("policy"):
            solved = (p.status == "optimal").mean()
            ms = 1000 * p.py_seconds.sum() / max(p.py_calls.sum(), 1)
            S["policies"][pol] = dict(runs=int(len(p)), solved=float(solved),
                                      sgm_cpu=sgm(p.cpu, 1), sgm_wall=sgm(p.time, 1),
                                      sgm_nodes=sgm(p.nodes, 100), median_cpu=float(p.cpu.median()),
                                      py_ms_per_call=float(ms),
                                      py_share_of_cpu=float(p.py_seconds.sum() / p.cpu.sum()))
            lines.append(f"  {pol:10s} runs {len(p):3d} | solved {solved:6.1%} | SGM CPU {sgm(p.cpu, 1):7.2f}s "
                         f"(wall {sgm(p.time, 1):7.2f}s) | SGM nodes {sgm(p.nodes, 100):8.0f}"
                         + (f" | python {ms:.2f} ms/call, {S['policies'][pol]['py_share_of_cpu']:.0%} of CPU"
                            if pol == "learned" else ""))
        seeds_all = sorted(int(s) for s in g.seed.unique())
        for pol in [p for p in ["learned", "pscost", "fullstrong"] if p in g.policy.unique()]:
            seeds = sorted(int(s) for s in g[g.policy == pol].seed.unique())
            for metric, key in [("cpu", "paired_cpu"), ("nodes", "paired_nodes")]:
                S[key][pol] = paired(g, pol, metric, seeds, rng)
            c, n = S["paired_cpu"][pol], S["paired_nodes"][pol]
            ss = ", ".join(f"{v:.2f}" for v in c["single_seed"].values())
            lines.append(f"  {pol:10s} vs default over seeds {seeds}: CPU x{c['ratio']:.2f} "
                         f"[{c['lo']:.2f}, {c['hi']:.2f}], nodes x{n['ratio']:.2f} "
                         f"[{n['lo']:.2f}, {n['hi']:.2f}] | faster on {c['instances_better']}/"
                         f"{c['instances']} instances, {c['share_runs_better']:.0%} of runs | "
                         f"one seed at a time: {ss}")
        if {"learned", "pscost"} <= set(g.policy.unique()):
            seeds = sorted(int(s) for s in g[g.policy == "learned"].seed.unique())
            S["learned_vs_pscost"] = {
                "cpu": paired(g, "learned", "cpu", seeds, rng, ref="pscost"),
                "nodes": paired(g, "learned", "nodes", seeds, rng, ref="pscost"),
                "cpu_net": paired(g, "learned", "cpu_net", seeds, rng, ref="pscost", ref_metric="cpu"),
            }
            S["learned_net_vs_default"] = paired(g, "learned", "cpu_net", seeds, rng, ref_metric="cpu")
            lp = S["learned_vs_pscost"]
            lines.append(f"  learned vs pscost: CPU x{lp['cpu']['ratio']:.2f} [{lp['cpu']['lo']:.2f}, "
                         f"{lp['cpu']['hi']:.2f}], nodes x{lp['nodes']['ratio']:.2f} "
                         f"[{lp['nodes']['lo']:.2f}, {lp['nodes']['hi']:.2f}] | faster on "
                         f"{lp['cpu']['instances_better']}/{lp['cpu']['instances']} instances | "
                         f"CPU net of the Python callback x{lp['cpu_net']['ratio']:.2f} "
                         f"[{lp['cpu_net']['lo']:.2f}, {lp['cpu_net']['hi']:.2f}]")
            ln = S["learned_net_vs_default"]
            lines.append(f"  learned net of Python callback vs default: CPU x{ln['ratio']:.2f} "
                         f"[{ln['lo']:.2f}, {ln['hi']:.2f}]")
        if len(seeds_all) > 1:
            S["placebo_cpu"] = placebo(g, "cpu", seeds_all)
            S["placebo_nodes"] = placebo(g, "nodes", seeds_all)
            pc, pn = S["placebo_cpu"], S["placebo_nodes"]
            lines.append(f"  placebo (default vs default, {len(pc['ratios'])} ordered seed pairs): CPU ratio "
                         f"{pc['min']:.2f}..{pc['max']:.2f}; nodes ratio {pn['min']:.2f}..{pn['max']:.2f}")
            lines.append(f"  default across seeds, per instance: slowest/fastest CPU median "
                         f"{pc['per_instance_max_over_min_median']:.2f}x, worst "
                         f"{pc['per_instance_max_over_min_max']:.2f}x; nodes median "
                         f"{pn['per_instance_max_over_min_median']:.2f}x, worst "
                         f"{pn['per_instance_max_over_min_max']:.2f}x; within-instance sd of "
                         f"log CPU {pc['mean_within_instance_sd_log']:.2f}")
        if len(seeds_all) > 1:
            p = g[g.policy == REF].pivot(index="instance", columns="seed", values="cpu")
            S["default_cpu_by_instance"] = {int(i): [float(v) for v in row] for i, row in p.iterrows()}
        summary[split] = S
    print("\n".join(lines))
    (config.RESULTS / f"e4_log{sfx}.txt").write_text("\n".join(lines).lstrip() + "\n")
    (config.RESULTS / f"e4_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    fig_imitation(metrics, config.FIGURES / ("fig1_quick.png" if quick else "fig1_imitation.png"))
    if all("placebo_cpu" in summary[s] for s in summary):
        fig_placebo(summary, config.FIGURES / ("fig2_quick.png" if quick else "fig2_placebo.png"))
    fig_nodes_time(summary, config.FIGURES / ("fig3_quick.png" if quick else "fig3_nodes_time.png"))
    if any("default_cpu_by_instance" in summary[s] for s in summary):
        fig_spread(summary, config.FIGURES / ("fig4_quick.png" if quick else "fig4_seed_spread.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
