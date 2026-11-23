"""Curvature as a prior: flat and hyperbolic embeddings of real and controlled hierarchies.

Companion code for the Medium article that tests the claim that curved (hyperbolic)
geometry should replace "flat" AI for nested, hierarchical context. Prints every number in
the article to results/output.txt, writes results/summary.json, and draws every figure.
Calls no language model.

Sections
  1  The data: WordNet 3.0 (pinned NLTK package, verified), the mammal and food noun
     hierarchies, the mammal hierarchy's transitive closure, a complete tree and a grid;
     size, diameter and Gromov four-point delta of each.
  2  Dimension sweep: each graph embedded in flat and in hyperbolic space with 2 to 200
     dimensions; average and worst-case distortion and neighbour-ranking MAP.
  3  What the sweep says: the flat/hyperbolic distortion ratio by dimension, how many
     flat dimensions match a few hyperbolic ones, and the curvature each hyperbolic fit learned.
  4  The dial: random cross-links added to the mammal tree, from 0.5% to 50% of its size;
     the hyperbolic advantage at 5 and 50 dimensions against tree-likeness.
  5  Checks: longer training, other initialisations, and where the leaves of the complete
     tree put their parent in the flat embeddings.

Run:  python curvature_prior.py > results/output.txt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import embed as E
import figures
import graphs as G
import wordnet as W

HERE = Path(__file__).resolve().parent
DIMS = (2, 5, 10, 20, 50, 100, 200)
FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
LINK_SEEDS = (7, 8, 9)
DELTA_SAMPLES = 200_000


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def rounded(obj, sig: int = 8):
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: rounded(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, sig) for v in obj]
    return obj


def run_pair(g, d: int, seed: int = 0, steps: int = 1500) -> dict:
    out = {}
    for geo in ("euclidean", "hyperbolic"):
        out[geo] = E.evaluate(g, E.fit(g, geo, d, seed=seed, steps=steps))
    return out


def main() -> None:
    t0 = time.time()
    summary: dict = {}

    # ------------------------------------------------------------------------------
    header("1  The data")
    spec = W.contract()
    f = spec["files"][0]
    W.fetch()
    print(f"WordNet 3.0 from {f['url']}")
    print(f"  {f['bytes']:,} bytes, SHA-256 {f['sha256']} (verified)")
    print(f"  registry record {spec['record']} at {spec['repository']}@{spec['commit'][:12]}")
    path = W.fetch()
    syn = W.read_nouns(path)
    print(f"  noun synsets: {len(syn):,}")
    mammals = G.wordnet_graph("mammal")
    food = G.wordnet_graph("food")
    nodes, _, _ = W.subtree(syn, W.first_sense(path, "mammal"))
    root = nodes.index(W.first_sense(path, "mammal"))
    closure = G.closure(mammals, root)
    tree = G.complete_tree(3, 6)
    grid = G.grid(33)
    graphs = [mammals, food, closure, tree, grid]
    keys = ["mammals", "food", "closure", "tree", "grid"]
    summary["graphs"] = {}
    for k, g in zip(keys, graphs):
        dl = G.gromov_delta(g.dist, DELTA_SAMPLES, seed=1)
        extra = g.m - (g.n - 1)
        summary["graphs"][k] = {"name": g.name, "n": g.n, "m": g.m, "extra_edges": extra, **dl}
        print(f"{g.name:40s} n={g.n:5d}  edges={g.m:5d}  beyond a tree={extra:5d}  diameter={dl['diameter']:4.0f}  "
              f"delta={dl['delta']:.1f}  2*delta/diam={dl['relative']:.3f}  mean delta={dl['delta_mean']:.4f}")
    depth = mammals.dist[root]
    print(f"mammal hierarchy: root 'mammal', depth {int(depth.max())}, "
          f"{int((mammals.dist[root] == 1).sum())} children of the root, {len(mammals.neighbours()[root])} neighbours")

    # ------------------------------------------------------------------------------
    header("2  Dimension sweep")
    sweep = {}
    fits_2d = {}
    for k, g in zip(keys, graphs):
        sweep[k] = {}
        print(f"\n{g.name}")
        print(f"  {'d':>4}  {'flat D_avg':>10} {'hyp D_avg':>10} {'ratio':>6}   {'flat worst':>10} {'hyp worst':>10}   "
              f"{'flat MAP':>8} {'hyp MAP':>8}   {'hyp radius':>10}")
        for d in DIMS:
            res = {}
            for geo in ("euclidean", "hyperbolic"):
                fit = E.fit(g, geo, d, seed=0)
                res[geo] = E.evaluate(g, fit)
                if k == "tree" and d == 2:
                    fits_2d[geo] = fit
            sweep[k][d] = res
            e, h = res["euclidean"], res["hyperbolic"]
            print(f"  {d:4d}  {e['D_avg']:10.4f} {h['D_avg']:10.4f} {e['D_avg'] / h['D_avg']:6.2f}   "
                  f"{e['worst']:10.2f} {h['worst']:10.2f}   {e['MAP']:8.4f} {h['MAP']:8.4f}   {h['radius_max']:10.2f}",
                  flush=True)
    summary["sweep"] = {k: {str(d): v for d, v in s.items()} for k, s in sweep.items()}

    # ------------------------------------------------------------------------------
    header("3  What the sweep says")
    for k, g in zip(keys, graphs):
        s = sweep[k]
        ratios = [s[d]["euclidean"]["D_avg"] / s[d]["hyperbolic"]["D_avg"] for d in DIMS]
        print(f"\n{g.name}: flat/hyperbolic distortion ratio by dimension "
              + ", ".join(f"{d}: {r:.2f}" for d, r in zip(DIMS, ratios)))
        for ref in (5, 10):
            hd, hm = s[ref]["hyperbolic"]["D_avg"], s[ref]["hyperbolic"]["MAP"]
            match_d = next((d for d in DIMS if s[d]["euclidean"]["D_avg"] <= hd), None)
            match_m = next((d for d in DIMS if s[d]["euclidean"]["MAP"] >= hm), None)
            print(f"  hyperbolic d={ref:2d}: D_avg {hd:.4f}, MAP {hm:.4f}; flat dimensions to match the distortion: "
                  f"{match_d if match_d else 'none up to 200'}; to match the MAP: {match_m if match_m else 'none up to 200'}")
        best_e = min(s[d]["euclidean"]["D_avg"] for d in DIMS)
        best_h = min(s[d]["hyperbolic"]["D_avg"] for d in DIMS)
        print(f"  best D_avg over all dimensions: flat {best_e:.4f}, hyperbolic {best_h:.4f}")
        us = [float(np.exp(s[d]["hyperbolic"]["log_u"])) for d in DIMS]
        diam = float(g.dist.max())
        print("  learned curvature of the hyperbolic fit: graph steps per unit of hyperbolic distance u "
              + ", ".join(f"{d}: {u:.2f}" for d, u in zip(DIMS, us)))
        print("    curvature in graph units, -1/u^2: " + ", ".join(f"{d}: {-1 / u ** 2:.3g}" for d, u in zip(DIMS, us)))
        print("    diameter in radii of curvature, diameter/u: " + ", ".join(f"{d}: {diam / u:.1f}" for d, u in zip(DIMS, us)))

    # ------------------------------------------------------------------------------
    header("4  The dial: cross-links added to the mammal tree")
    dial = []
    for frac in FRACTIONS:
        for s_i, ls in enumerate(LINK_SEEDS):
            if frac == 0.0 and s_i > 0:
                continue
            h = G.with_cross_links(mammals, frac, seed=ls)
            dl = G.gromov_delta(h.dist, DELTA_SAMPLES, seed=1)
            row = {"fraction": frac, "link_seed": ls, "links": h.m - mammals.m, "delta_mean": dl["delta_mean"],
                   "delta": dl["delta"], "diameter": dl["diameter"], "relative": dl["relative"]}
            dims = (5, 50) if s_i == 0 else (50,)
            for d in dims:
                res = run_pair(h, d)
                row[f"d{d}"] = {geo: {m: res[geo][m] for m in ("D_avg", "MAP", "worst")} for geo in res}
            dial.append(row)
            line = (f"  links={row['links']:4d} ({frac:5.1%}) seed {ls}: diameter {dl['diameter']:2.0f}, "
                    f"mean delta {dl['delta_mean']:.3f}, max delta {dl['delta']:.1f}")
            for d in dims:
                e, hh = row[f"d{d}"]["euclidean"], row[f"d{d}"]["hyperbolic"]
                line += (f" | d={d}: D_avg flat {e['D_avg']:.4f} hyp {hh['D_avg']:.4f} ratio {e['D_avg'] / hh['D_avg']:.2f}, "
                         f"MAP flat {e['MAP']:.3f} hyp {hh['MAP']:.3f}")
            print(line, flush=True)
    summary["dial"] = dial
    print("\nratio at d=50 by fraction (mean over link placements, min to max):")
    for frac in FRACTIONS:
        rs = [r["d50"]["euclidean"]["D_avg"] / r["d50"]["hyperbolic"]["D_avg"] for r in dial if r["fraction"] == frac]
        print(f"  {frac:5.1%}: {np.mean(rs):.2f} ({min(rs):.2f} to {max(rs):.2f}), n={len(rs)}")

    # ------------------------------------------------------------------------------
    header("5  Checks")
    conv = {}
    for geo, d in (("euclidean", 200), ("hyperbolic", 50)):
        r = E.evaluate(mammals, E.fit(mammals, geo, d, seed=0, steps=4500))
        base = sweep["mammals"][d][geo]
        conv[geo] = {"d": d, "D_avg_1500": base["D_avg"], "D_avg_4500": r["D_avg"], "MAP_1500": base["MAP"], "MAP_4500": r["MAP"]}
        print(f"mammals, {geo}, d={d}: three times the steps moves D_avg {base['D_avg']:.4f} -> {r['D_avg']:.4f}, "
              f"MAP {base['MAP']:.4f} -> {r['MAP']:.4f}")
    seeds = {}
    for geo in ("euclidean", "hyperbolic"):
        vals = [E.evaluate(mammals, E.fit(mammals, geo, 10, seed=s)) for s in (0, 1, 2)]
        seeds[geo] = [{"D_avg": v["D_avg"], "MAP": v["MAP"]} for v in vals]
        print(f"mammals, {geo}, d=10, initialisations 0, 1, 2: D_avg "
              + ", ".join(f"{v['D_avg']:.4f}" for v in vals) + "; MAP " + ", ".join(f"{v['MAP']:.4f}" for v in vals))
    leaves = {}
    nb = tree.neighbours()
    deg = np.array([len(x) for x in nb])
    for d in (20, 200):
        D = E.all_distances(E.fit(tree, "euclidean", d, seed=0))
        ranks, internal, ratio = [], [], []
        for u in range(tree.n):
            row = D[u].copy()
            row[u] = np.inf
            rank = np.empty(tree.n, dtype=np.int64)
            rank[np.argsort(row, kind="stable")] = np.arange(1, tree.n + 1)
            rk = np.sort(rank[nb[u]])
            if deg[u] == 1:
                p = nb[u][0]
                sib = [v for v in nb[p] if deg[v] == 1 and v != u]
                ranks.append(int(rk[0]))
                ratio.append(float(np.mean(D[u, sib]) / D[u, p]))
            else:
                internal.append(float(np.mean(np.arange(1, len(rk) + 1) / rk)))
        counts = {str(k): int(c) for k, c in zip(*np.unique(ranks, return_counts=True))}
        leaves[f"d{d}"] = {"leaves": len(ranks), "parent_rank_counts": counts, "MAP": float(np.mean(internal)),
                           "sibling_over_parent": float(np.median(ratio)),
                           "sibling_over_parent_min": float(min(ratio)), "sibling_over_parent_max": float(max(ratio))}
        print(f"complete tree, euclidean, d={d}: rank of the parent among a leaf's nearest points "
              + ", ".join(f"{k}: {c} leaves" for k, c in counts.items())
              + f" (of {len(ranks)}); internal nodes' mean AP {np.mean(internal):.4f}; "
              f"distance to a sibling / distance to the parent, median {np.median(ratio):.2f} "
              f"({min(ratio):.2f} to {max(ratio):.2f}; 2 in the tree)")
    summary["checks"] = {"convergence": conv, "seeds": seeds, "leaves": leaves}

    # ------------------------------------------------------------------------------
    header("Figures")
    paths = [figures.figure_two_dimensions(tree, 0, fits_2d, sweep["tree"][2]),
             figures.figure_distortion(sweep, graphs, keys, DIMS),
             figures.figure_ranking(sweep, graphs, keys, DIMS),
             figures.figure_dial(dial, FRACTIONS)]
    for p in paths:
        print("  wrote", Path(p).relative_to(HERE))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(rounded(summary), indent=1) + "\n")
    print(f"\n(run time is printed to stderr)")
    import sys
    print(f"total run time {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
