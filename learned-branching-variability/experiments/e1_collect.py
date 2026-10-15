"""Experiment 1: record full strong-branching scores at sampled nodes.

The default solver (reliability pseudocost branching) drives the search; at a
random half of the nodes, up to 40 per instance, every candidate is scored by
full strong branching without changing the solver state.

Outputs
  results/samples_{train,valid}.npz   features X, score y, the two child gains,
                                      infeasibility flags, cutoff gap, node id per row
  results/e1_log.txt                  counts and timings
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import get_context

import numpy as np

import config
from learnbranch.solve import collect


def _task(args):
    split, k, quick = args
    inst = config.instance(split, k, quick)
    max_nodes = config.QUICK_COLLECT_MAX_NODES if quick else config.COLLECT_MAX_NODES
    t0 = time.perf_counter()
    samples = collect(inst, seed=0, rng=np.random.default_rng([7, k, len(split)]),
                      prob=config.COLLECT_PROB, max_nodes=max_nodes)
    return split, k, samples, time.perf_counter() - t0


def main(quick: bool = False):
    suffix = "_quick" if quick else ""
    lines = []
    tasks = [(s, k, quick) for s in ["train", "valid"] for k in range(config.n_instances(s, quick))]
    t0 = time.perf_counter()
    out = {"train": [], "valid": []}
    secs = {"train": 0.0, "valid": 0.0}
    with get_context("fork").Pool(config.WORKERS) as pool:
        for split, k, samples, sec in pool.imap_unordered(_task, tasks):
            out[split].append((k, samples))
            secs[split] += sec
            print(f"{split} instance {k}: {len(samples)} nodes, {sec:.0f}s", flush=True)
    for split, items in out.items():
        X, sb, node, inst = [], [], [], []
        nid = 0
        for k, samples in sorted(items, key=lambda t: t[0]):
            for Xs, info in samples:
                X.append(Xs)
                sb.append(info)
                node.append(np.full(len(Xs), nid))
                inst.append(np.full(len(Xs), k))
                nid += 1
        X = np.vstack(X).astype(np.float32)
        cols = {k: np.concatenate([d[k] for d in sb]) for k in sb[0]}
        y = cols.pop("score")
        node, inst = np.concatenate(node), np.concatenate(inst)
        np.savez_compressed(config.RESULTS / f"samples_{split}{suffix}.npz",
                            X=X, y=y, node=node, instance=inst, **cols)
        msg = (f"{split}: {len(items)} instances, {nid} nodes, {len(y)} candidate rows, "
               f"{len(y) / max(nid, 1):.1f} candidates per node, "
               f"{secs[split] / 60:.1f} CPU-minutes")
        print(msg)
        lines.append(msg)
    lines.append(f"wall time {time.perf_counter() - t0:.0f}s with {config.WORKERS} workers")
    (config.RESULTS / f"e1_log{suffix}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
