"""Experiment 2: fit the branching scorer and measure imitation quality.

A gradient-boosted regressor maps candidate features to a strong-branching
score normalised within the node. Two targets are compared on the validation
instances and the one whose chosen candidates have the higher strong-branching
score (relative to the node's best) is kept.

Imitation is reported against three cheap rules that need no training: the
pseudocost score, the most fractional variable, and a random choice.

Outputs
  results/model.joblib    the fitted regressor (rebuilt by this script)
  results/e2_metrics.json validation metrics per scorer
  results/e2_log.txt      the same, readable
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from learnbranch.branching import FEATURES


def node_groups(node: np.ndarray):
    order = np.argsort(node, kind="stable")
    bounds = np.flatnonzero(np.diff(node[order])) + 1
    return np.split(order, bounds)


def targets(y, node, kind):
    t = np.empty_like(y, dtype=float)
    for g in node_groups(node):
        r = y[g] / y[g].max()
        t[g] = r if kind == "ratio" else np.clip(np.log(np.maximum(r, 1e-12)), -12, 0)
    return t


def decision_metrics(pred, y, node, rng=None):
    top1 = top5 = 0
    ratio = []
    groups = node_groups(node)
    for g in groups:
        p = pred[g] if rng is None else rng.random(len(g))
        top = y[g] >= y[g].max() * (1 - 1e-9)      # ties at the best score all count
        chosen = np.argmax(p)
        top1 += top[chosen]
        top5 += top[np.argsort(-p)[:5]].any()
        ratio.append(y[g][chosen] / y[g].max())
    n = len(groups)
    return dict(nodes=n, top1=top1 / n, top5=top5 / n, mean_score_ratio=float(np.mean(ratio)),
                median_score_ratio=float(np.median(ratio)))


def main(quick: bool = False):
    sfx = "_quick" if quick else ""
    tr = np.load(config.RESULTS / f"samples_train{sfx}.npz")
    va = np.load(config.RESULTS / f"samples_valid{sfx}.npz")
    lines, metrics, models = [], {}, {}
    for kind in ["ratio", "logratio"]:
        reg = HistGradientBoostingRegressor(max_iter=60 if quick else 300, learning_rate=0.05,
                                            max_leaf_nodes=31, l2_regularization=1.0,
                                            random_state=0)
        reg.fit(tr["X"], targets(tr["y"], tr["node"], kind))
        models[kind] = reg
        metrics[f"learned_{kind}"] = decision_metrics(reg.predict(va["X"]), va["y"], va["node"])
    j = FEATURES.index
    metrics["pseudocost"] = decision_metrics(va["X"][:, j("ps_score_rel")], va["y"], va["node"])
    metrics["most_fractional"] = decision_metrics(va["X"][:, j("frac_dist")], va["y"], va["node"])
    metrics["random"] = decision_metrics(None, va["y"], va["node"], rng=np.random.default_rng(0))
    best = max(["ratio", "logratio"], key=lambda k: metrics[f"learned_{k}"]["mean_score_ratio"])
    metrics["chosen_target"] = best
    metrics["train_rows"], metrics["train_nodes"] = int(len(tr["y"])), int(len(np.unique(tr["node"])))
    joblib.dump(models[best], config.RESULTS / f"model{sfx}.joblib")
    ties = [np.sum(va["y"][g] >= va["y"][g].max() * (1 - 1e-9)) for g in node_groups(va["node"])]
    metrics["valid_nodes_with_tied_best"] = float(np.mean(np.array(ties) > 1))
    metrics["valid_median_tied_best"] = float(np.median(ties))
    lines.append(f"training rows {metrics['train_rows']:,} from {metrics['train_nodes']:,} nodes; "
                 f"validation nodes {metrics['random']['nodes']:,}; nodes where several candidates "
                 f"tie for the best score {metrics['valid_nodes_with_tied_best']:.1%}")
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"  {k:18s} picks a best candidate {v['top1']:.1%}  best in its top 5 {v['top5']:.1%}  "
                         f"chosen/best SB score: mean {v['mean_score_ratio']:.3f}, "
                         f"median {v['median_score_ratio']:.3f}")
    lines.append(f"kept target: {best}")
    print("\n".join(lines))
    (config.RESULTS / f"e2_log{sfx}.txt").write_text("\n".join(lines) + "\n")
    (config.RESULTS / f"e2_metrics{sfx}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
