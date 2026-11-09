"""Part 1 of the study: what the train-period x test-period matrix looks like.

    python matrix_sims.py --reps 300 --workers 2

For each scenario in ``dgp.scenarios`` (600 rows, one break at row 300 where there is
one), an AR(1) is fitted on each of ten 60-row blocks and scored on every other block
(6-row gap). The matrices are averaged over replications and compared with the
population matrix implied by the true parameters (no estimation noise).

Writes ``results/matrix_sims.npz`` and ``results/matrix_summary.txt``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from backward_eval import (
    ARGaussian,
    ar1_moments,
    embed_ar,
    population_excess_nlpd,
    transfer_matrix,
    variance_ratio_excess,
)
from dgp import LABELS, scenarios, simulate

ROWS, BREAK_ROW, BLOCK, GAP, P = 600, 300, 60, 6, 1
K = ROWS // BLOCK
OUT = Path(__file__).parent / "results"


def one(args: tuple[str, int]) -> dict[str, np.ndarray]:
    name, seed = args
    rng = np.random.default_rng(seed)
    mu, phi, sigma = scenarios(ROWS + P, BREAK_ROW + P)[name]()
    y = simulate(mu, phi, sigma, rng)
    X, z, _ = embed_ar(y, P)
    res = transfer_matrix(X, z, ARGaussian, BLOCK, gap=GAP, folds=5, seed=seed)
    return {
        "nlpd": res.excess("nlpd"),
        "sqerr": res.excess("sqerr"),
        "crps": res.excess("crps"),
        "cover90": res.mean_matrix("cover90"),
        "zstat_last_row": np.array([res.paired(K - 1, j)[0] / res.paired(K - 1, j)[1] for j in range(K - 1)]),
    }


def population(name: str) -> np.ndarray:
    """Population excess log loss using each block's average parameters."""
    mu, phi, sigma = scenarios(ROWS + P, BREAK_ROW + P)[name]()
    mu, phi, sigma = mu[P:], phi[P:], sigma[P:]
    par = []
    for k in range(K):
        sl = slice(k * BLOCK, (k + 1) * BLOCK)
        m, f, s = mu[sl].mean(), phi[sl].mean(), sigma[sl].mean()
        par.append((np.array([m * (1 - f), f]), s, ar1_moments(m, f, s)))
    E = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            E[i, j] = population_excess_nlpd(par[i][0], par[i][1], par[j][0], par[j][1], par[j][2])
    return E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261109)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    names = list(scenarios(ROWS + P, BREAK_ROW + P))
    store: dict[str, np.ndarray] = {}
    lines = [
        f"Train-period x test-period matrices, AR(1), {ROWS} rows, {K} blocks of {BLOCK}, gap {GAP}, "
        f"break at row {BREAK_ROW}, {args.reps} replications",
        "Excess log loss = mean NLPD of block i's model on block j minus the within-block "
        "cross-validated NLPD on block j (nats per forecast).",
        "",
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for si, name in enumerate(names):
            seeds = [args.seed + 100_000 * si + r for r in range(args.reps)]
            outs = list(ex.map(one, [(name, s) for s in seeds], chunksize=8))
            for key in outs[0]:
                arr = np.stack([o[key] for o in outs])
                store[f"{name}__{key}__mean"] = arr.mean(axis=0)
                store[f"{name}__{key}__sd"] = arr.std(axis=0, ddof=1)
            pop = population(name)
            store[f"{name}__population_nlpd"] = pop
            E = store[f"{name}__nlpd__mean"]
            S = store[f"{name}__sqerr__mean"]
            C = store[f"{name}__cover90__mean"]
            off = ~np.eye(K, dtype=bool)
            pre, post = slice(0, K // 2), slice(K // 2, K)
            lines.append(f"[{name}] {LABELS[name]}")
            lines.append(f"  mean off-diagonal excess NLPD: {E[off].mean():.3f} (population {pop[off].mean():.3f})")
            lines.append(f"  within regime (both blocks before or both after): "
                         f"{np.nanmean(np.r_[E[pre, pre][~np.eye(K // 2, dtype=bool)], E[post, post][~np.eye(K // 2, dtype=bool)]]):.3f}")
            lines.append(f"  early model on late blocks (forward across the break): {E[pre, post].mean():.3f} "
                         f"(population {pop[pre, post].mean():.3f})")
            lines.append(f"  late model on early blocks (backward across the break): {E[post, pre].mean():.3f} "
                         f"(population {pop[post, pre].mean():.3f})")
            lines.append(f"  squared-error ratio, forward across: {S[pre, post].mean():.3f}; backward across: {S[post, pre].mean():.3f}")
            lines.append(f"  90% coverage, forward across: {C[pre, post].mean():.3f}; backward across: {C[post, pre].mean():.3f}; "
                         f"within regime: {np.mean(np.r_[C[pre, pre][~np.eye(K // 2, dtype=bool)], C[post, post][~np.eye(K // 2, dtype=bool)]]):.3f}; "
                         f"native diagonal: {np.diag(C).mean():.3f}")
            last = E[K - 1, : K - 1]
            lines.append("  last row (today's model on past blocks, oldest first): " + " ".join(f"{v:.3f}" for v in last))
            zl = store[f"{name}__zstat_last_row__mean"]
            lines.append("  mean z statistic on the last row, oldest first: " + " ".join(f"{v:.1f}" for v in zl))
            lines.append("")

    lines.append("Closed form for a pure variance change (right mean, wrong variance), excess = (r - 1 - log r)/2, "
                 "r = test variance / training variance:")
    for r in (4.0, 0.25, 2.0, 0.5):
        lines.append(f"  r = {r:g}: {variance_ratio_excess(r):.3f}")
    np.savez_compressed(OUT / "matrix_sims.npz", **store)
    (OUT / "matrix_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
