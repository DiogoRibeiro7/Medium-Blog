"""Simulated series with one known kind of change, to check what the matrix reads.

    python sims.py

Each scenario is an AR(1), ``y[t] = mu[t] + phi[t] * (y[t-1] - mu[t-1]) + sigma[t] * e[t]``
with Student-t(5) shocks, 10 blocks of 120 rows, and one change in the middle:

    none       nothing changes
    level      the mean moves up by 2 (in units of the shock scale) from block 6 on
    spread     the shock scale doubles from block 6 on
    dynamics   the persistence rises from 0.2 to 0.8 from block 6 on (shock scale adjusted so
               the unconditional variance stays the same)
    episode    block 5 alone has three times the shock scale and a mean 2 higher; the series
               returns to its old behaviour afterwards

The model is an AR(1) with Student-t errors. For every scenario and replicate the script
records the excess log loss of every cell and its three parts, and summarises them over
the cells that cross the change (and over the cells that do not). Writes
``results/sims_summary.txt`` and ``results/sims_cells.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bwdiag import LinearModel, lag_rows, transfer_matrix

OUT = Path(__file__).resolve().parent / "results"
BLOCK, K, BREAK, EPISODE = 120, 10, 6, 4  # 0-based block index of the change / the episode block
SCENARIOS = ("none", "level", "spread", "dynamics", "episode")
DF = 5.0


def paths(scenario: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, phi, sigma = np.zeros(n), np.full(n, 0.5), np.ones(n)
    b = 1 + BREAK * BLOCK  # first row of block BREAK (one lag row in front)
    if scenario == "level":
        mu[b:] = 2.0
    elif scenario == "spread":
        sigma[b:] = 2.0
    elif scenario == "dynamics":
        phi[:b], phi[b:] = 0.2, 0.8
        sigma[:b] = np.sqrt(1 - 0.2**2)
        sigma[b:] = np.sqrt(1 - 0.8**2)
    elif scenario == "episode":
        a, e = 1 + EPISODE * BLOCK, 1 + (EPISODE + 1) * BLOCK
        mu[a:e], sigma[a:e] = 2.0, 3.0
    return mu, phi, sigma


def simulate(scenario: str, rng: np.random.Generator) -> np.ndarray:
    n = K * BLOCK + 1
    mu, phi, sigma = paths(scenario, n)
    e = rng.standard_t(DF, n) * np.sqrt((DF - 2) / DF)
    y = np.empty(n)
    y[0] = mu[0] + sigma[0] * e[0]
    for t in range(1, n):
        y[t] = mu[t] + phi[t] * (y[t - 1] - mu[t - 1]) + sigma[t] * e[t]
    return y


def factory() -> LinearModel:
    return LinearModel(df=DF)


def crosses(scenario: str, i: int, j: int) -> bool:
    if scenario == "episode":
        return (i == EPISODE) != (j == EPISODE)
    return (i < BREAK) != (j < BREAK)


def run(reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in SCENARIOS:
        for rep in range(reps):
            X, z, _ = lag_rows(simulate(s, rng), 1)
            M = transfer_matrix(X, z, factory, BLOCK, gap=1, folds=5, seed=rep)
            for i in range(M.k):
                for j in range(M.k):
                    if i == j:
                        continue
                    rows.append(dict(scenario=s, rep=rep, train=i, test=j, crosses=crosses(s, i, j),
                                     excess=M.excess[i, j], level=M.level[i, j], spread=M.spread[i, j],
                                     structure=M.structure[i, j]))
            if rep == 0:
                costs = M.regime_costs(4)
                regimes, _ = M.regimes(2 if s != "episode" else 3)
                rows[-1]["regimes"] = (str([(int(a), int(b)) for a, b in regimes])
                                       + "; cost of 1-4 regimes: " + ", ".join(f"{c:.1f}" for c in costs))
    return pd.DataFrame(rows)


def summarise(cells: pd.DataFrame, reps: int) -> list[str]:
    lines = [f"Simulated checks: AR(1) with Student-t(5) shocks, {K} blocks of {BLOCK} rows, {reps} replicates per scenario.",
             "Mean over cells of the excess log loss and its parts (nats per row); 'share' = part / excess.", ""]
    for s in SCENARIOS:
        c = cells[cells.scenario == s]
        for cr in (True, False):
            x = c[c.crosses == cr]
            if not len(x):
                continue
            m = x[["excess", "level", "spread", "structure"]].mean()
            tag = "cells across the change" if cr else "cells within a regime "
            share = "" if m.excess < 0.02 else (f"  shares {m.level / m.excess:.2f} / {m.spread / m.excess:.2f} / "
                                                f"{m.structure / m.excess:.2f}")
            lines.append(f"  {s:>8} {tag}: excess {m.excess:+.3f} = level {m.level:.3f} + spread {m.spread:.3f} "
                         f"+ structure {m.structure:+.3f}{share}  ({len(x)} cells)")
    lines.append("")
    lines.append("Replicate 0: best split into 2 regimes (3 for the episode; inclusive block ranges, 0-based; the change "
                 f"is at block {BREAK}, the episode is block {EPISODE}) and the cost of the best split into 1-4 regimes. "
                 "A split is always returned; whether it means anything shows in how far the cost falls:")
    for s in SCENARIOS:
        r = cells[(cells.scenario == s) & cells.get("regimes", pd.Series(dtype=object)).notna()]
        if len(r):
            lines.append(f"  {s:>8}: {r.regimes.iloc[0]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cells = run(a.reps, a.seed)
    cells.drop(columns=[c for c in ["regimes"] if c in cells]).to_csv(OUT / "sims_cells.csv.gz", index=False,
                                                                       float_format="%.5f")
    lines = summarise(cells, a.reps)
    (OUT / "sims_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
