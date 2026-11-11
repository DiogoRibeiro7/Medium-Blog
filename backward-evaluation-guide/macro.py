"""Worked example on monthly data: US CPI inflation and industrial production growth.

    python macro.py

Data: ``../backward-evaluation/data/fred_monthly.csv`` (the first study's snapshot). Monthly
CPI inflation (annualised log changes, % per year) and industrial production growth (log
changes, % per month) from 1960; an AR(3) with Student-t(5) errors, and with Gaussian errors
for comparison; blocks of 60 months aligned to the end of the sample, a 12-month gap.

Writes ``results/macro_cells.csv`` (every cell, both distributions) and
``results/macro_summary.txt`` (today's model on the past, the best splits into regimes, and the
column of the block with the 2020 shock).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bwdiag import LinearModel, lag_rows, transfer_matrix

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DATA = HERE.parent / "backward-evaluation" / "data" / "fred_monthly.csv"
P, BLOCK, GAP = 3, 60, 12
DISTS = {"t": 5.0, "gaussian": np.inf}
NAMES = {"cpi_inflation": "CPI inflation", "ip_growth": "Industrial production growth"}


def load() -> dict[str, pd.Series]:
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    cpi = (1200 * np.log(df["CPIAUCSL"]).diff()).loc["1960-01-01":].dropna()
    ip = (100 * np.log(df["INDPRO"]).diff()).loc["1960-01-01":].dropna()
    return {"cpi_inflation": cpi, "ip_growth": ip}


def matrices() -> dict:
    out = {}
    for name, s in load().items():
        X, z, t = lag_rows(s.to_numpy(), P)
        dates = s.index[t]
        for dist, df in DISTS.items():
            M = transfer_matrix(X, z, lambda df=df: LinearModel(df=df), BLOCK, gap=GAP, folds=5, seed=2026)
            M.labels = [f"{dates[a]:%Y-%m}" for a, b in M.blocks]
            M.meta = dict(ends=[f"{dates[b - 1]:%Y-%m}" for a, b in M.blocks], first=f"{dates[0]:%Y-%m}",
                          last=f"{dates[-1]:%Y-%m}")
            out[(name, dist)] = M
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mats = matrices()
    rows, lines = [], ["US monthly series: train-period x test-period matrices of an AR(3), blocks of 60 months, "
                       "12-month gap. Excess log loss (nats per month) = level + spread + structure.", ""]
    for (name, dist), M in mats.items():
        for i in range(M.k):
            for j in range(M.k):
                if i != j and np.isfinite(M.loss[i, j]):
                    rows.append(dict(series=name, dist=dist, train=i, test=j, train_label=M.labels[i],
                                     test_label=M.labels[j], excess=M.excess[i, j], level=M.level[i, j],
                                     spread=M.spread[i, j], structure=M.structure[i, j], rows=int(M.rows[i, j])))
        lines.append(f"[{NAMES[name]}, {dist}] {M.meta['first']} to {M.meta['last']}, {M.k} blocks starting "
                     + ", ".join(M.labels))
        lines.append(M.describe())
        last = M.row(-1)
        tot = {p: float(np.mean([r[p] for r in last])) for p in ("excess", "level", "spread", "structure")}
        lines.append("  mean over the past blocks: " + ", ".join(f"{p} {v:+.3f}" for p, v in tot.items()))
        costs = M.regime_costs(6)
        lines.append("  cost of the best split into 1..6 regimes: " + ", ".join(f"{c:.1f}" for c in costs))
        for n in (2, 3, 4):
            reg, _ = M.regimes(n)
            lines.append(f"  {n} regimes: " + "; ".join(f"{M.labels[a]} to {M.meta['ends'][b]}" for a, b in reg))
        j = [k for k, lab in enumerate(M.labels) if lab <= "2020-03" <= M.meta["ends"][k]][0]
        col = M.column(j)
        cm = {p: float(np.mean([r[p] for r in col])) for p in ("excess", "level", "spread", "structure")}
        lines.append(f"  every other block's model on the block {M.labels[j]} to {M.meta['ends'][j]} (mean): "
                     + ", ".join(f"{p} {v:+.3f}" for p, v in cm.items()))
        lines.append("")
    pd.DataFrame(rows).to_csv(OUT / "macro_cells.csv", index=False, float_format="%.5f")
    (OUT / "macro_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
