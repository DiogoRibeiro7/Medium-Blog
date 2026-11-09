"""The conjecture's premise: does a strong rejection predict a lasting change?

    python premise.py

For each series (AR(3), Gaussian errors, blocks of 60 months, gap 12; sensitivity with 36),
and for each run of three consecutive blocks k-1, k, k+1:

    R_k = z statistic of block k's model on block k-1 against block k-1's own model
    P_k = E[k-1, k+1] - E[k, k+1]   (> 0: the change persisted; < 0: it reverted)

Writes ``results/premise_triples.csv`` and ``results/premise_summary.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backward-evaluation"))

from backward_eval import ARGaussian, embed_ar, transfer_matrix  # noqa: E402
from common import IN_SAMPLE, SHORT, TEST, load  # noqa: E402

P, GAP = 3, 12
Z_CRIT = float(norm.ppf(1 - 0.05 / 8))
OUT = Path(__file__).parent / "results"


def triples(sid: str, s: pd.Series, block: int) -> list[dict]:
    X, z, t = embed_ar(s.to_numpy(), P)
    dates = s.index[t]
    res = transfer_matrix(X, z, ARGaussian, block, gap=GAP, folds=5, seed=99)
    E = res.excess("nlpd")
    recs = []
    for k in range(1, res.k - 1):
        m, se = res.paired(k, k - 1)
        a, b = res.blocks[k]
        recs.append({"series": sid, "block_len": block, "k": k, "block_start": f"{dates[a]:%Y-%m}",
                     "block_end": f"{dates[b - 1]:%Y-%m}", "R": m / se if se > 0 else np.nan,
                     "excess_on_previous": m, "P": E[k - 1, k + 1] - E[k, k + 1],
                     "old_model_on_next": E[k - 1, k + 1], "new_model_on_next": E[k, k + 1]})
    return recs


def summarize(df: pd.DataFrame, label: str) -> list[str]:
    lines = []
    for block in sorted(df.block_len.unique()):
        d = df[df.block_len == block].dropna(subset=["R", "P"])
        rho, p = spearmanr(d.R, d.P)
        strong = d[d.R > Z_CRIT]
        mid = d[(d.R > 1.645) & (d.R <= Z_CRIT)]
        weak = d[d.R <= 1.645]
        lines.append(f"  {label}, {block}-month blocks: {len(d)} triples; Spearman(R, P) = {rho:+.3f} (p = {p:.3f})")
        for nm, g in (("R > 2.498", strong), ("1.645 < R <= 2.498", mid), ("R <= 1.645", weak)):
            if len(g):
                lines.append(f"    {nm}: {len(g)} triples, change persisted (P > 0) in {(g.P > 0).sum()} "
                             f"({100 * (g.P > 0).mean():.0f}%), median P {g.P.median():+.3f}")
        top = d.sort_values("R", ascending=False).head(8)
        lines.append("    strongest rejections: " + "; ".join(
            f"{SHORT[r.series]} {r.block_start[:4]}-{r.block_end[2:4]} R {r.R:.1f} P {r.P:+.2f}" for r in top.itertuples()))
    return lines


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load()
    recs = []
    for sid in [*TEST, *IN_SAMPLE]:
        for block in (60, 36):
            recs += triples(sid, data[sid], block)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "premise_triples.csv", index=False, float_format="%.5g")
    lines = ["Premise test: rejection strength R (z of block k's model on block k-1) against persistence "
             "P = E[k-1,k+1] - E[k,k+1] (> 0: persisted). AR(3), Gaussian errors, gap 12.", ""]
    lines += summarize(df[df.series.isin(TEST)], "eight test series")
    lines += summarize(df[df.series.isin(IN_SAMPLE)], "in-sample (industrial production, CPI)")
    lines.append("")
    lines.append("Per test series, 60-month blocks (Spearman R vs P; triples with R > 2.498 persisted / total):")
    for sid in TEST:
        d = df[(df.series == sid) & (df.block_len == 60)].dropna(subset=["R", "P"])
        rho = spearmanr(d.R, d.P)[0]
        st = d[d.R > Z_CRIT]
        lines.append(f"  {SHORT[sid]}: rho {rho:+.2f}; strong {int((st.P > 0).sum())}/{len(st)}")
    (OUT / "premise_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
