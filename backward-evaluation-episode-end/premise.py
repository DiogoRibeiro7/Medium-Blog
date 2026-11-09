"""S2 of PREREGISTRATION.md: does the real-versus-financial split replicate on new data?

    python premise.py

For each test series (AR(3), Gaussian errors, 60-month blocks, gap 12) and each run of three
consecutive blocks k-1, k, k+1:

    R_k = z statistic of block k's model on block k-1 against block k-1's own model
    P_k = E[k-1, k+1] - E[k, k+1]   (> 0: the change persisted; < 0: it reverted)

as in ``../backward-evaluation-pooling/premise.py``. Predictions: among triples with R > 2.498 the
share with P > 0 is higher in group B (rates, exchange rates) than in group A (real activity), and
Spearman(R, P) is higher in group B. Writes ``results/premise_triples.csv`` and
``results/premise_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

import common
from backward_eval import ARGaussian, embed_ar, transfer_matrix

BLOCK = 60


def triples(key: str, s: pd.Series) -> list[dict]:
    X, z, t = embed_ar(s.to_numpy(), common.AR_ORDER)
    dates = s.index[t]
    res = transfer_matrix(X, z, ARGaussian, BLOCK, gap=common.GAP, folds=5, seed=99)
    E = res.excess("nlpd")
    recs = []
    for k in range(1, res.k - 1):
        m, se = res.paired(k, k - 1)
        a, b = res.blocks[k]
        recs.append({"series": key, "group": common.GROUP[key], "k": k, "block_start": f"{dates[a]:%Y-%m}",
                     "block_end": f"{dates[b - 1]:%Y-%m}", "R": m / se if se > 0 else np.nan,
                     "P": E[k - 1, k + 1] - E[k, k + 1]})
    return recs


def main() -> None:
    common.OUT.mkdir(exist_ok=True)
    data = common.load("test")
    df = pd.DataFrame([r for k, s in data.items() for r in triples(k, s)]).dropna(subset=["R", "P"])
    df.to_csv(common.OUT / "premise_triples.csv", index=False, float_format="%.5g")
    lines = ["S2: rejection strength R against persistence P on the test series (AR(3), Gaussian, 60-month blocks, gap 12).", ""]
    rho, p = spearmanr(df.R, df.P)
    lines.append(f"  all test series: {len(df)} triples; Spearman(R, P) = {rho:+.3f} (p = {p:.3f})")
    counts = {}
    for grp, label in (("A", "group A, real activity"), ("B", "group B, rates and exchange rates")):
        d = df[df.group == grp]
        rho, p = spearmanr(d.R, d.P)
        strong = d[d.R > common.Z_CRIT]
        counts[grp] = (int((strong.P > 0).sum()), int((strong.P <= 0).sum()))
        lines.append(f"  {label}: {len(d)} triples; Spearman(R, P) = {rho:+.3f} (p = {p:.3f}); R > 2.498: {len(strong)} triples, "
                     f"persisted in {counts[grp][0]}" + (f" ({100 * counts[grp][0] / len(strong):.0f}%)" if len(strong) else ""))
        for k in [k for k in common.TEST if common.GROUP[k] == grp and k in data]:
            dk = d[d.series == k]
            sk = dk[dk.R > common.Z_CRIT]
            r_k = spearmanr(dk.R, dk.P)[0] if len(dk) > 2 else np.nan
            lines.append(f"    {common.SHORT[k]}: {len(dk)} triples, rho {r_k:+.2f}; strong {int((sk.P > 0).sum())}/{len(sk)}")
    table = [[counts["B"][0], counts["B"][1]], [counts["A"][0], counts["A"][1]]]
    if min(sum(table[0]), sum(table[1])) > 0:
        odds, pf = fisher_exact(table, alternative="greater")
        lines.append(f"  one-sided Fisher exact test, persisted share B > A: p = {pf:.3f}")
    top = df.sort_values("R", ascending=False).head(10)
    lines.append("  strongest rejections: " + "; ".join(
        f"{common.SHORT[r.series]} {r.block_start[:4]}-{r.block_end[2:4]} R {r.R:.1f} P {r.P:+.2f}" for r in top.itertuples()))
    (common.OUT / "premise_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
