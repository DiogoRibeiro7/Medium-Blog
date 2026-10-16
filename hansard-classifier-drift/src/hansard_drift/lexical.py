"""How the Chair's vocabulary changes between eras.

Two descriptions, both label-free:

* the terms that most distinguish each era's Chair from all the others, scored by the
  log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and Quinn, 2008);
* the Jensen-Shannon divergence between the eras' word distributions.

A fixed dictionary written against one era's way of speaking is only as good in another
era as the overlap between the two vocabularies, so these tables are the label-free half
of the drift argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import Config

TOKEN = r"(?u)\b[a-z][a-z']*[a-z]\b|\b[a-z]\b"


def era_counts(norm: pd.Series, era: pd.Series, eras: list[str], ngram_max: int = 2, min_count: int = 50):
    vec = CountVectorizer(token_pattern=TOKEN, ngram_range=(1, ngram_max), lowercase=False, min_df=5, dtype=np.int64)
    x = vec.fit_transform(norm.fillna("").astype(str))
    codes = pd.Categorical(era, categories=eras).codes
    ok = codes >= 0
    ind = sparse.csr_matrix((np.ones(ok.sum()), (codes[ok], np.flatnonzero(ok))), shape=(len(eras), x.shape[0]))
    counts = np.asarray((ind @ x).todense())
    terms = np.array(vec.get_feature_names_out())
    keep = counts.sum(axis=0) >= min_count
    return counts[:, keep], terms[keep]


def fightin_words(counts: np.ndarray, prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """z-scored log-odds of each term in each row against all other rows.

    counts: eras x terms. The prior is the pooled term distribution scaled to
    ``prior_strength`` pseudo-counts, so each term is shrunk towards its own pooled
    frequency and rare terms cannot top the list on a handful of occurrences.
    """
    total = counts.sum(axis=0).astype(float)
    alpha = prior_strength * total / total.sum()
    a0 = alpha.sum()
    delta = np.zeros_like(counts, dtype=float)
    z = np.zeros_like(counts, dtype=float)
    for i in range(counts.shape[0]):
        yi = counts[i].astype(float)
        yj = total - yi
        ni, nj = yi.sum(), yj.sum()
        li = np.log((yi + alpha) / (ni + a0 - yi - alpha))
        lj = np.log((yj + alpha) / (nj + a0 - yj - alpha))
        delta[i] = li - lj
        z[i] = delta[i] / np.sqrt(1.0 / (yi + alpha) + 1.0 / (yj + alpha))
    return delta, z


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def lexical_drift(norm: pd.Series, era: pd.Series, cfg: Config) -> dict[str, pd.DataFrame]:
    lx = cfg.lexical
    eras = [e for e in cfg.era_names if (era == e).any()]
    counts, terms = era_counts(norm, era, eras, ngram_max=2, min_count=int(lx["min_count"]))
    _, z = fightin_words(counts, float(lx["prior_strength"]))
    rows = []
    for i, e in enumerate(eras):
        order = np.argsort(-z[i])[: int(lx["top_n"])]
        share = counts[i] / counts[i].sum() * 10_000
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "era": e,
                    "rank": rank,
                    "term": terms[j],
                    "z": z[i, j],
                    "count": int(counts[i, j]),
                    "per_10k": share[j],
                }
            )
    top = pd.DataFrame(rows)

    uni = np.array([" " not in t for t in terms])
    jsd = pd.DataFrame(index=eras, columns=eras, dtype=float)
    for i, a in enumerate(eras):
        for j, b in enumerate(eras):
            jsd.loc[a, b] = jensen_shannon(counts[i, uni].astype(float), counts[j, uni].astype(float))
    return {"top": top, "jsd": jsd}
