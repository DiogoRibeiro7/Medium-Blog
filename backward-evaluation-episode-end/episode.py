"""The backward rule with an end-of-episode check (PREREGISTRATION.md).

Uses the matrix, losses, models and backward walk of ``../backward-evaluation`` and the linear
pool of ``../backward-evaluation-pooling`` (imported, not copied).

At each re-selection origin ``o`` the backward walk runs as before. If it stops at block ``j``
(backward window ``[o - jL, o)``, rejected block ``[o - (j+1)L, o - jL)``):

* the *change model* is fitted on ``[o - jL, o - r)``,
* the *past model* on the rejected block,
* both score the last ``r`` rows; ``d`` = change-model log loss minus past-model log loss,
  ``z_end = mean(d) / se(d)`` (Newey-West);
* if ``z_end > c`` the change has ended and the rule trains on the expanding window, otherwise on
  the backward window.

Schemes: ``E`` expanding, ``B`` plain backward rule, ``C`` checked rule, ``P0.25`` pool of ``B``
and ``E``, ``PC0.25`` pool of ``C`` and ``E``, ``AveW`` pool over window lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import common  # noqa: F401  (puts the companion folders on sys.path)
from backward_eval import ARGaussian, backward_window, mean_and_se, row_losses
from pooling import Scores, _dens_cdf, pool
from schemes import CumOLS, run_avew

SCHEMES = ("E", "B", "C", "P0.25", "PC0.25", "AveW")


def end_check(X: np.ndarray, z: np.ndarray, origin: int, j: int, block_len: int, r: int,
              model_factory) -> tuple[float, float]:
    """``(z_end, mean d)`` for a walk that stopped at block ``j`` (``z_end > c``: the change ended)."""
    lo_w, lo_p = origin - j * block_len, origin - (j + 1) * block_len
    if lo_p < 0 or origin - r - lo_w < 10:
        return 0.0, 0.0
    change = model_factory().fit(X[lo_w : origin - r], z[lo_w : origin - r])
    past = model_factory().fit(X[lo_p:lo_w], z[lo_p:lo_w])
    rec = slice(origin - r, origin)
    lc = row_losses(z[rec], *change.predict(X[rec]), getattr(change, "df", np.inf))["nlpd"]
    lp = row_losses(z[rec], *past.predict(X[rec]), getattr(past, "df", np.inf))["nlpd"]
    m, se = mean_and_se(lc - lp)
    return (m / se if se > 0 else 0.0), float(m)


@dataclass
class Decision:
    origin: int
    stopped_at: int | None
    rows_B: int
    z_star: float | None
    z_end: dict[int, float] = field(default_factory=dict)  # by r


@dataclass
class Selection:
    start_B: np.ndarray
    stopped: np.ndarray
    z_end: dict[int, np.ndarray]  # by r, nan where the walk did not stop
    decisions: list[Decision]

    def start_C(self, r: int, c: float) -> np.ndarray:
        """Window start of the checked rule: 0 (expanding) where the change has ended."""
        ended = self.stopped & (np.nan_to_num(self.z_end[r], nan=-np.inf) > c)
        return np.where(ended, 0, self.start_B)


def select(X, z, origins, model_factory, block_len, max_back, z_crit, gap, reselect, folds, seed,
           rs=(12,)) -> Selection:
    """Plain-rule windows and end-of-episode statistics, re-chosen every ``reselect`` origins."""
    rng = np.random.default_rng(seed)
    n = len(origins)
    start_B = np.zeros(n, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    z_end = {r: np.full(n, np.nan) for r in rs}
    decisions: list[Decision] = []
    cur: tuple = (0, False, {r: np.nan for r in rs})
    for i, o in enumerate(origins):
        o = int(o)
        if i % reselect == 0:
            ch = backward_window(X, z, o, model_factory, block_len, max_back, z_crit, gap=gap, folds=folds, rng=rng)
            st = ch.stopped_at is not None
            ze = {r: (end_check(X, z, o, ch.stopped_at, block_len, r, model_factory)[0] if st else np.nan) for r in rs}
            cur = (o - ch.rows, st, ze)
            decisions.append(Decision(o, ch.stopped_at, ch.rows, ch.stats[-1][1] if st else None, ze))
        start_B[i], stopped[i] = cur[0], cur[1]
        for r in rs:
            z_end[r][i] = cur[2][r]
    return Selection(start_B, stopped, z_end, decisions)


def _uses_backward(start: np.ndarray, stopped: np.ndarray) -> np.ndarray:
    return stopped & (start > 0)


def run_fast(X, z, origins, block_len, max_back, z_crit, gap, reselect=12, folds=5, seed=0, avew_step=12,
             rs=(12,), cs=(1.645,), primary=(12, 1.645)) -> tuple[dict[str, Scores], Selection]:
    """All schemes for least-squares AR models with Gaussian errors (cumulative sums).

    ``C`` and ``PC0.25`` use ``primary``; variants ``C[r=..,c=..]`` are added for every other
    ``(r, c)`` in ``rs x cs``.
    """
    cum = CumOLS(X, z)
    sel = select(X, z, origins, ARGaussian, block_len, max_back, z_crit, gap, reselect, folds, seed, rs)
    zv = z[origins]
    n = len(origins)

    def window(starts):
        b, s = cum.fit(starts, origins)
        return np.einsum("ok,ok->o", X[origins], b), s

    mE, sE = window(np.zeros(n, dtype=int))
    mB, sB = window(sel.start_B)
    useB = _uses_backward(sel.start_B, sel.stopped)
    out = {"E": pool(zv, mB, sB, mE, sE, np.zeros(n)),
           "B": pool(zv, mB, sB, mE, sE, useB.astype(float)),
           "P0.25": pool(zv, mB, sB, mE, sE, np.where(useB, 0.25, 0.0))}
    for r in rs:
        for c in cs:
            stC = sel.start_C(r, c)
            mC, sC = window(stC)
            useC = _uses_backward(stC, sel.stopped)
            key = "C" if (r, c) == primary else f"C[r={r},c={c:g}]"
            out[key] = pool(zv, mC, sC, mE, sE, useC.astype(float))
            if (r, c) == primary:
                out["PC0.25"] = pool(zv, mC, sC, mE, sE, np.where(useC, 0.25, 0.0))
    av = run_avew(cum, origins, block_len, avew_step)
    out["AveW"] = Scores(nlpd=av.nlpd, sqerr=av.sqerr, pit=av.pit, weight=np.full(n, np.nan))
    return out, sel


def run_generic(X, z, origins, model_factory, block_len, max_back, z_crit, gap, reselect=12, folds=5, seed=0,
                avew_step=12, avew=True, rs=(12,), cs=(1.645,), primary=(12, 1.645)) -> tuple[dict[str, Scores], Selection]:
    """The same schemes for any model, refitted on its window at every origin."""
    sel = select(X, z, origins, model_factory, block_len, max_back, z_crit, gap, reselect, folds, seed, rs)
    n = len(origins)
    zv = z[origins]
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    df = np.inf

    def fc(a: int, o: int) -> tuple[float, float]:
        nonlocal df
        key = (a, o)
        if key not in cache:
            m = model_factory().fit(X[a:o], z[a:o])
            df = getattr(m, "df", np.inf)
            mu, sc = m.predict(X[o : o + 1])
            cache[key] = (float(mu[0]), float(sc[0]))
        return cache[key]

    def window(starts):
        ms = np.empty(n)
        ss = np.empty(n)
        for i, (a, o) in enumerate(zip(starts, origins)):
            ms[i], ss[i] = fc(int(a), int(o))
        return ms, ss

    mE, sE = window(np.zeros(n, dtype=int))
    mB, sB = window(sel.start_B)
    useB = _uses_backward(sel.start_B, sel.stopped)
    out = {"E": pool(zv, mB, sB, mE, sE, np.zeros(n), df),
           "B": pool(zv, mB, sB, mE, sE, useB.astype(float), df),
           "P0.25": pool(zv, mB, sB, mE, sE, np.where(useB, 0.25, 0.0), df)}
    for r in rs:
        for c in cs:
            stC = sel.start_C(r, c)
            mC, sC = window(stC)
            useC = _uses_backward(stC, sel.stopped)
            key = "C" if (r, c) == primary else f"C[r={r},c={c:g}]"
            out[key] = pool(zv, mC, sC, mE, sE, useC.astype(float), df)
            if (r, c) == primary:
                out["PC0.25"] = pool(zv, mC, sC, mE, sE, np.where(useC, 0.25, 0.0), df)
    if avew:
        nl, se, pit = np.empty(n), np.empty(n), np.empty(n)
        for i, o in enumerate(origins):
            o = int(o)
            ws = np.arange(block_len, o + 1, avew_step)
            if ws[-1] != o:
                ws = np.append(ws, o)
            pr = [fc(o - w, o) for w in ws]
            ms, ss = np.array([p[0] for p in pr]), np.array([p[1] for p in pr])
            d, cdf = _dens_cdf(np.full(len(ms), z[o]), ms, ss, df)
            nl[i], se[i], pit[i] = -np.log(max(d.mean(), 1e-300)), (z[o] - ms.mean()) ** 2, cdf.mean()
        out["AveW"] = Scores(nlpd=nl, sqerr=se, pit=pit, weight=np.full(n, np.nan))
    return out, sel


def criteria(gains: dict[str, dict[str, float]], units: list[str], scheme: str = "C") -> dict:
    """H1-H3 of PREREGISTRATION.md over ``units``; ``gains[unit][scheme]`` = loss(E) - loss(scheme)."""
    pos = [u for u in units if gains[u]["B"] > 0]
    neg = [u for u in units if gains[u]["B"] < 0]
    sB_pos, sC_pos = sum(gains[u]["B"] for u in pos), sum(gains[u][scheme] for u in pos)
    sB_neg, sC_neg = sum(gains[u]["B"] for u in neg), sum(gains[u][scheme] for u in neg)
    mean = {k: float(np.mean([gains[u][k] for u in units])) for k in gains[units[0]]}
    defaults = {k: mean[k] for k in ("AveW", "P0.25") if k in mean}
    h1 = (sC_neg >= 0.5 * sB_neg) if neg else None
    h2 = (sC_pos >= 0.5 * sB_pos) if pos else None
    h3 = (mean[scheme] >= max(defaults.values())) if defaults else None
    if h1 is None:
        verdict = "supported" if (h2 is not False and h3) else "not supported"
    elif h1 and (h2 is not False) and h3:
        verdict = "supported"
    elif h1 and (h2 is not False):
        verdict = "repairs the rule but does not beat the defaults"
    else:
        verdict = "not supported"
    return {"B_gains_on": pos, "B_loses_on": neg, "sum_B_where_B_loses": sB_neg, "sum_C_there": sC_neg,
            "H1": h1, "sum_B_where_B_gains": sB_pos, "sum_C_there_": sC_pos, "H2": h2,
            "mean": mean, "H3": h3, "verdict": verdict}
