"""Pools of the backward-window forecast and the expanding-window forecast.

Uses the matrix, losses, models and backward rule of the companion project
``../backward-evaluation`` (imported from there, not copied).

Schemes (see PREREGISTRATION.md):

* ``E``      expanding window
* ``B``      the plain backward rule: the backward window whenever the walk stops
* ``EWP``    evidence-weighted pool: ``w f_B + (1 - w) f_E`` with ``w = Phi(z* - z_crit)``,
             ``z*`` the z statistic of the block where the walk stopped
* ``P0.25``, ``P0.5``, ``P0.75``  fixed pools with that weight on ``f_B``
* ``AveW``   equal-weight pool over window lengths (Pesaran and Pick, 2011)

The walk, the window and the weight are re-chosen every ``reselect`` origins and held fixed
in between. When the walk does not stop, the backward window is the expanding window and
every pool reduces to ``E``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backward-evaluation"))

from backward_eval import ARGaussian, backward_window  # noqa: E402
from schemes import CumOLS, run_avew  # noqa: E402

FIXED = (0.25, 0.5, 0.75)
SCHEMES = ("E", "B", "EWP", "P0.25", "P0.5", "P0.75", "AveW")


def evidence_weight(z_star: float | None, z_crit: float) -> float:
    """Weight on the backward window: Phi(z* - z_crit); 0 if the walk did not stop."""
    if z_star is None:
        return 0.0
    return float(norm.cdf(z_star - z_crit))


@dataclass
class Selection:
    origin: int
    rows: int
    stopped: bool
    z_star: float | None


def select(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    model_factory,
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    reselect: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Selection]]:
    """Window start, stop flag and z* for each origin (re-chosen every ``reselect``)."""
    rng = np.random.default_rng(seed)
    starts = np.empty(len(origins), dtype=int)
    stopped = np.zeros(len(origins), dtype=bool)
    zstar = np.full(len(origins), np.nan)
    sel: list[Selection] = []
    cur = (0, False, np.nan)
    for i, o in enumerate(origins):
        if i % reselect == 0:
            ch = backward_window(X, z, int(o), model_factory, block_len, max_back, z_crit, gap=gap,
                                 folds=folds, rng=rng)
            zs = ch.stats[-1][1] if ch.stopped_at is not None else None
            cur = (int(o) - ch.rows, ch.stopped_at is not None, np.nan if zs is None else zs)
            sel.append(Selection(int(o), ch.rows, ch.stopped_at is not None, zs))
        starts[i], stopped[i], zstar[i] = cur
    return starts, stopped, zstar, sel


@dataclass
class Scores:
    nlpd: np.ndarray
    sqerr: np.ndarray
    pit: np.ndarray
    weight: np.ndarray  # weight on the backward window (nan for E and AveW)

    @property
    def cover90(self) -> np.ndarray:
        return ((self.pit >= 0.05) & (self.pit <= 0.95)).astype(float)


def _dens_cdf(zv: np.ndarray, m: np.ndarray, s: np.ndarray, df: float) -> tuple[np.ndarray, np.ndarray]:
    w = (zv - m) / s
    if np.isinf(df):
        return norm.pdf(w) / s, norm.cdf(w)
    return student_t.pdf(w, df) / s, student_t.cdf(w, df)


def pool(zv, mB, sB, mE, sE, w, df=np.inf) -> Scores:
    """Linear pool ``w f_B + (1 - w) f_E`` scored at ``zv`` (all arrays per origin)."""
    dB, cB = _dens_cdf(zv, mB, sB, df)
    dE, cE = _dens_cdf(zv, mE, sE, df)
    dens = w * dB + (1 - w) * dE
    return Scores(nlpd=-np.log(np.maximum(dens, 1e-300)), sqerr=(zv - (w * mB + (1 - w) * mE)) ** 2,
                  pit=w * cB + (1 - w) * cE, weight=np.asarray(w, dtype=float))


def run_fast(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    reselect: int = 12,
    folds: int = 5,
    seed: int = 0,
    avew_step: int = 12,
) -> tuple[dict[str, Scores], list[Selection]]:
    """All schemes for least-squares AR models with Gaussian errors (cumulative sums)."""
    cum = CumOLS(X, z)
    starts, stopped, zstar, sel = select(X, z, origins, ARGaussian, block_len, max_back, z_crit, gap,
                                         reselect, folds, seed)
    bB, sB = cum.fit(starts, origins)
    bE, sE = cum.fit(np.zeros(len(origins), dtype=int), origins)
    mB = np.einsum("ok,ok->o", X[origins], bB)
    mE = np.einsum("ok,ok->o", X[origins], bE)
    zv = z[origins]
    w_ev = np.where(stopped, norm.cdf(zstar - z_crit), 0.0)
    w_ev = np.nan_to_num(w_ev)
    out = {
        "E": pool(zv, mB, sB, mE, sE, np.zeros(len(origins))),
        "B": pool(zv, mB, sB, mE, sE, stopped.astype(float)),
        "EWP": pool(zv, mB, sB, mE, sE, w_ev),
    }
    for c in FIXED:
        out[f"P{c:g}"] = pool(zv, mB, sB, mE, sE, np.where(stopped, c, 0.0))
    av = run_avew(cum, origins, block_len, avew_step)
    out["AveW"] = Scores(nlpd=av.nlpd, sqerr=av.sqerr, pit=av.pit, weight=np.full(len(origins), np.nan))
    return out, sel


def run_generic(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    model_factory,
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    reselect: int = 12,
    folds: int = 5,
    seed: int = 0,
    avew_step: int = 12,
) -> dict[str, Scores]:
    """E, B, EWP, P0.5 and AveW for any model (refitted at every origin)."""
    starts, stopped, zstar, _ = select(X, z, origins, model_factory, block_len, max_back, z_crit, gap,
                                       reselect, folds, seed)
    n = len(origins)
    mB, sB, mE, sE = (np.empty(n) for _ in range(4))
    df = np.inf
    for i, (o, a) in enumerate(zip(origins, starts)):
        mE_ = model_factory().fit(X[:o], z[:o])
        mu, sc = mE_.predict(X[o : o + 1])
        mE[i], sE[i] = mu[0], sc[0]
        df = getattr(mE_, "df", np.inf)
        if a == 0:
            mB[i], sB[i] = mE[i], sE[i]
        else:
            mb = model_factory().fit(X[a:o], z[a:o])
            mu, sc = mb.predict(X[o : o + 1])
            mB[i], sB[i] = mu[0], sc[0]
    zv = z[origins]
    w_ev = np.nan_to_num(np.where(stopped, norm.cdf(zstar - z_crit), 0.0))
    out = {
        "E": pool(zv, mB, sB, mE, sE, np.zeros(n), df),
        "B": pool(zv, mB, sB, mE, sE, stopped.astype(float), df),
        "EWP": pool(zv, mB, sB, mE, sE, w_ev, df),
        "P0.5": pool(zv, mB, sB, mE, sE, np.where(stopped, 0.5, 0.0), df),
    }
    nl, se, pit = np.empty(n), np.empty(n), np.empty(n)
    for i, o in enumerate(origins):
        ws = np.arange(block_len, o + 1, avew_step)
        if ws[-1] != o:
            ws = np.append(ws, o)
        ms, ss = [], []
        for w in ws:
            m = model_factory().fit(X[o - w : o], z[o - w : o])
            mu, sc = m.predict(X[o : o + 1])
            ms.append(mu[0])
            ss.append(sc[0])
        d, c = _dens_cdf(np.full(len(ms), z[o]), np.array(ms), np.array(ss), df)
        nl[i], se[i], pit[i] = -np.log(max(d.mean(), 1e-300)), (z[o] - np.mean(ms)) ** 2, c.mean()
    out["AveW"] = Scores(nlpd=nl, sqerr=se, pit=pit, weight=np.full(n, np.nan))
    return out


def criteria(gains: dict[str, dict[str, float]], units: list[str]) -> dict:
    """C1-C3 of PREREGISTRATION.md over ``units`` (series or scenarios).

    ``gains[unit][scheme]`` is ``mean log loss(E) - mean log loss(scheme)``.
    """
    pos = [u for u in units if gains[u]["B"] > 0]
    neg = [u for u in units if gains[u]["B"] < 0]
    sB_pos = sum(gains[u]["B"] for u in pos)
    sE_pos = sum(gains[u]["EWP"] for u in pos)
    sB_neg = sum(gains[u]["B"] for u in neg)
    sE_neg = sum(gains[u]["EWP"] for u in neg)
    mean_ewp = float(np.mean([gains[u]["EWP"] for u in units]))
    fixed = {f"P{c:g}": float(np.mean([gains[u][f"P{c:g}"] for u in units])) for c in FIXED}
    best = max(fixed, key=fixed.get)
    return {
        "units_B_gains": pos, "units_B_loses": neg,
        "sum_G_B_where_B_gains": sB_pos, "sum_G_EWP_there": sE_pos,
        "C1_gains_kept": (sE_pos >= 0.5 * sB_pos) if pos else None,
        "sum_G_B_where_B_loses": sB_neg, "sum_G_EWP_there_": sE_neg,
        "C2_losses_removed": (sE_neg >= 0.5 * sB_neg) if neg else None,
        "mean_G_EWP": mean_ewp, "mean_G_fixed": fixed, "best_fixed": best,
        "C3_weighting_matters": mean_ewp >= fixed[best],
    }
