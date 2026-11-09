"""Forward (pseudo out-of-sample) comparison of training-window schemes.

At every origin ``o`` each scheme fits an AR model by least squares on a window of rows
before ``o`` and issues a Gaussian forecast for row ``o``. Window schemes:

* ``expanding``   all rows before the origin
* ``rolling_w``   the last ``w`` rows
* ``avew``        equal-weight linear pool of the forecasts from every window length on
                  a grid (Pesaran and Pick, 2011, averaged point forecasts; we pool the
                  predictive densities so that the log score is defined)
* ``backward``    the window chosen by ``backward_eval.backward_window`` (re-chosen
                  every ``reselect`` origins; the start date is kept in between)
* ``roc``         the window chosen by ``backward_eval.reverse_cusumsq_window``
* ``forward_cv``  the window length whose rolling forecasts had the lowest log loss over
                  the last ``val_len`` origins (a simple version of the cross-validation
                  window choice in Pesaran and Timmermann, 2007), re-chosen like ``backward``
* ``oracle``      (simulations only) rows since the last true break

Least squares fits for many windows use cumulative cross-products, so a fit is O(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from backward_eval import ARGaussian, LOG_2PI, Z90, backward_window, reverse_cusumsq_window


class CumOLS:
    """Least squares on any window ``[a, b)`` of rows from cumulative cross-products."""

    def __init__(self, X: np.ndarray, z: np.ndarray) -> None:
        self.X, self.z = X, z
        n, k = X.shape
        self.k = k
        self.Cxx = np.zeros((n + 1, k, k))
        self.Cxx[1:] = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
        self.Cxz = np.zeros((n + 1, k))
        self.Cxz[1:] = np.cumsum(X * z[:, None], axis=0)
        self.Czz = np.zeros(n + 1)
        self.Czz[1:] = np.cumsum(z * z)

    def fit(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(a)
        b = np.asarray(b)
        XtX = self.Cxx[b] - self.Cxx[a]
        Xtz = self.Cxz[b] - self.Cxz[a]
        zz = self.Czz[b] - self.Czz[a]
        beta = np.linalg.solve(XtX, Xtz[..., None])[..., 0]
        rss = zz - 2.0 * np.einsum("...k,...k->...", beta, Xtz) + np.einsum(
            "...k,...kl,...l->...", beta, XtX, beta
        )
        dof = np.maximum((b - a) - self.k, 1)
        s = np.sqrt(np.maximum(rss, 1e-12) / dof)
        return beta, s


@dataclass
class SchemeOutput:
    nlpd: np.ndarray
    sqerr: np.ndarray
    pit: np.ndarray
    window: np.ndarray  # rows used at each origin (for pooled schemes: nan)

    @property
    def cover90(self) -> np.ndarray:
        return ((self.pit >= 0.05) & (self.pit <= 0.95)).astype(float)


def _gauss(z: np.ndarray, m: np.ndarray, s: np.ndarray) -> SchemeOutput:
    w = (z - m) / s
    return SchemeOutput(
        nlpd=0.5 * LOG_2PI + np.log(s) + 0.5 * w**2,
        sqerr=(z - m) ** 2,
        pit=norm.cdf(w),
        window=np.full(len(z), np.nan),
    )


def run_window_scheme(cum: CumOLS, origins: np.ndarray, starts: np.ndarray) -> SchemeOutput:
    """Forecast row ``o`` from rows ``[start, o)`` for each origin."""
    beta, s = cum.fit(starts, origins)
    m = np.einsum("ok,ok->o", cum.X[origins], beta)
    out = _gauss(cum.z[origins], m, s)
    out.window = (origins - starts).astype(float)
    return out


def run_avew(cum: CumOLS, origins: np.ndarray, w_min: int, step: int) -> SchemeOutput:
    """Equal-weight linear pool over window lengths ``w_min, w_min + step, ..., o``."""
    nl, se, pit = (np.empty(len(origins)) for _ in range(3))
    for idx, o in enumerate(origins):
        ws = np.arange(w_min, o + 1, step)
        if ws[-1] != o:
            ws = np.append(ws, o)
        a = o - ws
        beta, s = cum.fit(a, np.full(len(a), o))
        m = beta @ cum.X[o]
        zt = cum.z[o]
        dens = norm.pdf(zt, m, s)
        nl[idx] = -np.log(max(dens.mean(), 1e-300))
        se[idx] = (zt - m.mean()) ** 2
        pit[idx] = norm.cdf(zt, m, s).mean()
    return SchemeOutput(nlpd=nl, sqerr=se, pit=pit, window=np.full(len(origins), np.nan))


def forward_cv_starts(
    cum: CumOLS, origins: np.ndarray, candidates: tuple[int, ...], val_len: int, reselect: int
) -> tuple[np.ndarray, list]:
    """Window start per origin from forward validation over the last ``val_len`` origins.

    ``candidates`` are window lengths in rows; 0 means "all rows" (expanding).
    """
    k = cum.k
    starts = np.empty(len(origins), dtype=int)
    details: list = []
    best = 0
    for idx, o in enumerate(origins):
        if idx % reselect == 0:
            vo = np.arange(max(o - val_len, k + 2), o)
            scores = []
            for w in candidates:
                st = np.zeros(len(vo), dtype=int) if w == 0 else np.maximum(vo - w, 0)
                ok = (vo - st) > k + 1
                if ok.sum() < len(vo) // 2:
                    scores.append(np.inf)
                    continue
                beta, sd = cum.fit(st[ok], vo[ok])
                m = np.einsum("ok,ok->o", cum.X[vo[ok]], beta)
                w_ = (cum.z[vo[ok]] - m) / sd
                scores.append(float(np.mean(np.log(sd) + 0.5 * w_**2)))
            best = candidates[int(np.argmin(scores))]
            details.append((int(o), int(o) if best == 0 else min(best, int(o)), None))
        starts[idx] = 0 if best == 0 else max(int(o) - best, 0)
    return starts, details


def selected_starts(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    rule: str,
    reselect: int,
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    folds: int,
    seed: int,
    roc_min: int,
) -> tuple[np.ndarray, list]:
    """Window start for each origin under a data-driven rule re-chosen every
    ``reselect`` origins (the start date is held fixed in between)."""
    rng = np.random.default_rng(seed)
    starts = np.empty(len(origins), dtype=int)
    details: list = []
    current = 0
    for idx, o in enumerate(origins):
        if idx % reselect == 0:
            if rule == "backward":
                ch = backward_window(
                    X, z, int(o), ARGaussian, block_len, max_back, z_crit, gap=gap, folds=folds, rng=rng
                )
                current = int(o) - ch.rows
                details.append((int(o), ch.rows, ch.stopped_at))
            elif rule == "roc":
                rows = reverse_cusumsq_window(X, z, int(o), roc_min, int(o))
                current = int(o) - rows
                details.append((int(o), rows, None))
            else:
                raise ValueError(rule)
        starts[idx] = current
    return starts, details


def run_all(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    rolling: tuple[int, ...],
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    reselect: int = 12,
    folds: int = 5,
    seed: int = 0,
    avew_min: int | None = None,
    avew_step: int = 12,
    roc_min: int | None = None,
    oracle_starts: np.ndarray | None = None,
    cv_len: int | None = None,
    extra_z: tuple[float, ...] = (),
) -> tuple[dict[str, SchemeOutput], dict[str, list]]:
    """Every scheme on the same origins. ``backward`` uses ``z_crit``; each value in
    ``extra_z`` adds a ``backward_z{value:g}`` variant with the same blocks."""
    cum = CumOLS(X, z)
    out: dict[str, SchemeOutput] = {}
    details: dict[str, list] = {}
    out["expanding"] = run_window_scheme(cum, origins, np.zeros(len(origins), dtype=int))
    for w in rolling:
        out[f"rolling_{w}"] = run_window_scheme(cum, origins, np.maximum(origins - w, 0))
    out["avew"] = run_avew(cum, origins, avew_min or block_len, avew_step)
    for name, zc in [("backward", z_crit)] + [(f"backward_z{v:g}", v) for v in extra_z]:
        starts, det = selected_starts(X, z, origins, "backward", reselect, block_len, max_back, zc,
                                      gap, folds, seed, 0)
        out[name] = run_window_scheme(cum, origins, starts)
        details[name] = det
    starts, det = selected_starts(X, z, origins, "roc", reselect, block_len, max_back, z_crit, gap,
                                  folds, seed, roc_min or block_len // 2)
    out["roc"] = run_window_scheme(cum, origins, starts)
    details["roc"] = det
    cv_starts, cv_det = forward_cv_starts(cum, origins, tuple(rolling) + (0,), cv_len or block_len, reselect)
    out["forward_cv"] = run_window_scheme(cum, origins, cv_starts)
    details["forward_cv"] = cv_det
    if oracle_starts is not None:
        out["oracle"] = run_window_scheme(cum, origins, oracle_starts)
    return out, details


# ------------------------------------------------- any model (slow path, real data)


def _score(zv: float, mean: float, scale: float, df: float) -> tuple[float, float, float]:
    w = (zv - mean) / scale
    if np.isinf(df):
        return 0.5 * LOG_2PI + np.log(scale) + 0.5 * w * w, (zv - mean) ** 2, float(norm.cdf(w))
    from scipy.stats import t as student_t

    return float(np.log(scale) - student_t.logpdf(w, df)), (zv - mean) ** 2, float(student_t.cdf(w, df))


def _density(zv: float, mean: np.ndarray, scale: np.ndarray, df: float) -> tuple[float, float]:
    """Mean density and mean CDF of an equal-weight pool at ``zv``."""
    w = (zv - mean) / scale
    if np.isinf(df):
        return float(np.mean(norm.pdf(w) / scale)), float(np.mean(norm.cdf(w)))
    from scipy.stats import t as student_t

    return float(np.mean(student_t.pdf(w, df) / scale)), float(np.mean(student_t.cdf(w, df)))


def run_generic(
    X: np.ndarray,
    z: np.ndarray,
    origins: np.ndarray,
    model_factory,
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int,
    rolling: tuple[int, ...] = (120,),
    reselect: int = 12,
    folds: int = 5,
    seed: int = 0,
    avew_step: int = 12,
    cv_len: int | None = None,
) -> dict[str, SchemeOutput]:
    """The main schemes for any model (refitted at every origin; slow but general)."""
    n_o = len(origins)

    def window(starts: np.ndarray) -> SchemeOutput:
        nl, se, pit = np.empty(n_o), np.empty(n_o), np.empty(n_o)
        for i, (o, a) in enumerate(zip(origins, starts)):
            m = model_factory().fit(X[a:o], z[a:o])
            mean, sc = m.predict(X[o : o + 1])
            nl[i], se[i], pit[i] = _score(z[o], mean[0], sc[0], getattr(m, "df", np.inf))
        out = SchemeOutput(nlpd=nl, sqerr=se, pit=pit, window=(origins - starts).astype(float))
        return out

    out: dict[str, SchemeOutput] = {}
    out["expanding"] = window(np.zeros(n_o, dtype=int))
    for w in rolling:
        out[f"rolling_{w}"] = window(np.maximum(origins - w, 0))

    # equal-weight pool over window lengths
    nl, se, pit = np.empty(n_o), np.empty(n_o), np.empty(n_o)
    for i, o in enumerate(origins):
        ws = np.arange(block_len, o + 1, avew_step)
        if ws[-1] != o:
            ws = np.append(ws, o)
        means, scales, df = [], [], np.inf
        for w in ws:
            m = model_factory().fit(X[o - w : o], z[o - w : o])
            mu, sc = m.predict(X[o : o + 1])
            means.append(mu[0])
            scales.append(sc[0])
            df = getattr(m, "df", np.inf)
        dens, cdf = _density(z[o], np.array(means), np.array(scales), df)
        nl[i], se[i], pit[i] = -np.log(max(dens, 1e-300)), (z[o] - np.mean(means)) ** 2, cdf
    out["avew"] = SchemeOutput(nlpd=nl, sqerr=se, pit=pit, window=np.full(n_o, np.nan))

    # backward rule
    rng = np.random.default_rng(seed)
    starts = np.empty(n_o, dtype=int)
    cur = 0
    for i, o in enumerate(origins):
        if i % reselect == 0:
            ch = backward_window(X, z, int(o), model_factory, block_len, max_back, z_crit, gap=gap,
                                 folds=folds, rng=rng)
            cur = int(o) - ch.rows
        starts[i] = cur
    out["backward"] = window(starts)

    # forward validation over the last cv_len origins
    cands = tuple(rolling) + (0,)
    cv_len = cv_len or block_len
    best = 0
    for i, o in enumerate(origins):
        if i % reselect == 0:
            vo = np.arange(o - cv_len, o)
            sc_ = []
            for w in cands:
                tot = 0.0
                for v in vo:
                    a = 0 if w == 0 else max(v - w, 0)
                    m = model_factory().fit(X[a:v], z[a:v])
                    mu, s = m.predict(X[v : v + 1])
                    tot += _score(z[v], mu[0], s[0], getattr(m, "df", np.inf))[0]
                sc_.append(tot)
            best = cands[int(np.argmin(sc_))]
        starts[i] = 0 if best == 0 else max(int(o) - best, 0)
    out["forward_cv"] = window(starts.copy())
    return out
