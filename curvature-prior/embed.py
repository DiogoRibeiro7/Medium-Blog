"""Fitting the same graph metric in flat and in hyperbolic space.

Both geometries are fitted the same way: every node gets a point, and the loss is the mean
squared relative error between embedded and graph distances, (u * d_emb / d_graph - 1)^2,
over a fixed sample of pairs that includes every edge. The scale u is learned too; in
hyperbolic space, where the curvature is fixed at -1, learning u is the same as learning the
curvature, so each geometry picks its own best scale.

Hyperbolic points are stored in polar form, as tangent vectors z at the origin of the
hyperbolic space (radius r = |z|, direction z / r); the map is a global chart, so plain Adam
can be used, and Adam's steps move the radius linearly. Distances use the hyperbolic law of
cosines in a form that is a sum of non-negative terms (see cross_hyper), which stays accurate
far from the origin. Radii are capped at R_MAX = 12; no fit here comes close to the cap.

Training uses all pairs (i, j), i != j, with 256 rows i per step drawn by a seeded permutation.

Quality is measured on all pairs:
  average distortion D_avg = mean |c * d_emb / d_graph - 1|, with c the scale that minimises
    the squared error (so neither geometry is penalised for its units);
  worst-case distortion = max ratio / min ratio of d_emb / d_graph;
  MAP = mean average precision of recovering each node's graph neighbours by ranking all
    other nodes by embedded distance (the reconstruction score of Nickel and Kiela, 2017).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
EPS = 1e-15


def cross_euclid(A, Z):
    d2 = jnp.maximum(jnp.sum(A * A, 1)[:, None] + jnp.sum(Z * Z, 1)[None, :] - 2.0 * A @ Z.T, 0.0)
    return jnp.sqrt(d2 + EPS)


def cross_hyper(A, Z):
    """Hyperbolic distances for points given in polar form by tangent vectors at the origin:
    radius r = |z|, direction z / r. With theta the angle between directions,
    cosh d = cosh(r1 - r2) + 2 sinh r1 sinh r2 sin^2(theta / 2), written as
    d = 2 asinh(sqrt(sinh^2((r1 - r2) / 2) + sinh r1 sinh r2 sin^2(theta / 2))),
    a sum of non-negative terms that stays accurate far from the origin."""
    ra = jnp.sqrt(jnp.sum(A * A, axis=1) + EPS)
    rz = jnp.sqrt(jnp.sum(Z * Z, axis=1) + EPS)
    sin2 = jnp.clip((1.0 - (A / ra[:, None]) @ (Z / rz[:, None]).T) / 2.0, 0.0, 1.0)
    a = jnp.sinh((ra[:, None] - rz[None, :]) / 2.0) ** 2 + jnp.sinh(ra)[:, None] * jnp.sinh(rz)[None, :] * sin2
    return 2.0 * jnp.arcsinh(jnp.sqrt(a + EPS))


CROSS = {"euclidean": cross_euclid, "hyperbolic": cross_hyper}
R_MAX = 12.0  # hyperbolic radius cap: beyond it float64 loses the small angles (see README)


@dataclass
class Fit:
    geometry: str
    d: int
    X: np.ndarray
    log_u: float
    final_loss: float


def fit(g, geometry: str, d: int, seed: int, steps: int = 1500, lr: float = 0.05, batch: int = 256) -> Fit:
    """Adam on the relative squared error, over all pairs (i, j), i != j, with `batch` rows i
    per step drawn by a seeded permutation (every row once per epoch)."""
    cross = CROSS[geometry]
    DG = jnp.asarray(g.dist)
    INV = jnp.asarray(np.where(g.dist > 0, 1.0 / np.maximum(g.dist, 1e-300), 0.0))
    rng = np.random.default_rng(seed)
    params = {"X": jnp.asarray(1e-2 * rng.standard_normal((g.n, d))), "log_u": jnp.asarray(0.0)}

    def loss_rows(p, rows):
        de = cross(p["X"][rows], p["X"])
        inv = INV[rows]
        err = (jnp.exp(p["log_u"]) * de * inv - 1.0) * (inv > 0)
        return jnp.sum(err ** 2) / jnp.sum(inv > 0)

    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(p, m, v, t, lr_t, rows):
        val, gr = jax.value_and_grad(loss_rows)(p, rows)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, gr)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, gr)
        p = jax.tree_util.tree_map(
            lambda a, mm, vv: a - lr_t * (mm / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + eps), p, m, v)
        if geometry == "hyperbolic":
            r = jnp.sqrt(jnp.sum(p["X"] ** 2, axis=1, keepdims=True) + EPS)
            p = {"X": p["X"] * jnp.minimum(1.0, R_MAX / r), "log_u": p["log_u"]}
        return p, m, v, val

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    batch = min(batch, g.n)
    order = np.empty(0, dtype=np.int64)
    for t in range(1, steps + 1):
        if len(order) < batch:
            order = np.concatenate([order, rng.permutation(g.n)])
        rows, order = order[:batch], order[batch:]
        lr_t = lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t / steps)))  # cosine to 5%
        params, m, v, val = step(params, m, v, float(t), lr_t, jnp.asarray(rows))
    full = jax.jit(lambda p: loss_rows(p, jnp.arange(g.n)))(params)
    return Fit(geometry, d, np.asarray(params["X"]), float(params["log_u"]), float(full))


def all_distances(fitted: Fit) -> np.ndarray:
    Z = jnp.asarray(fitted.X)
    D = np.array(jax.jit(CROSS[fitted.geometry])(Z, Z))
    np.fill_diagonal(D, 0.0)
    return D


def evaluate(g, fitted: Fit) -> dict:
    D = all_distances(fitted)
    iu = np.triu_indices(g.n, 1)
    r = D[iu] / g.dist[iu]
    c = r.sum() / (r * r).sum()
    davg = float(np.mean(np.abs(c * r - 1.0)))
    worst = float(r.max() / max(r.min(), 1e-300))
    aps = []
    for u, nb in enumerate(g.neighbours()):
        row = D[u].copy()
        row[u] = np.inf
        order = np.argsort(row, kind="stable")
        rank = np.empty(g.n, dtype=np.int64)
        rank[order] = np.arange(1, g.n + 1)
        rk = np.sort(rank[nb])
        aps.append(np.mean(np.arange(1, len(rk) + 1) / rk))
    return {"geometry": fitted.geometry, "d": fitted.d, "D_avg": davg, "worst": worst,
            "MAP": float(np.mean(aps)), "loss": fitted.final_loss, "log_u": fitted.log_u,
            "radius_max": float(np.max(np.linalg.norm(fitted.X, axis=1)))}  # hyperbolic: the radius itself
