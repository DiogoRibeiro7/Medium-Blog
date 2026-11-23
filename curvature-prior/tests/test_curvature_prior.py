"""Fast checks of the pieces the article rests on."""

import jax.numpy as jnp
import numpy as np

import embed as E
import graphs as G


def test_hyperbolic_distance_matches_the_hyperboloid():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((20, 4)) * 2.0
    r = np.linalg.norm(Z, axis=1)
    x0 = np.cosh(r)
    v = np.sinh(r)[:, None] * Z / r[:, None]
    lor = x0[:, None] * x0[None, :] - v @ v.T
    ref = np.arccosh(np.maximum(lor, 1.0))
    got = np.array(E.cross_hyper(jnp.asarray(Z), jnp.asarray(Z)))
    np.fill_diagonal(ref, 0.0)
    np.fill_diagonal(got, 0.0)
    assert np.allclose(got, ref, atol=1e-6)


def test_hyperbolic_distance_is_accurate_far_out():
    # two points at radius 11 on the same ray, one unit apart: the distance must be 1
    Z = jnp.asarray([[11.0, 0.0], [10.0, 0.0]])
    assert abs(float(E.cross_hyper(Z, Z)[0, 1]) - 1.0) < 1e-9


def test_tree_has_zero_delta_and_grid_does_not():
    assert G.gromov_delta(G.complete_tree(3, 4).dist, 20_000, 1)["delta"] == 0.0
    assert G.gromov_delta(G.grid(9).dist, 20_000, 1)["delta"] > 0.0


def test_closure_joins_every_ancestor():
    t = G.complete_tree(2, 3)
    c = G.closure(t, 0)
    # 15 nodes; each node at depth k has k ancestors: 0 + 2*1 + 4*2 + 8*3 = 34 edges
    assert c.n == 15 and c.m == 34 and c.dist.max() == 2


def test_cross_links_are_new_edges():
    t = G.complete_tree(3, 3)
    h = G.with_cross_links(t, 0.1, seed=3)
    assert h.m == t.m + round(0.1 * t.n)


def test_hyperbolic_beats_flat_on_a_small_tree_in_two_dimensions():
    t = G.complete_tree(3, 4)
    e = E.evaluate(t, E.fit(t, "euclidean", 2, seed=0, steps=600))
    h = E.evaluate(t, E.fit(t, "hyperbolic", 2, seed=0, steps=600))
    assert h["D_avg"] < e["D_avg"]


def test_flat_is_not_worse_on_a_small_grid():
    g = G.grid(10)
    e = E.evaluate(g, E.fit(g, "euclidean", 2, seed=0, steps=600))
    h = E.evaluate(g, E.fit(g, "hyperbolic", 2, seed=0, steps=600))
    assert e["D_avg"] <= h["D_avg"] + 0.01
