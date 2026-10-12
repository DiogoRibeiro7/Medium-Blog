import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eos.mlp import Problem, init_params
from eos.sharpness import adaptive_sharpness, sharpness
from eos.symmetry import imbalance_increment, layer_imbalance, rescale


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    y = np.eye(3)[rng.integers(0, 3, 20)]
    sizes = [5, 6, 4, 3]
    params = init_params(1, sizes)
    out = {}
    for act in ["relu", "tanh"]:
        out[act] = (Problem.build(sizes, act, x, y, params), params)
    return out


def test_hvp_matches_dense_hessian(tiny):
    prob, params = tiny["tanh"]
    flat = prob.flatten(params)
    dense = np.asarray(jax.hessian(prob.loss)(flat))
    v = np.random.default_rng(1).normal(size=flat.size)
    np.testing.assert_allclose(np.asarray(prob.hvp(flat, jnp.asarray(v))), dense @ v, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_lanczos_sharpness_matches_dense(tiny, act):
    prob, params = tiny[act]
    flat = prob.flatten(params)
    dense = np.asarray(jax.hessian(prob.loss)(flat))
    s, _ = sharpness(prob, flat)
    assert s == pytest.approx(np.linalg.eigvalsh(dense)[-1], rel=1e-7)


def test_rescaling_keeps_function_and_adaptive_sharpness_but_not_sharpness(tiny):
    prob, params = tiny["relu"]
    flat = prob.flatten(params)
    flat2 = prob.flatten(rescale(params, 0, 3.0))
    np.testing.assert_allclose(np.asarray(prob.predict(flat, prob.x)), np.asarray(prob.predict(flat2, prob.x)), rtol=1e-12, atol=1e-12)
    s1, _ = sharpness(prob, flat)
    s2, _ = sharpness(prob, flat2)
    assert abs(s2 - s1) / s1 > 0.5
    a1, _ = adaptive_sharpness(prob, flat)
    a2, _ = adaptive_sharpness(prob, flat2)
    assert a2 == pytest.approx(a1, rel=1e-6)


def test_gd_changes_layer_imbalance_by_exactly_eta_squared_term(tiny):
    prob, params = tiny["relu"]
    flat = prob.flatten(params)
    eta = 0.37
    for layer in [0, 1]:
        g = prob.unravel(prob.grad(flat))
        new = prob.unravel(flat - eta * prob.grad(flat))
        lhs = layer_imbalance(new, layer) - layer_imbalance(params, layer)
        rhs = imbalance_increment(g, layer, eta)
        assert lhs == pytest.approx(rhs, rel=1e-8, abs=1e-12)


def test_gd_on_quadratic_threshold_is_two_over_lambda():
    lam = 7.0
    for eta, converges in [(1.99 / lam, True), (2.01 / lam, False)]:
        x = 1.0
        for _ in range(5000):
            x = x - eta * lam * x
        assert (abs(x) < 1e-3) == converges
