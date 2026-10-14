import numpy as np
import pytest

import jax.numpy as jnp

from pinncross.dmc import run_dmc
from pinncross.pinn import PinnConfig, local_energy_fn, train_pinn
from pinncross.problems import Problem


def test_dmc_oscillator_energy_within_statistical_error():
    P = Problem("oscillator", 2)
    r = run_dmc(P, n_walkers=2000, tau=0.04, n_steps=2000, n_equil=100, seed=3)
    assert abs(r["E_potential"] - P.E0) < 5 * r["se_potential"] + 2e-3


@pytest.mark.parametrize("name,d", [("oscillator", 3), ("coupled", 4)])
def test_local_energy_of_exact_ground_state_is_constant(name, d):
    P = Problem(name, d)
    EL = local_energy_fn(lambda x: -P.phi(x, xp=jnp), P)
    X = np.random.default_rng(0).normal(scale=0.8, size=(16, d))
    vals = np.array([float(EL(jnp.asarray(x))) for x in X])
    np.testing.assert_allclose(vals, P.E0, atol=1e-10)


def test_short_pinn_run_is_finite_and_improves_on_adam_only():
    cfg = PinnConfig(width=16, depth=2, adam_steps=200, adam_batch=128,
                     lbfgs_steps=60, lbfgs_batch=256, eval_points=4000)
    r = train_pinn(Problem("oscillator", 1), cfg, seed=0)
    assert np.isfinite(r["E"])
    assert r["rel_energy_error"] < r["adam_only"]["rel_energy_error"]
    assert r["rel_energy_error"] < 1e-2
