"""A physics-informed neural network for the ground state, in JAX (float64, CPU).

The network represents the wavefunction in log form on the box [-L, L]^d,

    psi(x) = prod_k (1 - x_k^2 / L^2) * exp(NN(x)),

so psi vanishes on the walls and is positive inside. A positive solution of
H psi = E psi with those boundary values can only be the ground state, so the
positivity is what stops training from landing on an excited state.

The loss is the standard PINN residual |H psi - E psi|^2 averaged over collocation
points drawn from N(0, sigma^2 I) inside the box, with E the Rayleigh quotient on the
same points and the whole thing divided by the average of psi^2, so that shrinking psi
cannot lower it. It is computed through the local energy
E_L = H psi / psi = -(Lap u + |grad u|^2) / 2 + V with u = log psi, because
|H psi - E psi|^2 = psi^2 (E_L - E)^2. It is zero exactly when H psi = E psi on the
sampled region. (``weighting="none"`` drops the psi^2 factor, for comparison.)

Training is Adam on fresh points at every step, then L-BFGS on a fixed set of points.
The energy and the wavefunction error are measured afterwards on fresh exact draws
from |psi0|^2, independent of the training points:

    E = <psi|H|psi> / <psi|psi>       (an importance-weighted average of E_L),
    relative L2 error = min_c ||c psi - psi0|| / ||psi0||.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
import optax

from .problems import Problem

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class PinnConfig:
    width: int = 64
    depth: int = 3            # hidden layers
    L: float = 5.0            # box half-width (same as the grid solvers)
    sigma: float = 1.0        # collocation points ~ N(0, sigma^2 I), restricted to the box
    adam_steps: int = 2000
    adam_batch: int = 512
    lr_start: float = 1e-3
    lr_end: float = 1e-5
    lbfgs_steps: int = 2000
    lbfgs_batch: int = 2048
    eval_points: int = 500_000
    weighting: str = "psi2"   # "psi2": standard residual |H psi - E psi|^2; "none": |E_L - E|^2


def init_params(key, d: int, cfg: PinnConfig):
    sizes = [d] + [cfg.width] * cfg.depth + [1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (a, b)) * jnp.sqrt(1.0 / a)
        params.append((W, jnp.zeros(b)))
    return params


def n_params(params) -> int:
    return int(sum(W.size + b.size for W, b in params))


def local_energy_fn(u, problem: Problem):
    """E_L(x) = H psi / psi for psi = exp(u(x)): -(Lap u + |grad u|^2) / 2 + V(x)."""

    def EL(x):
        g = jax.grad(u)(x)
        H = jax.hessian(u)(x)
        return -0.5 * (jnp.trace(H) + g @ g) + problem.potential(x, xp=jnp)

    return EL


def make_model(problem: Problem, cfg: PinnConfig):
    L = cfg.L

    def net(params, x):
        h = x
        for W, b in params[:-1]:
            h = jnp.tanh(h @ W + b)
        W, b = params[-1]
        return (h @ W + b)[0]

    def log_psi(params, x):
        return jnp.sum(jnp.log1p(-(x / L) ** 2)) + net(params, x)

    def local_energy(params, x):
        return local_energy_fn(lambda y: log_psi(params, y), problem)(x)

    batched_EL = jax.vmap(local_energy, in_axes=(None, 0))
    batched_logpsi = jax.vmap(log_psi, in_axes=(None, 0))

    def loss(params, X):
        EL = batched_EL(params, X)
        if cfg.weighting == "none":
            return jnp.mean((EL - jnp.mean(EL)) ** 2)
        # standard residual |H psi - E psi|^2 = psi^2 (E_L - E)^2 with E the Rayleigh
        # quotient on the batch, normalised by sum psi^2 so shrinking psi cannot help
        u = batched_logpsi(params, X)
        w = jnp.exp(2.0 * (u - jax.lax.stop_gradient(jnp.max(u))))
        w = w / jnp.sum(w)
        E = jnp.sum(w * EL)
        return jnp.sum(w * (EL - E) ** 2)

    return log_psi, local_energy, batched_EL, batched_logpsi, loss


def collocation(rng: np.random.Generator, m: int, d: int, cfg: PinnConfig) -> np.ndarray:
    lim = 0.98 * cfg.L
    X = rng.normal(0.0, cfg.sigma, size=(m, d))
    bad = np.any(np.abs(X) >= lim, axis=1)
    while bad.any():
        X[bad] = rng.normal(0.0, cfg.sigma, size=(int(bad.sum()), d))
        bad = np.any(np.abs(X) >= lim, axis=1)
    return X


def evaluate(problem: Problem, cfg: PinnConfig, params, batched_EL, batched_logpsi,
             seed: int, chunk: int = 20_000) -> dict:
    rng = np.random.default_rng(10_000 + seed)
    X = problem.sample_psi0_squared(cfg.eval_points, rng)
    X = X[np.all(np.abs(X) < cfg.L, axis=1)]
    EL_f = jax.jit(batched_EL)
    lp_f = jax.jit(batched_logpsi)
    EL, logrho = [], []
    for i in range(0, X.shape[0], chunk):
        xb = jnp.asarray(X[i:i + chunk])
        EL.append(np.asarray(EL_f(params, xb)))
        logrho.append(np.asarray(lp_f(params, xb)) + problem.phi(X[i:i + chunk]))
    EL = np.concatenate(EL)
    logrho = np.concatenate(logrho)
    rho = np.exp(logrho - logrho.max())          # psi / psi0 up to a constant
    w = rho**2
    E = float(np.sum(w * EL) / np.sum(w))
    se = float(np.sqrt(np.sum(w**2 * (EL - E) ** 2)) / np.sum(w))
    l2 = float(np.sqrt(max(0.0, 1.0 - np.mean(rho) ** 2 / np.mean(rho**2))))
    return {"E": E, "se_E": se, "rel_energy_error": abs(E - problem.E0) / problem.E0,
            "rel_se": se / problem.E0, "l2_error": l2, "eval_points": int(X.shape[0])}


def train_pinn(problem: Problem, cfg: PinnConfig, seed: int = 0, log_every: int = 0) -> dict:
    d = problem.d
    rng = np.random.default_rng(seed)
    params = init_params(jax.random.PRNGKey(seed), d, cfg)
    _, _, batched_EL, batched_logpsi, loss = make_model(problem, cfg)
    history = []

    t_cpu, t_wall = time.process_time(), time.perf_counter()

    # --- stage 1: Adam on fresh collocation points -----------------------------------
    sched = optax.cosine_decay_schedule(cfg.lr_start, cfg.adam_steps,
                                        alpha=cfg.lr_end / cfg.lr_start)
    adam = optax.adam(sched)
    a_state = adam.init(params)

    @jax.jit
    def adam_step(params, state, X):
        value, grads = jax.value_and_grad(loss)(params, X)
        updates, state = adam.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    for step in range(cfg.adam_steps):
        X = jnp.asarray(collocation(rng, cfg.adam_batch, d, cfg))
        params, a_state, value = adam_step(params, a_state, X)
        if log_every and step % log_every == 0:
            history.append(("adam", step, float(value)))

    cpu_adam = time.process_time() - t_cpu
    wall_adam = time.perf_counter() - t_wall
    params_adam = params

    # --- stage 2: L-BFGS on a fixed set of points ------------------------------------
    Xfix = jnp.asarray(collocation(rng, cfg.lbfgs_batch, d, cfg))

    def fixed_loss(p):
        return loss(p, Xfix)

    lbfgs = optax.lbfgs()
    l_state = lbfgs.init(params)
    vg = optax.value_and_grad_from_state(fixed_loss)

    @jax.jit
    def lbfgs_step(params, state):
        value, grads = vg(params, state=state)
        updates, state = lbfgs.update(grads, state, params, value=value, grad=grads,
                                      value_fn=fixed_loss)
        return optax.apply_updates(params, updates), state, value

    for step in range(cfg.lbfgs_steps):
        params, l_state, value = lbfgs_step(params, l_state)
        if log_every and step % log_every == 0:
            history.append(("lbfgs", step, float(value)))
        if not np.isfinite(float(value)):
            break

    cpu = time.process_time() - t_cpu
    wall = time.perf_counter() - t_wall

    ev = evaluate(problem, cfg, params, batched_EL, batched_logpsi, seed)
    ev_adam = evaluate(problem, cfg, params_adam, batched_EL, batched_logpsi, seed)
    return {
        "problem": problem.name, "d": d, "g": problem.g, "method": "pinn", "seed": seed,
        "n_params": n_params(params), "config": cfg.__dict__,
        "cpu_s": cpu, "wall_s": wall, "cpu_adam_s": cpu_adam, "wall_adam_s": wall_adam,
        "final_loss": float(fixed_loss(params)),
        # peak resident memory of the whole process so far (JAX runtime included):
        # an upper bound on what this run needed
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        **ev,
        "adam_only": {k: ev_adam[k] for k in ("E", "rel_energy_error", "l2_error")},
        "history": history,
    }
