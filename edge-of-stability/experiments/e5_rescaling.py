"""Experiment 5: what survives a change of coordinates (ReLU network, digits).

Part A  Train once (eta = 0.05, a stable run). Apply the function-preserving
        layer rescaling alpha to the trained network and continue gradient
        descent with the same step size from each rescaled copy. Same function,
        different coordinates: does gradient descent leave it alone?
Part B  Track the layer imbalance Q_0 = ||W_0||^2 + ||b_0||^2 - ||W_1||^2 at
        every step for a stable run and a large-step run, and check the exact
        identity  Q' - Q = eta^2 (||g_{W_0}||^2 + ||g_{b_0}||^2 - ||g_{W_1}||^2).
Part C  Start from the same initial function in two coordinate systems
        (alpha = 1 and alpha = 4) and train with a small and a large step,
        for the same amount of "time" eta * steps.

Outputs: results/e5_rescaling.csv, results/e5_tracking.npz, results/e5_init.csv
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from config import INIT_GAIN, N_TRAIN, RESULTS, SIZES
from eos.data import digits_split
from eos.mlp import Problem, accuracy, init_params
from eos.sharpness import adaptive_sharpness, sharpness
from eos.symmetry import imbalance_increment, layer_imbalance, rescale
from eos.train import run_gd

ETA = 0.05


def predictions(prob, theta, x):
    return np.asarray(prob.predict(theta, jnp.asarray(x)))


def part_a(prob, data, theta0, quick):
    train_steps, cont_steps = (500, 300) if quick else (5000, 2000)
    tr = run_gd(prob, theta0, ETA, train_steps, 100)
    base = prob.unravel(jnp.asarray(tr.params))
    alphas = np.geomspace(0.05, 10.0, 9 if quick else 41)
    rows = []
    for a in alphas:
        p_a = rescale(base, 0, float(a))
        th = prob.flatten(p_a)
        s0, _ = sharpness(prob, th)
        as0, _ = adaptive_sharpness(prob, th)
        out_before = predictions(prob, th, data.x_test)
        cont = run_gd(prob, th, ETA, cont_steps, cont_steps)
        if cont.diverged:
            rows.append({"alpha": a, "sharpness_start": s0, "adaptive_start": as0, "diverged": True})
            continue
        out_after = predictions(prob, cont.params, data.x_test)
        s1, _ = sharpness(prob, cont.params)
        as1, _ = adaptive_sharpness(prob, cont.params)
        rows.append({
            "alpha": a, "diverged": False,
            "sharpness_start": s0, "adaptive_start": as0,
            "sharpness_end": s1, "adaptive_end": as1,
            "loss_start": float(cont.loss[0]), "loss_max": float(cont.loss.max()), "loss_end": float(cont.loss[-1]),
            "frac_test_predictions_changed": float(np.mean(out_before.argmax(1) != out_after.argmax(1))),
            "mean_abs_output_change": float(np.mean(np.abs(out_before - out_after))),
            "test_acc_start": float(np.mean(out_before.argmax(1) == data.labels_test)),
            "test_acc_end": float(np.mean(out_after.argmax(1) == data.labels_test)),
            "Q0_start": layer_imbalance(p_a, 0),
            "Q0_end": layer_imbalance(prob.unravel(jnp.asarray(cont.params)), 0),
        })
        r = rows[-1]
        print(f"alpha={a:.3f} S={s0:.2f} (2/eta={2/ETA}) adaptive={as0:.5f} maxloss/start={r['loss_max']/r['loss_start']:.2f} "
              f"changed={r['frac_test_predictions_changed']:.3f} S_end={s1:.2f} Q0 {r['Q0_start']:.3f}->{r['Q0_end']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("e5_rescaling_quick.csv" if quick else "e5_rescaling.csv"), index=False)


def tracked_run(prob, theta0, eta, steps, every):
    step = jax.jit(lambda th: (prob.value_and_grad(th)))
    th = jnp.asarray(theta0)
    q, inc, losses, ev, sh = [], [], [], [], []
    v0 = None
    for t in range(steps):
        if t % every == 0:
            s, v0 = sharpness(prob, th, v0=v0)
            ev.append(t)
            sh.append(s)
        loss, g = step(th)
        p, gp = prob.unravel(th), prob.unravel(g)
        q.append(layer_imbalance(p, 0))
        inc.append(imbalance_increment(gp, 0, eta))
        losses.append(float(loss))
        th = th - eta * g
    q.append(layer_imbalance(prob.unravel(th), 0))
    return np.array(losses), np.array(q), np.array(inc), np.array(ev), np.array(sh), np.asarray(th)


def part_b(prob, theta0, quick):
    steps = 300 if quick else 3000
    out = {}
    for eta in [0.05, 0.25]:
        losses, q, inc, ev, sh, _ = tracked_run(prob, theta0, eta, steps, 25)
        err = np.max(np.abs((q[1:] - q[:-1]) - inc))
        out.update({f"loss_{eta}": losses, f"Q0_{eta}": q, f"inc_{eta}": inc, f"steps_{eta}": ev, f"sharp_{eta}": sh})
        moved = q[-1] - q[0]
        first = q[min(200, steps)] - q[0]
        print(f"eta={eta}: Q0 {q[0]:.4f} -> {q[-1]:.4f} (moved {moved:+.4f}; first 200 steps {first:+.4f}); "
              f"max |dQ - eta^2 term| = {err:.2e}; final sharpness {sh[-1]:.2f}", flush=True)
    np.savez(RESULTS / ("e5_tracking_quick.npz" if quick else "e5_tracking.npz"), **out)


def part_c(prob, data, params, quick):
    horizon = 25.0 if quick else 250.0  # eta * steps
    rows = []
    finals = {}
    for alpha in [1.0, 4.0]:
        p_a = rescale(params, 0, alpha)
        th0 = prob.flatten(p_a)
        s_init, _ = sharpness(prob, th0)
        for eta in [0.01, 0.05]:
            steps = int(round(horizon / eta))
            tr = run_gd(prob, th0, eta, steps, steps // 10)
            pf = prob.unravel(jnp.asarray(tr.params))
            s_end, _ = sharpness(prob, tr.params)
            a_end, _ = adaptive_sharpness(prob, tr.params)
            finals[(alpha, eta)] = predictions(prob, tr.params, data.x_test)
            rows.append({
                "alpha": alpha, "eta": eta, "ceiling": 2 / eta, "steps": steps,
                "sharpness_init": s_init, "sharpness_end": s_end, "adaptive_end": a_end,
                "Q0_init": layer_imbalance(p_a, 0), "Q0_end": layer_imbalance(pf, 0),
                "max_loss": float(tr.loss.max()), "train_loss_end": float(tr.loss[-1]),
                "test_acc": accuracy(prob, tr.params, data.x_test, data.labels_test),
            })
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    for eta in [0.01, 0.05]:
        diff = np.mean(finals[(1.0, eta)].argmax(1) != finals[(4.0, eta)].argmax(1))
        for r in rows:
            if r["eta"] == eta:
                r["frac_test_pred_differ_between_inits"] = float(diff)
        print(f"eta={eta}: test predictions differing between alpha=1 and alpha=4 starts: {diff:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / ("e5_init_quick.csv" if quick else "e5_init.csv"), index=False)


def main(quick: bool = False, parts: str = "abc"):
    data = digits_split(N_TRAIN)
    params = init_params(0, SIZES, INIT_GAIN)
    prob = Problem.build(SIZES, "relu", data.x_train, data.y_train, params)
    theta0 = prob.flatten(params)
    if "a" in parts:
        part_a(prob, data, theta0, quick)
    if "b" in parts:
        part_b(prob, theta0, quick)
    if "c" in parts:
        part_c(prob, data, params, quick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="abc")
    a = ap.parse_args()
    main(a.quick, a.parts)
