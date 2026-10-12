# Gradient descent at the edge of stability

Companion code for the Medium article *Gradient Descent at the Edge of Stability: How the Step Size Bounds Curvature, from a Two-Parameter Model to a Neural Network*.

## Question

Full-batch gradient descent with step size η can only settle at a point where the largest Hessian eigenvalue is below 2/η. What does that inequality do to the minimum that training ends in, and how much of the "sharpness" measured after training belongs to the optimiser rather than to the model?

The project answers this in three settings:

1. **A two-parameter model** `f(u, v) = u v`, loss `(uv − 1)²/2`, where everything is exact. Gradient flow conserves the imbalance `D = u² − v²`; gradient descent changes it by the exact factor `D' = D (1 − η² r²)`, with `r = uv − 1`. Experiment 1 compares where flow and descent stop from 3000 random starts per step size.
2. **The same model past the threshold.** On `u = v = s`, gradient descent is the cubic map `s ↦ s(1 + η − η s²)`. The 2-cycle is stable for `1 < η < √5 − 1`, the period-doubling ratios converge to Feigenbaum's δ ≈ 4.669, and a boundary crisis at η = 2 ends bounded motion (experiment 2).
3. **A small network on real data** (scikit-learn's handwritten digits, 1000 training / 797 test images, squared-error loss, float64). Sharpness is the top Hessian eigenvalue, computed with Lanczos on exact Hessian-vector products. Experiments 3–6: sharpness traces at four step sizes, a 12 × 4 step-size × seed sweep, the layer-rescaling symmetry of ReLU networks with the exact discrete identity for the layer imbalance, and heavy-ball momentum.

## Data

No download is needed: the digits data ships with scikit-learn. The train/test split is fixed (seed 0) and independent of the model seeds.

## Run

From this folder:

```bash
python -m pip install -e ".[dev]"
pytest                                   # identities and estimators, ~10 s
cd experiments
python e1_toy_landing.py                 # ~1 min
python e2_toy_chaos.py                   # ~5 min
python e3_network_traces.py              # ~3 min
python e4_lr_sweep.py                    # ~30 min (48 runs)
python e5_rescaling.py                   # ~15 min
python e6_momentum.py                    # ~2 min
python make_figures.py                   # draws every figure from results/
```

Every experiment script accepts `--quick` for a smoke run; quick runs write `*_quick.*` files and never overwrite the full results. Timings are for a two-core CPU.

## Outputs

`results/` holds the saved outputs of the full runs that the article quotes, and `figures/` the nine figures, drawn only from those files.

| Path | Contents |
|---|---|
| `src/eos/toy.py` | the two-parameter model: Hessian, flow endpoint, GD map, balanced cubic map, orbits, Lyapunov exponent |
| `src/eos/mlp.py` | fully connected networks on a flat float64 parameter vector, exact gradients and Hessian-vector products (JAX) |
| `src/eos/sharpness.py` | top Hessian eigenvalue and adaptive sharpness `λmax(diag|θ| H diag|θ|)`, matrix-free Lanczos |
| `src/eos/symmetry.py` | ReLU layer rescaling, layer imbalance, exact per-step imbalance increment |
| `src/eos/train.py` | full-batch GD / heavy-ball with per-step loss and periodic sharpness |
| `tests/` | the identities the article relies on, checked numerically |
| `experiments/` | one script per experiment, plus `make_figures.py` |

## What the tests check

* The Hessian at a minimiser of `(uv − 1)²/2` has eigenvalues `0` and `u² + v²`.
* Gradient flow conserves `u² − v²`; gradient descent satisfies `D' = D(1 − η² r²)` to rounding error.
* The 2-cycle appears just beyond η = 1.
* Lanczos sharpness equals the top eigenvalue of the dense Hessian on a small network.
* Rescaling a ReLU network leaves its outputs and its adaptive sharpness unchanged and changes its sharpness.
* One GD step changes the layer imbalance by exactly `η² (‖g_l‖² − ‖g_{l+1}‖²)`.

## Automation

`.github/workflows/edge-of-stability.yml` runs the tests and the quick versions of the experiments when files in this folder change.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt.
