# PINNs against classical ground-state solvers, one to eight dimensions

Companion code for a Medium article on where a physics-informed neural network (PINN) starts to beat classical methods for a quantum ground state, and how much that answer depends on which classical method it is compared with.

## Question

A PINN represents the solution as a network and needs no grid, so its cost grows slowly with dimension. A tensor-product grid with N points per axis needs N^d unknowns. Somewhere the grid must lose. Where that happens depends on N, and N depends on the discretisation: a second-order stencil needs far more points per axis than a fourth-order one, and a spectral grid needs fewer still. Above the dimensions where any grid fits, the classical alternative is Monte Carlo, whose cost does not depend on a grid at all.

This project measures, for each dimension from 1 to 8:

1. the energy error a PINN reaches with a fixed training budget;
2. what each classical method needs (points per axis, memory, CPU time) to reach the same error;
3. how far each grid method can go in dimension before it no longer fits in memory.

## Problems

Both have an exact ground state `psi0 = exp(-phi)` and energy `E0 = d/2` (units hbar = m = 1).

* **Oscillator**: `V = |x|^2 / 2`. Separable, so the exact answer of any tensor-product grid in any dimension follows from one 1D eigenproblem; this is used to state memory needs beyond what fits on the machine without extrapolating anything.
* **Coupled quartic**: `phi = |x|^2/2 + kappa * sum_{i<j} x_i^2 x_j^2`, `kappa = 1/(d-1)`, and the potential built backwards from it, `V = E0 + (|grad phi|^2 - Lap phi)/2`. The coupling term is neither a sum of one-coordinate functions nor radial, so the problem does not separate. psi0 is positive, hence the ground state.

## Methods

| Method | What it is | Code |
|---|---|---|
| 2nd-order finite differences | 3-point stencil on a box, matrix-free LOBPCG | `src/pinncross/grids.py` |
| 4th-order finite differences | 5-point stencil, same solver | `src/pinncross/grids.py` |
| Spectral (sinc) grid | Colbert–Miller sinc DVR, same solver | `src/pinncross/grids.py` |
| Diffusion Monte Carlo | no trial function, symmetric split, fixed population | `src/pinncross/dmc.py` |
| PINN | `psi = prod(1 - x_k^2/L^2) exp(NN(x))`, residual loss, Adam then L-BFGS (JAX, float64) | `src/pinncross/pinn.py` |

The grid solver is single-vector LOBPCG with a kinetic-energy preconditioner applied by discrete sine transforms and scaled by the potential, so the iteration count barely grows with N. It starts from the constant vector. Nothing in any classical method uses the known answer, and none of them exploits separability (the oscillator shortcut is only used to report memory needs beyond the budget, and is checked against full d-dimensional solves wherever they fit).

## Run

From this folder:

```bash
python -m pip install -e ".[dev]"
pytest                       # identities, solver checks, ~30 s
cd experiments
./run_all.sh                 # about 5 hours on a two-core CPU
```

or step by step:

```bash
python e1_convergence.py         # ~10 min: convergence in N, box size
python e2_grids_by_dimension.py  # ~2 h: d-dimensional grid solves up to 5 GB
python e4_dmc.py                 # ~20 min
python e3_pinn.py --seeds 0      # ~1.5 h: 15 PINN runs
python e3_pinn.py --seeds 1 2 --dims 3 5 8
python summarize.py              # comparisons at matched accuracy
python make_figures.py
```

Every script accepts `--quick` for a smoke run that writes `*_quick` files and never overwrites the full results. The grid and PINN steps save as they go and skip work already saved.

## Results

RESULTS_TABLE

## Outputs

| Path | Contents |
|---|---|
| `results/e1_convergence.json` | grid convergence in N, box-size checks |
| `results/e2_grids.json` | every d-dimensional grid solve: error, CPU and wall time, iterations, peak memory |
| `results/e4_dmc.json` | DMC time-step check and runs by dimension |
| `results/pinn/*.json` | one file per PINN run, with the training history |
| `results/summary.json`, `summary.md` | the matched-accuracy comparisons |
| `figures/` | the figures, drawn only from `results/` |

## Limits

* Two problems, both with nodeless ground states. Positivity is what the PINN's `exp(NN)` form and plain DMC both rely on; fermionic ground states have nodes and neither method applies as written.
* One network size and one training budget for the PINN in every dimension; three seeds at d = 3, 5 and 8, one elsewhere.
* Memory is the peak traced by `tracemalloc` for this implementation, about 17 arrays of the grid's size. A leaner eigensolver could hold about 6, which would lower every grid curve by a factor of about 3 without changing the scaling.
* CPU times are process CPU time on one machine (two cores, 7.7 GB RAM), measured with nothing else running.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt.
