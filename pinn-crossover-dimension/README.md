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
| Diffusion Monte Carlo | no trial function, symmetric split, fixed population, Richardson-extrapolated in the time step | `src/pinncross/dmc.py` |
| PINN | `psi = prod(1 - x_k^2/L^2) exp(NN(x))`, residual loss, Adam then L-BFGS (JAX, float64) | `src/pinncross/pinn.py` |

The grid solver is single-vector LOBPCG with a kinetic-energy preconditioner applied by discrete sine transforms and scaled by the potential. On the oscillator it never needs more than about 40 iterations; on the coupled problem the count grows with N for the finite-difference grids (up to 470). It starts from the constant vector and stops when the eigenvalue changes by less than 1e-13 (relative) over three iterations or the residual falls below 1e-10 of it. Nothing in any classical method uses the known answer, and none of them exploits separability (the oscillator shortcut is only used to report memory needs beyond the budget, and is checked against full d-dimensional solves wherever they fit).

## Run

From this folder:

```bash
python -m pip install -e ".[dev]"
pytest                       # identities, solver checks, ~30 s
cd experiments
./run_all.sh                 # about 7 hours on a two-core CPU
```

or step by step:

```bash
python e1_convergence.py         # ~10 min: convergence in N, box size
python e2_grids_by_dimension.py  # ~2.5 h: 672 d-dimensional grid solves up to 5 GB
python e4_dmc.py                 # ~25 min: time-step check, extrapolated runs
python e3_pinn.py --seeds 0      # ~2.5 h: 15 PINN runs
python e3_pinn.py --seeds 1 2 --dims 3 5 8                         # ~2 h
python e5_dmc_at_pinn_budget.py  # ~45 min: DMC given the PINN's CPU time
python e3_pinn.py --problems coupled --dims 8 --seeds 0 --lbfgs-refresh 500
python summarize.py              # comparisons at matched accuracy
python make_figures.py
```

Every script accepts `--quick` for a smoke run that writes `*_quick` files and never overwrites the full results. The grid and PINN steps save as they go and skip work already saved.

## Results

Headline numbers (two-core CPU, 5 GB memory budget per grid solve):

* The PINN's median relative energy error is between 3.4e-6 and 1.2e-4 on the oscillator in d = 1-8 after 3-14 min of training, and between 3.2e-6 and 9.5e-4 on the coupled problem in d = 2-7; on the coupled problem in d = 8 it fails in all three seeds (and with refreshed L-BFGS points).
* The spectral (sinc) grid matches the PINN's energy in every dimension where the PINN works, with at most 2.4 GB and about 3 min of CPU. The fourth-order stencil fits through d = 4, the second-order stencil through d = 2.
* Plain DMC (no trial function, extrapolated to zero time step) has relative error x sqrt(CPU seconds) of about 7e-3 in every dimension. Run for the PINN's training time it reaches 5.1e-4, 1.2e-4 and 4.0e-4 on the coupled problem in d = 6, 7, 8 (PINN: 3.9e-4, 9.5e-4, failed) and 5.5e-4 on the oscillator in d = 8 (PINN: 1.1e-4).

`results/summary.md`, produced by `summarize.py`:

| problem | d | PINN energy error | PINN CPU | FD2 to match | FD4 to match | sinc to match | DMC CPU to match |
|---|---|---|---|---|---|---|---|
| oscillator | 1 | 1.2e-04 | 3.0 min | N=192, 3.26e-05 GB, 0.0 s | N=32, 9.66e-06 GB, 0.0 s | N=8, 8.8e-06 GB, 0.0 s | 0.84 h |
| oscillator | 2 | 5.3e-05 | 5.3 min | N=384, 0.0202 GB, 0.7 s | N=40, 0.000249 GB, 0.0 s | N=8, 1.95e-05 GB, 0.0 s | 4.5 h |
| oscillator | 3 | 2.3e-05 | 6.3 min | N=407, 9.17 GB (exact, 1D) | N=48, 0.0152 GB, 0.7 s | N=10, 0.000157 GB, 0.1 s | 13 h |
| oscillator | 4 | 3.4e-06 | 8.1 min | N=957, 1.14e+05 GB (exact, 1D) | N=66, 2.58 GB (exact, 1D) | N=10, 0.00145 GB, 0.1 s | 1.1e+03 h |
| oscillator | 5 | 9.8e-06 | 9.8 min | N=609, 1.14e+07 GB (exact, 1D) | N=52, 51.7 GB (exact, 1D) | N=10, 0.0144 GB, 1.0 s | 84 h |
| oscillator | 6 | 1.4e-05 | 11.4 min | N=524, 2.82e+09 GB (exact, 1D) | N=49, 1.88e+03 GB (exact, 1D) | N=10, 0.144 GB, 9.6 s | 57 h |
| oscillator | 7 | 3.9e-05 | 13.3 min | N=315, 4.19e+10 GB (exact, 1D) | N=38, 1.56e+04 GB (exact, 1D) | N=8, 0.302 GB, 16.3 s | 6.6 h |
| oscillator | 8 | 1.1e-04 | 14.1 min | N=188, 2.12e+11 GB (exact, 1D) | N=29, 6.8e+04 GB (exact, 1D) | N=8, 2.42 GB, 193.9 s | 1 h |
| coupled | 2 | 4.2e-06 | 5.6 min | N=2048, 0.571 GB, 437.6 s | N=160, 0.00362 GB, 0.9 s | N=32, 0.000163 GB, 0.1 s | 1.3e+03 h |
| coupled | 3 | 3.2e-06 | 6.5 min | not reached | N=128, 0.285 GB, 102.2 s | N=24, 0.002 GB, 0.2 s | 1.1e+03 h |
| coupled | 4 | 3.6e-05 | 8.2 min | N=768, 4.73e+04 GB (est.) | N=64, 2.28 GB, 535.6 s | N=16, 0.00945 GB, 1.1 s | 13 h |
| coupled | 5 | 7.0e-05 | 9.8 min | N=512, 4.79e+06 GB (est.) | N=64, 146 GB (est.) | N=14, 0.0775 GB, 7.2 s | 3 h |
| coupled | 6 | 3.9e-04 | 11.7 min | N=192, 6.81e+06 GB (est.) | N=40, 557 GB (est.) | N=12, 0.43 GB, 46.0 s | 0.082 h |
| coupled | 7 | 9.5e-04 | 13.1 min | N=128, 7.66e+07 GB (est.) | N=32, 4.67e+03 GB (est.) | N=10, 1.44 GB, 177.6 s | 0.016 h |
| coupled | 8 | 5.6e+00 (failed) | 14.3 min | - | - | - | - |

Matched on the wavefunction (relative L2 error; nodal for the grids):

| problem | d | PINN L2 error | FD2 to match | FD4 to match | sinc to match |
|---|---|---|---|---|---|
| oscillator | 1 | 2.1e-03 | N=48, 1.18e-05 GB, 0.0 s | N=16, 7.46e-06 GB, 0.0 s | N=8, 8.8e-06 GB, 0.0 s |
| oscillator | 2 | 2.0e-03 | N=48, 0.000356 GB, 0.0 s | N=16, 4.5e-05 GB, 0.0 s | N=8, 1.95e-05 GB, 0.0 s |
| oscillator | 3 | 1.8e-03 | N=64, 0.0358 GB, 1.3 s | N=20, 0.00121 GB, 0.0 s | N=8, 8.63e-05 GB, 0.1 s |
| oscillator | 4 | 1.4e-03 | N=66, 2.58 GB (exact, 1D) | N=20, 0.0218 GB, 1.4 s | N=8, 0.000597 GB, 0.1 s |
| oscillator | 5 | 2.1e-03 | N=55, 68.4 GB (exact, 1D) | N=20, 0.435 GB, 26.3 s | N=8, 0.00473 GB, 0.2 s |
| oscillator | 6 | 4.0e-03 | N=42, 747 GB (exact, 1D) | N=16, 2.28 GB, 221.5 s | N=8, 0.0378 GB, 2.3 s |
| oscillator | 7 | 5.2e-03 | N=38, 1.56e+04 GB (exact, 1D) | N=14, 14.3 GB (exact, 1D) | N=8, 0.302 GB, 16.3 s |
| oscillator | 8 | 9.8e-03 | N=29, 6.8e+04 GB (exact, 1D) | N=12, 58.5 GB (exact, 1D) | N=6, 0.242 GB, 19.3 s |
| coupled | 2 | 3.0e-04 | N=192, 0.00515 GB, 4.3 s | N=64, 0.000629 GB, 0.1 s | N=28, 0.000126 GB, 0.1 s |
| coupled | 3 | 5.1e-04 | N=128, 0.285 GB, 89.1 s | N=40, 0.00883 GB, 2.0 s | N=18, 0.000864 GB, 0.1 s |
| coupled | 4 | 2.9e-03 | N=64, 2.28 GB, 428.3 s | N=24, 0.0451 GB, 5.2 s | N=12, 0.00299 GB, 0.4 s |
| coupled | 5 | 4.3e-03 | N=48, 34.7 GB (est.) | N=20, 0.435 GB, 57.0 s | N=12, 0.0358 GB, 4.3 s |
| coupled | 6 | 1.3e-02 | N=24, 26 GB (est.) | N=16, 2.28 GB, 341.1 s | N=10, 0.144 GB, 16.0 s |
| coupled | 7 | 2.5e-02 | N=24, 624 GB (est.) | N=10, 1.36 GB (est.) | N=6, 0.0403 GB, 3.9 s |

`(exact, 1D)`: oscillator, from the one-dimensional eigenproblem. `(est.)`: coupled problem, from the points per axis that reached the same target in a lower dimension. "DMC CPU to match" uses DMC's measured error law.

## Outputs

| Path | Contents |
|---|---|
| `results/e1_convergence.json` | grid convergence in N, box-size checks |
| `results/e2_grids.json` | every d-dimensional grid solve: error, CPU and wall time, iterations, peak memory |
| `results/e4_dmc.json` | DMC time-step check and extrapolated runs by dimension |
| `results/e5_dmc_at_pinn_budget.json` | DMC run for the PINN's training time, d = 8 oscillator and d = 6-8 coupled |
| `results/pinn_variants/*.json` | the PINN with fresh L-BFGS points every 500 iterations (8D coupled) |
| `results/pinn/*.json` | one file per PINN run, with the training history |
| `results/summary.json`, `summary.md` | the matched-accuracy comparisons |
| `figures/` | the figures, drawn only from `results/` |

## Limits

* Two problems, both with nodeless ground states. Positivity is what the PINN's `exp(NN)` form and plain DMC both rely on; fermionic ground states have nodes and neither method applies as written.
* One network size and one training budget for the PINN in every dimension; three seeds at d = 3, 5 and 8, one elsewhere.
* Memory is the peak traced by `tracemalloc` for this implementation, about 17 arrays of the grid's size. The LOBPCG iteration itself needs about 7, so a leaner implementation would lower every grid curve by a factor of 2-3 without changing the scaling.
* Where the PINN's energy error is below about 1e-4 it is within ~1.5 standard errors of its 500,000-point evaluation (oscillator d <= 6, coupled d = 2-3); the true error may be smaller, which would only raise the grids' requirements.
* The long DMC runs sit 2-3 standard errors above the exact energy in three of four cases, consistent with a residual population-control bias of a few parts in 1e4 at 4,000 walkers.
* CPU times are process CPU time on one machine (two cores, 7.7 GB RAM), measured with nothing else running.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt.
