# HMC and the geometry of Bayesian inverse PDEs

This project is a reproducible computational companion to the Medium essay
*When Physics Meets Probability: Hamiltonian Monte Carlo and the Limits of Physics-Informed Neural Networks*.

## What it does—and what it does not

It uses a **known analytic forward solution**, not a trained PINN. This design
sets approximation error to zero, isolating posterior geometry and the behaviour
of HMC, a Gaussian mean-field variational approximation and a MAP estimator.
The manuscript discusses how these phenomena carry over to B-PINNs and where
additional neural approximation and residual-likelihood assumptions enter.

The exact solution of the one-dimensional heat equation is

```text
u(x,t; A, κ) = A sin(πx) exp(-κπ²t),  x in [0,1], t ≥ 0.
```

It satisfies `u_t = κu_xx`, zero boundary values and initial condition
`u(x,0)=A sin(πx)` for all positive `A` and `κ`. Their log parameters have
independent Gaussian priors. Twenty-four synthetic noisy observations are
drawn at either narrowly or widely distributed measurement times; their spatial
locations and noise variance are held fixed.

## Reproduce

Tested with Python 3.12, NumPy 2.3, SciPy 1.17, and Matplotlib 3.10. From this
directory, use an isolated environment of your choice:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python experiment.py --output results --seed 20261116 --replications 100
```

The command writes `results/metrics.json` and five PNG figures under
`results/figures/`. No data are downloaded, and all random seeds are fixed.

## What is compared

1. A 480×480 deterministic integration of the two-dimensional posterior,
   treated as the numerical reference rather than an MCMC result.
2. MAP for the same posterior.
3. Reverse-KL, independent Gaussian variational inference, optimized using
   24×24 Gauss–Hermite quadrature (no stochastic VI noise).
4. Four-chain leapfrog HMC with a unit mass matrix.
5. Four-chain leapfrog HMC with a fixed mass matrix equal to the Hessian of the
   negative log posterior at the MAP point. This is a local metric, not full
   Riemannian HMC.
6. Repeated synthetic-data coverage of the **exact-grid** posterior intervals;
   these are not claims of full B-PINN calibration.

HMC step sizes are adapted only during warm-up; path lengths are randomly
jittered. Results include acceptance, rough FFT-based ESS, split-Rhat and the
number of gradient calls. Chain diagnostics are intentionally minimal: users
should not interpret split-Rhat or ESS alone as proof that all posterior regions
were explored. High-dimensional applications should use modern rank-normalized
diagnostics and stronger chain/pathology checks.

## Scientific boundaries

- The analytic forward model is correctly specified by construction.
- The experiment is low-dimensional (two unknown log parameters), not a
  demonstration of scaling to thousands of neural weights.
- The residual term of a soft-constrained PINN is absent because the equation
  holds exactly. A full B-PINN would need an explicit residual-discrepancy
  model, prior over weights and inference over those weights.
- Credible interval coverage is evaluated over draws from a fixed parameter
  pair, so the observed value can differ from nominal coverage even under a
  correctly specified Bayesian model, especially for finite samples.
- The MAP Hessian mass is fixed across the trajectory. It need not solve a
  globally curved posterior; the study tests that limitation.

## Outputs

| Figure | Subject |
| --- | --- |
| `posterior_geometry.png` | Narrow time design: curved parameter ridge and VI contour |
| `predictive_uncertainty.png` | Exact posterior/VI solution bands against truth |
| `observation_geometry.png` | Posterior geometry under two time designs |
| `sampler_diagnostics.png` | Trace and autocorrelation under two HMC masses |
| `repeated_coverage.png` | Exact posterior repeated-data 90% interval coverage |

The generated figures are original and can be reused with attribution to
Diogo Ribeiro and the companion project.
