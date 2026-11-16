"""Follow-up studies for the inverse-diffusion benchmark.

Each function answers one question the essay raises about the narrow-window
posterior and returns plain numbers for ``metrics.json``:

* identification: what the likelihood alone says (profile likelihood), how far
  the interval moves with the prior, and how the ridge's local shape changes
  along its length;
* samplers: HMC under three constant mass matrices, replicated over seeds and
  path lengths, with rank-normalised diagnostics, divergences and the leapfrog
  stability margin along the ridge;
* design: local Fisher-information design (Box–Lucas and a kappa-targeted
  two-time design), checked with the exact grid posterior on repeated data;
* calibration: coverage of the exact-grid intervals as the true kappa moves
  away from the prior centre, and with parameters drawn from the prior.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import design
import diagnostics
from experiment import (
    OBSERVATION_SIGMA,
    PRIOR_MEAN,
    PRIOR_SD,
    TRUE_AMPLITUDE,
    TRUE_DIFFUSIVITY,
    FloatArray,
    GridPosterior,
    Observations,
    Posterior,
    adaptive_grid,
    hmc,
    posterior_grid,
    simulate,
    simulate_at,
    solution,
)

# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def profile_likelihood(data: Observations, kappas: FloatArray) -> tuple[FloatArray, float]:
    """Profile negative log-likelihood over kappa (amplitude maximised in closed form).

    u is linear in A, so for each kappa the best A >= 0 is a least-squares
    coefficient.  Returns the curve minus its minimum and the limit at kappa = 0,
    on the same scale.
    """
    sin_x = np.sin(np.pi * data.x)

    def nll(kappa: float) -> float:
        f = sin_x * np.exp(-kappa * np.pi**2 * data.t)
        amp = max(0.0, float(f @ data.y) / float(f @ f))
        r = (amp * f - data.y) / data.sigma
        return 0.5 * float(r @ r)

    values = np.array([nll(k) for k in kappas])
    best = minimize_scalar(lambda b: nll(np.exp(b)), bounds=(np.log(kappas.min()), np.log(kappas.max())),
                           method="bounded", options={"xatol": 1e-10})
    floor = min(float(best.fun), float(values.min()))
    return values - floor, nll(0.0) - floor


def likelihood_interval(kappas: FloatArray, delta: FloatArray, level: float) -> tuple[float, float, bool]:
    """Profile-likelihood interval {kappa: 2 delta <= chi2_1(level)}; flag if it reaches the grid edge."""
    threshold = 0.5 * float(norm.ppf(0.5 + level / 2) ** 2)
    inside = kappas[delta <= threshold]
    return float(inside.min()), float(inside.max()), bool(delta[0] <= threshold)


def sensitivity_grid(posterior: Posterior) -> GridPosterior:
    """Grid wide enough for diffuse priors, with log-kappa spacing of at most 0.01."""
    lo = posterior.prior_mean[1] - 5 * posterior.prior_sd[1]
    hi = min(posterior.prior_mean[1] + 5 * posterior.prior_sd[1], 1.2)
    n_beta = int(np.ceil((hi - lo) / 0.01)) + 1
    a_lo = posterior.prior_mean[0] - 5 * posterior.prior_sd[0]
    a_hi = posterior.prior_mean[0] + 5 * posterior.prior_sd[0]
    return posterior_grid(posterior, resolution=(480, n_beta), bounds=((a_lo, a_hi), (lo, hi)))


def prior_sensitivity(data: Observations) -> dict[str, object]:
    """Exact-grid 90% interval for kappa as the log-kappa prior is widened or moved."""
    out: dict[str, object] = {"scale": {}, "centre": {}}
    for scale in (0.35, 0.70, 1.40, 2.80):
        post = Posterior(data, prior_sd=np.array([PRIOR_SD[0], scale]))
        grid = sensitivity_grid(post)
        k = np.exp(grid.points[:, 1])
        out["scale"][f"{scale:.2f}"] = {
            "interval90_diffusivity": [float(v) for v in grid.kappa_quantiles((0.05, 0.95))],
            "median_diffusivity": float(grid.kappa_quantiles((0.5,))[0]),
            "boundary_mass": grid.boundary_mass,
        }
    for centre in (0.11, 0.22, 0.44):
        post = Posterior(data, prior_mean=np.array([PRIOR_MEAN[0], np.log(centre)]))
        grid = sensitivity_grid(post)
        k = np.exp(grid.points[:, 1])
        out["centre"][f"{centre:.2f}"] = {
            "interval90_diffusivity": [float(v) for v in grid.kappa_quantiles((0.05, 0.95))],
            "median_diffusivity": float(grid.kappa_quantiles((0.5,))[0]),
            "boundary_mass": grid.boundary_mass,
        }
    return out


@dataclass
class RidgePoint:
    """Local geometry at the conditional mode of log A for a given kappa quantile."""

    level: float
    q: FloatArray
    hessian: FloatArray

    def soft_direction(self) -> FloatArray:
        _, vec = np.linalg.eigh(self.hessian)
        d = vec[:, 0]
        return d * np.sign(d[1])


def ridge_points(posterior: Posterior, grid: GridPosterior, levels: tuple[float, ...]) -> list[RidgePoint]:
    """Conditional modes along the ridge at posterior quantiles of kappa."""
    points = []
    for level in levels:
        beta = float(np.log(grid.kappa_quantiles((level,))[0]))
        alpha = minimize_scalar(lambda a: float(posterior.potential(np.array([a, beta]))),
                                bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-10}).x
        q = np.array([alpha, beta])
        points.append(RidgePoint(level, q, posterior.local_hessian(q)))
    return points


def metric_root(mass: FloatArray) -> FloatArray:
    """Symmetric square root M^{1/2}; HMC with mass M is plain HMC in z = M^{1/2} q."""
    vals, vecs = np.linalg.eigh(mass)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def ridge_in_metric(points: list[RidgePoint], mass: FloatArray) -> list[dict[str, float]]:
    """Local widths of the ridge in the coordinates in which HMC with this mass is isotropic.

    For a locally Gaussian target, leapfrog with step eps is stable only when
    eps < 2 * (thinnest local SD in these coordinates) (Neal, 2011, "What stepsize?").
    """
    root = metric_root(mass)
    inv_root = np.linalg.inv(root)
    rows = []
    for point in points:
        local = inv_root.T @ point.hessian @ inv_root
        sd = 1.0 / np.sqrt(np.linalg.eigvalsh(local))  # ascending eigenvalues -> descending SD
        rows.append({
            "level": point.level,
            "kappa": float(np.exp(point.q[1])),
            "long_sd": float(sd[0]),
            "thin_sd": float(sd[1]),
        })
    return rows


def identification(data: Observations, grid: GridPosterior, posterior: Posterior) -> dict[str, object]:
    """Profile likelihood, local vs global correlation and ridge geometry for one data set."""
    kappas = np.exp(np.linspace(np.log(1e-5), np.log(2.0), 4000))
    delta, at_zero = profile_likelihood(data, kappas)
    mode = posterior.map_estimate()
    hess = posterior.local_hessian(mode)
    _, cov = grid.log_moments()
    levels = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    points = ridge_points(posterior, grid, levels)
    mode_dir = RidgePoint(0.0, mode, hess).soft_direction()
    rows = []
    for p in points:
        vals = np.linalg.eigvalsh(p.hessian)
        d = p.soft_direction()
        rows.append({
            "level": p.level,
            "kappa": float(np.exp(p.q[1])),
            "local_correlation": float(-p.hessian[0, 1] / np.sqrt(p.hessian[0, 0] * p.hessian[1, 1])),
            "long_sd": float(1 / np.sqrt(vals[0])),
            "thin_sd": float(1 / np.sqrt(vals[1])),
            "angle_from_mode_direction_deg": float(np.degrees(np.arccos(min(1.0, abs(d @ mode_dir))))),
        })
    lo90, hi90, open90 = likelihood_interval(kappas, delta, 0.90)
    lo95, hi95, open95 = likelihood_interval(kappas, delta, 0.95)
    return {
        "profile_interval90": [lo90, hi90],
        "profile_interval90_reaches_grid_edge": open90,
        "profile_interval95": [lo95, hi95],
        "profile_interval95_reaches_grid_edge": open95,
        "profile_delta_nll_at_kappa_zero": float(at_zero),
        "profile_argmax_kappa": float(kappas[np.argmin(delta)]),
        "local_correlation_at_map": float(-hess[0, 1] / np.sqrt(hess[0, 0] * hess[1, 1])),
        "global_correlation_log": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        "mean_field_sd_log_kappa_predicted_from_map_hessian": float(1 / np.sqrt(hess[1, 1])),
        "mean_field_sd_log_kappa_predicted_from_global_precision": float(np.sqrt(1 / np.linalg.inv(cov)[1, 1])),
        "laplace_sd_log_kappa": float(np.sqrt(np.linalg.inv(hess)[1, 1])),
        "ridge": rows,
    }


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def _hmc_job(args: tuple) -> dict[str, float]:
    data, mass, start, seed, steps = args
    post = Posterior(data)
    result = hmc(post, start, mass=mass, seed=seed, leapfrog_range=steps)
    k = np.exp(result.draws[..., 1])
    return {
        "step_size": float(np.mean(result.step_sizes)),
        "acceptance": float(np.mean(result.acceptance)),
        "ess_bulk": float(diagnostics.ess_bulk(k)),
        "ess_tail": float(diagnostics.ess_tail(k)),
        "rhat_rank": float(diagnostics.rhat_rank(k)),
        "q05": float(np.quantile(k, 0.05)),
        "q95": float(np.quantile(k, 0.95)),
        "mcse_q05": float(diagnostics.mcse_quantile(k, 0.05)),
        "mcse_q95": float(diagnostics.mcse_quantile(k, 0.95)),
        "divergences": int(result.divergences),
        "gradients": int(result.gradients),
    }


def path_range(mean_steps: int) -> tuple[int, int]:
    """Jitter the number of leapfrog steps by about ±40% around its mean (15 -> 9..21)."""
    return max(1, round(0.6 * mean_steps)), round(1.4 * mean_steps)


def sampler_study(
    data: Observations,
    masses: dict[str, FloatArray],
    start: FloatArray,
    *,
    seeds: int,
    path_means: tuple[int, ...],
    base_seed: int,
    workers: int = 2,
) -> dict[str, object]:
    """Replicate HMC runs over seeds and path lengths for each constant mass matrix."""
    jobs, keys = [], []
    for name, mass in masses.items():
        for mean_steps in path_means:
            for s in range(seeds):
                jobs.append((data, mass, start, base_seed + 7919 * s + 101 * mean_steps, path_range(mean_steps)))
                keys.append((name, mean_steps))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_hmc_job, jobs))
    table: dict[str, dict[str, list[dict[str, float]]]] = {}
    for (name, mean_steps), res in zip(keys, results, strict=True):
        table.setdefault(name, {}).setdefault(str(mean_steps), []).append(res)
    summary: dict[str, object] = {}
    for name, by_path in table.items():
        summary[name] = {}
        for mean_steps, runs in by_path.items():
            arr = {k: np.array([r[k] for r in runs]) for k in runs[0]}
            summary[name][mean_steps] = {
                "runs": len(runs),
                "step_size_median": float(np.median(arr["step_size"])),
                "acceptance_median": float(np.median(arr["acceptance"])),
                "ess_bulk_per_1000_gradients_median": float(np.nanmedian(1000 * arr["ess_bulk"] / arr["gradients"])),
                "ess_tail_per_1000_gradients_median": float(np.nanmedian(1000 * arr["ess_tail"] / arr["gradients"])),
                "ess_tail_min": float(np.nanmin(arr["ess_tail"])),
                "rhat_rank_max": float(np.nanmax(arr["rhat_rank"])),
                "q05_mean": float(arr["q05"].mean()),
                "q05_sd_across_seeds": float(arr["q05"].std(ddof=1)),
                "q05_mcse_median": float(np.nanmedian(arr["mcse_q05"])),
                "q95_mean": float(arr["q95"].mean()),
                "q95_sd_across_seeds": float(arr["q95"].std(ddof=1)),
                "q95_mcse_median": float(np.nanmedian(arr["mcse_q95"])),
                "divergences_median": float(np.median(arr["divergences"])),
                "divergences_total": int(arr["divergences"].sum()),
                "q05_values": arr["q05"].tolist(),
                "q95_values": arr["q95"].tolist(),
            }
    return summary


def acceptance_by_position(result_probabilities: FloatArray, start_states: FloatArray,
                           grid: GridPosterior, bins: int = 10) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Mean Metropolis acceptance probability by posterior-quantile bin of the starting kappa."""
    edges = grid.kappa_quantiles(tuple(np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    k = np.exp(start_states[..., 1]).ravel()
    p = result_probabilities.ravel()
    idx = np.digitize(k, edges[1:-1])
    means = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(bins)])
    counts = np.array([np.sum(idx == b) for b in range(bins)], dtype=float)
    centres = (np.arange(bins) + 0.5) / bins
    return centres, means, counts


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


def design_study(x: FloatArray, *, replications: int, base_seed: int, resolution: int) -> dict[str, object]:
    """Compare time designs by local Fisher information and by exact-grid posteriors on repeated data."""
    narrow_t = np.linspace(0.218, 0.262, x.size)
    wide_t = np.linspace(0.06, 0.57, x.size)
    first = 0.06
    designs: dict[str, FloatArray] = {"narrow": narrow_t, "wide": wide_t}
    half = design.TwoTimeDesign(first, design.box_lucas_second_time(first, TRUE_DIFFUSIVITY), x.size // 2, x.size)
    designs["box_lucas_at_truth"] = design.assign_times(x, half)
    planned: dict[str, design.TwoTimeDesign] = {"box_lucas_at_truth": half}
    for label, guess in (("kappa_targeted_at_truth", TRUE_DIFFUSIVITY),
                         ("kappa_targeted_at_prior_centre", float(np.exp(PRIOR_MEAN[1]))),
                         ("kappa_targeted_at_guess_0.08", 0.08),
                         ("kappa_targeted_at_guess_0.60", 0.60)):
        plan = design.two_time_design(x, first, TRUE_AMPLITUDE, guess, OBSERVATION_SIGMA, PRIOR_SD, latest_time=2.0)
        planned[label] = plan
        designs[label] = design.assign_times(x, plan)

    out: dict[str, object] = {"designs": {}}
    for label, t in designs.items():
        info = design.fisher_information(x, t, TRUE_AMPLITUDE, TRUE_DIFFUSIVITY, OBSERVATION_SIGMA)
        sds, widths, covered = [], [], 0
        for r in range(replications):
            data = simulate_at(base_seed + 1009 * r, x, t)
            grid = adaptive_grid(Posterior(data), resolution=resolution)
            _, cov = grid.log_moments()
            lo, hi = grid.kappa_quantiles((0.05, 0.95))
            sds.append(np.sqrt(cov[1, 1]))
            widths.append(hi - lo)
            covered += bool(lo <= TRUE_DIFFUSIVITY <= hi)
        entry = {
            "laplace_sd_log_kappa_at_truth": design.laplace_sd_log_kappa(info, PRIOR_SD),
            "log_det_fisher_at_truth": float(np.log(np.linalg.det(info))),
            "exact_sd_log_kappa_median": float(np.median(sds)),
            "exact_interval90_width_kappa_median": float(np.median(widths)),
            "coverage90_kappa": covered / replications,
            "replications": replications,
            "latest_time": float(np.max(t)),
            "signal_at_latest_time_centre": float(solution(np.array(0.5), np.array(np.max(t)),
                                                           TRUE_AMPLITUDE, TRUE_DIFFUSIVITY)),
        }
        if label in planned:
            p = planned[label]
            entry |= {"early_time": p.early, "late_time": p.late, "n_early": p.n_early}
        out["designs"][label] = entry
    out["box_lucas_rule"] = "t2 = t1 + 1/(kappa pi^2), half of the observations at each time"
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def coverage_by_truth(kappas: tuple[float, ...], *, replications: int, base_seed: int,
                      resolution: int) -> dict[str, object]:
    """Coverage of exact-grid 90% kappa intervals at several fixed true diffusivities."""
    out: dict[str, object] = {}
    for design_name in ("narrow", "wide"):
        rows = {}
        for kappa in kappas:
            covered = below = 0
            for r in range(replications):
                data = simulate(base_seed + 1009 * r, design_name, diffusivity=kappa)
                grid = adaptive_grid(Posterior(data), resolution=resolution)
                lo, hi = grid.kappa_quantiles((0.05, 0.95))
                covered += bool(lo <= kappa <= hi)
                below += bool(kappa < lo)
            rows[f"{kappa:.3f}"] = {"coverage": covered / replications,
                                    "truth_below_interval": below / replications}
        out[design_name] = rows
    out["replications"] = replications
    return out


def coverage_prior_drawn(*, replications: int, base_seed: int, resolution: int) -> dict[str, float]:
    """Coverage when (A, kappa) are drawn from the prior: nominal by construction if the grid is right."""
    rng = np.random.default_rng(base_seed)
    covered = 0
    for r in range(replications):
        alpha, beta = rng.normal(PRIOR_MEAN, PRIOR_SD)
        data = simulate(base_seed + 1009 * r, "narrow", amplitude=float(np.exp(alpha)),
                        diffusivity=float(np.exp(beta)))
        grid = adaptive_grid(Posterior(data), resolution=resolution)
        lo, hi = grid.kappa_quantiles((0.05, 0.95))
        covered += bool(lo <= np.exp(beta) <= hi)
    return {"replications": replications, "coverage": covered / replications,
            "binomial_se_at_nominal": float(np.sqrt(0.09 / replications))}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {"identity": "#b2632f", "map_hessian": "#24618c", "posterior_covariance": "#5b8c3a"}
LABELS = {"identity": "Identity mass", "map_hessian": "MAP-Hessian mass",
          "posterior_covariance": "Posterior-covariance mass"}


def plot_identification(observed: dict[str, Observations], path) -> None:
    """Profile likelihood against posteriors under several priors, narrow and wide designs."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    kappas = np.exp(np.linspace(np.log(1e-3), np.log(1.0), 1500))
    for ax, name in zip(axes, ("narrow", "wide"), strict=True):
        data = observed[name]
        delta, _ = profile_likelihood(data, kappas)
        ax.plot(kappas, np.exp(-delta), color="#1d1d1f", lw=2.2, label="Profile likelihood (data only)")
        for scale, ls in ((0.35, ":"), (0.70, "-"), (2.80, "--")):
            post = Posterior(data, prior_sd=np.array([PRIOR_SD[0], scale]))
            grid = sensitivity_grid(post)
            axis, dens = grid.marginal_log_diffusivity()
            dens = dens / dens.max()
            ax.plot(np.exp(axis), dens, color="#3279ad", ls=ls, lw=1.6,
                    label=f"Posterior, prior SD of log κ = {scale:.2f}")
        prior_axis = np.linspace(np.log(1e-3), 0, 600)
        prior = np.exp(-0.5 * ((prior_axis - PRIOR_MEAN[1]) / PRIOR_SD[1]) ** 2)
        ax.plot(np.exp(prior_axis), prior, color="#c75536", lw=1.2, alpha=0.7, label="Prior (SD 0.70), scaled")
        ax.axvline(TRUE_DIFFUSIVITY, color="#777", lw=0.9, ls="-.")
        ax.set_xscale("log")
        ax.set(xlim=(1e-3, 1.0), ylim=(0, 1.05), xlabel="Diffusivity κ (log scale)",
               title=f"{name.capitalize()} time design")
    axes[0].set_ylabel("Relative likelihood / density (peak = 1)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_sampler(runs: dict[str, object], grid: GridPosterior, margins: dict[str, list[dict[str, float]]],
                 steps: dict[str, float], sweep: dict[str, object], path) -> None:
    """Traces, acceptance by position, stability margin and tail efficiency for three masses."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    ax = axes[0, 0]
    for name in ("identity", "map_hessian"):
        series = np.exp(runs[name].draws[0, :500, 1])
        ax.plot(series, color=COLORS[name], lw=0.8, alpha=0.9, label=LABELS[name])
    q05, q95 = grid.kappa_quantiles((0.05, 0.95))
    for v in (q05, q95):
        ax.axhline(v, color="black", ls="--", lw=0.9)
    ax.set(xlabel="Retained iteration (chain 1, first 500)", ylabel="κ",
           title="Traces; dashed lines: exact 5% and 95% quantiles")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    for name, res in runs.items():
        centres, means, _ = acceptance_by_position(res.accept_probability, res.start_states, grid)
        ax.plot(100 * centres, means, marker="o", ms=4, color=COLORS[name], label=LABELS[name])
    ax.set(xlabel="Posterior percentile of κ at the start of the transition",
           ylabel="Mean acceptance probability", ylim=(0, 1.02),
           title="Where proposals are rejected")
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[1, 0]
    for name, rows in margins.items():
        pct = [100 * r["level"] for r in rows]
        ratio = [steps[name] / r["thin_sd"] for r in rows]
        ax.plot(pct, ratio, marker="s", ms=4, color=COLORS[name], label=LABELS[name])
    ax.axhline(2.0, color="#a33", ls="--", lw=1.0)
    ax.text(2, 2.08, "leapfrog unstable above 2", color="#a33", fontsize=8)
    ax.set(xlabel="Posterior percentile of κ along the ridge", ylabel="Tuned step size / thinnest local SD",
           title="Stability margin of the warm-up step size")
    ax.legend(fontsize=8, loc="upper center")

    ax = axes[1, 1]
    for name, by_path in sweep.items():
        paths = sorted(int(p) for p in by_path)
        tail = [by_path[str(p)]["ess_tail_per_1000_gradients_median"] for p in paths]
        bulk = [by_path[str(p)]["ess_bulk_per_1000_gradients_median"] for p in paths]
        ax.plot(paths, tail, marker="o", color=COLORS[name], label=f"{LABELS[name]}: tail")
        ax.plot(paths, bulk, marker="^", ls=":", color=COLORS[name], label=f"{LABELS[name]}: bulk")
    ax.set(xlabel="Mean leapfrog steps per transition", ylabel="ESS per 1,000 gradients (median over seeds)",
           title="Efficiency across path lengths (solid: tail, dotted: bulk)")
    ax.set_xticks(sorted(int(p) for p in next(iter(sweep.values()))))
    top = max(max(v["ess_tail_per_1000_gradients_median"], v["ess_bulk_per_1000_gradients_median"])
              for by_path in sweep.values() for v in by_path.values())
    ax.set_ylim(0, 1.32 * top)
    handles = [plt.Line2D([], [], color=COLORS[n], lw=2, label=LABELS[n]) for n in sweep]
    ax.legend(handles=handles, fontsize=7.5, loc="upper center", ncol=3)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_design(x: FloatArray, study: dict[str, object], path) -> None:
    """Laplace SD of log kappa over uniform windows, and the two-time trade-off curve."""
    starts = np.linspace(0.0, 0.8, 161)
    widths = np.linspace(0.01, 1.0, 199)
    sd_map = design.window_map(x, starts, widths, TRUE_AMPLITUDE, TRUE_DIFFUSIVITY, OBSERVATION_SIGMA, PRIOR_SD)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    ax = axes[0]
    levels = np.array([0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6])
    cs = ax.contourf(starts, widths, sd_map, levels=levels, cmap="Blues_r", extend="both")
    ax.contour(starts, widths, sd_map, levels=levels, colors="white", linewidths=0.4)
    fig.colorbar(cs, ax=ax, label="Laplace posterior SD of log κ")
    ax.scatter([0.218], [0.044], marker="o", s=60, color="#c75536", zorder=3)
    ax.annotate("narrow", (0.218, 0.044), xytext=(0.30, 0.12), color="#c75536",
                arrowprops={"arrowstyle": "-", "color": "#c75536"})
    ax.scatter([0.06], [0.51], marker="o", s=60, color="#1d1d1f", zorder=3)
    ax.annotate("wide", (0.06, 0.51), xytext=(0.16, 0.62), color="#1d1d1f",
                arrowprops={"arrowstyle": "-", "color": "#1d1d1f"})
    ax.set(xlabel="First measurement time", ylabel="Width of the measurement window",
           title="24 evenly spaced times: where to put the window")

    ax = axes[1]
    first = 0.06
    late = np.linspace(first + 0.01, 1.6, 400)
    c = TRUE_DIFFUSIVITY * np.pi**2
    for n_early, color, label in ((12, "#3279ad", "12 early / 12 late (D-optimal split)"),
                                  (study["designs"]["kappa_targeted_at_truth"]["n_early"], "#1d1d1f",
                                   f"{study['designs']['kappa_targeted_at_truth']['n_early']} early / "
                                   f"{x.size - study['designs']['kappa_targeted_at_truth']['n_early']} late")):
        mask = design.early_mask(x, n_early)
        sd = [design.laplace_sd_log_kappa(design.fisher_information(x, np.where(mask, first, tl), TRUE_AMPLITUDE,
                                                                    TRUE_DIFFUSIVITY, OBSERVATION_SIGMA), PRIOR_SD)
              for tl in late]
        ax.plot(late, sd, color=color, lw=2, label=label)
    ax.axvline(first + 1 / c, color="#3279ad", ls="--", lw=1)
    ax.text(first + 1 / c - 0.055, 0.16, "t₁ + 1/(κπ²)", color="#3279ad", fontsize=8, rotation=90)
    late_best = study["designs"]["kappa_targeted_at_truth"]["late_time"]
    ax.axvline(late_best, color="#1d1d1f", ls=":", lw=1)
    ax.text(late_best + 0.015, 0.16, "best late time", color="#1d1d1f", fontsize=8, rotation=90)
    narrow_sd = study["designs"]["narrow"]["laplace_sd_log_kappa_at_truth"]
    ax.axhline(narrow_sd, color="#c75536", ls="-.", lw=1)
    ax.text(0.9, narrow_sd - 0.025, "24 times in the narrow window", color="#c75536", fontsize=8)
    ax.set(xlabel="Second measurement time (first fixed at 0.06)", ylabel="Laplace posterior SD of log κ",
           ylim=(0, 0.42), title="Two measurement times: dispersion against decay")
    ax.legend(fontsize=8, loc="center right")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_coverage_by_truth(cov: dict[str, object], path) -> None:
    """Coverage of 90% intervals as the true diffusivity moves relative to the prior."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    n = cov["replications"]
    for name, color, marker in (("narrow", "#3279ad", "o"), ("wide", "#1d1d1f", "s")):
        rows = cov[name]
        ks = np.array([float(k) for k in rows])
        vals = np.array([rows[k]["coverage"] for k in rows])
        ax.plot(ks, vals, marker=marker, color=color, lw=1.6, label=f"{name.capitalize()} time design")
    se = np.sqrt(0.09 / n)
    ax.axhspan(0.9 - 1.96 * se, 0.9 + 1.96 * se, color="#e9cc9e", alpha=0.35, label="Nominal 90% ± 1.96 binomial SE")
    ax.axhline(0.9, color="#af552e", ls="--", lw=1)
    ax.axvline(np.exp(PRIOR_MEAN[1]), color="#777", ls=":", lw=1)
    ax.text(np.exp(PRIOR_MEAN[1]) * 0.97, 0.45, "prior median", color="#555", fontsize=8, ha="right")
    ax.set(xscale="log", xlabel="True diffusivity κ (log scale)", ylabel="Coverage of the 90% interval for κ",
           ylim=(0, 1.02), title=f"Exact-grid interval coverage at fixed true κ ({n} data sets per point)")
    ticks = [float(k) for k in cov["narrow"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{k:g}" for k in ticks])
    ax.minorticks_off()
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=190)
    plt.close(fig)
