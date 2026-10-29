"""Reproducible FCFS queue experiments for the accompanying Medium article.

All times are in minutes.  This module simulates a GI/G/c queue with infinite
waiting room, no abandonment, and independent interarrival/service sequences.
For Poisson arrivals it checks sample means against exact M/G/1 or M/M/c
results; for renewal arrivals it reports a clearly labelled approximation.

Run ``python analysis.py`` to regenerate the full numerical evidence and
figures, or ``python analysis.py --quick`` for separate smoke-test outputs.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ArrivalKind = Literal["exponential", "hyperexponential"]
ServiceKind = Literal["deterministic", "exponential", "lognormal"]

SEED: int = 20261029
ARRIVAL_RATE: float = 0.8  # Requests / minute.
THRESHOLDS: tuple[float, ...] = (0.01, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80)
UTILIZATION_GRID: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class QueueSpec:
    """One queue configuration, with the same exogenous demand mean unless noted."""

    slug: str
    label: str
    arrival_kind: ArrivalKind
    service_kind: ServiceKind
    servers: int = 1
    mean_service: float = 1.0

    def __post_init__(self) -> None:
        if not self.slug or not self.label:
            raise ValueError("slug and label must be nonempty")
        if self.servers < 1 or not isinstance(self.servers, int):
            raise ValueError("servers must be a positive integer")
        if not math.isfinite(self.mean_service) or self.mean_service <= 0:
            raise ValueError("mean_service must be finite and positive")
        if ARRIVAL_RATE * self.mean_service >= self.servers:
            raise ValueError("the specified queue must be stable")

    @property
    def arrival_cv2(self) -> float:
        """Squared coefficient of variation of an interarrival time."""
        return 4.0 if self.arrival_kind == "hyperexponential" else 1.0

    @property
    def service_cv2(self) -> float:
        """Squared coefficient of variation of a service duration."""
        return {"deterministic": 0.0, "exponential": 1.0, "lognormal": 4.0}[
            self.service_kind
        ]

    @property
    def utilization(self) -> float:
        return ARRIVAL_RATE * self.mean_service / self.servers


SPECS: tuple[QueueSpec, ...] = (
    QueueSpec("mm1", "M/M/1 · baseline", "exponential", "exponential"),
    QueueSpec("md1", "M/D/1 · regular service", "exponential", "deterministic"),
    QueueSpec("mg1", "M/G/1 · variable service", "exponential", "lognormal"),
    QueueSpec("gim1", "GI/M/1 · bursty arrivals", "hyperexponential", "exponential"),
    QueueSpec("mm1_fast", "M/M/1 · 15% faster", "exponential", "exponential", mean_service=0.85),
    QueueSpec("mm2", "M/M/2 · add a server", "exponential", "exponential", servers=2),
)


def sample_interarrivals(
    rng: np.random.Generator,
    n: int,
    rate: float,
    kind: ArrivalKind,
) -> NDArray[np.float64]:
    """Draw iid interarrival times with mean ``1/rate``.

    For hyperexponential arrivals, the balanced two-phase mixture has SCV 4:
    weights p, 1-p; phase rates 2*p*rate and 2*(1-p)*rate.  This is a
    renewal process, not a correlated or time-varying arrival process.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    if kind == "exponential":
        return rng.exponential(scale=1.0 / rate, size=n)
    if kind == "hyperexponential":
        p = (1.0 + math.sqrt(0.6)) / 2.0
        phase = rng.random(n) < p
        rates = np.where(phase, 2 * p * rate, 2 * (1 - p) * rate)
        return rng.exponential(scale=1.0 / rates)
    raise ValueError(f"unsupported arrival kind: {kind}")


def sample_services(
    rng: np.random.Generator,
    n: int,
    mean: float,
    kind: ServiceKind,
) -> NDArray[np.float64]:
    """Draw iid durations with controlled mean and squared CV (0, 1, or 4)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("mean must be finite and positive")
    if kind == "deterministic":
        return np.full(n, mean, dtype=np.float64)
    if kind == "exponential":
        return rng.exponential(scale=mean, size=n)
    if kind == "lognormal":
        sigma2 = math.log(5.0)  # CV^2 = exp(sigma^2)-1 = 4.
        return rng.lognormal(
            mean=math.log(mean) - sigma2 / 2.0,
            sigma=math.sqrt(sigma2),
            size=n,
        )
    raise ValueError(f"unsupported service kind: {kind}")


def simulate_fcfs(
    arrival_times: NDArray[np.float64],
    service_times: NDArray[np.float64],
    servers: int = 1,
) -> NDArray[np.float64]:
    """Return each arrival's waiting time before service under FCFS.

    Incoming jobs are assigned to the first server that can serve them.  The
    heap holds each server's next-free time.  This is exact for infinite
    patience, non-preemptive FCFS and identical work-conserving servers.
    """
    if (
        arrival_times.ndim != 1
        or service_times.ndim != 1
        or arrival_times.shape != service_times.shape
        or arrival_times.size == 0
    ):
        raise ValueError("arrival_times and service_times must be equal nonempty 1-D arrays")
    if not isinstance(servers, int) or servers < 1:
        raise ValueError("servers must be a positive integer")
    if (
        not np.all(np.isfinite(arrival_times))
        or not np.all(np.isfinite(service_times))
        or np.any(service_times <= 0)
        or np.any(np.diff(arrival_times) < 0)
    ):
        raise ValueError("times must be finite, arrivals sorted, and service times positive")

    waits = np.empty(arrival_times.size, dtype=np.float64)
    if servers == 1:
        available = 0.0
        for i, (arrival, duration) in enumerate(zip(arrival_times, service_times)):
            start = max(float(arrival), available)
            waits[i] = start - float(arrival)
            available = start + float(duration)
        return waits

    available_times = [0.0] * servers
    heapq.heapify(available_times)
    for i, (arrival, duration) in enumerate(zip(arrival_times, service_times)):
        start = max(float(arrival), available_times[0])
        waits[i] = start - float(arrival)
        heapq.heapreplace(available_times, start + float(duration))
    return waits


def expected_mg1_wait(rate: float, mean_service: float, service_cv2: float) -> float:
    """Exact stationary FCFS M/G/1 queue wait (Pollaczek–Khinchine)."""
    rho = rate * mean_service
    if rate <= 0 or mean_service <= 0 or service_cv2 < 0 or rho >= 1:
        raise ValueError("M/G/1 requires a positive rate, finite SCV and rho < 1")
    return rate * (mean_service**2) * (1.0 + service_cv2) / (2.0 * (1.0 - rho))


def expected_mmc_wait(rate: float, mean_service: float, servers: int) -> float:
    """Exact stationary M/M/c Erlang-C waiting time under FCFS."""
    if not isinstance(servers, int) or servers < 1 or rate <= 0 or mean_service <= 0:
        raise ValueError("M/M/c requires positive rate, mean, integer servers")
    offered = rate * mean_service
    rho = offered / servers
    if rho >= 1:
        raise ValueError("M/M/c is unstable when rho >= 1")
    idle_normalizer = sum(offered**k / math.factorial(k) for k in range(servers))
    all_busy_weight = offered**servers / (math.factorial(servers) * (1 - rho))
    p_wait = all_busy_weight / (idle_normalizer + all_busy_weight)
    return p_wait * mean_service / (servers - offered)


def kingman_wait(rate: float, mean_service: float, arrival_cv2: float, service_cv2: float) -> float:
    """Kingman's heavy-traffic GI/G/1 mean-wait approximation; not an identity."""
    rho = rate * mean_service
    if rho <= 0 or rho >= 1 or min(arrival_cv2, service_cv2) < 0:
        raise ValueError("Kingman's approximation requires 0 < rho < 1 and nonnegative SCVs")
    return rho / (1.0 - rho) * (arrival_cv2 + service_cv2) / 2.0 * mean_service


def prediction(spec: QueueSpec) -> tuple[str, float]:
    """Return an exact baseline or explicitly approximate formula."""
    if spec.arrival_kind == "hyperexponential":
        return (
            "Kingman approximation",
            kingman_wait(ARRIVAL_RATE, spec.mean_service, spec.arrival_cv2, spec.service_cv2),
        )
    if spec.servers == 1:
        return (
            "Pollaczek–Khinchine exact",
            expected_mg1_wait(ARRIVAL_RATE, spec.mean_service, spec.service_cv2),
        )
    if spec.service_kind == "exponential":
        return ("Erlang C exact", expected_mmc_wait(ARRIVAL_RATE, spec.mean_service, spec.servers))
    raise ValueError("no exact formula implemented for this queue")


def one_replication(
    spec: QueueSpec,
    repetition: int,
    n_observed: int,
    n_warmup: int,
    *,
    seed: int = SEED,
) -> NDArray[np.float64]:
    """Simulate from empty, discard warm-up arrivals, return retained waits.

    The same Poisson arrivals and exponential-service uniform streams are
    reused across policy comparisons within the same repetition.  Different
    repetitions are independent under different deterministic seeds.
    """
    if not isinstance(repetition, int) or repetition < 0 or n_observed < 10 or n_warmup < 0:
        raise ValueError("repetition and sample lengths must be valid integers")
    n = n_observed + n_warmup
    arrival_key = 0 if spec.arrival_kind == "exponential" else 1
    service_key = {"deterministic": 0, "exponential": 1, "lognormal": 2}[spec.service_kind]
    a_rng = np.random.default_rng(np.random.SeedSequence([seed, repetition, arrival_key]))
    s_rng = np.random.default_rng(np.random.SeedSequence([seed, repetition, 100 + service_key]))
    arrivals = np.cumsum(sample_interarrivals(a_rng, n, ARRIVAL_RATE, spec.arrival_kind))
    services = sample_services(s_rng, n, spec.mean_service, spec.service_kind)
    return simulate_fcfs(arrivals, services, spec.servers)[n_warmup:]


def _confidence_interval(values: NDArray[np.float64]) -> tuple[float, float]:
    """Normal-approximation interval across independent simulation repetitions."""
    center = float(np.mean(values))
    if values.size < 2:
        raise ValueError("at least two independent repetitions are required")
    radius = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return center - radius, center + radius


def run_experiment(quick: bool = False) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Compare matched-mean demand scenarios and an independent utilization sweep."""
    if not isinstance(quick, bool):
        raise TypeError("quick must be bool")
    repetitions = 3 if quick else 16
    observed = 3_000 if quick else 60_000
    warmup = 1_500 if quick else 12_000
    rows: list[dict[str, object]] = []
    tails: list[dict[str, object]] = []
    for spec in SPECS:
        sample_mean: list[float] = []
        sample_p95: list[float] = []
        sample_sla: list[float] = []
        exceed = np.empty((repetitions, len(THRESHOLDS)), dtype=np.float64)
        for repetition in range(repetitions):
            waits = one_replication(spec, repetition, observed, warmup)
            sample_mean.append(float(waits.mean()))
            sample_p95.append(float(np.quantile(waits, 0.95)))
            sample_sla.append(float(np.mean(waits > 5.0)))
            exceed[repetition] = np.mean(waits[:, None] > np.asarray(THRESHOLDS)[None, :], axis=0)
        means = np.asarray(sample_mean)
        low, high = _confidence_interval(means)
        theory_type, theory = prediction(spec)
        rows.append(
            {
                "scenario": spec.slug,
                "label": spec.label,
                "arrival_process": spec.arrival_kind,
                "service_distribution": spec.service_kind,
                "servers": spec.servers,
                "mean_service_min": spec.mean_service,
                "arrival_cv2": spec.arrival_cv2,
                "service_cv2": spec.service_cv2,
                "utilization_per_server": spec.utilization,
                "repetitions": repetitions,
                "recorded_arrivals_per_rep": observed,
                "warmup_arrivals_per_rep": warmup,
                "theory_type": theory_type,
                "predicted_mean_wait_min": theory,
                "simulated_mean_wait_min": float(means.mean()),
                "ci95_low_min": low,
                "ci95_high_min": high,
                "simulated_p95_wait_min": float(np.mean(sample_p95)),
                "fraction_wait_over_5min": float(np.mean(sample_sla)),
                "mean_system_time_min": float(means.mean()) + spec.mean_service,
            }
        )
        for j, threshold in enumerate(THRESHOLDS):
            tails.append(
                {
                    "scenario": spec.slug,
                    "threshold_min": threshold,
                    "fraction_exceeding": float(exceed[:, j].mean()),
                }
            )

    sensitivity: list[dict[str, object]] = []
    sweep_reps = 2 if quick else 8
    for rho in UTILIZATION_GRID:
        # The mixing time increases sharply near saturation.  Longer runs
        # reduce start-empty bias for rho=0.90 and especially rho=0.95.
        sweep_observed = 3_000 if quick else (150_000 if rho >= 0.95 else 80_000 if rho >= 0.90 else 30_000)
        sweep_warmup = 1_500 if quick else (50_000 if rho >= 0.95 else 20_000 if rho >= 0.90 else 7_000)
        # The sweep changes only arrival intensity; it always has one server.
        for service in ("deterministic", "exponential", "lognormal"):
            cv2 = {"deterministic": 0.0, "exponential": 1.0, "lognormal": 4.0}[service]
            rep_means: list[float] = []
            for repetition in range(sweep_reps):
                n = sweep_observed + sweep_warmup
                a_rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, int(rho * 100), repetition]))
                s_rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, 8, int(rho * 100), repetition, int(cv2)])
                )
                arrivals = np.cumsum(sample_interarrivals(a_rng, n, rho, "exponential"))
                services = sample_services(s_rng, n, 1.0, service)
                waits = simulate_fcfs(arrivals, services)[sweep_warmup:]
                rep_means.append(float(waits.mean()))
            vals = np.asarray(rep_means)
            low, high = _confidence_interval(vals)
            sensitivity.append(
                {
                    "rho": rho,
                    "service": service,
                    "cv2_service": cv2,
                    "repetitions": sweep_reps,
                    "simulated_mean_wait_min": float(vals.mean()),
                    "ci95_low_min": low,
                    "ci95_high_min": high,
                    "exact_mean_wait_min": expected_mg1_wait(rho, 1.0, cv2),
                }
            )
    return rows, tails, sensitivity


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write stable column-ordered CSV with an explicit UTF-8 encoding."""
    if not rows:
        raise ValueError("at least one row must be provided")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(
    summary: list[dict[str, object]],
    tails: list[dict[str, object]],
    sensitivity: list[dict[str, object]],
    output: Path,
    *,
    suffix: str = "",
) -> None:
    """Render article figures from recorded estimates only, never hardcoded points."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

    ordered = list(summary)
    x = np.arange(len(ordered))
    observed = np.array([float(item["simulated_mean_wait_min"]) for item in ordered])
    lower = observed - np.array([float(item["ci95_low_min"]) for item in ordered])
    upper = np.array([float(item["ci95_high_min"]) for item in ordered]) - observed
    predicted = np.array([float(item["predicted_mean_wait_min"]) for item in ordered])
    fig, ax = plt.subplots(figsize=(11.5, 5.4), constrained_layout=True)
    ax.errorbar(x - 0.09, observed, yerr=[lower, upper], fmt="o", capsize=3, label="Simulation: mean ± 95% CI")
    ax.scatter(x + 0.09, predicted, marker="x", s=68, label="Analytic value (GI/M/1: approximation)")
    ax.set_xticks(x, [str(item["label"]).replace(" · ", "\n") for item in ordered])
    ax.set_ylabel("Mean queue wait (minutes)")
    ax.set_title("Same mean demand, different service and arrival processes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / f"fig1_same_mean_different_wait{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for service, caption in (
        ("deterministic", "Deterministic service (CV² = 0)"),
        ("exponential", "Exponential service (CV² = 1)"),
        ("lognormal", "Lognormal service (CV² = 4)"),
    ):
        points = sorted((r for r in sensitivity if r["service"] == service), key=lambda r: float(r["rho"]))
        rho = [float(p["rho"]) for p in points]
        exact = [float(p["exact_mean_wait_min"]) for p in points]
        estimate = [float(p["simulated_mean_wait_min"]) for p in points]
        line = ax.plot(rho, exact, label=caption)[0]
        ax.scatter(rho, estimate, s=22, color=line.get_color())
    ax.set_xlabel("Utilization ρ of the single server")
    ax.set_ylabel("Mean queue wait (minutes)")
    ax.set_title("Near saturation, a small capacity change is not a small wait change")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output / f"fig2_utilization_cliff{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for slug, caption in (
        ("mm1", "M/M/1"),
        ("md1", "M/D/1"),
        ("mg1", "M/G/1, CV²(service) = 4"),
        ("gim1", "GI/M/1, CV²(arrivals) = 4"),
    ):
        points = sorted((r for r in tails if r["scenario"] == slug), key=lambda r: float(r["threshold_min"]))
        ax.plot(
            [float(p["threshold_min"]) for p in points],
            [float(p["fraction_exceeding"]) for p in points],
            marker="o",
            label=caption,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Queue-wait threshold (minutes)")
    ax.set_ylabel("Fraction of customers waiting longer")
    ax.set_title("An average forecast misses the waiting-time tail")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output / f"fig3_waiting_tails{suffix}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Recreate the tables and figures for either the full or quick experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small independent smoke test; preserve full outputs")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not isinstance(args.output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    summary, tails, sensitivity = run_experiment(quick=bool(args.quick))
    suffix = "_quick" if args.quick else ""
    result_dir = args.output_root / "results"
    write_csv(result_dir / f"summary{suffix}.csv", summary)
    write_csv(result_dir / f"tails{suffix}.csv", tails)
    write_csv(result_dir / f"sensitivity{suffix}.csv", sensitivity)
    make_figures(summary, tails, sensitivity, args.output_root / "figures", suffix=suffix)
    for row in summary:
        print(
            f"{row['scenario']:9}  theory={float(row['predicted_mean_wait_min']):7.3f}  "
            f"sim={float(row['simulated_mean_wait_min']):7.3f}  "
            f"95% CI=({float(row['ci95_low_min']):.3f}, {float(row['ci95_high_min']):.3f})  "
            f"p95={float(row['simulated_p95_wait_min']):7.3f}  "
            f"P(wait>5)={float(row['fraction_wait_over_5min']):.3f}"
        )


if __name__ == "__main__":
    main()
