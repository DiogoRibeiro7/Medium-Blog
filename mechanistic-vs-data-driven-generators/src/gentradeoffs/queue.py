"""Exact event-by-event simulation of one contact-centre day.

The centre is a first-come-first-served multi-server queue with abandonment
(M/G/c+G when arrivals are Poisson), optionally with redials: a caller who
hangs up may call again later, and the redial joins the queue as a new call.

Calls are processed in the order they arrive. Each is assigned the agent who
can start serving it earliest, taking shifts into account, and is abandoned
without occupying anyone if that start lies beyond the caller's patience.
Waiting calls are therefore served in arrival order and abandoners never
hold an agent, which reproduces FCFS service exactly. A redial is scheduled
at the moment of hanging up plus a delay, which is always later than the
call being processed, so a heap of pending redials keeps the processing
order correct.

Staffing is piecewise constant over half-hour intervals. Agent ``k`` is on
shift in interval ``j`` when ``k < staff[j]``, so when staffing falls the
highest-numbered agents finish their current call and log off. The agents of
the last interval stay after closing until the queue is empty.

Staffing is either given, or set during the day by the planners' rule
(``use_policy``): a square-root plan on the forecast, revised from the
fourth interval onwards by comparing the calls offered so far with the
forecast, then thinned by absence. Because redials make the offered calls
depend on how the queue went, the revision has to happen inside the
simulation, at the moment the planners would make it.

All random numbers are drawn by the caller and passed in, so one set of
callers can be pushed through different staffing plans (common random
numbers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

INTERVAL_SECONDS = 1800.0
N_INTERVALS = 24
SL_THRESHOLD = 20.0
CLOSE = INTERVAL_SECONDS * N_INTERVALS

# Columns of the per-interval output.
OFFERED, ANSWERED, ABANDONED, WITHIN, TOTAL_WAIT, TOTAL_SERVICE = range(6)

# Layout of the policy parameter vector.
P_BETA, P_AHT, P_FROM, P_ELAST, P_ADD, P_CUT, P_ATTEND = range(7)
MAX_SLOTS = 256


@njit(cache=True)
def _next_on_shift(k, t, staff, length):
    """Earliest time >= t at which agent k is on shift (inf if never)."""
    n_int = staff.shape[0]
    j = int(t // length)
    if j < 0:
        j = 0
    while j < n_int:
        if k < staff[j]:
            start = j * length
            return t if t > start else start
        j += 1
    if k < staff[n_int - 1]:
        end = n_int * length
        return t if t > end else end
    return np.inf


@njit(cache=True)
def _present(target, attend_row, attendance):
    """Agents who turn up out of ``target`` rostered."""
    n = 0
    for k in range(min(target, attend_row.shape[0])):
        if attend_row[k] < attendance:
            n += 1
    return max(n, 1)


@njit(cache=True)
def _heap_push(ht, hi, size, t, i):
    pos = size
    ht[pos] = t
    hi[pos] = i
    while pos > 0:
        parent = (pos - 1) // 2
        if ht[parent] <= ht[pos]:
            break
        ht[parent], ht[pos] = ht[pos], ht[parent]
        hi[parent], hi[pos] = hi[pos], hi[parent]
        pos = parent
    return size + 1


@njit(cache=True)
def _heap_pop(ht, hi, size):
    t, i = ht[0], hi[0]
    size -= 1
    ht[0], hi[0] = ht[size], hi[size]
    pos = 0
    while True:
        left = 2 * pos + 1
        if left >= size:
            break
        child = left
        if left + 1 < size and ht[left + 1] < ht[left]:
            child = left + 1
        if ht[pos] <= ht[child]:
            break
        ht[pos], ht[child] = ht[child], ht[pos]
        hi[pos], hi[child] = hi[child], hi[pos]
        pos = child
    return t, i, size


@njit(cache=True)
def simulate_core(arrivals, service, patience, retry_u, retry_delay,
                  pool_service, pool_patience, pool_u, pool_delay,
                  p_retry, staff, use_policy, forecast, policy, attend,
                  extra, length, threshold):
    """Simulate a day. ``staff`` is read (given staffing) or written
    (policy staffing); ``extra`` agents are added to every interval."""
    n_int = staff.shape[0]
    out = np.zeros((n_int, 6))
    load = forecast * policy[P_AHT] / length
    sched = np.zeros(n_int, np.int64)
    kmax = 0
    if use_policy:
        for j in range(n_int):
            sched[j] = int(np.ceil(load[j] + policy[P_BETA]
                                   * np.sqrt(load[j])))
            staff[j] = max(_present(sched[j], attend[j], policy[P_ATTEND])
                           + extra, 1)
            bound = sched[j] + int(policy[P_ADD]) + max(extra, 0)
            if bound > kmax:
                kmax = bound
        decided = int(policy[P_FROM])
    else:
        for j in range(n_int):
            staff[j] = max(staff[j] + extra, 1)
            if staff[j] > kmax:
                kmax = staff[j]
        decided = n_int
    free = np.zeros(max(kmax, 1))

    n_pool = pool_service.shape[0]
    ht = np.empty(n_pool + 1)
    hi = np.empty(n_pool + 1, np.int64)
    hsize = 0
    next_pool = 0
    p = 0
    n_fresh = arrivals.shape[0]
    while True:
        if p < n_fresh and (hsize == 0 or arrivals[p] <= ht[0]):
            a = arrivals[p]
            s = service[p]
            pat = patience[p]
            u = retry_u[p]
            dly = retry_delay[p]
            p += 1
        elif hsize > 0:
            a, idx, hsize = _heap_pop(ht, hi, hsize)
            s = pool_service[idx]
            pat = pool_patience[idx]
            u = pool_u[idx]
            dly = pool_delay[idx]
        else:
            break
        # Planners revise interval j once interval j-2 is complete.
        while decided < n_int and a >= (decided - 1) * length:
            j = decided
            seen = 0.0
            expected = 0.0
            for i in range(j - 1):
                seen += out[i, OFFERED]
                expected += forecast[i]
            ratio = (seen / expected) ** policy[P_ELAST]
            rev = load[j] * ratio
            target = int(np.ceil(rev + policy[P_BETA] * np.sqrt(rev)))
            lo = sched[j] - int(policy[P_CUT])
            hi_ = sched[j] + int(policy[P_ADD])
            target = min(max(target, lo), hi_)
            staff[j] = max(_present(target, attend[j], policy[P_ATTEND])
                           + extra, 1)
            decided += 1
        best = np.inf
        bk = -1
        for k in range(kmax):
            t0 = free[k] if free[k] > a else a
            st = _next_on_shift(k, t0, staff, length)
            if st < best:
                best = st
                bk = k
        j = int(a // length)
        if j >= n_int:
            j = n_int - 1
        wait = best - a
        out[j, OFFERED] += 1.0
        if wait > pat:
            out[j, ABANDONED] += 1.0
            out[j, TOTAL_WAIT] += pat
            if u < p_retry and next_pool < n_pool:
                t_new = a + pat + dly
                if t_new < n_int * length:
                    hsize = _heap_push(ht, hi, hsize, t_new, next_pool)
                    next_pool += 1
        else:
            out[j, ANSWERED] += 1.0
            out[j, TOTAL_WAIT] += wait
            out[j, TOTAL_SERVICE] += s
            if wait <= threshold:
                out[j, WITHIN] += 1.0
            free[bk] = best + s
    return out


@dataclass
class Callers:
    """Everything random about one day's callers, drawn up front."""

    arrivals: np.ndarray
    service: np.ndarray
    patience: np.ndarray
    retry_u: np.ndarray
    retry_delay: np.ndarray
    pool_service: np.ndarray
    pool_patience: np.ndarray
    pool_u: np.ndarray
    pool_delay: np.ndarray


def uniform_arrivals(rng: np.random.Generator, counts: np.ndarray,
                     length: float = INTERVAL_SECONDS) -> np.ndarray:
    """Sorted times with ``counts[j]`` arrivals uniform in interval j.

    Given its count, a Poisson process in an interval has independent
    uniform arrival times, so this is the process conditioned on counts.
    """
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.repeat(np.arange(counts.size) * length, counts)
    times = starts + rng.uniform(0.0, length, size=starts.size)
    times.sort()
    return times


def run(callers: Callers, *, staff: np.ndarray | None = None,
        p_retry: float = 0.0, policy: np.ndarray | None = None,
        forecast: np.ndarray | None = None, attend: np.ndarray | None = None,
        extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run a day with either given ``staff`` or the planners' ``policy``.

    Returns the per-interval output and the staffing that was used.
    """
    use_policy = staff is None
    if use_policy:
        staff_arr = np.zeros(N_INTERVALS, np.int64)
    else:
        staff_arr = np.asarray(staff, dtype=np.int64).copy()
        policy = np.zeros(7)
        forecast = np.ones(N_INTERVALS)
        attend = np.zeros((N_INTERVALS, 1))
    out = simulate_core(callers.arrivals, callers.service, callers.patience,
                        callers.retry_u, callers.retry_delay,
                        callers.pool_service, callers.pool_patience,
                        callers.pool_u, callers.pool_delay, float(p_retry),
                        staff_arr, use_policy, np.asarray(forecast, float),
                        np.asarray(policy, float), attend, int(extra),
                        INTERVAL_SECONDS, SL_THRESHOLD)
    return out, staff_arr


def summarise(out: np.ndarray, staff: np.ndarray) -> dict[str, np.ndarray]:
    """Turn simulation output into the columns of the interval table."""
    offered = out[:, OFFERED]
    answered = out[:, ANSWERED]
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_offered = np.where(offered > 0, offered, np.nan)
        safe_answered = np.where(answered > 0, answered, np.nan)
        return {
            "offered": offered,
            "agents": np.asarray(staff, dtype=float),
            "abandon_rate": out[:, ABANDONED] / safe_offered,
            "sl": out[:, WITHIN] / safe_offered,
            "awt": out[:, TOTAL_WAIT] / safe_offered,
            "aht": out[:, TOTAL_SERVICE] / safe_answered,
        }
