"""Legal duties and pairings: the two resource-constrained path layers of crew pairing.

A *duty* is a sequence of flights flown by one crew between a report and a release,
with every connection long enough and the flight duty period (FDP) within EASA
Table 2. Duties are paths in the flight-connection network with two resources (elapsed
FDP and number of sectors); they are enumerated once by depth-first search with
pruning, which is exact because the Table 2 limit never grows with more sectors.

A *pairing* is a sequence of duties that starts and ends at the crew base, separated
by rests away from base. With at most two duties per pairing (the configuration used
here), pairings are paths of length one or two in the duty network, and the legal
two-duty paths are listed once as index arrays so that pricing can evaluate all of them
with vectorised arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from airline_ops.network.model import DAY, Network
from airline_ops.planning.rules import PlanningRules


@dataclass(frozen=True, slots=True)
class Duty:
    """A legal duty period.

    Times in ``dep_offsets`` and ``arr_offsets`` are minutes after the report time.
    """

    idx: int
    flights: tuple[int, ...]
    report_utc: int
    length: int
    fdp: int
    start_station: str
    end_station: str
    dep_offsets: tuple[int, ...]
    arr_offsets: tuple[int, ...]
    short_connections: tuple[tuple[int, int], ...]
    change_connections: tuple[tuple[int, int, int], ...]  # (from, to, ground minutes) when changing aircraft

    @property
    def sectors(self) -> int:
        """Number of flights in the duty."""
        return len(self.flights)


@dataclass(frozen=True)
class DutySet:
    """All legal duties of a network under one set of rules, with incidence matrices."""

    duties: tuple[Duty, ...]
    short_connections: tuple[tuple[int, int], ...]
    flight_incidence: NDArray[np.int32]
    short_incidence: NDArray[np.int32]
    masks: NDArray[np.uint64]

    @property
    def size(self) -> int:
        """Number of duties."""
        return len(self.duties)


def enumerate_duties(
    network: Network, rules: PlanningRules, *, allowed_short: set[tuple[int, int]] | None = None
) -> DutySet:
    """Enumerate every legal duty.

    Args:
        network: The daily schedule.
        rules: Connection and FTL rules (including any robustness buffer).
        allowed_short: When given, short connections (crew stays with the aircraft
            because the connection is too short to change aircraft) are allowed only
            if they are in this set, which is how a fixed aircraft routing restricts
            crew pairing in the sequential approach.

    """
    flights = network.flights
    outgoing: dict[int, list[tuple[int, int, bool]]] = {f.idx: [] for f in flights}
    for i in flights:
        station = i.destination
        same_min = rules.crew_same_aircraft_min(station)
        change_min = rules.crew_change_aircraft_min()
        for j in flights:
            if j.origin != station or j.idx == i.idx:
                continue
            gap = (j.dep - i.arr) % DAY
            if gap < same_min or gap > rules.max_sit_min:
                continue
            short = gap < change_min
            if short and allowed_short is not None and (i.idx, j.idx) not in allowed_short:
                continue
            outgoing[i.idx].append((j.idx, gap, short))

    duties: list[Duty] = []
    report_before = rules.operator.report_before_min
    post = rules.operator.post_flight_min

    def extend(
        seq: list[int],
        deps: list[int],
        arrs: list[int],
        shorts: list[tuple[int, int]],
        changes: list[tuple[int, int, int]],
        report: int,
        report_local: int,
    ) -> None:
        fdp = arrs[-1]
        duties.append(
            Duty(
                idx=len(duties),
                flights=tuple(seq),
                report_utc=report % DAY,
                length=fdp + post,
                fdp=fdp,
                start_station=flights[seq[0]].origin,
                end_station=flights[seq[-1]].destination,
                dep_offsets=tuple(deps),
                arr_offsets=tuple(arrs),
                short_connections=tuple(shorts),
                change_connections=tuple(changes),
            )
        )
        if len(seq) >= rules.max_sectors:
            return
        last = seq[-1]
        for j, gap, short in outgoing[last]:
            if j in seq:
                continue
            dep_j = arrs[-1] + gap
            arr_j = dep_j + flights[j].block
            if arr_j > rules.max_fdp(report_local, len(seq) + 1):
                continue
            new_shorts = [*shorts, (last, j)] if short else shorts
            new_changes = changes if short else [*changes, (last, j, gap)]
            extend([*seq, j], [*deps, dep_j], [*arrs, arr_j], new_shorts, new_changes, report, report_local)

    for f in flights:
        report = f.dep - report_before
        report_local = network.airports[f.origin].local_minute(report)
        first_arr = report_before + f.block
        if first_arr > rules.max_fdp(report_local, 1):
            continue
        extend([f.idx], [report_before], [first_arr], [], [], report, report_local)

    shorts = sorted({sc for d in duties for sc in d.short_connections})
    short_index = {sc: k for k, sc in enumerate(shorts)}
    words = (network.n_flights + 63) // 64
    masks = np.zeros((len(duties), words), dtype=np.uint64)
    for d in duties:
        for f_idx in d.flights:
            masks[d.idx, f_idx // 64] |= np.uint64(1) << np.uint64(f_idx % 64)
    flight_incidence = _ragged(duties, lambda d: d.flights)
    short_incidence = _ragged(duties, lambda d: tuple(short_index[sc] for sc in d.short_connections))
    return DutySet(
        duties=tuple(duties),
        short_connections=tuple(shorts),
        flight_incidence=flight_incidence,
        short_incidence=short_incidence,
        masks=masks,
    )


def _ragged(duties: list[Duty], key: object) -> NDArray[np.int32]:
    """Pack variable-length index lists into an (n, k_max) array padded with -1."""
    rows = [key(d) for d in duties]  # type: ignore[operator]
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), max(width, 1)), -1, dtype=np.int32)
    for n, r in enumerate(rows):
        if r:
            out[n, : len(r)] = r
    return out


@dataclass(frozen=True)
class PairingSpace:
    """All legal pairings, implicitly: one-duty pairings and arrays of legal two-duty pairs.

    ``pair_rest`` is the rest in minutes between the two duties and ``pair_nights`` the
    hotel nights it implies.
    """

    single: NDArray[np.int64]
    pair_first: NDArray[np.int64]
    pair_second: NDArray[np.int64]
    pair_rest: NDArray[np.int64]
    pair_nights: NDArray[np.int64]

    @property
    def size(self) -> int:
        """Number of legal pairings."""
        return int(self.single.size + self.pair_first.size)


def build_pairing_space(duty_set: DutySet, rules: PlanningRules) -> PairingSpace:
    """List every legal pairing of one or two duties that starts and ends at the base."""
    duties = duty_set.duties
    base = rules.base
    single = np.array([d.idx for d in duties if d.start_station == base and d.end_station == base], dtype=np.int64)
    firsts: list[NDArray[np.int64]] = []
    seconds: list[NDArray[np.int64]] = []
    rests: list[NDArray[np.int64]] = []
    if rules.max_duties >= 2:
        outstations = sorted({d.end_station for d in duties if d.start_station == base and d.end_station != base})
        start = np.array([d.report_utc for d in duties], dtype=np.int64)
        length = np.array([d.length for d in duties], dtype=np.int64)
        masks = duty_set.masks
        for station in outstations:
            a_idx = np.array(
                [d.idx for d in duties if d.start_station == base and d.end_station == station], dtype=np.int64
            )
            b_idx = np.array(
                [d.idx for d in duties if d.start_station == station and d.end_station == base], dtype=np.int64
            )
            if a_idx.size == 0 or b_idx.size == 0:
                continue
            end_a = start[a_idx] + length[a_idx]
            min_rest = np.maximum(length[a_idx], 600)
            base_gap = (start[b_idx][None, :] - end_a[:, None]) % DAY
            need = np.maximum(0, -(-(min_rest[:, None] - base_gap) // DAY))
            rest = base_gap + need * DAY
            ok = rest <= rules.max_rest_min
            overlap = np.zeros(ok.shape, dtype=bool)
            for w in range(masks.shape[1]):
                overlap |= (masks[a_idx, w][:, None] & masks[b_idx, w][None, :]) != 0
            ok &= ~overlap
            ai, bi = np.nonzero(ok)
            firsts.append(a_idx[ai])
            seconds.append(b_idx[bi])
            rests.append(rest[ai, bi])
    if firsts:
        pf, ps, pr = np.concatenate(firsts), np.concatenate(seconds), np.concatenate(rests)
    else:
        pf = ps = pr = np.zeros(0, dtype=np.int64)
    nights = 1 + pr // DAY
    return PairingSpace(single=single, pair_first=pf, pair_second=ps, pair_rest=pr, pair_nights=nights)
