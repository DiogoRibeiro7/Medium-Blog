"""Isolation risk under two fault models calibrated to the same number of cable faults.

Independent model: every cable segment near the island fails on its own, as a Poisson
process with a rate proportional to its length inside the zone.

Incident model: incidents happen at random points of the zone; an incident of radius R
severs every segment that passes within R of its centre. The incident rate is set so that
the expected number of segment faults per year equals the independent model's.

Only faults inside the zone are modelled, in both models alike.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import unary_union

from .cuts import build_graph, mainland_units, make_zone, unit_node
from .data import Network

HOURS_PER_YEAR = 8766.0


@dataclass
class Params:
    faults_per_km_year: float = 200 / 1.5e6     # ~200 faults a year on ~1.5 million km
    regional_share: float = 0.10                # share of incidents that are regional
    local_median_km: float = 0.3                # radius of an ordinary incident
    regional_median_km: float = 15.0            # radius of a regional incident
    radius_sigma: float = 0.7                   # log-sd of both radii
    radius_cap_km: float = 150.0
    wait_median_days: float = 21.0              # time before a repair starts (shared per incident)
    wait_sigma: float = 0.7
    fix_days: tuple = (3.0, 10.0)               # repair itself, per segment, uniform
    zone_km: float = 100.0
    samples: int = 200_000
    seed: int = 7

    def mean_outage_days(self) -> float:
        return self.wait_median_days * np.exp(self.wait_sigma ** 2 / 2) + sum(self.fix_days) / 2


class IslandModel:
    """The pieces of the network an island's isolation depends on."""

    def __init__(self, net: Network, u: int, zone_km: float, G: nx.MultiGraph | None = None):
        self.net, self.u = net, u
        G = build_graph(net) if G is None else G
        self.zone = make_zone(net, u, zone_km)
        self.region = unary_union([Point(p).buffer(zone_km, quad_segs=32) for p in self.zone.landings_xy])
        breakable = set(self.zone.edge_ids)
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        for a, b, d in G.edges(data=True):
            if d["kind"] == "land" or d["idx"] not in breakable:
                H.add_edge(a, b)
        comp = {}
        for k, cc in enumerate(nx.connected_components(H)):
            for n in cc:
                comp[n] = k
        self.src = comp[unit_node(u)]
        self.sinks = {comp[unit_node(m)] for m in mainland_units(net) if unit_node(m) in comp}
        # breakable edges that join two different components, as (local index, comp a, comp b)
        self.edges = []
        self.lines = []
        self.length_in_zone = []
        for k, e in enumerate(self.zone.edge_ids):
            a, b = comp[net.edges[e].u], comp[net.edges[e].v]
            if a == b:
                continue
            self.edges.append((e, a, b))
            line = self.zone.lines[k]
            self.lines.append(line)
            self.length_in_zone.append(float(line.intersection(self.region).length))
        self.length_in_zone = np.array(self.length_in_zone)
        self.ncomp = len(set(comp.values()))

    def isolated(self, broken: frozenset) -> bool:
        """True when, with the given local edge indices broken, no path reaches a continent."""
        if self.src in self.sinks:
            return False
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        for i, (_, a, b) in enumerate(self.edges):
            if i in broken:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        root = find(self.src)
        return all(find(s) != root for s in self.sinks)

    def minimal_cuts(self, max_size: int = 4) -> list:
        """All minimal sets of at most max_size breakable segments whose loss isolates the island."""
        m = len(self.edges)
        cuts = []
        for size in range(1, min(max_size, m) + 1):
            for combo in itertools.combinations(range(m), size):
                s = frozenset(combo)
                if any(c <= s for c in cuts):
                    continue
                if self.isolated(s):
                    cuts.append(s)
        return cuts


CUT_SEARCH_MAX = 4


def independent_risk(model: IslandModel, p: Params, cuts=None) -> dict:
    """Unavailability from coincident independent faults (rare-event sum over minimal cuts).

    Minimal cuts are enumerated up to CUT_SEARCH_MAX segments. When none exists, the
    result is reported as an upper bound: every cut has at least CUT_SEARCH_MAX + 1
    segments, and there are at most C(m, CUT_SEARCH_MAX + 1) of them.
    """
    from math import comb
    cuts = model.minimal_cuts(CUT_SEARCH_MAX) if cuts is None else cuts
    m_years = p.mean_outage_days() / 365.25
    nu = p.faults_per_km_year * model.length_in_zone
    q = nu * m_years / (1 + nu * m_years)
    unavail = sum(float(np.prod(q[list(c)])) for c in cuts)
    onset = sum(sum(nu[e] * float(np.prod(q[[f for f in c if f != e]])) for e in c) for c in cuts)
    bound = None
    if not cuts and len(q):
        s = CUT_SEARCH_MAX + 1
        bound = comb(len(q), s) * float(np.max(q)) ** s * HOURS_PER_YEAR if len(q) >= s else 0.0
    return {"hours_per_year": unavail * HOURS_PER_YEAR, "episodes_per_year": onset,
            "faults_per_year": float(nu.sum()), "min_cut_size": min((len(c) for c in cuts), default=0),
            "n_cuts": len(cuts), "upper_bound_hours_if_no_small_cut": bound}


def incident_risk(model: IslandModel, p: Params, cuts=None) -> dict:
    """Unavailability from single incidents that sever enough segments by themselves, plus
    the coincidence term of the independent model for cuts of two or more segments."""
    rng = np.random.default_rng(p.seed)
    minx, miny, maxx, maxy = model.region.bounds
    pts = []
    need = p.samples
    while need > 0:
        x = rng.uniform(minx, maxx, need * 2)
        y = rng.uniform(miny, maxy, need * 2)
        inside = shapely.contains_xy(model.region, x, y)
        pts.append(np.column_stack([x[inside], y[inside]])[:need])
        need -= int(inside.sum())
    pts = np.vstack(pts)[:p.samples]
    n = len(pts)
    regional = rng.random(n) < p.regional_share
    med = np.where(regional, p.regional_median_km, p.local_median_km)
    R = np.minimum(med * np.exp(p.radius_sigma * rng.standard_normal(n)), p.radius_cap_km)
    P = shapely.points(pts[:, 0], pts[:, 1])
    hits = np.zeros((n, len(model.lines)), dtype=bool)
    for j, line in enumerate(model.lines):
        hits[:, j] = shapely.distance(P, line) <= R
    per_incident = hits.sum(axis=1)
    total_faults = p.faults_per_km_year * float(model.length_in_zone.sum())
    mean_hits = per_incident.mean()
    rate = total_faults / mean_hits if mean_hits > 0 else 0.0     # incidents per year
    keys = {}
    iso = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(per_incident):
        key = frozenset(np.flatnonzero(hits[i]).tolist())
        if key not in keys:
            keys[key] = model.isolated(key)
        iso[i] = keys[key]
    p_iso = iso.mean()
    # duration of isolation for isolating incidents: shared wait, individual repairs
    durations = []
    for i in np.flatnonzero(iso):
        broken = sorted(np.flatnonzero(hits[i]).tolist())
        wait = p.wait_median_days * np.exp(p.wait_sigma * rng.standard_normal())
        done = wait + rng.uniform(*p.fix_days, len(broken))
        order = np.argsort(done)
        down = set(broken)
        for k in order:
            down.discard(broken[k])
            if not model.isolated(frozenset(down)):
                durations.append(done[k])
                break
    mean_days = float(np.mean(durations)) if durations else 0.0
    single_hours = rate * p_iso * mean_days * 24.0
    # second-order term: separate events overlapping in time. A one-segment cut is already
    # covered by the incidents themselves, so only cuts of two or more segments are added.
    cuts = model.minimal_cuts(CUT_SEARCH_MAX) if cuts is None else cuts
    indep = independent_risk(model, p, [c for c in cuts if len(c) >= 2])
    return {"hours_per_year": single_hours + indep["hours_per_year"],
            "single_incident_hours_per_year": single_hours,
            "episodes_per_year": rate * p_iso + indep["episodes_per_year"],
            "incidents_per_year": rate, "p_isolating": float(p_iso),
            "mean_isolation_days": mean_days, "mean_segments_per_incident": float(mean_hits),
            "faults_per_year": total_faults, "n_isolating_samples": int(iso.sum())}
