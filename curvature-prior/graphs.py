"""The graphs, their shortest-path metrics, and how tree-like each one is.

Two real hierarchies come from WordNet (the mammal and the food noun subtrees), with the
hypernym links taken as undirected edges. Two controls have known shape: a complete tree
with branching 3 and depth 6, which is exactly a tree, and a 33 x 33 grid, which is flat.
A dial adds random cross-links to the mammal tree to make it gradually less tree-like.

Tree-likeness is measured by Gromov's four-point delta: for four points, form the three
sums d(x,y)+d(z,w), d(x,z)+d(y,w), d(x,w)+d(y,z); delta is half the gap between the two
largest. A metric tree has delta = 0 for every quadruple. The relative version divides
2*delta by the diameter. Here delta is the maximum over a fixed random sample of
quadruples, so it is a lower bound on the exact value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

import wordnet as W


@dataclass
class Graph:
    name: str
    n: int
    edges: np.ndarray  # (m, 2), i < j
    dist: np.ndarray = None  # (n, n) shortest-path distances

    def __post_init__(self):
        if self.dist is None:
            self.dist = distances(self.n, self.edges)

    @property
    def m(self) -> int:
        return len(self.edges)

    def neighbours(self) -> list[np.ndarray]:
        nb = [[] for _ in range(self.n)]
        for i, j in self.edges:
            nb[i].append(j)
            nb[j].append(i)
        return [np.array(sorted(x)) for x in nb]


def distances(n: int, edges: np.ndarray) -> np.ndarray:
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    ncomp, _ = connected_components(a, directed=False)
    if ncomp != 1:
        raise ValueError("graph is not connected")
    return shortest_path(a, method="D", directed=False, unweighted=True)


def wordnet_graph(lemma: str) -> Graph:
    path = W.fetch()
    syn = W.read_nouns(path)
    nodes, edges, _ = W.subtree(syn, W.first_sense(path, lemma))
    return Graph(f"WordNet {lemma}s" if lemma != "food" else "WordNet food", len(nodes), np.array(edges))


def complete_tree(branching: int = 3, depth: int = 6) -> Graph:
    edges, nxt, frontier = [], 1, [0]
    for _ in range(depth):
        new = []
        for p in frontier:
            for _ in range(branching):
                edges.append((p, nxt))
                new.append(nxt)
                nxt += 1
        frontier = new
    return Graph(f"tree (branching {branching}, depth {depth})", nxt, np.array(edges))


def grid(side: int = 33) -> Graph:
    idx = lambda r, c: r * side + c
    edges = [(idx(r, c), idx(r, c + 1)) for r in range(side) for c in range(side - 1)]
    edges += [(idx(r, c), idx(r + 1, c)) for r in range(side - 1) for c in range(side)]
    return Graph(f"grid ({side} x {side})", side * side, np.array(sorted(edges)))


def with_cross_links(base: Graph, fraction: float, seed: int) -> Graph:
    """Add round(fraction * n) random edges between nodes not already joined."""
    rng = np.random.default_rng(seed)
    k = int(round(fraction * base.n))
    have = {tuple(e) for e in base.edges.tolist()}
    added = []
    while len(added) < k:
        i, j = sorted(rng.choice(base.n, 2, replace=False).tolist())
        if (i, j) not in have:
            have.add((i, j))
            added.append((i, j))
    edges = np.array(sorted(have))
    return Graph(f"{base.name} + {k} cross-links", base.n, edges)


def gromov_delta(dist: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    q = rng.integers(0, n, size=(samples, 4))
    x, y, z, w = q.T
    s = np.sort(np.stack([dist[x, y] + dist[z, w], dist[x, z] + dist[y, w], dist[x, w] + dist[y, z]]), axis=0)
    d = (s[2] - s[1]) / 2.0
    diam = float(dist.max())
    return {"delta": float(d.max()), "delta_mean": float(d.mean()), "diameter": diam,
            "relative": float(2 * d.max() / diam), "samples": samples}


def closure(base: Graph, root: int) -> Graph:
    """The transitive closure of a rooted tree-like hierarchy: every node joined to every one
    of its ancestors (the object Nickel and Kiela, 2017, embed)."""
    parent = {}
    order = np.argsort(base.dist[root], kind="stable")
    for v in order[1:]:
        nb = [u for u in base.neighbours()[v] if base.dist[root, u] == base.dist[root, v] - 1]
        parent[int(v)] = [int(u) for u in nb]
    anc = {int(root): set()}
    for v in order[1:]:
        s = set()
        for p in parent[int(v)]:
            s |= anc[p] | {p}
        anc[int(v)] = s
    edges = sorted({(min(v, a), max(v, a)) for v, s in anc.items() for a in s})
    return Graph(f"{base.name}, transitive closure", base.n, np.array(edges))
