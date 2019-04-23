from collections import namedtuple
from union_find import DisjointSet


# Putting weight as the first element means Edges will sort by weight first,
# then source and target (lexicographically).
Edge = namedtuple('Edge', ['weight', 'source', 'target'])


def kruskal_mst(n, edges):
  """
  Given a positive integer n (number of vertices) and a collection of Edge
  namedtuple objects representing the undirected edges of a graph, returns a
  list of edges forming a minimal spanning tree of the graph. Assumes the
  vertices are numbers in the range 0 to n - 1.  Also assumes input is a
  valid connected undirected graph and that for two vertices v and w only one
  of (v, w) or (w, v) is an edge in the input. Output is undefined if these
  assumptions are not satisfied.
  """
  d = DisjointSet(n)
  mst_tree = []
  for edge in sorted(edges):
    if d.find(edge.source) != d.find(edge.target):
      mst_tree.append(edge)
      if len(mst_tree) == n - 1:
        break
      d.union(edge.source, edge.target)
  return mst_tree



import unittest

class KruskalMSPTest(unittest.TestCase):
  def test_single_vertex_graph(self):
    self.assertEqual(kruskal_mst(1, []), [])
  def test_single_edge_graph(self):
    edges = [Edge(source=0, target=1, weight=10)]
    self.assertEqual(kruskal_mst(2, edges), edges)
  def test_cycle_5(self):
    edges = [
      Edge(source=0, target=1, weight=50),
      Edge(source=1, target=2, weight=30),
      Edge(source=2, target=3, weight=60),
      Edge(source=3, target=4, weight=20),
      Edge(source=4, target=0, weight=10),
    ]
    # Everything except the heaviest edge. Output sorted by weight.
    self.assertEqual(kruskal_mst(5, edges), [
      Edge(source=4, target=0, weight=10),
      Edge(source=3, target=4, weight=20),
      Edge(source=1, target=2, weight=30),
      Edge(source=0, target=1, weight=50),
    ])
  def test_complete_graph_4(self):
    edges = [
      Edge(source=0, target=1, weight=10),
      Edge(source=0, target=2, weight=30),
      Edge(source=0, target=3, weight=40),
      Edge(source=1, target=2, weight=20),
      Edge(source=1, target=3, weight=50),
      Edge(source=2, target=3, weight=60),
    ]
    self.assertEqual(kruskal_mst(4, edges), [
      Edge(source=0, target=1, weight=10),
      Edge(source=1, target=2, weight=20),
      Edge(source=0, target=3, weight=40),
    ])
