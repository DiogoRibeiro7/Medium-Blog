def dfs(graph, source):
  """
  Given a directed graph (format described below), and a source vertex,
  returns a set of vertices reachable from source.
  The graph parameter is expected to be a dictionary mapping each vertex to a
  list of vertices indicating outgoing edges. For example if vertex v has
  outgoing edges to u and w we have graph[v] = [u, w].
  """
  visited = set()
  def _recurse(v):
    if v in visited:
      return
    visited.add(v)
    for w in graph[v]:
      _recurse(w)
  _recurse(source)
  return visited
  
  
#And the accompanied unit test:
import unittest

class DFSTest(unittest.TestCase):
  def test_single_vertex(self):
    graph = {0: []}
    self.assertEqual(dfs(graph, 0), {0})
  def test_single_vertex_with_loop(self):
    graph = {0: [0]}
    self.assertEqual(dfs(graph, 0), {0})
  def test_two_vertices_no_path(self):
    graph = {
      0: [],
      1: [],
    }
    self.assertEqual(dfs(graph, 0), {0})
    self.assertEqual(dfs(graph, 1), {1})
  def test_two_vertices_with_simple_path(self):
    graph = {
      0: [1],
      1: [],
    }
    self.assertEqual(dfs(graph, 0), {0, 1})
    self.assertEqual(dfs(graph, 1), {1})
  def test_complete_graph(self):
    def _complete_graph(n):
      return {v: list(set(range(n)) - {v}) for v in range(n)}
    for n in range(2, 10):
      graph = _complete_graph(n)
      for v in range(n):
        self.assertEqual(dfs(graph, v), set(range(n)))
  def test_cycle_5(self):
    graph = {
      0: [1],
      1: [2],
      2: [3],
      3: [4],
      4: [0],
    }
    for v in range(5):
      self.assertEqual(dfs(graph, v), {0, 1, 2, 3, 4})
