from collections import deque


def bfs(graph, source, target):
  """
  Given a directed graph (format described below), and source and target
  vertices, returns a shortest unweighted path as a list of vertices going
  from source to target, or None if no such path exists. Returned path will
  not include the source vertex in it.
  The graph parameter is expected to be a dictionary mapping each vertex to a
  list of vertices indicating outgoing edges. For example if vertex v has
  outgoing edges to u and w we have graph[v] = [u, w].
  """
  q = deque([source])
  # previous_vertex[v] holds the immediate vertex before v in the shortest
  # path from source to v. This dictionary also acts as our "visited" set
  # since we set previous_vertex[v] as soon as the vertex enters our queue.
  previous_vertex = {source: source}
  while q:
    v = q.popleft()
    if v == target:
      return _construct_path(previous_vertex, source, target)
    for w in graph[v]:
      if w not in previous_vertex:
        previous_vertex[w] = v
        q.append(w)
  return None


def _construct_path(previous_vertex, source, target):
  if source == target:
    return []
  return _construct_path(previous_vertex, source,
               previous_vertex[target]) + [target]


#And the accompanied unit test:
import unittest

class BFSTest(unittest.TestCase):
  def test_single_vertex(self):
    graph = {0: []}
    self.assertEqual(bfs(graph, 0, 0), [])
  def test_single_vertex_with_loop(self):
    graph = {0: [0]}
    self.assertEqual(bfs(graph, 0, 0), [])
  def test_two_vertices_no_path(self):
    graph = {
      0: [],
      1: [],
    }
    self.assertEqual(bfs(graph, 0, 1), None)
  def test_two_vertices_with_simple_path(self):
    graph = {
      0: [1],
      1: [],
    }
    self.assertEqual(bfs(graph, 0, 1), [1])
  def test_complete_graph(self):
    def _complete_graph(n):
      return {v: list(set(range(n)) - {v}) for v in range(n)}
    for n in range(2, 10):
      graph = _complete_graph(n)
      for v in range(n):
        for w in range(n):
          self.assertEqual(bfs(graph, v, w),
                   [] if v == w else [w])
  def test_cycle_5(self):
    graph = {
      0: [4, 1],
      1: [0, 2],
      2: [1, 3],
      3: [2, 4],
      4: [3, 0],
    }
    self.assertEqual(bfs(graph, 0, 2), [1, 2])
    self.assertEqual(bfs(graph, 0, 3), [4, 3])
