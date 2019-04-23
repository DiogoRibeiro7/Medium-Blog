from collections import namedtuple, defaultdict
from Queue import PriorityQueue
Edge = namedtuple('Edge', ['target', 'weight'])


def dijkstra(graph, source, target):
  """
  Given a directed graph (format described below), and source and target
  vertices, returns a shortest path as a list of vertices going from source
  to target, along with the distance of the shortest path, or None if no such
  path exists. Returned path will not include the source vertex in it.
  Assumes non-negative weights.
  The graph parameter is expected to be a dictionary mapping each vertex to a
  list of Edge named tuples indicating the vertex's outgoing edges. For
  example if vertex v has outgoing edges to u and w with weights 10 and 20
  respectively, we have graph[v] = [Edge(u, 10), Edge(w, 20)].
  """
  q = PriorityQueue()
  q.put((0, source))
  # previous_vertex[v] holds the immediate vertex before v in the shortest
  # path from source to v. This dictionary also acts as our "visited" set
  # since we set previous_vertex[v] as soon as the vertex enters our queue.
  previous_vertex = {source: source}
  # Arguably not the best way to represent infinity but it works for the sake
  # of learning the algorithm.
  shortest_distance = defaultdict(lambda: float('inf'))
  shortest_distance[source] = 0
  while not q.empty():
    (distance, v) = q.get()
    if v == target:
      return (distance, _construct_path(previous_vertex, source, target))
    for edge in graph[v]:
      alt_distance = edge.weight + distance
      if alt_distance < shortest_distance[edge.target]:
        shortest_distance[edge.target] = alt_distance
        q.put((alt_distance, edge.target))
        previous_vertex[edge.target] = v
  return None


def _construct_path(previous_vertex, source, target):
  if source == target:
    return []
  return _construct_path(previous_vertex, source,
               previous_vertex[target]) + [target]



import unittest


class DijkstraTest(unittest.TestCase):
  def test_single_vertex(self):
    graph = {0: []}
    self.assertEqual(dijkstra(graph, 0, 0), (0, []))
  def test_two_vertices_no_path(self):
    graph = {
      0: [],
      1: [],
    }
    self.assertEqual(dijkstra(graph, 0, 1), None)
  def test_two_vertices_with_path(self):
    graph = {
      0: [Edge(target=1, weight=10)],
      1: [],
    }
    self.assertEqual(dijkstra(graph, 0, 1), (10, [1]))
  def test_cycle_3(self):
    graph = {
      0: [Edge(target=1, weight=10), Edge(target=2, weight=30)],
      1: [Edge(target=0, weight=10), Edge(target=2, weight=10)],
      2: [Edge(target=0, weight=30), Edge(target=1, weight=30)],
    }
    self.assertEqual(dijkstra(graph, 0, 2), (20, [1, 2]))
  def test_clrs_example(self):
    graph = {
      's': [
        Edge(target='t', weight=3),
        Edge(target='y', weight=5),
      ],
      't': [
        Edge(target='x', weight=6),
        Edge(target='y', weight=2),
      ],
      'y': [
        Edge(target='t', weight=1),
        Edge(target='z', weight=6),
      ],
      'x': [
        Edge(target='z', weight=2),
      ],
      'z': [
        Edge(target='x', weight=7),
        Edge(target='s', weight=3),
      ],
    }
    distance, path = dijkstra(graph, 's', 'z')
    self.assertEqual(distance, 11)
    self.assertIn(path, [
      ['y', 'z'],
      ['t', 'y', 'x', 'z'],
    ])
    distance, path = dijkstra(graph, 's', 'x')
    self.assertEqual(distance, 9)
    self.assertIn(path, [
      ['t', 'x'],
      ['y', 'x'],
    ])
