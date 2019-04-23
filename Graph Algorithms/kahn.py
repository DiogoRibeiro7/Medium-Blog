from collections import deque, namedtuple
Vertex = namedtuple('Vertex', ['name', 'incoming', 'outgoing'])


def build_doubly_linked_graph(graph):
  """
  Given a graph with only outgoing edges, build a graph with incoming and
  outgoing edges. The returned graph will be a dictionary mapping vertex to a
  Vertex namedtuple with sets of incoming and outgoing vertices.
  """
  g = {v:Vertex(name=v, incoming=set(), outgoing=set(o))
     for v, o in graph.items()}
  for v in g.values():
    for w in v.outgoing:
      if w in g:
        g[w].incoming.add(v.name)
      else:
        g[w] = Vertex(name=w, incoming={v}, outgoing=set())
  return g


def kahn_top_sort(graph):
  """
  Given an acyclic directed graph (format described below), returns a
  dictionary mapping vertex to sequence such that sorting by the sequence
  component will result in a topological sort of the input graph. Output is
  undefined if input is a not a valid DAG.
  The graph parameter is expected to be a dictionary mapping each vertex to a
  list of vertices indicating outgoing edges. For example if vertex v has
  outgoing edges to u and w we have graph[v] = [u, w].
  """
  g = build_doubly_linked_graph(graph)
  # sequence[v] < sequence[w] implies v should be before w in the topological
  # sort.
  q = deque(v.name for v in g.values() if not v.incoming)
  sequence = {v: 0 for v in q}
  while q:
    v = q.popleft()
    for w in g[v].outgoing:
      g[w].incoming.remove(v)
      if not g[w].incoming:
        sequence[w] = sequence[v] + 1
        q.append(w)
  return sequence


#And the accompanied unit test:
import unittest

class KahnTopSortTest(unittest.TestCase):
  def test_single_vertex(self):
    graph = {
      0: [],
    }
    self.assertEqual(kahn_top_sort(graph), {
      0: 0,
    })
  def test_total_order_2(self):
    graph = {
      0: [1],
      1: [],
    }
    self.assertEqual(kahn_top_sort(graph), {
      0: 0,
      1: 1,
    })
  def test_total_order_3(self):
    graph = {
      0: [1],
      1: [2],
      2: [],
    }
    self.assertEqual(kahn_top_sort(graph), {
      0: 0,
      1: 1,
      2: 2,
    })
  def test_two_independent_total_orders(self):
    # 0 -> 1 -> 2
    # 3 -> 4 -> 5
    graph = {
      0: [1],
      1: [2],
      2: [],
      3: [4],
      4: [5],
      5: [],
    }
    self.assertEqual(kahn_top_sort(graph), {
      0: 0,
      3: 0,
      1: 1,
      4: 1,
      2: 2,
      5: 2,
    })
  def test_simple_dag_1(self):
    # 0 -> 1 -> 2
    #   \ /
    #  3
    graph = {
      0: [1, 3],
      1: [2],
      2: [],
      3: [1],
    }
    self.assertEqual(kahn_top_sort(graph), {
      0: 0,
      3: 1,
      1: 2,
      2: 3,
    })
