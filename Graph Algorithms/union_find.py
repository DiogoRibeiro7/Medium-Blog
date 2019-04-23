class DisjointSet(object):
  def __init__(self, n):
    """
    Initializes a disjoint set structure consisting of n disjoint sets.
    """
    self.parent = list(range(n))
  def find(self, x):
    """Returns the representative element of the set x belongs to."""
    if self.parent[x] != x:
      self.parent[x] = self.find(self.parent[x])
    return self.parent[x]
  def union(self, x, y):
    """Joins the sets containing x and y."""
    self.parent[self.find(x)] = self.find(y)
    
    
    
    
import unittest
class DisjointSetTest(unittest.TestCase):
  def test_initialized_state(self):
    d = DisjointSet(3)
    self.assertEqual(d.find(0), 0)
    self.assertEqual(d.find(1), 1)
    self.assertEqual(d.find(2), 2)
  def test_basic_union(self):
    d = DisjointSet(3)
    d.union(0, 1)
    self.assertEqual(d.find(0), d.find(1))
    self.assertNotEqual(d.find(1), d.find(2))
  def test_basic_union_idempotent(self):
    d = DisjointSet(2)
    d.union(0, 1)
    d.union(0, 1)
    self.assertEqual(d.find(0), d.find(1))
  def test_union_all(self):
    d = DisjointSet(100)
    for i in range(1, 100):
      d.union(i - 1, i)
    for i in range(1, 100):
      self.assertEqual(d.find(0), d.find(i))
