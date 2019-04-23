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
    
