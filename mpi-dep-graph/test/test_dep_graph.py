import unittest
from dep_graph import DependencyGraph, VertexType

"""
Unit tests for the dependency graph.
"""


class TestDependencyGraph(unittest.TestCase):
    def test_add_vertex(self):
        """
        Tests the add_vertex() method.
        """
        g = DependencyGraph(1)
        g.add_vertex(VertexType.SEND, 0, 1, 4, 1)
        g.add_vertex(VertexType.CALC, 0, 2, 10000)

        self.assertEqual(g.num_vertices(), 2)
        self.assertEqual(g.num_edges(), 0)
    
    def test_add_edge_requires(self):
        """
        Tests the add_edge() method when the
        edge is a requires edge.
        """
        g = DependencyGraph(2)
        v1 = g.add_vertex(VertexType.SEND, 0, 1, 4, 1)
        v2 = g.add_vertex(VertexType.RECV, 1, 1, 4, 1)
        g.add_edge(0, 1, 1, 1)

        v3 = g.add_vertex(VertexType.CALC, 0, 2, 10000)
        g.add_edge(0, 1, 0, 2)
        g.finalize()
        self.assertEqual(g.num_vertices(), 3)
        self.assertEqual(g.num_edges(), 2)
        self.assertEqual(g.get_successors(v1), [v2, v3])

    def test_add_edge_irequires(self):
        """
        Tests the add_edge() method when the
        edge is an irequires edge.
        """
        g = DependencyGraph(2)
        v1 = g.add_vertex(VertexType.SEND, 0, 1, 4, 1)
        v2 = g.add_vertex(VertexType.CALC, 0, 2, 10000)
        g.add_edge_by_global_index(v2, v1)
        v3 = g.add_vertex(VertexType.CALC, 0, 3, 2000)
        g.add_edge(0, 1, 0, 3, True)
        g.finalize()
        self.assertEqual(g.num_vertices(), 3)
        self.assertEqual(g.num_edges(), 2)
        self.assertEqual(g.get_successors(v2), [v1, v3])

if __name__ == "__main__":
    unittest.main()