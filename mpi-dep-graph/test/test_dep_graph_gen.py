import os
import unittest
from dep_graph_generator import DependencyGraphGenerator
from dep_graph import VertexType

"""
Integration tests for the dependency graph generator.
"""

class TestDepGraphGenerator(unittest.TestCase):
    def test_generate_dep_graph_blocking(self):
        """
        Tests the dependency graph generator with the goal
        file from a simple blocking MPI program.
        The graph should look like this:
        1 (l2) -> 0 (l1) -> 2 (l3) : Rank 0
                    |
                    |
                    V
        4 (l2) -> 3 (l1) -> 5 (l3) : Rank 1
        """
        goal_path = "mpi-dep-graph/test/data/blocking.goal"
        generator = DependencyGraphGenerator(goal_path)
        dep_graph = generator.generate()

        # Checks the number of vertices and edges
        self.assertEqual(dep_graph.num_vertices(), 6)
        self.assertEqual(dep_graph.num_edges(), 5)

        # Checks the vertex types
        # Rank 0
        self.assertEqual(dep_graph.graph.vs[0]["type"], VertexType.SEND)
        self.assertEqual(dep_graph.graph.vs[1]["type"], VertexType.CALC)
        self.assertEqual(dep_graph.graph.vs[2]["type"], VertexType.CALC)
        # Rank 1
        self.assertEqual(dep_graph.graph.vs[3]["type"], VertexType.RECV)
        self.assertEqual(dep_graph.graph.vs[4]["type"], VertexType.CALC)
        self.assertEqual(dep_graph.graph.vs[5]["type"], VertexType.CALC)

        # Checks the edges
        self.assertEqual(dep_graph.get_successors(dep_graph.start_v), [1, 4])
        self.assertEqual(dep_graph.get_successors(0), [2, 3])
        self.assertEqual(dep_graph.get_successors(1), [0])
        self.assertEqual(dep_graph.get_successors(2), [dep_graph.end_v])
        self.assertEqual(dep_graph.get_successors(3), [5])
        self.assertEqual(dep_graph.get_successors(4), [3])
        self.assertEqual(dep_graph.get_successors(5), [dep_graph.end_v])


    def test_generate_dep_graph_nonblocking(self):
        """
        Tests the dependency graph generator with the goal
        file from a simple non-blocking MPI program.
        The graph should look like this:
              --> 2 (l3) -----
             /               |
            /                V
        1 (l2) -> 0 (l1) -> 3 (l4) : Rank 0
                    |
                    |
                    V
        5 (l2) -> 4 (l1) -> 6 (l3) : Rank 1
        """
        goal_path = "mpi-dep-graph/test/data/non_blocking.goal"
        generator = DependencyGraphGenerator(goal_path)
        dep_graph = generator.generate()

        # Checks the number of vertices and edges
        self.assertEqual(dep_graph.num_vertices(), 7)
        self.assertEqual(dep_graph.num_edges(), 7)

        # Checks the vertex types
        # Rank 0
        self.assertEqual(dep_graph.graph.vs[0]["type"], VertexType.SEND)
        self.assertEqual(dep_graph.graph.vs[1]["type"], VertexType.CALC)
        self.assertEqual(dep_graph.graph.vs[2]["type"], VertexType.CALC)
        self.assertEqual(dep_graph.graph.vs[3]["type"], VertexType.CALC)
        # Rank 1
        self.assertEqual(dep_graph.graph.vs[4]["type"], VertexType.RECV)
        self.assertEqual(dep_graph.graph.vs[5]["type"], VertexType.CALC)
        self.assertEqual(dep_graph.graph.vs[6]["type"], VertexType.CALC)

        # Checks the edges
        self.assertEqual(dep_graph.get_successors(dep_graph.start_v), [1, 5])
        self.assertEqual(dep_graph.get_successors(0), [3, 4])
        self.assertEqual(dep_graph.get_successors(1), [0, 2])
        self.assertEqual(dep_graph.get_successors(2), [3])
        self.assertEqual(dep_graph.get_successors(3), [dep_graph.end_v])
        self.assertEqual(dep_graph.get_successors(4), [6])
        self.assertEqual(dep_graph.get_successors(5), [4])
        self.assertEqual(dep_graph.get_successors(6), [dep_graph.end_v])



if __name__ == "__main__":
    unittest.main()