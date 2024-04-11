import unittest
from dep_graph_generator import DependencyGraphGenerator
from lp_converter import LPConverter
from lp_analyzer import LPAnalyzer
from topology import NetTopology


class TestGraphAnalysis(unittest.TestCase):
    def test_net_lat_sen_blocking(self) -> None:
        """
        Tests the network latency sensitivity metric from
        the dependency graph generated from the goal file of a
        simple blocking MPI program.
        1 (l2) -> 0 (l1) -> 2 (l3) : Rank 0
        [1000]      |[1500]  [2000]
                    | 
                    | L + (4 - 1) * 6
                    |
                    V
        4 (l2) -> 3 (l1) -> 5 (l3) : Rank 1
        [3000]      [1500]  [2000]
        """
        goal_path = "mpi-dep-graph/test/data/blocking.goal"
        generator = DependencyGraphGenerator(goal_path)
        dep_graph = generator.generate()
        topology = NetTopology.default_topology(dep_graph.num_ranks)
        lp_model = LPConverter(dep_graph, topology).convert_to_lp()
        analyzer = LPAnalyzer()
        net_lat_sen = analyzer.get_net_lat_sensitivity(lp_model)
        print(net_lat_sen.critical_latencies)
        self.assertEqual(len(net_lat_sen.critical_latencies), 1)
        self.assertEqual(tuple(net_lat_sen.critical_latencies[0]), (0, 1))

    def test_net_lat_sen_non_blocking(self) -> None:
        """
        Tests the network latency sensitivity metric from
        the dependency graph generated from the goal file of a
        simple blocking MPI program.
            ---> 2 (l3) ------
           /     [2000]      |
          /                  V
        1 (l2) -> 0 (l1) -> 3 (l4) : Rank 0
        [1000]      |[1500]  [4000]
                    | 
                    | L + (4 - 1) * 6
                    |
                    V
        5 (l2) -> 4 (l1) -> 6 (l3)  : Rank 1
        [3000]      [1500]  [2000]
        """
        goal_path = "mpi-dep-graph/test/data/non_blocking.goal"
        generator = DependencyGraphGenerator(goal_path)
        dep_graph = generator.generate()
        topology = NetTopology.default_topology(dep_graph.num_ranks)
        lp_model = LPConverter(dep_graph, topology).convert_to_lp()
        analyzer = LPAnalyzer()
        net_lat_sen = analyzer.get_net_lat_sensitivity(lp_model)
        self.assertEqual(len(net_lat_sen.critical_latencies), 2)
        self.assertEqual(tuple(net_lat_sen.critical_latencies[0]), (0, 0))
        self.assertEqual(tuple(net_lat_sen.critical_latencies[1]), (482, 1))

if __name__ == "__main__":
    unittest.main()