import unittest
from dep_graph_generator import DependencyGraphGenerator
from lp_converter import LPConverter
from lp_analyzer import LPAnalyzer
from topology import NetTopology


class TestGraphAnalysis(unittest.TestCase):
    def _run_sensitivity(self, goal_path: str):
        generator = DependencyGraphGenerator(goal_path)
        dep_graph = generator.generate(is_loggps=True)
        topology = NetTopology.default_topology(dep_graph.num_ranks)
        lp_model = LPConverter(dep_graph).convert_to_lp(
            verbose=False,
            topology=topology,
            G=0.018,
        )
        analyzer = LPAnalyzer()
        return analyzer.get_net_lat_sensitivity(
            lp_model,
            L_lb=3000,
            L_ub=5000,
            step=1000,
            verbose=False,
        )

    def test_net_lat_sen_blocking(self) -> None:
        net_lat_sen = self._run_sensitivity("mpi-dep-graph/test/data/blocking.goal")
        self.assertGreaterEqual(len(net_lat_sen.runtime), 3)
        ls = [point[0] for point in net_lat_sen.runtime]
        runtimes = [point[1] for point in net_lat_sen.runtime]
        self.assertEqual(ls, sorted(ls))
        self.assertTrue(all(runtimes[i] <= runtimes[i + 1] for i in range(len(runtimes) - 1)))

    def test_net_lat_sen_non_blocking(self) -> None:
        net_lat_sen = self._run_sensitivity("mpi-dep-graph/test/data/non_blocking.goal")
        self.assertGreaterEqual(len(net_lat_sen.runtime), 3)
        ls = [point[0] for point in net_lat_sen.runtime]
        runtimes = [point[1] for point in net_lat_sen.runtime]
        self.assertEqual(ls, sorted(ls))
        self.assertTrue(all(runtimes[i] <= runtimes[i + 1] for i in range(len(runtimes) - 1)))

if __name__ == "__main__":
    unittest.main()
