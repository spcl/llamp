import unittest
import tempfile

from dep_graph_generator import DependencyGraphGenerator
from trace_compression import (
    EventSignature,
    RepeatToken,
    compress_sequence_iter_template,
    compress_sequence_lossless,
)


class TestTraceCompression(unittest.TestCase):
    def test_event_signature_hash_and_equality(self):
        s1 = EventSignature(
            kind="send",
            size_or_cost=1024,
            peer_rank=1,
            tag=7,
            group_id=None,
            stream_id=3,
            channel_id=0,
            attrs=(("cpu", "3"), ("nic", "0")),
        )
        s2 = EventSignature(
            kind="send",
            size_or_cost=1024,
            peer_rank=1,
            tag=7,
            group_id=None,
            stream_id=3,
            channel_id=0,
            attrs=(("cpu", "3"), ("nic", "0")),
        )
        self.assertEqual(s1, s2)
        self.assertEqual(s1.stable_hash(), s2.stable_hash())
        self.assertEqual(s1.to_compact_str(), s2.to_compact_str())

    def test_lossless_repeat_detection_and_indexer_roundtrip(self):
        seq = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5]
        program = compress_sequence_lossless(seq, max_window=8)
        self.assertEqual(program.materialize(), seq)
        self.assertTrue(any(isinstance(t, RepeatToken) for t in program.tokens))
        for ordinal in range(len(seq)):
            loc = program.indexer.ordinal_to_location(ordinal)
            self.assertEqual(program.indexer.location_to_ordinal(loc), ordinal)

    def test_iter_template_materialization(self):
        seq = [99, 98] + [1, 2, 3] * 6 + [77]
        program = compress_sequence_iter_template(seq)
        self.assertEqual(program.materialize(), seq)

    def test_lossless_mode_matches_baseline_graph(self):
        goal_path = "mpi-dep-graph/test/data/non_blocking.goal"
        generator = DependencyGraphGenerator(goal_path)
        baseline = generator.generate()
        compressed = generator.generate(trace_compress="lossless")

        self.assertEqual(baseline.num_vertices(), compressed.num_vertices())
        self.assertEqual(baseline.num_edges(), compressed.num_edges())
        self.assertEqual(baseline.graph.vcount(), compressed.graph.vcount())
        self.assertEqual(baseline.graph.ecount(), compressed.graph.ecount())

        attrs = ("type", "r", "l", "cost", "dst_r", "src_r", "src_idx", "loc_idx", "i_idx")
        for idx in range(baseline.graph.vcount()):
            base_v = baseline.graph.vs[idx]
            comp_v = compressed.graph.vs[idx]
            for attr in attrs:
                self.assertEqual(attr in base_v.attributes(), attr in comp_v.attributes())
                if attr in base_v.attributes():
                    self.assertEqual(base_v[attr], comp_v[attr])

        def edge_sig(edge):
            return (
                edge.source,
                edge.target,
                edge["i"],
                edge.attributes().get("v", False),
            )

        baseline_edges = sorted(edge_sig(e) for e in baseline.graph.es)
        compressed_edges = sorted(edge_sig(e) for e in compressed.graph.es)
        self.assertEqual(baseline_edges, compressed_edges)

    def test_collective_ring_semantics_builds_explicit_comm_edges(self):
        goal = """num_ranks 2

rank 0 {
l1: calc 10
l2: AllGather 1024 bytes comm 0 gpu 0 stream 0 seq 7 end
l2 requires l1
l3: calc 5
l3 requires l2
}

rank 1 {
l1: calc 11
l2: AllGather 1024 bytes comm 0 gpu 1 stream 0 seq 7 end
l2 requires l1
l3: calc 6
l3 requires l2
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".goal") as f:
            f.write(goal)
            f.flush()
            graph = DependencyGraphGenerator(f.name).generate(
                trace_compress="lossless",
                collective_semantics="ring",
            )

        sends = [v for v in graph.graph.vs if v["type"].name == "SEND"]
        recvs = [v for v in graph.graph.vs if v["type"].name == "RECV"]
        self.assertEqual(len(sends), 2)
        self.assertEqual(len(recvs), 2)

        cross_rank_comm_edges = 0
        for edge in graph.graph.es:
            src_v = graph.graph.vs[edge.source]
            dst_v = graph.graph.vs[edge.target]
            if src_v["type"].name == "SEND" and dst_v["type"].name == "RECV":
                if src_v["r"] != dst_v["r"]:
                    cross_rank_comm_edges += 1
        self.assertEqual(cross_rank_comm_edges, 2)


if __name__ == "__main__":
    unittest.main()
