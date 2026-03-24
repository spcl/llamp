import os
from collections import defaultdict
from typing import Optional, List, Union, Tuple
from file_parser import *
from goal_elem import *
from dep_graph import DependencyGraph, VertexType


class DependencyGraphGenerator(object):
    """
    An object that generates the dependency graph from
    the parsed goal file and communication dependency file.
    """
    def __init__(self, goal_file: str,
                 comm_dep_file: Optional[str] = None,
                 same_rank_dep_mode: str = "edge") -> None:
        """
        Initialize the dependency graph generator.
        @param goal_file: The path to the goal file.
        @param comm_dep_file: The path to the communication dependency file.
        If it is None, then the dependency graph generator will attempt
        to match the sends and recvs in the goal file simply by message tags.
        """
        # Makes sure that the goal file exists
        if not os.path.exists(goal_file):
            directory = os.getcwd()
            raise FileNotFoundError(f"[ERROR] Goal file {goal_file} does not exist. {directory}")
        self.goal_file = goal_file
        self.comm_dep_file = comm_dep_file
        if same_rank_dep_mode not in {"edge", "metadata"}:
            raise ValueError(
                f"[ERROR] Unsupported same_rank_dep_mode: {same_rank_dep_mode}"
            )
        self.same_rank_dep_mode = same_rank_dep_mode
        self.goal_parser = GoalFileParser()
        self.comm_dep_parser = CommDepFileParser()
    
    def generate(self, is_loggps: bool = False) -> DependencyGraph:
        """
        Generates the dependency graph from the goal file and
        the communication dependency file.
        @param is_loggps: Whether the dependency graph produced
        is for the LogGPS model. If it is True, additional edges will
        be added between the local dependency of a RECV vertex and
        its corresponding SEND vertex. Since this uses additional
        memory, it is suggested to set this to False if the dependency
        graph is not for the LogGPS model.
        @return: The generated dependency graph.
        """
        dep_graph = None
        curr_rank = None
        # Parses the goal file
        goal_file = open(self.goal_file, "r")
        model_name = "LogGPS" if is_loggps else "LogGP"
        print(f"[INFO] Generating dependency graph for {model_name} model...", flush=True)
        # Keeps track of send/recv operations keyed by (src_rank, dst_rank, tag).
        # This is always populated so we can still recover unambiguous base
        # send->recv matches even when a richer replay sidecar is supplied.
        sends = defaultdict(list)
        recvs = defaultdict(list)
        # LogGOPSim comm-dep sidecars record per-rank operation offsets rather
        # than GOAL labels. Full traces often use contiguous labels so the two
        # coincide, but reduced/truncated witnesses keep original labels and
        # need an explicit offset map.
        rank_op_offset_to_global_index = defaultdict(dict)
        rank_op_offsets = defaultdict(int)

        line_count = 0
        end_v = None
        start_v = None
        for line in goal_file:
            elem = self.goal_parser.parse_line(line)
            # Checks if the line is empty
            if elem is None:
                continue
            
            if isinstance(elem, GlobalRanks):
                # Initializes the dependency graph with the number of ranks
                # in the MPI program
                dep_graph = DependencyGraph(elem.num_ranks, is_loggps)
            
            elif isinstance(elem, RankStart):
                # Sets the current rank
                curr_rank = elem.rank
                start_v = None
            
            elif isinstance(elem, RankEnd):
                dep_graph.rank_to_end_v[curr_rank] = end_v
                curr_rank = None
                end_v = None
            
            elif isinstance(elem, SendOp):
                # Adds the send operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.SEND, curr_rank,
                                           elem.label, elem.data_size,
                                           elem.dst, elem.cpu, elem.nic)
                rank_op_offset_to_global_index[curr_rank][rank_op_offsets[curr_rank]] = idx
                rank_op_offsets[curr_rank] += 1
                key = (curr_rank, elem.dst, elem.tag)
                sends[key].append(idx)

            elif isinstance(elem, RecvOp):
                # Adds the recv operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.RECV, curr_rank,
                                           elem.label, elem.data_size,
                                           elem.src, elem.cpu, elem.nic)
                rank_op_offset_to_global_index[curr_rank][rank_op_offsets[curr_rank]] = idx
                rank_op_offsets[curr_rank] += 1
                key = (elem.src, curr_rank, elem.tag)
                recvs[key].append(idx)

            elif isinstance(elem, CalcOp):
                # Adds the calc operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.CALC, curr_rank,
                                     elem.label, elem.cost, cpu=elem.cpu)
                rank_op_offset_to_global_index[curr_rank][rank_op_offsets[curr_rank]] = idx
                rank_op_offsets[curr_rank] += 1
                end_v = idx
                if start_v is None and elem.cost > 0:
                    start_v = idx
                    dep_graph.rank_to_start_v[curr_rank] = start_v
            
            elif isinstance(elem, Dependency):
                # Adds the dependency to the dependency graph
                dep_graph.add_edge(curr_rank, elem.src_label,
                                   curr_rank, elem.dst_label,
                                   False, elem.is_irequire)

            else:
                raise ValueError(f"[ERROR] Invalid line: {line}")


            line_count += 1
            if line_count % 1000000 == 0:
                print(f"[INFO] Parsed {line_count} lines in the goal file.", flush=True)
        goal_file.close()

        def add_unique_tag_matches(require_all: bool) -> None:
            unmatched = []
            all_keys = set(sends.keys()) | set(recvs.keys())
            for key in all_keys:
                send_vertices = sends.get(key, [])
                recv_vertices = recvs.get(key, [])
                if len(send_vertices) == 1 and len(recv_vertices) == 1:
                    send_idx = send_vertices[0]
                    recv_idx = recv_vertices[0]
                    recv_v = dep_graph.graph.vs[recv_idx]
                    if "src_idx" in recv_v.attributes() and recv_v["src_idx"] == send_idx:
                        continue
                    dep_graph.add_edge_by_global_index(send_idx, recv_idx, True)
                elif send_vertices or recv_vertices:
                    unmatched.append((key, len(send_vertices), len(recv_vertices)))
            if require_all and unmatched:
                for key, ns, nr in unmatched[:20]:
                    print(f"[DEBUG] Unmatched key {key}: sends={ns}, recvs={nr}")
                raise AssertionError(
                    "[ERROR] There are unmatched sends and recvs in the goal file.\n"
                    "Communication dependency file is required to match them."
                )

        if self.comm_dep_file is not None:
            # Parses the communication dependency file if given.
            # Only true cross-rank SEND->RECV message matches belong in the DAG.
            # Richer replay/order relations are stored as side metadata and
            # consumed later as LP constraints so they do not create graph cycles.
            replay_nic_pairs = []
            replay_prec_pairs = []
            comm_dep_file = open(self.comm_dep_file, "r")

            def resolve_comm_dep_global(rank: int, offset: int) -> int:
                if offset in rank_op_offset_to_global_index[rank]:
                    return rank_op_offset_to_global_index[rank][offset]
                legacy_label = offset + 1
                if legacy_label in dep_graph.local_index_to_global_index[rank]:
                    return dep_graph.local_index_to_global_index[rank][legacy_label]
                raise KeyError(
                    f"[ERROR] Could not resolve comm-dep offset {offset} on rank {rank}. "
                    "This usually means the sidecar was generated from a differently "
                    "labeled trace."
                )

            for line in comm_dep_file:
                src, dst = self.comm_dep_parser.parse_line(line)
                src_global = resolve_comm_dep_global(src[0], src[1])
                dst_global = resolve_comm_dep_global(dst[0], dst[1])
                src_type = dep_graph.graph.vs[src_global]["type"]
                dst_type = dep_graph.graph.vs[dst_global]["type"]

                # Exact replay sidecars may include:
                # - same-rank local replay edges, and
                # - cross-rank precedence edges between non-send/recv ops.
                # Only plain SEND/MACRO -> RECV/MACRO pairs should be treated
                # as message-match communication edges.
                is_comm = (
                    src[0] != dst[0]
                    and src_type in {VertexType.SEND, VertexType.MACRO}
                    and dst_type in {VertexType.RECV, VertexType.MACRO}
                )
                if src[0] == dst[0]:
                    pair = (src_global, dst_global)
                    replay_nic_pairs.append(pair)
                    replay_prec_pairs.append(pair)
                    continue

                # Rich exact replay sidecars also contain cross-rank precedence
                # relations between local compute/recv/send operations. Those
                # are not message-match edges and can form cycles if inserted
                # into the DAG directly, so keep them as replay metadata.
                if not is_comm:
                    replay_prec_pairs.append((src_global, dst_global))
                    continue

                if src[0] != dst[0] or self.same_rank_dep_mode == "edge":
                    dep_graph.add_edge_by_global_index(src_global, dst_global, is_comm)
            comm_dep_file.close()
            # Recover any remaining unambiguous base send->recv matches that the
            # replay sidecar did not spell out explicitly.
            add_unique_tag_matches(require_all=False)
        else:
            add_unique_tag_matches(require_all=True)

        dep_graph.finalize()
        if self.comm_dep_file is not None:
            dep_graph.graph["replay_nic_pairs"] = replay_nic_pairs
            dep_graph.graph["replay_prec_pairs"] = replay_prec_pairs
        return dep_graph
