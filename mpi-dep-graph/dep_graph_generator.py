import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean
from time import time
from typing import Dict, List, Optional, Tuple, Union

import psutil

from file_parser import GoalFileParser, CommDepFileParser
from goal_elem import (
    GlobalRanks,
    RankStart,
    RankEnd,
    SendOp,
    RecvOp,
    CalcOp,
    CollectiveOp,
    Dependency,
)
from dep_graph import DependencyGraph, VertexType
from trace_compression import (
    CompressedProgram,
    EventSignature,
    LiteralToken,
    TraceCompressMode,
    compress_sequence_iter_template,
    compress_sequence_lossless,
    deduplicate_programs,
)


@dataclass
class RankTrace:
    signature_ids: List[int] = field(default_factory=list)
    label_to_ordinal: Dict[int, int] = field(default_factory=dict)
    local_dep_labels: List[Tuple[int, int, bool]] = field(default_factory=list)
    local_deps: List[Tuple[int, int, bool]] = field(default_factory=list)


class DependencyGraphGenerator(object):
    """
    An object that generates the dependency graph from
    the parsed goal file and communication dependency file.
    """

    def __init__(self, goal_file: str,
                 comm_dep_file: Optional[str] = None) -> None:
        if not os.path.exists(goal_file):
            directory = os.getcwd()
            raise FileNotFoundError(
                f"[ERROR] Goal file {goal_file} does not exist. {directory}"
            )
        self.goal_file = goal_file
        self.comm_dep_file = comm_dep_file
        self.goal_parser = GoalFileParser()
        self.comm_dep_parser = CommDepFileParser()

    @staticmethod
    def __rss_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    def generate(
        self,
        is_loggps: bool = False,
        trace_compress: Union[str, TraceCompressMode] = "off",
        trace_compress_materialize: bool = False,
        collective_semantics: str = "marker",
        trace_compress_rank_parametric: bool = False,
    ) -> DependencyGraph:
        """
        Generates the dependency graph from goal/comm-dep inputs.
        Compression is applied before graph construction when enabled.
        """
        mode = trace_compress
        if isinstance(mode, str):
            mode = TraceCompressMode.from_str(mode)
        if collective_semantics not in {"marker", "ring"}:
            raise ValueError(
                f"[ERROR] Unsupported collective semantics '{collective_semantics}'. "
                "Use one of ['marker', 'ring']."
            )

        # Fast baseline path preserved for default behavior.
        if (
            mode == TraceCompressMode.OFF
            and collective_semantics == "marker"
            and not trace_compress_materialize
            and not trace_compress_rank_parametric
        ):
            return self.__generate_baseline(is_loggps)

        return self.__generate_compressed(
            is_loggps=is_loggps,
            mode=mode,
            trace_compress_materialize=trace_compress_materialize,
            collective_semantics=collective_semantics,
            trace_compress_rank_parametric=trace_compress_rank_parametric,
        )

    def __generate_baseline(self, is_loggps: bool = False) -> DependencyGraph:
        """
        Original code path with no trace compression.
        Collective markers are preserved as local marker events.
        """
        dep_graph = None
        curr_rank = None
        model_name = "LogGPS" if is_loggps else "LogGP"
        print(
            f"[INFO] Generating dependency graph for {model_name} model...",
            flush=True,
        )
        if self.comm_dep_file is None:
            sends = {}
            recvs = {}

        line_count = 0
        end_v = None
        start_v = None
        with open(self.goal_file, "r") as goal_file:
            for line in goal_file:
                elem = self.goal_parser.parse_line(line)
                if elem is None:
                    continue

                if isinstance(elem, GlobalRanks):
                    dep_graph = DependencyGraph(elem.num_ranks, is_loggps)

                elif isinstance(elem, RankStart):
                    curr_rank = elem.rank
                    start_v = None

                elif isinstance(elem, RankEnd):
                    dep_graph.rank_to_end_v[curr_rank] = end_v
                    curr_rank = None
                    end_v = None

                elif isinstance(elem, SendOp):
                    idx = dep_graph.add_vertex(
                        VertexType.SEND,
                        curr_rank,
                        elem.label,
                        elem.data_size,
                        elem.dst,
                    )
                    if self.comm_dep_file is None:
                        key = (curr_rank, elem.dst, elem.tag)
                        if key in recvs:
                            dep_graph.add_edge_by_global_index(idx, recvs[key], True)
                            del recvs[key]
                        else:
                            sends[key] = idx

                elif isinstance(elem, RecvOp):
                    idx = dep_graph.add_vertex(
                        VertexType.RECV,
                        curr_rank,
                        elem.label,
                        elem.data_size,
                        elem.src,
                    )
                    if self.comm_dep_file is None:
                        key = (elem.src, curr_rank, elem.tag)
                        if key in sends:
                            dep_graph.add_edge_by_global_index(sends[key], idx, True)
                            del sends[key]
                        else:
                            recvs[key] = idx

                elif isinstance(elem, CalcOp):
                    idx = dep_graph.add_vertex(
                        VertexType.CALC,
                        curr_rank,
                        elem.label,
                        elem.cost,
                    )
                    end_v = idx
                    if start_v is None and elem.cost > 0:
                        start_v = idx
                        dep_graph.rank_to_start_v[curr_rank] = start_v

                elif isinstance(elem, CollectiveOp):
                    # Baseline keeps legacy marker semantics.
                    idx = dep_graph.add_vertex(
                        VertexType.CALC,
                        curr_rank,
                        elem.label,
                        0,
                    )
                    end_v = idx
                    if start_v is None:
                        start_v = idx
                        dep_graph.rank_to_start_v[curr_rank] = start_v

                elif isinstance(elem, Dependency):
                    dep_graph.add_edge(
                        curr_rank,
                        elem.src_label,
                        curr_rank,
                        elem.dst_label,
                        False,
                        elem.is_irequire,
                    )

                else:
                    raise ValueError(f"[ERROR] Invalid line: {line}")

                line_count += 1
                if line_count % 1000000 == 0:
                    print(
                        f"[INFO] Parsed {line_count} lines in the goal file.",
                        flush=True,
                    )

        if self.comm_dep_file is not None:
            with open(self.comm_dep_file, "r") as comm_dep_file:
                for line in comm_dep_file:
                    src, dst = self.comm_dep_parser.parse_line(line)
                    dep_graph.add_edge(src[0], src[1] + 1, dst[0], dst[1] + 1, True)
        else:
            if len(sends) != 0 or len(recvs) != 0:
                for _, idx in sends.items():
                    print(
                        "[DEBUG] Unmatched send: l"
                        + str(dep_graph.graph.vs[idx]["l"])
                        + " (rank "
                        + str(dep_graph.graph.vs[idx]["r"])
                        + ")"
                    )
                for _, idx in recvs.items():
                    print(
                        "[DEBUG] Unmatched recv: l"
                        + str(dep_graph.graph.vs[idx]["l"])
                        + " (rank "
                        + str(dep_graph.graph.vs[idx]["r"])
                        + ")"
                    )
            assert len(sends) == 0 and len(recvs) == 0, (
                "[ERROR] There are unmatched sends and recvs in the goal file.\n"
                "Communication dependency file is required to match them."
            )

        dep_graph.finalize()
        return dep_graph

    @staticmethod
    def __get_meta_int(
        metadata: Dict[str, Union[int, str]],
        *keys: str,
    ) -> Optional[int]:
        for key in keys:
            if key not in metadata:
                continue
            try:
                return int(metadata[key])
            except (TypeError, ValueError):
                return None
        return None

    def __to_signature(self, elem: Union[SendOp, RecvOp, CalcOp, CollectiveOp]) -> EventSignature:
        metadata = dict(getattr(elem, "metadata", {}) or {})
        stream_id = self.__get_meta_int(metadata, "stream", "cpu")
        channel_id = self.__get_meta_int(metadata, "channel", "nic")
        group_id = self.__get_meta_int(metadata, "group", "group_id", "collective")
        attrs = tuple(sorted((str(k), str(v)) for k, v in metadata.items()))

        if isinstance(elem, SendOp):
            return EventSignature(
                kind="send",
                size_or_cost=elem.data_size,
                peer_rank=elem.dst,
                tag=elem.tag,
                group_id=group_id,
                stream_id=stream_id,
                channel_id=channel_id,
                attrs=attrs,
            )
        if isinstance(elem, RecvOp):
            return EventSignature(
                kind="recv",
                size_or_cost=elem.data_size,
                peer_rank=elem.src,
                tag=elem.tag,
                group_id=group_id,
                stream_id=stream_id,
                channel_id=channel_id,
                attrs=attrs,
            )
        if isinstance(elem, CollectiveOp):
            return EventSignature(
                kind=f"collective:{elem.op_name.lower()}",
                size_or_cost=elem.data_size,
                peer_rank=None,
                tag=self.__get_meta_int(metadata, "seq"),
                group_id=self.__get_meta_int(metadata, "comm", "group", "group_id"),
                stream_id=stream_id,
                channel_id=channel_id,
                attrs=attrs,
            )
        return EventSignature(
            kind="calc",
            size_or_cost=elem.cost,
            peer_rank=None,
            tag=None,
            group_id=group_id,
            stream_id=stream_id,
            channel_id=channel_id,
            attrs=attrs,
        )

    def __intern_signature(
        self,
        signature: EventSignature,
        signatures: List[EventSignature],
        signature_to_id: Dict[EventSignature, int],
    ) -> int:
        signature_id = signature_to_id.get(signature)
        if signature_id is not None:
            return signature_id
        signature_id = len(signatures)
        signatures.append(signature)
        signature_to_id[signature] = signature_id
        return signature_id

    def __parse_trace_ir(
        self,
    ) -> Tuple[List[RankTrace], List[EventSignature], List[Tuple[int, int, int, int]]]:
        rank_traces: List[RankTrace] = []
        signatures: List[EventSignature] = []
        signature_to_id: Dict[EventSignature, int] = {}
        comm_deps: List[Tuple[int, int, int, int]] = []
        pending_sends: Dict[Tuple[int, int, Optional[int]], deque] = defaultdict(deque)
        pending_recvs: Dict[Tuple[int, int, Optional[int]], deque] = defaultdict(deque)

        curr_rank = None
        line_count = 0
        with open(self.goal_file, "r") as goal_file:
            for line in goal_file:
                elem = self.goal_parser.parse_line(line)
                if elem is None:
                    continue

                if isinstance(elem, GlobalRanks):
                    rank_traces = [RankTrace() for _ in range(elem.num_ranks)]
                elif isinstance(elem, RankStart):
                    curr_rank = elem.rank
                elif isinstance(elem, RankEnd):
                    curr_rank = None
                elif isinstance(elem, (SendOp, RecvOp, CalcOp, CollectiveOp)):
                    assert curr_rank is not None
                    trace = rank_traces[curr_rank]
                    ordinal = len(trace.signature_ids)
                    trace.label_to_ordinal[elem.label] = ordinal
                    signature = self.__to_signature(elem)
                    signature_id = self.__intern_signature(
                        signature,
                        signatures,
                        signature_to_id,
                    )
                    trace.signature_ids.append(signature_id)

                    if self.comm_dep_file is None:
                        if isinstance(elem, SendOp):
                            key = (curr_rank, elem.dst, elem.tag)
                            if len(pending_recvs[key]) > 0:
                                dst_rank, dst_ordinal = pending_recvs[key].popleft()
                                comm_deps.append(
                                    (curr_rank, ordinal, dst_rank, dst_ordinal)
                                )
                            else:
                                pending_sends[key].append((curr_rank, ordinal))
                        elif isinstance(elem, RecvOp):
                            key = (elem.src, curr_rank, elem.tag)
                            if len(pending_sends[key]) > 0:
                                src_rank, src_ordinal = pending_sends[key].popleft()
                                comm_deps.append(
                                    (src_rank, src_ordinal, curr_rank, ordinal)
                                )
                            else:
                                pending_recvs[key].append((curr_rank, ordinal))
                elif isinstance(elem, Dependency):
                    assert curr_rank is not None
                    rank_traces[curr_rank].local_dep_labels.append(
                        (elem.src_label, elem.dst_label, elem.is_irequire)
                    )
                else:
                    raise ValueError(f"[ERROR] Invalid line: {line}")

                line_count += 1
                if line_count % 1000000 == 0:
                    print(
                        f"[INFO] Parsed {line_count} lines in the goal file.",
                        flush=True,
                    )

        for rank, trace in enumerate(rank_traces):
            for src_label, dst_label, is_irequire in trace.local_dep_labels:
                if src_label not in trace.label_to_ordinal:
                    raise KeyError(
                        f"[ERROR] Missing src label l{src_label} in rank {rank}"
                    )
                if dst_label not in trace.label_to_ordinal:
                    raise KeyError(
                        f"[ERROR] Missing dst label l{dst_label} in rank {rank}"
                    )
                src_ordinal = trace.label_to_ordinal[src_label]
                dst_ordinal = trace.label_to_ordinal[dst_label]
                trace.local_deps.append((src_ordinal, dst_ordinal, is_irequire))

        if self.comm_dep_file is not None:
            with open(self.comm_dep_file, "r") as comm_dep_file:
                for line in comm_dep_file:
                    src, dst = self.comm_dep_parser.parse_line(line)
                    src_rank, src_label = src[0], src[1] + 1
                    dst_rank, dst_label = dst[0], dst[1] + 1
                    try:
                        src_ordinal = rank_traces[src_rank].label_to_ordinal[src_label]
                        dst_ordinal = rank_traces[dst_rank].label_to_ordinal[dst_label]
                    except KeyError as exc:
                        raise KeyError(
                            "[ERROR] Communication dependency references a label not "
                            f"present in GOAL: src=({src_rank}, l{src_label}), "
                            f"dst=({dst_rank}, l{dst_label})"
                        ) from exc
                    comm_deps.append((src_rank, src_ordinal, dst_rank, dst_ordinal))
        else:
            unmatched_sends = sum(len(v) for v in pending_sends.values())
            unmatched_recvs = sum(len(v) for v in pending_recvs.values())
            if unmatched_sends > 0 or unmatched_recvs > 0:
                raise AssertionError(
                    "[ERROR] There are unmatched sends and recvs in the goal file.\n"
                    "Communication dependency file is required to match them."
                )

        return rank_traces, signatures, comm_deps

    def __program_parametric_key(self,
                                 rank: int,
                                 program: CompressedProgram,
                                 signatures: List[EventSignature],
                                 num_ranks: int) -> Tuple:
        def normalize_signature(signature_id: int) -> Tuple:
            sig = signatures[signature_id]
            rel_peer = None
            if sig.peer_rank is not None:
                rel_peer = (sig.peer_rank - rank) % num_ranks
            return (
                sig.kind,
                sig.size_or_cost,
                rel_peer,
                sig.tag,
                sig.group_id,
                sig.stream_id,
                sig.channel_id,
                sig.attrs,
            )

        key = []
        for token in program.tokens:
            if isinstance(token, LiteralToken):
                key.append(("L", normalize_signature(token.signature_id)))
            else:
                template = tuple(
                    normalize_signature(signature_id)
                    for signature_id in program.templates[token.template_id].signature_ids
                )
                key.append(("R", template, token.count))
        return tuple(key)

    def __count_parametric_unique(self,
                                  rank_programs: List[CompressedProgram],
                                  signatures: List[EventSignature],
                                  num_ranks: int) -> int:
        seen = set()
        for rank, program in enumerate(rank_programs):
            seen.add(self.__program_parametric_key(rank, program, signatures, num_ranks))
        return len(seen)

    def __compress_rank_programs(
        self,
        rank_traces: List[RankTrace],
        signatures: List[EventSignature],
        mode: TraceCompressMode,
        trace_compress_rank_parametric: bool,
    ) -> Tuple[List[CompressedProgram], List[CompressedProgram], List[int], int]:
        rank_programs: List[CompressedProgram] = []
        for trace in rank_traces:
            if mode == TraceCompressMode.OFF:
                program = CompressedProgram(
                    templates=[],
                    tokens=[LiteralToken(signature_id=s) for s in trace.signature_ids],
                    materialized_length=len(trace.signature_ids),
                    mode=mode,
                )
            elif mode == TraceCompressMode.ITER_TEMPLATE:
                program = compress_sequence_iter_template(trace.signature_ids)
            else:
                program = compress_sequence_lossless(
                    trace.signature_ids,
                    mode=mode,
                )
            rank_programs.append(program)

        unique_programs, rank_to_program_id = deduplicate_programs(rank_programs)
        parametric_unique_count = len(unique_programs)
        if trace_compress_rank_parametric:
            parametric_unique_count = self.__count_parametric_unique(
                rank_programs,
                signatures,
                len(rank_traces),
            )

        return (
            rank_programs,
            unique_programs,
            rank_to_program_id,
            parametric_unique_count,
        )

    def __print_compression_report(
        self,
        rank_traces: List[RankTrace],
        rank_programs: List[CompressedProgram],
        unique_programs: List[CompressedProgram],
        rank_to_program_id: List[int],
        parametric_unique_count: int,
        trace_compress_rank_parametric: bool,
    ) -> None:
        raw_per_rank = [len(trace.signature_ids) for trace in rank_traces]
        token_per_rank = [len(program.tokens) for program in rank_programs]
        raw_total = sum(raw_per_rank)

        template_lengths = []
        for program in unique_programs:
            template_lengths.extend(len(t.signature_ids) for t in program.templates)

        templates_count = len(template_lengths)
        avg_template_len = mean(template_lengths) if templates_count > 0 else 0.0
        max_template_len = max(template_lengths) if templates_count > 0 else 0

        per_rank_units = [
            rank_programs[i].estimated_units() for i in range(len(rank_programs))
        ]
        unique_units = sum(program.estimated_units() for program in unique_programs)
        total_units_no_dedup = sum(per_rank_units)
        savings_no_dedup = 0.0
        if raw_total > 0:
            savings_no_dedup = 1.0 - (total_units_no_dedup / raw_total)
        savings_with_dedup = 0.0
        if raw_total > 0:
            savings_with_dedup = 1.0 - (unique_units / raw_total)

        print("[TRACE-COMPRESS] summary")
        print(
            "[TRACE-COMPRESS] raw_events_total={}, num_ranks={}".format(
                raw_total,
                len(rank_traces),
            )
        )
        print(
            "[TRACE-COMPRESS] raw_per_rank(min/avg/max)={}/{:.2f}/{}".format(
                min(raw_per_rank) if raw_per_rank else 0,
                mean(raw_per_rank) if raw_per_rank else 0.0,
                max(raw_per_rank) if raw_per_rank else 0,
            )
        )
        print(
            "[TRACE-COMPRESS] tokens_per_rank(min/avg/max)={}/{:.2f}/{}".format(
                min(token_per_rank) if token_per_rank else 0,
                mean(token_per_rank) if token_per_rank else 0.0,
                max(token_per_rank) if token_per_rank else 0,
            )
        )
        print(
            "[TRACE-COMPRESS] templates={}, avg_template_len={:.2f}, max_template_len={}".format(
                templates_count,
                avg_template_len,
                max_template_len,
            )
        )
        if trace_compress_rank_parametric:
            print(
                "[TRACE-COMPRESS] unique_programs(strict/parametric)={}/{}".format(
                    len(unique_programs),
                    parametric_unique_count,
                )
            )
        else:
            print(
                "[TRACE-COMPRESS] unique_programs={}/{}".format(
                    len(unique_programs),
                    len(rank_programs),
                )
            )
        print(
            "[TRACE-COMPRESS] estimated_units(raw/no_dedup/dedup)={}/{}/{}".format(
                raw_total,
                total_units_no_dedup,
                unique_units,
            )
        )
        print(
            "[TRACE-COMPRESS] estimated_savings(no_dedup/dedup)={:.2%}/{:.2%}".format(
                savings_no_dedup,
                savings_with_dedup,
            )
        )

        for rank in range(min(8, len(rank_traces))):
            program_id = rank_to_program_id[rank]
            print(
                "[TRACE-COMPRESS] rank={} raw={} tokens={} templates={} canonical_program={}".format(
                    rank,
                    raw_per_rank[rank],
                    token_per_rank[rank],
                    len(rank_programs[rank].templates),
                    program_id,
                )
            )

    @staticmethod
    def __signature_to_vertex_type(kind: str) -> VertexType:
        if kind == "send":
            return VertexType.SEND
        if kind == "recv":
            return VertexType.RECV
        if kind.startswith("collective:"):
            return VertexType.CALC
        if kind == "calc":
            return VertexType.CALC
        raise ValueError(f"[ERROR] Unsupported event kind: {kind}")

    def __print_materialized_debug(
        self,
        rank_programs: List[CompressedProgram],
        signatures: List[EventSignature],
        max_ranks: int = 4,
        max_events_per_rank: int = 12,
    ) -> None:
        print(
            f"[TRACE-COMPRESS] materialize preview first "
            f"{max_events_per_rank} events per rank"
        )
        for rank in range(min(max_ranks, len(rank_programs))):
            sample = []
            for i, signature_id in enumerate(rank_programs[rank].iter_materialized()):
                if i >= max_events_per_rank:
                    break
                sample.append(signatures[signature_id].to_compact_str())
            print(f"[TRACE-COMPRESS] rank={rank} preview={sample}")

    def __build_dep_graph_from_compressed(
        self,
        rank_traces: List[RankTrace],
        signatures: List[EventSignature],
        rank_programs: List[CompressedProgram],
        comm_deps: List[Tuple[int, int, int, int]],
        is_loggps: bool,
        collective_semantics: str,
    ) -> DependencyGraph:
        dep_graph = DependencyGraph(len(rank_traces), is_loggps)
        ordinal_start_idx: List[List[int]] = [
            [0 for _ in range(len(trace.signature_ids))]
            for trace in rank_traces
        ]
        ordinal_end_idx: List[List[int]] = [
            [0 for _ in range(len(trace.signature_ids))]
            for trace in rank_traces
        ]
        collective_groups: Dict[Tuple, List[Tuple[int, int, int]]] = defaultdict(list)

        for rank, trace in enumerate(rank_traces):
            program = rank_programs[rank]
            start_v = None
            local_label = 1

            for ordinal, signature_id in enumerate(program.iter_materialized()):
                signature = signatures[signature_id]
                if collective_semantics == "ring" and signature.kind.startswith("collective:"):
                    send_idx = dep_graph.add_vertex(
                        VertexType.SEND,
                        rank,
                        local_label,
                        signature.size_or_cost,
                        rank,
                    )
                    local_label += 1
                    recv_idx = dep_graph.add_vertex(
                        VertexType.RECV,
                        rank,
                        local_label,
                        signature.size_or_cost,
                        rank,
                    )
                    local_label += 1
                    # Local completion edge (sets loc_idx for recv).
                    dep_graph.add_edge_by_global_index(send_idx, recv_idx, is_comm=False)
                    ordinal_start_idx[rank][ordinal] = send_idx
                    ordinal_end_idx[rank][ordinal] = recv_idx
                    collective_key = (
                        signature.kind,
                        signature.group_id,
                        signature.tag,
                        signature.size_or_cost,
                    )
                    collective_groups[collective_key].append((rank, send_idx, recv_idx))
                    if start_v is None:
                        start_v = send_idx
                    continue

                vertex_type = self.__signature_to_vertex_type(signature.kind)
                other_rank = signature.peer_rank
                cost = signature.size_or_cost
                if signature.kind.startswith("collective:"):
                    # Marker fallback for collectives.
                    cost = 0
                idx = dep_graph.add_vertex(
                    vertex_type,
                    rank,
                    local_label,
                    cost,
                    other_rank,
                )
                local_label += 1
                ordinal_start_idx[rank][ordinal] = idx
                ordinal_end_idx[rank][ordinal] = idx

                if (
                    start_v is None
                    and vertex_type == VertexType.CALC
                    and signature.kind == "calc"
                    and signature.size_or_cost > 0
                ):
                    start_v = idx

            if len(trace.signature_ids) == 0:
                raise ValueError(f"[ERROR] Rank {rank} has no events.")
            if start_v is None:
                start_v = ordinal_start_idx[rank][0]
            dep_graph.rank_to_start_v[rank] = start_v
            dep_graph.rank_to_end_v[rank] = ordinal_end_idx[rank][-1]

        # Local dependencies preserve per-rank program order semantics.
        for rank, trace in enumerate(rank_traces):
            for src_ord, dst_ord, is_irequire in trace.local_deps:
                dep_graph.add_edge_by_global_index(
                    ordinal_end_idx[rank][src_ord],
                    ordinal_start_idx[rank][dst_ord],
                    is_comm=False,
                    is_irequires=is_irequire,
                )

        # Point-to-point communication edges from comm-dep or tag matching.
        for src_rank, src_ord, dst_rank, dst_ord in comm_deps:
            src_program = rank_programs[src_rank]
            dst_program = rank_programs[dst_rank]

            src_loc = src_program.indexer.ordinal_to_location(src_ord)
            dst_loc = dst_program.indexer.ordinal_to_location(dst_ord)
            src_norm_ord = src_program.indexer.location_to_ordinal(src_loc)
            dst_norm_ord = dst_program.indexer.location_to_ordinal(dst_loc)

            dep_graph.add_edge_by_global_index(
                ordinal_start_idx[src_rank][src_norm_ord],
                ordinal_start_idx[dst_rank][dst_norm_ord],
                is_comm=True,
            )

        # Optional collective communication semantics: ring coupling.
        if collective_semantics == "ring":
            for _, participants in collective_groups.items():
                if len(participants) < 2:
                    continue
                participants = sorted(participants, key=lambda x: (x[0], x[1]))
                for i, (src_rank, src_send_idx, _) in enumerate(participants):
                    dst_rank, _, dst_recv_idx = participants[(i + 1) % len(participants)]
                    dep_graph.graph.vs[src_send_idx]["dst_r"] = dst_rank
                    dep_graph.graph.vs[dst_recv_idx]["src_r"] = src_rank
                    dep_graph.add_edge_by_global_index(
                        src_send_idx,
                        dst_recv_idx,
                        is_comm=True,
                    )

        dep_graph.finalize()
        return dep_graph

    def __generate_compressed(
        self,
        is_loggps: bool,
        mode: TraceCompressMode,
        trace_compress_materialize: bool,
        collective_semantics: str,
        trace_compress_rank_parametric: bool,
    ) -> DependencyGraph:
        model_name = "LogGPS" if is_loggps else "LogGP"
        print(
            f"[INFO] Generating dependency graph for {model_name} model "
            f"with trace compression mode '{mode.value}' and collective "
            f"semantics '{collective_semantics}'...",
            flush=True,
        )

        parse_start = time()
        rank_traces, signatures, comm_deps = self.__parse_trace_ir()
        parse_t = time() - parse_start
        print(
            "[TRACE-COMPRESS] phase=parse time_s={:.3f} rss_mb={:.2f}".format(
                parse_t,
                self.__rss_mb(),
            )
        )

        compress_start = time()
        (
            rank_programs,
            unique_programs,
            rank_to_program_id,
            parametric_unique_count,
        ) = self.__compress_rank_programs(
            rank_traces,
            signatures,
            mode,
            trace_compress_rank_parametric,
        )
        compress_t = time() - compress_start
        print(
            "[TRACE-COMPRESS] phase=compress time_s={:.3f} rss_mb={:.2f}".format(
                compress_t,
                self.__rss_mb(),
            )
        )

        self.__print_compression_report(
            rank_traces,
            rank_programs,
            unique_programs,
            rank_to_program_id,
            parametric_unique_count,
            trace_compress_rank_parametric,
        )
        if trace_compress_materialize:
            self.__print_materialized_debug(rank_programs, signatures)

        build_start = time()
        dep_graph = self.__build_dep_graph_from_compressed(
            rank_traces,
            signatures,
            rank_programs,
            comm_deps,
            is_loggps,
            collective_semantics,
        )
        build_t = time() - build_start
        print(
            "[TRACE-COMPRESS] phase=build time_s={:.3f} rss_mb={:.2f}".format(
                build_t,
                self.__rss_mb(),
            )
        )
        return dep_graph
