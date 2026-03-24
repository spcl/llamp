from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from dep_graph import DependencyGraph, VertexType
from file_parser import CommDepFileParser, GoalFileParser
from goal_elem import (
    CalcOp,
    Dependency,
    GlobalRanks,
    RankEnd,
    RankStart,
    RecvOp,
    SendOp,
)


_ALGO_MAP = {
    "0": "tree",
    "1": "ring",
    "2": "collnet_direct",
    "3": "collnet_chain",
    "4": "nvls",
    "5": "nvls_tree",
    "6": "pat",
}
_PROTO_MAP = {
    "0": "ll",
    "1": "ll128",
    "2": "simple",
}


@dataclass
class ObservedCollectiveOp:
    label: int
    kind: str
    algo: str
    proto: str
    nranks: int
    data_size: int
    type_size: int
    chunk_size: int
    channels: int
    regime_id: str
    instance_id: str
    participant_rank: Optional[int] = None
    root: Optional[int] = None
    redop: Optional[str] = None
    cpu: Optional[int] = None
    nic: Optional[int] = None
    ring_prevs: Tuple[int, ...] = field(default_factory=tuple)
    ring_nexts: Tuple[int, ...] = field(default_factory=tuple)
    goal_ranks: Optional[int] = None
    inter_goal_prevs: Tuple[int, ...] = field(default_factory=tuple)
    inter_goal_nexts: Tuple[int, ...] = field(default_factory=tuple)
    remote_prevs: Tuple[int, ...] = field(default_factory=tuple)
    remote_nexts: Tuple[int, ...] = field(default_factory=tuple)
    remote_recv_counts: Tuple[int, ...] = field(default_factory=tuple)
    remote_send_counts: Tuple[int, ...] = field(default_factory=tuple)
    channel_local_ns: Tuple[int, ...] = field(default_factory=tuple)
    channel_work_bytes: Tuple[int, ...] = field(default_factory=tuple)
    channel_chunk_bytes: Tuple[int, ...] = field(default_factory=tuple)
    tree_parent: Optional[int] = None
    tree_children: Tuple[int, ...] = field(default_factory=tuple)
    compression: Optional[str] = None

    def normalized_kind(self) -> str:
        return self.kind.lower()

    def normalized_algo(self) -> str:
        return _ALGO_MAP.get(self.algo, self.algo.lower())

    def normalized_proto(self) -> str:
        return _PROTO_MAP.get(self.proto, self.proto.lower())

    def resolved_participant_rank(self, fallback_rank: int) -> int:
        if self.participant_rank is None:
            return fallback_rank
        return self.participant_rank

    def regime_key(self) -> Tuple[str, str, str, int, int, int, Optional[str], Optional[int]]:
        return (
            self.normalized_kind(),
            self.normalized_algo(),
            self.normalized_proto(),
            self.nranks,
            self.channels,
            self.chunk_size,
            self.redop,
            self.root,
        )

    def regime_string(self) -> str:
        parts = [
            f"kind={self.normalized_kind()}",
            f"algo={self.normalized_algo()}",
            f"proto={self.normalized_proto()}",
            f"nranks={self.nranks}",
            f"channels={self.channels}",
            f"chunk_size={self.chunk_size}",
        ]
        if self.redop is not None:
            parts.append(f"redop={self.redop}")
        if self.root is not None:
            parts.append(f"root={self.root}")
        return ", ".join(parts)


RING_ALLGATHER_REGIMES = {
    ("allgather", "ring", "simple", 16, 8, 2_097_152, None, None),
    ("allgather", "ring", "simple", 8, 8, 2_097_152, None, None),
    ("allgather", "ring", "ll", 16, 1, 131_072, None, None),
    ("allgather", "ring", "ll", 16, 1, 32_768, None, None),
    ("allgather", "ring", "ll", 8, 1, 32_768, None, None),
}
RING_REDUCESCATTER_REGIMES = {
    ("reducescatter", "ring", "simple", 16, 8, 2_097_152, "0", None),
    ("reducescatter", "ring", "simple", 8, 8, 2_097_152, "0", None),
}
RING_BROADCAST_REGIMES = {
    ("broadcast", "ring", "ll", 16, 1, 32_768, None, 0),
    ("broadcast", "ring", "ll", 8, 1, 32_768, None, 0),
}
TREE_ALLREDUCE_REGIMES = {
    ("allreduce", "tree", "ll", 16, 1, 32_768, "0", None),
    ("allreduce", "tree", "ll", 16, 1, 32_768, "3", None),
    ("allreduce", "tree", "ll", 8, 1, 32_768, "0", None),
    ("allreduce", "tree", "ll", 8, 1, 32_768, "3", None),
}
LOCAL_DEGENERATE_KINDS = {"allgather", "reducescatter", "allreduce", "broadcast"}


def _parse_csv_ints(value: str) -> Tuple[int, ...]:
    if value == "":
        return tuple()
    return tuple(int(item) for item in value.split(","))


class MacroIRParser(object):
    """
    Parser for a compact LLAMP-specific IR that keeps supported collectives
    intact until dependency-graph generation.
    """

    def __init__(self) -> None:
        self.goal_parser = GoalFileParser()
        self.allgather_ring_simple_re = re.compile(
            r"^l(\d+)\s*:\s*allgather_ring_simple\s+([A-Za-z0-9_.-]+)\s+"
            r"(\d+)b\s+nranks\s+(\d+)\s+chunk_size\s+(\d+)"
            r"(?:\s+cpu\s+(\d+))?(?:\s+nic\s+(\d+))?$"
        )
        self.observed_collective_re = re.compile(
            r"^l(\d+)\s*:\s*observed_collective\s+(.+)$"
        )
        self.generic_collective_re = re.compile(r"^l\d+\s*:\s*([A-Za-z_][A-Za-z0-9_-]*)\b")

    def _parse_observed_collective(self, label: int, body: str) -> ObservedCollectiveOp:
        tokens = body.strip().split()
        if len(tokens) % 2 != 0:
            raise ValueError(f"[ERROR] Invalid observed_collective line: l{label}: {body}")
        values = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens), 2)}
        required = [
            "kind",
            "algo",
            "proto",
            "nranks",
            "data_size",
            "type_size",
            "chunk_size",
            "channels",
            "regime_id",
            "instance_id",
        ]
        missing = [field for field in required if field not in values]
        if missing:
            raise ValueError(
                f"[ERROR] observed_collective is missing required fields {missing}: l{label}: {body}"
            )
        return ObservedCollectiveOp(
            label=label,
            kind=values["kind"],
            algo=values["algo"],
            proto=values["proto"],
            nranks=int(values["nranks"]),
            data_size=int(values["data_size"]),
            type_size=int(values["type_size"]),
            chunk_size=int(values["chunk_size"]),
            channels=int(values["channels"]),
            regime_id=values["regime_id"],
            instance_id=values["instance_id"],
            participant_rank=int(values["participant"]) if "participant" in values else None,
            root=int(values["root"]) if "root" in values else None,
            redop=values.get("redop"),
            cpu=int(values["cpu"]) if "cpu" in values else None,
            nic=int(values["nic"]) if "nic" in values else None,
            ring_prevs=_parse_csv_ints(values.get("ring_prevs", "")),
            ring_nexts=_parse_csv_ints(values.get("ring_nexts", "")),
            goal_ranks=int(values["goal_ranks"]) if "goal_ranks" in values else None,
            inter_goal_prevs=_parse_csv_ints(values.get("inter_goal_prevs", "")),
            inter_goal_nexts=_parse_csv_ints(values.get("inter_goal_nexts", "")),
            remote_prevs=_parse_csv_ints(values.get("remote_prevs", "")),
            remote_nexts=_parse_csv_ints(values.get("remote_nexts", "")),
            remote_recv_counts=_parse_csv_ints(values.get("remote_recv_counts", "")),
            remote_send_counts=_parse_csv_ints(values.get("remote_send_counts", "")),
            channel_local_ns=_parse_csv_ints(values.get("channel_local_ns", "")),
            channel_work_bytes=_parse_csv_ints(values.get("channel_work_bytes", "")),
            channel_chunk_bytes=_parse_csv_ints(values.get("channel_chunk_bytes", "")),
            tree_parent=int(values["tree_parent"]) if "tree_parent" in values else None,
            tree_children=_parse_csv_ints(values.get("tree_children", "")),
            compression=values.get("compression"),
        )

    def parse_line(self, line: str):
        if len(line.strip()) == 0:
            return None

        match = self.allgather_ring_simple_re.match(line)
        if match:
            nranks = int(match.group(4))
            chunk_size = int(match.group(5))
            return ObservedCollectiveOp(
                label=int(match.group(1)),
                kind="allgather",
                algo="ring",
                proto="simple",
                nranks=nranks,
                data_size=int(match.group(3)),
                type_size=1,
                chunk_size=chunk_size,
                channels=1,
                regime_id=f"legacy_allgather_ring_simple_{nranks}r_chunk{chunk_size}",
                instance_id=match.group(2),
                cpu=int(match.group(6)) if match.group(6) is not None else None,
                nic=int(match.group(7)) if match.group(7) is not None else None,
                channel_work_bytes=(int(match.group(3)),),
                channel_chunk_bytes=(chunk_size,),
            )

        match = self.observed_collective_re.match(line)
        if match:
            return self._parse_observed_collective(int(match.group(1)), match.group(2))

        try:
            return self.goal_parser.parse_line(line)
        except ValueError:
            match = self.generic_collective_re.match(line)
            if match is not None:
                op_name = match.group(1)
                raise ValueError(
                    f"[ERROR] Unsupported collective in macro IR: {op_name}. "
                    "Supported macro collectives are emitted as observed_collective records."
                )
            raise


class MacroIRGraphGenerator(object):
    """
    Generates a dependency graph directly from compact macro IR without
    decomposing supported collectives into send/recv microevents.
    """

    def __init__(
        self,
        macro_ir_file: str,
        comm_dep_file: Optional[str] = None,
        same_rank_dep_mode: str = "edge",
        ranks_per_node: int = 1,
    ) -> None:
        if not os.path.exists(macro_ir_file):
            raise FileNotFoundError(f"[ERROR] Macro IR file {macro_ir_file} does not exist.")
        if comm_dep_file is not None and not os.path.exists(comm_dep_file):
            raise FileNotFoundError(f"[ERROR] Communication dependency file {comm_dep_file} does not exist.")
        if same_rank_dep_mode not in {"edge", "metadata"}:
            raise ValueError(f"[ERROR] Unsupported same_rank_dep_mode: {same_rank_dep_mode}")
        self.macro_ir_file = macro_ir_file
        self.comm_dep_file = comm_dep_file
        self.same_rank_dep_mode = same_rank_dep_mode
        self.ranks_per_node = ranks_per_node
        self.parser = MacroIRParser()
        self.comm_dep_parser = CommDepFileParser()
        self.next_local_index: Dict[int, int] = {}
        self.next_trace_order: Dict[int, int] = {}

    def _alloc_local_index(self, rank: int) -> int:
        curr = self.next_local_index.get(rank, -1)
        self.next_local_index[rank] = curr - 1
        return curr

    def _alloc_trace_order(self, rank: int) -> int:
        curr = self.next_trace_order.get(rank, 0)
        self.next_trace_order[rank] = curr + 1
        return curr

    def _set_trace_order(self, dep_graph: DependencyGraph, v_idx: int, rank: int) -> None:
        dep_graph.graph.vs[v_idx]["trace_order"] = self._alloc_trace_order(rank)

    def _chunk_bytes(self, total_work_bytes: int, chunk_bytes: int) -> List[int]:
        remaining = total_work_bytes
        chunks: List[int] = []
        while remaining > 0:
            curr = min(chunk_bytes, remaining)
            chunks.append(curr)
            remaining -= curr
        return chunks

    def _create_calc_vertex(self, dep_graph: DependencyGraph, rank: int, cost: int = 0) -> int:
        v_idx = dep_graph.add_vertex(
            VertexType.CALC,
            rank,
            self._alloc_local_index(rank),
            cost=cost,
        )
        self._set_trace_order(dep_graph, v_idx, rank)
        return v_idx

    def _create_macro_vertex(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        cost: int,
        cpu: Optional[int],
        nic: Optional[int],
        attrs: Dict[str, object],
    ) -> int:
        v_idx = dep_graph.add_vertex(
            VertexType.MACRO,
            rank,
            self._alloc_local_index(rank),
            cost=cost,
            cpu=cpu,
            nic=nic,
        )
        self._set_trace_order(dep_graph, v_idx, rank)
        v_obj = dep_graph.graph.vs[v_idx]
        for key, value in attrs.items():
            v_obj[key] = value
        return v_idx

    def _wrap_entry_exit(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        start_vertices: List[int],
        end_vertices: List[int],
    ) -> Tuple[int, int]:
        if len(start_vertices) == 1:
            entry = start_vertices[0]
        else:
            entry = self._create_calc_vertex(dep_graph, rank, cost=0)
            for start_v in start_vertices:
                dep_graph.add_edge_by_global_index(entry, start_v)
        if len(end_vertices) == 1:
            exit_v = end_vertices[0]
        else:
            exit_v = self._create_calc_vertex(dep_graph, rank, cost=0)
            for end_v in end_vertices:
                dep_graph.add_edge_by_global_index(end_v, exit_v)
        return entry, exit_v

    def _resolve_collective_handler(self, op: ObservedCollectiveOp) -> str:
        if op.nranks == 1 and op.normalized_kind() in LOCAL_DEGENERATE_KINDS:
            return "degenerate"
        if op.compression == "intergoal_roles":
            key = op.regime_key()
            if key in RING_ALLGATHER_REGIMES or key in RING_REDUCESCATTER_REGIMES:
                return "ring_remote_roles"
            if (
                op.normalized_kind() in {"allgather", "reducescatter"}
                and op.normalized_algo() == "ring"
                and len(op.channel_work_bytes) == op.channels
                and len(op.channel_chunk_bytes) == op.channels
                and len(op.remote_prevs) == op.channels
                and len(op.remote_nexts) == op.channels
            ):
                return "ring_remote_roles"
            raise ValueError(
                "[ERROR] intergoal_roles compression is only supported for observed ring "
                f"allgather/reducescatter regimes, got {op.regime_string()} "
                f"(regime_id={op.regime_id})"
            )
        if op.compression == "ring_transfer":
            key = op.regime_key()
            if key in RING_ALLGATHER_REGIMES or key in RING_REDUCESCATTER_REGIMES:
                return "ring_transfer"
            raise ValueError(
                "[ERROR] ring_transfer compression is only supported for observed ring "
                f"allgather/reducescatter regimes, got {op.regime_string()} "
                f"(regime_id={op.regime_id})"
            )
        if op.regime_id.startswith("legacy_allgather_ring_simple_"):
            return "ring_allgather"
        key = op.regime_key()
        if key in RING_ALLGATHER_REGIMES:
            return "ring_allgather"
        if key in RING_REDUCESCATTER_REGIMES:
            return "ring_reducescatter"
        if key in RING_BROADCAST_REGIMES:
            return "ring_broadcast"
        if key in TREE_ALLREDUCE_REGIMES:
            return "tree_allreduce"
        raise ValueError(
            "[ERROR] Unsupported collective regime in macro IR: "
            f"{op.regime_string()} (regime_id={op.regime_id})"
        )

    def _add_ring_collective(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
        collective_name: str,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        if not op.channel_work_bytes or not op.channel_chunk_bytes:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} is missing channel bytes metadata."
            )
        if len(op.channel_work_bytes) != op.channels or len(op.channel_chunk_bytes) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent channel metadata."
            )
        if op.ring_prevs and len(op.ring_prevs) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent ring_prevs metadata."
            )
        if op.ring_nexts and len(op.ring_nexts) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent ring_nexts metadata."
            )

        channel_vertices: List[List[List[int]]] = []
        entry_vertices: List[int] = []
        exit_vertices: List[int] = []
        stage_count = op.goal_ranks if op.goal_ranks is not None else op.nranks
        prevs = (
            list(op.inter_goal_prevs)
            if op.inter_goal_prevs
            else list(op.ring_prevs)
            if op.ring_prevs
            else [((participant_rank - 1) % op.nranks)] * op.channels
        )
        nexts = (
            list(op.inter_goal_nexts)
            if op.inter_goal_nexts
            else list(op.ring_nexts)
            if op.ring_nexts
            else [((participant_rank + 1) % op.nranks)] * op.channels
        )

        for channel_idx in range(op.channels):
            channel_chunks = self._chunk_bytes(
                op.channel_work_bytes[channel_idx],
                op.channel_chunk_bytes[channel_idx],
            )
            per_channel: List[List[int]] = []
            for chunk_idx, chunk_bytes in enumerate(channel_chunks):
                chunk_vertices: List[int] = []
                for stage_idx in range(stage_count):
                    attrs = {
                        "collective": collective_name,
                        "collective_instance_id": op.instance_id,
                        "collective_regime_id": op.regime_id,
                        "collective_label": op.label,
                        "participant_rank": participant_rank,
                        "collective_channel": channel_idx,
                        "collective_chunk": chunk_idx,
                        "collective_stage": stage_idx,
                        "lat_coeff": 0 if stage_idx == 0 else 1,
                        "bw_coeff": 0 if stage_idx == 0 else chunk_bytes,
                        "post_o_coeff": 1 if stage_idx in (0, stage_count - 1) else 2,
                        "local_ns": 0,
                        "emits_send": stage_idx < stage_count - 1,
                        "send_o_coeff": 1 if stage_idx < stage_count - 1 else 0,
                        "ring_prev_participant": prevs[channel_idx],
                        "ring_next_participant": nexts[channel_idx],
                        "effective_nranks": stage_count,
                    }
                    v_idx = self._create_macro_vertex(
                        dep_graph,
                        rank,
                        chunk_bytes,
                        op.cpu,
                        op.nic,
                        attrs,
                    )
                    chunk_vertices.append(v_idx)
                per_channel.append(chunk_vertices)
                for prev_idx, next_idx in zip(chunk_vertices, chunk_vertices[1:]):
                    dep_graph.add_edge_by_global_index(prev_idx, next_idx)
            for prev_chunk, next_chunk in zip(per_channel, per_channel[1:]):
                dep_graph.add_edge_by_global_index(prev_chunk[-1], next_chunk[0])
            channel_vertices.append(per_channel)
            entry_vertices.append(per_channel[0][0])
            exit_vertices.append(per_channel[-1][-1])

        entry, exit_v = self._wrap_entry_exit(dep_graph, rank, entry_vertices, exit_vertices)
        return entry, exit_v, {
            "mode": "ring",
            "participant_rank": participant_rank,
            "goal_rank": rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
            "kind": collective_name,
            "channels": channel_vertices,
            "ring_prevs": tuple(prevs),
            "ring_nexts": tuple(nexts),
        }

    def _add_ring_collective_transfer(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
        collective_name: str,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        if not op.channel_work_bytes or not op.channel_chunk_bytes:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} is missing channel bytes metadata."
            )
        if len(op.channel_work_bytes) != op.channels or len(op.channel_chunk_bytes) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent channel metadata."
            )
        if op.ring_prevs and len(op.ring_prevs) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent ring_prevs metadata."
            )
        if op.ring_nexts and len(op.ring_nexts) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent ring_nexts metadata."
            )

        effective_ranks = op.goal_ranks if op.goal_ranks is not None else op.nranks
        prevs = (
            list(op.inter_goal_prevs)
            if op.inter_goal_prevs
            else list(op.ring_prevs)
            if op.ring_prevs
            else [((participant_rank - 1) % op.nranks)] * op.channels
        )
        nexts = (
            list(op.inter_goal_nexts)
            if op.inter_goal_nexts
            else list(op.ring_nexts)
            if op.ring_nexts
            else [((participant_rank + 1) % op.nranks)] * op.channels
        )
        transfer_o_coeff = 2 * effective_ranks - 2

        entry_v = self._create_calc_vertex(dep_graph, rank, cost=0)
        channel_vertices: List[List[int]] = []
        channel_last_vertices: List[int] = []
        for channel_idx in range(op.channels):
            chunks = self._chunk_bytes(
                op.channel_work_bytes[channel_idx],
                op.channel_chunk_bytes[channel_idx],
            )
            chunk_vertices: List[int] = []
            for chunk_idx, curr_chunk_bytes in enumerate(chunks):
                attrs = {
                    "collective": collective_name,
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": channel_idx,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "chunk_transfer",
                    "ring_prev_participant": prevs[channel_idx],
                    "ring_next_participant": nexts[channel_idx],
                    "transfer_o_coeff": transfer_o_coeff,
                    "transfer_const_ns": 0,
                    "transfer_src_indices": tuple(),
                    "transfer_lat_coeffs": tuple(),
                    "transfer_bw_coeffs": tuple(),
                    "post_o_coeff": 0,
                    "local_ns": 0,
                    "emits_send": False,
                    "send_o_coeff": 0,
                }
                v_idx = self._create_macro_vertex(
                    dep_graph,
                    rank,
                    curr_chunk_bytes,
                    op.cpu,
                    op.nic,
                    attrs,
                )
                chunk_vertices.append(v_idx)
            channel_vertices.append(chunk_vertices)
            channel_last_vertices.append(chunk_vertices[-1])

        exit_v = channel_last_vertices[0]
        if len(channel_last_vertices) > 1:
            exit_v = self._create_calc_vertex(dep_graph, rank, cost=0)
            for last_v in channel_last_vertices:
                dep_graph.add_edge_by_global_index(last_v, exit_v)

        return entry_v, exit_v, {
            "mode": "ring_transfer",
            "participant_rank": participant_rank,
            "goal_rank": rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
            "effective_nranks": effective_ranks,
            "kind": collective_name,
            "entry_vertex": entry_v,
            "exit_vertex": exit_v,
            "channels": channel_vertices,
            "ring_prevs": tuple(prevs),
            "ring_nexts": tuple(nexts),
        }

    def _add_ring_collective_remote_roles(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
        collective_name: str,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        if not op.channel_work_bytes or not op.channel_chunk_bytes:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} is missing channel bytes metadata."
            )
        if len(op.channel_work_bytes) != op.channels or len(op.channel_chunk_bytes) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent channel metadata."
            )
        if len(op.remote_prevs) != op.channels or len(op.remote_nexts) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} is missing remote role metadata."
            )
        recv_counts = list(op.remote_recv_counts) if op.remote_recv_counts else [1 if prev >= 0 else 0 for prev in op.remote_prevs]
        send_counts = list(op.remote_send_counts) if op.remote_send_counts else [1 if nxt >= 0 else 0 for nxt in op.remote_nexts]
        channel_local_ns = list(op.channel_local_ns) if op.channel_local_ns else [0] * op.channels
        if len(recv_counts) != op.channels or len(send_counts) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent remote count metadata."
            )
        if len(channel_local_ns) != op.channels:
            raise ValueError(
                f"[ERROR] Ring collective {op.instance_id} has inconsistent channel_local_ns metadata."
            )

        channel_vertices: List[List[int]] = []
        entry_vertices: List[int] = []
        exit_vertices: List[int] = []
        for channel_idx in range(op.channels):
            remote_prev = op.remote_prevs[channel_idx]
            remote_next = op.remote_nexts[channel_idx]
            if remote_prev < 0 and remote_next < 0:
                channel_vertices.append([])
                continue
            chunks = self._chunk_bytes(
                op.channel_work_bytes[channel_idx],
                op.channel_chunk_bytes[channel_idx],
            )
            recv_o_coeff = (recv_counts[channel_idx] / len(chunks)) if remote_prev >= 0 else 0
            send_o_coeff = (send_counts[channel_idx] / len(chunks)) if remote_next >= 0 else 0
            local_ns_per_chunk = channel_local_ns[channel_idx] / len(chunks)
            chunk_vertices: List[int] = []
            for chunk_idx, curr_chunk_bytes in enumerate(chunks):
                attrs = {
                    "collective": collective_name,
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": channel_idx,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "intergoal_role",
                    "lat_coeff": 1 if remote_prev >= 0 else 0,
                    "bw_coeff": curr_chunk_bytes if remote_prev >= 0 else 0,
                    "post_o_coeff": recv_o_coeff + send_o_coeff,
                    "local_ns": local_ns_per_chunk,
                    "emits_send": remote_next >= 0,
                    "send_o_coeff": send_o_coeff,
                    "ring_prev_participant": remote_prev,
                    "ring_next_participant": remote_next,
                    "effective_nranks": 2 if (remote_prev >= 0 or remote_next >= 0) else 1,
                }
                v_idx = self._create_macro_vertex(
                    dep_graph,
                    rank,
                    curr_chunk_bytes,
                    op.cpu,
                    op.nic,
                    attrs,
                )
                chunk_vertices.append(v_idx)
            for prev_idx, next_idx in zip(chunk_vertices, chunk_vertices[1:]):
                dep_graph.add_edge_by_global_index(prev_idx, next_idx)
            channel_vertices.append(chunk_vertices)
            entry_vertices.append(chunk_vertices[0])
            exit_vertices.append(chunk_vertices[-1])

        if not entry_vertices or not exit_vertices:
            local_v = self._create_calc_vertex(dep_graph, rank, cost=0)
            return local_v, local_v, {
                "mode": "ring_remote_roles",
                "participant_rank": participant_rank,
                "goal_rank": rank,
                "instance_id": op.instance_id,
                "nranks": op.nranks,
                "kind": collective_name,
                "entry_vertex": local_v,
                "exit_vertex": local_v,
                "channels": channel_vertices,
                "remote_prevs": tuple(op.remote_prevs),
                "remote_nexts": tuple(op.remote_nexts),
            }

        # Remote-role collectives need an explicit entry anchor even for the
        # single-channel case so chunk 0 can depend on the peer collective's
        # pre-collective state without creating a ring cycle.
        entry = self._create_calc_vertex(dep_graph, rank, cost=0)
        for start_v in entry_vertices:
            dep_graph.add_edge_by_global_index(entry, start_v)
        if len(exit_vertices) == 1:
            exit_v = exit_vertices[0]
        else:
            exit_v = self._create_calc_vertex(dep_graph, rank, cost=0)
            for end_v in exit_vertices:
                dep_graph.add_edge_by_global_index(end_v, exit_v)
        return entry, exit_v, {
            "mode": "ring_remote_roles",
            "participant_rank": participant_rank,
            "goal_rank": rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
            "kind": collective_name,
            "entry_vertex": entry,
            "exit_vertex": exit_v,
            "channels": channel_vertices,
            "remote_prevs": tuple(op.remote_prevs),
            "remote_nexts": tuple(op.remote_nexts),
        }

    def _add_ring_broadcast(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        if op.root is None:
            raise ValueError(f"[ERROR] Broadcast {op.instance_id} is missing a root rank.")
        if not op.channel_work_bytes or not op.channel_chunk_bytes:
            raise ValueError(
                f"[ERROR] Broadcast {op.instance_id} is missing channel bytes metadata."
            )
        if len(op.channel_work_bytes) != op.channels or len(op.channel_chunk_bytes) != op.channels:
            raise ValueError(
                f"[ERROR] Broadcast {op.instance_id} has inconsistent channel metadata."
            )
        prevs = list(op.ring_prevs) if op.ring_prevs else [((participant_rank - 1) % op.nranks)] * op.channels
        nexts = list(op.ring_nexts) if op.ring_nexts else [((participant_rank + 1) % op.nranks)] * op.channels

        channel_vertices: List[List[int]] = []
        entry_vertices: List[int] = []
        exit_vertices: List[int] = []
        for channel_idx in range(op.channels):
            chunk_vertices: List[int] = []
            is_root = participant_rank == op.root
            is_tail = nexts[channel_idx] == op.root
            for chunk_idx, chunk_bytes in enumerate(
                self._chunk_bytes(op.channel_work_bytes[channel_idx], op.channel_chunk_bytes[channel_idx])
            ):
                attrs = {
                    "collective": "broadcast_ring",
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": channel_idx,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "broadcast",
                    "lat_coeff": 0 if is_root else 1,
                    "bw_coeff": 0 if is_root else chunk_bytes,
                    "post_o_coeff": 1 if (is_root or is_tail) else 2,
                    "local_ns": 0,
                    "emits_send": not is_tail,
                    "send_o_coeff": 1 if not is_tail else 0,
                    "ring_prev_participant": prevs[channel_idx],
                    "ring_next_participant": nexts[channel_idx],
                    "broadcast_root_participant": op.root,
                }
                v_idx = self._create_macro_vertex(dep_graph, rank, chunk_bytes, op.cpu, op.nic, attrs)
                chunk_vertices.append(v_idx)
            for prev_idx, next_idx in zip(chunk_vertices, chunk_vertices[1:]):
                dep_graph.add_edge_by_global_index(prev_idx, next_idx)
            channel_vertices.append(chunk_vertices)
            entry_vertices.append(chunk_vertices[0])
            exit_vertices.append(chunk_vertices[-1])

        entry, exit_v = self._wrap_entry_exit(dep_graph, rank, entry_vertices, exit_vertices)
        return entry, exit_v, {
            "mode": "ring_broadcast",
            "participant_rank": participant_rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
            "root": op.root,
            "channels": channel_vertices,
            "ring_prevs": tuple(prevs),
            "ring_nexts": tuple(nexts),
        }

    def _add_tree_allreduce(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        if not op.channel_work_bytes or not op.channel_chunk_bytes:
            raise ValueError(
                f"[ERROR] Tree collective {op.instance_id} is missing channel bytes metadata."
            )
        if op.channels != 1:
            raise ValueError(
                f"[ERROR] Tree allreduce currently supports one observed channel, got {op.channels}."
            )
        if op.tree_parent is None:
            raise ValueError(f"[ERROR] Tree allreduce {op.instance_id} is missing tree_parent.")

        work_bytes = op.channel_work_bytes[0]
        chunk_bytes = op.channel_chunk_bytes[0]
        chunks = self._chunk_bytes(work_bytes, chunk_bytes)
        parent = op.tree_parent
        children = tuple(child for child in op.tree_children if child >= 0)

        up_recv_vertices_by_child: Dict[int, List[int]] = {child: [] for child in children}
        up_exit_vertices: List[int] = []
        down_recv_vertices: List[int] = []
        down_send_vertices_by_child: Dict[int, List[int]] = {child: [] for child in children}
        prev_local: Optional[int] = None
        first_entry: Optional[int] = None
        last_exit: Optional[int] = None

        for chunk_idx, curr_chunk_bytes in enumerate(chunks):
            local_cursor = prev_local

            for child in children:
                recv_attrs = {
                    "collective": "allreduce_tree_ll",
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": 0,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "up_recv",
                    "lat_coeff": 1,
                    "bw_coeff": curr_chunk_bytes,
                    "post_o_coeff": 1,
                    "local_ns": 0,
                    "emits_send": False,
                    "send_o_coeff": 0,
                    "tree_child_participant": child,
                }
                recv_v = self._create_macro_vertex(dep_graph, rank, curr_chunk_bytes, op.cpu, op.nic, recv_attrs)
                if local_cursor is not None:
                    dep_graph.add_edge_by_global_index(local_cursor, recv_v)
                up_recv_vertices_by_child[child].append(recv_v)
                local_cursor = recv_v
                if first_entry is None:
                    first_entry = recv_v

            up_exit = local_cursor
            if parent != -1:
                up_send_attrs = {
                    "collective": "allreduce_tree_ll",
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": 0,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "up_send",
                    "lat_coeff": 0,
                    "bw_coeff": 0,
                    "post_o_coeff": 1,
                    "local_ns": 0,
                    "emits_send": True,
                    "send_o_coeff": 1,
                    "tree_parent_participant": parent,
                }
                up_send_v = self._create_macro_vertex(dep_graph, rank, curr_chunk_bytes, op.cpu, op.nic, up_send_attrs)
                if up_exit is not None:
                    dep_graph.add_edge_by_global_index(up_exit, up_send_v)
                up_exit = up_send_v
                if first_entry is None:
                    first_entry = up_send_v

            if up_exit is None:
                raise ValueError(
                    f"[ERROR] Tree allreduce {op.instance_id} generated no up-phase vertex for participant {participant_rank}."
                )
            up_exit_vertices.append(up_exit)

            if parent != -1:
                down_recv_attrs = {
                    "collective": "allreduce_tree_ll",
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": 0,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "down_recv",
                    "lat_coeff": 1,
                    "bw_coeff": curr_chunk_bytes,
                    "post_o_coeff": 1,
                    "local_ns": 0,
                    "emits_send": False,
                    "send_o_coeff": 0,
                    "tree_parent_participant": parent,
                }
                down_recv_v = self._create_macro_vertex(
                    dep_graph,
                    rank,
                    curr_chunk_bytes,
                    op.cpu,
                    op.nic,
                    down_recv_attrs,
                )
                dep_graph.add_edge_by_global_index(up_exit, down_recv_v)
                local_cursor = down_recv_v
                down_recv_vertices.append(down_recv_v)
                if first_entry is None:
                    first_entry = down_recv_v
            else:
                local_cursor = up_exit

            for child in children:
                down_send_attrs = {
                    "collective": "allreduce_tree_ll",
                    "collective_instance_id": op.instance_id,
                    "collective_regime_id": op.regime_id,
                    "collective_label": op.label,
                    "participant_rank": participant_rank,
                    "collective_channel": 0,
                    "collective_chunk": chunk_idx,
                    "collective_phase": "down_send",
                    "lat_coeff": 0,
                    "bw_coeff": 0,
                    "post_o_coeff": 1,
                    "local_ns": 0,
                    "emits_send": True,
                    "send_o_coeff": 1,
                    "tree_child_participant": child,
                }
                send_v = self._create_macro_vertex(dep_graph, rank, curr_chunk_bytes, op.cpu, op.nic, down_send_attrs)
                dep_graph.add_edge_by_global_index(local_cursor, send_v)
                down_send_vertices_by_child[child].append(send_v)
                local_cursor = send_v
                if first_entry is None:
                    first_entry = send_v

            last_exit = local_cursor
            prev_local = local_cursor

        if first_entry is None or last_exit is None:
            raise ValueError(
                f"[ERROR] Tree allreduce {op.instance_id} generated an empty macro block for participant {participant_rank}."
            )
        entry, exit_v = self._wrap_entry_exit(dep_graph, rank, [first_entry], [last_exit])
        return entry, exit_v, {
            "mode": "tree_allreduce",
            "participant_rank": participant_rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
            "parent": parent,
            "children": children,
            "up_recv_vertices_by_child": up_recv_vertices_by_child,
            "up_exit_vertices": up_exit_vertices,
            "down_recv_vertices": down_recv_vertices,
            "down_send_vertices_by_child": down_send_vertices_by_child,
        }

    def _add_degenerate_collective(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
    ) -> Tuple[int, int, Dict[str, object]]:
        participant_rank = op.resolved_participant_rank(rank)
        attrs = {
            "collective": f"{op.normalized_kind()}_degenerate",
            "collective_instance_id": op.instance_id,
            "collective_regime_id": op.regime_id,
            "collective_label": op.label,
            "participant_rank": participant_rank,
            "collective_channel": 0,
            "collective_chunk": 0,
            "collective_phase": "local",
            "lat_coeff": 0,
            "bw_coeff": 0,
            "post_o_coeff": 0,
            "local_ns": 0,
            "emits_send": False,
            "send_o_coeff": 0,
        }
        v_idx = self._create_macro_vertex(dep_graph, rank, op.data_size, op.cpu, op.nic, attrs)
        return v_idx, v_idx, {
            "mode": "degenerate",
            "participant_rank": participant_rank,
            "instance_id": op.instance_id,
            "nranks": op.nranks,
        }

    def _add_collective(
        self,
        dep_graph: DependencyGraph,
        rank: int,
        op: ObservedCollectiveOp,
    ) -> Tuple[int, int, Dict[str, object]]:
        handler = self._resolve_collective_handler(op)
        if handler == "ring_allgather":
            return self._add_ring_collective(dep_graph, rank, op, "allgather_ring")
        if handler == "ring_reducescatter":
            return self._add_ring_collective(dep_graph, rank, op, "reducescatter_ring")
        if handler == "ring_transfer":
            collective_name = "allgather_ring" if op.normalized_kind() == "allgather" else "reducescatter_ring"
            return self._add_ring_collective_transfer(dep_graph, rank, op, collective_name)
        if handler == "ring_remote_roles":
            collective_name = "allgather_ring" if op.normalized_kind() == "allgather" else "reducescatter_ring"
            return self._add_ring_collective_remote_roles(dep_graph, rank, op, collective_name)
        if handler == "ring_broadcast":
            return self._add_ring_broadcast(dep_graph, rank, op)
        if handler == "tree_allreduce":
            return self._add_tree_allreduce(dep_graph, rank, op)
        if handler == "degenerate":
            return self._add_degenerate_collective(dep_graph, rank, op)
        raise AssertionError(f"Unexpected collective handler: {handler}")

    def _connect_ring_collectives(
        self,
        dep_graph: DependencyGraph,
        instance_id: str,
        participants: Dict[int, Dict[str, object]],
        nranks: int,
    ) -> None:
        if sorted(participants.keys()) != list(range(nranks)):
            raise ValueError(
                f"[ERROR] Collective {instance_id} is missing participants. "
                f"Expected 0..{nranks - 1}, found {sorted(participants.keys())}."
            )
        sample = next(iter(participants.values()))
        channels: List[List[List[int]]] = sample["channels"]  # type: ignore[index]
        for participant_rank, data in participants.items():
            ring_prevs: Tuple[int, ...] = data["ring_prevs"]  # type: ignore[index]
            channel_vertices: List[List[List[int]]] = data["channels"]  # type: ignore[index]
            for channel_idx, chunk_lists in enumerate(channel_vertices):
                prev_participant = ring_prevs[channel_idx]
                prev_channel_vertices: List[List[List[int]]] = participants[prev_participant]["channels"]  # type: ignore[index]
                if len(chunk_lists) != len(prev_channel_vertices[channel_idx]):
                    raise ValueError(
                        f"[ERROR] Ring collective {instance_id} has inconsistent chunking between "
                        f"participant {prev_participant} and {participant_rank} on channel {channel_idx}."
                    )
                for chunk_vertices, prev_chunk_vertices in zip(
                    chunk_lists,
                    prev_channel_vertices[channel_idx],
                ):
                    for stage_idx in range(1, len(chunk_vertices)):
                        dep_graph.add_edge_by_global_index(
                            prev_chunk_vertices[stage_idx - 1],
                            chunk_vertices[stage_idx],
                            is_comm=True,
                        )

    def _count_inter_goal_ring_hops_impl(
        self,
        src_participant: int,
        dst_participant: int,
        next_map: Dict[int, int],
        goal_rank_map: Dict[int, int],
        ranks_per_node: int = 1,
    ) -> Tuple[int, int]:
        """Returns (inter_hops, intra_hops) between src and dst in the ring.
        When ranks_per_node > 1, uses physical node assignment (rank // rpn)
        instead of goal_rank_map for inter/intra classification."""
        curr = src_participant
        inter_hops = 0
        intra_hops = 0
        steps = 0
        while curr != dst_participant:
            nxt = next_map[curr]
            if ranks_per_node > 1:
                same_node = (curr // ranks_per_node == nxt // ranks_per_node)
            else:
                same_node = (goal_rank_map[curr] == goal_rank_map[nxt])
            if same_node:
                intra_hops += 1
            else:
                inter_hops += 1
            curr = nxt
            steps += 1
            if steps > len(next_map):
                break
        else:
            return inter_hops, intra_hops

        # Fallback: collapse to goal-level ring
        src_goal = goal_rank_map[src_participant]
        dst_goal = goal_rank_map[dst_participant]
        if src_goal == dst_goal:
            return 0, 0

        goal_next_map: Dict[int, int] = {}
        for participant, nxt in next_map.items():
            curr_goal = goal_rank_map[participant]
            next_goal = goal_rank_map[nxt]
            if curr_goal == next_goal:
                continue
            existing = goal_next_map.get(curr_goal)
            if existing is not None and existing != next_goal:
                raise ValueError(
                    f"[ERROR] Inconsistent inter-goal ring topology while collapsing transfer "
                    f"from {src_participant} to {dst_participant}."
                )
            goal_next_map[curr_goal] = next_goal

        if src_goal not in goal_next_map:
            raise ValueError(
                f"[ERROR] Invalid ring topology while computing transfer from "
                f"{src_participant} to {dst_participant}."
            )

        curr_goal = src_goal
        goal_hops = 0
        visited_goals = 0
        while curr_goal != dst_goal:
            curr_goal = goal_next_map[curr_goal]
            goal_hops += 1
            visited_goals += 1
            if visited_goals > len(goal_next_map):
                raise ValueError(
                    f"[ERROR] Invalid collapsed inter-goal ring topology while computing transfer from "
                    f"{src_participant} to {dst_participant}."
                )
        return goal_hops, 0

    def _count_inter_and_intra_goal_ring_hops(
        self,
        src_participant: int,
        dst_participant: int,
        next_map: Dict[int, int],
        goal_rank_map: Dict[int, int],
    ) -> Tuple[int, int]:
        """Returns (inter_hops, intra_hops) pair."""
        inter, intra = self._count_inter_goal_ring_hops_impl(
            src_participant, dst_participant, next_map, goal_rank_map,
            ranks_per_node=self.ranks_per_node,
        )
        return inter, intra

    def _count_inter_goal_ring_hops(
        self,
        src_participant: int,
        dst_participant: int,
        next_map: Dict[int, int],
        goal_rank_map: Dict[int, int],
    ) -> int:
        curr = src_participant
        inter_hops = 0
        steps = 0
        while curr != dst_participant:
            nxt = next_map[curr]
            if goal_rank_map[curr] != goal_rank_map[nxt]:
                inter_hops += 1
            curr = nxt
            steps += 1
            if steps > len(next_map):
                break
        else:
            return inter_hops

        src_goal = goal_rank_map[src_participant]
        dst_goal = goal_rank_map[dst_participant]
        if src_goal == dst_goal:
            return 0

        goal_next_map: Dict[int, int] = {}
        for participant, nxt in next_map.items():
            curr_goal = goal_rank_map[participant]
            next_goal = goal_rank_map[nxt]
            if curr_goal == next_goal:
                continue
            existing = goal_next_map.get(curr_goal)
            if existing is not None and existing != next_goal:
                raise ValueError(
                    f"[ERROR] Inconsistent inter-goal ring topology while collapsing transfer "
                    f"from {src_participant} to {dst_participant}."
                )
            goal_next_map[curr_goal] = next_goal

        if src_goal not in goal_next_map:
            raise ValueError(
                f"[ERROR] Invalid ring topology while computing transfer from "
                f"{src_participant} to {dst_participant}."
            )

        curr_goal = src_goal
        goal_hops = 0
        visited_goals = 0
        while curr_goal != dst_goal:
            curr_goal = goal_next_map[curr_goal]
            goal_hops += 1
            visited_goals += 1
            if visited_goals > len(goal_next_map):
                raise ValueError(
                    f"[ERROR] Invalid collapsed inter-goal ring topology while computing transfer from "
                    f"{src_participant} to {dst_participant}."
                )
        return goal_hops

    def _connect_ring_transfer_collectives(
        self,
        dep_graph: DependencyGraph,
        instance_id: str,
        participants: Dict[int, Dict[str, object]],
        nranks: int,
    ) -> None:
        if sorted(participants.keys()) != list(range(nranks)):
            raise ValueError(
                f"[ERROR] Collective {instance_id} is missing participants. "
                f"Expected 0..{nranks - 1}, found {sorted(participants.keys())}."
            )
        goal_rank_map = {
            participant_rank: int(data["goal_rank"])
            for participant_rank, data in participants.items()
        }
        sample = next(iter(participants.values()))
        sample_channels: List[List[int]] = sample["channels"]  # type: ignore[index]
        effective_nranks = int(sample.get("effective_nranks", nranks))
        for channel_idx in range(len(sample_channels)):
            next_map = {
                participant_rank: int(data["ring_nexts"][channel_idx])  # type: ignore[index]
                for participant_rank, data in participants.items()
            }
            chunk_counts = {
                participant_rank: len(data["channels"][channel_idx])  # type: ignore[index]
                for participant_rank, data in participants.items()
            }
            if len(set(chunk_counts.values())) != 1:
                raise ValueError(
                    f"[ERROR] Ring transfer collective {instance_id} has inconsistent chunk counts "
                    f"on channel {channel_idx}: {chunk_counts}"
                )
            num_chunks = next(iter(chunk_counts.values()))
            for dst_participant, data in participants.items():
                dst_chunks: List[int] = data["channels"][channel_idx]  # type: ignore[index]
                for chunk_idx, dst_v in enumerate(dst_chunks):
                    chunk_bytes = int(dep_graph.graph.vs[dst_v]["cost"])
                    src_indices: List[int] = []
                    lat_coeffs: List[int] = []
                    lat_intra_coeffs: List[int] = []
                    bw_coeffs: List[int] = []
                    o_coeffs: List[int] = []
                    const_ns: List[int] = []
                    for src_participant, src_data in participants.items():
                        if chunk_idx == 0:
                            src_v = int(src_data["entry_vertex"])  # type: ignore[index]
                        else:
                            src_channel_chunks: List[int] = src_data["channels"][channel_idx]  # type: ignore[index]
                            src_v = src_channel_chunks[chunk_idx - 1]
                        inter_hops, intra_hops = self._count_inter_and_intra_goal_ring_hops(
                            src_participant,
                            dst_participant,
                            next_map,
                            goal_rank_map,
                        )
                        src_indices.append(src_v)
                        lat_coeffs.append(inter_hops)
                        lat_intra_coeffs.append(intra_hops)
                        bw_coeffs.append(inter_hops * chunk_bytes)
                        o_coeffs.append(2 * effective_nranks - 2)
                        const_ns.append(0)
                        dep_graph.add_edge_by_global_index(src_v, dst_v, is_comm=False)
                    dst_obj = dep_graph.graph.vs[dst_v]
                    dst_obj["transfer_src_indices"] = tuple(src_indices)
                    dst_obj["transfer_lat_coeffs"] = tuple(lat_coeffs)
                    dst_obj["transfer_lat_intra_coeffs"] = tuple(lat_intra_coeffs)
                    dst_obj["transfer_bw_coeffs"] = tuple(bw_coeffs)
                    dst_obj["transfer_o_coeffs"] = tuple(o_coeffs)
                    dst_obj["transfer_const_ns"] = tuple(const_ns)

    def _connect_ring_remote_role_collectives(
        self,
        dep_graph: DependencyGraph,
        instance_id: str,
        participants: Dict[int, Dict[str, object]],
        nranks: int,
    ) -> None:
        if sorted(participants.keys()) != list(range(nranks)):
            raise ValueError(
                f"[ERROR] Collective {instance_id} is missing participants. "
                f"Expected 0..{nranks - 1}, found {sorted(participants.keys())}."
            )
        for participant_rank, data in participants.items():
            remote_prevs: Tuple[int, ...] = data["remote_prevs"]  # type: ignore[index]
            channel_vertices: List[List[int]] = data["channels"]  # type: ignore[index]
            for channel_idx, chunk_vertices in enumerate(channel_vertices):
                prev_participant = remote_prevs[channel_idx]
                if prev_participant < 0:
                    continue
                prev_channel_vertices: List[List[int]] = participants[prev_participant]["channels"]  # type: ignore[index]
                prev_chunks = prev_channel_vertices[channel_idx]
                if len(prev_chunks) != len(chunk_vertices):
                    raise ValueError(
                        f"[ERROR] Ring remote-role collective {instance_id} has inconsistent chunking between "
                        f"participant {prev_participant} and {participant_rank} on channel {channel_idx}."
                    )
                prev_entry = int(participants[prev_participant]["entry_vertex"])  # type: ignore[index]
                for chunk_idx, curr_v in enumerate(chunk_vertices):
                    src_v = prev_entry if chunk_idx == 0 else prev_chunks[chunk_idx - 1]
                    dep_graph.add_edge_by_global_index(src_v, curr_v, is_comm=True)

    def _connect_tree_allreduce(
        self,
        dep_graph: DependencyGraph,
        instance_id: str,
        participants: Dict[int, Dict[str, object]],
        nranks: int,
    ) -> None:
        if sorted(participants.keys()) != list(range(nranks)):
            raise ValueError(
                f"[ERROR] Collective {instance_id} is missing participants. "
                f"Expected 0..{nranks - 1}, found {sorted(participants.keys())}."
            )
        for participant_rank, data in participants.items():
            children: Tuple[int, ...] = data["children"]  # type: ignore[index]
            up_recv_vertices_by_child: Dict[int, List[int]] = data["up_recv_vertices_by_child"]  # type: ignore[index]
            up_exit_vertices: List[int] = data["up_exit_vertices"]  # type: ignore[index]
            down_recv_vertices: List[int] = data["down_recv_vertices"]  # type: ignore[index]
            for child_participant in children:
                child_up_exit_vertices: List[int] = participants[child_participant]["up_exit_vertices"]  # type: ignore[index]
                parent_recv_vertices = up_recv_vertices_by_child[child_participant]
                if len(child_up_exit_vertices) != len(parent_recv_vertices):
                    raise ValueError(
                        f"[ERROR] Tree collective {instance_id} has inconsistent reduce chunks "
                        f"between participant {child_participant} and {participant_rank}."
                    )
                for child_up, parent_up in zip(child_up_exit_vertices, parent_recv_vertices):
                    dep_graph.add_edge_by_global_index(child_up, parent_up, is_comm=True)

            parent = data["parent"]  # type: ignore[index]
            if parent == -1:
                continue
            parent_down_send_vertices: Dict[int, List[int]] = participants[parent]["down_send_vertices_by_child"]  # type: ignore[index]
            parent_send_vertices = parent_down_send_vertices[participant_rank]
            if len(parent_send_vertices) != len(down_recv_vertices):
                raise ValueError(
                    f"[ERROR] Tree collective {instance_id} has inconsistent broadcast chunks "
                    f"between participant {parent} and {participant_rank}."
                )
            for parent_down, child_down in zip(parent_send_vertices, down_recv_vertices):
                dep_graph.add_edge_by_global_index(parent_down, child_down, is_comm=True)

    def _connect_ring_broadcast(
        self,
        dep_graph: DependencyGraph,
        instance_id: str,
        participants: Dict[int, Dict[str, object]],
        nranks: int,
    ) -> None:
        if sorted(participants.keys()) != list(range(nranks)):
            raise ValueError(
                f"[ERROR] Collective {instance_id} is missing participants. "
                f"Expected 0..{nranks - 1}, found {sorted(participants.keys())}."
            )
        for participant_rank, data in participants.items():
            root = data["root"]  # type: ignore[index]
            if participant_rank == root:
                continue
            ring_prevs: Tuple[int, ...] = data["ring_prevs"]  # type: ignore[index]
            channel_vertices: List[List[int]] = data["channels"]  # type: ignore[index]
            for channel_idx, chunk_vertices in enumerate(channel_vertices):
                prev_participant = ring_prevs[channel_idx]
                prev_chunk_vertices: List[List[int]] = participants[prev_participant]["channels"]  # type: ignore[index]
                if len(prev_chunk_vertices[channel_idx]) != len(chunk_vertices):
                    raise ValueError(
                        f"[ERROR] Ring broadcast {instance_id} has inconsistent chunking between "
                        f"participant {prev_participant} and {participant_rank} on channel {channel_idx}."
                    )
                for prev_v, curr_v in zip(prev_chunk_vertices[channel_idx], chunk_vertices):
                    dep_graph.add_edge_by_global_index(prev_v, curr_v, is_comm=True)

    def generate(self, is_loggps: bool = False) -> DependencyGraph:
        sections: Dict[int, List[object]] = {}
        num_ranks: Optional[int] = None
        curr_rank: Optional[int] = None

        with open(self.macro_ir_file, "r") as in_f:
            for line in in_f:
                elem = self.parser.parse_line(line)
                if elem is None:
                    continue
                if isinstance(elem, GlobalRanks):
                    num_ranks = elem.num_ranks
                    sections = {rank: [] for rank in range(num_ranks)}
                    continue
                if isinstance(elem, RankStart):
                    curr_rank = elem.rank
                    continue
                if isinstance(elem, RankEnd):
                    curr_rank = None
                    continue
                if curr_rank is None:
                    raise ValueError(f"[ERROR] Element outside rank block: {line.strip()}")
                sections[curr_rank].append(elem)

        if num_ranks is None:
            raise ValueError("[ERROR] Macro IR is missing the num_ranks header.")

        dep_graph = DependencyGraph(num_ranks, is_loggps)
        self.next_local_index = {rank: -1 for rank in range(num_ranks)}
        self.next_trace_order = {rank: 0 for rank in range(num_ranks)}

        if self.comm_dep_file is None:
            sends: Dict[Tuple[int, int, Optional[int]], int] = {}
            recvs: Dict[Tuple[int, int, Optional[int]], int] = {}
        pending_local_deps: List[Tuple[int, Dependency]] = []
        label_to_entry: List[Dict[int, int]] = [{} for _ in range(num_ranks)]
        label_to_exit: List[Dict[int, int]] = [{} for _ in range(num_ranks)]
        rank_op_offset_to_global_index: List[Dict[int, int]] = [{} for _ in range(num_ranks)]
        rank_op_offsets: List[int] = [0 for _ in range(num_ranks)]
        collective_instances: Dict[str, Dict[str, object]] = {}

        for rank in range(num_ranks):
            first_entry: Optional[int] = None
            last_exit: Optional[int] = None
            for elem in sections[rank]:
                if isinstance(elem, Dependency):
                    pending_local_deps.append((rank, elem))
                    continue

                if isinstance(elem, CalcOp):
                    idx = dep_graph.add_vertex(
                        VertexType.CALC,
                        rank,
                        elem.label,
                        cost=elem.cost,
                        cpu=elem.cpu,
                    )
                    self._set_trace_order(dep_graph, idx, rank)
                    entry = exit_v = idx
                elif isinstance(elem, SendOp):
                    idx = dep_graph.add_vertex(
                        VertexType.SEND,
                        rank,
                        elem.label,
                        elem.data_size,
                        elem.dst,
                        elem.cpu,
                        elem.nic,
                    )
                    dep_graph.graph.vs[idx]["trace_order"] = elem.label
                    if self.comm_dep_file is None:
                        key = (rank, elem.dst, elem.tag)
                        if key in recvs:
                            dep_graph.add_edge_by_global_index(idx, recvs[key], True)
                            del recvs[key]
                        else:
                            sends[key] = idx
                    entry = exit_v = idx
                elif isinstance(elem, RecvOp):
                    idx = dep_graph.add_vertex(
                        VertexType.RECV,
                        rank,
                        elem.label,
                        elem.data_size,
                        elem.src,
                        elem.cpu,
                        elem.nic,
                    )
                    dep_graph.graph.vs[idx]["trace_order"] = elem.label
                    if self.comm_dep_file is None:
                        key = (elem.src, rank, elem.tag)
                        if key in sends:
                            dep_graph.add_edge_by_global_index(sends[key], idx, True)
                            del sends[key]
                        else:
                            recvs[key] = idx
                    entry = exit_v = idx
                elif isinstance(elem, ObservedCollectiveOp):
                    entry, exit_v, collective_data = self._add_collective(dep_graph, rank, elem)
                    instance = collective_instances.setdefault(
                        elem.instance_id,
                        {
                            "kind": collective_data["mode"],
                            "nranks": elem.nranks,
                            "participants": {},
                        },
                    )
                    participant_rank = collective_data["participant_rank"]  # type: ignore[index]
                    instance["participants"][participant_rank] = collective_data  # type: ignore[index]
                else:
                    raise ValueError(f"[ERROR] Unsupported macro IR element: {type(elem).__name__}")

                label_to_entry[rank][elem.label] = entry
                label_to_exit[rank][elem.label] = exit_v
                rank_op_offset_to_global_index[rank][rank_op_offsets[rank]] = entry
                rank_op_offsets[rank] += 1
                if first_entry is None:
                    first_entry = entry
                last_exit = exit_v

            if first_entry is None or last_exit is None:
                raise ValueError(f"[ERROR] Rank {rank} has no operations in macro IR.")
            dep_graph.rank_to_start_v[rank] = first_entry
            dep_graph.rank_to_end_v[rank] = last_exit

        if self.comm_dep_file is not None:
            replay_nic_pairs = []
            replay_prec_pairs = []

            def resolve_comm_dep_global(rank: int, offset: int) -> int:
                if offset in rank_op_offset_to_global_index[rank]:
                    return rank_op_offset_to_global_index[rank][offset]
                legacy_label = offset + 1
                if legacy_label in dep_graph.local_index_to_global_index[rank]:
                    return dep_graph.local_index_to_global_index[rank][legacy_label]
                raise KeyError(
                    f"[ERROR] Could not resolve comm-dep offset {offset} on rank {rank}. "
                    "This usually means the sidecar was generated from a differently "
                    "labeled macro trace."
                )

            with open(self.comm_dep_file, "r") as comm_dep_file:
                for line in comm_dep_file:
                    src, dst = self.comm_dep_parser.parse_line(line)
                    src_global = resolve_comm_dep_global(src[0], src[1])
                    dst_global = resolve_comm_dep_global(dst[0], dst[1])
                    src_type = dep_graph.graph.vs[src_global]["type"]
                    dst_type = dep_graph.graph.vs[dst_global]["type"]
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

                    if not is_comm:
                        replay_prec_pairs.append((src_global, dst_global))
                        continue

                    if is_comm or self.same_rank_dep_mode == "edge":
                        dep_graph.add_edge_by_global_index(src_global, dst_global, is_comm)
        elif sends or recvs:
            raise ValueError("[ERROR] There are unmatched sends and recvs in the macro IR.")

        for instance_id, spec in collective_instances.items():
            participants = spec["participants"]  # type: ignore[index]
            nranks = spec["nranks"]  # type: ignore[index]
            mode = next(iter(participants.values()))["mode"]  # type: ignore[index]
            if mode == "ring":
                self._connect_ring_collectives(dep_graph, instance_id, participants, nranks)
            elif mode == "ring_transfer":
                self._connect_ring_transfer_collectives(dep_graph, instance_id, participants, nranks)
            elif mode == "ring_remote_roles":
                self._connect_ring_remote_role_collectives(dep_graph, instance_id, participants, nranks)
            elif mode == "ring_broadcast":
                self._connect_ring_broadcast(dep_graph, instance_id, participants, nranks)
            elif mode == "tree_allreduce":
                self._connect_tree_allreduce(dep_graph, instance_id, participants, nranks)
            elif mode == "degenerate":
                continue
            else:
                raise AssertionError(f"Unexpected collective mode for {instance_id}: {mode}")

        for rank, dep in pending_local_deps:
            src_idx = label_to_exit[rank][dep.src_label]
            dst_idx = label_to_entry[rank][dep.dst_label]
            dep_graph.add_edge_by_global_index(
                src_idx,
                dst_idx,
                is_comm=False,
                is_irequires=dep.is_irequire,
            )

        dep_graph.finalize()
        if self.comm_dep_file is not None:
            dep_graph.graph["replay_nic_pairs"] = replay_nic_pairs
            dep_graph.graph["replay_prec_pairs"] = replay_prec_pairs
        return dep_graph
