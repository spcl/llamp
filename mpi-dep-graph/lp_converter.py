from __future__ import annotations

import numpy as np
import igraph
import os
import heapq
try:
    import gurobipy as gp
except ImportError:
    gp = None
from functools import lru_cache
from time import time
from tqdm import tqdm
from ortools.linear_solver import pywraplp
from typing import List, Dict, Tuple, Set, Optional, Union
from dep_graph import DependencyGraph, VertexType
from topology import NetTopology
from utils import *


LOGGPS_ERROR_MSG = "[ERROR] It seems that your LP model was not created with the LogGPS support"

class LPConverter(object):
    """
    A object that converts the dependency graph into a linear program.
    """
    def __init__(self, dep_graph: DependencyGraph,
                 o: int = 1500,
                 S: Optional[int] = None) -> None:
        """
        @param dep_graph: The dependency graph to be converted.
        @param o: The overhead parameter o in the LogGPS model.
        @param S: The S parameter in the LogGPS model. If None, then
        LogGP will be used instead of LogGPS.
        """
        self.dep_graph = dep_graph
        self.o = o
        self.S = S if S is not None else float("inf")
        if S is not None:
            print(f"[INFO] Using LogGPS, o: {o}, S: {S}")
        else:
            print(f"[INFO] Using LogGP, o: {o}")
        self.use_gurobi = is_gurobi_installed()
    
    def __add_variables(self, model: Union[gp.Model, pywraplp.Solver],
                        num_ranks: int,
                        pairwise_analysis: bool = False,
                        G: Optional[float] = None,
                        verbose: bool = True,
                        ranks_per_node: int = 1) -> Tuple:
        """
        A private helper function that adds the required variables
        to the given model. If `pairwise_analysis` is True, then
        a new variable `l` will be created for each pair of ranks.

        If the given `G` is None when the constructor is called,
        then the bandwidth between ranks will be considered as
        variable/variables in the analysis. Otherwise, it will
        be considered as a constant value. Similar to `l` above,
        a new variable `g` will be created for each pair of ranks
        if `pairwise_analysis` is True.
        
        @return A tuple that contains the following:
        - The variable `l` if pairwise analysis is disabled, otherwise an 
        matrix that contains the variables `l` between each pair of ranks.
        - If `G` is not None, then we will define element will be a constant 
        value.
        - The type of the return value will be
        Tuple[Union[var, np.ndarray], Union[var, np.ndarray, float]]
        """
        if self.use_gurobi:
            g = None
            l = None
            # If GUROBI is installed
            if pairwise_analysis:
                num_ls = (1 + (num_ranks - 1)) * (num_ranks - 1) // 2
                # Creates a list of solver variables for l
                ls = model.addVars(num_ls, name="l")
                # Creates a list of solver variables for g
                if not G:
                    gs = model.addVars(num_ls, name="g")

                # Creates a matrix of objects
                # that can be indexed by two integers
                # Each entry i,j represents the latency between
                # rank i and rank j
                l = np.empty((num_ranks, num_ranks), dtype=gp.Var)
                g = np.empty((num_ranks, num_ranks), dtype=gp.Var)

                c = 0
                for i in range(num_ranks):
                    # Special case of self-messaging, seen in NPB CG
                    # FIXME: The latency and bandwidth costs of self-messaging
                    # are not zero
                    l[i, i] = 0
                    g[i, i] = 0
                    for j in range(i + 1, num_ranks):
                        l[i, j] = l[j, i] = ls[c]
                        if not G:
                            g[i, j] = g[j, i] = gs[c]
                        else:
                            g[i, j] = g[j, i] = G
                        c += 1
            else:
                l = model.addVar(name="l")
                l_intra = model.addVar(name="l_intra") if ranks_per_node > 1 else l
                if not G:
                    g = model.addVar(name="g")
                else:
                    g = G
            
            N = model.addVar(name="N", lb=num_ranks, ub=num_ranks)
            o = model.addVar(name="o", lb=self.o, ub=self.o)
            if self.S != float("inf"):
                S = model.addVar(name="S", lb=self.S, ub=self.S)
            model.update()
            return l, g, l_intra
        else:
            # If GUROBI is not installed, then use the default ortools model
            print("[WARNING] GUROBI is not installed. Using the default ortools model.")
            if pairwise_analysis:
                raise NotImplementedError("Pairwise analysis is not implemented for OR-Tools models.")
            l = model.NumVar(0.0, model.infinity(), "l")
            l_intra = model.NumVar(0.0, model.infinity(), "l_intra") if ranks_per_node > 1 else l
            if G is None:
                g = model.NumVar(0.0, model.infinity(), "g")
            else:
                g = G
            # Encodes the number of ranks in the model
            model.NumVar(num_ranks, num_ranks, "N")

            return l, g, l_intra

    @lru_cache(maxsize=128)
    def __get_msg_cpu_overhead(self, msg_size: int) -> float:
        """
        A helper function that returns the CPU overhead
        of a message of the given size.
        """
        return 0
        # if msg_size <= 1024:
        #     return int(1500 + 9 * msg_size)
        # else:
        #     return int(12000 + msg_size * 0.2)



    def __get_constr_name(self, v1: int, v2: int) -> str:
        """
        A helper function that returns the name of the constraint
        between the two given vertices. This is used to encode
        information in the name of the constraint so that a graph
        does not need to be present for the analysis later.
        The rules are as follows, assuming v1 is the predecessor of v2:
        - The format of the constraint name is "<v1>_<v2>", where
        v1 and v2 are the global indices of the vertices.
        """
        constr_name = f"{v1}_{v2}"

        return constr_name

    @measure_time("LP conversion")
    def convert_to_lp(self, verbose: bool = True,
                      pairwise_analysis: bool = False,
                      is_mpich: bool = False,
                      native_nic_serialize_sends: bool = False,
                      native_nic_send_mode: str = "same_ready_cohort",
                      native_nic_send_window: int = 1,
                      native_nic_resource_scope: str = "rank",
                      native_recv_nic_serialize_recvs: bool = False,
                      native_cpu_serialize_ops: bool = False,
                      native_cpu_mode: str = "full_timeline",
                      native_cpu_scope: str = "all_ops",
                      native_cpu_resource_scope: str = "rank",
                      rendezvous_hold_sender_nic: bool = False,
                      rendezvous_hold_sender_cpu: bool = False,
                      serialize_same_nic_sends: bool = False,
                      serialize_same_nic_recvs: bool = False,
                      serialize_node_shared_nic_sends: bool = False,
                      serialize_node_shared_nic_recvs: bool = False,
                      serialize_replay_comm_ops: bool = False,
                      serialize_same_cpu_ops: bool = False,
                      serialize_same_cpu_comm_ops: bool = False,
                      native_nic_debug_limit: int = 0,
                      msg_gap: float = 0.0,
                      G: Optional[float] = None,
                      ready_proxy_G: Optional[float] = None,
                      topology: Optional[NetTopology] = None,
                      unit: str = "ns",
                      ranks_per_node: int = 1) \
                    -> Union[gp.Model, pywraplp.Solver]:
        """
        Converts the dependency graph into a linear program as
        per the given topology and the parameters in the LogGP model.
        @param verbose: If True, will print out the progress and more
        information about the conversion.
        @param pairwise_analysis: If True, a new variable l will be created
        for each pair of ranks. It is only used when pairwise rank
        network latency sensitivity analysis is enabled.
        @param is_mpich: If True, then the LP model will be constructed
        as per the MPICH implementation. Otherwise, it will be constructed
        as per the ideal LogGP/LogGPS model. The difference is that for
        MPICH, only rendezvous messages can be overlapped by computation,
        when performing Isend. In the eager case, the send operation will
        always be blocking. In addition, the RNDV_RTS message will
        also be blocking.
        @param G: The bandwidth parameter G in the LogGP model. If None,
        then the bandwidth between ranks will be considered as a variable
        in the analysis. Otherwise, it will be considered as a constant value.
        @param ready_proxy_G: Optional nominal G used only for the static
        ready-time proxy that drives native resource ordering. This is useful
        when the actual LP keeps g symbolic for a bandwidth sweep but we still
        want the ordering heuristic to reflect a realistic operating point.
        @param topology: The network topology of the MPI program. If None,
        then the default topology will be used.
        @param unit: The unit of time to be used in the LP model. The default
        is nanoseconds. The other options are "us" for microseconds and "ms"
        for milliseconds.
        """
        scale = 1
        if unit == "us":
            scale = 1000
        elif unit == "ms":
            scale = 1000000
        if native_nic_serialize_sends and (serialize_same_nic_sends or serialize_node_shared_nic_sends):
            raise ValueError(
                "[ERROR] Native NIC serialization cannot be combined with legacy same-NIC send serialization."
            )
        valid_native_send_modes = {"same_ready_cohort", "full_timeline", "windowed_timeline", "credit_timeline"}
        if native_nic_send_mode not in valid_native_send_modes:
            raise ValueError(
                f"[ERROR] Unsupported native NIC send mode: {native_nic_send_mode}. "
                f"Expected one of {sorted(valid_native_send_modes)}."
            )
        if native_nic_send_window < 1:
            raise ValueError(
                f"[ERROR] Native NIC send window must be >= 1, got {native_nic_send_window}."
            )
        valid_native_nic_scopes = {"rank", "node_shared"}
        if native_nic_resource_scope not in valid_native_nic_scopes:
            raise ValueError(
                f"[ERROR] Unsupported native NIC resource scope: {native_nic_resource_scope}. "
                f"Expected one of {sorted(valid_native_nic_scopes)}."
            )
        valid_native_cpu_modes = {"full_timeline", "same_ready_cohort"}
        if native_cpu_mode not in valid_native_cpu_modes:
            raise ValueError(
                f"[ERROR] Unsupported native CPU mode: {native_cpu_mode}. "
                f"Expected one of {sorted(valid_native_cpu_modes)}."
            )
        valid_native_cpu_scopes = {"all_ops", "comm_only"}
        if native_cpu_scope not in valid_native_cpu_scopes:
            raise ValueError(
                f"[ERROR] Unsupported native CPU scope: {native_cpu_scope}. "
                f"Expected one of {sorted(valid_native_cpu_scopes)}."
            )
        valid_native_cpu_resource_scopes = {"rank", "node_shared"}
        if native_cpu_resource_scope not in valid_native_cpu_resource_scopes:
            raise ValueError(
                f"[ERROR] Unsupported native CPU resource scope: {native_cpu_resource_scope}. "
                f"Expected one of {sorted(valid_native_cpu_resource_scopes)}."
            )
        if native_recv_nic_serialize_recvs and (serialize_same_nic_recvs or serialize_node_shared_nic_recvs):
            raise ValueError(
                "[ERROR] Native recv-NIC serialization cannot be combined with legacy same-NIC recv serialization."
            )
        if native_cpu_serialize_ops and (serialize_same_cpu_ops or serialize_same_cpu_comm_ops):
            raise ValueError(
                "[ERROR] Native CPU serialization cannot be combined with legacy same-CPU serialization."
            )
        print(f"[INFO] Time unit: {unit}")
        o = self.o / scale
        msg_gap_scaled = msg_gap / scale
        is_loggps = self.dep_graph.is_loggps
        materialize_native_nic_release = native_nic_serialize_sends and native_nic_debug_limit > 0
        self.last_lp_build_metadata = {
            "native_nic_send": {
                "enabled": bool(native_nic_serialize_sends),
                "mode": native_nic_send_mode if native_nic_serialize_sends else "disabled",
                "window": int(native_nic_send_window) if native_nic_serialize_sends else 0,
                "materialized_release_vars": bool(materialize_native_nic_release),
                "rendezvous_hold_sender_nic": bool(rendezvous_hold_sender_nic),
                "resource_scope": (
                    "node_nic" if native_nic_resource_scope == "node_shared" else "rank_nic"
                ),
                "order_heuristic": (
                    "same_ready_cohort_then_trace_order_with_topological_safety"
                    if native_nic_send_mode == "same_ready_cohort"
                    else (
                        "credit_capacity_timeline_then_trace_order_with_topological_safety"
                        if native_nic_send_mode == "credit_timeline"
                        else (
                        "windowed_ready_proxy_timeline_then_trace_order_with_topological_safety"
                        if native_nic_send_mode == "windowed_timeline"
                        else "full_ready_proxy_timeline_then_trace_order_with_topological_safety"
                        )
                    )
                ),
                "num_groups": 0,
                "num_ready_cohorts": 0,
                "num_send_vertices": 0,
                "num_release_vars": 0,
                "num_order_edges": 0,
                "ready_proxy_G": (
                    float(ready_proxy_G)
                    if ready_proxy_G is not None
                    else (float(G) if G is not None else None)
                ),
                "detailed_groups_dumped": 0,
                "groups": [],
            },
            "native_recv_nic": {
                "enabled": bool(native_recv_nic_serialize_recvs),
                "mode": "native_ready_proxy_topo_full_chain" if native_recv_nic_serialize_recvs else "disabled",
                "resource_scope": "rank_nic",
                "num_groups": 0,
                "num_recv_vertices": 0,
                "num_order_edges": 0,
            },
            "native_cpu": {
                "enabled": bool(native_cpu_serialize_ops),
                "mode": native_cpu_mode if native_cpu_serialize_ops else "disabled",
                "scope": native_cpu_scope if native_cpu_serialize_ops else "disabled",
                "rendezvous_hold_sender_cpu": bool(rendezvous_hold_sender_cpu),
                "resource_scope": (
                    "node_cpu" if native_cpu_resource_scope == "node_shared" else "rank_cpu"
                ),
                "num_groups": 0,
                "num_vertices": 0,
                "num_order_edges": 0,
            },
            "legacy_same_nic_send": {
                "enabled": bool(serialize_same_nic_sends or serialize_node_shared_nic_sends),
                "node_shared": bool(serialize_node_shared_nic_sends),
                "num_groups": 0,
                "num_send_vertices": 0,
                "num_order_edges": 0,
            },
            "legacy_same_nic_recv": {
                "enabled": bool(serialize_same_nic_recvs or serialize_node_shared_nic_recvs),
                "node_shared": bool(serialize_node_shared_nic_recvs),
                "num_groups": 0,
                "num_recv_vertices": 0,
                "num_order_edges": 0,
            },
        }
        self.last_lp_debug_exprs = {
            "start": {},
            "finish": {},
            "nic_release": {},
            "cpu_release": {},
        }
        
        # Create the LP model
        num_ranks = self.dep_graph.num_ranks
        if self.use_gurobi:
            model = gp.Model("Dependency Graph LP Model")
        else:
            ortools_solver = os.environ.get("LLAMP_ORTOOLS_SOLVER", "CLP")
            model = pywraplp.Solver.CreateSolver(ortools_solver)
            if model is None:
                raise RuntimeError(f"[ERROR] OR-Tools solver '{ortools_solver}' is not available.")
        
        if verbose:
            print(f"[INFO] Number of ranks: {num_ranks}")
            print(f"[INFO] Pairwise analysis: {pairwise_analysis}")

        if G is not None and verbose:
            print(f"[INFO] Using constant bandwidth parameter G: {G}")

        # Adds the required variables to the model
        l, g, l_intra = self.__add_variables(model, num_ranks, pairwise_analysis, G, verbose, ranks_per_node)

        if G is not None:
            g /= scale
        
        end_v = self.dep_graph.end_v

        def create_new_var(var_index: int) \
            -> Union[gp.Var, pywraplp.Variable]:
            """
            A helper function that creates a new solver variable
            with the given index.
            @param var_index: The index of the variable.
            """
            if var_index == end_v:
                var_name = "t"
            else:
                var_name = f"y{var_index}"
            
            if self.use_gurobi:
                return model.addVar(name=var_name)
            else:
                return model.NumVar(0.0, model.infinity(), var_name)

        def create_aux_var(var_name: str) \
            -> Union[gp.Var, pywraplp.Variable]:
            if self.use_gurobi:
                return model.addVar(name=var_name)
            return model.NumVar(0.0, model.infinity(), var_name)
        
        def remove_var(var: Union[gp.Var, pywraplp.Variable]) -> None:
            """
            A helper function that removes the given variable from the model.
            """
            if self.use_gurobi:
                model.remove(var)
            else:
                # OR-Tools does not support removing variables from a built model.
                # Leaving an unused variable in place is harmless for the LP solve.
                return

        def get_vertex_cost(v: igraph.Vertex) \
            -> Tuple[Union[gp.LinExpr, float], float]:
            """
            A helper function that returns the cost of the given vertex
            either as a solver variable or a constant value based
            on the type of the vertex.
            Returns a tuple that contains the cost expressed as a
            linear expression and the cost evaluated with the initial L.
            """
            type = v["type"]
            cost = v["cost"] / scale
            if type == VertexType.SEND or type == VertexType.RECV:
                cost = o

            return cost


        def add_constr(constr: Union[gp.Constr, pywraplp.Constraint],
                       name: Optional[str] = None) -> None:
            """
            A helper function that adds a new constraint to the model
            """
            # print(f"[DEBUG] Added constraint {name}: {constr}")
            if self.use_gurobi:
                if name is not None:
                    model.addConstr(constr, name=name)
                else:
                    model.addConstr(constr)
            else:
                if name is not None:
                    model.Add(constr, name)
                else:
                    model.Add(constr)
        
        def get_l_g_vars(src_rank: int, dst_rank: int) -> Tuple:
            """
            A helper function that returns the l and g variables
            between the given source and destination ranks.
            Uses l_intra for same-node rank pairs when ranks_per_node > 1.
            """
            if pairwise_analysis:
                return l[src_rank, dst_rank], g[src_rank, dst_rank]
            if ranks_per_node > 1 and (src_rank // ranks_per_node == dst_rank // ranks_per_node):
                return l_intra, g
            return l, g

        def get_vertex_post_cost(v_obj: igraph.Vertex) -> float:
            type = v_obj["type"]
            if type == VertexType.CALC:
                return v_obj["cost"] / scale
            if type == VertexType.SEND or type == VertexType.RECV:
                return o
            if type == VertexType.MACRO:
                local_ns = (
                    v_obj["local_ns"] / scale
                    if "local_ns" in v_obj.attributes() and v_obj["local_ns"] is not None
                    else 0
                )
                post_o_coeff = v_obj["post_o_coeff"] if "post_o_coeff" in v_obj.attributes() else 0
                return post_o_coeff * o + local_ns
            return 0

        def get_nic_occupancy_expr(v_obj: igraph.Vertex):
            """
            Returns the traced NIC occupancy of a send-capable operation.
            This mirrors LogGOPSim's send-side nextgs update:
                nextgs = send_start + msg_gap + bytes * G
            and intentionally does not use CPU overhead o.
            """
            type = v_obj["type"]
            if type == VertexType.SEND:
                dst_rank = v_obj["dst_r"]
                src_rank = v_obj["r"]
                _, g_var = get_l_g_vars(src_rank, dst_rank)
                return msg_gap_scaled + (v_obj["cost"] * g_var)
            if type == VertexType.RECV:
                return msg_gap_scaled
            if type == VertexType.MACRO:
                if not ("emits_send" in v_obj.attributes() and v_obj["emits_send"]):
                    return 0
                bw_coeff = v_obj["bw_coeff"] if "bw_coeff" in v_obj.attributes() else 0
                if pairwise_analysis:
                    send_peer = None
                    if "ring_nexts" in v_obj.attributes() and v_obj["ring_nexts"]:
                        send_peer = v_obj["ring_nexts"][0]
                    elif "tree_parent" in v_obj.attributes() and v_obj["tree_parent"] is not None:
                        send_peer = v_obj["tree_parent"]
                    elif "tree_children" in v_obj.attributes() and v_obj["tree_children"]:
                        send_peer = v_obj["tree_children"][0]
                    if send_peer is None:
                        raise ValueError(
                            f"[ERROR] Cannot resolve send peer for macro vertex {v_obj.index} "
                            "under pairwise NIC serialization."
                        )
                    _, g_var = get_l_g_vars(v_obj["r"], send_peer)
                else:
                    g_var = g
                return msg_gap_scaled + (bw_coeff * g_var)
            return 0

        def get_cpu_occupancy_expr(v_obj: igraph.Vertex):
            """
            Returns the local CPU service window that should serialize on a
            traced per-CPU resource.

            The returned occupancy is measured from get_vertex_start_expr():
            - CALC: local calc duration
            - SEND: sender CPU overhead o
            - RECV: receiver CPU overhead o
              (the recv start proxy already lands after the recv NIC gap)
            - MACRO: local post-cost exposed by the compressed vertex
            """
            type = v_obj["type"]
            if type == VertexType.CALC:
                return v_obj["cost"] / scale
            if type == VertexType.SEND or type == VertexType.RECV:
                return o
            if type == VertexType.MACRO:
                return get_vertex_post_cost(v_obj)
            return 0

        def should_native_cpu_serialize_vertex(v_obj: igraph.Vertex) -> bool:
            if v_obj["r"] < 0:
                return False
            if "cpu" not in v_obj.attributes() or v_obj["cpu"] is None:
                return False
            if native_cpu_scope == "comm_only" and v_obj["type"] == VertexType.CALC:
                return False
            occupancy = get_cpu_occupancy_expr(v_obj)
            try:
                return float(occupancy) > 0.0
            except TypeError:
                return True

        def get_send_peer_rank(v_obj: igraph.Vertex) -> int | None:
            if v_obj["type"] == VertexType.SEND:
                return int(v_obj["dst_r"])
            if v_obj["type"] != VertexType.MACRO:
                return None
            if not ("emits_send" in v_obj.attributes() and v_obj["emits_send"]):
                return None
            if "ring_next_participant" in v_obj.attributes():
                peer = v_obj["ring_next_participant"]
                if peer is not None and int(peer) >= 0:
                    return int(peer)
            if "tree_parent_participant" in v_obj.attributes():
                peer = v_obj["tree_parent_participant"]
                if peer is not None and int(peer) >= 0:
                    return int(peer)
            if "tree_child_participant" in v_obj.attributes():
                peer = v_obj["tree_child_participant"]
                if peer is not None and int(peer) >= 0:
                    return int(peer)
            return None

        def is_rendezvous_send_vertex(v_obj: igraph.Vertex) -> bool:
            if not is_loggps or self.S is None:
                return False
            if v_obj["type"] == VertexType.SEND:
                effective_size = max(int(v_obj["cost"]), 1)
                return effective_size > self.S
            if v_obj["type"] == VertexType.MACRO:
                if not ("emits_send" in v_obj.attributes() and v_obj["emits_send"]):
                    return False
                bw_coeff = int(v_obj["bw_coeff"]) if "bw_coeff" in v_obj.attributes() else 0
                effective_size = max(bw_coeff, 1)
                return effective_size > self.S
            return False

        def is_network_visible_send(v_obj: igraph.Vertex) -> bool:
            """
            Same-NIC fairness should model the network interface, not local NVLink
            bookkeeping sends. For rank-per-GPU V2 traces, intra-node sends appear
            as 0-byte SEND operations and should not occupy the NIC queue.
            """
            if v_obj["type"] == VertexType.SEND:
                if int(v_obj["cost"]) <= 0:
                    return False
                if ranks_per_node <= 1:
                    return True
                peer = get_send_peer_rank(v_obj)
                if peer is None:
                    return True
                return (int(v_obj["r"]) // ranks_per_node) != (peer // ranks_per_node)

            if v_obj["type"] == VertexType.MACRO:
                if not ("emits_send" in v_obj.attributes() and v_obj["emits_send"]):
                    return False
                if "bw_coeff" in v_obj.attributes() and float(v_obj["bw_coeff"]) <= 0:
                    return False
                if ranks_per_node <= 1:
                    return True
                peer = get_send_peer_rank(v_obj)
                if peer is None:
                    return True
                rank = int(v_obj["r"])
                return (rank // ranks_per_node) != (peer // ranks_per_node)

            return False

        def get_vertex_start_expr(v_obj: igraph.Vertex):
            """
            Reconstructs an operation start timestamp from the exported
            finish-time expression. When an explicit communication start
            variable exists we use it directly; otherwise we back out the
            post-start local cost from the finish expression.
            """
            if v_obj.index in comm_vars:
                return comm_vars[v_obj.index]
            return var_map[v_obj.index] - get_vertex_post_cost(v_obj)

        def get_native_send_nic_release_expr(v_obj: igraph.Vertex):
            release = get_vertex_start_expr(v_obj) + get_nic_occupancy_expr(v_obj)
            if rendezvous_hold_sender_nic and is_rendezvous_send_vertex(v_obj):
                release = var_map[v_obj.index] - o
            return release

        def get_native_nic_send_credit(v_obj: igraph.Vertex) -> float:
            """
            Returns how much of the local NIC's bounded-outstanding capacity a
            send-capable vertex consumes.

            Raw SEND vertices remain fully serialized because each explicit
            micro-send occupies one full credit. Compressed MACRO send vertices
            represent an already-aggregated wave of sends, so they consume a
            fractional credit that allows a bounded number of macro waves to be
            outstanding without reverting to the overly weak baseline model.
            """
            if native_nic_send_mode != "credit_timeline":
                return 1.0
            if v_obj["type"] == VertexType.SEND:
                return 1.0
            if v_obj["type"] == VertexType.MACRO and "emits_send" in v_obj.attributes() and v_obj["emits_send"]:
                return 1.0 / float(native_nic_send_window)
            return 1.0

        def get_native_cpu_release_expr(v_obj: igraph.Vertex):
            release = get_vertex_start_expr(v_obj) + get_cpu_occupancy_expr(v_obj)
            if rendezvous_hold_sender_cpu and v_obj["type"] == VertexType.SEND and is_rendezvous_send_vertex(v_obj):
                release = var_map[v_obj.index]
            return release

        def serialization_sort_key(idx: int) -> tuple[int, int]:
            """
            Use a DAG-safe order for resource-serialization constraints.
            Some compressed graphs preserve a replay/trace_order tag that is
            not globally topologically consistent after macro lowering. If we
            serialize purely by trace_order, we can introduce cycles such as
            A -> ... -> B in the DAG plus a resource edge B -> A.
            Prioritize topological order and use trace_order only as a
            secondary tiebreaker when it is available.
            """
            v = self.dep_graph.graph.vs[idx]
            trace_order = (
                int(v["trace_order"])
                if "trace_order" in v.attributes() and v["trace_order"] is not None
                else topo_pos[idx]
            )
            return topo_pos[idx], trace_order

        def issue_order_sort_key(idx: int) -> tuple[int, int, int]:
            v = self.dep_graph.graph.vs[idx]
            trace_order = (
                int(v["trace_order"])
                if "trace_order" in v.attributes() and v["trace_order"] is not None
                else int(v["l"]) if "l" in v.attributes() and v["l"] is not None else topo_pos[idx]
            )
            local_label = int(v["l"]) if "l" in v.attributes() and v["l"] is not None else topo_pos[idx]
            return trace_order, local_label, idx

        def get_nic_group_id(v_obj: igraph.Vertex, node_shared: bool) -> int:
            rank = int(v_obj["r"])
            if not node_shared or ranks_per_node <= 1:
                return rank
            return rank // ranks_per_node

        def compute_proxy_timestamps() -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
            """
            Computes a deterministic lower-bound timing proxy from LLAMP graph
            semantics only. The proxy is intentionally simple: it ignores any
            external replay order and only uses the DAG plus fixed operation
            costs. It is used to bias the native NIC order while the final
            ordering remains topologically safe.
            """
            proxy_g_value = ready_proxy_G if ready_proxy_G is not None else G
            nominal_g = (float(proxy_g_value) / scale) if proxy_g_value is not None else 0.0
            ready_proxy: Dict[int, float] = {}
            finish_proxy: Dict[int, float] = {}
            comm_start_proxy: Dict[int, float] = {}

            def preds_finish_max(indices: List[int]) -> float:
                if not indices:
                    return 0.0
                return max(finish_proxy[idx] for idx in indices)

            for idx in vs:
                v_obj = self.dep_graph.graph.vs[idx]
                preds = self.dep_graph.get_predecessors(idx)
                if not preds:
                    ready_proxy[idx] = 0.0
                    finish_proxy[idx] = 0.0
                    continue

                v_type = v_obj["type"]
                if v_type == VertexType.SEND:
                    real_preds = []
                    for pred in preds:
                        edge = self.dep_graph.get_edge(pred, idx)
                        assert edge is not None
                        if ("v" not in edge.attributes()) or (not edge["v"]):
                            real_preds.append(pred)
                    ready = preds_finish_max(real_preds)
                    ready_proxy[idx] = ready
                    finish_proxy[idx] = ready + o
                    comm_start_proxy[idx] = ready
                    continue

                if v_type == VertexType.RECV:
                    src_idx = v_obj["src_idx"]
                    local_preds = [pred for pred in preds if pred != src_idx]
                    remote_ready = comm_start_proxy.get(src_idx, finish_proxy.get(src_idx, 0.0))
                    remote_ready += o + (v_obj["cost"] * nominal_g)
                    ready = max(preds_finish_max(local_preds), remote_ready)
                    ready_proxy[idx] = ready
                    finish_proxy[idx] = ready + msg_gap_scaled + o
                    continue

                if v_type == VertexType.MACRO:
                    if "transfer_src_indices" in v_obj.attributes() and v_obj["transfer_src_indices"]:
                        finish = 0.0
                        transfer_srcs = list(v_obj["transfer_src_indices"])
                        transfer_bw_coeffs = list(v_obj["transfer_bw_coeffs"])
                        transfer_o_coeffs = list(v_obj["transfer_o_coeffs"])
                        transfer_const_ns = list(v_obj["transfer_const_ns"])
                        for src_idx, bw_coeff, o_coeff, const_ns in zip(
                            transfer_srcs,
                            transfer_bw_coeffs,
                            transfer_o_coeffs,
                            transfer_const_ns,
                        ):
                            finish = max(
                                finish,
                                finish_proxy[src_idx] + (bw_coeff * nominal_g) + (o_coeff * o) + (const_ns / scale),
                            )
                        ready_proxy[idx] = finish
                        finish_proxy[idx] = finish
                        continue

                    local_pred = v_obj["loc_idx"] if "loc_idx" in v_obj.attributes() else None
                    ready = finish_proxy[local_pred] if local_pred is not None else 0.0
                    remote_srcs: List[int] = []
                    if "src_indices" in v_obj.attributes():
                        existing = v_obj["src_indices"]
                        remote_srcs = list(existing) if existing is not None else []
                    elif "src_idx" in v_obj.attributes() and v_obj["src_idx"] is not None:
                        remote_srcs = [v_obj["src_idx"]]
                    bw_coeff = v_obj["bw_coeff"] if "bw_coeff" in v_obj.attributes() else 0
                    for src_idx in remote_srcs:
                        ready = max(ready, finish_proxy[src_idx] + (bw_coeff * nominal_g))
                    ready_proxy[idx] = ready
                    finish_proxy[idx] = ready + get_vertex_post_cost(v_obj)
                    if "emits_send" in v_obj.attributes() and v_obj["emits_send"]:
                        comm_start_proxy[idx] = ready
                    continue

                ready = preds_finish_max(preds)
                ready_proxy[idx] = ready
                finish_proxy[idx] = ready + get_vertex_cost(v_obj)

            return ready_proxy, finish_proxy, comm_start_proxy

        def build_priority_topological_positions(ready_proxy: Dict[int, float]) -> Dict[int, int]:
            """
            Builds a deterministic topological order biased by the static
            ready-time proxy. This preserves DAG safety while allowing
            same-resource sends that are simultaneously available to be
            ordered by LLAMP-native readiness semantics instead of a replayed
            external schedule.
            """
            indegrees = self.dep_graph.graph.indegree()
            heap: List[tuple[float, tuple[int, int, int], int]] = []
            for idx, indegree in enumerate(indegrees):
                if indegree == 0:
                    heapq.heappush(heap, (ready_proxy.get(idx, 0.0), issue_order_sort_key(idx), idx))

            order: List[int] = []
            while heap:
                _, _, idx = heapq.heappop(heap)
                order.append(idx)
                for succ in self.dep_graph.get_successors(idx):
                    indegrees[succ] -= 1
                    if indegrees[succ] == 0:
                        heapq.heappush(
                            heap,
                            (ready_proxy.get(succ, 0.0), issue_order_sort_key(succ), succ),
                        )

            if len(order) != self.dep_graph.graph.vcount():
                raise RuntimeError("[ERROR] Failed to build native NIC priority topological order.")
            return {idx: pos for pos, idx in enumerate(order)}

        def add_same_nic_send_constraints() -> None:
            """
            Serializes sends that share the same traced NIC in trace order.
            This is required for fair V1 comparisons against LogGOPSim.
            """
            replay_pairs = []
            use_replay_pairs = not serialize_node_shared_nic_sends
            if use_replay_pairs and "replay_nic_pairs" in self.dep_graph.graph.attributes():
                replay_pairs = list(self.dep_graph.graph["replay_nic_pairs"])
            if replay_pairs:
                seen = set()
                for prev_idx, next_idx in replay_pairs:
                    if (prev_idx, next_idx) in seen:
                        continue
                    seen.add((prev_idx, next_idx))
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    next_v = self.dep_graph.graph.vs[next_idx]
                    if prev_v["r"] != next_v["r"]:
                        continue
                    prev_is_send_like = (
                        prev_v["type"] == VertexType.SEND
                        or (
                            prev_v["type"] == VertexType.MACRO
                            and "emits_send" in prev_v.attributes()
                            and prev_v["emits_send"]
                        )
                    )
                    next_is_send_like = (
                        next_v["type"] == VertexType.SEND
                        or (
                            next_v["type"] == VertexType.MACRO
                            and "emits_send" in next_v.attributes()
                            and next_v["emits_send"]
                        )
                    )
                    if not prev_is_send_like or not next_is_send_like:
                        continue
                    if not is_network_visible_send(prev_v) or not is_network_visible_send(next_v):
                        continue
                    if "nic" in prev_v.attributes() and "nic" in next_v.attributes():
                        if prev_v["nic"] != next_v["nic"]:
                            continue
                    constr_name = f"nic_{prev_idx}_{next_idx}"
                    prev_start = get_vertex_start_expr(prev_v)
                    next_start = get_vertex_start_expr(next_v)
                    add_constr(
                        next_start >= prev_start + get_nic_occupancy_expr(prev_v),
                        constr_name,
                    )
                return

            nic_sends = {}
            for v in self.dep_graph.graph.vs:
                if v["type"] == VertexType.SEND:
                    if not is_network_visible_send(v):
                        continue
                elif v["type"] == VertexType.MACRO and "emits_send" in v.attributes() and v["emits_send"]:
                    if not is_network_visible_send(v):
                        continue
                else:
                    continue
                nic = v["nic"] if "nic" in v.attributes() else 0
                key = (get_nic_group_id(v, serialize_node_shared_nic_sends), nic)
                nic_sends.setdefault(key, []).append(v.index)

            legacy_stats = self.last_lp_build_metadata["legacy_same_nic_send"]
            legacy_stats["num_groups"] = len(nic_sends)
            legacy_stats["num_send_vertices"] = sum(len(vertices) for vertices in nic_sends.values())

            for send_vertices in nic_sends.values():
                send_vertices.sort(key=serialization_sort_key)
                for prev_idx, next_idx in zip(send_vertices, send_vertices[1:]):
                    constr_name = f"nic_{prev_idx}_{next_idx}"
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    prev_start = get_vertex_start_expr(prev_v)
                    next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                    add_constr(
                        next_start >= prev_start + get_nic_occupancy_expr(prev_v),
                        constr_name,
                    )
                    legacy_stats["num_order_edges"] += 1

        def add_same_nic_recv_constraints() -> None:
            replay_pairs = []
            use_replay_pairs = not serialize_node_shared_nic_recvs
            if use_replay_pairs and "replay_nic_pairs" in self.dep_graph.graph.attributes():
                replay_pairs = list(self.dep_graph.graph["replay_nic_pairs"])
            if replay_pairs:
                seen = set()
                for prev_idx, next_idx in replay_pairs:
                    if (prev_idx, next_idx) in seen:
                        continue
                    seen.add((prev_idx, next_idx))
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    next_v = self.dep_graph.graph.vs[next_idx]
                    if prev_v["r"] != next_v["r"]:
                        continue
                    if prev_v["type"] != VertexType.RECV or next_v["type"] != VertexType.RECV:
                        continue
                    if "nic" in prev_v.attributes() and "nic" in next_v.attributes():
                        if prev_v["nic"] != next_v["nic"]:
                            continue
                    constr_name = f"recv_nic_{prev_idx}_{next_idx}"
                    prev_start = get_vertex_start_expr(prev_v)
                    next_start = get_vertex_start_expr(next_v)
                    add_constr(
                        next_start >= prev_start + get_nic_occupancy_expr(prev_v),
                        constr_name,
                    )
                return

            nic_recvs = {}
            for v in self.dep_graph.graph.vs:
                if v["type"] != VertexType.RECV:
                    continue
                nic = v["nic"] if "nic" in v.attributes() else 0
                key = (get_nic_group_id(v, serialize_node_shared_nic_recvs), nic)
                nic_recvs.setdefault(key, []).append(v.index)

            legacy_stats = self.last_lp_build_metadata["legacy_same_nic_recv"]
            legacy_stats["num_groups"] = len(nic_recvs)
            legacy_stats["num_recv_vertices"] = sum(len(vertices) for vertices in nic_recvs.values())

            for recv_vertices in nic_recvs.values():
                recv_vertices.sort(key=serialization_sort_key)
                for prev_idx, next_idx in zip(recv_vertices, recv_vertices[1:]):
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    prev_start = get_vertex_start_expr(prev_v)
                    next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                    add_constr(
                        next_start >= prev_start + get_nic_occupancy_expr(prev_v),
                        f"recv_nic_{prev_idx}_{next_idx}",
                    )
                    legacy_stats["num_order_edges"] += 1

        def add_same_cpu_constraints(comm_only: bool = False) -> None:
            """
            Serializes operations that share the same traced CPU in trace order.
            This is required for fair node-aggregated V1 comparisons against
            LogGOPSim, which honors per-CPU resource assignments in GOAL.
            """
            replay_pairs = []
            if comm_only and "replay_nic_pairs" in self.dep_graph.graph.attributes():
                replay_pairs = list(self.dep_graph.graph["replay_nic_pairs"])
            allowed_comm_types = {VertexType.SEND, VertexType.RECV, VertexType.MACRO}
            if replay_pairs:
                seen = set()
                for prev_idx, next_idx in replay_pairs:
                    if (prev_idx, next_idx) in seen:
                        continue
                    seen.add((prev_idx, next_idx))
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    next_v = self.dep_graph.graph.vs[next_idx]
                    if prev_v["r"] != next_v["r"]:
                        continue
                    if comm_only:
                        if prev_v["type"] not in allowed_comm_types:
                            continue
                        if next_v["type"] not in allowed_comm_types:
                            continue
                    if "cpu" not in prev_v.attributes() or "cpu" not in next_v.attributes():
                        continue
                    if prev_v["cpu"] != next_v["cpu"]:
                        continue
                    prev_finish = var_map[prev_idx]
                    next_start = get_vertex_start_expr(next_v)
                    constr_name = f"cpu_{prev_idx}_{next_idx}"
                    add_constr(next_start >= prev_finish, constr_name)
                return

            cpu_ops = {}
            for v in self.dep_graph.graph.vs:
                if v["r"] < 0:
                    continue
                if "cpu" not in v.attributes() or v["cpu"] is None:
                    continue
                if comm_only and v["type"] not in allowed_comm_types:
                    continue
                key = (v["r"], int(v["cpu"]))
                cpu_ops.setdefault(key, []).append(v.index)

            for op_vertices in cpu_ops.values():
                op_vertices.sort(key=serialization_sort_key)
                for prev_idx, next_idx in zip(op_vertices, op_vertices[1:]):
                    prev_finish = var_map[prev_idx]
                    next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                    constr_name = f"cpu_{prev_idx}_{next_idx}"
                    add_constr(next_start >= prev_finish, constr_name)

        def add_replay_precedence_constraints() -> None:
            """
            Applies exact replay/order relations that should constrain the LP
            without becoming structural DAG edges. This is the generic path
            for richer V2 sidecars that mix message matches with additional
            schedule-order semantics.
            """
            replay_pairs = []
            if "replay_prec_pairs" in self.dep_graph.graph.attributes():
                replay_pairs = list(self.dep_graph.graph["replay_prec_pairs"])
            if not replay_pairs:
                return

            seen = set()
            for prev_idx, next_idx in replay_pairs:
                if (prev_idx, next_idx) in seen:
                    continue
                seen.add((prev_idx, next_idx))
                prev_finish = var_map[prev_idx]
                next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                add_constr(
                    next_start >= prev_finish,
                    f"replay_{prev_idx}_{next_idx}",
                )

        def add_replay_comm_constraints() -> None:
            """
            Serializes same-rank communication operations in exact replay order
            using a communication-engine occupancy view rather than a plain
            precedence edge. This is broader than same-NIC ordering and is
            intended for richer V2 traces where local stream/engine pressure is
            not fully exposed by the raw DAG.
            """
            replay_pairs = []
            if "replay_nic_pairs" in self.dep_graph.graph.attributes():
                replay_pairs = list(self.dep_graph.graph["replay_nic_pairs"])
            if not replay_pairs:
                return

            seen = set()
            allowed = {VertexType.SEND, VertexType.RECV, VertexType.MACRO}
            for prev_idx, next_idx in replay_pairs:
                if (prev_idx, next_idx) in seen:
                    continue
                seen.add((prev_idx, next_idx))
                prev_v = self.dep_graph.graph.vs[prev_idx]
                next_v = self.dep_graph.graph.vs[next_idx]
                if prev_v["r"] != next_v["r"]:
                    continue
                if prev_v["type"] not in allowed or next_v["type"] not in allowed:
                    continue
                prev_start = get_vertex_start_expr(prev_v)
                next_start = get_vertex_start_expr(next_v)
                occupancy = get_nic_occupancy_expr(prev_v)
                if prev_v["type"] == VertexType.RECV:
                    occupancy += o
                add_constr(
                    next_start >= prev_start + occupancy,
                    f"replay_comm_{prev_idx}_{next_idx}",
                )
        
        # Obtains topological ordering of all the vertices
        vs = self.dep_graph.get_topological_sort(mode="out")
        topo_pos = {idx: order for order, idx in enumerate(vs)}
        ready_proxy, finish_proxy, comm_start_proxy = compute_proxy_timestamps()
        native_priority_topo_pos = build_priority_topological_positions(ready_proxy)
        # vs = self.dep_graph.bfs()

        # A dictionary that maps the global index of each
        # vertex to either a linear expression or a constant value.
        # Conceptually, each entry represents the timestamp
        # at which each vertex finishes execution.
        var_map = {}

        # A dictionary that maps the global index of SEND vertex
        # to a solver variable that denotes 
        comm_vars = {}
        nic_release_vars = {}
        rendezvous_end_vars = {}

        def ensure_native_nic_release_var(v_obj: igraph.Vertex):
            if not native_nic_serialize_sends or not materialize_native_nic_release:
                return None
            v = v_obj.index
            if v not in comm_vars:
                raise RuntimeError(
                    f"[ERROR] Native NIC serialization expected a communication start variable for vertex {v}."
                )
            if v in nic_release_vars:
                return nic_release_vars[v]
            release_var = create_aux_var(f"nic_rel_{v}")
            nic_release_vars[v] = release_var
            add_constr(
                release_var >= get_native_send_nic_release_expr(v_obj),
                f"nicrelease_{v}",
            )
            self.last_lp_build_metadata["native_nic_send"]["num_release_vars"] += 1
            return release_var

        def add_native_nic_send_constraints() -> None:
            native_stats = self.last_lp_build_metadata["native_nic_send"]
            nic_sends = {}
            use_node_shared_scope = native_nic_resource_scope == "node_shared"
            for v in self.dep_graph.graph.vs:
                if not is_network_visible_send(v):
                    continue
                if v["type"] == VertexType.SEND:
                    pass
                elif v["type"] == VertexType.MACRO and "emits_send" in v.attributes() and v["emits_send"]:
                    pass
                else:
                    continue
                nic = v["nic"] if "nic" in v.attributes() else 0
                key = (get_nic_group_id(v, use_node_shared_scope), int(nic))
                nic_sends.setdefault(key, []).append(v.index)

            native_stats["num_groups"] = len(nic_sends)
            native_stats["num_send_vertices"] = sum(len(vertices) for vertices in nic_sends.values())

            dumped_groups = 0
            for (resource_id, nic), send_vertices in sorted(nic_sends.items()):
                send_vertices.sort(
                    key=lambda idx: (
                        ready_proxy.get(idx, 0.0),
                        native_priority_topo_pos[idx],
                        issue_order_sort_key(idx),
                    )
                )
                cohorts: List[List[int]] = []
                curr: List[int] = []
                curr_key: Optional[float] = None
                for idx in send_vertices:
                    key = round(float(ready_proxy.get(idx, 0.0)), 9)
                    if curr_key is None or key == curr_key:
                        curr.append(idx)
                        curr_key = key
                        continue
                    cohorts.append(curr)
                    curr = [idx]
                    curr_key = key
                if curr:
                    cohorts.append(curr)
                native_stats["num_ready_cohorts"] += len(cohorts)

                if materialize_native_nic_release:
                    for cohort in cohorts:
                        for idx in cohort:
                            ensure_native_nic_release_var(self.dep_graph.graph.vs[idx])
                if native_nic_debug_limit > 0 and dumped_groups < native_nic_debug_limit:
                    native_stats["groups"].append(
                        {
                            "resource_id": int(resource_id),
                            "rank": int(resource_id) if not use_node_shared_scope else None,
                            "node": int(resource_id) if use_node_shared_scope else None,
                            "nic": int(nic),
                            "mode": native_nic_send_mode,
                            "window": int(native_nic_send_window),
                            "resource_scope": native_nic_resource_scope,
                            "ordered_send_vertices": [int(idx) for idx in send_vertices],
                            "credits": [
                                float(get_native_nic_send_credit(self.dep_graph.graph.vs[idx]))
                                for idx in send_vertices
                            ],
                            "cohorts": [
                                {
                                    "ready_proxy": float(ready_proxy.get(cohort[0], 0.0)),
                                    "send_vertices": [int(idx) for idx in cohort],
                                    "trace_order": [
                                        int(self.dep_graph.graph.vs[idx]["trace_order"])
                                        if "trace_order" in self.dep_graph.graph.vs[idx].attributes()
                                        and self.dep_graph.graph.vs[idx]["trace_order"] is not None
                                        else None
                                        for idx in cohort
                                    ],
                                }
                                for cohort in cohorts
                            ],
                        }
                    )
                    dumped_groups += 1
                ordered_sequences: List[List[int]]
                if native_nic_send_mode == "same_ready_cohort":
                    ordered_sequences = cohorts
                else:
                    ordered_sequences = [send_vertices]
                for sequence in ordered_sequences:
                    sequence.sort(
                        key=lambda idx: (
                            native_priority_topo_pos[idx],
                            issue_order_sort_key(idx),
                        )
                    )
                    if native_nic_send_mode == "windowed_timeline":
                        paired_indices = zip(sequence, sequence[native_nic_send_window:])
                    elif native_nic_send_mode != "credit_timeline":
                        paired_indices = zip(sequence, sequence[1:])
                    else:
                        paired_indices = []
                        start_order_pairs = list(zip(sequence, sequence[1:]))
                        left = 0
                        outstanding_credits = 0.0
                        for curr_pos, curr_idx in enumerate(sequence):
                            curr_credit = float(get_native_nic_send_credit(self.dep_graph.graph.vs[curr_idx]))
                            while left < curr_pos and (outstanding_credits + curr_credit) > (1.0 + 1e-12):
                                paired_indices.append((sequence[left], curr_idx))
                                outstanding_credits -= float(
                                    get_native_nic_send_credit(self.dep_graph.graph.vs[sequence[left]])
                                )
                                left += 1
                            outstanding_credits += curr_credit
                        for prev_idx, next_idx in start_order_pairs:
                            add_constr(
                                get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                                >= get_vertex_start_expr(self.dep_graph.graph.vs[prev_idx]),
                                f"nicnativeorder_{prev_idx}_{next_idx}",
                            )
                            native_stats["num_order_edges"] += 1
                    for prev_idx, next_idx in paired_indices:
                        prev_v = self.dep_graph.graph.vs[prev_idx]
                        next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                        if materialize_native_nic_release:
                            add_constr(
                                next_start >= nic_release_vars[prev_idx],
                                f"nicnative_{prev_idx}_{next_idx}",
                            )
                        else:
                            add_constr(
                                next_start >= get_native_send_nic_release_expr(prev_v),
                                f"nicnative_{prev_idx}_{next_idx}",
                            )
                        native_stats["num_order_edges"] += 1
            native_stats["detailed_groups_dumped"] = dumped_groups

        def add_native_recv_nic_constraints() -> None:
            native_stats = self.last_lp_build_metadata["native_recv_nic"]
            nic_recvs = {}
            for v in self.dep_graph.graph.vs:
                if v["type"] != VertexType.RECV:
                    continue
                nic = v["nic"] if "nic" in v.attributes() else 0
                key = (int(v["r"]), int(nic))
                nic_recvs.setdefault(key, []).append(v.index)

            native_stats["num_groups"] = len(nic_recvs)
            native_stats["num_recv_vertices"] = sum(len(vertices) for vertices in nic_recvs.values())

            for recv_vertices in nic_recvs.values():
                recv_vertices.sort(
                    key=lambda idx: (
                        ready_proxy.get(idx, 0.0),
                        native_priority_topo_pos[idx],
                        issue_order_sort_key(idx),
                    )
                )
                for prev_idx, next_idx in zip(recv_vertices, recv_vertices[1:]):
                    prev_v = self.dep_graph.graph.vs[prev_idx]
                    prev_start = get_vertex_start_expr(prev_v)
                    next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                    add_constr(
                        next_start >= prev_start + get_nic_occupancy_expr(prev_v),
                        f"recvnative_{prev_idx}_{next_idx}",
                    )
                    native_stats["num_order_edges"] += 1

        def add_native_cpu_constraints() -> None:
            native_stats = self.last_lp_build_metadata["native_cpu"]
            cpu_ops = {}
            use_node_shared_scope = native_cpu_resource_scope == "node_shared"
            for v in self.dep_graph.graph.vs:
                if not should_native_cpu_serialize_vertex(v):
                    continue
                key = (get_nic_group_id(v, use_node_shared_scope), int(v["cpu"]))
                cpu_ops.setdefault(key, []).append(v.index)

            native_stats["num_groups"] = len(cpu_ops)
            native_stats["num_vertices"] = sum(len(vertices) for vertices in cpu_ops.values())

            for op_vertices in cpu_ops.values():
                op_vertices.sort(
                    key=lambda idx: (
                        ready_proxy.get(idx, 0.0),
                        native_priority_topo_pos[idx],
                        issue_order_sort_key(idx),
                    )
                )
                sequences: List[List[int]]
                if native_cpu_mode == "same_ready_cohort":
                    cohorts: List[List[int]] = []
                    curr: List[int] = []
                    curr_key: Optional[float] = None
                    for idx in op_vertices:
                        key = round(float(ready_proxy.get(idx, 0.0)), 9)
                        if curr_key is None or key == curr_key:
                            curr.append(idx)
                            curr_key = key
                            continue
                        cohorts.append(curr)
                        curr = [idx]
                        curr_key = key
                    if curr:
                        cohorts.append(curr)
                    sequences = cohorts
                else:
                    sequences = [op_vertices]
                for sequence in sequences:
                    for prev_idx, next_idx in zip(sequence, sequence[1:]):
                        prev_v = self.dep_graph.graph.vs[prev_idx]
                        next_start = get_vertex_start_expr(self.dep_graph.graph.vs[next_idx])
                        add_constr(
                            next_start >= get_native_cpu_release_expr(prev_v),
                            f"cpunative_{prev_idx}_{next_idx}",
                        )
                        native_stats["num_order_edges"] += 1


        def compute_parallel_calc_time(v: int, dep: int, pred: int) -> None:
            """
            A helper function that computes the return time of a CALC
            vertex when it has an irequire dependency on another vertex.
            """
            dep_obj = self.dep_graph.graph.vs[dep]
            assert dep_obj["type"] == VertexType.SEND or \
                dep_obj["type"] == VertexType.RECV, \
                "The irequire dependency of a vertex must be a SEND or RECV"

            cost = get_vertex_cost(self.dep_graph.graph.vs[v])

            msg_size = dep_obj["cost"]
            is_rndv = msg_size > self.S and is_loggps
            
            # cpu_overhead = self.__get_msg_cpu_overhead(msg_size)
            cpu_overhead = 0
            async_overhead = 5000 / scale

            if dep_obj["type"] == VertexType.RECV:
                # If the dependency is a RECV vertex
                if is_rndv and is_mpich:
                    # Retrieves the send vertex that the recv vertex depends on
                    src_idx = dep_obj["src_idx"]
                    # If s_var does not exist, then create a new one
                    if src_idx not in comm_vars:
                        s_var = create_new_var(src_idx)
                        comm_vars[src_idx] = s_var
                    else:
                        s_var = comm_vars[src_idx]
                    
                    var_map[v] = s_var + cost + async_overhead + cpu_overhead
                else:
                    var_map[v] = var_map[pred] + cost + async_overhead + cpu_overhead
                
                return
            
            # If the dependency is a SEND vertex
            var_map[v] = var_map[pred] + cost + async_overhead + cpu_overhead

        def compute_send_time(v_obj: igraph.Vertex, preds: List[int]) -> None:
            """
            A helper function that computes the return time of a
            send vertex based on its predecessors.
            """
            v = v_obj.index
            msg_size = v_obj["cost"]

            # cpu_overhead = self.__get_msg_cpu_overhead(msg_size)
            cpu_overhead = 0

            # Checks whether the rendezvous protocol is triggered
            is_rndv = msg_size > self.S and is_loggps
            dst_rank = v_obj["dst_r"]
            src_rank = v_obj["r"]
            l_var, g_var = get_l_g_vars(src_rank, dst_rank)

            if topology:
                # If the network topology is provided
                num_links, switch_lat = topology.get_cost(src_rank, dst_rank)
                switch_lat /= scale
                l_var = (num_links * l_var + switch_lat)

            # Creates a new variable s_var if it does not already exist
            if v not in comm_vars:
                s_var = create_new_var(v)
                comm_vars[v] = s_var
            else:
                s_var = comm_vars[v]

            # Creates constraints for s_var
            # The local comp vertex on which the send vertex depends
            for pred in preds:
                edge = self.dep_graph.get_edge(pred, v)
                assert edge is not None
                # A new constraint is added for s_var under the following conditions:
                # - If the edge is a virtual edge and RNDV is triggered
                # - If the edge is not a virtual edge
                if ("v" not in edge.attributes()) or (not edge["v"]):
                    constr_name = self.__get_constr_name(pred, v)
                    rhs = var_map[pred]
                    add_constr(s_var >= rhs, constr_name)
            # Computes when the sender operation will actually finish
            if is_rndv:
                end = create_new_var(v)
                rendezvous_end_vars[v] = end
                add_constr(end >= s_var + o + cpu_overhead, self.__get_constr_name(v, v))
            else:
                # If RNDV is not triggered
                # if is_mpich:
                #     end = s_var + self.o + l_var + (msg_size * g_var) + wireup_cost
                # else:
                # The send operation returns after the message is sent
                # comm_vars[v] = s_var + self.o
                # end = s_var + self.o + wireup_cost
                end = s_var + o + cpu_overhead
                
            var_map[v] = end
        

        def compute_recv_time(v_obj: igraph.Vertex, preds: List[int]) -> None:
            """
            A helper function that computes the return time of a
            recv vertex based on its predecessors.
            """
            v = v_obj.index
            msg_size = v_obj["cost"]

            # cpu_overhead = self.__get_msg_cpu_overhead(msg_size)
            cpu_overhead = 0

            # Checks whether the rendezvous protocol is triggered
            is_rndv = msg_size > self.S and is_loggps
            dst_rank = v_obj["r"]
            src_rank = v_obj["src_r"]
            l_var, g_var = get_l_g_vars(src_rank, dst_rank)
            
            # FIXME Code duplication
            if topology:
                # If the network topology is provided
                num_links, switch_lat = topology.get_cost(src_rank, dst_rank)
                switch_lat /= scale
                l_var = (num_links * l_var + switch_lat)

            # Retrieves the global index of the remote send vertex
            src_idx = v_obj['src_idx']
            local_preds = [pred for pred in preds if pred != src_idx]
            assert local_preds, f"[ERROR] RECV vertex {v} has no local predecessor."
            # Creates a new variable s_var
            s_var = create_new_var(v)
            # Creates constraints for local predecessors. Replay-order inputs
            # may legitimately introduce additional same-rank recv->recv edges.
            for loc_idx in local_preds:
                constr_name = self.__get_constr_name(loc_idx, v)
                add_constr(s_var >= var_map[loc_idx], constr_name)
            # Creates a constraint for the remote dependency.
            # LogGOPSim releases the receive-side CPU after the sender's local
            # o, network latency/bandwidth, then a receive-side NIC gap.
            constr_name = self.__get_constr_name(src_idx, v)
            constr = s_var >= comm_vars[src_idx] + o + l_var + (msg_size * g_var)
            add_constr(constr, constr_name)
            var_map[v] = s_var + msg_gap_scaled + o + cpu_overhead
            if is_rndv and src_idx in rendezvous_end_vars:
                add_constr(
                    rendezvous_end_vars[src_idx] >= var_map[v],
                    self.__get_constr_name(v, src_idx),
                )

        def compute_macro_time(v_obj: igraph.Vertex, preds: List[int]) -> None:
            """
            Computes the finish time of a compact collective stage vertex.
            The stage either has:
            - only a local predecessor (send-only stage), or
            - a local predecessor plus one or more remote predecessors.
            """
            v = v_obj.index
            if "transfer_src_indices" in v_obj.attributes() and v_obj["transfer_src_indices"]:
                transfer_srcs = list(v_obj["transfer_src_indices"])
                transfer_lat_coeffs = list(v_obj["transfer_lat_coeffs"])
                transfer_lat_intra_coeffs = (
                    list(v_obj["transfer_lat_intra_coeffs"])
                    if "transfer_lat_intra_coeffs" in v_obj.attributes() and v_obj["transfer_lat_intra_coeffs"]
                    else [0] * len(transfer_srcs)
                )
                transfer_bw_coeffs = list(v_obj["transfer_bw_coeffs"])
                transfer_o_coeffs = list(v_obj["transfer_o_coeffs"])
                transfer_const_ns = list(v_obj["transfer_const_ns"])
                if not (
                    len(transfer_srcs)
                    == len(transfer_lat_coeffs)
                    == len(transfer_bw_coeffs)
                    == len(transfer_o_coeffs)
                    == len(transfer_const_ns)
                ):
                    raise ValueError(
                        f"[ERROR] Transfer macro vertex {v} has inconsistent source coefficient metadata."
                    )
                finish_var = create_new_var(v)
                dst_rank = v_obj["r"]
                for src_idx, lat_coeff, lat_intra_coeff, bw_coeff, o_coeff, const_ns in zip(
                    transfer_srcs,
                    transfer_lat_coeffs,
                    transfer_lat_intra_coeffs,
                    transfer_bw_coeffs,
                    transfer_o_coeffs,
                    transfer_const_ns,
                ):
                    src_rank = self.dep_graph.graph.vs[src_idx]["r"]
                    _, g_var = get_l_g_vars(src_rank, dst_rank)
                    # For transfer constraints, lat_coeff/lat_intra_coeff already
                    # encode inter vs intra hop counts, so use l/l_intra directly
                    # rather than relying on get_l_g_vars rank pairing.
                    lat_cost = lat_coeff * l + lat_intra_coeff * l_intra
                    if topology:
                        num_links, switch_lat = topology.get_cost(src_rank, dst_rank)
                        switch_lat /= scale
                        lat_cost = num_links * l + switch_lat
                    constr_name = self.__get_constr_name(src_idx, v)
                    add_constr(
                        finish_var
                        >= var_map[src_idx]
                        + lat_cost
                        + (bw_coeff * g_var)
                        + (o_coeff * o)
                        + (const_ns / scale),
                        constr_name,
                    )
                var_map[v] = finish_var
                return

            local_ns = v_obj["local_ns"] / scale if "local_ns" in v_obj.attributes() else 0
            post_o_coeff = v_obj["post_o_coeff"] if "post_o_coeff" in v_obj.attributes() else 0
            post_cost = post_o_coeff * o + local_ns

            remote_srcs: List[int] = []
            if "src_indices" in v_obj.attributes():
                existing = v_obj["src_indices"]
                remote_srcs = list(existing) if existing is not None else []
            elif "src_idx" in v_obj.attributes():
                if v_obj["src_idx"] is not None:
                    remote_srcs = [v_obj["src_idx"]]

            local_pred = v_obj["loc_idx"] if "loc_idx" in v_obj.attributes() else None
            if not remote_srcs:
                if local_pred is None:
                    if len(preds) != 1:
                        raise ValueError(
                            f"[ERROR] Macro vertex {v} has no recorded local predecessor "
                            f"and {len(preds)} total predecessors."
                        )
                    local_pred = preds[0]
                if "emits_send" in v_obj.attributes() and v_obj["emits_send"]:
                    if (
                        native_nic_serialize_sends
                        or native_cpu_serialize_ops
                        or serialize_same_nic_sends
                        or serialize_node_shared_nic_sends
                    ):
                        s_var = create_new_var(v)
                        comm_vars[v] = s_var
                        constr_name = self.__get_constr_name(local_pred, v)
                        add_constr(s_var >= var_map[local_pred], constr_name)
                        var_map[v] = s_var + post_cost
                    else:
                        comm_vars[v] = var_map[local_pred]
                        var_map[v] = var_map[local_pred] + post_cost
                else:
                    if native_cpu_serialize_ops and should_native_cpu_serialize_vertex(v_obj):
                        s_var = create_new_var(v)
                        constr_name = self.__get_constr_name(local_pred, v)
                        add_constr(s_var >= var_map[local_pred], constr_name)
                        var_map[v] = s_var + post_cost
                    else:
                        var_map[v] = var_map[local_pred] + post_cost
                return

            s_var = create_new_var(v)
            if "emits_send" in v_obj.attributes() and v_obj["emits_send"]:
                comm_vars[v] = s_var
            if local_pred is not None:
                constr_name = self.__get_constr_name(local_pred, v)
                add_constr(s_var >= var_map[local_pred], constr_name)

            lat_coeff = v_obj["lat_coeff"] if "lat_coeff" in v_obj.attributes() else 0
            bw_coeff = v_obj["bw_coeff"] if "bw_coeff" in v_obj.attributes() else 0
            dst_rank = v_obj["r"]
            for src_idx in remote_srcs:
                src_rank = self.dep_graph.graph.vs[src_idx]["r"]
                l_var, g_var = get_l_g_vars(src_rank, dst_rank)
                if topology:
                    num_links, switch_lat = topology.get_cost(src_rank, dst_rank)
                    switch_lat /= scale
                    l_var = (num_links * l_var + switch_lat)
                constr_name = self.__get_constr_name(src_idx, v)
                add_constr(
                    s_var >= var_map[src_idx] + (lat_coeff * l_var) + (bw_coeff * g_var),
                    constr_name,
                )
            var_map[v] = s_var + post_cost

        # ========================================================
        # Iterates through all the vertices in the graph
        # ========================================================
        for v in tqdm(vs, disable=not verbose):
            v_obj = self.dep_graph.graph.vs[v]
            if v in var_map:
                # Skips the vertex if it has already been computed
                continue
            
            # Checks the predecessors of the current vertex
            preds = self.dep_graph.get_predecessors(v)
            num_pred = len(preds)
            # =================================
            # The vertex is a starting vertex
            # =================================
            if num_pred == 0:
                # If v is the starting vertex
                var_map[v] = 0
                continue

            # ================================
            # If the vertex has one predecessor
            # ================================
            cost = get_vertex_cost(v_obj)
            if num_pred == 1:
                pred = preds[0]
                # Fetches the edge between the predecessor and the current vertex
                edge = self.dep_graph.get_edge(pred, v)
                # Checks if the edge is an irequire edge
                if edge["i"]:
                    dep = v_obj["i_idx"]
                    # If the edge is an irequire edge, then some additional
                    # cost will be added to the current vertex
                    compute_parallel_calc_time(v, dep, pred)
                else:
                    # Special case when LogGPS is not used
                    if v_obj["type"] == VertexType.SEND:
                        msg_size = v_obj["cost"]
                        # cpu_overhead = self.__get_msg_cpu_overhead(msg_size)
                        cpu_overhead = 0
                        is_rndv = msg_size > self.S and is_loggps
                        if is_loggps:
                            # FIXME: Probably not very efficient
                            # Well, for the sake of consistency
                            # it will be like this for now
                            if v not in comm_vars:
                                s_var = create_new_var(v)
                                comm_vars[v] = s_var
                            else:
                                s_var = comm_vars[v]
                            constr_name = self.__get_constr_name(pred, v)
                            add_constr(s_var >= var_map[pred], constr_name)
                            if is_rndv:
                                end_var = create_new_var(v)
                                rendezvous_end_vars[v] = end_var
                                add_constr(
                                    end_var >= s_var + o + cpu_overhead,
                                    self.__get_constr_name(v, v),
                                )
                                var_map[v] = end_var
                            else:
                                var_map[v] = s_var + cost + cpu_overhead
                        else:
                            if (
                                native_nic_serialize_sends
                                or native_cpu_serialize_ops
                                or serialize_same_nic_sends
                                or serialize_node_shared_nic_sends
                            ):
                                s_var = create_new_var(v)
                                comm_vars[v] = s_var
                                constr_name = self.__get_constr_name(pred, v)
                                add_constr(s_var >= var_map[pred], constr_name)
                            else:
                                s_var = var_map[pred]
                                comm_vars[v] = s_var
                            var_map[v] = s_var + cost + cpu_overhead
                    elif v_obj["type"] == VertexType.MACRO:
                        compute_macro_time(v_obj, preds)
                    else:
                        if native_cpu_serialize_ops and should_native_cpu_serialize_vertex(v_obj):
                            s_var = create_new_var(v)
                            constr_name = self.__get_constr_name(pred, v)
                            add_constr(s_var >= var_map[pred], constr_name)
                            var_map[v] = s_var + cost
                        else:
                            var_map[v] = var_map[pred] + cost
                continue
            
            # ================================
            # If the vertex has more than one predecessor
            # ================================
            
            # Creates a new variable
            type = v_obj["type"]
            if type == VertexType.SEND:
                # =============================
                # If the vertex is a SEND vertex
                # =============================
                compute_send_time(v_obj, preds)
                
            elif type == VertexType.RECV:
                # =============================
                # If the vertex is a RECV vertex
                # =============================
                compute_recv_time(v_obj, preds)
            elif type == VertexType.MACRO:
                # =============================
                # If the vertex is a compact collective stage
                # =============================
                compute_macro_time(v_obj, preds)
            else:
                # =============================
                # If the vertex is a CALC vertex
                # =============================
                s_var = create_new_var(v)
                for pred in preds:
                    constr_name = self.__get_constr_name(pred, v)
                    add_constr(s_var >= var_map[pred], constr_name)
                var_map[v] = s_var + cost
            
        obj_var = var_map[self.dep_graph.end_v]

        add_replay_precedence_constraints()
        if native_nic_serialize_sends:
            add_native_nic_send_constraints()
        if native_recv_nic_serialize_recvs:
            add_native_recv_nic_constraints()
        if native_cpu_serialize_ops:
            add_native_cpu_constraints()
        if serialize_same_nic_sends or serialize_node_shared_nic_sends:
            add_same_nic_send_constraints()
        if serialize_same_nic_recvs or serialize_node_shared_nic_recvs:
            add_same_nic_recv_constraints()
        if serialize_replay_comm_ops:
            add_replay_comm_constraints()
        if serialize_same_cpu_ops:
            add_same_cpu_constraints()
        if serialize_same_cpu_comm_ops:
            add_same_cpu_constraints(comm_only=True)

        for idx in vs:
            v_obj = self.dep_graph.graph.vs[idx]
            self.last_lp_debug_exprs["start"][idx] = get_vertex_start_expr(v_obj)
            self.last_lp_debug_exprs["finish"][idx] = var_map[idx]
            self.last_lp_debug_exprs["nic_release"][idx] = get_native_send_nic_release_expr(v_obj)
            self.last_lp_debug_exprs["cpu_release"][idx] = get_native_cpu_release_expr(v_obj)

        # Sets the objective function
        if self.use_gurobi:
            model.setObjective(obj_var, gp.GRB.MINIMIZE)
            model.update()
        else:
            model.Minimize(obj_var)
        return model


    @experimental
    @measure_time("reassign S")
    def reassign_S(self, model: gp.Model, S: int) -> None:
        """
        FIXME Currently only supports GUROBI model.
        FIXME Function is too long
        Re-inserts some of the constraints in the given LP model as per the
        dependency graph as well as the new S value (eager protocol threshold).
        """
        print(f"[INFO] Reassigning rendezvous threshold to {S} in the LP model...")

        l_matrix = \
            get_rank_latency_variable_matrix_gurobi(model, self.dep_graph.num_ranks)
        g_matrix = \
            get_rank_bandwidth_variable_matrix_gurobi(model, self.dep_graph.num_ranks)
        
        o = self.o

        def get_l_g_vars(src_rank: int, dst_rank: int) -> Tuple:
            """
            A helper function that returns the l and g variables
            between the given source and destination ranks.
            """
            return l_matrix[src_rank, dst_rank], g_matrix[src_rank, dst_rank]


        # Iterates through all the vertices in the dependency graph
        for v in tqdm(self.dep_graph.graph.vs.indices):
            v_obj = self.dep_graph.graph.vs[v]
            if v_obj["type"] != VertexType.RECV:
                continue
            
            # If the vertex is a RECV vertex
            s = v_obj["cost"]
            # If the send operation triggers the rendezvous protocol
            # based on the given S parameter
            is_rendezvous = s > S
            loc_idx = v_obj["loc_idx"]
            src_idx = v_obj["src_idx"]
            # Checks whether in the previous assignment, the send operation
            # triggers the rendezvous protocol
            # Retrieves the constraint
            constr_name = f"{loc_idx}_{src_idx}"
            constr = model.getConstrByName(constr_name)
            assert constr is not None, LOGGPS_ERROR_MSG
            # Checks whether the previous assignment is a rendezvous protocol
            # by checking the RHS of the constraint. If it is negative
            # then it did not trigger the rendezvous protocol in the
            # previous assignment
            prev_is_rendezvous = constr.RHS > 0
            # If the previous assignment matches the current assignment
            # then nothing needs to be done
            if prev_is_rendezvous == is_rendezvous:
                continue
            src_rank = v_obj["src_r"]
            dst_rank = v_obj["r"]

            l_var, g_var = get_l_g_vars(src_rank, dst_rank)
            src_preds = self.dep_graph.get_predecessors(src_idx)
            constr.RHS *= -1
            # Otherwise, constraints need to be changed
            if not is_rendezvous:
                # If the current assignment does not trigger
                # the rendezvous protocol

                # Changes all the constraints related to the sending vertex
                for pred in src_preds:
                    if pred == loc_idx:
                        continue
                    constr_name = f"{pred}_{src_idx}"
                    constr = model.getConstrByName(constr_name)
                    assert constr is not None, LOGGPS_ERROR_MSG
                    # Changes the coefficient of l_var for all
                    # the constraints related to the sending vertex
                    model.chgCoeff(constr, l_var, 0)

                # Changes the constraint representing the end time
                # of the sending vertex
                constr_name = f"{src_idx}e"
                e_var = model.getVarByName(f"y{src_idx}e")
                e_constr = model.getConstrByName(constr_name)
                assert constr is not None and e_var is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(e_constr, l_var, 0)
                model.chgCoeff(e_constr, g_var, 0)
                e_constr.RHS = o

                # Changes the constraint representing the end time
                # of the receiving vertex
                constr_name = f"{v}e"
                e_constr = model.getConstrByName(constr_name)
                assert e_constr is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(e_constr, l_var, 0)
                model.chgCoeff(e_constr, g_var, 0)
                e_constr.RHS = o

                # Changes the constraint connecting the sending vertex
                # and the receiving vertex
                constr_name = f"{src_idx}_{v}"
                s_var = model.getVarByName(f"y{src_idx}")
                constr = model.getConstrByName(constr_name)
                assert constr is not None and s_var is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(constr, l_var, -1)
                model.chgCoeff(constr, g_var, -(s - 1))
                model.chgCoeff(constr, e_var, -1)
                model.chgCoeff(constr, s_var, 0)
            
            else:
                # If the current assignment triggers the rendezvous protocol
                
                # Changes all the constraints related to the sending vertex
                for pred in src_preds:
                    if pred == loc_idx:
                        continue
                    constr_name = f"{pred}_{src_idx}"
                    constr = model.getConstrByName(constr_name)
                    assert constr is not None, LOGGPS_ERROR_MSG
                    # Changes the coefficient of l_var for all
                    # the constraints related to the sending vertex
                    model.chgCoeff(constr, l_var, -1)
                
                # Changes the constraint representing the end time
                # of the sending vertex
                constr_name = f"{src_idx}e"
                e_var = model.getVarByName(f"y{src_idx}e")
                e_constr = model.getConstrByName(constr_name)
                assert constr is not None and e_var is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(e_constr, l_var, -3)
                model.chgCoeff(e_constr, g_var, -(s - 1))
                e_constr.RHS = 4 * o

                # Changes the constraint representing the end time
                # of the receiving vertex
                constr_name = f"{v}e"
                e_constr = model.getConstrByName(constr_name)
                assert e_constr is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(e_constr, l_var, -2)
                model.chgCoeff(e_constr, g_var, -(s - 1))
                e_constr.RHS = 3 * o

                # Changes the constraint connecting the sending vertex
                # and the receiving vertex
                constr_name = f"{src_idx}_{v}"
                s_var = model.getVarByName(f"y{src_idx}")
                constr = model.getConstrByName(constr_name)
                assert constr is not None and s_var is not None, LOGGPS_ERROR_MSG
                model.chgCoeff(constr, l_var, 0)
                model.chgCoeff(constr, g_var, 0)
                model.chgCoeff(constr, e_var, 0)
                model.chgCoeff(constr, s_var, -1)
                
        model.update()
        print(f"[INFO] Reassigned rendezvous threshold to {S} in the LP model.")
