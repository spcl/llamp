import numpy as np
import igraph
import gurobipy as gp
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
                        verbose: bool = True) -> Tuple:
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
                if not G:
                    g = model.addVar(name="g")
                else:
                    g = G
            
            N = model.addVar(name="N", lb=num_ranks, ub=num_ranks)
            o = model.addVar(name="o", lb=self.o, ub=self.o)
            if self.S != float("inf"):
                S = model.addVar(name="S", lb=self.S, ub=self.S)
            model.update()
            return l, g
        else:
            # If GUROBI is not installed, then use the default ortools model
            print("[WARNING] GUROBI is not installed. Using the default ortools model.")
            model = pywraplp.Solver.CreateSolver("GLOP")
            l = model.NumVar(0.0, 1.0, "l")
            g = model.NumVar(0.0, 1.0, "g")
            # Encodes the number of ranks in the model
            N = model.NumVar(name="N", lb=num_ranks, ub=num_ranks)

            return l, g

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
                      G: Optional[float] = None,
                      topology: Optional[NetTopology] = None,
                      unit: str = "ns") \
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
        print(f"[INFO] Time unit: {unit}")
        o = self.o / scale
        is_loggps = self.dep_graph.is_loggps
        
        # Create the LP model
        num_ranks = self.dep_graph.num_ranks
        if self.use_gurobi:
            model = gp.Model("Dependency Graph LP Model")
        else:
            model = pywraplp.Solver.CreateSolver("GLOP")
        
        if verbose:
            print(f"[INFO] Number of ranks: {num_ranks}")
            print(f"[INFO] Pairwise analysis: {pairwise_analysis}")

        if G is not None and verbose:
            print(f"[INFO] Using constant bandwidth parameter G: {G}")

        # Adds the required variables to the model
        l, g = self.__add_variables(model, num_ranks, pairwise_analysis, G, verbose)

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
                return model.NumVar(0.0, 1.0, var_name)
        
        def remove_var(var: Union[gp.Var, pywraplp.Variable]) -> None:
            """
            A helper function that removes the given variable from the model.
            """
            if self.use_gurobi:
                model.remove(var)
            else:
                model.Remove(var)

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
                model.addConstr(constr)
            else:
                model.Add(constr)
        
        def get_l_g_vars(src_rank: int, dst_rank: int) -> Tuple:
            """
            A helper function that returns the l and g variables
            between the given source and destination ranks.
            """
            if pairwise_analysis:
                return l[src_rank, dst_rank], g[src_rank, dst_rank]
            return l, g
        
        # Obtains topological ordering of all the vertices
        vs = self.dep_graph.get_topological_sort(mode="out")
        # vs = self.dep_graph.bfs()

        # A dictionary that maps the global index of each
        # vertex to either a linear expression or a constant value.
        # Conceptually, each entry represents the timestamp
        # at which each vertex finishes execution.
        var_map = {}

        # A dictionary that maps the global index of SEND vertex
        # to a solver variable that denotes 
        comm_vars = {}


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
                    # If the edge is not a virtual edge
                    constr_name = self.__get_constr_name(pred, v)
                    rhs = var_map[pred] + l_var if is_rndv else var_map[pred]
                    add_constr(s_var >= rhs, constr_name)
                elif is_rndv and edge["v"]:
                    # If the edge is a virtual edge and RNDV is triggered
                    constr_name = self.__get_constr_name(pred, v)
                    add_constr(s_var >= var_map[pred], constr_name)
            # FIXME Code duplication
            # Computes when the sender operation will actually finish
            if is_rndv:
                # comm_vars[v] = s_var + l_var
                end = s_var + 3 * o + 3 * l_var + (msg_size * g_var) + cpu_overhead
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
            assert len(preds) == 2
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

            # Retrieves the global index of the local calc vertex
            # on which the recv vertex depends
            loc_idx = v_obj["loc_idx"]
            # Retrieves the global index of the remote send vertex
            src_idx = v_obj['src_idx']
            # Creates a new variable s_var
            s_var = create_new_var(v)
            # Creates constraints for s_var
            if not is_rndv:
                # If RNDV is not triggered
                # Creates a constraint for the local dependency
                constr_name = self.__get_constr_name(loc_idx, v)
                add_constr(s_var >= var_map[loc_idx], constr_name)
                # Creates a constraint for the remote dependency
                constr_name = self.__get_constr_name(src_idx, v)
                constr = s_var >= comm_vars[src_idx] \
                    + l_var + (msg_size * g_var) + o + cpu_overhead
                add_constr(constr, constr_name)
                var_map[v] = s_var + o + cpu_overhead
            else:
                # If RNDV is triggered
                # coeff = 4 if is_mpich else 3
                assert src_idx in comm_vars
                var_map[v] = comm_vars[src_idx] + \
                    2 * o + 3 * l_var + (msg_size * g_var) + cpu_overhead
                # Removes the added solver variable
                remove_var(s_var)

        # ========================================================
        # Iterates through all the vertices in the graph
        # ========================================================
        for v in tqdm(vs):
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
                        else:
                            s_var = var_map[pred]
                            comm_vars[v] = s_var
                        var_map[v] = s_var + cost + cpu_overhead
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
