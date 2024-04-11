import numpy as np
import copy
import gurobipy as gp
import igraph
from time import time
from tqdm import tqdm
from ortools.linear_solver import pywraplp
from typing import List, Dict, Tuple, Set, Optional, Iterable
from metrics import *
from utils import *
from dep_graph import DependencyGraph, VertexType
from architecture_desc import Architecture, TwoLevelHierarchy


# FIXME: Still not very maintainable in terms of providing support
# for different solvers
class LPAnalyzer(object):
    """
    A object that computes different metrics based on the
    given linear program.
    """
    def __init__(self) -> None:
        self.use_gurobi = is_gurobi_installed()
        if self.use_gurobi:
            print("[INFO] GUROBI found. Use GUROBI to compute metrics.")
        else:
            print("[WARNING] GUROBI not found. Use ortools to compute metrics.")

    # ======================================================================
    # Network latency sensitivity
    # ======================================================================

    def __get_net_lat_sensitivity_ortools(self, model: pywraplp.Solver,
                                          L_ub: int, L_lb: int,
                                          verbose: bool) -> NetLatSensitivity:
        # TODO: Implement this
        raise NotImplementedError("Only GUROBI is supported at the moment.")

    def __get_net_lat_sensitivity_gurobi(self,
                                         model: gp.Model,
                                         L_ub: int,
                                         L_lb: int,
                                         step: int,
                                         verbose: bool) -> NetLatSensitivity:
        """
        Computes the network latency sensitivity of the given
        linear program by determining all critical latencies.
        """
        model.setParam("Threads", 0)
        # model.setParam("LPWarmStart", 1)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", 0)

        critical_latencies = []
        runtimes = []
        lat_costs = []

        l = model.getVarByName("l")
        # Sets the lower bound of `g` to a constant
        g = model.getVarByName("g")
        if g is not None:
            g.lb = 0.018

        if verbose:
            print("[INFO] Presolving the model...")
        presolved_model = model.presolve()
        presolved_model.printStats()
        reset_model = False
        if presolved_model.NumVars == 0:
            reset_model = True
        
        if verbose:
            print(f"[INFO] Presolve complete. Reset model: {reset_model}")
        if reset_model:
            model.reset()

        assert l is not None
        l.lb = L_ub
        eps = 0.01
        
        # Initial solve to get the starting value
        start_time = time()
        model.optimize()
        print(f"[DEBUG] Initial solve time: {time() - start_time:.2f}s")
        assert model.status == gp.GRB.OPTIMAL
        curr_a = l.RC
        curr_interval = [l.SALBLow, l.SALBUp]
        l.lb = l.SALBLow - eps
        # print(f"[DEBUG] l.lb: {l.lb}")
        runtimes.append((l.x, model.objVal))
        lat_costs.append((l.x, l.RC * l.x / model.objVal))
        
        iter = 1
        FREQ = 3
        print(f"[DEBUG] Start a: {curr_a}, intervals: {curr_interval}")
        print(f"[DEBUG] Predicted runtime: {model.objVal / (10 ** 9):.3f}s")
        start_val = curr_interval[0] - eps
        
        progress_bar = tqdm(total=int(start_val - L_lb), smoothing=1)
        # progress_bar.update()

        while l.SALBLow > L_lb:
            if reset_model:
                model.reset()
            model.optimize()
            interval = [l.SALBLow, l.SALBUp]
            a = l.RC
            
            runtimes.append((l.x, model.objVal))
            lat_costs.append((l.x, a * l.x / model.objVal))

            if a != curr_a:
                # If the current a value is different from the previous a value
                # then we have found a critical latency
                critical_latencies.insert(0, (curr_interval[0], curr_a))
                curr_a = a
                curr_interval = interval
            else:
                # If the current a value is the same as the previous a value
                # then we need to update the lower bound of l
                curr_interval[0] = interval[0]
            
            # Resets the lower bound of l
            l.lb = min(l.SALBLow - eps, l.x - step)
            iter += 1

            if iter % FREQ == 0:
                progress_bar.n = int(start_val - l.lb)
                time_per_solve = (time() - start_time) / iter
                progress_bar.desc = f"T: {model.objVal / (10 ** 9):.3f}s | L: {l.x:.0f} | avg_t: {time_per_solve:.2f}s"
                progress_bar.refresh()

        progress_bar.refresh()
        progress_bar.close()
    
        if len(critical_latencies) > 0:
            if critical_latencies[0][0] < 1:
                critical_latencies[0][0] = 0

            if a != critical_latencies[0][1] and critical_latencies[0][0] > 1:
                critical_latencies.insert(0, (L_lb, curr_a))
        else:
            critical_latencies.append((L_lb, curr_a))
        critical_latencies.append((L_ub, critical_latencies[-1][1]))
            
        assert model.status == gp.GRB.OPTIMAL

        if verbose:
            print(critical_latencies)
            print(f"[INFO] Number of iterations: {iter}")
            print(f"[INFO] Status: {model.status}")
            print(f"[INFO] l = ", l.x)
        
        return NetLatSensitivity(critical_latencies, runtimes, lat_costs)
    
    @measure_time("network latency sensitivity")
    def get_net_lat_sensitivity(self, model: Union[gp.Model, pywraplp.Solver],
                                L_ub: int = 10 ** 9,
                                L_lb: int = 0,
                                step: int = 1000,
                                verbose: bool = True) -> NetLatSensitivity:
        """
        Computes the network latency sensitivity of the given
        linear program.
        @param model: The linear program to be analyzed.
        @param L_ub: The upper bound of the network latency.
        @param L_lb: The lower bound of the network latency.
        @param step: The step size of the network latency.
        @param verbose: If True, will print out the progress and more
        information about the conversion.
        """
        if self.use_gurobi:
            res = self.__get_net_lat_sensitivity_gurobi(model, L_ub, L_lb, 
                                                        step, verbose)
        else:
            res = self.__get_net_lat_sensitivity_ortools(model, L_ub, L_lb, 
                                                         step, verbose)
        return res
    

    # ======================================================================
    # Network latency buffer
    # ======================================================================

    def __get_net_lat_buffer_gurobi(self, model: gp.Model,
                                    threshold: float,
                                    baseline_runtime: Optional[int],
                                    verbose: bool) -> float:
        """
        Computes the network latency buffer of the given linear program
        using GUROBI.
        """
        model.setParam("Threads", 0)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", int(verbose))

        l = model.getVarByName("l")
        g = model.getVarByName("g")
        assert l is not None
        if g is not None:
            g.lb = 0.018

        t = model.getVarByName("t")
        assert t is not None, "[ERROR] Decision variable 't' not found"
        
        # Calculates the baseline runtime on the spot
        if baseline_runtime is None:
            if verbose:
                print("[INFO] Baseline runtime not provided. Calculating the baseline runtime...")
            # TODO Should not hardcode the value of l
            l.lb = 1400
            # Solves the LP
            model.optimize()
            assert model.status == gp.GRB.OPTIMAL
            baseline_runtime = model.objVal

        max_runtime = baseline_runtime * (1 + threshold)        
        if verbose:
            print(f"[INFO] Baseline runtime: {baseline_runtime / (10 ** 9):.3f}s")
            print(f"[INFO] Calculating the network latency buffer: {max_runtime} ({threshold * 100:.2f}%)")

        # Changes the objective function of the model to maximize l
        model.setObjective(l, gp.GRB.MAXIMIZE)
        # Adds the constraint that the runtime should 
        # not exceed the maximum runtime
        model.addConstr(t <= max_runtime)
        model.update()
        model.reset()
        model.optimize()
        assert model.status == gp.GRB.OPTIMAL
        # Restores the original objective function
        return l.x


    @measure_time("network latency buffer")
    def get_net_lat_buffer(self, model: Union[gp.Model, pywraplp.Solver],
                           threshold: float, baseline_runtime: Optional[int],
                           verbose: bool = True) -> float:
        """
        Computes the network latency buffer of the given linear program.
        The network latency buffer is defined as the maximum network
        latency that can be tolerated without a significant increase
        in the runtime.
        @param model: The linear program to be analyzed.
        @param threshold: The maximum percentage increase in runtime
        that can be tolerated.
        @param baseline_runtime: The runtime of the linear program
        with the minimum network latency.
        """
        if self.use_gurobi:
            return self.__get_net_lat_buffer_gurobi(model, threshold, baseline_runtime, verbose)
        else:
            raise NotImplementedError("Only GUROBI is supported at the moment.")


    # ======================================================================
    # Solve the given LP
    # ======================================================================
    @measure_time("solve LP")
    def solve_lp(self, model: Union[gp.Model, pywraplp.Solver],
                 l_min: int, l_max: int, step: int,
                 verbose: bool = True) -> None:
        """
        Solves the given linear program in the range of L specified by
        `l_min` and `l_max`, the results will be saved to "runtime.csv"
        and "net_lat_sen.csv".
        """
        runtime = {}
        net_lat_sen = {}
        
        if self.use_gurobi:
            model.setParam("Threads", 0)
            # model.setParam("LPWarmStart", 1)
            model.setParam("Method", -1)
            model.setParam("LogToConsole", 0)
            l = model.getVarByName("l")
            g = model.getVarByName("g")
            if g is not None:
                g.lb = 0.018
        else:
            # Fetches the L variable from the Ortools model
            pass

        for L in tqdm(range(l_min, l_max + 1, step)):
            if self.use_gurobi:
                l.lb = L
                model.optimize()
                assert model.status == gp.GRB.OPTIMAL
                runtime[L] = model.objVal
                net_lat_sen[L] = l.RC
            else:
                pass
        
        # Saves the results to a CSV file
        save_dict_to_csv(runtime, "runtime.csv",
                        ["L", "runtime"], verbose)
        save_dict_to_csv(net_lat_sen, "net_lat_sen.csv",
                         ["L", "sensitivity"], verbose)
            

    # ======================================================================
    # Critical computations
    # ======================================================================

    def __get_critical_computations_given_L_gurobi(self,
                                                   model: gp.Model,
                                                   L: int,
                                                   verbose: bool = True) \
                                                   -> CriticalCompute:
        """
        Uses GUROBI to solve the linear program in order to obtain a list
        of critical computation vertices in the dependency graph.
        """
        model.setParam("Threads", 0)
        # model.setParam("LPWarmStart", 1)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", 0)
        
        l = model.getVarByName("l")
        l.lb = L
        l.ub = 10 ** 9
        model.update()
        # print(model.display())
        model.optimize()
        # Makes sure that the model is optimal and the solution is feasible
        assert model.status == gp.GRB.OPTIMAL
        print(f"[DEBUG] Objective value = {model.objVal}")
        # Iterates through all the constraints
        # and checks if the constraint is tight
        # In addition, checks the sensitivity bounds of the constraints
        # as well as their shadow prices
        # TODO: Not sure how to interpret the sensitivity bounds
        critical_computations = []
        for constr in tqdm(model.getConstrs()):
            if constr.Slack == 0 and constr.ConstrName[0] == 'C' and \
                constr.Pi > 0:
                # If it is a tight constraint
                interval = [constr.SARHSLow, constr.SARHSUp]
                interval_len = interval[1] - interval[0]
                shadow_price = constr.Pi
                print(f"[DEBUG] Constraint {constr.ConstrName} is tight. "
                      f"Interval = {interval} interval len = {interval_len} Shadow price = {shadow_price}")


    @measure_time("critical computations")
    def get_critical_computations_given_L(self,
                                          model: Union[gp.Model, pywraplp.Solver],
                                          L: int = 10 ** 4, verbose: bool = True) \
                                    -> List[int]:
        """
        Obtains a list of critical computation vertices 
        in the dependency graph given a specific network latency L.
        """
        
        if self.use_gurobi:
            res = self.__get_critical_computations_given_L_gurobi(model, L, verbose)
        else:
            pass
        
        return res

    # ======================================================================
    # Pairwise rank sensitivity
    # ======================================================================
    
    def __get_pairwise_rank_sensitivity_gurobi(self, model: gp.Model,
                                               arch: Architecture,
                                               verbose: bool = True) \
                                                -> np.ndarray:
        """
        Computes the pairwise rank sensitivity of the given linear program
        within the given network latency interval with GUROBI.
        """
        model.setParam("Threads", 0)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", 0)

        num_ranks = model.getVarByName("N")
        assert num_ranks is not None and num_ranks.lb == num_ranks.ub
        num_ranks = int(num_ranks.lb)
        assignment = arch.generate_round_robin_proc_assignment(num_ranks)

        # Retrieves a list of solver variables from the model
        l_matrix = get_rank_latency_variable_matrix_gurobi(model, num_ranks)
        g_matrix = get_rank_bandwidth_variable_matrix_gurobi(model, num_ranks)

        # Sets the lower bound of the latency variables to L
        arch.assign_variable_lb(assignment, l_matrix, g_matrix, self.use_gurobi)
        # Computes the pairwise rank sensitivity
        model.optimize()
        assert model.status == gp.GRB.OPTIMAL

        res = get_pairwise_sensitivity_matrix_gurobi(l_matrix, False)
        
        return res
    
    @measure_time("pairwise rank sensitivity")
    def get_pairwise_rank_sensitivity(self,
                                      model: Union[gp.Model, pywraplp.Solver],
                                      arch: Architecture,
                                      verbose: bool = True) -> np.ndarray:
        """
        TODO Pairwise rank sensitivity is a tentative name, might be
        changed later.

        Computes the pairwise rank sensitivity of the given linear program
        within the given network latency interval.
        It uses the same iterative approach as in the network latency
        sensitivity curve calculation.

        @param model: The linear program to be analyzed.
        @param arch: An object that describes the architecture of the system.
        @param verbose: If True, will print out the progress and more
        information about the conversion.
        """
        if self.use_gurobi:
            res = self.__get_pairwise_rank_sensitivity_gurobi(model, arch, verbose)
        else:
            pass
        return res
    

    # ======================================================================
    # [EXPERIMENTAL] Process placement analysis
    # ======================================================================
    
    def __get_process_placement_analysis_gurobi(self, model: gp.Model,
                                                arch: Architecture,
                                                verbose: bool = True) \
                                                -> List[List[int]]:
        """
        Performs process placement analysis on the given linear program
        and the cluster description with the GUROBI solver.
        This is done through an iterative approach.
        TODO: Provide support for general architecture and not only
        two-level hierarchy.
        """
        model.setParam("Threads", 8)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", 0)

        num_ranks = model.getVarByName("N")
        assert num_ranks is not None and num_ranks.lb == num_ranks.ub
        num_ranks = int(num_ranks.lb)

        l_matrix = get_rank_latency_variable_matrix_gurobi(model, num_ranks)
        g_matrix = get_rank_bandwidth_variable_matrix_gurobi(model, num_ranks)

        # assignment = arch.generate_random_proc_assignment(num_ranks)
        # assignment = [[2,3], [0,7], [1,5], [4,6]]
        # assignment = [[0, 3], [1, 2]]
        # The baseline will be the round-robin assignment
        assignment = \
            arch.generate_round_robin_proc_assignment(num_ranks)
        # assignment = [[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]]

        print(f"[DEBUG] Initial assignment: {assignment}")
        # Current objective value
        baseline_obj_val = float("inf")
        MAX_TOLERANCE = 10
        MIN_TOLERANCE = 3
        tolerance = 0
        best_assignment = copy.deepcopy(assignment)
        best_obj_val = float("inf")

        # Parameters for simulated-annealing-like algorithm
        # a random value will be generated and if it is
        # below the threshold, then a random swap will be performed;
        # otherwise, the swap that has the highest potential
        # performance improvement will be performed.
        # The threshold will be decreased by a factor of alpha after
        # each swap

        # Stores the list of swaps that can be performed
        # to potentially improve the objective value
        swap_options = []
        # If threshold is equal to 0, the algorithm is equivalent to
        # simply performing the swap that has the highest potential
        # performance improvement
        MIN_THRESHOLD = 0.1
        threshold = 1
        alpha = 0.99

        # Reference solve
        arch.assign_variable_lb(assignment, l_matrix, g_matrix, self.use_gurobi)

        solve_start = time()
        model.optimize()
        solve_time = time() - solve_start
        print(f"[DEBUG] Model solved in {solve_time:.2f} seconds", flush=True)
        
        # Dynamically adjusts max_tolerance according to the solve time
        # If the time taken to solve the model is more than 10 seconds,
        # then we set the tolerance to be the minimum tolerance
        if solve_time > 10:
            tolerance = MIN_TOLERANCE
        else:
            tolerance = int(min(MAX_TOLERANCE / (solve_time * 2), MAX_TOLERANCE))
            MAX_TOLERANCE = tolerance

        print(f"[INFO] Initial tolerance: {tolerance}")
        assert model.status == gp.GRB.OPTIMAL

        baseline_obj_val = best_obj_val = model.objVal
        print(f"[INFO] Baseline objective value: {baseline_obj_val / (10 ** 9)}s", flush=True)

        # Iteratively performs the process placement analysis
        # until no more swaps can be performed
        while True:
            # Computes the pairwise rank sensitivity matrix
            # from the current solution
            sens_l_matrix = get_pairwise_sensitivity_matrix_gurobi(l_matrix)
            sens_g_matrix = get_pairwise_sensitivity_matrix_gurobi(g_matrix)
            
            lat_sens_ratio, comm_cost = \
                arch.calc_comm_sens_ratio(sens_l_matrix, sens_g_matrix, assignment)
            comm_ratio = comm_cost / model.objVal
            print(f"[INFO] Latency cost ratio: {lat_sens_ratio}")
            print(f"[INFO] Communication cost ratio: {comm_ratio}")
            # visualize_heatmap(curr_sens_matrix, f"prs_matrix_{num_swap}.png")

            # ====================================
            # Selects the ranks to be swapped
            # ====================================
            # FIXME different swaps should be generated from an object
            num_clusters = len(assignment)
            max_change = -float("inf")
            best_swap = None
            # Iterates through all ranks that can be swapped
            for cluster_x in range(num_clusters):
                cluster = assignment[cluster_x]
                for i in range(len(cluster)):
                    rank_x = cluster[i]
                    # Iterates through all ranks that can be swapped with rank_x
                    for cluster_y in range(cluster_x + 1, num_clusters):
                        for j, rank_y in enumerate(assignment[cluster_y]):
                            l_change, g_change = \
                                arch.calc_performance_change(rank_x, cluster_x,
                                                             rank_y, cluster_y,
                                                             sens_l_matrix,
                                                             sens_g_matrix,
                                                             assignment)
                            change = l_change + g_change
                            if change > max_change:
                                max_change = change
                                best_swap = (change, cluster_x, cluster_y, i, j)
                            if change > 0:
                                # If the performance change is positive, then
                                # we store the swap as a potential option
                                swap_options.append((change, cluster_x, cluster_y, i, j))
            
            if len(swap_options) == 0:
                # If no swap can be performed at all, then we have reached the
                # potential optimal assignment
                if verbose:
                    print("[INFO] No more swaps can be performed. "
                          "Process placement analysis completed.", flush=True)
                break
            else:
                # Otherwise, we perform the swap
                # Generates a random number
                rand = np.random.uniform()
                if rand < threshold:
                    # If the random number is below the threshold, then
                    # we perform a random swap
                    swap_idx = np.random.choice(len(swap_options))
                    swap = swap_options[swap_idx]
                else:
                    swap = best_swap

                threshold = max(threshold * alpha, MIN_THRESHOLD)
                # threshold *= alpha
                # Performs the swap
                _, swap_cluster_x, swap_cluster_y, pos_x, pos_y = swap
                swap_x = assignment[swap_cluster_x][pos_x]
                swap_y = assignment[swap_cluster_y][pos_y]
                assignment[swap_cluster_x][pos_x] = swap_y
                assignment[swap_cluster_y][pos_y] = swap_x
                # Clears the swap options
                swap_options.clear()
                if verbose:
                    print(f"[INFO] Swapped {swap_x} and {swap_y} "
                          f"between cluster {swap_cluster_x} and {swap_cluster_y}")
                    print(f"[INFO] Curr assignment: {assignment}")
                    print(f"[INFO] Curr threshold: {threshold}", flush=True)

            
            # =======================================
            # Re-solves the linear program
            # =======================================
            model.reset()
            arch.assign_variable_lb(assignment, l_matrix, g_matrix, self.use_gurobi)
            
            # Solves the linear program
            solve_start = time()
            model.optimize()
            solve_time = time() - solve_start
            print(f"[DEBUG] Model solved in {solve_time:.2f} seconds", flush=True)

            assert model.status == gp.GRB.OPTIMAL
            
            print(f"[INFO] Curr objective value: {model.objVal / (10 ** 9)}s", flush=True)
            if model.objVal >= best_obj_val:
                if tolerance == 0:
                    # If the objective value does not improve, then we have
                    # reached the potential optimal assignment
                    if verbose:
                        print(f"[INFO] Objective value {model.objVal} >= {best_obj_val}", flush=True)
                    # Reverses the swaps that have been performed
                    break
                else:
                    tolerance -= 1
                    print(f"[DEBUG] Tolerance = {tolerance}", flush=True)
            else:
                best_obj_val = model.objVal
                # Resets tolerance value
                tolerance = MAX_TOLERANCE
                best_assignment = copy.deepcopy(assignment)
                print(f"[DEBUG] Best assignment: {best_assignment}")
        
        print(f"[INFO] Baseline objective value: {baseline_obj_val / (10 ** 9)}s")
        print(f"[INFO] Projected objective value after reassignment: {best_obj_val / (10 ** 9)}s")
        return best_assignment

    @measure_time("process placement analysis")
    @experimental("process placement analysis")
    def placement_analysis(self, model: Union[gp.Model, pywraplp.Solver],
                           arch: Architecture, verbose: bool = True) \
        -> List[List[int]]:
        """
        TODO: Need to change the return type to a more general type
        Performs process placement analysis on the given linear program
        and the cluster description.
        @param model: The linear program to be analyzed.
        @param architecture: An object that describes the architecture
        of the system on which the processes should be mapped.
        @param verbose: If True, will print out the progress and more
        information about the conversion.
        @return: A list of lists of integers. Each inner tuple represents
        the ranks that are assigned to the same node.
        """
        if self.use_gurobi:
            placement = \
                self.__get_process_placement_analysis_gurobi(model, arch, verbose)
        else:
            pass

        # Sorts the ranks in each cluster
        for cluster in placement:
            cluster.sort()
        return placement
        
    
    # ======================================================================
    # [EXPERIMENTAL] Communication cost analysis
    # ======================================================================
    def __comm_cost_analysis_gurobi(self, model: gp.Model,
                                     graph: DependencyGraph,
                                     verbose: bool) \
                                        -> CommCostMetrics:
        """
        Performs wait state analysis on the application given its
        corresponding linear program and the dependency graph.
        """
        model.setParam("Threads", 4)
        model.setParam("Method", -1)
        model.setParam("LogToConsole", 0)

        res = CommCostMetrics(graph.num_ranks)

        # FIXME Temporary solution
        # The architecture and assignment should be
        # passed in from the outside
        arch = TwoLevelHierarchy(256, 180, 0.01, 19000, 0.06)
        # arch = TwoLevelHierarchy(16, 1000, 0, 1000, 0)
        num_ranks = graph.num_ranks
        assignment = arch.generate_round_robin_proc_assignment(num_ranks)
        l_matrix = get_rank_latency_variable_matrix_gurobi(model, num_ranks)
        g_matrix = get_rank_bandwidth_variable_matrix_gurobi(model, num_ranks)
        # Sets the lower bound of l and g variables
        arch.assign_variable_lb(assignment, l_matrix, g_matrix, True)
        model.update()

        o = model.getVarByName("o")
        assert o is not None and o.lb == o.ub
        # o = o.lb
        o = 470
        # S = model.getVarByName("S")
        S = 1024
        # if S is not None:
        #     assert S.lb == S.ub
        #     S = S.lb
        print(f"[DEBUG] o = {o}, S = {S}")
        # else:
        #     S = float('inf')
        #     print(f"[DEBUG] o = {o}")

        def get_vertex_cost(v_obj: igraph.Vertex, is_rendezvous: bool,
                            is_local: bool) -> float:
            """
            A helper function that returns the cost of the given vertex.
            `is_local` indicates whether the returned cost is the
            local cost or the communication cost.
            If the given vertex is a send, then the cost would differ
            depending on its successor, if the successor is a recv, then
            `is_local` should be set to False, otherwise, it should be set
            to True.
            """
            v_type = v_obj["type"]
            cost = v_obj["cost"]
            # FIXME Inefficient to do the type check again in this function
            if v_type == VertexType.CALC:
                return cost
            else:
                s = cost
                # Obtains the src and dst ranks
                if v_type == VertexType.SEND:
                    # If the vertex is a send
                    src = v_obj["r"]
                    dst = v_obj["dst_r"]
                    if src == dst:
                        l = 0
                        g = 0
                    else:
                        l = l_matrix[src, dst].lb
                        g = g_matrix[src, dst].lb
                    if is_rendezvous:                        
                        if is_local:
                            return 4 * o + 3 * l + (s - 1) * g
                        
                        return l
                    else:
                        if is_local:
                            return o
                        return o + l + (s - 1) * g
                else:
                    # If the vertex is a recv
                    src = v_obj["src_r"]
                    dst = v_obj["r"]
                    # Obtains the values of latency and bandwidth as per
                    # the src and dst ranks
                    if src == dst:
                        l = 0
                        g = 0
                    else:
                        l = l_matrix[src, dst].lb
                        g = g_matrix[src, dst].lb

                    if is_rendezvous:
                        return 3 * o + 2 * l + (s - 1) * g
                    return o
        
        # model.optimize()
        # assert model.status == gp.GRB.OPTIMAL
        
        # res.add_total_time(model.objVal)
        # print(f"[DEBUG] Objective value = {model.objVal}")
        # A dictionary that keeps track of when each calc vertex starts
        comp_start = {}
        comp_start[graph.start_v] = 0
        # A dictionary that keeps track of when each send vertex starts
        # based on the starting time of its corresponding recv vertex.
        # Only used when LogGPS is enabled.
        send_remote_start = {}
        # Performs topological sort on the dependency graph
        # to obtain the order of the vertices
        vs = graph.get_topological_sort(mode='out')

        is_loggps = graph.is_loggps
        rendezvous_count = 0
        # Traverses the vertices in the topological order
        # and performs communication cost analysis
        for v in tqdm(vs[:-1]):
            v_obj = graph.graph.vs[v]
            v_type = v_obj["type"]
            succ = graph.get_successors(v)
            # Removes the current vertex from the dictionary
            # to save memory
            # s_time = comp_start.pop(v)
            s_time = comp_start[v]
            v_rank = v_obj["r"]

            if v_type == VertexType.CALC:
                # =============================
                # If the vertex is a CALC
                # =============================
                cost = get_vertex_cost(v_obj, False, True)
                succ_start = s_time + cost
                # Updates the start time of the successors
                for s in succ:
                    if is_loggps:
                        edge = graph.graph.get_eid(v, s)
                        # If the edge is a loggps virtual edge
                        # we do not update the start time of the successor
                        if graph.graph.es[edge]['v']:
                            send_remote_start[s] = succ_start
                            continue
                    
                    s_type = graph.graph.vs[s]["type"]
                    curr = comp_start.get(s)
                    if s_type == VertexType.RECV and curr is not None:
                        comp_start[s] = (succ_start, curr)
                    else:
                        if curr is None:
                            comp_start[s] = succ_start
                        else:
                            # Should be rare
                            comp_start[s] = succ_start if succ_start > curr else curr
                res.add_compute_interval(v_rank, (s_time, succ_start))
            else:
                is_rendezvous = v_obj["cost"] > S
                if v_type == VertexType.SEND:
                    # =============================
                    # If the vertex is a SEND
                    # =============================
                    # Calculates the local and communication cost of
                    # the send operation separately
                    local_cost = get_vertex_cost(v_obj, is_rendezvous, True)
                    comm_cost = get_vertex_cost(v_obj, is_rendezvous, False)
                    calc_start = s_time + local_cost
                    comm_start = s_time + comm_cost

                    if is_loggps:
                        if is_rendezvous:
                            assert v in send_remote_start
                            remote_start = send_remote_start.pop(v)
                            calc_start = \
                                (remote_start if remote_start > comm_start else comm_start) + local_cost
                            rendezvous_count += 1
                        # e_var = model.getVarByName(f"y{v}e")
                        # assert e_var is not None
                        # # assert e_var.X == calc_start, f"e_var.X = {e_var.X}, calc_start = {calc_start}"
                        # if e_var.X != calc_start:
                        #     print(f"[ERROR] s_time: {s_time}, comm_cost: {comm_cost}")
                        #     print(f"[ERROR] remote start: {remote_start} comm start: {comm_start}")
                        #     print(f"[ERROR] y{v}.X = {model.getVarByName(f'y{v}').X}")
                        #     print(f"[ERROR] Local cost: {local_cost}")
                        #     print(f"[ERROR] e_var.X = {e_var.X}, calc_start = {calc_start}")
                        #     graph.print_vertex_info(v)
                        #     exit(-1)

                    for s in succ:
                        s_type = graph.graph.vs[s]["type"]
                        curr = comp_start.get(s)
                        if s_type == VertexType.CALC:
                            # If the successor is a calc
                            if curr is None:
                                comp_start[s] = calc_start
                            else:
                                comp_start[s] = calc_start if calc_start > curr else curr

                        elif s_type == VertexType.RECV:
                            # If the successor is a recv
                            if curr is not None:
                                comp_start[s] = (curr, comm_start)
                            else:
                                comp_start[s] = comm_start
                    res.add_comm_interval(v_rank, (s_time, calc_start))
                else:
                    # =============================
                    # If the vertex is a RECV
                    # =============================
                    cost = get_vertex_cost(v_obj, is_rendezvous, True)
                    assert len(s_time) == 2
                    # The start time of the recv vertex should be
                    # a tuple that contains the start time of the
                    # corresponding to the send operation and the
                    # local calc operation
                    recv_time, send_time = s_time
                    is_receiver_waiting = send_time > recv_time
                    if is_rendezvous and not is_receiver_waiting:
                        src_rank = v_obj["src_r"]
                        res.add_wait_interval(src_rank, (send_time, recv_time))
                    elif is_receiver_waiting:
                        res.add_wait_interval(v_rank, (recv_time, send_time))
                    # Updates the start time of the successor
                    succ_start = (send_time if is_receiver_waiting else recv_time) + cost
                    for s in succ:
                        curr = comp_start.get(s)
                        if curr is None:
                            comp_start[s] = succ_start
                        else:
                            comp_start[s] = succ_start if succ_start > curr else curr
                    
                    res.add_comm_interval(v_rank, (recv_time, succ_start))
                    # if is_loggps:
                    #     e_var = model.getVarByName(f"y{v}e")
                    #     assert e_var is not None
                    #     if e_var.X != succ_start:
                    #         print(f"[ERROR] recv_time = {recv_time}, send_time = {send_time}")
                    #         print(f"[ERROR] y{v}.X = {model.getVarByName(f'y{v}').X}")
                    #         print(f"[ERROR] Local cost: {local_cost}")
                    #         print(f"[ERROR] e_var.X = {e_var.X}, succ_start = {succ_start}")
                    #         graph.print_vertex_info(v)
                    #         exit(-1)

        print(f"[DEBUG] Rendezvous was triggered {rendezvous_count} times")
        res.add_total_time(comp_start[graph.end_v])
        # print(f"[DEBUG] Objective value = {model.objVal / (10 ** 9):.3f}s")
        print(f"[DEBUG] Total time = {res.total_time / (10**9):.3f}s")
        # print(f"[DEBUG] Objective value = {model.objVal} ns")
        # print(f"[DEBUG] Total time = {res.total_time} ns")
        # assert abs(model.objVal - res.total_time) < 1e-3
        res.finalize()
        print(f"[INFO] Wait time: {res.wait_time}")
        print(f"[INFO] Wait overlap time: {res.wait_overlap_time}")
        print(f"[INFO] Communication time: {res.comm_time}")
        print(f"[INFO] Communication overlap time: {res.comm_overlap_time}")
        # Computes the communication cost without overlap
        for i in range(graph.num_ranks):
            comm_cost = (res.comm_time[i] - res.comm_overlap_time[i]) / 10**9
            print(f"[INFO] Rank {i} net communication cost: {comm_cost:.3f}s")
        return res

    @measure_time("Communication cost analysis")
    @experimental("Communication cost analysis")
    def comm_cost_analysis(self, model: Union[gp.Model, pywraplp.Solver],
                            graph: DependencyGraph,
                            verbose: bool = True) \
                            -> CommCostMetrics:
        """
        Performs communication cost analysis on the application given its
        corresponding linear program and the dependency graph.
        Communication cost analysis include the following:
        - Wait state analysis for each rank
        - Overlap analysis for each rank
        - Estimation of the total communication cost
        @param model: The linear program to be analyzed.
        @param graph: The dependency graph of the application
        @param include_overlap: If True, will include overlap analysis
        in the results.
        @param verbose: If True, will print out the progress and more
        information the algorithm.
        @return: A WaitStateMetrics object that stores the results
        of the wait state analysis.
        """
        if self.use_gurobi:
            res = self.__comm_cost_analysis_gurobi(model, graph, verbose)
        else:
            pass

        return res