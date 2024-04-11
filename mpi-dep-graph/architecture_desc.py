from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set, Optional
from utils import *


# =======================================================
# TODO: Support architecture topology of a general system
# =======================================================

class Architecture(ABC):
    """
    An abstract class to represent the architecture description of a system.
    Should be implemented by the subclasses.
    """
    @abstractmethod
    def generate_random_proc_assignment(self, num_ranks: int, seed: int):
        pass

    @abstractmethod
    def generate_round_robin_proc_assignment(self, num_ranks: int):
        pass

    @abstractmethod
    def assign_variable_lb(self, assignment, l_matrix: np.ndarray,
                           g_matrix: np.ndarray, use_gurobi: bool):
        pass

    @abstractmethod
    def calc_performance_change(self, x: int, x_cluster: int,
                                y: int, y_cluster: int,
                                sens_l_matrix: np.ndarray,
                                sens_g_matrix: np.ndarray,
                                curr_assignment) -> float:
        pass
    
    @abstractmethod
    def calc_comm_sens_ratio(self, sens_l_matrix: np.ndarray,
                             sens_g_matrix: np.ndarray,
                             assignment) -> float:
        pass


class TwoLevelHierarchy(Architecture):
    """
    A class to represent the architecture description of a system,
    the MPI ranks will be mapped to the architecture description
    after the rank placement algorithm is executed.

    This class represents the architecture description of
    a two-level hierarchical system, i.e., a node is composed of
    several cores.
    Similar to the TLeaf topology in SCOTCH.
    """
    def __init__(self, num_cores_per_node: int,
                 intra_node_L: float, intra_node_G: float,
                 inter_node_L: float, inter_node_G: float) -> None:
        """
        :param num_nodes: number of nodes in the system
        :param intra_node_L: latency of intra-node communication,
        The L parameter in the LogGP model.
        :param intra_node_G: gap per byte for the communication between
        two ranks in a single node. A representation of the bandwidth of
        intra-node communication. The G parameter in the LogGP model.
        :param inter_node_L: latency of inter-node communication.
        :param inter_node_G: gap per byte for the communication between
        two ranks in different nodes.
        """
        self.num_cores_per_node = num_cores_per_node
        self.intra_node_L = intra_node_L
        self.intra_node_G = intra_node_G
        self.inter_node_L = inter_node_L
        self.inter_node_G = inter_node_G
    
    def generate_random_proc_assignment(self, num_ranks: int, seed: int = 42) \
        -> List[List[int]]:
        """
        Generates a random processor assignment for the given cluster
        description and the number of ranks.
        @param num_ranks: The number of ranks in the MPI program.
        @param seed: The random seed.
        """
        assignment = []
        remaining_ranks = set(range(num_ranks))
        while len(remaining_ranks) > 0:
            # Randomly selects a set of ranks on the node
            ranks_on_node = set(random.sample(tuple(remaining_ranks),
                                              k=self.num_cores_per_node))
            assignment.append(list(ranks_on_node))
            remaining_ranks -= ranks_on_node
    
        return assignment
    
    def generate_round_robin_proc_assignment(self, num_ranks: int) \
        -> List[List[int]]:
        """
        Generates a round-robin processor assignment. The ranks are
        assigned to the nodes in a round-robin fashion, as an example,
        if there are 4 ranks per node and 8 ranks in total, the ranks
        will be assigned to the nodes as follows:
        [[0, 1, 2, 3], [4, 5, 6, 7]]
        @param num_ranks: The number of ranks in the MPI program.
        """
        assignment = []
        num_cores_per_node = self.num_cores_per_node
        num_clusters = math.ceil(num_ranks / num_cores_per_node)
        for i in range(num_clusters):
            # If this is not the last cluster
            if i < num_clusters - 1:
                ranks_on_node = list(range(i * num_cores_per_node,
                                        (i + 1) * num_cores_per_node))
            else:
                ranks_on_node = list(range(i * num_cores_per_node, num_ranks))
            assignment.append(ranks_on_node)
        
        return assignment
    
    def assign_variable_lb(self, assignment: List[List[int]],
                           l_matrix: np.ndarray, g_matrix: np.ndarray,
                           use_gurobi: bool) -> None:
        """
        Assigns the lower bound of the latency and bandwidth variables
        based on the given assignment.
        """
        num_clusters = len(assignment)
        if use_gurobi:
            # Implementation for GUROBI model
            for cluster_x in range(num_clusters):
                cluster = assignment[cluster_x]
                for i in range(len(cluster)):
                    rank_x = cluster[i]
                    # Assigns the intra-node latency and bandwidth
                    for j in range(i + 1, len(cluster)):
                        rank_y = cluster[j]
                        l_matrix[rank_x, rank_y].LB = self.intra_node_L
                        g_matrix[rank_x, rank_y].LB = self.intra_node_G
                    
                    # Assigns the inter-node latency and bandwidth
                    for cluster_y in range(cluster_x + 1, num_clusters):
                        for rank_y in assignment[cluster_y]:
                            l_matrix[rank_x, rank_y].LB = self.inter_node_L
                            g_matrix[rank_x, rank_y].LB = self.inter_node_G
        else:
            raise NotImplementedError("[ERROR] Not implemented yet.")


    def calc_performance_change(self, x: int, x_cluster: int,
                                y: int, y_cluster: int,
                                sens_l_matrix: np.ndarray,
                                sens_g_matrix: np.ndarray,
                                curr_assignment: List[List[int]]) \
                                    -> Tuple[float, float]:
        """
        Calculates the potential performance change if we swap the given
        rank x and y in the current assignment as per the sensitivity
        matrices. Returns the potential change in latency cost and
        the potential change in bandwidth cost as a tuple.
        """
        # Potential change of the latency cost
        l_change = 0
        # Potential change of the bandwidth cost
        g_change = 0
        # print(f"[DEBUG] x: {x}, y: {y}")
        # print(f"[DEBUG] l sensitivity matrix: {sens_l_matrix}")
        # print(f"[DEBUG] g sensitivity matrix: {sens_g_matrix}")

        for rank in curr_assignment[x_cluster]:
            if rank == x:
                continue
            # All communication between x and the other ranks in the
            # same cluster as x will be affected by the swap negatively
            l_change -= sens_l_matrix[x, rank]
            g_change -= sens_g_matrix[x, rank]
            # All communication between y and the other ranks in the
            # same cluster as x will be affected by the swap positively
            l_change += sens_l_matrix[y, rank]
            g_change += sens_g_matrix[y, rank]

        for rank in curr_assignment[y_cluster]:
            if rank == y:
                continue
            # All communication between y and the other ranks in the
            # same cluster as y will be affected by the swap negatively
            l_change -= sens_l_matrix[y, rank]
            g_change -= sens_g_matrix[y, rank]
            # All communication between x and the other ranks in the
            # same cluster as y will be affected by the swap positively
            l_change += sens_l_matrix[x, rank]
            g_change += sens_g_matrix[x, rank]

        # The total potential change in latency cost
        l_change = l_change * (self.inter_node_L - self.intra_node_L)
        # The total potential change in bandwidth cost
        g_change = g_change * (self.inter_node_G - self.intra_node_G)
        # The total potential change in performance will be
        # the sum of the potential change in latency cost and
        # the potential change in bandwidth cost
        return l_change, g_change
    

    def calc_comm_sens_ratio(self, sens_l_matrix: np.ndarray,
                             sens_g_matrix: np.ndarray,
                             assignment: List[List[int]],
                             total_cost: bool = True) \
                                -> Union[float, Tuple[float, float]]:
        """
        Calculates the communication sensitivity ratio based on the given
        sensitivity matrices and the current assignment.
        The communication sensitivity ratio is defined as the ratio
        of the total latency cost over the total communication cost.
        @param sens_l_matrix: The sensitivity matrix of the latency cost.
        @param sens_g_matrix: The sensitivity matrix of the bandwidth cost.
        @param assignment: The current assignment.
        @param total_cost: If True, will return the total cost of the
        communication (i.e., latency and bandwidth) along with the ratio.
        """
        l_cost = 0
        g_cost = 0

        for cluster_x in range(len(assignment)):
            cluster = assignment[cluster_x]
            for i in range(len(cluster)):
                rank_x = cluster[i]
                # Computes the intra-node latency and bandwidth costs
                for j in range(i + 1, len(cluster)):
                    rank_y = cluster[j]
                    l_cost += sens_l_matrix[rank_x, rank_y] * self.intra_node_L
                    g_cost += sens_g_matrix[rank_x, rank_y] * self.intra_node_G
                
                # Computes the inter-node latency and bandwidth costs
                for cluster_y in range(cluster_x + 1, len(assignment)):
                    for rank_y in assignment[cluster_y]:
                        l_cost += sens_l_matrix[rank_x, rank_y] * self.inter_node_L
                        g_cost += sens_g_matrix[rank_x, rank_y] * self.inter_node_G
        
        res = l_cost / (g_cost + l_cost)

        if total_cost:
            return res, l_cost + g_cost
        else:
            return res