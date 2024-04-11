from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from utils import experimental



class NetTopology(object):

    @experimental("Topology")
    def __init__(self, cost_matrix: np.array, switch_latency: float = 108) \
        -> None:
        """
        Initialize the NetTopology object.
        @param cost_matrix: A 2D numpy array that represents the cost
        matrix of the network topology. The cost matrix is a square matrix
        of size num_ranks x num_ranks, where num_ranks is the number of ranks
        in the MPI program. The cost matrix is symmetric, and the diagonal
        elements are all 0.
        @param switch_latency: The latency of a switch in ns.
        """
        self.cost_matrix: np.array = cost_matrix
        self.switch_latency: float = switch_latency

    def get_num_links(self, src: int, dst: int) -> int:
        """
        Returns the cost of sending a message from src rank to dst rank
        in terms of the number of links between them.
        The cost is the coefficient in front of the L parameter
        in the LogGP model, which is used to indicate the number of
        hops between the two ranks. For example, if the cost is 2 between
        rank 1 and rank 2, then the number of hops between them is 1.
        """
        return self.cost_matrix[src][dst]
    
    def get_switch_latency(self, src: int, dst: int) -> float:
        """
        Returns the latency incurred by the switch when sending 
        a message from src rank to dst rank in ns.
        """
        if src == dst:
            return 0
        return (self.cost_matrix[src][dst] - 1) * self.switch_latency

    def get_cost(self, src: int, dst: int) -> Tuple[int, float]:
        """
        Returns the cost of sending a message from src rank to dst rank.
        The cost is a tuple of two values: the number of links between
        the two ranks, and the latency incurred by the switch.
        """
        return self.get_num_links(src, dst), self.get_switch_latency(src, dst)

    @staticmethod
    def from_file(filename: str) -> NetTopology:
        """
        Creates a NetTopology object from a file.
        """
        raise NotImplementedError("[ERROR] from_file method is not implemented yet.")

    @staticmethod
    def default_topology(num_ranks: int, switch_latency: float = 108) -> NetTopology:
        """
        Creates a default NetTopology object with the given number of ranks.
        In the default topology, the cost of sending a message from rank i
        to rank j is 1 if i != j, and 0 if i == j.
        """
        cost_matrix = np.ones((num_ranks, num_ranks))
        np.fill_diagonal(cost_matrix, 0)
        return NetTopology(cost_matrix, switch_latency)
    

    @staticmethod
    def fat_tree(num_nodes: int, k: int, 
                 num_layers: int = 3, switch_latency: float = 108) -> NetTopology:
        """
        Creates a Fat-Tree network topology.
        @param k: The number of ports in each switch, which is also the
        number of pods as well as the number of nodes in each pod.
        @param num_layers: The number of layers in the Fat-Tree network.
        @param switch_latency: The latency of a switch in ns.
        """
        print(f"[INFO] Creating Fat-Tree topology with parameters:\n"
              f"k={k}, layers={num_layers}, switch latency={switch_latency} ns")
        num_nodes_per_edge_switch = k // 2
        num_nodes_per_pod = k
        
        cost_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            # Checks which edge switch the node is connected to
            i_edge_switch = i // num_nodes_per_edge_switch
            # Checks which pod the node is connected to
            i_pod = i // num_nodes_per_pod
            for j in range(i + 1, num_nodes):
                # Checks which edge switch the dst node is connected to
                j_edge_switch = j // num_nodes_per_edge_switch
                # Checks which pod the dst node is connected to
                j_pod = j // num_nodes_per_pod
                if i_edge_switch == j_edge_switch:
                    # If the two nodes are connected to the same edge switch
                    cost_matrix[i][j] = 2
                    cost_matrix[j][i] = 2
                elif i_pod == j_pod:
                    # If the two nodes are connected to the same pod
                    cost_matrix[i][j] = 4
                    cost_matrix[j][i] = 4
                else:
                    # If the two nodes are connected to different pods
                    cost_matrix[i][j] = num_layers * 2
                    cost_matrix[j][i] = num_layers * 2
        return NetTopology(cost_matrix, switch_latency)


    @staticmethod
    def dragonfly(g: int, a: int, p: int, num_nodes: int,
                  switch_latency: float = 108) -> NetTopology:
        """
        Creates a Dragonfly network topology given the parameters.
        Note that parameters such as h and k are not needed in this case.
        @param g: The number of groups.
        @param a: The number of routers per group.
        @param p: The number of nodes per switch.
        @param num_nodes: The total number of nodes in the network.
        """
        assert num_nodes <= g * a * p, \
            f"[ERROR] The number of nodes {num_nodes} exceeds the maximum number of nodes {g * a * p}"
        print(f"[INFO] Creating Dragonfly topology with parameters:\n"
              f"g={g}, a={a}, p={p}, switch latency={switch_latency} ns")
        cost_matrix = np.zeros((num_nodes, num_nodes))
        num_nodes_per_group = a * p
        num_nodes_per_switch = p

        for i in range(num_nodes):
            i_switch = i // num_nodes_per_switch
            i_group = i // num_nodes_per_group
            for j in range(i + 1, num_nodes):
                j_switch = j // num_nodes_per_switch
                j_group = j // num_nodes_per_group
                if i_group == j_group:
                    # If the two nodes are in the same group
                    if i_switch == j_switch:
                        # If the two nodes are in the same switch
                        cost_matrix[i][j] = 2
                        cost_matrix[j][i] = 2
                    else:
                        # If the two nodes are in different switch
                        cost_matrix[i][j] = 3
                        cost_matrix[j][i] = 3
                else:
                    # If the two nodes are in different groups
                    cost_matrix[i][j] = 4
                    cost_matrix[j][i] = 4
        return NetTopology(cost_matrix, switch_latency)