import sys
import igraph
import subprocess
import numpy as np
from time import time
from array import array
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import gurobipy as gp
from scipy.sparse import lil_matrix
from typing import List, Tuple, Dict, Optional, Set, Union
from topology import NetTopology
from dep_graph import DependencyGraph, VertexType
from metrics import *
from utils import *


class DependencyGraphAnalyzer(object):
    """
    An object that analyzes the dependency graph so as to
    measure different types of performance metrics.
    """
    def __init__(self) -> None:
        pass

    @experimental("communication pattern analysis")
    def comm_pattern_analysis(self, graph: DependencyGraph) -> None:
        """
        Potentially recognizes the communication pattern in the dependency graph by first converting the graph into a sparse adjacency matrix.
        """

        # Initializes a sparse adjacency matrix whose size
        # is the number of vertices in the dependency graph
        # and whose data type is boolean
        num_vertices = graph.graph.vcount()
        adj_matrix = lil_matrix((num_vertices, num_vertices), dtype=bool)

        # Iterates through all the vertices in the dependency graph
        for v in tqdm(graph.graph.vs.indices):
            v_type = graph.graph.vs[v]['type']
            if v_type == VertexType.SEND:
                for u in graph.get_successors(v):
                    if graph.graph.vs[u]['type'] == VertexType.RECV:
                        adj_matrix[v, u] = True
                        break
        
        plt.spy(adj_matrix)
        # Marks the start of each rank
        alpha = 0.5
        for start_v in graph.rank_to_start_v:
            # Reduce the opacity of the lines
            plt.axvline(x=start_v, color='k', linestyle='--', alpha=alpha)
            plt.axhline(y=start_v, color='k', linestyle='--', alpha=alpha)
        
        # for end_v in graph.rank_to_end_v:
        #     # Reduce the opacity of the lines
        #     plt.axvline(x=end_v, color='k', linestyle='--', alpha=alpha)
        #     plt.axhline(y=end_v, color='k', linestyle='--', alpha=alpha)
        

        plt.savefig("adj_matrix.png")


