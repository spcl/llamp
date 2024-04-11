import gurobipy as gp
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Tuple, Set, List, Callable, Optional
import numpy as np
import random
import math
import functools
from time import time
from ortools.linear_solver.python import model_builder
from ortools.linear_solver import pywraplp
from dep_graph import DependencyGraph, VertexType

"""
This file contains utility functions that are used by the
main scripts.
"""

def is_gurobi_installed() -> bool:
    """
    Checks if the GUROBI solver is installed on the machine
    by running the command "gurobi_cl --version".
    """
    try:
        subprocess.check_output(["gurobi_cl", "--version"])
        return True
    except:
        return False



def load_model(model_path: str, verbose: bool = True) \
                -> Union[gp.Model, pywraplp.Solver]:
    """
    Loads the LP model in MPS format from the given path.
    """
    if verbose:
        print(f"[INFO] Loading the LP model from {model_path}...", flush=True)
    
    if is_gurobi_installed():
        # Loads a GUROBI model
        model = gp.read(model_path)
    else:
        # Loads an ortools model
        # https://github.com/google/or-tools/issues/523
        model = model_builder.ModelBuilder()
        # Checks the file extension
        if model_path.endswith(".mps"):
            model.import_from_mps_file(model_path)
        else:
            raise ValueError(f"[ERROR] Unsupported file extension: {model_path}")
        
    if verbose:
        print("[INFO] Load LP model: SUCCESS", flush=True)

    return model


def visualize_heatmap(data: np.ndarray,
                      filename: str = "pairwise_latency_sensitivity.png") -> None:
    """
    Visualizes the given 2D array as a heatmap.
    """
    # Masks the lower triangular matrix
    mask = np.zeros_like(data)
    mask[np.tril_indices_from(data)] = True
    plt.figure(figsize=(10, 10))
    with sns.axes_style("white"):
        sns.heatmap(data, annot=True, mask=mask, fmt="g", cmap="YlGnBu")
    plt.savefig(filename)
    plt.close()
    print(f"[DEBUG] Saved the heatmap to {filename}.")


def get_rank_latency_variable_matrix_gurobi(model: gp.Model,
                                            num_ranks: int) -> np.ndarray:
    """
    Extracts the latency variable matrix from the GUROBI model.
    """
    # The number of l variables should be equal to the number of
    # pairs of ranks
    num_ls = (1 + (num_ranks - 1)) * (num_ranks - 1) // 2
    l_matrix = np.empty((num_ranks, num_ranks), dtype=gp.Var)
    c = 0
    for i in range(num_ranks):
        l_matrix[i][i] = 0
        for j in range(i + 1, num_ranks):
            l = model.getVarByName(f"l[{c}]")
            if l is None:
                raise ValueError(f"[ERROR] Variable l[{c}] does not exist in "
                                 "the model. Make sure pairwise analysis is "
                                 "enabled.")
            l_matrix[i, j] = l_matrix[j, i] = l
            c += 1
    assert c == num_ls, f"[ERROR] Number of l variables in the model {c} != {num_ls}"
    return l_matrix



def get_rank_bandwidth_variable_matrix_gurobi(model: gp.Model,
                                              num_ranks: int) -> np.ndarray:
    """
    Extracts the bandwidth variable matrix from the GUROBI model.
    FIXME: Code duplication with get_rank_latency_variable_matrix_gurobi
    """
    # The number of g variables should be equal to the number of
    # pairs of ranks
    num_gs = (1 + (num_ranks - 1)) * (num_ranks - 1) // 2
    g_matrix = np.empty((num_ranks, num_ranks), dtype=gp.Var)
    c = 0
    for i in range(num_ranks):
        g_matrix[i][i] = 0
        for j in range(i + 1, num_ranks):
            g = model.getVarByName(f"g[{c}]")
            if g is None:
                raise ValueError(f"[ERROR] Variable g[{c}] does not exist in "
                                 "the model. Make sure pairwise analysis is "
                                 "enabled.")
            g_matrix[i, j] = g_matrix[j, i] = g
            c += 1
    assert c == num_gs, f"[ERROR] Number of g variables in the model {c} != {num_gs}"
    return g_matrix


def get_pairwise_sensitivity_matrix_gurobi(var_matrix: np.ndarray,
                                           symmetric: bool = True) \
                                            -> np.ndarray:
    """
    Extracts the pairwise sensitivity matrix from the given matrix of
    GUROBI variables.
    @param var_matrix: A matrix of variables.
    @param symmetric: If True, the returned matrix is symmetric, otherwise
    returns an upper triangular matrix.
    @return: A symmetric matrix of integers.
    """
    res = np.zeros_like(var_matrix, dtype=int)
    for i in range(var_matrix.shape[0]):
        for j in range(i + 1, var_matrix.shape[1]):
            if symmetric:
                res[i, j] = res[j, i] = var_matrix[i, j].RC
            else:
                res[i, j] = var_matrix[i, j].RC
    return res


def get_pairwise_sensitivity_matrix_ortools(l_matrix: np.ndarray,
                                            symmetric: bool = True) \
                                            -> np.ndarray:
    """
    Extracts the pairwise sensitivity matrix from the given matrix of
    latency variables.
    @param l_matrix: A matrix of latency variables.
    @param symmetric: If True, the returned matrix is symmetric, otherwise
    returns an upper triangular matrix.
    @return: A symmetric matrix of integers.
    """
    res = np.zeros_like(l_matrix, dtype=int)
    for i in range(l_matrix.shape[0]):
        for j in range(i + 1, l_matrix.shape[1]):
            if symmetric:
                res[i, j] = res[j, i] = l_matrix[i, j].reduced_cost()
            else:
                res[i, j] = l_matrix[i, j].reduced_cost()

# ====================================
# =========== Decorators =============
# ====================================

def measure_time(info: str):
    """
    A decorator that measures the execution time of the
    given function.
    """
    def wrapper_with_info(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            verbose = kwargs.get("verbose", False)
            if verbose:
                print(f"[INFO] Computing {info}...", flush=True)
            start = time()
            res = func(*args, **kwargs)
            time_taken = time() - start
            if verbose:
                print(f"[INFO] {info.capitalize()} computed in {time_taken:.3f} seconds.")
            return res
        return wrapper
    return wrapper_with_info


def experimental(info: str):
    """
    A decorator that marks the given function as experimental.
    """
    def wrapper_with_info(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[WARNING] {info.capitalize()} is an experimental feature", flush=True)
            return func(*args, **kwargs)
        return wrapper
    return wrapper_with_info


def write_placement_to_file(placement: List[List[int]], filename: str,
                            format: str = "slurm",
                            host_names: Optional[List[str]] = None) -> None:
    """
    Writes the given rank placement to the output file in a format that
    is compatible with the given MPI implementation specified by
    the `format` argument.
    If the format is "cray_mpich", the placement will be a list of ranks
    where the integer at each position indicates the rank to which the
    the process should be mapped.
    E.g., if the placement is [[0,2], [1,3]], the output will be "0,2,1,3".
    If the format is "mpich", the placement will be
    a list of node IDs where the integer at each position indicates the
    node to which the process should be mapped.
    E.g., if the placement is [[0,2], [1,3]], the output will be "0,1,0,1".
    If the format is "slurm", the placement will be a list of hostnames
    where the name at each position indicates the host to which the
    process should be mapped.
    E.g., if the placement is [[0, 2], [1, 3]],
    the output will be
    "host0\nhost1\nhost0\nhost1"
    """
    assert len(placement) > 0, "[ERROR] Empty placement."
    assert len(placement[0]) > 0, "[ERROR] Empty placement."

    res = [0 for _ in range(len(placement) * len(placement[0]))]
    for n in range(len(placement)):
        for r in range(len(placement[n])):
            res[placement[n][r]] = n
    
    # TODO: Add support for other formats
    print(f"[INFO] Using format: '{format}' to write the placement to '{filename}")
    if format == "mpich":
        with open(filename, "w") as f:
            f.write(",".join(map(str, res)) + "\n")
    elif format == "cray_mpich":
        placement.sort()
        flattened_placement = [str(rank) for ranks in placement for rank in ranks]
        with open(filename, "w") as f:
            f.write(",".join(flattened_placement) + "\n")
    elif format == "slurm":
        if host_names is None:
            raise ValueError("[ERROR] Missing host names for OpenMPI format.")
        with open(filename, "w") as f:
            for node_id in enumerate(res):
                f.write(f"{host_names[node_id]}\n")
    else:
        raise ValueError(f"[ERROR] Unsupported format: {format}")
    


def save_dict_to_csv(data: dict, filename: str, column_names: List[str],
                     verbose: bool) -> None:
    """
    Saves the given dictionary to a CSV file with the given filename 
    and column names.
    """
    with open(filename, "w") as f:
        f.write(",".join(column_names) + "\n")
        for key, value in data.items():
            f.write(f"{key},{value}\n")
    if verbose:
        print(f"[INFO] Saved the dictionary to {filename}.", flush=True)
    