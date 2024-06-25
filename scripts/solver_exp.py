
import argparse
import os
import subprocess
import psutil
from subprocess import run, PIPE, TimeoutExpired
from typing import Union
from time import time
import gurobipy as gp

"""
This file contains code needed to run experiments to compare
the speed of the Gurobi LP solver to LogGOPSim.

FIXME I know there is a lot of code duplication here, but I don't
really care since it's just a one-off script. Bite me.
"""
# ============================================================
# Utility functions
# ============================================================

def is_gurobi_installed() -> bool:
    """
    Checks if the Gurobi solver is installed on the machine
    by running the command "gurobi_cl --version".
    """
    try:
        subprocess.check_output(["gurobi_cl", "--version"])
        return True
    except:
        return False


def load_model(model_path: str, verbose: bool = True, solver: str = "gurobi") \
                -> gp.Model:
    """
    Loads the LP model in MPS format from the given path.
    """
    if verbose:
        print(f"[INFO] Loading the LP model from {model_path}...", flush=True)
    
    if solver == "gurobi":
        assert is_gurobi_installed(), "[ERROR] GUROBI is not installed on the machine"
        # Loads a GUROBI model
        model = gp.read(model_path)
    else:
        raise ValueError(f"[ERROR] Unsupported solver: {solver}")
        
        
    if verbose:
        print("[INFO] Load LP model: SUCCESS", flush=True)

    return model

# ============================================================
# LogGOPSim speed test
# ============================================================
def loggopsim_speed_test(loggopsim_path: str, bin_path: str, num_runs: int = 10):
    """
    Runs the speed test for the LogGOPSim LP solver.
    """
    # asserts that the loggopsim path exists
    assert os.path.exists(loggopsim_path), f"[ERROR] LogGOPSim does not exist: {loggopsim_path}"
    # asserts that the model path exists
    assert os.path.exists(bin_path), f"[ERROR] Bin path does not exist: {bin_path}"
    # Runs the speed test
    print("[INFO] Running speed test for LogGOPSim...", flush=True)
    total_time = 0
    L = 3000
    for i in range(num_runs):
        command = f"{loggopsim_path} -f {bin_path} -L {L}"
        L += 1000
        print(f"[INFO] Command to execute: {command}", flush=True)
        start_time = time()
        result = run(command, shell=True, stdout=PIPE, stderr=PIPE, timeout=1800)
        if result.returncode != 0:
            print(f"[ERROR] LogGOPSim failed: {result.stderr.decode('utf-8')}", flush=True)
            return
        end_time = time()
        total_time += end_time - start_time
        print(f"[INFO] Trial {i + 1} / {num_runs} time: {end_time - start_time} seconds", flush=True)
    print(f"[INFO] LogGOPSim speed test: {end_time - start_time} seconds", flush=True)
    print(f"[INFO] Average time: {total_time / num_runs} seconds", flush=True)

# ============================================================
# Gurobi speed test
# ============================================================

def gurobi_speed_test(model_path: str, num_runs: int = 10):
    """
    Runs the speed test for the GUROBI LP solver.
    """
    assert is_gurobi_installed(), "[ERROR] GUROBI is not installed on the machine"
    # asserts that the model path exists
    assert os.path.exists(model_path), f"[ERROR] Model path does not exist: {model_path}"
    # Loads the file using the utility function
    model = load_model(model_path)
    print(f"[INFO] Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB", flush=True)
    assert model is not None, "[ERROR] Failed to load the model"

    model.setParam("Threads", 0)
    # model.setParam("LPWarmStart", 1)
    model.setParam("Method", -1)
    model.setParam("LogToConsole", 1)
    # model.setParam("Presolve", 2)
        # model.setParam("BarOrder", 1)
    l = model.getVarByName("l")
    g = model.getVarByName("g")
    assert l is not None, "[ERROR] Variable l does not exist"
    # assert g is not None, "[ERROR] Variable g does not exist"
    if g is not None:
        g.lb = 1
    l.lb = 3000
    print(f"[INFO] Initial lower bound for l: {l.lb}")
    
    print("[INFO] Presolving the model...")
    presolved_model = model.presolve()
    presolved_model.printStats()
    reset_model = False
    if presolved_model.NumVars == 0:
        reset_model = True
    model.reset()

    print(f"[INFO] Reset model: {reset_model}")

    # Runs the speed test
    print("[INFO] Running speed test for GUROBI...", flush=True)
    total_time = 0
    for i in range(num_runs):
        if reset_model:
            model.reset()
        start_time = time()
        model.optimize()
        l.lb += 1000
        end_time = time()
        total_time += end_time - start_time
        print(f"[INFO] Trial {i + 1} / {num_runs} time: {end_time - start_time} seconds", flush=True)
    print(f"[INFO] GUROBI speed test: {end_time - start_time} seconds", flush=True)
    print(f"[INFO] Average time: {total_time / num_runs} seconds", flush=True)


if __name__ == "__main__":
    # Parses user args
    parser = argparse.ArgumentParser(description="Run speed tests on different LP solvers")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the LP model in lp format if using Gurobi."
                        "Path to the bin file if using LogGOPSim.")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs to average over")
    parser.add_argument("--solver", type=str, required=True, choices=["gurobi", "loggopsim"],
                        help="Solver to test")
    parser.add_argument("--loggopsim-bin", type=str,
                        default="/users/sshen/workspace/lgs-mpi/LogGOPSim/LogGOPSim",
                        help="Path to the LogGOPSim binary file [PLEASE CHANGE THIS TO YOUR PATH]")

    args = parser.parse_args()
    print(f"[INFO] Running speed test for {args.solver} solver...", flush=True)
    print(f"[INFO] Model path: {args.model}", flush=True)
    print(f"[INFO] Number of runs: {args.num_runs}", flush=True)

    if args.solver == "gurobi":
        gurobi_speed_test(args.model, args.num_runs)
    elif args.solver == "loggopsim":
        loggopsim_speed_test(args.loggopsim_bin, args.model, args.num_runs)
    else:
        raise ValueError(f"[ERROR] Unsupported solver: {args.solver}")
