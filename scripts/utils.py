import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from subprocess import run, PIPE
from typing import Optional, List, Tuple

"""
Utility functions used by the scripts in this directory.
"""

GIB_TO_B = 1073741824

def collect_net_params(hostfile: Optional[str] = None,
                       file_path: str = "netparams.csv") -> None:
    """
    Collects the network parameters using Netgauge and saves
    the results to a CSV file.
    If hostfile is provided, the network parameters are collected
    in the inter-node context.
    """
    print(f"[INFO] Collecting network parameters in ns...", flush=True)

    # Runs Netgauge for 60 seconds
    hostfile_arg = "" if hostfile is None else f"-f {hostfile}"
    command = f"mpirun -env INJECTED_LATENCY=0 -np 2 {hostfile_arg} netgauge -m mpi -x loggp"
    print(f"[INFO] Command to execute: {command}", flush=True)
    result = run(command, shell=True, stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        print(f"[ERROR] Netgauge failed: {result.stderr.decode('utf-8')}", flush=True)
        return
    
    params_file = open(file_path, "w")
    params_file.write("L,s,o,G\n")
    params_file.flush()

    # Parses the output
    output = result.stdout.decode("utf-8")
    for line in output.split("\n"):
        line = line.strip()
        if not line.startswith("L"):
            continue
        tokens = line.split()
        assert len(tokens) == 9, f"[ERROR] Invalid line: {line}"
        L = int(float(tokens[0].split("=")[1]) * 1000)
        s = int(tokens[1].split("=")[1])
        o = int(float(tokens[2].split("=")[1]) * 1000)
        bandwidth = float(tokens[6][1:]) * GIB_TO_B
        G = 1 / bandwidth * (10 ** 9)
        params_file.write(f"{L},{s},{o},{G}\n")
    
    params_file.close()
    print(f"[INFO] Network parameters saved to {file_path}", flush=True)



def get_net_params(s: int, netparam_file: str = "netparams.csv") -> Tuple:
    """
    Returns the network parameters L, o, and G for a given s
    as per the netparam file.
    """
    assert os.path.exists(netparam_file), \
        f"[ERROR] Netparam file does not exist: {netparam_file}"

    df = pd.read_csv(netparam_file)
    # Finds the row that has the closest value of s
    row = df.iloc[(df["s"] - s).abs().argsort()[:1]]
    L = row["L"].values[0]
    o = row["o"].values[0]
    # G is the average G value across the rows after the frist 3 rows
    # Makes sure that G is positive
    G = df["G"].values[3:].mean()
    assert G > 0, "[ERROR] Invalid value for G"
    return L, o, G


def get_avg_message_size(goal_file: str) -> int:
    """
    Computes the average message size from the goal file.
    """
    total_bytes = 0
    message_count = 0
    with open(goal_file, "r") as f:
        for line in f:
            if line.startswith("l"):
                tokens = line.split()
                if tokens[1] == "send" and tokens[2][-1] == 'b':
                    total_bytes += int(tokens[2][:-1])
                    message_count += 1

    return int(total_bytes / message_count)



def get_baseline_runtime(runtime_file: str) -> float:
    """
    Returns the predicted baseline runtime for the model, i.e., predicted
    runtime of the program when the injected L is 0.
    """
    df = pd.read_csv(runtime_file)
    # Returns the value of runtime with the smallest L
    return df["runtime"].min()




def choose_schedgen_allreduce_algorithm(schedgen_dir: str,
                                        allreduce_alg: str = "recdoub") -> None:
    """
    Chooses the allreduce algorithm used by Schedgen to produce
    the execution graph. The options are 'recdoub' and 'ring'.
    This requires recompiling 'schedgen' inside the 'schedgen_dir' directory.
    """
    assert allreduce_alg in ["recdoub", "ring"], \
        f"[ERROR] Invalid allreduce algorithm: {allreduce_alg}"
    
    # Makes sure that the file 'process_trace.cpp' exists
    # in the 'schedgen_dir' directory
    process_trace_file = f"{schedgen_dir}/process_trace.cpp"
    assert os.path.exists(process_trace_file), \
        f"[ERROR] process_trace.cpp does not exist: {process_trace_file}"
    
    # If 'recdoub' is chosen, comments out line 1333 and 
    # uncomments line 1334 in 'process_trace.cpp' using sed
    if allreduce_alg == "recdoub":
        # Checks if line 1333 is already commented out, if it is not,
        # comments out line 1333 and uncomments line 1334 in 'process_trace.cpp' using sed
        command = f"sed -n '1333p' {process_trace_file} | grep '//'"
        proc = run(command, shell=True, stdout=PIPE, stderr=PIPE)
        if proc.returncode == 1:
            if os.system(f"sed -i '1333s/^/\\/\\//' {process_trace_file}") != 0:
                print(f"[ERROR] Failed to comment out line 1333 in {process_trace_file}")
                exit(1)
            if os.system(f"sed -i '1334s|//||' {process_trace_file}") != 0:
                print(f"[ERROR] Failed to uncomment line 1334 in {process_trace_file}")
                exit(1)
    else:
        # If 'ring' is chosen, comments out line 1334 and
        # uncomments line 1333 in 'process_trace.cpp' using sed
        command = f"sed -n '1334p' {process_trace_file} | grep '//'"
        proc = run(command, shell=True, stdout=PIPE, stderr=PIPE)
        # Checks if line 1334 is already commented out, if it is not,
        # comments out line 1334 and uncomments line 1333 in 'process_trace.cpp' using sed
        
        if proc.returncode == 1:
            if os.system(f"sed -i '1334s/^/\\/\\//' {process_trace_file}") != 0:
                print(f"[ERROR] Failed to comment out line 1334 in {process_trace_file}")
                exit(1)
            if os.system(f"sed -i '1333s|//||' {process_trace_file}") != 0:
                print(f"[ERROR] Failed to uncomment line 1333 in {process_trace_file}")
                exit(1)
    
    # Recompiles 'schedgen'
    # Changes the current directory to 'schedgen_dir'
    os.chdir(schedgen_dir)
    if os.system("make clean") != 0:
        print(f"[ERROR] Failed to clean schedgen")
        exit(1)
    if os.system("make") != 0:
        print(f"[ERROR] Failed to compile schedgen")
        exit(1)

    assert os.path.exists("schedgen"), "[ERROR] schedgen does not exist"
    print(f"[INFO] Successfully compiled schedgen with {allreduce_alg} algorithm")