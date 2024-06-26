import os
import pandas as pd
import warnings
from subprocess import run, PIPE
from typing import Optional, List, Tuple
warnings.filterwarnings("ignore")

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
    # G is the smallest positivie G across the rows after the frist 3 rows
    # Makes sure that G is positive
    G = df["G"].values[3:].min()
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



def 