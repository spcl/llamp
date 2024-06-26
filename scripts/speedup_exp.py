import os
import argparse
import re
from typing import List, Optional
from subprocess import run, PIPE, TimeoutExpired
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import matplotlib.patches as patches
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore')

"""
Script used to measure the speedup of the Gurobi solver over LogGOPSim
over a number of selected benchmarks. This is used to reproduce the
results from figure 7 in the LLAMP paper.
"""

def collect_data(data_dir: str, size_param: str,
                 hostfile: Optional[str], verbose: bool) -> None:
    """
    Collects the traces from NPB benchmarks LULESH and LAMMPS
    """
    if verbose:
        print(f"[INFO] Using size parameter: {size_param}", flush=True)
        print(f"[INFO] hostfile: {hostfile}", flush=True)

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"{hostfile_arg} --envall"

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["INJECTED_LATENCY"] = "0"
    # Collect traces for NPB
    npb_benchmarks = [ "ep", "mg", "ft", "cg", "bt", "sp", "lu" ]
    if size_param == "test":
        CLASS = "A"
        np = 16
    else:
        CLASS = "C"
        np = 256
    
    v_arg = "-v" if verbose else ""
    for benchmark in npb_benchmarks:
        benchmark_name = f"{benchmark}.{CLASS}.x"
        command = f"mpirun {mpirun_args} -np {np} ../apps/NPB3.4.3/NPB3.4-MPI/bin/{benchmark_name}"
        out_dir = f"{data_dir}/{benchmark_name}"
        if os.path.exists(out_dir):
            print(f"[INFO] Deleting trace directory: {out_dir}")
            os.system(f"rm -rf {out_dir}")
        
        # Uses the script lp_gen.py to generate the LP file
        # for the given benchmark
        lp_gen_command = f'python3 lp_gen.py -c "{command}" -p {benchmark_name} {v_arg} --f77 --timeout 600'
        print(f"[INFO] Running command: {lp_gen_command}")
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to generate LP file for {benchmark_name}")
            exit(1)
        
        # Moves the project directory from the parent directory to out_dir
        if os.system(f"mv ../{benchmark_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{benchmark_name}' to {out_dir}")
            exit(1)
    
    # Collect traces for LULESH
    if size_param == "test":
        np = 8
    else:
        np = 216
    
    assert os.path.exists("../apps/lulesh/build/lulesh2.0"), "[ERROR] LULESH does not exist"
    command = f"mpirun {mpirun_args} -np {np} ../apps/lulesh/build/lulesh2.0 -i 1000 -s 8"
    out_dir = f"{data_dir}/lulesh"
    if os.path.exists(out_dir):
        print(f"[INFO] Deleting trace directory: {out_dir}")
        os.system(f"rm -rf {out_dir}")

    lp_gen_command = f'python3 lp_gen.py -c "{command}" -p lulesh -v --timeout 600'
    print(f"[INFO] Running command: {lp_gen_command}")
    if os.system(lp_gen_command) != 0:
        print(f"[ERROR] Failed to generate LP file for LULESH")
        exit(1)
    # Moves the project directory from the parent directory to out_dir
    if os.system(f"mv ../lulesh {out_dir}") != 0:
        print(f"[ERROR] Failed to move lulesh to {out_dir}")
        exit(1)

    # Collect traces for LAMMPS
    # Copies the file Cu.eam to the current directory
    lammps_dir = "../apps/lammps/"
    if os.system(f"cp {lammps_dir}bench/Cu_u3.eam .") != 0:
        print("[ERROR] Failed to copy Cu_u3.eam file")
        exit(1)
    # Replaces line 3 in in.eam with the following:
    # 'variable         x index 4'
    if size_param == "test":
        size = 2
        np = 16
    else:
        size = 4
        np = 512
    
    if os.system(f"sed -i '3s/.*/variable         x index {size}/' {lammps_dir}bench/in.eam") != 0:
        print("[ERROR] Failed to replace line 3 in in.eam")
        exit(1)
    # Replaces line 4 in in.eam with the following:
    # 'variable         y index 4'
    if os.system(f"sed -i '4s/.*/variable         y index {size}/' {lammps_dir}bench/in.eam") != 0:
        print("[ERROR] Failed to replace line 4 in in.eam")
        exit(1)
    # Replaces line 5 in in.eam with the following:
    # 'variable         z index 4'
    if os.system(f"sed -i '5s/.*/variable         z index {size}/' {lammps_dir}bench/in.eam") != 0:
        print("[ERROR] Failed to replace line 5 in in.eam")
        exit(1)
    
    command = f"mpirun {mpirun_args} -np {np} {lammps_dir}build/lmp -in {lammps_dir}bench/in.eam"

    out_dir = f"{data_dir}/lammps"
    if os.path.exists(out_dir):
        print(f"[INFO] Deleting trace directory: {out_dir}")
        os.system(f"rm -rf {out_dir}")
    
    lp_gen_command = f'python3 lp_gen.py -c "{command}" -p lammps -v --timeout 600'
    print(f"[INFO] Running command: {lp_gen_command}")
    if os.system(lp_gen_command) != 0:
        print(f"[ERROR] Failed to generate LP file for LAMMPS")
        exit(1)

    # Moves the project directory from the parent directory to out_dir
    if os.system(f"mv ../lammps {out_dir}") != 0:
        print(f"[ERROR] Failed to move lammps to {out_dir}")
        exit(1)


def test_speed(data_dir: str, verbose: bool) -> None:
    """
    Runs the speed test for Gurobi and LogGOPSim and produces a csv file
    that contains the results.
    """
    res_file_path = f"{data_dir}/speedup_results.csv"
    if os.path.exists(res_file_path):
        os.system(f"rm {res_file_path}")

    res_file = open(res_file_path, "w")
    res_file.write("benchmark,method,runtime\n")
    res_file.flush()

    if verbose:
        print("[INFO] Running speed test for Gurobi...", flush=True)
    
    num_runs = 10
    for benchmark in os.listdir(data_dir):
        benchmark_path = f"{data_dir}/{benchmark}"
        if not os.path.isdir(benchmark_path):
            # Should not happen
            continue
        print(f"[INFO] Running speed test for {benchmark}...", flush=True)
        model_path = f"{benchmark_path}/output/{benchmark}.lp"
        assert os.path.exists(model_path), f"[ERROR] Model path does not exist: {model_path}"

        # Runs the Gurobi solver with the solver_exp.py script
        command = f"python3 solver_exp.py --model {model_path} --solver gurobi --num-runs {num_runs}"
        # Gurobi speed test
        if verbose:
            print(f"[INFO] Running Gurobi for {benchmark}...", flush=True)
        proc = run(command.split(), stdout=PIPE, stderr=PIPE)
        rc = proc.returncode
        stdout = proc.stdout.decode("utf-8")
        stderr = proc.stderr.decode("utf-8")
        if rc != 0:
            print(f"[ERROR] Gurobi failed for {benchmark}: {stderr}", flush=True)
            continue
        # Extracts the time from the stdout with the given regex pattern
        # Trial \d+ \/ \d+ time: (\d+\.\d+) seconds
        pattern = r"Trial \d+ \/ \d+ time: (\d+\.\d+) seconds"
        times = re.findall(pattern, stdout)
        assert len(times) == num_runs, f"[ERROR] Number of times does not match: {len(times)} != {num_runs}"
        # Writes the results to the csv file
        for time in times:
            res_file.write(f"{benchmark},gurobi,{time}\n")
            res_file.flush()

        # Runs the LogGOPSim solver with the solver_exp.py script
        bin_file_path = f"{benchmark_path}/{benchmark}.bin"
        assert os.path.exists(bin_file_path), f"[ERROR] Bin file does not exist: {bin_file_path}"
        loggopsim_path = os.environ.get("LOGGOPSIM_PATH")
        assert loggopsim_path is not None, "[ERROR] LOGGOPSIM_PATH is not set"
        command = f"python3 solver_exp.py --model {bin_file_path} --solver loggopsim --num-runs {num_runs} --loggopsim-bin {loggopsim_path}"

        if verbose:
            print(f"[INFO] Running LogGOPSim for {benchmark}...", flush=True)
        proc = run(command.split(), stdout=PIPE, stderr=PIPE)
        rc = proc.returncode
        stdout = proc.stdout.decode("utf-8")
        stderr = proc.stderr.decode("utf-8")
        if rc != 0:
            print(f"[ERROR] LogGOPSim failed for {benchmark}: {stderr}", flush=True)
            continue
        # Extracts the time from the stdout with the given regex pattern
        # Trial \d+ \/ \d+ time: (\d+\.\d+) seconds
        pattern = r"Trial \d+ \/ \d+ time: (\d+\.\d+) seconds"
        times = re.findall(pattern, stdout)
        assert len(times) == num_runs, f"[ERROR] Number of times does not match: {len(times)} != {num_runs}"
        # Writes the results to the csv file
        for time in times:
            res_file.write(f"{benchmark},loggopsim,{time}\n")
            res_file.flush()

    res_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speedup script")
    parser.add_argument("-d", "--data-dir", type=str, default="speedup_data",
                        help="Path to the directory containing all the collected data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-s", "--size", type=str, default="test",
                        help="Size of the parameters to run for the applications. Options are 'test' and 'paper' "\
                            "where 'test' runs the applications with a small input size and 'paper' runs the applications "\
                            "with a large input size. Default is 'test'.")
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Path to the hostfile. If provided, will execute the applications on multiple hosts.')

    args = parser.parse_args()

    # Checks if the data directory exists
    if os.path.exists(args.data_dir):
        # Deletes the directory
        print(f"[INFO] Deleting data directory: {args.data_dir}")
        os.system(f"rm -rf {args.data_dir}")
    print(f"[INFO] Creating data directory: {args.data_dir}")
    os.makedirs(args.data_dir)

    # The script consists of 2 steps
    # 1. Collect trace data
    # 2. Run the speedup test for Gurobi and LogGOPSim
    collect_data(args.data_dir, args.size, args.hostfile, args.verbose)
    
    # Run the speedup test for Gurobi and LogGOPSim
    test_speed(args.data_dir, args.verbose)
