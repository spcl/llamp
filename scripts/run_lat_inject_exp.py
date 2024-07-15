# Runs latency injection experiments

import os
import re
import argparse
import pandas as pd
from time import time
from typing import List, Optional, Dict, Tuple, Union, Callable
from subprocess import run, PIPE, TimeoutExpired


ENV_VAR_NAME = "INJECTED_LATENCY"
TIMER_LIB = os.environ.get("LIBTIMER_C", None)
TIMER_LIB_F77 = os.environ.get("LIBTIMER_F77", None)

def parse_icon_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from LULESH and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Total runtime:\s+(\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime

def parse_milc_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from MILC and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime
    

def parse_lammps_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from LAMMPS and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime

def parse_openmx_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from OpenMX and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime


def parse_namd_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from NAMD and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Wallclock: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime


def parse_lulesh_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from LULESH and obtains the
    runtime of the program.
    """
    assert out_str is not None, "[ERROR] Invalid output"
    
    # regex_pattern = r"(\d+\.\d+)\soverall"
    # matches = re.findall(regex_pattern, out_str)
    # assert len(matches) > 0, "[ERROR] Invalid output"
    # return float(matches[0])
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime
    

def parse_hpcg_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from HPCG and obtains the runtime
    of the program.
    """
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime
    
    # Deletes all the output files produced by HPCG
    proc = run(['rm -f hpcg2024* HPCG-Benchmark_*'], stdout=PIPE, stderr=PIPE, shell=True)
    if proc.returncode != 0:
        # If an error occurred while deleting files
        print("[ERROR] Process error:")
        print(proc.stderr.decode())
        exit(1)

    return max_runtime
    

def parse_examinimd_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from ExaMiniMD and obtains the runtime
    of the program.
    """
    regex_pattern = r"Total runtime: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime

def parse_microbenchmark_runtime(out_str: Optional[str]) -> float:
    """
    Parses the output from the microbenchmark and obtains the runtime
    of the program.
    """
    regex_pattern = r"Total time: (\d+.\d+)"
    matches = re.findall(regex_pattern, out_str)
    assert len(matches) > 0, "[ERROR] Invalid output"
    
    max_runtime = 0
    for match in matches:
        runtime = float(match)
        if runtime > max_runtime:
            max_runtime = runtime

    return max_runtime

# A dictionary that maps each application to
# a function that parses its output in order
# to obtain an accurate measure of the runtime
# of the corresponding program
runtime_parse_fn: Dict[str, Callable] = {
    "lulesh": parse_lulesh_runtime,
    "hpcg"  : parse_hpcg_runtime,
    "minimd": parse_examinimd_runtime,
    "namd"  : parse_namd_runtime,
    "openmx": parse_openmx_runtime,
    "lammps": parse_lammps_runtime,
    "milc"  : parse_milc_runtime,
    "icon"  : parse_icon_runtime,
    "microbenchmark": parse_microbenchmark_runtime,
}


def exec_command(command: Union[List[str], str], timeout: int,
                 return_output: bool = True) -> Optional[str]:
    """
    A helper function that runs the given command in a subprocess.
    If `return_output` is set to True, will return the stdout of the
    subprocess as output.
    """
    if isinstance(command, str):
        command = command.split()
    
    proc = None
    # If an error occurs when running the experiment, just rerun the
    # same command
    while True:
        try:
            proc = run(command, stdout=PIPE, stderr=PIPE, timeout=timeout)

            if proc.returncode == 0:
                break

            if proc.returncode != 0:
                # If an error occurred while the script was running
                print("[ERROR] Process error:")
                print(proc.stderr.decode(), flush=True)
                print("[INFO] Restarting the trial", flush=True)
                # cleanup_proc = run(["./cleanup_icon.sh"], stdout=PIPE, stderr=PIPE, timeout=5)
                # assert cleanup_proc.returncode == 0
                # print("[INFO] Cleaning up...", flush=True)
        except TimeoutExpired:
            print("[WARNING] Timed out, restarting the trial", flush=True)
            cleanup_proc = run(["./cleanup_icon.sh"], stdout=PIPE, stderr=PIPE, timeout=5)
            assert cleanup_proc.returncode == 0
            print("[INFO] Cleaning up...", flush=True)
            
    if return_output:
        return proc.stdout.decode()


def run_experiment(command: str, app: str, num_procs: int, res_file: str,
                   L_interval: Tuple[int, int], step: int, num_trials: int,
                   hostfile: Optional[str], timeout: int,
                   omp_num_threads: int, size_param: str) -> None:
    """
    Performs the latency injection experiment.
    Executes the given `command` with `mpirun` the injected latencies
    are defined by `L_interval` and `step`, and they will be between
    `L_min` and `L_max`. For each L, we will execute `mpirun` `num_trial` times.
    The result will be saved to `res_file` as csv.
    """
    L_min, L_max = L_interval
    mpirun_bin = "mpirun"
    assert mpirun_bin is not None
    hostfile_arg = "" if hostfile is None else f"-f {hostfile}"
    if app == "icon":
        # ICON has its own run script
        base_command = f'{command} "mpirun -np {num_procs} {hostfile_arg} ' \
                          f'-envall -env OMP_NUM_THREADS {omp_num_threads} -env MPICH_ASYNC_PROGRESS 1 ' \
                            '-env UCX_RNDV_THRESH 256000 -env UCX_RC_VERBS_SEG_SIZE 256000 '
        sim_time = "00:30:00" if size_param == "test" else "06:00:00"
        
    else:
        mpirun_command = f"{mpirun_bin} -np {num_procs} {hostfile_arg} " \
                         f"-envall -env OMP_NUM_THREADS {omp_num_threads} -env MPICH_ASYNC_PROGRESS 1 " \
                         "-env UCX_RNDV_THRESH 256000 -env UCX_RC_VERBS_SEG_SIZE 256000 " \
                         f"-env LD_PRELOAD {TIMER_LIB} " \
                         f"{command}"
    
                     # "-env MPICH_PROGRESS_THREAD_AFFINITY auto " \

        print(f"[INFO] mpirun command to execute: {mpirun_command}")

    # The result CSV should contain the following columns
    # "L", "runtime"
    res_columns = ["L", "runtime"]
    res = []
    
    # Opens the output file that flushes immediately
    output_file = open(res_file, "w")
    output_file.write("L,runtime\n")
    num_runs = ((L_max + step - L_min) // step + 1) * num_trials

    print(f"[INFO] Total number of trials: {num_runs}")
    curr_runs = 0
    start_time = time()

    for L in range(L_min, L_max + step, step):
        print(f"[INFO] Running experiment for L: {L}")
        total_runtime = 0
        if app == "icon":
            mpirun_command = base_command + f'-env INJECTED_LATENCY {L}" "{sim_time}"'
            print(f"[INFO] mpirun command to execute: {mpirun_command}")
        for n in range(num_trials):
            print(f"Trial {n + 1} / {num_trials}:", flush=True)
            # Sets the enviornment variable INJECTED_LATENCY
            os.environ[ENV_VAR_NAME] = str(L)
            # Run experiments for one L many times
            out = exec_command(mpirun_command, timeout)
            fn = runtime_parse_fn.get(app)
            if fn is None:
                print(f"[ERROR] App {app} not supported")
                exit(-1)
            runtime = fn(out)
            total_runtime += runtime
            print("{:.4f} s".format(runtime), flush=True)
            output_file.write(f"{L},{runtime}\n")
            output_file.flush()

        curr_runs += num_trials
        print("[INFO] Average runtime for L={}: {:.3f} s".format(L, total_runtime / num_trials))
        end_L_time = time()
        time_elapsed = end_L_time - start_time
        print(f"[INFO] Time elapsed: {time_elapsed:.3f} s", flush=True)
        eta_secs = time_elapsed / curr_runs * (num_runs - curr_runs)
        print(f"[INFO] ETA: {eta_secs:.3f} s ({eta_secs / 3600:.2f} hrs)", flush=True)
        
    output_file.close()
    print(f"[INFO] Saved results to {res_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Run experiments for latency injection")
    
    parser.add_argument("-c", "--command", dest="command", default=None, required=True,
                        type=str, help="Path to the csv result file")
    parser.add_argument("-n", "--np", dest="num_procs", default=8, 
                       help="Number of processes to run for the MPI application")
    parser.add_argument("-r", "--result", dest="res_file", default="result.csv",
                        type=str, help="Command to execute after mpirun")
    parser.add_argument("--l-min", dest="L_min", default=1, type=int,
                        help="The minimum injected latency in us")
    parser.add_argument("--l-max", dest="L_max", default=100, type=int,
                        help="The maximum injected latency in us")
    parser.add_argument("-s", "--step", dest="step", default=1, type=int,
                        help="The increment from L_min to L_max")
    parser.add_argument("-t", "--num-trials", dest="num_trials", default=10, type=int,
                        help="Number of trials an experiment for a specific L will be run")
    parser.add_argument("-a", "--app", dest="app", required=True, type=str,
                        help="Name of the application.")
    parser.add_argument("--hostfile", dest="hostfile", default=None,
                        help="The hostfile for the MPI application")
    parser.add_argument("--timeout", dest="timeout", type=int, default=20,
                        help="Timeout for each trial of the experiment")
    parser.add_argument("--size-param", dest="size_param", default="test",
                        type=str, help="Size parameter for the application (test or paper). "
                        "Only has an effect when the application is ICON.")
    parser.add_argument("-o", "--omp-num-threads", dest="omp_num_threads", default=4,
                        type=int, help="Number of OpenMP threads to use")

    args = parser.parse_args()

    L_min = args.L_min
    L_max = args.L_max
    assert L_min <= L_max
    print(f"[INFO] Running latency injection experiment: {args.app}...")
    print(f"[INFO] Command: {args.command}")
    print(f"[INFO] Num processes: {args.num_procs}")
    print(f"[INFO] L: [{L_min}, {L_max}], step: {args.step}")
    print(f"[INFO] Num trials per L: {args.num_trials}");
    print(f"[INFO] Timeout: {args.timeout}")
    print(f"[INFO] Result file: {args.res_file}")
    
    run_experiment(args.command, args.app, args.num_procs, args.res_file,
                   (L_min, L_max), args.step, args.num_trials,
                   args.hostfile, args.timeout, args.omp_num_threads,
                   args.size_param)
    
