import os
import argparse
from typing import List, Optional
from subprocess import run, PIPE, TimeoutExpired

"""
Script used to measure the speedup of the Gurobi solver over LogGOPSim
over a number of selected benchmarks. This is used to reproduce the
results from figure 7 in the LLAMP paper.
"""

# def collect_traces(trace_dir: str, command: str,
#                    is_fortran: bool, timeout: int,
#                    verbose: bool = False) -> str:
#     """
#     Collects MPI traces using the given command and saves them to the
#     provided trace directory.
#     @param trace_dir: Path to the directory where the traces will be saved.
#     @param project: Name of the project.
#     @param command: The command or script to run in order to execute the
#     application and generate the MPI traces.
#     @param is_fortran: Flag to indicate if the application is a
#     Fortran application. This is used to set the appropriate
#     tracing library to use.
#     @param timeout: Timeout in seconds for running the command.
#     """
#     tokens = command.split()
#     # Sets the trace directory
#     os.environ["HTOR_PMPI_FILE_PREFIX"] = f"{trace_dir}/pmpi-trace-rank-"
#     # Sets the liballprof shared library
#     if is_fortran:
#         assert "LIBALLPROF_F77" in os.environ, "[ERROR] LIBALLPROF_F is not set"
#         os.environ["LD_PRELOAD"] = os.environ["LIBALLPROF_F77"]
#     else:
#         assert "LIBALLPROF_C" in os.environ, "[ERROR] LIBALLPROF_C is not set"
#         os.environ["LD_PRELOAD"] = os.environ["LIBALLPROF_C"]

#     try:
#         proc = run(tokens, stdout=PIPE, stderr=PIPE, timeout=timeout)
#         rc = proc.returncode
#         if verbose:
#             print("[INFO] Command stdout:")
#             print(proc.stdout.decode("utf-8"))
        
#         if rc == 0 and verbose:
#             print("[INFO] Command execution: SUCCESS", flush=True)
        
#         if proc.returncode != 0:
#             print(f"[ERROR] Command failed {rc}: {proc.stderr.decode('utf-8')}", flush=True)
#             exit(1)

#     except TimeoutExpired:
#         print(f"[ERROR] Command timed out after {timeout} seconds.", flush=True)
#         exit(1)

    
#     # Checks if the trace directory is empty
#     if not os.listdir(trace_dir):
#         print("[ERROR] No trace files were generated. Exiting...", flush=True)
#         exit(1)


def collect_data(data_dir: str, size_param: str, verbose: bool) -> None:
    """
    Collects the traces from NPB benchmarks LULESH and LAMMPS
    """
    os.environ["OMP_NUM_THREADS"] = "1"
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
        command = f"mpirun -np {np} ../apps/NPB3.4.3/NPB3.4-MPI/bin/{benchmark_name}"
        out_dir = f"{data_dir}/{benchmark_name}"
        if os.path.exists(out_dir):
            print(f"[INFO] Deleting trace directory: {out_dir}")
            os.system(f"rm -rf {out_dir}")
        
        # Uses the script lp_gen.py to generate the LP file
        # for the given benchmark
        project_name = f"npb_{benchmark}"
        lp_gen_command = f'python3 lp_gen.py -c "{command}" -p {project_name} {v_arg} --f77 --timeout 600'
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to generate LP file for {benchmark_name}")
            exit(1)
        
        # Moves the project directory from the parent directory to out_dir
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move {project_name} to {out_dir}")
            exit(1)
    
    # Collect traces for LULESH
    if size_param == "test":
        np = 8
    else:
        np = 216
    
    assert os.path.exists("../apps/lulesh/build/lulesh2.0"), "[ERROR] LULESH does not exist"
    command = f"mpirun -np {np} ../apps/lulesh/build/lulesh2.0 -i 1000 -s 8"
    out_dir = f"{data_dir}/lulesh"
    if os.path.exists(out_dir):
        print(f"[INFO] Deleting trace directory: {out_dir}")
        os.system(f"rm -rf {out_dir}")

    lp_gen_command = f'python3 lp_gen.py -c "{command}" -p lulesh -v --timeout 600'
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
    
    command = f"mpirun -np {np} {lammps_dir}build/lmp -in {lammps_dir}bench/in.eam"
    out_dir = f"{data_dir}/lammps"
    if os.path.exists(out_dir):
        print(f"[INFO] Deleting trace directory: {out_dir}")
        os.system(f"rm -rf {out_dir}")
    
    lp_gen_command = f'python3 lp_gen.py -c "{command}" -p lammps -v --timeout 600'
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

    if verbose:
        print("[INFO] Running speed test for Gurobi...", flush=True)
    
    # Gurobi speed test
    
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speedup script")
    parser.add_argument("-d", "--data-dir", type=str, default="speedup_data",
                        help="Path to the directory containing all the collected data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-s", "--size", type=str, default="test",
                        help="Size of the parameters to run for the applications. Options are 'test' and 'paper' "\
                            "where 'test' runs the applications with a small input size and 'paper' runs the applications "\
                            "with a large input size. Default is 'test'.")

    args = parser.parse_args()

    # Checks if the data directory exists
    if os.path.exists(args.data_dir):
        # Deletes the directory
        print(f"[INFO] Deleting data directory: {args.data_dir}")
        os.system(f"rm -rf {args.data_dir}")
    print(f"[INFO] Creating data directory: {args.data_dir}")
    os.makedirs(args.data_dir)
        

    # The script consists of 3 steps
    # 1. Collect trace data
    # 2. Run the speedup test for Gurobi and LogGOPSim
    # 3. Generate the plot
    collect_data(args.data_dir, args.size, args.verbose)
    
    # Run the speedup test for Gurobi and LogGOPSim
    test_speed(args.verbose)
