import os
import re
import argparse
import pandas as pd
import warnings
from typing import Optional, List, Dict
from subprocess import run, PIPE
from utils import *
warnings.filterwarnings("ignore")

"""
This file contains code needed to run experiments for the
validation experiments.
"""

NET_PARAMS_FILE = "netparams.csv"


RES_FILE_NAMES = ["_lat_ratio.csv", "_lat_sensitivity.csv", "_lat_tolerance.csv", "_runtime.csv"]

def generate_lp_lulesh(data_dir: str, size_param: str = 'test',
                       hostfile: Optional[str] = None,
                       verbose: bool = False) -> None:
    """
    Generates the LP files for LULESH
    """
    if size_param == 'test':
        num_procs = [8, 27]
        data_size = 8
        iterations = 500
    else:
        num_procs = [8, 27, 64]
        data_size = 16
        iterations = 1000
    
    v_arg = "-v" if verbose else ""
    
    # Checks if the directory <data_dir>/lulesh exists
    if not os.path.exists(f"{data_dir}/lulesh"):
        print(f"[INFO] Creating directory: {data_dir}/lulesh")
        os.makedirs(f"{data_dir}/lulesh")

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg} -env UCX_RC_VERBS_SEG_SIZE 256000 "\
        "-env UCX_RNDV_THRESHOLD 256000 -env MPICH_ASYNC_PROGRESS 1"
    lulesh_exec = "../apps/lulesh/build/lulesh2.0"
    assert os.path.exists(lulesh_exec), "[ERROR] LULESH executable does not exist"
    for num_proc in num_procs:
        project_name = f"lulesh_{num_proc}"
        out_dir = f"{data_dir}/lulesh/{project_name}"
        lp_file = f"{out_dir}/output/{project_name}.lp"
        if os.path.exists(lp_file):
            print(f"[INFO] LP file {lp_file} already exists. Skipping LP generation for {project_name}")
            continue
        print(f"[INFO] Running LULESH with {num_proc} processes")
        command = f"mpirun {mpirun_args} -np {num_proc} {lulesh_exec} -s {data_size} -i {iterations}"
        lp_gen_command = f'python3 lp_gen.py -c "{command}" -p {project_name} {v_arg} --netparam_file {NET_PARAMS_FILE}'
        print(f"[INFO] Command to execute: {lp_gen_command}")
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to run the command: {lp_gen_command}")
            exit(1)
        
        # Moves the generated trace to the data directory
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{project_name}' to {out_dir}")
            exit(1)

        
def generate_lp_hpcg(data_dir: str, size_param: str = 'test',
                     hostfile: Optional[str] = None,
                     verbose: bool = False) -> None:
    """
    Generates the LP files for HPCG
    """
    if size_param == 'test':
        num_procs = [8, 16]
        data_size = 16
    else:
        num_procs = [8, 32, 64]
        data_size = 48
    
    v_arg = "-v" if verbose else ""
    
    # Checks if the directory <data_dir>/hpcg exists
    if not os.path.exists(f"{data_dir}/hpcg"):
        print(f"[INFO] Creating directory: {data_dir}/hpcg")
        os.makedirs(f"{data_dir}/hpcg")

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg} -env UCX_RC_VERBS_SEG_SIZE 256000 "\
        "-env UCX_RNDV_THRESHOLD 256000 -env MPICH_ASYNC_PROGRESS 1"
    hpcg_exec = "../apps/hpcg/build/bin/xhpcg"

    assert os.path.exists(hpcg_exec), "[ERROR] HPCG executable does not exist"

    for num_proc in num_procs:
        project_name = f"hpcg_{num_proc}"
        out_dir = f"{data_dir}/hpcg/{project_name}"
        lp_file = f"{out_dir}/output/{project_name}.lp"
        if os.path.exists(lp_file):
            print(f"[INFO] LP file {lp_file} already exists. Skipping LP generation for {project_name}")
            continue
        print(f"[INFO] Running HPCG with {num_proc} processes")
        command = f"mpirun {mpirun_args} -np {num_proc} {hpcg_exec} {data_size} {data_size} {data_size}"
        lp_gen_command = f'python3 lp_gen.py -c "{command}" -p {project_name} {v_arg} --netparam_file {NET_PARAMS_FILE}'
        print(f"[INFO] Command to execute: {lp_gen_command}")
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to run the command: {lp_gen_command}")
            exit(1)
        
        # Moves the generated trace to the data directory
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{project_name}' to {out_dir}")
            exit(1)
    
    # Deletes all the output files produced by HPCG
    if os.system("rm -rf HPCG-Benchmark-* hpcg2024*") != 0:
        print("[ERROR] Failed to delete the output files produced by HPCG")
        exit(1)
    

def generate_lp_milc(data_dir: str, size_param: str = 'test',
                     hostfile: Optional[str] = None,
                     verbose: bool = False) -> None:
    """
    Generates the LP files for MILC
    """
    if size_param == 'test':
        num_procs = [8, 16]
    else:
        num_procs = [8, 32, 64]
    
    v_arg = "-v" if verbose else ""

    # Checks if the directory <data_dir>/milc exists
    if not os.path.exists(f"{data_dir}/milc"):
        print(f"[INFO] Creating directory: {data_dir}/milc")
        os.makedirs(f"{data_dir}/milc")

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg} -env UCX_RC_VERBS_SEG_SIZE 256000 "\
        "-env UCX_RNDV_THRESHOLD 256000 -env MPICH_ASYNC_PROGRESS 1"

    milc_exec = "../apps/milc_qcd/ks_imp_dyn/su3_rmd"
    assert os.path.exists(milc_exec), "[ERROR] MILC executable does not exist"

    conf_file = "../validation/milc/milc.in"
    assert os.path.exists(conf_file), "[ERROR] MILC configuration file does not exist"

    for num_proc in num_procs:
        project_name = f"milc_{num_proc}"
        out_dir = f"{data_dir}/milc/{project_name}"

        lp_file = f"{out_dir}/output/{project_name}.lp"
        if os.path.exists(lp_file):
            print(f"[INFO] LP file {lp_file} already exists. Skipping LP generation for {project_name}")
            continue

        print(f"[INFO] Running MILC with {num_proc} processes")
        command = f"mpirun {mpirun_args} -np {num_proc} {milc_exec} {conf_file}"
        lp_gen_command = f'python3 lp_gen.py -c "{command}" -p {project_name} {v_arg} --netparam_file {NET_PARAMS_FILE}'
        print(f"[INFO] Command to execute: {lp_gen_command}")
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to run the command: {lp_gen_command}")
            exit(1)
        
        # Moves the generated trace to the data directory
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{project_name}' to {out_dir}")
            exit(1)


def generate_lp_icon(data_dir: str, size_param: str = 'test',
                     hostfile: Optional[str] = None,
                     verbose: bool = False) -> None:
    """
    Generates the LP files for ICON
    """
    if size_param == 'test':
        num_procs = [8, 16]
    else:
        num_procs = [8, 32, 64]
    
    v_arg = "-v" if verbose else ""

    if not os.path.exists(f"{data_dir}/icon"):
        print(f"[INFO] Creating directory: {data_dir}/icon")
        os.makedirs(f"{data_dir}/icon")

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg} -env UCX_RC_VERBS_SEG_SIZE 256000 "\
        "-env UCX_RNDV_THRESHOLD 256000 -env MPICH_ASYNC_PROGRESS 1 -env INJECTED_LATENCY 0"
    
    run_script = "../validation/icon/run-icon.sh"
    assert os.path.exists(run_script), "[ERROR] ICON run script does not exist"
    for num_proc in num_procs:
        project_name = f"icon_{num_proc}"
        out_dir = f"{data_dir}/icon/{project_name}"
        lp_file = f"{out_dir}/output/{project_name}.lp"
        if os.path.exists(lp_file):
            print(f"[INFO] LP file {lp_file} already exists. Skipping LP generation for {project_name}")
            continue
        print(f"[INFO] Running ICON with {num_proc} processes")
        command = f'{run_script} "mpirun {mpirun_args}"'
        lp_gen_command = f'python3 lp_gen.py -c \'{command}\' --icon --f77 -p {project_name} '\
            f' {v_arg} --netparam_file {NET_PARAMS_FILE}'
        print(f"[INFO] Command to execute: {lp_gen_command}")
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to run the command: {lp_gen_command}")
            exit(1)
        
        # Moves the generated trace to the data directory
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{project_name}' to {out_dir}")
            exit(1)


def generate_metrics(data_dir: str, app: str, L_params: Tuple,
                     verbose: bool = False) -> None:
    """
    Generates the metrics for LULESH
    """
    app_dir = f"{data_dir}/{app}"
    assert os.path.exists(app_dir), f"[ERROR] Directory {app_dir} does not exist"
    L_baseline, _, _ = get_net_params(8, NET_PARAMS_FILE)
    
    L_min, L_max, L_step = L_params
    L_min = L_min * 1000 + L_baseline
    L_max = L_max * 1000 + L_baseline
    L_step *= 1000

    v_arg = "-v" if verbose else ""
    for subdir in os.listdir(app_dir):
        project_dir = f"{app_dir}/{subdir}"
        output_dir = f"{project_dir}/output"
        # Checks if all the result files exist
        all_res_files_exist = all(
            [os.path.exists(f"{output_dir}/{subdir}{res_file_suffix}") for res_file_suffix in RES_FILE_NAMES]
        )

        if all_res_files_exist:
            print(f"[INFO] All result files already exist for {subdir}. Skipping the metrics generation")
            continue

        assert os.path.exists(project_dir), f"[ERROR] Directory {project_dir} does not exist"
        lp_file = f"{output_dir}/{subdir}.lp"
        assert os.path.exists(lp_file), f"[ERROR] LP file {lp_file} does not exist"

        main_script = os.path.join(os.getcwd(), "..", "mpi-dep-graph", "main.py")
        assert os.path.exists(main_script), f"[ERROR] Main script {main_script} does not exist"
        command = f"python3 {main_script} --load-lp-model-path {lp_file} -a sensitivity "\
            f"--l-min {L_min} --l-max {L_max} --step {L_step} --output-dir {output_dir} {v_arg}"
        
        if os.system(command) != 0:
            print(f"[ERROR] Failed to run the command: {command}")
            exit(1)

        runtime_file = f"{output_dir}/{subdir}_runtime.csv"
        assert os.path.exists(runtime_file), f"[ERROR] Runtime file {runtime_file} does not exist"
        lat_buf_baseline = int(get_baseline_runtime(runtime_file))

        lat_tolerance_res_file = open(f"{output_dir}/{subdir}_lat_tolerance.csv", "w")
        lat_tolerance_res_file.write("latency_threshold,latency_tolerance\n")
        lat_tolerance_res_file.flush()

        for lat_buf_thresh in [0.01, 0.02, 0.05]:
            command = f"python3 {main_script} --load-lp-model-path {lp_file} -a buffer "\
                      f"--lat-buf-baseline {lat_buf_baseline} --lat-buf-thresh {lat_buf_thresh} {v_arg}"
            print(f"[INFO] Command to execute: {command}")
            proc = run(command, shell=True, stdout=PIPE, stderr=PIPE)
            if proc.returncode != 0:
                print(f"[ERROR] Command failed: {proc.stderr.decode('utf-8')}")
                exit(1)
            
            # Parses the output
            output = proc.stdout.decode("utf-8")
            if verbose:
                print(output)
            pattern = r"degradation: (\d+\.\d+) ns"
            match = re.search(pattern, output)
            if match is None:
                print(f"[ERROR] Invalid output: {output}")
                exit(1)
            
            latency_tolerance = float(match.group(1))
            lat_tolerance_res_file.write(f"{lat_buf_thresh},{latency_tolerance}\n")
            lat_tolerance_res_file.flush()
            print(f"[INFO] Latency tolerance for {lat_buf_thresh}: {latency_tolerance} ns")
        print(f"[INFO] Latency tolerance results written to {lat_tolerance_res_file.name}")


def collect_metrics(data_dir: str, app: str,
                    num_threads: int, L_params: Tuple,
                    size_param: str = 'test',
                    hostfile: Optional[str] = None,
                    verbose: bool = False) -> None:
    """
    Collects the metrics for the validation experiments
    """
    print(f"[INFO] Using size parameter: {size_param}", flush=True)

    if app == "all":
        apps = ["lulesh", "hpcg", "milc", "icon"]
    else:
        apps = [app]
    
    os.environ["INJECTED_LATENCY"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    for app in apps:
        print(f"[INFO] Generate LP files for {app}")
        if app == "lulesh":
            generate_lp_lulesh(data_dir, size_param, hostfile, verbose)
        elif app == "hpcg":
            generate_lp_hpcg(data_dir, size_param, hostfile, verbose)
        elif app == "milc":
            generate_lp_milc(data_dir, size_param, hostfile, verbose)
        elif app == "icon":
            generate_lp_icon(data_dir, size_param, hostfile, verbose)
        else:
            print(f"[ERROR] Invalid application: {app}")
            exit(1)

        print(f"[INFO] Generating metrics for {app}")

        generate_metrics(data_dir, app, L_params, verbose)



def run_lat_injection_lulesh(data_dir: str, L_params: Tuple,
                             num_trials: int, verbose: bool = False) -> None:
    
    """
    Runs the latency injection experiments for LULESH
    """



def run_lat_injection(data_dir: str, app: str, L_params: Tuple,
                      num_trials: int, verbose: bool = False) -> None:
    """
    Runs the latency injection experiments
    """
    if app == "all":
        apps = ["lulesh", "hpcg", "milc", "icon"]
    else:
        apps = [app]

    for app in apps:
        if app == "lulesh":
            run_lat_injection_lulesh(data_dir, L_params, num_trials, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the validation experiments.")
    parser.add_argument("-d", "--data-dir", type=str, default="validation_data",
                        help="Path to the directory containing the validation data.")
    parser.add_argument("-n", "--num-trials", dest="num_trials", default=10, type=int,
                        help="The number of trials to run for each injected latency")
    parser.add_argument("--app", dest="app", type=str, default="all",
                        help="The application to run the validation experiments on. "
                        "Options are 'all', 'lulesh', 'hpcg', 'milc', 'icon'.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Prints verbose output.")
    parser.add_argument("--size", dest="size_param", type=str, default='test',
                        help="Size of the parameters to run for the applications. Options are 'test' and 'paper' "\
                            "where 'test' runs the applications with a small input size and 'paper' runs the applications "\
                            "with a large input size. Default is 'test'.")
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                        help="The number of OMP threads to use for the applications.")
    parser.add_argument("--restart", dest="restart", action="store_true",
                        help="Restarts the experiments from the beginning by removing the existing data directory.")
    parser.add_argument('--hostfile', dest="hostfile", type=str, default=None,
                        help='Path to the hostfile. If provided, will execute the applications on multiple hosts.')
    
    args = parser.parse_args()

    # Creates the output directory
    if os.path.exists(args.data_dir):
        if args.restart:
            print(f"[INFO] Removing existing directory: {args.data_dir}")
            os.system(f"rm -rf {args.data_dir}")
            print(f"[INFO] Creating directory: {args.data_dir}")
            os.makedirs(args.data_dir)
        else:
            print(f"[WARNING] Directory {args.data_dir} already exists.")
    else:
        print(f"[INFO] Creating directory: {args.data_dir}")
        os.makedirs(args.data_dir)

    # The script consists of three 3 steps:
    # 1. Profiles the network to collect data on G and o
    # 2. Traces the applications to collect data about
    #   the application's latency sensitivity and tolerance
    # 3. Runs the validation experiments and collects data
    #   on the application's performance under different injected latency.
    size_param = args.size_param
    if size_param not in ['test', 'paper']:
        print(f"[ERROR] Invalid size parameter: {args.size_param}")
        exit(1)
    
    if args.size_param == 'test':
        L_params = (1, 100, 10)
    else:
        L_params = (1, 100, 1)
    
    
    # Step 1: Collect network parameters
    if not os.path.exists(NET_PARAMS_FILE):
        collect_net_params(file_path=NET_PARAMS_FILE)
    else:
        print(f"[INFO] Using existing network parameters file: {NET_PARAMS_FILE}")
    
    # Step 2: Trace the applications and collect metrics
    collect_metrics(args.data_dir, app=args.app, num_threads=args.num_threads,
                    L_params=L_params,
                    size_param=size_param, hostfile=args.hostfile,
                    verbose=args.verbose)

    # Step 3: Run the latency injection experiments
    run_lat_injection(args.data_dir, app=args.app,
                      L_params=L_params,
                      num_trials=args.num_trials, verbose=args.verbose)
