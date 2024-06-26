import os
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

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg}"
    lulesh_exec = "../apps/lulesh/build/lulesh2.0"
    assert os.path.exists(lulesh_exec), "[ERROR] LULESH executable does not exist"
    for num_proc in num_procs:
        project_name = f"lulesh_{num_proc}"
        out_dir = f"{data_dir}/{project_name}"
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
    pass

        

def generate_lp_milc(data_dir: str, size_param: str = 'test',
                     hostfile: Optional[str] = None,
                     verbose: bool = False) -> None:
    """
    Generates the LP files for MILC
    """
    pass


def generate_lp_icon(data_dir: str, size_param: str = 'test',
                     hostfile: Optional[str] = None,
                     verbose: bool = False) -> None:
    """
    Generates the LP files for ICON
    """
    pass


def collect_metrics(data_dir: str, app: str, size_param: str = 'test',
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

    for app in apps:
        print(f"[INFO] Generate LP files for {app}")
        if app == "lulesh":
            generate_lp_lulesh(data_dir, size_param, hostfile, verbose)
        elif app == "hpcg":
            generate_lp_hpcg(data_dir, size_param, hostfile, verbose)
        elif app == "milc":
            pass
        elif app == "icon":
            pass
        else:
            print(f"[ERROR] Invalid application: {app}")
            exit(1)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the validation experiments.")
    parser.add_argument("-d", "--data-dir", type=str, default="validation_data",
                        help="Path to the directory containing the validation data.")
    parser.add_argument("--l-min", dest="L_min", default=1, type=int,
                        help="The minimum injected latency in us")
    parser.add_argument("--l-max", dest="L_max", default=100, type=int,
                        help="The maximum injected latency in us")
    parser.add_argument("-s", "--step", dest="step", default=1, type=int,
                        help="The step size for the injected latency in us")
    parser.add_argument("-n", "--num-trials", dest="num_trials", default=10, type=int,
                        help="The number of trials to run for each injected latency")
    parser.add_argument("--app", type=str, default="all",
                        help="The application to run the validation experiments on. "
                        "Options are 'all', 'lulesh', 'hpcg', 'milc', 'icon'.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Prints verbose output.")
    parser.add_argument("-s", "--size", type=str, default='test',
                        help="Size of the parameters to run for the applications. Options are 'test' and 'paper' "\
                            "where 'test' runs the applications with a small input size and 'paper' runs the applications "\
                            "with a large input size. Default is 'test'.")
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Path to the hostfile. If provided, will execute the applications on multiple hosts.')
    
    args = parser.parse_args()

    # Creates the output directory
    if os.path.exists(args.data_dir):
        print(f"[INFO] Removing existing directory: {args.data_dir}")
        os.system(f"rm -rf {args.data_dir}")
    print(f"[INFO] Creating directory: {args.data_dir}")
    os.makedirs(args.data_dir)

    # The script consists of three 3 steps:
    # 1. Profiles the network to collect data on G and o
    # 2. Traces the applications to collect data about
    #   the application's latency sensitivity and tolerance
    # 3. Runs the validation experiments and collects data
    #   on the application's performance under different injected latency.

    # Step 1: Collect network parameters
    if not os.path.exists(NET_PARAMS_FILE):
        collect_net_params(file_path=NET_PARAMS_FILE)
    else:
        print(f"[INFO] Using existing network parameters file: {NET_PARAMS_FILE}")
    
    # Step 2: Trace the applications and collect metrics
    collect_metrics(args.data_dir, app=args.app,
                    size_param=args.size, hostfile=args.hostfile,
                    verbose=args.verbose)


