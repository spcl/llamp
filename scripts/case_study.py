import os
import re
import argparse
from typing import Optional, List, Dict
from utils import *


"""
Script to run the case study experiments.
"""

NET_PARAMS_FILE = "netparams.csv"
RES_FILE_NAMES = ["_lat_ratio.csv", "_lat_sensitivity.csv", "_runtime.csv", "_lat_tolerance.csv"]


def trace_icon(data_dir: str, size_param: str = 'test',
               hostfile: Optional[str] = None,
               num_threads: int = 1,
               verbose: bool = False) -> None:
    """
    Generates the LP files for ICON
    """
    if size_param == 'test':
        num_procs = [8, 16]
        sim_time = "2000-01-01T00:30:00Z"
    else:
        num_procs = [32, 64, 256]
        sim_time = "2000-01-02T00:00:00Z"
    
    v_arg = "-v" if verbose else ""

    hostfile_arg = f"-f {hostfile}" if hostfile is not None else ""
    mpirun_args = f"--envall {hostfile_arg} -env UCX_RC_VERBS_SEG_SIZE 256000 "\
        "-env UCX_RNDV_THRESHOLD 256000 -env MPICH_ASYNC_PROGRESS 1 -env INJECTED_LATENCY 0 "\
        f"-env OMP_NUM_THREADS {num_threads}"
    
    run_script = "../validation/icon/run-icon.sh"
    assert os.path.exists(run_script), "[ERROR] ICON run script does not exist"
    for num_proc in num_procs:
        project_name = f"icon_{num_proc}"
        out_dir = f"{data_dir}/{project_name}"
        trace_dir = f"{out_dir}/traces_cleaned"
        # If we already have <num_proc> trace files inside the trace directory,
        # skip this step
        if os.path.exists(out_dir) and len(os.listdir(trace_dir)) == num_proc:
            print(f"[INFO] Skipping tracing ICON {num_proc} as the traces already exist")
            continue
    
        print(f"[INFO] Running ICON with {num_proc} processes")
        command = f'{run_script} "mpirun {mpirun_args} -np {num_proc}" "{sim_time}"'
        lp_gen_command = f'python3 lp_gen.py -c \'{command}\' --icon --f77 -p {project_name} '\
            f' {v_arg} --netparam_file {NET_PARAMS_FILE} --only-tracing'
        print(f"[INFO] Command to execute: {lp_gen_command}", flush=True)
        if os.system(lp_gen_command) != 0:
            print(f"[ERROR] Failed to run the command: {lp_gen_command}")
            exit(1)
        
        # Moves the generated trace to the data directory
        if os.system(f"mv ../{project_name} {out_dir}") != 0:
            print(f"[ERROR] Failed to move '../{project_name}' to {out_dir}")
            exit(1)


def allerduce_generate_metrics(data_dir: str, size_param: str = "test",
                               verbose: bool = True) -> None:
    """
    Generates the metrics for ICON using two different algorithms for allreduce.
    """
    if size_param == 'test':
        num_procs = [8, 16]
    else:
        num_procs = [32, 64, 256]

    # Checks if the directory 'allreduce' exists in the data directory
    allreduce_dir = f"{data_dir}/allreduce"
    if not os.path.exists(allreduce_dir):
        print(f"[INFO] Creating directory: {allreduce_dir}")
        os.makedirs(allreduce_dir)
    
    # Checks if the 'recdoub' and 'ring' directories exist in the 'allreduce' directory
    recdoub_dir = f"{allreduce_dir}/recdoub"
    ring_dir = f"{allreduce_dir}/ring"
    if not os.path.exists(recdoub_dir):
        print(f"[INFO] Creating directory: {recdoub_dir}")
        os.makedirs(recdoub_dir)
    if not os.path.exists(ring_dir):
        print(f"[INFO] Creating directory: {ring_dir}")
        os.makedirs(ring_dir)

    v_arg = "-v" if verbose else ""
    def generate_metrics(allreduce_alg: str) -> None:
        """
        A helper function
        """
        for i, num_proc in enumerate(num_procs):
            if i == 0:
                L_params = (0, 200, 10) if size_param == 'test' else (0, 200, 1)
            else:
                L_params = (0, 100, 10) if size_param == 'test' else (0, 100, 1)
            
            project_name = f"icon_{num_proc}"
            trace_dir = f"{data_dir}/{project_name}/traces_cleaned"
            assert os.path.exists(trace_dir), f"[ERROR] Trace directory does not exist: {trace_dir}"
            out_dir = f"{data_dir}/allreduce/{allreduce_alg}/{project_name}"
            if not os.path.exists(out_dir):
                print(f"[INFO] Creating directory: {out_dir}")
                os.makedirs(out_dir)
            
            # Checks if the goal file exists
            goal_file = f"{out_dir}/{project_name}.goal"
            comm_dep_file = f"{out_dir}/{project_name}.comm-dep"
            if not os.path.exists(goal_file) or not os.path.exists(comm_dep_file):
                goal_gen_script = f"bash goal_gen.sh {trace_dir} {project_name} {out_dir}"
                if verbose:
                    print(f"[INFO] Command to execute: {goal_gen_script}")
                if os.system(goal_gen_script) != 0:
                    print(f"[ERROR] Failed to generate goal file for ICON {num_proc}")
                    exit(1)

                assert os.path.exists(f"{out_dir}/{project_name}.bin") and \
                    os.path.exists(f"{out_dir}/{project_name}.comm-dep"), \
                    f"[ERROR] Failed to generate goal file for ICON {num_proc}"
            else:
                print(f"[INFO] Skipping generating goal file for ICON {num_proc}", flush=True)
            

            all_res_files_exist = \
                all([os.path.exists(f"{out_dir}/{project_name}{res_file}") for res_file in RES_FILE_NAMES])

            if not all_res_files_exist:
                lp_file = f"{out_dir}/{project_name}.lp"
                # Checks if the lp file exists
                if os.path.exists(lp_file):
                    lp_arg = f"--load-lp-model-path {lp_file}"
                else:
                    lp_arg = f"--export-lp-model-path {lp_file}"

                avg_message_size = get_avg_message_size(goal_file)
                L_baseline, o, G = get_net_params(avg_message_size, NET_PARAMS_FILE)
                main_script = f"../mpi-dep-graph/main.py"

                L_min, L_max, L_step = L_params
                L_min = L_min * 1000 + L_baseline
                L_max = L_max * 1000 + L_baseline
                L_step *= 1000
                
                graph_arg = ""
                if allreduce_alg == "recdoub" and i == len(num_procs) - 1:
                    graph_arg = f"--export-graph-path {out_dir}/{project_name}.pkl"

                assert os.path.exists(main_script), "[ERROR] Main script does not exist"
                command = f"python3 {main_script} -g {goal_file} -c {comm_dep_file} -o {o} -G {G} -S 256000 "\
                    f"{v_arg} {lp_arg} --output-dir {out_dir} -a sensitivity "\
                    f"--l-min {L_min} --l-max {L_max} --step {L_step} {graph_arg}"
                
                if verbose:
                    print(f"[INFO] Command to execute: {command}", flush=True)
                if os.system(command) != 0:
                    print(f"[ERROR] Failed to generate metrics for ICON {num_proc}")
                    exit(1)

                assert all([os.path.exists(f"{out_dir}/{project_name}{res_file}") for res_file in RES_FILE_NAMES[:-1]]), \
                    f"[ERROR] Failed to generate metrics for ICON {num_proc}"
                

                runtime_file = f"{out_dir}/{project_name}_runtime.csv"
                assert os.path.exists(runtime_file), f"[ERROR] Runtime file does not exist: {runtime_file}"
                lat_buf_baseline = int(get_baseline_runtime(runtime_file))

                lat_tolerance_path = f"{out_dir}/{project_name}_lat_tolerance.csv"
                lat_tolerance_file = open(lat_tolerance_path, "w")
                lat_tolerance_file.write("latency_threshold,latency_tolerance\n")
                lat_tolerance_file.flush()


                for lat_buf_thresh in [0.01, 0.02, 0.05]:
                    command = f"python3 {main_script} --load-lp-model-path {lp_file} -a buffer {v_arg} "\
                        f"--lat-buf-thresh {lat_buf_thresh} --lat-buf-baseline {lat_buf_baseline}"
                    if verbose:
                        print(f"[INFO] Command to execute: {command}", flush=True)
                    
                    proc = run(command, shell=True, stdout=PIPE, stderr=PIPE)
                    if proc.returncode != 0:
                        print(f"[ERROR] Failed to generate metrics for ICON {num_proc}")
                        print(proc.stderr.decode("utf-8"))
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
                    lat_tolerance_file.write(f"{lat_buf_thresh},{latency_tolerance}\n")
                    lat_tolerance_file.flush()
                    print(f"[INFO] Latency tolerance for {lat_buf_thresh}: {latency_tolerance} ns")
                print(f"[INFO] Latency tolerance results written to {lat_tolerance_path}")
            else:
                print(f"[INFO] Skipping generating metrics for ICON {num_proc}", flush=True)


    # Generates the metrics for ICON using Recursive Doubling
    # Keeps track of the current working directory
    cwd = os.getcwd()
    choose_schedgen_allreduce_algorithm("../Schedgen", allreduce_alg="recdoub")
    os.chdir(cwd)
    
    generate_metrics("recdoub")

    # Generates the metrics for ICON using Ring
    # Keeps track of the current working directory
    cwd = os.getcwd()
    choose_schedgen_allreduce_algorithm("../Schedgen", allreduce_alg="ring")
    os.chdir(cwd)

    generate_metrics("ring")



def topology_generate_metrics(data_dir: str, size_param: str = "test",
                              verbose: bool = True) -> None:
    """
    Generates the metrics for ICON using different topologies, namely
    Fat Tree and Dragonfly.
    """
    num_procs = 16 if size_param == "test" else 256

    v_arg = "-v" if verbose else ""
    topology_dir = f"{data_dir}/topology"
    if not os.path.exists(topology_dir):
        print(f"[INFO] Creating directory: {topology_dir}")
        os.makedirs(topology_dir)
    
    name = f"icon_{num_procs}"
    graph_file = f"{data_dir}/allreduce/recdoub/{name}/{name}.pkl"
    assert os.path.exists(graph_file), f"[ERROR] Graph file does not exist: {graph_file}"


    topologies = ["fat_tree", "dragonfly"]
    for topology in topologies:
        out_dir = f"{topology_dir}/{topology}"
        if not os.path.exists(out_dir):
            print(f"[INFO] Creating directory: {out_dir}")
            os.makedirs(out_dir)
        
        if verbose:
            print(f"[INFO] Generating metrics for ICON using {topology} topology")
        
    
        # Checks if all result files exist
        all_res_files_exist = \
            all([os.path.exists(f"{out_dir}/{name}{res_file}") for res_file in RES_FILE_NAMES])
        
        if all_res_files_exist:
            print(f"[INFO] Skipping generating metrics for ICON using {topology} topology", flush=True)
            continue

        # Checks if the lp file exists
        lp_file = f"{out_dir}/{name}.lp"
        if os.path.exists(lp_file):
            lp_arg = f"--load-lp-model-path {lp_file}"
        else:
            lp_arg = f"--export-lp-model-path {lp_file}"
        
        L_min = 274
        L_max = 424
        L_step = 1

        main_script = "../mpi-dep-graph/main.py"
        assert os.path.exists(main_script), "[ERROR] Main script does not exist"
        command = f"python3 {main_script} {lp_arg} --load-graph-path {graph_file} "\
            f"-a sensitivity --output-dir {out_dir} --topology {topology} "\
            f"--l-min {L_min} --l-max {L_max} --step {L_step} {v_arg}"

        if verbose:
            print(f"[INFO] Command to execute: {command}", flush=True)
        
        if os.system(command) != 0:
            print(f"[ERROR] Failed to generate metrics for ICON using {topology} topology")
            exit(1)
        
        assert all([os.path.exists(f"{out_dir}/{name}{res_file}") for res_file in RES_FILE_NAMES[:-1]]), \
            f"[ERROR] Failed to generate metrics for ICON using {topology} topology"
        
        runtime_file = f"{out_dir}/{name}_runtime.csv"
        assert os.path.exists(runtime_file), f"[ERROR] Runtime file does not exist: {runtime_file}"
        lat_buf_baseline = int(get_baseline_runtime(runtime_file))

        lat_tolerance_path = f"{out_dir}/{name}_lat_tolerance.csv"
        lat_tolerance_file = open(lat_tolerance_path, "w")
        lat_tolerance_file.write("latency_threshold,latency_tolerance\n")
        lat_tolerance_file.flush()

        for lat_buf_thresh in [0.01]:
            command = f"python3 {main_script} --load-lp-model-path {lp_file} -a buffer {v_arg} "\
                f"--lat-buf-thresh {lat_buf_thresh} --lat-buf-baseline {lat_buf_baseline}"
            if verbose:
                print(f"[INFO] Command to execute: {command}", flush=True)
            
            proc = run(command, shell=True, stdout=PIPE, stderr=PIPE)
            if proc.returncode != 0:
                print(f"[ERROR] Failed to generate metrics for ICON using {topology} topology")
                print(proc.stderr.decode("utf-8"))
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
            lat_tolerance_file.write(f"{lat_buf_thresh},{latency_tolerance}\n")
            lat_tolerance_file.flush()
            print(f"[INFO] Latency tolerance for {lat_buf_thresh}: {latency_tolerance} ns")

        print(f"[INFO] Latency tolerance results written to {lat_tolerance_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the case study experiments.")
    parser.add_argument("-d", "--data-dir", type=str, default="case_study_data",
                        help="Path to the directory containing the case study data.")
    parser.add_argument("--size", type=str, default='test', dest="size_param",
                        help="Size of the parameters to run for the applications. Options are 'test' and 'paper' "\
                            "where 'test' runs the applications with a small input size and 'paper' runs the applications "\
                            "with a large input size. Default is 'test'.")
    parser.add_argument("--hostfile", type=str, default=None,
                        help="Path to the hostfile for the experiments. "\
                            "If provided, will execute the applications on multiple hosts.")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of OMP threads to use for ICON.")
    parser.add_argument("--restart", action="store_true",
                        help="Restarts the experiment from the beginning by removing the existing data directory.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Prints verbose output.")
    args = parser.parse_args()

    # Creates the output directory
    if os.path.exists(args.data_dir):
        if args.restart:
            print(f"[INFO] Removing existing directory: {args.data_dir}")
            os.system(f"rm -rf {args.data_dir}")
            print(f"[INFO] Creating directory: {args.data_dir}")
            os.makedirs(args.data_dir)
        else:
            print(f"[INFO] Using existing directory: {args.data_dir}")
    else:
        print(f"[INFO] Creating directory: {args.data_dir}")
        os.makedirs(args.data_dir)

    if not os.path.exists(NET_PARAMS_FILE):
        collect_net_params(NET_PARAMS_FILE)
    else:
        print(f"[INFO] Using existing netparams file: {NET_PARAMS_FILE}")


    size_param = args.size_param
    assert size_param in ['test', 'paper'], "[ERROR] Invalid size parameter. Options are 'test' and 'paper'."

    # The script consists of three steps:
    # 1. Collects the traces for ICON under three different node configurations
    trace_icon(args.data_dir, size_param=size_param,
               num_threads=args.num_threads,
               hostfile=args.hostfile,
               verbose=args.verbose)

    # 2. Produces metrics for the collected traces using two different
    #    algorithms for allreduce, namely Ring and Recursive Doubling.
    allerduce_generate_metrics(args.data_dir, size_param=size_param,
                               verbose=args.verbose)

    # 3. Analyzes the impact of topology on the performance of ICON.
    topology_generate_metrics(args.data_dir, size_param=size_param,
                              verbose=args.verbose)
