import argparse
import os
import sys
from typing import List, Optional
from subprocess import run, PIPE, TimeoutExpired
from utils import *


def collect_traces(trace_dir: str, project: str, command: str, icon: bool,
                   liballprof: str, timeout: int, verbose: bool = False) -> str:
    """
    Collects MPI traces using the given command and saves them to the
    trace directory.
    @param trace_dir: Path to the directory where the traces will be saved.
    @param project: Name of the project.
    @param command: The command or script to run in order to execute the
    application and generate the MPI traces.
    @param icon: Flag to indicate that the traces are from ICON.
    @param liballprof: Path to the liballprof shared library.
    @param timeout: Timeout in seconds for running the command.
    @param verbose: Flag to enable verbose mode.
    @return The actual path to the trace directory.
    """

    if verbose:
        print(f"[INFO] Collecting MPI traces for project {project}...", flush=True)
        print(f"[INFO] Command to run: {command}")
    
    # Splits the command into tokens
    tmp_tokens = command.split()
    tokens = [tmp_tokens[0]]
    curr_token = ""
    for token in tmp_tokens[1:]:
        if token.startswith("\"") and token.endswith("\""):
            tokens.append(token[1:-1])
        if token.startswith("\""):
            curr_token = token[1:]
        elif token.endswith("\""):
            curr_token += f" {token[:-1]}"
            tokens.append(curr_token)
            curr_token = ""
        elif curr_token:
            curr_token += f" {token}"
        else:
            tokens.append(token)
    
    print(f"[INFO] Tokens: {tokens}", flush=True)
    # Sets the trace directory
    os.environ["HTOR_PMPI_FILE_PREFIX"] = f"{trace_dir}/pmpi-trace-rank-"
    # Sets the liballprof shared library
    os.environ["LD_PRELOAD"] = liballprof
    if icon:
        os.environ["LD_PRELOAD"] = os.environ["LIBALLPROF2_F77"]

    try:
        proc = run(tokens, stdout=PIPE, stderr=PIPE, timeout=timeout)
        rc = proc.returncode
        if verbose:
            print("[INFO] Command stdout:")
            print(proc.stdout.decode("utf-8"))
            print("[INFO] Command stderr:")
            print(proc.stderr.decode("utf-8"))
        
        if rc == 0 and verbose:
            print("[INFO] Command execution: SUCCESS", flush=True)
        
        if proc.returncode != 0:
            print(f"[ERROR] Command failed {rc}: {proc.stderr.decode('utf-8')}", flush=True)
            print(f"[ERROR] stdout: {proc.stdout.decode('utf-8')}", flush=True)
            exit(1)

    except TimeoutExpired:
        print(f"[ERROR] Command timed out after {timeout} seconds.", flush=True)
        exit(1)

    
    # Checks if the trace directory is empty
    if not os.listdir(trace_dir):
        print("[ERROR] No trace files were generated. Exiting...", flush=True)
        exit(1)
    

    if icon:
        # Cleans up the ICON traces
        if verbose:
            print("[INFO] Cleaning up ICON traces...", flush=True)
        # Executes the icon_cleanup.py script
        cleanup_script = os.path.join(os.path.dirname(__file__), "icon_cleanup.py")
        assert os.path.exists(cleanup_script), f"[ERROR] icon_cleanup.py does not exist: {cleanup_script}"
        icon_clean_dir = trace_dir + "_cleaned"
        cleanup_command = f"python3 {cleanup_script} -i {trace_dir} -o {icon_clean_dir}"
        if verbose:
            print(f"[INFO] Cleanup command to run: {cleanup_command}")

        try:
            proc = run(cleanup_command.split(), stdout=PIPE, stderr=PIPE, timeout=timeout)
            rc = proc.returncode
            if verbose:
                print("[INFO] Cleanup stdout:")
                print(proc.stdout.decode("utf-8"))
            
            if rc == 0 and verbose:
                print("[INFO] Cleanup execution: SUCCESS", flush=True)
            
            if proc.returncode != 0:
                print(f"[ERROR] Cleanup failed {rc}: {proc.stderr.decode('utf-8')}", flush=True)
                exit(1)

        except TimeoutExpired:
            print(f"[ERROR] Cleanup timed out after {timeout} seconds.", flush=True)
            exit(1)

        return icon_clean_dir

    return trace_dir
        


def convert_trace_to_goal(project: str, trace_dir: str, project_dir: str,
                           verbose: bool = False) -> str:
    """
    Converts the MPI traces to a goal file.
    @param project: Name of the project.
    @param trace_dir: Path to the directory where the traces are saved.
    @param project_dir: Path to the project directory.
    @param verbose: Flag to enable verbose mode.
    @return The path to the goal file
    """
    if verbose:
        print(f"[INFO] Converting MPI traces to a graph for project {project}...", flush=True)
    
    # Calls the script goal_gen.sh
    goal_gen_script = os.path.join(os.path.dirname(__file__), "goal_gen.sh")
    assert os.path.exists(goal_gen_script), f"[ERROR] goal_gen.sh does not exist: {goal_gen_script}"
    goal_gen_command = f"bash {goal_gen_script} {trace_dir} {project} {proj_dir}"
    if verbose:
        print(f"[INFO] goal_gen.sh command to run: {goal_gen_command}")
    try:
        proc = run(goal_gen_command.split(), stdout=PIPE, stderr=PIPE)
        rc = proc.returncode
        if verbose:
            print("[INFO] goal_gen.sh stdout:")
            print(proc.stdout.decode("utf-8"))
        
        if rc == 0 and verbose:
            print("[INFO] goal_gen.sh execution: SUCCESS", flush=True)
        
        if proc.returncode != 0:
            print(f"[ERROR] goal_gen.sh failed {rc}: {proc.stderr.decode('utf-8')}", flush=True)
            exit(1)
    except Exception as e:
        print(f"[ERROR] goal_gen.sh failed: {e}", flush=True)
        exit(1)


    goal_file = os.path.join(project_dir, f"{project}.goal")
    if not os.path.exists(goal_file):
        print(f"[ERROR] Failed to generate the goal file: {goal_file}", flush=True)
        exit(1)
    
    comm_dep_file = os.path.join(project_dir, f"{project}.comm-dep")
    if not os.path.exists(comm_dep_file):
        print(f"[ERROR] Failed to generate the comm-dep file: {comm_dep_file}", flush=True)
        exit(1)
    
    if verbose:
        print(f"[INFO] Generated goal file: {goal_file}")
        print(f"[INFO] Generated comm_dep file: {comm_dep_file}", flush=True)
    
    return goal_file



def convert_goal_to_lp(project: str, proj_dir: str, out_dir: str,
                       o_val: int, G_val: float, S_val: Optional[int],
                       topology: str, save_graph: bool,
                       verbose: bool) -> None:
    """
    Converts the goal file to an LP model.
    """
    # Uses the python script main.py in ../mpi-dep-graph to 
    # convert the goal file to an LP model
    main_script = os.path.join(os.getcwd(), "..", "mpi-dep-graph", "main.py")
    assert os.path.exists(main_script), f"[ERROR] main.py does not exist: {main_script}"
    
    goal_file = os.path.join(proj_dir, f"{project}.goal")
    assert os.path.exists(goal_file), f"[ERROR] Goal file does not exist: {goal_file}"
    comm_dep_file = os.path.join(proj_dir, f"{project}.comm-dep")
    assert os.path.exists(comm_dep_file), f"[ERROR] Comm-dep file does not exist: {comm_dep_file}"

    S_arg = f"-S {S_val}" if S_val else ""
    v_arg = "-v" if verbose else ""
    lp_file = f"{out_dir}/{project}.lp"
    
    if save_graph:
        graph_path = f"{out_dir}/{project}.pkl"
        save_graph_arg = f"--export-graph-path {graph_path}"
    else:
        save_graph_arg = ""


    lp_gen_command = f"python3 {main_script} -g {goal_file} -c {comm_dep_file} -o {o_val} -G {G_val} {S_arg} --topology {topology} {v_arg} --export-lp-model-path {lp_file} {save_graph_arg}"
    if verbose:
        print("[INFO] Converting goal file to LP model...")
        print(f"[INFO] Command to run: {lp_gen_command}")
    try:
        proc = run(lp_gen_command.split(), stdout=sys.stdout, stderr=PIPE)
        rc = proc.returncode
        if rc == 0 and verbose:
            print("[INFO] main.py execution: SUCCESS", flush=True)
        
        if proc.returncode != 0:
            print(f"[ERROR] main.py failed {rc}: {proc.stderr.decode('utf-8')}", flush=True)
            exit(1)
    
    except Exception as e:
        print(f"[ERROR] main.py failed: {e}", flush=True)
        exit(1)


if __name__ == "__main__":
    # Parses arguments
    parser = argparse.ArgumentParser(description="Generates LP model directly from MPI traces")
    parser.add_argument("-t", "--trace_dir", type=str, required=False,
                        help="Path to the MPI trace file directory. If not provided, the default will be "
                        "a directory named 'traces' inside the project directory.")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Path to the output directory. If not provided, the default will be "
                        "a directory named 'output' inside the project directory.")
    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Name of the project. The script will create a directory with the project name.")
    parser.add_argument("--icon", action="store_true",
                        help="Flag to indicate that the traces are from ICON.")
    parser.add_argument("-c", "--command", type=str, required=True,
                        help="The command or script to run in order to execute the application and generate the MPI traces.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds for running the command.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Flag to enable verbose mode.")
    parser.add_argument("--f77", action="store_true",
                        help="Flag to indicate that the traces are from a FORTRAN 77 application.")
    parser.add_argument("--liballprof-dir", type=str, required=False,
                        help="Path to the liballprof directory. If not provided, "
                        "the script will set it to ../liballprof/.libs/")
    parser.add_argument("-o", dest="o", type=int, default=5000,
                        help="Specify the o in the LogGPS model. [DEFAULT: 5000]")
    parser.add_argument("-G", dest="G", type=float, default=0.018,
                        help="Specify the G in the LogGPS model. [DEFAULT: 0.018]")
    parser.add_argument("-S", dest="S", default=None, type=int,
                        help="Specify the S in the LogGPS model. [DEFAULT: None]")
    parser.add_argument("--netparam_file", dest="netparam_file", type=str, default=None,
                        help="Path to the network parameters file. If provided, "
                        "will ignore the o, G arguments and use the values from "
                        "the file. [DEFAULT: None]")
    parser.add_argument("--topology", dest="topology", choices=["default", "fat_tree", "dragonfly"],
                        required=False, default="default",
                        help="If given, will use the specified network topology to be used in the model.")
    parser.add_argument("-s", "--skip-tracing", action="store_true",
                        help="Flag to skip the tracing and only convert the goal file to LP model.")
    parser.add_argument("--only-tracing", action="store_true",
                        help="Flag to only run the tracing and skip the goal and LP generation.")
    parser.add_argument("--save-graph", action="store_true",
                        help="Flag to save the intermediate MPI exeuction graph that "
                        "is used to generate the LP in the output directory.")
        
    args = parser.parse_args()

    assert not (args.skip_tracing and args.only_tracing), "[ERROR] Cannot use both --skip-tracing and --only-tracing flags together."

    verbose = args.verbose
    # Checks if the project directory exists
    proj_dir = os.path.join(os.getcwd(), "..", args.project)
    if not os.path.exists(proj_dir):
        # Creates the project directory
        os.makedirs(proj_dir)
        if verbose:
            print(f"[INFO] Created project directory: {proj_dir}")
        
    # Generate the MPI traces
    trace_dir = args.trace_dir if args.trace_dir else os.path.join(proj_dir, "traces")
    # Checks if the trace directory exists
    if not os.path.exists(trace_dir):
        # Creates the trace directory
        os.makedirs(trace_dir)
        if verbose:
            print(f"[INFO] Created trace directory: {trace_dir}")
    if verbose:
        print(f"[INFO] Trace directory: {trace_dir}")

    
    # Checks if the output directory exists
    out_dir = args.output_dir if args.output_dir else os.path.join(proj_dir, "output")
    if not os.path.exists(out_dir):
        # Creates the output directory
        os.makedirs(out_dir)
        if verbose:
            print(f"[INFO] Created output directory: {out_dir}")
    if verbose:
        print(f"[INFO] Output directory: {out_dir}")


    if args.liballprof_dir:
        liballprof = args.liballprof_dir
    else:
        liballprof = os.path.join(os.getcwd(), "..", "liballprof", ".libs")
    
    liballprof = os.path.join(liballprof, "liballprof.so" if not args.f77 else "liballprof_f77.so")
    assert os.path.exists(liballprof), f"[ERROR] liballprof.so does not exist: {liballprof}"
    if verbose:
        print(f"[INFO] liballprof.so: {liballprof}")


    if not args.skip_tracing:
        trace_dir = collect_traces(trace_dir, args.project, args.command, args.icon,
                                liballprof, args.timeout, args.verbose)
    else:
        if args.icon:
            trace_dir = trace_dir + "_cleaned"
        # Double checks the trace directory to make sure it is not empty
        if not os.listdir(trace_dir):
            print("[ERROR] No trace files were generated. Exiting...", flush=True)
            exit(1)
        print(f"[INFO] Skipping tracing. Using existing traces in {trace_dir}")

    if verbose:
        print(f"[INFO] Traces saved to {trace_dir}", flush=True)

    if not args.only_tracing:    
        # Converts the MPI traces to goal
        goal_file = convert_trace_to_goal(args.project, trace_dir, proj_dir, args.verbose)

        if args.netparam_file is not None:
            assert os.path.exists(args.netparam_file), \
                f"[ERROR] Netparam file does not exist: {args.netparam_file}"
            # Calculates the average message size from the goal file
            avg_message_size = get_avg_message_size(goal_file)
            if verbose:
                print(f"[INFO] Average message size: {avg_message_size} bytes")
            # Reads the network parameters from the netparam file
            _, o, G = get_net_params(avg_message_size, args.netparam_file)
        else:
            o = args.o
            G = args.G

        # Converts the goal file to LP model
        convert_goal_to_lp(args.project, proj_dir, out_dir,
                        o, G, args.S, args.topology, args.save_graph,
                        args.verbose)
    exit(0)
