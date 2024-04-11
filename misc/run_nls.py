import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Tuple
from subprocess import run, PIPE
from pathlib import Path
from tqdm import tqdm

MIN_LATENCY = 200
MAX_LATENCY = 100000
STEP = 1000
RESULTS_PATTERN = r"Host\s(\d+):\s(\d+)"

def get_result_from_output(out_str: str, verbose: bool) -> int:
    """
    Parses the output string and retrieves runtime of the MPI application
    by comparing the runtime of individual ranks. The final result of
    one trial will be determined by the rank with the longest runtime.
    """
    matches = re.findall(RESULTS_PATTERN, out_str)
    assert len(matches) > 0
    # FIXME Could use higher oder functions such as reduce
    res = 0
    slowest_rank = None
    for rank, runtime_str in matches:
        runtime = int(runtime_str)
        if runtime > res:
            slowest_rank = rank
            res = runtime
    if verbose:
        print(f"[RUN-NLS][INFO] Slowest rank: {slowest_rank}, time: {res}")
    return res


def run_cmd(cmd: Union[List[str], str], verbose: bool,
            return_output: bool = False) -> Union[None, str]:
    """
    A helper function that runs the given command in a subprocess.
    If `return_output` is set to True, will return the stdout of the
    subprocess as output.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"[RUN-NLS] Running command: {' '.join(cmd)}")
    process = run(cmd, stdout=PIPE, stderr=PIPE)

    if verbose:
        print("[RUN-NLS][INFO] Process stdout:")
        print(process.stdout.decode())
    
    if process.returncode != 0:
        # If an error occurred while the script was running
        print("[RUN-NLS][ERROR] Process stderr:")
        print(process.stderr.decode())
        exit(1)

    if return_output:
        return process.stdout.decode()

def run_nls_exp(bin_file: str, latencies: range, verbose: bool) -> List:
    """
    Given the binary goal file, runs the experiment with increasing
    network latency (parameter L in the LogGP model), and returns
    the predicted runtime of each trial as a list.
    """
    res = []
    print("[RUN-NLS] Evaluating application runtime given network latencies "
          f"in the range ({MIN_LATENCY}, {MAX_LATENCY}, {STEP})")
    
    lgs_cmd = f"./LogGOPSim/LogGOPSim -f {bin_file} -S 4096 -G 1 -o 3000"

    for L in latencies:
        # Specifies the L parameter for each run
        cmd = f"{lgs_cmd} -L {L}"
        print(f"[RUN-NLS] Evaluating with L {L}")
        out = run_cmd(cmd, verbose, True)
        runtime = get_result_from_output(out, verbose)
        res.append(runtime)
    print("[RUN-NLS] Application evaluation: SUCCESS")
    return res


def save_res(res: List[Tuple[int, int]], output_file: str, verbose: bool) \
    -> None:
    """
    Saves the experiment result to the given output file as a csv.
    """
    print(f"[RUN-NLS] Saving results to {output_file}")
    df = pd.DataFrame(res, columns=["L", "runtime"])
    df.to_csv(output_file, index=False)
    print(f"[RUN-NLS] Save results to {output_file}: SUCCESS")


"""
Performs the experiment to measure the network latency sensitivity
of a given MPI application through the LogGOPSim toolchain. In
this case, we use LogGP as the network backend.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MPI network latency sensitivity experiment")
    parser.add_argument("-i", "--mpi-prog", dest="mpi_prog",
                        help="Path to the binary MPI program to be evaluated")
    parser.add_argument("--args", dest="prog_args", default=[], nargs="+",
                        help="Arguments that will be passed to the program during tracing.")
    parser.add_argument("-n", "--num_ranks", dest="num_ranks", default=2,
                        help="Number of ranks for the program. [Default: 2]")
    parser.add_argument("-t", "--type", dest="type", default="c",
                        help="MPI program type, determines which shared library to use. Options are ['c', 'f77']. [Default: 'c']")
    parser.add_argument("--trace-dir", dest="trace_dir", default=None,
                        help="Directory to store all the MPI trace files. "
                        "[Default: '<mpi-prog-name>-<num-ranks>-trace']")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False,
                        action="store_true",
                        help="Enable verbose output [Default: False]")
    parser.add_argument("--bin", dest="bin_file", default=None,
                        help="If provided, will skip over the tracing and bin file generation phases.")
    parser.add_argument("-o", "--output-file", dest="output_file", default=None,
                        help="Output csv file that stores the results. [Default: <mpi-prog-name>-<num-ranks>.csv]")
    args = parser.parse_args()
    verbose = args.verbose
    if args.bin_file is None:
    
        assert os.path.exists(args.mpi_prog)

        prog_name = f"{Path(args.mpi_prog).stem}-{args.num_ranks}"
        if args.trace_dir is None:
            # Sets the default trace directory
            trace_dir = f"{prog_name}-trace"
        else:
            trace_dir = args.trace_dir

        # Traces the given MPI program with trace.sh script
        if args.prog_args is None:
            trace_args = args.mpi_prog
        else:

            trace_args = f"{args.mpi_prog} {' '.join(args.prog_args)}"
        # trace_cmd = f'bash trace.sh {trace_args} {args.num_ranks} {args.type} {trace_dir}'
        trace_cmd = ["bash", "trace.sh", trace_args, str(args.num_ranks),
                     args.type, trace_dir]
        run_cmd(trace_cmd, verbose)
        print(f"[RUN-NLS] Trace {args.mpi_prog}: SUCCESS")

        # Converts the trace into bin
        goal_gen_cmd = f"bash goal_gen.sh {trace_dir} {prog_name}"
        run_cmd(goal_gen_cmd, verbose)
        print(f"[RUN-NLS] Convert trace to bin: SUCCESS")

        bin_file = f"{prog_name}.bin"
    else:
        print(f"[RUN-NLS] bin file provided, skipping to experiment");
        bin_file = args.bin_file
        assert os.path.exists(bin_file)

        prog_name = Path(bin_file).stem
    
    # Run experiments in LogGOPSim to evaluate the performance
    # the given MPI program with different network latency
    latencies = range(MIN_LATENCY, MAX_LATENCY + STEP, STEP)
    res = run_nls_exp(bin_file, latencies, verbose)
    res = list(zip(latencies, res))

    if args.output_file is None:
        output_file = f"{prog_name}.csv"
    else:
        output_file = args.output_file

    save_res(res, output_file, verbose)
