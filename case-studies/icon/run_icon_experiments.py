import os
import pandas as pd
from typing import List
from subprocess import run, PIPE
from argparse import ArgumentParser

DEFAULT_RUN_SCRIPT="/scratch/sshen/lgs-mpi/case-studies/icon/icon.sh"

def save_res(res: List[float], result_file: str) -> None:
    """
    Saves the results to the given output CSV file.
    """
    df = pd.DataFrame(res, columns=["runtime"])
    df.to_csv(result_file, index=False)
    print(f"[INFO] Result file saved to: {result_file}")


def get_runtime_from_output(run_script_output: str) -> float:
    """
    Parses the output of ICON and searches for the rows
    "nh_solve" and "nh_hdiff" in the timer report at end of a run.
    """
    solve_time = 0
    hdiff_time = 0
    # Reads the output from ICON's run script
    assert os.path.exists(run_script_output), \
        f"[ERROR] Run script output {run_script_output} does not exist"
    
    output_file = open(run_script_output, "r")
    for line in output_file:
        if solve_time > 0 and hdiff_time > 0:
            break
        stripped = line.strip()
        if stripped.startswith("L nh_solve"):
            tokens = stripped.split()
            assert len(tokens) == 13
            solve_time = float(tokens[-3])
        elif stripped.startswith("L nh_hdiff"):
            tokens = stripped.split()
            assert len(tokens) == 13
            hdiff_time = float(tokens[-3])
            
    output_file.close()

    if solve_time == 0 or hdiff_time == 0:
        raise Exception("[ERROR] Wrong output format, cannot find row nh_solve or nh_hdiff")
    
    return solve_time + hdiff_time

def run_icon(run_script: str, rankmap_file: str, run_script_output: str,
             num_procs: int, timeout: int) -> float:
    """
    Performs one trial of the experiment.
    Returns the time spent in ICON's dycore. This is obtained
    by looking at the rows containing nh_solve and nh_hdiff in the
    timer report at the end of a run. The final result would
    be the sum of those two numbers.
    """
    cmd = f"{run_script} {rankmap_file} {num_procs}"
    print(f"[INFO] Running command: {cmd}", flush=True)
    proc = run(cmd.split(), stdout=PIPE, stderr=PIPE, timeout=timeout)
    if proc.returncode != 0:
        # If an error occurred
        raise Exception(proc.stderr.decode())
    else:
        try:
            return get_runtime_from_output(run_script_output)
        except Exception as ex:
            raise ex


def run_experiment(run_script: str, rankmap_file: str, result_file: str,
                   run_script_output: str, num_procs: int,
                   num_trials: int, timeout: int, ignore_ex: bool) -> None:
    """
    Runs ICON for the given number of trials. Outputs the result to the
    path specified by `result_file`.
    @param run_script: Path to ICON's run script. The script should be
    able to accept 2 arguments the first of which specifies the number
    of processes, while the second argument specifies `rankmap_file`.
    @param rankmap_file: Path to the file containing the mapping of
    ranks to nodes.
    @param result_file: The path of the result CSV file.
    @param run_script_output: The path to the output of the ICON run script.
    @param num_procs: Number of processes to run, it needs to be
    divisible by 16.
    @param num_trials: Number of trials to run.
    @param timeout: Number of seconds to wait before a single trial times out.
    @param ignore_ex: If True, will 
    """
    print("[INFO] Running experiments for ICON")
    assert (num_procs % 16 == 0), "[ERROR] Number of processes must be divisible by 16"
    res = []
    for n in range(num_trials):
        print(f"[INFO] Trial {n + 1} / {num_trials}", flush=True)
        try:
            runtime = run_icon(run_script, rankmap_file, run_script_output,
                               num_procs, timeout)
            res.append(runtime)
            print(f"[INFO] Dycore time taken: {runtime} s")
        except Exception as ex:
            if ignore_ex:
                print("[WARNING] An exception occurred, ignoring ...")
                print(ex)
            else:
                print("[ERROR] An exception ocurred, stopping ...")
                print(ex)
                exit(-1)
    save_res(res, result_file)

            

if __name__ == "__main__":
    # Parses the command line arguments
    parser = ArgumentParser(description="Run ICON experiments")
    parser.add_argument("-s", "--run-script", dest="run_script",
                        default=DEFAULT_RUN_SCRIPT,
                        help="Path to ICON's run script.")
    parser.add_argument("-n", "--np", dest="num_procs", default=64, type=int,
                        help="Number of processes to run. It has to be divisble by 16.")
    parser.add_argument("-f", "--rankmap-file", dest="rankmap_file",
                        default="rankmap.txt",
                        help="Path to the file containing mappings of all the ranks to nodes.")
    parser.add_argument("-r", "--result-file",
                        dest="result_file", default="results.csv",
                        help="Path to a CSV file that stores the experiment results.")
    parser.add_argument("-o", "--run_script_output", dest="run_script_output",
                        default="run.out",
                        help="Path to the output of the ICON run script.")
    parser.add_argument("-t", "--num-trials", dest="num_trials",
                        default=10, type=int,
                        help="Number of trials to run.")
    parser.add_argument("--timeout", dest="timeout", default=600, type=int,
                        help="Number of seconds to wait before a single trial times out.")
    parser.add_argument("-e", "--ignore-exceptions", dest="ignore_ex", action="store_true",
                        help="If set, will keep running the experiments even if an exception occurs.")
    args = parser.parse_args()
    
    print("Files:")
    print(f"Run script       : {args.run_script}")
    print(f"Rankmap file     : {args.rankmap_file}")
    print(f"Result file      : {args.result_file}")
    print(f"Run script output: {args.run_script_output}")
    print("Experiment config:")
    print(f"Number of trials   : {args.num_trials}")
    print(f"Number of processes: {args.num_procs}")
    print(f"Timeout            : {args.timeout} s")
    print(f"Ignore exception   : {args.ignore_ex}")
    
    run_experiment(args.run_script, args.rankmap_file, args.result_file,
                   args.run_script_output, args.num_procs, args.num_trials,
                   args.timeout, args.ignore_ex)
