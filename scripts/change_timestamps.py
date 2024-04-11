import os
from typing import List
from argparse import ArgumentParser

"""
A script for changing all the timestamps in the trace files
collected from the Slimfly cluster. This is done by first
reading the "Current timestamp" from the start of the trace
files, then subtracting this value from all the timestamps.
"""

def change_timestamps(in_file: str, out_file: str) -> None:
    """
    Changes the timestamps in the given trace file and writes
    the new file to the output directory.
    """
    # Iterates through all the lines in the trace file
    in_file = open(in_file, "r")
    out_file = open(out_file, "w")
    changed_lines = []
    current_timestamp = 0
    for i, line in enumerate(in_file):
        line = line.strip()
        # Checks if the line starts with "Current timestamp"
        # If so, sets the current timestamp to the value
        # specified in the line
        if line.startswith("# Current timestamp"):
            current_timestamp = int(float(line.split(":")[1]))
            print(f"[INFO] Current timestamp: {current_timestamp}")
        
        # Checks if the line contains an MPI function
        # If so, changes the timestamp to the new value
        if (line != "") and (line[0] != "#"):
            tokens = line.split(":")
            assert len(tokens) >= 2, f"[ERROR] Invalid line {i + 1}: {line}"
            # Changes the timestamp from the old value
            # by subtracting the current timestamp
            if tokens[1] != "-":
                # Start timestamp
                tokens[1] = str(int(float(tokens[1])) - current_timestamp)
            
            if tokens[-1] != "-":
                # End timestamp
                tokens[-1] = str(int(float(tokens[-1])) - current_timestamp)
            changed_lines.append(":".join(tokens) + "\n")
        else:
            changed_lines.append(line + "\n")
    
    out_file.writelines(changed_lines)
    in_file.close()
    out_file.close()


def change_files_in_dir(trace_dir: str, output_dir: str,
                        in_file_ptrn: str, out_file_ptrn: str) -> None:
    """
    Changes the timestamps in the trace files in `trace_dir`
    and writes the new files to `output_dir`.
    """
    # Makes sure that the trace directory exists
    if not os.path.exists(trace_dir):
        raise FileNotFoundError(f"[ERROR] Trace directory {trace_dir} does not exist.")
    
    # Makes sure that the output directory exists
    if not os.path.exists(output_dir):
        print(f"[INFO] Output directory {output_dir} does not exist. Creating one...")
        os.makedirs(output_dir)
    
    # Iterates through all the trace files
    in_file_prefix = in_file_ptrn.split("*")[0]
    in_file_suffix = in_file_ptrn.split("*")[1]
    out_file_prefix = out_file_ptrn.split("*")[0]
    out_file_suffix = out_file_ptrn.split("*")[1]
    num_ranks = 0
    in_files = []
    while True:
        in_file = f"{trace_dir}/{in_file_prefix}{num_ranks}{in_file_suffix}"
        if not os.path.exists(in_file):
            # No more trace files
            break
        in_files.append(in_file)
        num_ranks += 1
    
    print(f"[INFO] Found {num_ranks} trace files.")

    for r, in_file in enumerate(in_files):
        print(f"[INFO] Processing trace file {in_file}...")
        # Changes the timestamps in the trace file
        out_file = f"{output_dir}/{out_file_prefix}{r}{out_file_suffix}"
        change_timestamps(in_file, out_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Change the timestamps in the trace files.")
    parser.add_argument("-i", "--trace_dir", type=str, help="The directory containing the trace files.")
    parser.add_argument("-o", "--output_dir", type=str, help="The directory to write the new trace files.")
    parser.add_argument("-f", "--in_file_ptrn", type=str,
                        default="pmpi-trace-rank-*.txt",
                        help="The pattern of the input trace files.")
    parser.add_argument("-g", "--out_file_ptrn", type=str,
                        default="pmpi-trace-rank-*.txt",
                        help="The pattern of the output trace files.")
    args = parser.parse_args()
    change_files_in_dir(args.trace_dir, args.output_dir, args.in_file_ptrn, args.out_file_ptrn)