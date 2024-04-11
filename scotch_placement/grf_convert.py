import os
import re
import numpy as np
from typing import Optional, List, Union, Tuple
from argparse import ArgumentParser


def get_comm_matrix_from_goal(path: str) -> np.ndarray:
    """
    Convert the output of the goal file produced by Schedgen
    to a GRF file that can be used by Scotch.
    """
    print(f"[INFO] Reading from {path}")
    input_file = open(path, "r")
    comm_matrix = None
    # Reads the first line which has the format "num_ranks ${num_ranks}"
    # Extracts the number of ranks
    lines = input_file.readlines()
    num_ranks = int(lines[0].split()[-1])
    assert num_ranks > 0, "Number of ranks must be positive"
    comm_matrix = np.zeros((num_ranks, num_ranks), dtype=np.int32)
    print(f"[INFO] Extracting communication matrix: {num_ranks} ranks")
    cur_rank = -1
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) < 2:
            continue

        if tokens[0] == "rank":
            curr_rank = int(tokens[1])
            print(f"[INFO] Processing rank {curr_rank}")
            continue

        if tokens[1] == "send":
            num_bytes = int(tokens[2][:-1])
            dest = int(tokens[4])
            comm_matrix[curr_rank, dest] += num_bytes
    return comm_matrix

def get_comm_matrix_from_mpitrace(path: str) -> np.ndarray:
    """
    Convert the output of the communication matrix from IBM mpitrace
    to a GRF file that can be used by Scotch.
    """
    print(f"[INFO] Reading from {path}")
    input_file = open(path, "r")
    # First line is "got nranks = <num-ranks>"
    # Extracts the number of ranks
    num_ranks = int(input_file.readline().split()[-1])
    assert num_ranks > 0, "Number of ranks must be positive"

    # Extracts the communication matrix
    comm_matrix = np.zeros((num_ranks, num_ranks), dtype=np.int32)
    print(f"[INFO] Extracting communication matrix: {num_ranks} ranks")

    for line in input_file:
        # for each connection, i.e., the p2p communication one rank has with another
        # it is in the format of
        # "rank <src-rank> sent to rank <dst-rank> : messages = <num-messages>, bytes = <num-bytes>"
        # Extracts the ID of the source rank and the destination rank
        # as well as the number of bytes sent with regex
        match = re.match(r"rank (\d+) sent to rank (\d+) : messages = (\d+), bytes = (\d+)", line)
        if match:
            src_rank = int(match.group(1))
            dst_rank = int(match.group(2))
            num_bytes = int(match.group(4))
            comm_matrix[src_rank, dst_rank] = num_bytes

    input_file.close()

    print("[INFO] Extract communication matrix: SUCCESS")
    return comm_matrix



def save_comm_matrix_as_grf(comm_matrix: np.ndarray, output_path: str) -> None:
    """
    Save the communication matrix as a GRF file.
    """
    print(f"[INFO] Converting communication matrix to GRF and saving to {output_path}")
    # GRF file has the format:
    # first line: 0
    # second line: <num-ranks> <num-edges>
    # third line: 0 010
    # For each rank, there is a line that has the format:
    # <num-edges> <num-bytes> <dst-rank> <num-bytes> <dst-rank> ...
    # where <num-edges> is the number of edges that the rank has
    # and <num-bytes> is the number of bytes sent to the destination rank
    # and <dst-rank> is the destination rank
    num_ranks = comm_matrix.shape[0]
    num_edges = np.count_nonzero(comm_matrix)
    output_file = open(output_path, "w")

    # Writes the first line
    output_file.write("0\n")
    # Writes the second line
    output_file.write(f"{num_ranks} {num_edges}\n")
    # Writes the third line
    output_file.write("0 010\n")
    for src_rank in range(num_ranks):
        # Gets the number of edges that the current rank has
        num_edges = np.count_nonzero(comm_matrix[src_rank])
        # Writes the number of edges
        output_file.write(f"{num_edges}")
        # Writes the number of bytes sent to each destination rank
        for dst_rank in range(num_ranks):
            num_bytes = comm_matrix[src_rank, dst_rank]
            if num_bytes > 0:
                output_file.write(f" {num_bytes} {dst_rank}")
        output_file.write("\n")

    output_file.close()
    print(f"[INFO] Saved grf file: {output_path}")
    print("[INFO] Save communication matrix as GRF: SUCCESS")


if __name__ == "__main__":
    parser = ArgumentParser(prog="GRF Converter")
    parser.add_argument("-i", "--input", dest="input", default=None,
                        required=True,
                        help="Path to the input file that will be parsed.")
    parser.add_argument("-f", "--format", dest="format", 
                        default="mpitrace", choices=["mpitrace", "goal"],
                        help="Format of the input file. If not given, will try to infer from the file extension.")
    parser.add_argument("-o", "--output", dest="output",
                        required=False, default=None,
                        help="If given, will store the output in the given file. [Default: <input-filename>.grf]")
    
    parser = parser.parse_args()
    input_path = parser.input
    output_path = parser.output
    # Makes sure that the input file exists
    if not os.path.isfile(input_path):
        raise ValueError(f"Input file {input_path} does not exist.")

    if output_path is None:
        # Extracts the input filename without the extension or the path
        output_path = os.path.splitext(os.path.basename(input_path))[0] + ".grf"
    
    print(f"[INFO] Converting communication data file from {parser.format} format to .grf")
    
    if parser.format == "mpitrace":
        comm_matrix = get_comm_matrix_from_mpitrace(input_path)
    elif parser.format == "goal":
        comm_matrix = get_comm_matrix_from_goal(input_path)
    else:
        raise ValueError(f"Unsupported format: {parser.format}")
    
    # Saves the communication matrix to the output file in GRF format
    save_comm_matrix_as_grf(comm_matrix, output_path)

