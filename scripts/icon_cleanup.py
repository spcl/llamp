import os
from tqdm import tqdm
from argparse import ArgumentParser

"""
Script for cleaning up the trace files collected from ICON.
"""

# A dictionary mapping the data type to its size in bytes
# This is used to convert the data size in the trace file from
# a Fortran handle (a unique integer for each type) to
# the actual size in bytes.
# DATA_TYPE_TO_SIZE = {
#     "15": 8, # MPI_DOUBLE
#     "13": 4, # MPI_FLOAT
#     "11": 8, # MPI_LONG
#     "24": 4, # MPI_INT
#     "5": 1, # MPI_CHAR
# }

# FIXME: Only works for MPICH
DATA_TYPE_TO_SIZE = {
    "1275070505": 8, # MPI_DOUBLE
    "13": 4, # MPI_FLOAT
    "11": 8, # MPI_LONG
    "1275072547": 8, # MPI_REAL_DOUBLE
    "1275069467": 4, # MPI_INT,
    "1275068698": 1, # MPI_CHAR
    "1275069488": 4, # MPI I4
}

# A dictionary mapping the MPI operation that needs to be converted
# to a tuple of indices, where the first element indicates the
# index of the data type and the second element indicates the
# index of the communicator in the array of arguments after
# splitting the line by the ':' character.
OPS_TO_CONVERT = {
    "MPI_Send": (4, 7),
    "MPI_Isend": (4, 7),
    "MPI_Recv": (4, 7),
    "MPI_Irecv": (4, 7),
    "MPI_Reduce": (5, 8),
    "MPI_Bcast": (4, 6),
    "MPI_Allreduce": (5, 7),
}


def convert_line_to_lap_format(line: str, num_ranks: int,
                               curr_timestamp: float = 0) -> str:
    """
    Converts the given line in the trace file to the format
    used by the old liballprof.
    """
    tokens = line.split(":")
    op_name = tokens[0]
    # Changes the timestamp from microseconds to nanoseconds
    tokens[1] = str(int(float(tokens[1]) - curr_timestamp))
    tokens[-1] = str(int(float(tokens[-1]) - curr_timestamp))
    if op_name == "MPI_Init":
        return f"{op_name}:-:0:0:{tokens[-1]}\n"

    if op_name in OPS_TO_CONVERT:
        d_type_idx, comm_idx = OPS_TO_CONVERT[op_name]
        # Converts the data type from a Fortran handle to the actual size in bytes
        data_size = DATA_TYPE_TO_SIZE[tokens[d_type_idx]]
        tokens[d_type_idx] = f"0,{data_size},0"
        # Converts the communicator from a Fortran handle to the rank number
        comm = tokens[comm_idx]
        tokens[comm_idx] = f"{comm},0,{num_ranks}"

    return ":".join(tokens) + "\n"


def clean_up_trace(in_file: str, out_file: str, num_ranks: int) -> None:
    """
    Cleans up the given trace file collected from ICON and outputs the
    cleaned up trace file to the output file.
    """
    print(f"[INFO] Cleaning up trace file {in_file}...")
    # Iterates through all the lines in the trace file
    in_file = open(in_file, "r")
    out_file = open(out_file, "w")
    cleaned_up_lines = []
    current_timestamp = 0
    # A counter for the number of times the line
    # "Current timestamp" is encountered
    num_current_timestamp = 0
    end = False
    found_start_time = False
    for line in in_file:
        # Checks if the line starts with "Current timestamp"
        # If so, sets the current timestamp to the value
        # specified in the line
        if line.startswith("# Current timestamp: "):
            num_current_timestamp += 1
            if num_current_timestamp == 3:
                current_timestamp = float(line.split(":")[1].strip())
            elif num_current_timestamp == 4:
                end = f"MPI_Finalize:{int(float(line.split(':')[1].strip()) - current_timestamp)}:-"
                cleaned_up_lines.append(end)
                break
            # print(f"[INFO] Current timestamp: {current_timestamp}")
        # Checks if the line starts with "MPI_Alltoallv"
        # If so, removes the following lines until "MPI_Finalize"
        # is encountered
        # if line.startswith("MPI_Allgather") or line.startswith("MPI_Group_translate_ranks"):
        #     # Splits the line by the ':' character
        #     tokens = line.split(":")
        #     end = f"MPI_Finalize:{int(float(tokens[1]))}:-"
        #     cleaned_up_lines.append(end)
        #     break
        # Writes to the output file in bulk at the end
        
        # Removes everything after the '#' character in a line
        line = line.split("#")[0]
        line = line.strip()
        if line != "":
            cleaned_up_lines.append(
                convert_line_to_lap_format(line, num_ranks, current_timestamp))
    
    out_file.writelines(cleaned_up_lines)
    in_file.close()
    out_file.close()

def clean_up_dir(trace_dir: str, output_dir: str, in_file_ptrn: str,
                 out_file_ptr: str) \
    -> None:
    """
    Cleans up the trace files collected from ICON given the directory
    as well as the pattern of the file names. Outputs the cleaned up
    trace files to the output directory.
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
    out_file_prefix = out_file_ptr.split("*")[0]
    out_file_suffix = out_file_ptr.split("*")[1]
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
        # Cleans up the trace file
        out_file = f"{output_dir}/{out_file_prefix}{r}{out_file_suffix}"
        clean_up_trace(in_file, out_file, num_ranks)


if __name__ == "__main__":
    # Parses the command line arguments
    parser = ArgumentParser(description="ICON trace cleanup")
    parser.add_argument("-i", "--trace_dir", required=True,
                        help="The directory containing the trace files")
    parser.add_argument("-o", "--output_dir", default=None, required=True,
                        help="The directory to store the cleaned up trace files")
    parser.add_argument("-f", "--in_file_name", default="pmpi-trace-rank-*.txt",
                        help="The pattern of the input trace file names, where * is the rank number")
    parser.add_argument("-g", "--out_file_name", default="pmpi-trace-rank-*.txt",
                        help="The pattern of the output trace file names, where * is the rank number")
    args = parser.parse_args()

    # Cleans up the trace files
    clean_up_dir(args.trace_dir, args.output_dir,
                 args.in_file_name, args.out_file_name)
