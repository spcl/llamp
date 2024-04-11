#!/bin/bash

# Trace a given MPI application
# Checks the number of arguments passed in
if [ $# -lt 2 ]
then
    >&2 echo "[ERROR] Name of the MPI binary and the number of ranks must be provided"
    exit 1
fi

app=$1
num_ranks=$2
# Determines whether the application was in C or Fortran
if [ -z "$3" ]
then
    # Sets the default type to C if not specified by user
    type=c
else
    type="$3"
fi

# If the trace directory is not provided
if [ -z "$4" ]
then
    trace_dir=$(echo $app | tr ' .' '-')-${num_ranks}-trace
else
    trace_dir="$4"
fi

echo "[INFO] MPI traces will be saved to: ${trace_dir}"
rm -rf "${trace_dir}"
mkdir "${trace_dir}"

if [ "$type" = "c" ]
then
    trace_lib="liballprof.so"
elif [ "$type" = 'f77' ]
then
    trace_lib="liballprof_f77.so"
else
    >&2 echo "[ERROR] Invalid type ${type}"
    >&2 echo "[ERROR] The type provided must be either 'c' or 'f77'"
    exit 1
fi

echo "[INFO] Library used: '$trace_lib'"

# Remove all the trace files from previous runs
rm /tmp/pmpi-trace-rank-*.txt

# Start tracing
LIB_PATH="liballprof/.libs/${trace_lib}"
echo "[INFO] Start tracing ${app}"
echo "[INFO] Number of ranks ${num_ranks}"
LD_PRELOAD="$LIB_PATH" mpirun --bind-to hwthread -n ${num_ranks} ${app}

if [ $? -ne 0 ]
then
    >&2 echo "[ERROR] Trace ${app}: FAILED"
    exit 1
fi

# Move all the trace files in to trace_dir
mv /tmp/pmpi-trace-rank-*.txt "$trace_dir"

if [ $? -ne 0 ]
then
    >&2 echo "[ERROR] No trace file was produced. Maybe the program was not written in ${type}?"
    >&2 echo "[ERROR] Trace ${app}: FAILED"
    exit 1
fi

echo "[INFO] Trace ${app}: SUCCESS"