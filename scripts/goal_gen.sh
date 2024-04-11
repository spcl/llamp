#!/bin/bash

# Generates the goal file given the MPI traces and
# convert it to bin with txt2bin

EXTRAPOLATION=1
VERBOSE=0
SCHEDGEN=/cluster/home/sishen/workspace/lgs-mpi/Schedgen/schedgen
LOGGOPSIM_DIR=/cluster/home/sishen/workspace/lgs-mpi/LogGOPSim
TXT2BIN=${LOGGOPSIM_DIR}/txt2bin
LOGGOPSIM=${LOGGOPSIM_DIR}/LogGOPSim

if [ $# -eq 0 ]
then
    >&2 echo "[ERROR] The trace directory must be provided"
    exit 1
fi

trace_dir="$1"


if [ ! -d $trace_dir ]
then
    >&2 echo "[ERROR] The trace directory $trace_dir does not exist"
    exit 1
fi


if [ -z "$2" ]
then
    # If the application's name is not provided
    output_name=tmp
else
    output_name=$2
fi

if [ -n "$3" ]
then
    output_dir=$3
else
    output_dir=/cluster/home/sishen/workspace/lgs-mpi/
fi

echo "[INFO] Output directory: ${output_dir}"
echo "[INFO] Generating goal file from traces stored in ${trace_dir}"
echo "[INFO] Extrapolation Factor: ${EXTRAPOLATION}"

# output_dir=/cluster/scratch/sishen/data/
# output_dir=/cluster/home/sishen/workspace/lgs-mpi/
goal_file="${output_dir}/${output_name}.goal"
${SCHEDGEN} -p trace --traces "${trace_dir}/pmpi-trace-rank-0.txt" \
    -o "${goal_file}" --traces-extr=${EXTRAPOLATION} --traces-print=$VERBOSE

if [ $? -ne 0 ]
then
    >&2 echo "[ERROR] Generate goal file: FAILED"
    exit 1
fi


echo "[INFO] Generate goal file (${goal_file}): SUCCESS"

echo "[INFO] Converting ${goal_file} to bin"

bin_file="${output_dir}/${output_name}.bin"

${TXT2BIN} -i "$goal_file" -o "$bin_file"

if [ $? -ne 0 ]
then
    >&2 echo "[ERROR] Convert to bin file: FAILED"
    exit 1
fi

echo "[INFO] Convert goal file (${goal_file}) to bin (${bin_file}): SUCCESS"

comm_dep_file="${output_dir}/${output_name}.comm-dep"
echo "[INFO] Generating communication dependency file ..."
time ${LOGGOPSIM} -f "${bin_file}" --comm-dep "${comm_dep_file}" -G 1

if [ $? -ne 0 ]
then
    >&2 echo "[ERROR] Generate communication dependency file: FAILED"
    exit 1
fi

echo "[INFO] Generate communication dependency file (${comm_dep_file}): SUCCESS"