#!/bin/bash -l
#SBATCH --job-name="LP Model Analysis"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=100000

module load gurobi

# Generates lp model from a given graph file
DIR=/cluster/home/sishen/workspace/lgs-mpi/
MAIN=${DIR}mpi-dep-graph/main.py
NAME=npb_luC_64
LP_MODEL_PATH=${DIR}${NAME}.mps

python3 ${MAIN} --load-lp-model-path ${LP_MODEL_PATH} -v -a -p