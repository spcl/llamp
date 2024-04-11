#!/bin/bash -l
#SBATCH --job-name="graph gen milc 64"
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2500
#SBATCH --ntasks=1
#SBATCH --partition=normal.4h
#SBATCH --output=/cluster/scratch/sishen/out/milc_64_graph_gen.out

DIR=/cluster/home/sishen/workspace/lgs-mpi/
# DATA_DIR=/cluster/scratch/sishen/data/
DATA_DIR=/cluster/home/sishen/workspace/lgs-mpi/
NAME=milc_slimfly_64
# LP_MODEL_NAME=${NAME}_dragonfly.lp
LP_MODEL_NAME=${NAME}.lp

MAIN=${DIR}mpi-dep-graph/main.py
GOAL=${DATA_DIR}${NAME}.goal
COMM_DEP=${DATA_DIR}${NAME}.comm-dep
GRAPH_PATH=${DATA_DIR}${NAME}.pkl
LP_MODEL_PATH=${DATA_DIR}${LP_MODEL_NAME}

# ICON 256 o=6030
# ICON 256 ring o=3483
# ICON 64 o=7400
# ICON 64 ring o=6900
# ICON 32 o=8500
# ICON 32 ring o=8200
o=6000
echo "[INFO] ${NAME}"
echo "[INFO] Chosen o: ${o}"
echo "CPU info"
lscpu
python3 ${MAIN} -g ${GOAL} -c ${COMM_DEP} \
    --export-graph-path ${GRAPH_PATH} \
    --export-lp-model-path ${LP_MODEL_PATH} \
    -v -o ${o} -G 0.018 --topology="dragonfly"
# python3 ${MAIN} -g ${GOAL} -c ${COMM_DEP} \
#     --load-graph-path ${GRAPH_PATH} \
#     --export-lp-model-path ${LP_MODEL_PATH} \
#     -v -o ${o} -G 0.013 --topology="dragonfly"