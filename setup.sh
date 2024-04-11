#/bin/bash

echo "[INFO] Set up liballprof ..."
cd liballprof/ && bash ./setup.sh

echo "[INFO] Set up Schedgen"
cd ../Schedgen && make clean && make -j8

echo "[INFO] Set up LogGOPSim"
cd ../LogGOPSim && make clean && make -j8