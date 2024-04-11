# LLAMP: Assessing Network Latency Tolerance of HPC Applications with Linear Programming


LLAMP (**L**ogGPS and **L**inear Programming based **A**nalyzer for **M**PI **P**rograms) is a toolchain designed for efficient analysis and quantification of network latency sensitivity and tolerance in HPC applications. By leveraging the LogGOPSim framework, LLAMP records MPI traces of MPI programs and transforms them into execution graphs. These graphs, through the use of the LogGPS model, are then converted into linear programs. They can be solved rapidly by modern linear solvers, allowing us to efficiently gather valuable metrics, such as the predicted runtime of programs, and critical path metrics.

## Quick start
### Dependencies
- Python3
- [Gurobi](https://www.gurobi.com/downloads/) linear solver
- [igraph](https://igraph.org/)

To run the latency injection experiment, you need to download the following source code and follow the instructions in `validation/latency-injector/README.md`.
- [MPICH](https://www.mpich.org/downloads/) (4.1.2)
- [UCX](https://github.com/openucx/ucx) (1.16.x)


### Generating Linear Programs from MPI Traces

First, compile `LogGOPSim`, `liballprof`, and `Schedgen` in their corresponding directories.
- To compile LogGOPSim, make sure you have `graphviz` installed. Then, call `make` to build the executable.
- To compile Schedgen, simply call `make` to build the executable. If you want to change the p2p algorithm used for a specific collective operation, you have to modify the file `process_trace.cpp` and recompile schedgen.
- To compile liballprof, change the compilers in `setup.sh` according to your system, and type `bash setup.sh`. The library can then be found in `liballprof/.libs`.

After the LogGOPSim toolchain has been built, use the `lp_gen.py` script in the `scripts` directory to generate linear programs for MPI applications directly. Descriptions of the parameters can be obtained by `python3 lp_gen.py -h`.

For example, to create a linear program for LULESH, execute the following command if you are using Open MPI:
```console
> python3 lp_gen.py -c "mpirun -x LD_PRELOAD -np 8 <path-to-lulesh> -i 100 -s 8" -p lulesh_test -v
```
If you are using MPICH:
```console
> python3 lp_gen.py -c "mpirun -envall -np 8 <path-to-lulesh> -i 100 -s 8" -p lulesh_test -v
```
If you are running your application in a cluster that uses slurm:
```console
> python3 lp_gen.py -c "srun --export=ALL -N2 -n8 <path-to-lulesh> -i 100 -s 8" -p lulesh_test -v
```
This will create a directory named `lulesh_test` under the root project directory, which will contain the traces inside the `traces` folder. The generated linear programming model will be saved as both `.lp` file and `.mps` file.

If the traces have already been collected, and you want to try out different parameters, such as `o` or `G`, add the `-s` argument when running the script to skip tracing.

To generate linear programs for [ICON](https://icon-model.org/), a few changes need to be made. To start, follow the instructions in `case-studies/icon/README.md` to compile ICON. Then, build `liballprof2`. The script for running ICON can be found in `case-studies/icon/`, make sure to set the paths as well as the `START` command correctly in the script. Type the following command to trace and produce linear programs for ICON:
```console
> python3 lp_gen.py -c "bash ../case-studies/icon/run-icon.sh" --icon -v -p icon_test
```

### Linear Program Analysis

To perform analysis on linear programs, use the `main.py` script inside `mpi-dep-graph`.

#### Network Latency Sensitivity

Run the following command inside `mpi-dep-graph` to generate the network latency sensitivity curves for your application. The results will be stored as CSV files.
```console
> python3 main.py --load-lp-model-path ../lulesh_test/lulesh_test.lp -a sensitivity --output-dir ../lulesh_test/
```
The interval of interest for $L$ can be specified via the `--l-min` and `--l-max` arguments. Change the `--step` argument to set the resolution.

#### Network Latency Tolerance
Run the following command inside `mpi-dep-graph` to generate the 1% network latency sensitivity tolerance for your application:
```console
> python3 main.py --load-lp-model-path ../lulesh_test/lulesh_test.lp -a buffer
```
To change the performance degradation threshold, use the `--lat-buffer-thresh` argument. To specify the baseline of application runtime manually, use the `--lat-buf-baseline` argument.


### Misc

- If you only intend to generate performance forecast for your application, you can replace the `-a` argument with `--solve` when executing `main.py`.
- If you want to save the MPI execution graph and have access to it, use `--export-graph-path` to specify the path for the graph. The graph will be saved as a `pkl` file.
- We currently do not provide an interface to change the network topology easily, you will have to modify the `topology` variable in `main.py` manually to adjust the configurations for Fat Tree and Dragonfly topologies.


### Contributions
LLAMP, with its linear programming approach, opens up a world of analysis possibilities that are just waiting to be explored. It is really in the hands of the users to discover new metrics or come up with interesting uses for LLAMP. Inside the source code, you will find numerous experimental features, like tools for __MPI process placement__.  If you have any ideas or an improvement, do not hesitate to dive in and submit a pull request! We are looking forward to seeing the new applications brought forward by the community.