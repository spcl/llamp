#/bin/bash

# Defines the installation directory for UCX
export UCX_ROOT=$PWD/validation/ucx
export UCX_INSTALL_DIR=$UCX_ROOT/install
# Defines the installation directory for MPICH
export MPICH_ROOT=$PWD/validation/mpich
export MPICH_INSTALL_DIR=$MPICH_ROOT/install
# Defines the installation directory for netcdf-fortran
export AUTOCONF_INSTALL_DIR=$PWD/deps/autoconf/install
export HDF5_INSTALL_DIR=$PWD/deps/hdf5/install
export NETCDF_C_INSTALL_DIR=$PWD/deps/netcdf-c/install
export NETCDF_FORTRAN_INSTALL_DIR=$PWD/deps/netcdf-fortran/install
export JOBS=64
export CC=gcc
export CXX=g++
export FC=gfortran
export MPICC=mpicc

export LD_LIBRARY_PATH=$UCX_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PATH=$UCX_INSTALL_DIR/bin:$PATH
# Makes sure that the installation was successful
# Asserts that ucx_info is in the PATH
if ! command -v ucx_info &> /dev/null
then
    echo "[ERROR] ucx_info could not be found."
fi
export PATH=$MPICH_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPICH_INSTALL_DIR/lib:$LD_LIBRARY_PATH
# Makes sure that the installation was successful
# Asserts that mpicc is in the PATH
if ! command -v mpicc &> /dev/null
then
    echo "[ERROR] mpicc could not be found."
fi

export LIBALLPROF_C=$(pwd)/liballprof/.libs/liballprof.so
export LIBALLPROF_F77=$(pwd)/liballprof/.libs/liballprof_f77.so
export LIBTIMER_C=$(pwd)/liballprof/mpi_timer.so
export LIBTIMER_F77=$(pwd)/liballprof/mpi_timer_f77.so
export LIBALLPROF2_C=$PWD/liballprof2/liballprof2.so
export LIBALLPROF2_F77=$PWD/liballprof2/liballprof2_f77.so
export PATH=$PWD/Schedgen:$PATH
export PATH=$PWD/LogGOPSim:$PATH
export LOGGOPSIM_PATH=$PWD/LogGOPSim/LogGOPSim

export GUROBI_HOME=$PWD/deps/gurobi/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH

# Makes sure that the installation was successful
# Asserts that gurobi_cl is in the PATH
if ! command -v gurobi_cl &> /dev/null
then
    echo "[ERROR] gurobi_cl could not be found."
    exit 1
fi

export PATH=$AUTOCONF_INSTALL_DIR/bin:$PATH
# Makes sure that the installation was successful
# Asserts that autoconf is in the PATH
if ! command -v autoconf &> /dev/null
then
    echo "[ERROR] autoconf could not be found."
    exit 1
fi
export PATH=$HDF5_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_DIR/lib:$LD_LIBRARY_PATH
# Makes sure that the installation was successful
# Asserts that h5pfc is in the PATH
if ! command -v h5pfc &> /dev/null
then
    echo "[ERROR] h5pfc could not be found."
    exit 1
fi
export PATH=$NETCDF_C_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_C_INSTALL_DIR/lib:$LD_LIBRARY_PATH
# Asserts that the installation was successful
# Asserts that nc-config is in the PATH
if ! command -v nc-config &> /dev/null
then
    echo "[ERROR] nc-config could not be found."
    exit 1
fi
export PATH=$NETCDF_FORTRAN_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_FORTRAN_INSTALL_DIR/lib:$LD_LIBRARY_PATH
# Makes sure that the installation was successful
# Asserts that nf-config is in the PATH
if ! command -v nf-config &> /dev/null
then
    echo "[ERROR] nf-config could not be found."
    exit 1
fi

export PATH=$PWD/deps/netgauge/:$PATH
if ! command -v netgauge &> /dev/null
then
    echo "[ERROR] netgauge could not be found."
    exit 1
fi