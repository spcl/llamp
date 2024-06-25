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
export NETCDF_FORTRAN_INSTALL_DIR=$PWD/deps/netcdf-fortran/install
export JOBS=64
export CC=gcc
export CXX=g++
export FC=gfortran
export MPICC=mpicc
export MPIFC=mpifort


# echo "[INFO] Set up UCX ..."
# # Asserts that the UCX source code is in the "validation" directory
# if [ ! -d "validation/ucx" ]; then
#     echo "[ERROR] UCX source code not found in the 'validation' directory."
#     exit 1
# fi
# # Compiles UCX
# echo "[INFO] Compiling UCX ..."
# cd validation/ucx
# ./autogen.sh
# ./configure CC=$CC CXX=$CXX --prefix=$UCX_INSTALL_DIR
# if [ $? -ne 0 ]; then
#     echo "[ERROR] UCX configuration failed."
#     exit 1
# fi
# # Copies the changed source files
# cp ../latency-injector/ucx-src/ucp.h src/ucp/api/
# cp ../latency-injector/ucx-src/ucp_request.h ../latency-injector/ucx-src/ucp_request.inl src/ucp/core/
# cp ../latency-injector/ucx-src/tag_match.c ../latency-injector/ucx-src/tag_match.inl ucx-src/tag_recv.c src/ucp/tag/

# make -j$JOBS
# # Makes sure that the compilation was successful
# if [ $? -ne 0 ]; then
#     echo "[ERROR] UCX compilation failed."
#     exit 1
# fi
# make install
# if [ $? -ne 0 ]; then
#     echo "[ERROR] UCX installation failed."
#     exit 1
# fi
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

