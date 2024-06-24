#/bin/bash

# Defines the installation directory for UCX
export UCX_ROOT=$PWD/validation/ucx
export UCX_INSTALL_DIR=$UCX_ROOT/install
# Defines the installation directory for MPICH
export MPICH_ROOT=$PWD/validation/mpich
export MPICH_INSTALL_DIR=$MPICH_ROOT/install
export JOBS=64
export CC=gcc-11
export CXX=g++-11
export FC=gfortran-11


echo "[INFO] Set up UCX ..."
# Asserts that the UCX source code is in the "validation" directory
if [ ! -d "validation/ucx" ]; then
    echo "[ERROR] UCX source code not found in the 'validation' directory."
    exit 1
fi
# Compiles UCX
echo "[INFO] Compiling UCX ..."
cd validation/ucx
./autogen.sh
./configure CC=$CC CXX=$CXX --prefix=$UCX_INSTALL_DIR
if [ $? -ne 0 ]; then
    echo "[ERROR] UCX configuration failed."
    exit 1
fi
# Copies the changed source files
cp ../latency-injector/ucx-src/ucp.h src/ucp/api/
cp ../latency-injector/ucx-src/ucp_request.h ../latency-injector/ucx-src/ucp_request.inl src/ucp/core/
cp ../latency-injector/ucx-src/tag_match.c ../latency-injector/ucx-src/tag_match.inl ucx-src/tag_recv.c src/ucp/tag/

make -j$JOBS
# Makes sure that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] UCX compilation failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "[ERROR] UCX installation failed."
    exit 1
fi
export LD_LIBRARY_PATH=$UCX_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export PATH=$UCX_INSTALL_DIR/bin:$PATH
# Makes sure that the installation was successful
# Asserts that ucx_info is in the PATH
if ! command -v ucx_info &> /dev/null
then
    echo "[ERROR] ucx_info could not be found."
    exit 1
fi

echo "[INFO] UCX installation: SUCCESS"

cd ../mpich

# Compile MPICH and install it
echo "[INFO] Set up MPICH ..."
# Checks if the MPICH source code is in the "validation" directory
if [ ! -d "validation/mpich" ]; then
    echo "[INFO] MPICH source code not found in the 'validation' directory."
    echo "[INFO] Downloading MPICH source code ..."
    wget https://github.com/pmodels/mpich/releases/download/v4.1.2/mpich-4.1.2.tar.gz
    tar -xvf mpich-4.1.2.tar.gz 
    # Moves the source code to the "validation" directory
    mv mpich-4.1.2 validation/mpich
    rm mpich-4.1.2.tar.gz
fi

# Compiles MPICH
echo "[INFO] Compiling MPICH ..."
echo "[INFO] Make sure to install the fortran compiler (gfortran) before compiling MPICH."
cd validation/mpich
./configure CC=$CC F77=$FC --prefix=$MPICH_INSTALL_DIR --with-ucx=$UCX_INSTALL_DIR
if [ $? -ne 0 ]; then
    echo "[ERROR] MPICH configuration failed."
    exit 1
fi
cp ../latency-injector/mpich-src/mpir_request.h src/include/
cp ../latency-injector/mpich-src/ucx_recv.h src/mpid/ch4/netmod/ucx/
cp ../latency-injector/mpich-src/ch4_* src/mpid/ch4/src/

make -j$JOBS
if [ $? -ne 0 ]; then
    echo "[ERROR] MPICH compilation failed."
    exit 1
fi
make install
# Makes sure that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] MPICH installation failed."
    exit 1
fi
export PATH=$MPICH_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPICH_INSTALL_DIR/lib:$LD_LIBRARY_PATH
# Makes sure that the installation was successful
# Asserts that mpicc is in the PATH
if ! command -v mpicc &> /dev/null
then
    echo "[ERROR] mpicc could not be found."
    exit 1
fi
echo "[INFO] MPICH installation: SUCCESS"
cd ../../


echo "[INFO] Set up liballprof ..."
cd liballprof/ && source ./setup.sh
# Asserts that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] liballprof compilation failed."
    exit 1
fi

# echo "[INFO] Set up liballprof2"
# cd ../liballprof2 && make clean && make -j8
# # Asserts that the compilation was successful
# if [ $? -ne 0 ]; then
#     echo "[ERROR] liballprof2 compilation failed."
#     exit 1
# fi
echo "[INFO] liballprof2 setup: SUCCESS"

echo "[INFO] Set up Schedgen"
echo "[WARNING] Make sure to install the 'gengetopt' package before compiling Schedgen."
cd ../Schedgen && make clean && make -j8
# Asserts that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] Schedgen compilation failed."
    exit 1
fi
echo "[INFO] Schedgen setup: SUCCESS"
export PATH=$PWD:$PATH

echo "[INFO] Set up LogGOPSim"
echo "[WARNING] Make sure to install the 're2c' and 'graphviz-dev' packages before compiling Schedgen."
cd ../LogGOPSim && make clean && make -j8
# Asserts that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] LogGOPSim compilation failed."
    exit 1
fi
echo "[INFO] LogGOPSim setup: SUCCESS"
export PATH=$PWD:$PATH
cd ..
echo "[INFO] All setup: SUCCESS"