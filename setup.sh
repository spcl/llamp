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
export MPIFC=mpifort


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

echo "[INFO] Set up liballprof2"
cd ../liballprof2 && make clean && make -j8
# Asserts that the compilation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] liballprof2 compilation failed."
    exit 1
fi
export LIBALLPROF2_C=$PWD/liballprof2/liballprof2.so
export LIBALLPROF2_F77=$PWD/liballprof2/liballprof2_f77.so
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


# Compiles autoconf
echo "[INFO] Set up autoconf ..."
# Asserts that the autoconf source code is in the "deps" directory
if [ ! -d "deps/autoconf" ]; then
    echo "[WARNING] autoconf source code not found in the 'deps' directory."
    echo "[INFO] Downloading autoconf source code ..."
    wget https://ftp.gnu.org/gnu/autoconf/autoconf-2.71.tar.gz
    tar -xvf autoconf-2.71.tar.gz
    # Moves the source code to the "deps" directory
    mv autoconf-2.71 deps/autoconf
    rm autoconf-2.71.tar.gz
fi
cd deps/autoconf
./configure --prefix=$AUTOCONF_INSTALL_DIR
if [ $? -ne 0 ]; then
    echo "[ERROR] autoconf configuration failed."
    exit 1
fi
make -j$JOBS
if [ $? -ne 0 ]; then
    echo "[ERROR] autoconf compilation failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "[ERROR] autoconf installation failed."
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


# Build hdf5
echo "[INFO] Set up hdf5 ..."
# Asserts that the hdf5 source code is in the "deps" directory
if [ ! -d "deps/hdf5" ]; then
    echo "[ERROR] hdf5 source code not found in the 'deps' directory."
    exit 1
fi
# Runs autogen.sh
cd deps/hdf5
./autogen.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] autogen.sh failed."
    exit 1
fi
CC=$MPICC FC=$MPIFC CFLAGS=-fPIC ./configure --enable-shared --enable-parallel --enable-fortran --enable-fortran2003 --prefix=$HDF5_INSTALL_DIR
if [ $? -ne 0 ]; then
    echo "[ERROR] hdf5 configuration failed."
    exit 1
fi
make -j$JOBS
if [ $? -ne 0 ]; then
    echo "[ERROR] hdf5 compilation failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "[ERROR] hdf5 installation failed."
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

# Build netcdf-c
echo "[INFO] Set up netcdf-c ..."
# Asserts that the netcdf-c source code is in the "deps" directory
if [ ! -d "deps/netcdf-c" ]; then
    echo "[ERROR] netcdf-c source code not found in the 'deps' directory."
    exit 1
fi
# Compiles netcdf-c
echo "[INFO] Compiling netcdf-c ..."
cd deps/netcdf-c
CC=$MPICC LDFLAGS=-L$HDF5_INSTALL_DIR/lib LIBS=-lhdf5 CPPFLAGS=-I$HDF5_INSTALL_DIR/include ./configure --enable-parallel-tests --prefix=$NETCDF_C_INSTALL_DIR --disable-libxml2
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-c configuration failed."
    exit 1
fi
make -j$JOBS
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-c compilation failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-c installation failed."
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



# Build netcdf-fortran
echo "[INFO] Set up netcdf-fortran ..."
# Asserts that the netcdf-fortran source code is in the "deps" directory
if [ ! -d "deps/netcdf-fortran" ]; then
    echo "[ERROR] netcdf-fortran source code not found in the 'deps' directory."
    exit 1
fi
# Compiles netcdf-fortran
echo "[INFO] Compiling netcdf-fortran ..."
cd deps/netcdf-fortran
CC=$MPICC FC=$MPIFC LIBS=-lnetcdf CPPFLAGS=-I$NETCDF_C_INSTALL_DIR/include LDFLAGS=-L$NETCDF_C_INSTALL_DIR/lib ./configure --enable-parallel-tests --prefix=$NETCDF_FORTRAN_INSTALL_DIR
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-fortran configuration failed."
    exit 1
fi
make -j$JOBS
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-fortran compilation failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "[ERROR] netcdf-fortran installation failed."
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


echo "[INFO] All setup: SUCCESS"