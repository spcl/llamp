import os
import yaml
import argparse

# Script that builds all the benchmarks

def print_warning(message: str):
    """
    Prints a warning message in color orange.
    """
    CSTART = '\033[93m'
    CEND = '\033[0m'
    print(f"{CSTART}[WARNING] {message}{CEND}")


def print_error(message: str):
    """
    Prints an error message in color red.
    """
    CSTART = '\033[91m'
    CEND = '\033[0m'
    print(f"{CSTART}[ERROR] {message}{CEND}")


def print_success(message: str):
    """
    Prints a success message in color green.
    """
    CSTART = '\033[92m'
    CEND = '\033[0m'
    print(f"{CSTART}{message}{CEND}")



def build_lulesh(jobs: int = 1,
                 verbose: bool = False) -> bool:
    """
    Clones and builds the LULESH benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building LULESH...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("lulesh"):
        print_error("Directory 'lulesh' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to lulesh
    os.chdir("lulesh")
    # Removes the build directory if it exists
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False

    # Runs the command "mkdir build" to create a build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False
    
    # Changes the directory to build
    os.chdir("build")

    # Runs the command "cmake" to configure the build
    command = f"cmake .. -DWITH_MPI=ON -DWITH_OPENMP=ON"
    
    if os.system(command) != 0:
        print_error("Failed to configure the build")
        return False

    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    assert os.path.exists("lulesh2.0"), "lulesh executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../../")

    return True


def build_npb(jobs: int = 1,
              verbose: bool = False) -> bool:
    """
    Clones and builds the NPB benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building NPB benchmark...")
    
    npb_name = f"NPB3.4.3"
    download_link = "https://www.nas.nasa.gov/assets/npb/NPB3.4.3.tar.gz"

    # Makes sure that the directory is already cloned
    if not os.path.exists(npb_name):
        # Removes the tar file if it exists
        if verbose:
            print(f"[INFO] Downloading the NPB benchmark from '{download_link}'...")
        # Downloads the NPB benchmark from the given URL
        if os.system(f"wget {download_link}") != 0:
            print_error("Failed to download the NPB benchmark")
            return False

        # Extracts the tar file to a directory named `npb_name`
        if os.system(f"tar -xf {npb_name}.tar.gz") != 0:
            print_error("Failed to extract the NPB benchmark")
            return False
    
    # Changes the directory to `{npb_name}`
    os.chdir(npb_name)
    # Changes the directory to the subdirectory that ends in "-MPI"
    os.chdir([d for d in os.listdir() if d.endswith("-MPI")][0])

    # Copies the configuration 'make.def.template' to 'make.def'
    if os.system("cp config/make.def.template config/make.def") != 0:
        print_error("Failed to copy the configuration file")
        return False
    
    # Copies the suite configuration file to the current directory
    if os.system(f"cp config/suite.def.template config/suite.def") != 0:
        print_error("Failed to copy the suite configuration file")
        return False
    
    benchmarks = ["ft", "mg", "sp", "lu", "bt", "is", "ep", "cg", "ua"]
    # Updates the suite configuration file with the benchmarks
    with open("config/suite.def", "w") as file:
        for benchmark in benchmarks:
            # CLASS C
            file.write(f"{benchmark} C\n")
    
    # Executes the command to build the benchmark
    if os.system(f"make suite -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Changes the directory to the parent directory
    os.chdir("../../")

    return True




def build_hpcg(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Clones and builds the HPCG benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building HPCG...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("hpcg"):
        print_error(f"Directory 'hpcg' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `HPCG-benchmark`
    os.chdir("hpcg")

    # Deletes the build directory if it exists
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False
    
    # Creates the build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False
    
    arch = "MPI_GCC_OMP"
    print_warning(f"Make sure that all parameters are specified in the file setup/Make.{arch}")


    # Changes the directory to `build`
    os.chdir("build")
    
    # Runs the configure script
    if os.system(f"../configure {arch}") != 0:
        print_error("Failed to run the configure script")
        return False
    
    if os.system(f"CXX=mpicxx make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False

    # Changes the directory to the parent directory
    os.chdir("../../")

    return True


def build_lammps(jobs: int = 1,
                 verbose: bool = False) -> bool:
    """
    Builds the LAMMPS application.
    """
    print("[INFO] Building LAMMPS...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("lammps"):
        print_error("Directory 'lammps' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `lammps`
    os.chdir("lammps")

    # Makes sure that the build directory does not exist
    if os.path.exists("build"):
        if os.system("rm -rf build") != 0:
            print_error("Failed to remove the build directory")
            return False
    
    # Creates the build directory
    if os.system("mkdir build") != 0:
        print_error("Failed to create the build directory")
        return False

    # Changes the directory to `build`
    os.chdir("build")

    # Runs the command "cmake" to configure the build
    if os.system("cmake ../cmake -DBUILD_MPI=yes -DBUILD_OMP=yes -DPKG_MANYBODY=yes") != 0:
        print_error("Failed to configure the build")
        return False

    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Asserts that the 'lmp' executable exists
    assert os.path.exists("lmp"), "lmp executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../../")
    return True


def build_milc(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Builds the MILC application.
    """
    print("[INFO] Building MILC...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("milc_qcd"):
        print_error("Directory 'milc_qcd' does not exist. Make sure to clone the submodule")
        return False
    
    os.chdir("milc_qcd")

    # Checks if the scidac directory exists
    if os.path.exists("scidac"):
        # Removes the scidac directory
        if os.system("rm -rf scidac") != 0:
            print_error("Failed to remove the scidac directory")
            return False
    
    # Creates the scidac directory
    if os.system("mkdir scidac") != 0:
        print_error("Failed to create the scidac directory")
        return False

    print("[INFO] Building SCIDAC dependencies")
    # Changes the directory to `scidac`
    os.chdir("scidac")
    # Clones the https://github.com/usqcd-software/qmp.git repository
    if os.system("git clone https://github.com/usqcd-software/qmp.git") != 0:
        print_error("Failed to clone the qmp repository")
        return False
    
    # Changes the directory to `qmp`
    os.chdir("qmp")
    # Runs the command "autoreconf -f -i" to generate the configure script
    if os.system("autoreconf -f -i") != 0:
        print_error("Failed to generate the configure script")
        return False

    # Runs the command "./configure --prefix=$PWD/install" to configure the build
    if os.system("./configure --prefix=$PWD/install CC=mpicc --with-qmp-comms-type=MPI") != 0:
        print_error("Failed to configure the build")
        return False
    
    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build QMP")
        return False

    # Runs the command "make install" to install the benchmark
    if os.system("make install") != 0:
        print_error("Failed to install QMP")
        return False

    # Asserts that 'qmp-config' exists
    assert os.path.exists("install/bin/qmp-config"), "qmp-config does not exist"

    # Changes the directory to the ROOT directory of MILC
    os.chdir("../../")

    # Replaces line 28 in the Makefile with the following:
    # MPP ?= true
    if os.system("sed -i '28s/.*/MPP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 28 in the Makefile")
        return False
    
    # Replaces line 107 in the Makefile with the following:
    # OMP ?= true
    if os.system("sed -i '107s/.*/OMP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 107 in the Makefile")
        return False

    # Replaces line 293 in the Makefile with the following:
    # WANTQMP ?= true
    if os.system("sed -i '293s/.*/WANTQMP ?= true/' Makefile") != 0:
        print_error("Failed to replace line 293 in the Makefile")
        return False
    
    # Replaces line 301 in the Makefile with the following:
    # SCIDAC = $(shell pwd)/../scidac
    if os.system("sed -i '301s/.*/SCIDAC = $(shell pwd)\/..\/scidac/' Makefile") != 0:
        print_error("Failed to replace line 301 in the Makefile")
        return False

    # Replaces line 302 in the Makefile with the following:
    # TAG=/install
    if os.system("sed -i '302s/.*/TAG=\/install/' Makefile") != 0:
        print_error("Failed to replace line 302 in the Makefile")
        return False

    # Copies the Makefile to the directory ks_imp_dyn
    if os.system("cp Makefile ks_imp_dyn/") != 0:
        print_error("Failed to copy the Makefile to the ks_imp_dyn directory")
        return False
    
    # Changes the directory to `ks_imp_dyn`
    os.chdir("ks_imp_dyn")
    # Runs the command "make clean"
    if os.system("make clean") != 0:
        print_error("Failed to clean the directory")
        return False

    # Inserts the following line to line 46 of Make_template
    # '  reunitarize_ks.o \'
    if os.system("sed -i '46i\\  reunitarize_ks.o \\\\' Make_template") != 0:
        print_error("Failed to insert a line to Make_template")
        return False

    # Runs the command "make su3_rmd" to build the application
    if os.system(f"make su3_rmd -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False

    assert os.path.exists("su3_rmd"), "su3_rmd executable does not exist"
    return True



def build_icon(jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Builds the ICON application.
    """
    print("[INFO] Building ICON...")

    # Makes sure that the directory is already cloned
    if not os.path.exists("icon"):
        print_error("Directory 'icon' does not exist. Make sure to clone the submodule")
        return False
    
    # Changes the directory to `icon`
    os.chdir("icon")

    # Configures the build
    config_cmd = './configure --disable-loop-exchange --disable-jsbach --enable-mpi '\
    '--disable-gpu CFLAGS="-I${NETCDF_C_INSTALL_DIR}/include" FCFLAGS="-g -fallow-argument-mismatch '\
    '-I${NETCDF_FORTRAN_INSTALL_DIR}/include -I${NETCDF_C_INSTALL_DIR}/include" '\
    'LDFLAGS="-L${NETCDF_C_INSTALL_DIR}/lib -L${NETCDF_FORTRAN_INSTALL_DIR}/lib" '\
    'LIBS="-lnetcdff -lnetcdf -lopenblas" CC=mpicc FC=mpif90 '\
    '--disable-rte-rrtmgp --disable-mpi-rget --disable-coupling --enable-openmp'
    print(f"[INFO] Config command: {config_cmd}")
    if os.system(config_cmd) != 0:
        print_error("Failed to configure the build")
        return False

    mpi_src_dir = "../../case-studies/icon/icon-src-release"
    # Copies the file from '../../case-studies/icon/icon-src/mo_mpi.f90' to
    # src/mo_mpi.f90
    if os.system(f"cp {mpi_src_dir}/mo_mpi.f90 src/parallel_infrastructure/mo_mpi.f90") != 0:
        print_error("Failed to copy the file 'mo_mpi.f90'")
        return False
    
    # Copies the file from '../../case-studies/icon/icon-src/mo_atmo_nonhydrostatic.f90' to
    # src/drivers/mo_atmo_nonhydrostatic.f90
    if os.system(f"cp {mpi_src_dir}/mo_atmo_nonhydrostatic.f90 src/drivers/mo_atmo_nonhydrostatic.f90") != 0:
        print_error("Failed to copy the file 'mo_atmo_nonhydrostatic.f90'")
        return False

    # Runs 'make clean'
    if os.system("make clean") != 0:
        print_error("Failed to clean the directory")
        return False

    # Builds the application
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    assert os.path.exists("bin/icon"), "icon executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../")
    return True



# The build functions for each benchmark
build_funcs = {
    "lulesh": build_lulesh,
    "npb": build_npb,
    "hpcg": build_hpcg,
    "milc": build_milc,
    "lammps": build_lammps,
    "icon": build_icon,
}


def build_app(app: str,
              verbose: bool = False,
              jobs: int = 1):
    """
    Build the given application.
    """
    print(f"[INFO] Build application: {app}")
    os.chdir("apps")

    if app == "all":
        apps = [ "lulesh", "npb", "hpcg", "lammps", "milc", "icon" ]
    else:
        apps = [app]

    for app in apps:
        if app not in build_funcs:
            print_warning(f"Unknown benchmark '{app}'. Skipping...")
            continue
        
        if not build_funcs[app](jobs, verbose):
            print_error(f"Failed to build the benchmark '{app}'")
            return False
        
        print_success(f"Built the application '{app}'")
    
    print(f"[INFO] Build {app}: COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build all the benchmarks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--app', default='all', help='The application to build, options are [lulesh, npb, hpcg, lammps, icon]')
    parser.add_argument('-j', '--jobs', type=int, default=32, help='Number of jobs to run in parallel while compiling')
    args = parser.parse_args()

    build_app(app=args.app, verbose=args.verbose, jobs=args.jobs)