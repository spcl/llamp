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
    print(f"[WARNING] {CSTART}{message}{CEND}")


def print_error(message: str):
    """
    Prints an error message in color red.
    """
    CSTART = '\033[91m'
    CEND = '\033[0m'
    print(f"[ERROR] {CSTART}{message}{CEND}")


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
    print("[INFO] Building LULESH benchmark...")

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
    print("[INFO] Building HPCG benchmark...")

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
    
    arch = "Linux_MPI"
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
    os.chdir("..")

    return True



def build_lammps(verbose: bool = False) -> bool:
    pass



def build_icon(verbose: bool = False) -> bool:
    pass


# The build functions for each benchmark
build_funcs = {
    "lulesh": build_lulesh,
    "npb": build_npb,
    "hpcg": build_hpcg,
    "lammps": build_lammps,
    "gromacs": build_icon,
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
        apps = [ "lulesh", "npb", "hpcg", "lammps", "icon" ]
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