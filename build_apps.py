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



def build_lulesh(config: dict = None,
                 jobs: int = 1,
                 verbose: bool = False) -> bool:
    """
    Clones and builds the LULESH benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building LULESH benchmark...")
    if verbose:
        print(f"[INFO] Build configuration: {config}")

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
    use_omp = "ON" if config["use_omp"] else "OFF"
    command = f"cmake .. -DWITH_MPI=OFF -DWITH_OPENMP={use_omp}"
    
    if os.system(command) != 0:
        print_error("Failed to configure the build")
        return False

    # Runs the command "make" to build the benchmark
    if os.system(f"make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    assert os.path.exists("lulesh2.0"), "lulesh executable does not exist"

    # Changes the directory to the parent directory
    os.chdir("../..")

    return True


def build_npb(config: dict = None,
              jobs: int = 1,
              verbose: bool = False) -> bool:
    """
    Clones and builds the NPB benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building NPB benchmark...")
    if verbose:
        print(f"[INFO] Build configuration: {config}")
    
    npb_name = f"NPB{config['version']}{'-MZ' if config['multi_zone'] else ''}"

    # Makes sure that the directory is already cloned
    if not os.path.exists(npb_name):
        # Removes the tar file if it exists
        if verbose:
            print(f"[INFO] Downloading the NPB benchmark from '{config['download_link']}'...")
        # Downloads the NPB benchmark from the given URL
        if os.system(f"wget {config['download_link']}") != 0:
            print_error("Failed to download the NPB benchmark")
            return False

        # Extracts the tar file to a directory named `npb_name`
        if os.system(f"tar -xf {npb_name}.tar.gz") != 0:
            print_error("Failed to extract the NPB benchmark")
            return False
    
    # Changes the directory to `{npb_name}`
    os.chdir(npb_name)
    # Changes the directory to the subdirectory that ends in "-OMP"
    os.chdir([d for d in os.listdir() if d.endswith("-OMP")][0])

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
            file.write(f"{benchmark} {config['class']}\n")
    
    # Executes the command to build the benchmark
    if os.system(f"make suite -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Changes the directory to the parent directory
    os.chdir("../../")

    return True




def build_hpcg(config: dict = None,
               jobs: int = 1,
               verbose: bool = False) -> bool:
    """
    Clones and builds the HPCG benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building HPCG benchmark...")
    if verbose:
        print(f"[INFO] Build configuration: {config}")

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
    
    print_warning(f"Make sure that all parameters are specified in the file setup/Make.{config['arch']}")


    # Changes the directory to `build`
    os.chdir("build")
    
    # Runs the configure script
    if os.system(f"../configure {config['arch']}") != 0:
        print_error("Failed to run the configure script")
        return False
    
    if os.system(f"CXX=g++ make -j {jobs}") != 0:
        print_error("Failed to build the benchmark")
        return False

    # Changes the directory to the parent directory
    os.chdir("..")

    return True


def build_tpc(config: dict = None,
              jobs: int = 1,
              verbose: bool = False) -> bool:
    """
    Clones and builds the TPC benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building TPC benchmark...")
    if verbose:
        print(f"[INFO] Build configuration: {config}")

    # Makes sure that the directory is already cloned
    if not os.path.exists("tpcc"):
        print_error("Directory 'tpc' does not exist. Make sure to clone the submodule")
        return False
    
    print_warning("Make sure to have OpenJDK 21 installed")

    # Changes the directory to `tpcc`
    os.chdir("tpcc")

    # Runs ./mvnw clean package -P <profile name>
    if os.system(f"./mvnw clean package -P {config['profile']}") != 0:
        print_error("Failed to build the benchmark")
        return False
    
    # Changes the directory to `target`
    os.chdir("target")

    name = f"benchbase-{config['profile']}"
    zip_name = f"{name}.zip"

    # Unzips the zip file
    if os.system(f"unzip {zip_name}") != 0:
        print_error("Failed to unzip the file")
        return False
    
    # Changes the directory to the parent directory
    os.chdir("../..")

    return True


def build_ml_perf(config: dict = None,
                  jobs: int = 1,
                  verbose: bool = False) -> bool:
        """
        Clones and builds the MLPerf benchmark.
        Returns True if the build is successful, False otherwise.
        """
        print("[INFO] Building MLPerf benchmark...")
        if verbose:
            print(f"[INFO] Build configuration: {config}")
    
        # Makes sure that the directory is already cloned
        if not os.path.exists("ml-perf"):
            print_error("Directory 'ml-perf' does not exist. Make sure to clone the submodule")
            return False
        
        # Changes the directory to `mlperf`
        os.chdir("ml-perf/closed/Intel/code/automation")

        # Runs the command "pip install -r requirements.txt"
        if os.system("pip install -r requirements.txt") != 0:
            print_error("Failed to install the requirements")
            return False
        

        # Runs the download.sh script
        download_command = f"model={config['model']} output_dir={config['output_dir']} conda_path={config['conda_path']} " \
                            f"bash ./download.sh"
        if os.system(download_command) != 0:
            print_error("Failed to run the download script")
            return False
        
        # Changes the directory to the parent directory
        os.chdir("..")
    
        return True


def build_spec_cpu_2017(config: dict = None,
                        jobs: int = 1,
                        verbose: bool = False) -> bool:
    """
    Clones and builds the SPEC CPU 2017 benchmark.
    Returns True if the build is successful, False otherwise.
    """
    print("[INFO] Building SPEC CPU 2017 benchmark...")
    if verbose:
        print(f"[INFO] Build configuration: {config}")

    # Makes sure that the installation directory of SPEC CPU 2017 benchmark
    # is present
    if not os.path.exists(config["install_dir"]):
        print_error(f"Directory '{config['install_dir']}' does not exist.")
        print_error("Make sure to download and extract the SPEC CPU 2017 benchmark "
              "as per the instructions on https://www.spec.org/cpu2017/Docs/quick-start.html")
        return False

    # Changes the directory to the installation directory
    os.chdir(config["install_dir"])

    print_warning("MAKE SURE TO SOURCE THE 'shrc' FILE BEFORE RUNNING THIS SCRIPT")
    
    
    # Copies the example configuration file whose name is
    # "Example-{config['base_config']}.cfg" to the config directory
    example_config = f"config/Example-{config['base_config_name']}.cfg"

    print_warning(f"MAKE SURE TO EDIT THE CONFIGURATION FILE BEFORE RUNNING THE BENCHMARKS")

    if not os.path.exists(example_config):
        print_error(f"Example configuration file '{example_config}' does not exist")
        return False
    config_name = config["config_name"]
    config_file = f"config/{config_name}.cfg"

    if os.system(f"cp {example_config} {config_file}") != 0:
        print_error("Failed to copy the configuration file")
        return False

    if verbose:
        print(f"[INFO] Configuration file {config_file} copied successfully")
    

    # Runs the command "runcpu" to build the suites
    for suite in config["suites"]:
        # Runs the command runcpu to clean the files and directories
        if os.system(f"runcpu --config={config_name} --action=clean {suite}") != 0:
            print_error(f"Failed to clean the benchmark")
            return False

        if os.system(f"runcpu --config={config_name} --action=build {suite} "
                     f"--define build_ncpus={jobs}") != 0:
            print_error(f"Failed to build the suite '{suite}'")
            return False

    return True


# The build functions for each benchmark
build_funcs = {
    "black_scholes": build_black_scholes,
    "gromacs": build_gromacs,
    "lulesh": build_lulesh,
    "npb": build_npb,
    "gemm": build_gemm,
    "hpl_mxp": build_hpl_mxp,
    "graph500": build_graph500,
    "hpcg": build_hpcg,
    "tpc": build_tpc,
    "ml_perf": build_ml_perf,
    "spec_cpu_2017": build_spec_cpu_2017,
}


def build_all(config: str = None,
              verbose: bool = False):
    """
    Builds all the benchmarks.
    """
    print("[INFO] Building all the benchmarks...")
    print(f"[INFO] Configuration file: {config}")

    # Makes sure that the configuration file exists
    if not os.path.exists(config):
        print_error(f"Configuration file '{config}' does not exist")
        return False

    # Reading the configuration yaml file
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    jobs = config["jobs"]
    for benchmark_dict in config["benchmarks"]:
        benchmark_name = list(benchmark_dict.keys())[0]
        benchmark_config = benchmark_dict[benchmark_name]

        assert "skip" in benchmark_config, f"Missing 'skip' key in the configuration of '{benchmark_name}'"
        if benchmark_config["skip"]:
            print_warning(f"Skipping the benchmark '{benchmark_name}'...")
            continue

        if benchmark_name not in build_funcs:
            print_warning(f"Unknown benchmark '{benchmark_name}'. Skipping...")
            continue
        
        if not build_funcs[benchmark_name](benchmark_config, jobs, verbose):
            print_error(f"Failed to build the benchmark '{benchmark_name}'")
            return False
        
        print_success(f"Built the benchmark '{benchmark_name}'")
    
    print("[INFO] Build all benchmarks: COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build all the benchmarks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('-c', '--config', default="build_config.yaml",
                        type=str, help='Path to the configuration file')
    args = parser.parse_args()

    build_all(config=args.config, verbose=args.verbose)