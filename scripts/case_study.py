import os
import argparse

"""
Script to run the case study experiments.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the case study experiments.")
    parser.add_argument("-d", "--data-dir", type=str, default="case_study_data",
                        help="Path to the directory containing the case study data.")
    args = parser.parse_args()

    # Creates the output directory
    if os.path.exists(args.data_dir):
        print(f"[INFO] Removing existing directory: {args.data_dir}")
        os.system(f"rm -rf {args.data_dir}")
    print(f"[INFO] Creating directory: {args.data_dir}")
    os.makedirs(args.data_dir)

    # The script consists of three steps:
    # 1. Collects the traces for ICON under three different configurations
    # 2. Produces metrics for the collected traces using two different
    #    algorithms for allreduce, namely Ring and Recursive Doubling.
    # 3. Analyzes the impact of topology on the performance of ICON.
