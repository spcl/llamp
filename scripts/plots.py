import os
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
warnings.filterwarnings('ignore')

"""

"""



def plot_speedup_results(data_dir: str) -> None:
    """
    Plots the results from the speedup test
    """
    res_file_path = f"{data_dir}/speedup_results.csv"
    assert os.path.exists(res_file_path), f"[ERROR] Results file does not exist: {res_file_path}"
    df = pd.read_csv(res_file_path)

    def log10(x):
        return np.log10(x)
    
    # Calculates the average speedup of Gurobi over LogGOPSim
    speedups = {}
    # Groups the dataframe by benchmark and method
    grouped = df.groupby(["benchmark", "method"])
    benchmarks = df["benchmark"].unique()
    x_axis_labels = benchmarks
    for name, group in grouped:
        benchmark, method = name
        if method == "gurobi":
            gurobi_time = group["runtime"].mean()
        else:
            loggopsim_time = group["runtime"].mean()
            speedup = loggopsim_time / gurobi_time
            speedups[benchmark] = speedup
    
    # Change the value of "method" column to "Gurobi" or "LogGOPSim"
    df["method"] = df["method"].apply(lambda x: "Gurobi Solver" if x == "gurobi" else "LogGOPSim")
    gurobi_runtimes = []
    loggopsim_runtimes = []
    for benchmark in benchmarks:
        gurobi_runtimes.append(df[(df["benchmark"] == benchmark) & (df["method"] == "Gurobi Solver")]["runtime"].mean())
        loggopsim_runtimes.append(df[(df["benchmark"] == benchmark) & (df["method"] == "LogGOPSim")]["runtime"].mean())

    # Reorders the speedup values according to the benchmarks
    speedups = [speedups[benchmark] for benchmark in benchmarks]
    # Plots a bar chart with seaborn in log scale
    # palette = sns.color_palette("pastel")
    palette = ["#0E8A7D", "#92BA51"]

    # Plots the barplot with matplotlib
    plt.figure(figsize=(13, 3.5))
    # Enabels gridline
    plt.grid(True, axis='y', zorder=0)
    # Move gridline to the background
    plt.gca().set_axisbelow(True)
    # Fetches the mean runtime of Gurobi and LogGOPSim from mean_runtimes
    bar_width = 0.4
    fontsize = 14
    br1 = np.arange(len(gurobi_runtimes))
    br2 = [x + bar_width for x in br1]
    # Plots the average runtime of Gurobi and LogGOPSim together
    plt.bar(br1, gurobi_runtimes, label="LLAMP", width=bar_width, color=palette[0])
    plt.bar(br2, loggopsim_runtimes, label="LogGOPSim", width=bar_width, color=palette[1])

    # Adds the average runtime of Gurobi and LogGOPSim on top of the bars
    for i, runtime in enumerate(gurobi_runtimes):
        if runtime > 100:
            text = f"{runtime:.0f}"
        elif runtime > 1:
            text = f"{runtime:.1f}"
        else:
            text = f"{runtime:.2f}"
        if runtime > 1:
            plt.text(i, runtime + 0.1, text, ha='center', va='bottom')
        else:
            plt.text(i, runtime + 0.01, text, ha='center', va='bottom')
    for i, runtime in enumerate(loggopsim_runtimes):
        if runtime > 100:
            text = f"{runtime:.0f}"
        elif runtime > 1:
            text = f"{runtime:.1f}"
        else:
            text = f"{runtime:.2f}"
        if runtime > 1:
            plt.text(i + bar_width, runtime + 0.1, text, ha='center', va='bottom')
        else:
            plt.text(i + bar_width, runtime + 0.01, text, ha='center', va='bottom')

    # Plots an arrow that shows the difference between the two bars for each benchmark
    for i, speedup in enumerate(speedups):
        loggopsim_runtime = loggopsim_runtimes[i]
        gurobi_runtime = gurobi_runtimes[i]
        arrow_top = loggopsim_runtime
        arrow_bot = gurobi_runtime
        x = i + bar_width
        # Uses two-headed arrow to plot the difference between the two bars
        plt.annotate("", xy=(x, arrow_top), xytext=(x, arrow_bot),
                    arrowprops=dict(arrowstyle='<->', lw=1, color="black"), 
                    va='center', ha='center')
        label_y = 10 ** ((log10(arrow_top) + log10(arrow_bot)) / 2) * 0.7
        if speedup > 100:
            text = f"{speedup:.0f}x"
        else:
            text = f"{speedup:.1f}x"
        plt.text(x + 0.23, label_y, text, ha='center', va='bottom', color="#FF0510")

    # Log scale
    plt.yscale("log")
    # Sets the x-axis labels
    plt.xticks([r + bar_width / 2 for r in range(len(gurobi_runtimes))], x_axis_labels, rotation=0)
    # Sets the y lim to 10^0 and 10^4
    plt.ylim([10 ** -2, 10 ** 3.6])
    # Sets the y-axis label to "Runtime [s]"
    plt.ylabel("Runtime [s]", fontsize=fontsize)
    # Sets x ticks to fontsize
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # Sets the x-axis label to "Benchmark"
    plt.xlabel("")
    # Increase the font size of the axis labels
    plt.rcParams.update({'font.size': fontsize})
    # Remove legend border
    legend = plt.legend(title="", ncol=2, loc='upper left', columnspacing=0.5, fontsize=fontsize)
    # legend.get_frame().set_linewidth(0.0)
    plt.tight_layout()
    # Sets frameon=False

    plt.savefig(f"{data_dir}/speedup.png", bbox_inches='tight')
    plt.savefig(f"{data_dir}/speedup.pdf", bbox_inches='tight')
    print(f"[INFO] Speedup plot saved to {data_dir}/speedup.png and {data_dir}/speedup.pdf")


def plot_validation_results(data_dir: str) -> None:
    pass



def plot_case_study_results(data_dir: str) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting script')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the data')
    parser.add_argument('-e', "--experiment", default="speedup",
                        options=["speedup", "validation", "case-study"],
                        help="Experiment to plot: speedup, validation, case-study")

    args = parser.parse_args()

    if args.experiment == "speedup":
        plot_speedup_results(args.data_dir, verbose=True)
    elif args.experiment == "validation":
        plot_validation_results(args.data_dir, verbose=True)
    elif args.experiment == "case-study":
        plot_case_study_results(args.data_dir, verbose=True)