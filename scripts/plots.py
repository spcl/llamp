import os
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, List, Dict, Optional
from utils import *
warnings.filterwarnings('ignore')

"""
Script to plot the results of the experiments
"""
NET_PARAMS_FILE = "netparams.csv"

def compute_rrmse(act_runtime_file_path: str, est_runtime_file_path: str) \
    -> Tuple[float, float]:
    """
    Computes the RRMSE between the estimated runtime and the actual runtime
    """
    assert os.path.exists(act_runtime_file_path), f"[ERROR] Actual runtime file does not exist: {act_runtime_file_path}"
    assert os.path.exists(est_runtime_file_path), f"[ERROR] Estimated runtime file does not exist: {est_runtime_file_path}"

    act_df = pd.read_csv(act_runtime_file_path)
    est_df = pd.read_csv(est_runtime_file_path)

    L_0, _, _ = get_net_params(1024, NET_PARAMS_FILE)

    est_df = est_df.sort_values(by=['L'])
    est_df['L'] = (est_df['L'] - L_0) / 1000
    est_df["runtime"] = est_df["runtime"] / 1e9

    # Rounds the L column to the nearest 1
    est_df['L'] = est_df['L'].round()
    act_df = act_df.groupby('L').mean().reset_index()

    sum = 0
    sum_act = 0
    for l in est_df['L']:
        l = int(l)
        est_runtime = est_df.loc[est_df['L'] == l, 'runtime'].values[0]
        act_runtime = act_df.loc[act_df['L'] == l, 'runtime'].values[0]
        sum += (act_runtime - est_runtime) ** 2
        sum_act += act_runtime

    N = len(est_df)

    rmse = np.sqrt(sum / N)
    rrmse = rmse / (sum_act / N) * 100

    return rmse, rrmse


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
    """
    Plots the results from the validation experiments
    """
    WIDTH = 7
    HEIGHT = 3.4
    MARGIN = 0.03
    L_0, _, _ = get_net_params(1024, NET_PARAMS_FILE)

    assert os.path.exists(data_dir), \
        f"[ERROR] Data directory does not exist: {data_dir}"
    # Counts the number of subdirectories in the data directory
    # which signifies the number of applications
    app_dirs = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    print(app_dirs)
    N = len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])
    # Counts the number of subdirectories in every application directory
    # which signifies the number of node configurations
    num_configs = []
    plot_titles = {}
    for app_dir in app_dirs:
        node_dirs = [
            int(name.upper().split('_')[1]) for name in os.listdir(f"{data_dir}/{app_dir}") \
                if os.path.isdir(os.path.join(f"{data_dir}/{app_dir}", name))
        ]
        plot_titles[app_dir] = node_dirs
        num_configs.append(len(node_dirs))
    # Makes sure that every application has the same number of node configurations,
    # meaning that all numbers in num_configs are the same
    assert len(set(num_configs)) == 1, "[ERROR] Number of node configurations are not the same for all applications"
    # Fetches the number of node configurations
    M = num_configs[0]    
    
    fig, axs = plt.subplots(N, M, figsize=(WIDTH, HEIGHT))
    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    lat_buf_width = 2
    buffer_font = 10

    for i, (app, node_configs) in enumerate(plot_titles.items()):
        # Sorts the node configurations in ascending order
        node_configs.sort()
        for j in range(len(node_configs)):
            config_name = f"{app}_{node_configs[j]}"
            title = f"{app.upper()} {node_configs[j]}"
            # Obtains input
            out_dir = f"{data_dir}/{app}/{config_name}/output"
            assert os.path.exists(out_dir), f"[ERROR] Output directory does not exist: {out_dir}"
            lat_inject_file = f"{out_dir}/{config_name}_lat_injection.csv"
            assert os.path.exists(lat_inject_file), f"[ERROR] Runtime file does not exist: {lat_inject_file}"

            df = pd.read_csv(lat_inject_file)
            # Sets gridlines
            axs[i, j].grid(True, zorder=0)

            with sns.axes_style("whitegrid"):
                # Top plot in each subplot
                sns.lineplot(data=df, x="L", y="runtime", ax=axs[i, j], label="Measured", marker="o")

                # Loads the runtime estimation from our model
                runtime_file = f"{out_dir}/{config_name}_runtime.csv"

                assert os.path.exists(runtime_file), f"[ERROR] Runtime file does not exist: {runtime_file}"

                df2 = pd.read_csv(runtime_file)
                # Converts the runtime to seconds
                df2["runtime "]= df2["runtime"] / (10 ** 9)

                # Sets the minimum of L to 0
                df2["L"] = (df["L"] - L_0) / 1000

                sns.lineplot(data=df2, x="L", y="runtime", ax=axs[i, j], label="LLAMP", marker='X')

                axs[i, j].set_title(title, fontsize=font_size)

                # Reads the latency tolerance file
                lat_tolerance_file = f"{out_dir}/{config_name}_lat_tolerance.csv"
                assert os.path.exists(lat_tolerance_file), f"[ERROR] Latency buffer file does not exist: {lat_tolerance_file}"
                df_buffer = pd.read_csv(lat_tolerance_file)
                # Retrieves the three latency tolerance values by
                # reading the value from the column "latency_tolerance"
                # given different values for the column "latency_threshold"
                buffer1 = df_buffer[df_buffer["latency_threshold"] == 0.01]["latency_tolerance"].values[0]
                buffer2 = df_buffer[df_buffer["latency_threshold"] == 0.02]["latency_tolerance"].values[0]
                buffer3 = df_buffer[df_buffer["latency_threshold"] == 0.05]["latency_tolerance"].values[0]

                # Converts to microseconds
                buffer1 /= 1000
                buffer2 /= 1000
                buffer3 /= 1000

                # Sets the x and y limits of the axis according to
                # the actual runtime data
                diff = df["L"].max() * (1 + MARGIN) - df["L"].max()
                x_lim = (-diff, df["L"].max() * (1 + MARGIN))

                axs[i, j].set_xlim(xmin=x_lim[0], xmax=x_lim[1])
                
                # Removes the x-ticks
                axs[i, j].set_xticks([])

                # Groups the runtime data by L and computes the mean
                df_mean = df.groupby("L").mean()
                y_lim = (df_mean["runtime"].min() * (1 - MARGIN), df_mean["runtime"].max() * (1 + MARGIN))

                shade_y_lim = (y_lim[0] - 1000, y_lim[1] + 1000)
                text_y_pos = 0.9 * y_lim[1] + 0.1 * y_lim[0]
                # Draws a horizontal line
                axs[i, j].axvline(x=buffer3, color='red', linestyle='--', zorder=100, linewidth=lat_buf_width)
                axs[i, j].fill_betweenx(shade_y_lim, -100, buffer3, color='red', alpha=0.1)
                x_pos = (buffer3 + buffer2) / 2
                t = axs[i, j].text(x_pos, text_y_pos, f"{buffer3:.1f}", fontsize=buffer_font, color="red", ha="center", va="top")

                axs[i, j].axvline(x=buffer2, color='orange', linestyle='--', zorder=100, linewidth=lat_buf_width)
                axs[i, j].fill_betweenx(shade_y_lim, -100, buffer2, color='orange', alpha=0.1)
                x_pos = (buffer2 + buffer1) / 2
                axs[i, j].text(x_pos, text_y_pos, f"{buffer2:.1f}", fontsize=buffer_font, color="orange", ha="center", va="top")

                axs[i, j].axvline(x=buffer1, color='green', linestyle='--', zorder=100, linewidth=lat_buf_width)
                axs[i, j].fill_betweenx(shade_y_lim, -100, buffer1, color='green', alpha=0.1)
                x_pos = (buffer1 + x_lim[0]) / 2
                axs[i, j].text(x_pos, text_y_pos, f"{buffer1:.1f}", fontsize=buffer_font, color="green", ha="center", va="top")

                axs[i, j].set_xlabel('')
                axs[i, j].tick_params(axis='x', labelsize=14)
                axs[i, j].tick_params(axis='y', labelsize=14)

                # Labels RRMSE
                rrmse = compute_rrmse(lat_inject_file, runtime_file)[1]
                x_pos = (x_lim[0] + x_lim[1]) / 2
                y_pos = y_lim[0] + 0.2 * (y_lim[1] - y_lim[0])

                axs[i, j].text(x_pos, y_pos, f"RRMSE: {rrmse:.2f}%", fontsize=font_size - 1,
                               ha='center', va='top', color="black", zorder=200)
                
            
            # ===================================
            # Bottom plot
            # ===================================

            with sns.axes_style("whitegrid"):
                sen_file = f"{out_dir}/{config_name}_lat_sensitivity.csv"
                if not os.path.exists(sen_file):
                    print(f"[ERROR] Sensitivity file does not exist: {sen_file}")
                    raise FileNotFoundError
                df_sen = pd.read_csv(sen_file)

                # Converts the unit from ns to s
                df_sen["L"] = (df_sen["L"] - L_0) / 1000
                ax_inset = axs[i, j].inset_axes([0, -0.75, 1, 0.7], transform=axs[i, j].transAxes)
                ax_inset.step(df_sen["L"], df_sen["sensitivity"], where='post',
                              label="Sensitivity", marker=".", color="r")

                if i == N - 1:
                    ax_inset.set_xlabel("ΔL [μs]", fontsize=font_size)
                else:
                    ax_inset.set_xlabel("")
                
                if j == 0:
                    ax_inset.set_ylabel(r'$\lambda_L$', fontsize=font_size)
                
                ax_inset.yaxis.set_major_formatter(ticker.EngFormatter())
                axs[i, j].tick_params(labelbottom=False)

                if i == 0 and j == 0:
                    ax_inset.legend(loc="lower right", borderaxespad=0, fontsize=14).set_zorder(102)
                else:
                    axs[i, j].legend().set_visible(False)
                    ax_inset.legend().set_visible(False)
                
                # Moves the legend to the top left corner
                if i == 0 and j == 0:
                    ax_inset.legend(bbox_to_anchor=(0, 1), loc="upper left", borderaxespad=0, fontsize=14).set_zorder(102)
                
                ax_inset.tick_params(axis="x", labelsize=14)
                ax_inset.tick_params(axis="y", labelsize=14)

                # Reads the critical latency data
                crit_file = f"{out_dir}/{config_name}_lat_ratio.csv"

                if not os.path.exists(crit_file):
                    print(f"[ERROR] Critical latency file does not exist: {crit_file}")
                    raise FileNotFoundError
                
                df_crit = pd.read_csv(crit_file)
                df_crit["L"] = (df_crit["L"] - L_0) / 1000
                # Checks if the crit_lat is in percentage
                if df_crit["crit_lat"].max() < 1:
                    df_crit["crit_lat"] *= 100
                
                ax_inset.axvline(x=buffer1, color="g", linestyle="--", zorder=100, linewidth=lat_buf_width)
                ax_inset.axvline(x=buffer2, color="orange", linestyle="--", zorder=100, linewidth=lat_buf_width)
                ax_inset.axvline(x=buffer3, color="r", linestyle="--", zorder=100, linewidth=lat_buf_width)

                ax_inset.set_axisbelow(True)
                ax_inset2.set_axisbelow(True)
                ax_inset2.grid(False)
                ax_inset2.tick_params(axis='x', labelsize=14)
        
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams.update({'font.size': font_size})
    sns.set_theme()
    sns.set_style("whitegrid")
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(f"{data_dir}/validation.png", bbox_inches='tight')
    fig.savefig(f"{data_dir}/validation.pdf", bbox_inches='tight')



def plot_case_study_results(data_dir: str) -> None:
    """
    Plots the results from the case study experiments
    """

    HEIGHT = 2.5
    WIDTH = 6

    L_0, _, _ = get_net_params(1024, NET_PARAMS_FILE)
    assert os.path.exists(data_dir), f"[ERROR] Data directory does not exist: {data_dir}"
    # Counts the number of subdirectories in the case study allreduce/recdoub directory
    # which signifies the number of node configurations
    allreduce_dir = f"{data_dir}/allreduce"
    recdoub_dir = f"{allreduce_dir}/recdoub"
    ring_dir = f"{allreduce_dir}/ring"
    assert os.path.exists(allreduce_dir), f"[ERROR] Allreduce directory does not exist: {allreduce_dir}"

    confs = [
        int(name.upper().split('_')[1]) for name in os.listdir(recdoub_dir) \
            if os.path.isdir(os.path.join(recdoub_dir, name))
    ]
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting script')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory containing the data')
    parser.add_argument('-e', "--experiment", type=str, required=True,
                        help="Experiment to plot: speedup, validation, case-study")
    args = parser.parse_args()

    if args.experiment == "speedup":
        plot_speedup_results(args.data_dir)
    elif args.experiment == "validation":
        plot_validation_results(args.data_dir)
    elif args.experiment == "case-study":
        plot_case_study_results(args.data_dir)