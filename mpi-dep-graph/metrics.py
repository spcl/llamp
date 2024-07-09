import matplotlib
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set, Union
from utils import experimental


class NetLatSensitivity(object):
    """
    An object that represents the network latency sensitivity metric.
    """
    def __init__(self, critical_latencies: List[Tuple[float, int]],
                 runtime: List[Tuple[float, float]],
                 lat_ratios: List[Tuple[float, float]]) -> None:
        """
        Initializes the network latency sensitivity metric object.
        `critical_latencies` is a list of tuples, where each tuple
        represents a critical latency. The first element of the tuple
        is the network latency value, and the second element is the
        value of the metric at that latency value.
        @param critical_latencies: A list of critical latencies.
        @param runtime: A list of tuples, where each tuple represents
        the runtime of the MPI application at a given latency value.
        @param lat_ratios: A list of tuples, where each tuple represents
        the latency cost of the MPI application at a given latency value.
        Latency cost means the percentage of runtime that is due to
        the network latency.
        """
        self.critical_latencies = critical_latencies
        self.runtime = runtime
        self.lat_ratios = lat_ratios

    def to_csv(self, name: str, output_dir: str) -> None:
        """
        Saves the network latency sensitivity metric to a CSV file.
        """
        csv_file = f"{output_dir}/{name}_lat_sensitivity.csv"
        with open(csv_file, "w") as f:
            f.write("L,sensitivity\n")
            for L, val in self.critical_latencies:
                f.write(f"{L},{val}\n")
        print(f"[INFO] Saved the network latency sensitivity metric to {csv_file}")

        # Saves the predicted runtime to a separate CSV file
        runtime_file = f"{output_dir}/{name}_runtime.csv"
        with open(runtime_file, "w") as f:
            f.write("L,runtime\n")
            for L, val in self.runtime:
                f.write(f"{L},{val}\n")
            
        print(f"[INFO] Saved the predicted runtime to {runtime_file}")
        
        
        # Saves the latency cost to a separate CSV file
        lat_ratio_file = f"{output_dir}/{name}_lat_ratio.csv"
        with open(lat_ratio_file, "w") as f:
            f.write("L,crit_lat\n")
            for L, val in self.lat_ratios:
                f.write(f"{L},{val}\n")
        print(f"[INFO] Saved the latency ratio to {lat_ratio_file}")

    def visualize(self, L_ub: Optional[int] = None, 
                  L_lb: Optional[int] = None,
                  log_scale: bool = False) -> None:
        """
        Visualizes the network latency sensitivity metric
        as a plot with L on the x-axis and ∂T/∂L on the y-axis.
        On the second y-axis, the relative values are plotted.
        @param log_scale: If True, the x-axis will be in log scale.
        """
        _, ax1 = plt.subplots()
        # ax2= ax1.twinx()

        # First y-axis
        critical_latencies = self.critical_latencies
        if len(critical_latencies) == 1:
            y = critical_latencies[0][-1]
            # Plots a straight horizontal line
            ax1.axhline(y=y, color='r', linestyle='-')
            # Sets the y-axis limits to be between 0 and 1.2 * y
            ax1.set_ylim(0, 1.2 * y)
        else:
            # critical_latencies.insert(0, (0, 0))
            x_max, y_max = critical_latencies[-1]
            # Adds (x_max * 1.2, y_max) to the end of the list
            
            # critical_latencies.append((x_max * 1.8, y_max))
            # Plots the critical latencies as a step function
            x, y = zip(*critical_latencies)
            ax1.step(x, y, where='post')
            # Sets the y-axis limits to be between the lowest and highest
            ax1.set_ylim(critical_latencies[0][1] - 10, y_max + 10)
            # Adds labels to the critical latencies
            # for i, (x, y) in enumerate(critical_latencies[1:-1]):
            #     plt.text(x, y, f"{int(x)}", fontsize=10)
        
        # Second y-axis
        # x, y = zip(*self.rel_vals)
        # ax2.plot(x, y, color="green")

        if log_scale:
            ax1.set_xscale("log")
        ax1.set_xlabel("L [ns]")
        ax1.set_ylabel("∂T/∂L")

        # if log_scale:
        #     ax2.set_xscale("log")
        # ax2.set_ylabel("(∂T/∂L) / T", color="green")
        # ax2.tick_params(axis='y', labelcolor="green")
        # if log_scale:
        #     formatter = matplotlib.ticker.LogFormatter(labelOnlyBase=False,
        #                                             minor_thresholds=(2, 0.4))
        #     ax1.xaxis.set_minor_formatter(formatter)
            # ax2.xaxis.set_minor_formatter(formatter)
        
        plt.savefig("netlat_sensitivity.png")
        print("[INFO] Saved the network latency sensitivity plot to "
              "netlat_sensitivity.png")
        plt.close()

        # Plots the runtime as a function of L
        _, ax = plt.subplots()
        x, y = zip(*self.runtime)
        ax.plot(x, y)
        ax.set_xlabel("L [ns]")
        ax.set_ylabel("T [ns]")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        plt.savefig("runtime.png")


@experimental
class CriticalCompute(object):
    """
    An object that represents a list of critical compute vertices along
    the critical path of the dependency graph. Note that in order for
    a compute vertex to be considered critical, the constraint associated
    must satisfy certain conditions.
    """
    def __init__(self) -> None:
        """
        Initializes the critical compute object.
        """
        # A list of tuples, where each tuple represents a critical
        self.vs = []


@experimental
class CommCostMetrics(object):
    """
    An object that encapsulates all the metrics related to wait states.
    See paper https://dl.acm.org/doi/10.1145/2934661
    """
    def __init__(self, num_ranks: int) -> None:
        self.num_ranks = num_ranks
        # The amount of time each rank spends in wait state
        self.wait_time = [0.0 for _ in range(num_ranks)]

        # The amount of time each rank spends in communication
        self.comm_time = [0.0 for _ in range(num_ranks)]

        # The amount of time each rank's wait time overlaps with
        # its compute time
        self.wait_overlap_time = [0.0 for _ in range(num_ranks)]
        # The amount of time each rank's communication time overlaps with
        # its compute time
        self.comm_overlap_time = [0.0 for _ in range(num_ranks)]

        self.total_time = None
        # A list of intervals of time during which each rank is in wait state.
        # This is used to deal with parallel sends and receives
        # so that intervals that overlap with each other are
        # merged together. This greatly reduces the overestimation
        # of wait time for each rank.
        self.wait_intervals = [None for _ in range(num_ranks)]

        self.compute_intervals = [None for _ in range(num_ranks)]

        self.comm_intervals = [None for _ in range(num_ranks)]

    def add_comm_interval(self, rank: int, interval: Tuple[float, float]) \
        -> None:
        """
        Adds the given interval to the given rank. Checks the previous
        item in the list and merges the two intervals if they overlap.
        FIXME This function is very similar to `add_wait_interval`.
        """
        if not self.comm_intervals[rank]:
            # Adds the interval to the list
            self.comm_intervals[rank] = interval
            return

        prev_interval = self.comm_intervals[rank]
        # Checks if the two intervals overlap
        if prev_interval[1] >= interval[0]:
            # Merges the two intervals
            # Checks the upper bound of the two intervals
            if prev_interval[1] < interval[1]:
                new_interval = (prev_interval[0], interval[1])
            else:
                new_interval = prev_interval
            self.comm_intervals[rank] = new_interval
        else:
            # If the two do not overlap, removes the previous interval
            # and adds wait time represented by the interval to the
            # wait time of the rank
            self.comm_intervals[rank] = interval
            self.add_comm_time(rank, prev_interval[1] - prev_interval[0])
            # Before deallocating the previous interval, checks if
            # it overlaps with any of the wait intervals
            overlap_time = \
                self.__interval_overlap(prev_interval, self.compute_intervals[rank])
            if overlap_time > 0:
                self.add_overlap(rank, overlap_time, False)

    def add_compute_interval(self, rank: int, interval: Tuple[float, float]) \
        -> None:
        """
        Adds the given interval to the given rank. Checks the previous
        item in the list and merges the two intervals if they overlap.
        FIXME This function is very similar to `add_wait_interval`.
        """
        # print(f"[DEBUG] Adding compute interval {interval} to rank {rank}")
        if not self.compute_intervals[rank]:
            # Adds the interval to the list
            self.compute_intervals[rank] = interval
            return

        prev_interval = self.compute_intervals[rank]
        # Checks if the two intervals overlap
        if prev_interval[1] >= interval[0]:
            # Merges the two intervals
            # Checks the upper bound of the two intervals
            if prev_interval[1] < interval[1]:
                new_interval = (prev_interval[0], interval[1])
            else:
                new_interval = prev_interval
            self.compute_intervals[rank] = new_interval
        else:
            # If the two do not overlap, removes the previous interval
            # and adds wait time represented by the interval to the
            # wait time of the rank
            self.compute_intervals[rank] = interval
            # Before deallocating the previous interval, checks if
            # it overlaps with any of the wait and communication intervals
            wait_overlap_time = \
                self.__interval_overlap(prev_interval, self.wait_intervals[rank])
            if wait_overlap_time > 0:
                self.add_overlap(rank, wait_overlap_time, True)
            
            comm_overlap_time = \
                self.__interval_overlap(prev_interval, self.comm_intervals[rank])
            if comm_overlap_time > 0:
                self.add_overlap(rank, comm_overlap_time, False)

    def add_wait_interval(self, rank: int, interval: Tuple[float, float]) \
        -> None:
        """
        Adds the given interval to the given rank. Checks the previous
        item in the list and merges the two intervals if they overlap.
        """
        if not self.wait_intervals[rank]:
            # Adds the interval to the list
            self.wait_intervals[rank] = interval
            return
        
        prev_interval = self.wait_intervals[rank]
        # Checks if the two intervals overlap
        if prev_interval[1] >= interval[0]:
            # Merges the two intervals
            # Checks the upper bound of the two intervals
            if prev_interval[1] < interval[1]:
                new_interval = (prev_interval[0], interval[1])
            else:
                new_interval = prev_interval
            self.wait_intervals[rank] = new_interval
        else:
            # If the two do not overlap, removes the previous interval
            # and adds wait time represented by the interval to the
            # wait time of the rank
            self.add_wait_time(rank, prev_interval[1] - prev_interval[0])
            self.wait_intervals[rank] = interval

            # Before deallocating the previous interval, checks if
            # it overlaps with any of the compute intervals
            overlap_time = \
                self.__interval_overlap(prev_interval, self.compute_intervals[rank])
            if overlap_time > 0:
                self.add_overlap(rank, overlap_time, True)
        # print(f"[DEBUG] Current wait intervals: {self.wait_intervals}")

    def add_total_time(self, total_time: float) -> None:
        """
        Adds the total predicted run time to the metric. This is
        the value of the objective function after optimization.
        """
        self.total_time = total_time

    def add_wait_time(self, rank: int, wait_time: float) -> None:
        """
        Adds the given wait time to the given rank.
        """
        self.wait_time[rank] += wait_time
    
    def add_comm_time(self, rank: int, comm_time: float) -> None:
        """
        Adds the given communication time to the given rank.
        """
        self.comm_time[rank] += comm_time
    
    def __interval_overlap(self, interval1: Optional[Tuple[float, float]],
                        interval2: Optional[Tuple[float, float]]) -> float:
        """
        A helper function that calculates the overlap between two intervals.
        """
        if interval1 is None or interval2 is None:
            return 0
        s1, e1 = interval1
        s2, e2 = interval2
        # Checks if the intervals overlap
        # Visualization:
        # |---|  |---|
        # s1  e1 s2  e2
        # or
        # |---|  |---|
        # s2  e2 s1  e1
        if s1 > e2 or s2 > e1:
            # If they do not overlap, returns 0
            return 0
        
        # Calculates the overlap
        # Visualization:
        # |-------|
        # s1      e1
        #    |-------|
        #    s2      e2
        # or
        # |-------|
        # s2      e2
        #    |-------|
        #    s1      e1
        # e = min(e1, e2)
        e = e1 if e1 < e2 else e2
        # s = max(s1, s2)
        s = s1 if s1 > s2 else s2
        return e - s
    
    def add_overlap(self, rank: int, overlap_time: float,
                    is_wait_time: bool) -> None:
        """
        Adds the given overlap time to the given rank. If `is_wait_time`
        is True, the overlap time is added to `wait_overlap_time`.
        Otherwise, it is added to `comm_overlap_time`.
        """
        # print(f"[DEBUG] Adding overlap time {overlap_time} to rank {rank}")
        if is_wait_time:
            self.wait_overlap_time[rank] += overlap_time
        else:
            self.comm_overlap_time[rank] += overlap_time

    def finalize(self) -> None:
        """
        Finalizes the metric by adding the remaining wait time from
        `wait_intervals` to `wait_time` for each rank.
        """
        for rank in range(self.num_ranks):
            wait_interval = self.wait_intervals[rank]
            comm_interval = self.comm_intervals[rank]
            compute_interval = self.compute_intervals[rank]
            
            # FIXME Kind of ugly NGL
            if wait_interval:
                self.add_wait_time(rank, wait_interval[1] - wait_interval[0])
                if compute_interval:
                    wait_overlap_time = \
                        self.__interval_overlap(wait_interval, compute_interval)
                    if wait_overlap_time > 0:
                        self.add_overlap(rank, wait_overlap_time, True)
                    # Resets the compute interval
                    self.compute_intervals[rank] = None
                # Resets the wait interval
                self.wait_intervals[rank] = None
            
            if comm_interval:
                self.add_comm_time(rank, comm_interval[1] - comm_interval[0])
                if compute_interval:
                    comm_overlap_time = \
                        self.__interval_overlap(comm_interval, compute_interval)
                    if comm_overlap_time > 0:
                        self.add_overlap(rank, comm_overlap_time, False)
                    # Resets the compute interval
                    self.compute_intervals[rank] = None
                
        for rank in range(self.num_ranks):
            assert self.wait_overlap_time[rank] <= self.wait_time[rank] and \
                self.comm_overlap_time[rank] <= self.comm_time[rank]
    

    def save_csv(self, path: str = "comm_cost.csv") -> None:
        """
        Saves the communication cost metrics to a CSV file.
        """
        assert self.total_time is not None

        with open(path, "w") as f:
            f.write("rank,wait_time,wait_overlap_time,comm_time,comm_overlap_time,total\n")
            for rank in range(self.num_ranks):
                f.write(f"{rank},{self.wait_time[rank]},{self.wait_overlap_time[rank]},"
                        f"{self.comm_time[rank]},{self.comm_overlap_time[rank]},{self.total_time}\n")
        
        print(f"[INFO] Saved the communication cost metrics to {path}")

    def visualize_comm_cost(self, type: str = "bar",
                            log_scale: bool = False) -> None:
        """
        Visualizes the communication cost of each rank as per
        the given type. The default type is bar chart.
        @param type: The type of visualization.
        @param log_scale: If True, the y-axis will be in log scale.
        """
        if type == "bar":
            # Plots the wait time and communication time in two separate
            # bar charts
            # Creates a figure with two subplots
            _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            
            # Converts everything from ns to s
            total_time = self.total_time / 1e9
            wait_time = [x / 1e9 for x in self.wait_time]
            wait_overlap_time = [x / 1e9 for x in self.wait_overlap_time]
            comm_time = [x / 1e9 for x in self.comm_time]
            comm_overlap_time = [x / 1e9 for x in self.comm_overlap_time]
            # Calculates the net wait time and net communication time
            net_wait_time = [x - y for x, y in zip(wait_time, wait_overlap_time)]
            net_comm_time = [x - y for x, y in zip(comm_time, comm_overlap_time)]

            # Wait time analysis plot
            # Plots x-axis as rank and y-axis as wait time
            x_labels = [str(x) for x in range(self.num_ranks)]
            ax1.bar(x_labels, wait_time, label="Wait time", zorder=2)
            ax1.bar(x_labels, [total_time for _ in range(self.num_ranks)],
                   label="Total runtime", zorder=1)
            ax1.bar(x_labels, wait_overlap_time, label="Wait overlap time", zorder=3)
            # Labels the bars for net wait time at the top of the bars
            for i, v in enumerate(wait_time):
                label = net_wait_time[i]
                ax1.text(i, v, f"{label:.2f}", color='black', va='bottom', ha='center')
            
            # Sets the title for the plot
            ax1.set_title("Wait Time Analysis")
            ax1.set_ylabel("Time [s]")
            # Moves the legend to outside of the plot
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            if log_scale:
                ax1.set_yscale("log")

            # Communication cost analysis plot
            # Plots x-axis as rank and y-axis as communication time
            ax2.bar(x_labels, comm_time, label="Communication time", zorder=2)
            ax2.bar(x_labels, [total_time for _ in range(self.num_ranks)],
                     label="Total runtime", zorder=1)
            ax2.bar(x_labels, comm_overlap_time, label="Comm overlap time", zorder=3)
            ax2.set_xlabel("Rank")
            ax2.set_ylabel("Time [s]")
            # Labels the bars for net communication time at the top of the bars
            for i, v in enumerate(comm_time):
                label = net_comm_time[i]
                ax2.text(i, v, f"{label:.2f}", color='black', va='bottom', ha='center')
            # Moves the legend to outside of the plot
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            if log_scale:
                ax2.set_yscale("log")
            ax2.set_title("Communication Cost Analysis")

            plt.savefig("comm_cost.png", bbox_inches='tight')
            print("[INFO] Saved the plot to comm_cost.png")
            plt.close()
        else:
            raise ValueError(f"[ERROR] Invalid type: {type}")