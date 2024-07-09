import os
import networkx as nx
import argparse
from typing import Optional, List, Union, Tuple
from dep_graph_generator import DependencyGraphGenerator
from dep_graph import DependencyGraph
from lp_analyzer import LPAnalyzer
from lp_converter import LPConverter
from architecture_desc import TwoLevelHierarchy
from dep_graph import VertexType
from topology import NetTopology
from dep_graph_analyzer import DependencyGraphAnalyzer
import psutil
from time import time
from utils import *


SOLVER_TYPE = "GLOP"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MPI Dependency Graph Generator")
    parser.add_argument("-g", "--goal-file", dest="goal_file", default=None,
                        help="Path to the goal file that will be parsed.")
    parser.add_argument("-c", "--comm-dep-file", dest="comm_dep_file",
                        required=False,
                        help="Path to the communication dependency file that "
                        "records the dependencies between all the sends and recvs. "
                        "Will attempt to match the sends and recvs in the goal file "
                        "simply by message tags if this file is not given.")
    parser.add_argument("--mpich", dest="mpich", action="store_true",
                        default=False, help="If given, will formulate the "
                        "linear programming model that corresponds to the "
                        "MPICH implementation. Otherwise, will formulate the "
                        "linear programming model that corresponds to the "
                        "ideal scenario.")
    parser.add_argument("--viz-file", dest="viz_file",
                        required=False, default=None,
                        help="If given, will store the visualization of the "
                        "dependency graph in the given file.")
    parser.add_argument("--output-dir", dest="output_dir",
                        required=False, default=None,
                        help="If given, will store the results of the analysis"
                        "in the given directory.")
    parser.add_argument("-a", "--analysis", dest="analysis",
                        required=False, default=None, type=str,
                        help="If provided, it defines the type of analysis "
                        "to be performed. The options are ['sensitivity', 'buffer', 'placement']. "
                        "If 'sensitivity' is given, will perform network latency "
                        "sensitivity analysis. If 'buffer' is given, will perform "
                        "latency buffer analysis. If 'placement' is given, will "
                        "perform process placement analysis.")
    parser.add_argument("--lat-buf-thresh", dest="lat_buf_thresh",
                        required=False, default=0.01, type=float,
                        help="If given, will use the given latency buffer threshold "
                        "to perform latency buffer analysis. It has to be a positive "
                        "float representing the maximum percentage of the performance "
                        "degradation that can be tolerated. [DEFAULT: 0.01]")
    parser.add_argument("--lat-buf-baseline", dest="lat_buf_baseline",
                        required=False, default=None, type=int,
                        help="If given, will use the given latency buffer baseline "
                        "to perform latency buffer analysis. It has to be a positive "
                        "integer in ns representing the baseline runtime of the app. "
                        "If not given, will calculate the runtime during the analysis.")
    parser.add_argument("-S", dest="S", default=None, type=int,
                        help="Specify the value of S in the LogGPS model. "
                        "If None, will assume that LogGP model is used "
                        "instead of LogGPS model.")
    parser.add_argument("-o", dest="o", default=5000, type=int,
                        help="Specify the value of o in the LogGPS model. "
                        "If None, will assume that LogGP model is used "
                        "instead of LogGPS model. [DEFAULT: 5000]")
    parser.add_argument("--l-min", dest="l_min", default=3000, type=int,
                        help="Specify the minimum value of L in the sensitivity analysis. [DEFAULT: 3000]")
    parser.add_argument("--l-max", dest="l_max", default=103000, type=int,
                        help="Specify the maximum value of L in the sensitivity analysis. [DEFAULT: 103000]")
    parser.add_argument("--step", dest="step", default=1000, type=int,
                        help="Specify the step size of L in the sensitivity analysis. [DEFAULT: 1000]")
    parser.add_argument("--export-graph-path", dest="export_graph_path",
                        required=False, default=None,
                        help="If given, will export the generated dependency "
                        "graph to the given path.")
    parser.add_argument("--load-graph-path", dest="load_graph_path",
                        default=None,
                        help="If given, will load the dependency graph from "
                        "the given path instead of generating it from scratch."
                        "If this option is given, the goal file and the "
                        "communication dependency file will be ignored.")
    parser.add_argument("--solve", dest="solve", action="store_true",
                        default=False, help="If given, will solve the linear "
                        "programming model. In the given range of L.")
    parser.add_argument("--rm-file", dest="rm_file", type=str, default="rankmap.txt",
                        help="The file that contains the result of the process placement analysis."
                        "It can be used directly by the rankmap argument of the MPICH runner.")
    parser.add_argument("--load-lp-model-path", dest="load_lp_model_path",
                        default=None,
                        help="If given, will load the LP model from the given "
                        "path instead of generating it from scratch.")
    parser.add_argument("--export-lp-model-path", dest="export_lp_model_path",
                        default=None,
                        help="If given, will save the LP model to the given path.")
    parser.add_argument("-v", "--verbose", dest="verbose",
                        required=False, action="store_true", default=False,
                        help="If given, will print out more information.")
    parser.add_argument("--topology", dest="topology", choices=["default", "fat_tree", "dragonfly"],
                        required=False, default="default",
                        help="If given, will use the specified network topology to be used in the model.")
    parser.add_argument("-G", dest="G_val", required=False, default=0.018, type=float,
                        help="If given, will set the value of G in the LogGPS model "
                        "to be a constant when constructing the LP. [DEFAULT: 0.018]")
    

    args = parser.parse_args()
    goal_file = args.goal_file
    comm_dep_file = args.comm_dep_file
    model = None
    verbose = args.verbose
    proj_name = None

    # FIXME: VERY Messy code
    if args.load_lp_model_path is not None and (args.analysis or args.solve):
        proj_name = os.path.basename(args.load_lp_model_path).split(".")[0]
        if args.load_graph_path is not None:
            if not os.path.exists(args.load_graph_path):
                raise FileNotFoundError("[ERROR] Dependency graph file {} does not exist."
                                        .format(args.load_graph_path))
            print(f"[INFO] Loading dependency graph from {args.load_graph_path}", flush=True)
            dep_graph = DependencyGraph(0)
            dep_graph.load(args.load_graph_path)
            print(f"[INFO] Loaded dependency graph from {args.load_graph_path}. Skipping graph generation")

        # Checks if the LP model file exists
        if not os.path.exists(args.load_lp_model_path):
            raise FileNotFoundError("[ERROR] LP model file {} does not exist."
                                    .format(args.load_lp_model_path))
        # Loads the LP model
        model = load_model(args.load_lp_model_path, verbose)
        print(f"[INFO] Loaded LP model from {args.load_lp_model_path}. "
              "Skipping LP model generation.", flush=True)
        
    else:
        if args.load_graph_path is not None:
            if not os.path.exists(args.load_graph_path):
                raise FileNotFoundError("[ERROR] Dependency graph file {} does not exist."
                                        .format(args.load_graph_path))
            print(f"[INFO] Loading dependency graph from {args.load_graph_path}", flush=True)
            dep_graph = DependencyGraph(0)
            dep_graph.load(args.load_graph_path)
            print(f"[INFO] Loaded dependency graph from {args.load_graph_path}. Skipping graph generation")

        else:
            # Checks if the goal file exists
            if not os.path.exists(goal_file):
                raise FileNotFoundError("[ERROR] Goal file {} does not exist.".format(goal_file))
            # Checks if the communication dependency file exists
            if comm_dep_file is None:
                print("[INFO] No communication dependency file is given. "
                    "Will attempt to match sends and recvs only by message tags.")
            elif not os.path.exists(comm_dep_file):
                raise FileNotFoundError("[ERROR] Communication dependency file {} does not exist.".format(comm_dep_file))
            
            # Generates the dependency graph
            print("[INFO] Generating dependency graph...", flush=True)
            dep_graph_generator = DependencyGraphGenerator(goal_file, comm_dep_file)
            # cProfile.run("dep_graph = dep_graph_generator.generate()")
            is_loggps = args.S is not None
            start = time()
            dep_graph = dep_graph_generator.generate(is_loggps)
            print("[INFO] Generated dependency graph in {:.2f} seconds.".format(time() - start))

        print(f"[INFO] Graph statistics:")
        print(f"[INFO] Number of vertices: {dep_graph.num_vertices()}")
        print(f"[INFO] Number of edges: {dep_graph.num_edges()}")
        print(f"[INFO] Number of ranks: {dep_graph.num_ranks}", flush=True)

        # Stores the dependency graph
        if args.export_graph_path is not None:
            print(f"[INFO] Exporting dependency graph to {args.export_graph_path}...", flush=True)
            dep_graph.export(args.export_graph_path)
            print("[INFO] Exported dependency graph to {}.".format(args.export_graph_path), flush=True)
        
        # Visualizes the dependency graph
        if args.viz_file is not None:
            dep_graph.visualize(args.viz_file)
    print(f"[INFO] Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB", flush=True)
    # Iterates through  all vertices in the graph and prints their info
    # for v in dep_graph.graph.vs:
    #     print("==============================")
    #     dep_graph.print_vertex_info(v.index)
    placement_analysis = args.analysis == "placement"
    if model is None:
        # =================================================
        # Converts the dependency graph to a linear program
        # =================================================
        # topology = NetTopology.default_topology(dep_graph.num_ranks)
        converter = LPConverter(dep_graph, o=args.o, S=args.S)

        # =================================================
        # Network topology
        # =================================================
        if args.topology == "default":
            topology = None
        elif args.topology == "fat_tree":
            topology = NetTopology.fat_tree(dep_graph.num_ranks, 16)
        elif args.topology == "dragonfly":
            topology = NetTopology.dragonfly(8, 4, 8, dep_graph.num_ranks)
        else:
            raise ValueError(f"[ERROR] Unsupported topology: {args.topology}")
        
        model = converter.convert_to_lp(verbose,
                                        placement_analysis,
                                        topology=topology,
                                        is_mpich=args.mpich,
                                        G=args.G_val)
    if verbose:
        print(f"[INFO] Number of variables = {model.NumVars}")
        print(f"[INFO] Number of constraints = {model.NumConstrs}", flush=True)
    # Outputs memory usage of the process
    print(f"[INFO] Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB", flush=True)
    if args.export_lp_model_path is not None:

        if proj_name is None:
            proj_name = os.path.basename(args.export_lp_model_path).split(".")[0]
        
        print("[INFO] Exporting LP model...", flush=True)
        if is_gurobi_installed():
            model.write(args.export_lp_model_path)
            # Writes a MPS file as well
            model.write(args.export_lp_model_path.replace(".lp", ".mps"))
        else:
            # model.ExportModelAsMpsFormat(False, args.export_lp_model_path, False)
            model.ExportModelAsLpFormat(False, args.export_lp_model_path, False)
        print(f"[INFO] Exported LP model to {args.export_lp_model_path}.", flush=True)
    
    # Initializes the analyzer
    analyzer = LPAnalyzer()

    L_lb = args.l_min
    L_ub = args.l_max
    step = args.step
    # ========================================================
    # Solves the linear programming model
    # ========================================================
    if args.solve:
        analyzer.solve_lp(model, l_min=L_lb, l_max=L_ub,
                          step=step, verbose=verbose)

    # ========================================================
    # Performs analysis on the dependency graph
    # ========================================================
    if args.analysis:
        print("[INFO] Performing analysis...", flush=True)

        # FIXME (TODO): Messy code
        if args.analysis == "sensitivity":
            # ============================
            # Network latency sensitivity
            # ============================
            net_lat_sen = analyzer.get_net_lat_sensitivity(model,
                                                           L_ub=L_ub,
                                                           L_lb=L_lb,
                                                           step=step,
                                                           verbose=verbose)
            # # net_lat_sen.display_metric()
            net_lat_sen.visualize(L_ub=L_ub, L_lb=L_lb, log_scale=False)
            if proj_name is None:
                proj_name = "tmp"
            net_lat_sen.to_csv(proj_name, args.output_dir)

        elif args.analysis == "buffer":
            max_lat = analyzer.get_net_lat_buffer(model, args.lat_buf_thresh,
                                        args.lat_buf_baseline, verbose=verbose)
            print(f"[INFO] Maximum latency for {args.lat_buf_thresh * 100}% "
                  f"degradation: {max_lat} ns")

        elif placement_analysis:
            # ============================
            # Pairwise rank sensitivity
            # ============================
            # matrix = analyzer.get_pairwise_rank_sensitivity(model, verbose=verbose)
            # visualize_heatmap(matrix)

            # ============================
            # Process placement analysis
            # ============================
            arch = TwoLevelHierarchy(16, 650, 0.01, 2500, 0.02)
            placement = analyzer.placement_analysis(model, arch, verbose)
            write_placement_to_file(placement, args.rm_file)
            print(f"[INFO] Final placement: {placement}")
        else:
            raise ValueError(f"[ERROR] Invalid analysis type: {args.analysis}")
            
        print("[INFO] Analysis completed.", flush=True)

        exit(0)