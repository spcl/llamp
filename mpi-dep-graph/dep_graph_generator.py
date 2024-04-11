import os
from typing import Optional, List, Union, Tuple
from file_parser import *
from goal_elem import *
from dep_graph import DependencyGraph, VertexType


class DependencyGraphGenerator(object):
    """
    An object that generates the dependency graph from
    the parsed goal file and communication dependency file.
    """
    def __init__(self, goal_file: str,
                 comm_dep_file: Optional[str] = None) -> None:
        """
        Initialize the dependency graph generator.
        @param goal_file: The path to the goal file.
        @param comm_dep_file: The path to the communication dependency file.
        If it is None, then the dependency graph generator will attempt
        to match the sends and recvs in the goal file simply by message tags.
        """
        # Makes sure that the goal file exists
        if not os.path.exists(goal_file):
            directory = os.getcwd()
            raise FileNotFoundError(f"[ERROR] Goal file {goal_file} does not exist. {directory}")
        self.goal_file = goal_file
        self.comm_dep_file = comm_dep_file
        self.goal_parser = GoalFileParser()
        self.comm_dep_parser = CommDepFileParser()
    
    def generate(self, is_loggps: bool = False) -> DependencyGraph:
        """
        Generates the dependency graph from the goal file and
        the communication dependency file.
        @param is_loggps: Whether the dependency graph produced
        is for the LogGPS model. If it is True, additional edges will
        be added between the local dependency of a RECV vertex and
        its corresponding SEND vertex. Since this uses additional
        memory, it is suggested to set this to False if the dependency
        graph is not for the LogGPS model.
        @return: The generated dependency graph.
        """
        dep_graph = None
        curr_rank = None
        # Parses the goal file
        goal_file = open(self.goal_file, "r")
        model_name = "LogGPS" if is_loggps else "LogGP"
        print(f"[INFO] Generating dependency graph for {model_name} model...", flush=True)
        # Keeps track of all the sends and recvs operations
        # if no communication dependency file is given
        if self.comm_dep_file is None:
            # Sends and recvs are stored as a dictionary of
            # (rank, tag) -> global vertex index in the dependency graph
            sends = {}
            recvs = {}

        line_count = 0
        end_v = None
        start_v = None
        for line in goal_file:
            elem = self.goal_parser.parse_line(line)
            # Checks if the line is empty
            if elem is None:
                continue
            
            if isinstance(elem, GlobalRanks):
                # Initializes the dependency graph with the number of ranks
                # in the MPI program
                dep_graph = DependencyGraph(elem.num_ranks, is_loggps)
            
            elif isinstance(elem, RankStart):
                # Sets the current rank
                curr_rank = elem.rank
                start_v = None
            
            elif isinstance(elem, RankEnd):
                dep_graph.rank_to_end_v[curr_rank] = end_v
                curr_rank = None
                end_v = None
            
            elif isinstance(elem, SendOp):
                # Adds the send operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.SEND, curr_rank,
                                           elem.label, elem.data_size,
                                           elem.dst)
                
                if self.comm_dep_file is None:
                    key = (curr_rank, elem.dst, elem.tag)
                    # Checks if the send operation has a matching recv
                    if key in recvs:
                        dep_graph.add_edge_by_global_index(idx, recvs[key], True)
                        # Removes the recv from the dictionary
                        del recvs[key]
                    else:
                        sends[key] = idx

            elif isinstance(elem, RecvOp):
                # Adds the recv operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.RECV, curr_rank,
                                           elem.label, elem.data_size,
                                           elem.src)
                
                if self.comm_dep_file is None:
                    key = (elem.src, curr_rank, elem.tag)
                    # Checks if the recv operation has a matching send
                    if key in sends:
                        dep_graph.add_edge_by_global_index(sends[key], idx, True)
                        # Removes the send from the dictionary
                        del sends[key]
                    else:
                        recvs[key] = idx

            elif isinstance(elem, CalcOp):
                # Adds the calc operation to the dependency graph
                idx = dep_graph.add_vertex(VertexType.CALC, curr_rank,
                                     elem.label, elem.cost)
                end_v = idx
                if start_v is None and elem.cost > 0:
                    start_v = idx
                    dep_graph.rank_to_start_v[curr_rank] = start_v
            
            elif isinstance(elem, Dependency):
                # Adds the dependency to the dependency graph
                dep_graph.add_edge(curr_rank, elem.src_label,
                                   curr_rank, elem.dst_label,
                                   False, elem.is_irequire)

            else:
                raise ValueError(f"[ERROR] Invalid line: {line}")


            line_count += 1
            if line_count % 1000000 == 0:
                print(f"[INFO] Parsed {line_count} lines in the goal file.", flush=True)
        goal_file.close()

        if self.comm_dep_file is not None:
            # Parses the communication dependency file if given
            comm_dep_file = open(self.comm_dep_file, "r")
            for line in comm_dep_file:
                src, dst = self.comm_dep_parser.parse_line(line)
                # Adds the dependency to the dependency graph
                dep_graph.add_edge(src[0], src[1] + 1, dst[0], dst[1] + 1, True)
            comm_dep_file.close()
        else:
            # Checks if there are unmatched sends and recvs
            if len(sends) != 0 or len(recvs) != 0:
                for _, idx in sends.items():
                    print("[DEBUG] Unmatched send: l" + \
                          str(dep_graph.graph.vs[idx]["l"]) + \
                          " (rank " + str(dep_graph.graph.vs[idx]["r"]) + ")")
                for _, idx in recvs.items():
                    print("[DEBUG] Unmatched recv: l" + \
                          str(dep_graph.graph.vs[idx]["l"]) + \
                          " (rank " + str(dep_graph.graph.vs[idx]["r"]) + ")")
            assert len(sends) == 0 and len(recvs) == 0, \
                "[ERROR] There are unmatched sends and recvs in the goal file.\n" \
                "Communication dependency file is required to match them."
        
        dep_graph.finalize()
        return dep_graph
