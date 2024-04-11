from __future__ import annotations
from enum import Enum, auto
import igraph
from typing import Optional, List, Union, Tuple, Dict
from collections import defaultdict
import numpy as np
import networkx

class VertexType(Enum):
    SEND = 0
    RECV = 1
    CALC = 2


class DependencyGraph(object):
    """
    A class representation of a dependency graph of
    all the MPI operations. This should be generated
    from goal files and their corresponding communication
    dependency files. This class can also be treated
    as a wrapper around the igraph.Graph object.
    """
    def __init__(self, num_ranks: int, is_loggps: bool = False) -> None:
        """
        Initialize the dependency graph.
        @param num_ranks: The number of ranks in the MPI program
        @param is_loggps: Whether the dependency graph produced
        is for the LogGPS model. If it is True, additional edges will
        be added between the local dependency of a RECV vertex and
        its corresponding SEND vertex.
        @param is_ucx: Whether the dependency graph produced is for the
        UCX model. If it is True, an additional attribute "w" will be
        added to some vertices so that we know if the vertex 
        contains a wire up operation. If so, additional cost will be
        incurred due to the WIREUP ACK message.
        """
        self.graph = igraph.Graph(directed=True)
        self.num_ranks = num_ranks

        # A list of dictionaries that maps the local index of a vertex
        # to its global index in the dependency graph.
        # The list is indexed by the rank of the vertex.
        self.local_index_to_global_index: List[Dict[int, int]] = \
            [{} for _ in range(num_ranks)]
        # Pending edges will be added to the graph in bulk
        # once finalize() is called
        self.pending_edges = []
        self.preds = defaultdict(list)

        # The start and end vertices of the dependency graph
        # These are dummy vertices that are added to the graph
        # Will be added when finalize() is called
        self.start_v = None
        self.end_v = None

        # The global indices of the vertices that belong to the
        # marks the first vertex of each rank
        self.rank_to_start_v = [None for _ in range(num_ranks)]
        # The global indices of the vertices that belong to the
        # marks the last vertex of each rank
        self.rank_to_end_v = [None for _ in range(num_ranks)]

        self.is_loggps = is_loggps

        # [DEPRECATED]: Whether the dependency graph produced is for the
        # UCX model. If it is True, an additional attribute "w" will be
        # added to some vertices so that we know if the vertex
        self.is_ucx = False
        # A two dimensional array that stores whether a wire up
        # operation has been performed between two ranks.
        # Wire up only happens once between two ranks once
        # during a single run of the MPI program.
        self.wired_up = np.zeros((num_ranks, num_ranks), dtype=bool)

    def add_pending_edges(self) -> None:
        """
        Adds all the pending edges to the graph.
        """
        if self.pending_edges:
            # Unzips the pending edges list
            edges, is_irequires = list(zip(*self.pending_edges))
            self.graph.add_edges(edges, attributes={"i": is_irequires})
            self.pending_edges.clear()
            self.preds.clear()

    def add_virtual_edges_for_loggps(self) -> None:
        """
        Adds virtual edges for the LogGPS model.
        This is done by traversing the current list of vertices
        and adds additional edges between the local dependency
        of a RECV vertex and its corresponding SEND vertex.
        """
        virtual_edges = []
        # Traverses the current list of vertices
        # and adds additional edges between the local dependency
        # of a RECV vertex and its corresponding SEND vertex
        for v in self.graph.vs:
            if v["type"] == VertexType.RECV:
                # Finds the global index of the corresponding SEND vertex
                # in the same rank
                assert "src_idx" in v.attributes() \
                    and "loc_idx" in v.attributes()
                # Adds an edge from the local dependency of the RECV vertex
                # to the corresponding SEND vertex
                virtual_edges.append((v["loc_idx"], v["src_idx"]))
        # Adds an attribute 'v' to the all the virtual edges and set
        # it to True. This is done to distinguish the virtual edges
        # from the real edges
        self.graph.add_edges(virtual_edges, attributes={"v": True})


    def finalize(self) -> None:
        """
        Finalizes the dependency graph by adding all the pending edges
        to the graph. This is done to improve the performance of igraph.
        https://stackoverflow.com/questions/13974279/igraph-why-is-add-edge-function-so-slow-ompared-to-add-edges

        In addition, this method also adds a dummy parent vertex to all
        the starting vertices and a dummy child vertex to all the ending
        vertices. This is done to make sure that all the starting vertices
        have at least one predecessor, and all the ending vertices have
        at least one successor.

        @param is_loggps: Whether the dependency graph produced
        is for the LogGPS model. If it is True, additional edges will
        be added between the local dependency of a RECV vertex and
        its corresponding SEND vertex. Since this uses additional
        memory, it is suggested to set this to False if the dependency
        graph is not for the LogGPS model.
        """
        self.add_pending_edges()
        # Adds a dummy parent vertex to all the starting vertices and
        # a dummy child vertex to all the ending vertices
        start_vertices = self.get_starting_vertices()
        
        assert len(start_vertices) == self.num_ranks, \
            f"[ERROR] The number of starting vertices ({len(start_vertices)}) " \
            f"must be equal to the number of ranks ({self.num_ranks})."

        # Adds a dummy parent vertex
        dummy_start = self.add_vertex(VertexType.CALC, -1, -1, cost=0)
        dummy_end = self.add_vertex(VertexType.CALC, -1, -1, cost=0)
        for v in self.rank_to_start_v:
            self.add_edge_by_global_index(dummy_start, v)
        for v in self.rank_to_end_v:
            self.add_edge_by_global_index(v, dummy_end)
        self.add_pending_edges()

        if self.is_loggps:
            self.add_virtual_edges_for_loggps()

        self.start_v = dummy_start
        self.end_v = dummy_end

        # Clear the mapping from local index to global index
        # since it is no longer needed
        self.local_index_to_global_index.clear()

        # Adds a list of variables to the property of the graph
        self.graph["start_v"] = self.start_v
        self.graph["end_v"] = self.end_v
        self.graph["rank_start"] = self.rank_to_start_v
        self.graph["rank_end"] = self.rank_to_end_v
        self.graph["is_loggps"] = self.is_loggps


    def num_vertices(self) -> int:
        """
        Returns the number of vertices in the dependency graph.
        """
        v_count = self.graph.vcount()
        if self.start_v is not None:
            v_count -= 1
        if self.end_v is not None:
            v_count -= 1
        return v_count
    
    def num_edges(self) -> int:
        """
        Returns the number of edges in the dependency graph.
        """
        e_count = self.graph.ecount()
        if self.start_v is not None:
            e_count -= self.graph.vs[self.start_v].outdegree()
        if self.end_v is not None:
            e_count -= self.graph.vs[self.end_v].indegree()
        return e_count


    def bfs(self, return_iter: bool = True) -> Union[List[int], igraph.BFSIter]:
        """
        Performs a breadth-first search on the dependency graph.
        It is a wrapper around the igraph.Graph.bfs() method.
        @param return_iter: Whether to return an iterator or a list.
        """
        if return_iter:
            return self.graph.bfsiter(self.start_v, mode="out")
        
        return self.graph.bfs(self.start_v, mode="out")[0]
            


    def add_vertex(self, type: VertexType, rank: int,
                   local_index: int,
                   cost: Optional[int] = None,
                   other_rank: Optional[int] = None) -> int:
        """
        Add a single vertex to the dependency graph.
        @param type: The type of the vertex, can be SEND, RECV or CALC.
        @param rank: The rank on which the local operation was performed.
        @param local_index: The index of the vertex in its local rank,
        which is the same as the label of the operation.
        @param cost: If the vertex type is Calc, then this is the cost of
        computation of the vertex. Otherwise, this is the number of bytes
        sent or received by the vertex.
        @param other_rank: If the vertex type is Send or Recv, then this
        is the rank of the other vertex that the vertex is communicating with.
        @return: The index of the vertex.
        """
        # Adds the vertex to the graph and returns its global index
        attrs = {
            "type": type,
            "r": rank,
            "l": local_index,
            "cost": cost,
        }
        if other_rank is not None:
            if type == VertexType.SEND:
                attrs["dst_r"] = other_rank
            elif type == VertexType.RECV:
                attrs["src_r"] = other_rank
        idx = self.graph.add_vertex(**attrs).index
        # Adds the global index to the mapping
        self.local_index_to_global_index[rank][local_index] = idx
        return idx

    def get_topological_sort(self, mode: str = "out") -> List[int]:
        """
        Returns a topological sort of the dependency graph as
        a list of global indices of the vertices. It is a
        wrapper around the igraph.Graph.topological_sorting().
        @param mode: The mode of the topological sorting.
        Can be "out" or "in". Specifies how to use the direction of
        the edges. For “out”, the sorting order ensures that each
        node comes before all nodes to which it has edges, so nodes
        with no incoming edges go first. For “in”, it is quite the
        opposite: each node comes before all nodes from which it
        receives edges. Nodes with no outgoing edges go first.
        """
        return self.graph.topological_sorting(mode=mode)


    def add_edge(self, src_rank: int, src_label: int,
                 dst_rank: int, dst_label: int,
                 is_comm: bool = False,
                 is_irequire: bool = False,
                 immediate: bool = False) -> None:
        """
        Add a single edge to the dependency graph.
        @param src_rank: The rank of the source vertex.
        @param src_label: The label of the source vertex.
        @param dst_rank: The rank of the destination vertex.
        @param dst_label: The label of the destination vertex.
        @param is_comm: Whether the edge is a communication edge.
        @param is_irequire: Whether the edge is an irequire edge.
        @param immediate: If True, will add the edge immediately to
        the dependency graph. This is due to the fact the igraph library
        has a poor performance when adding edges one-by-one to the graph,
        as it needs to re-index the edges every time an edge is added.
        """
        # Make sure the src and dst are valid
        assert src_rank < self.num_ranks
        assert dst_rank < self.num_ranks
        assert src_label >= 0
        assert dst_label >= 0
        
        # src = self.graph.vs.find(rank=src_rank, local_index=src_label).index
        # dst = self.graph.vs.find(rank=dst_rank, local_index=dst_label).index
        src = self.local_index_to_global_index[src_rank][src_label]
        dst = self.local_index_to_global_index[dst_rank][dst_label]

        # Makes sure the src and dst are of the correct type
        # given that the src is a send operation
        if is_comm:
            assert self.graph.vs[src]["type"] == VertexType.SEND and \
                self.graph.vs[dst]["type"] == VertexType.RECV
            # Adds another attribute named "src_idx" to the dst vertex
            # that stores the global index of the src vertex
            self.graph.vs[dst]["src_idx"] = src
            if self.is_ucx and not self.wired_up[src_rank, dst_rank]:
                # Adds an additional attribute "w" to the dst vertex
                # to indicate that it contains a wire up operation
                self.graph.vs[src]["w"] = True
                self.graph.vs[dst]["w"] = True
                self.wired_up[src_rank, dst_rank] = True

        else:
            if self.graph.vs[dst]["type"] == VertexType.RECV:
                assert self.graph.vs[src]["type"] == VertexType.CALC
                # Stores the global index of the src vertex
                # in the dst vertex as its local computation dependency
                self.graph.vs[dst]["loc_idx"] = src
        
        # Checks if is_irequire is True
        # If so, then the dst vertex must be a calc operation
        if is_irequire:
            assert self.graph.vs[dst]["type"] == VertexType.CALC
            assert src_rank == dst_rank
            # Adds an attribute "i_idx" to the dst vertex
            # that stores the global index of the src vertex
            self.graph.vs[dst]["i_idx"] = src
            # # Sets the src of dst to the predecessor of src
            # # Makes sure that the new src belongs to the same
            # # rank as the original src
            # # FIXME: This is kind of ugly
            preds = self.preds[src]
            src = None
            for pred in preds:
                if self.graph.vs[pred]["r"] == src_rank:
                    src = pred
                    break
            assert src is not None, \
                f"Could not find a predecessor of {dst} in rank {src_rank}"

        self.preds[dst].append(src)

        if immediate:
            self.graph.add_edge(src, dst, is_irequire=is_irequire)
        else:
            self.pending_edges.append(((src, dst), is_irequire))
    
    def get_paths(self, start: int, end: int) -> List[List[int]]:
        """
        Returns all the paths from the start vertex to the end vertex.
        @param start: The global index of the start vertex.
        @param end: The global index of the end vertex.
        @return: A list of paths, where each path is a list of global
        indices of the vertices.
        """
        return self.graph.get_all_simple_paths(start, end)

    def get_starting_vertices(self) -> List[int]:
        """
        Returns a list of global indices of the vertices that
        do not have any predecessors.
        """
        return self.graph.vs.select(_indegree_eq=0).indices
    
    def get_edge(self, src: int, dst: int) -> Optional[igraph.Edge]:
        """
        Returns the edge object given the source vertex
        to the destination vertex. If the edge does not exist,
        returns None.
        """
        eid = self.graph.get_eid(src, dst, directed=True, error=False)
        if eid == -1:
            return None
        return self.graph.es[eid]

    def get_end_vertices(self) -> List[int]:
        """
        Returns a list of global indices of the vertices that
        do not have any successors.
        """
        return self.graph.vs.select(_outdegree_eq=0, type_eq=VertexType.CALC).indices

    def get_successors(self, src: int) -> List[int]:
        """
        Returns the global indices of the vertices that are
        the targets of the given vertex in a list. It is a
        wrapper around the igraph.Graph.successors() method.
        """
        return self.graph.successors(src)
    
    def get_predecessors(self, dst: int) -> List[int]:
        """
        Returns the global indices of the vertices that are
        the sources of the given vertex in a list. It is a
        wrapper around the igraph.Graph.predecessors() method.
        """
        return self.graph.predecessors(dst)
    
    @DeprecationWarning
    def get_is_irequire(self, src: int, dst: int) -> bool:
        """
        Returns whether the edge from the source vertex to the
        destination vertex is an irequire edge.
        """
        return self.graph.es[self.graph.get_eid(src, dst)]["is_irequire"]

    def add_edge_by_global_index(self, src: int, dst: int,
                                 is_comm: bool = False,
                                 is_irequires: bool = False,
                                 immediate: bool = False) -> bool:
        """
        Adds a single edge to the dependency graph by the global indices
        of the source and destination vertices.
        @param src: The global index of the source vertex.
        @param dst: The global index of the destination vertex.
        @param is_comm: Whether the edge is a communication edge.
        @param is_irequire: Whether the edge is an irequire edge.
        @param immediate: If True, will add the edge immediately to
        the dependency graph. This is due to the fact the igraph library
        has a poor performance when adding edges one-by-one to the graph,
        as it needs to re-index the edges every time an edge is added.
        """
        self.preds[dst].append(src)
        # FIXME: Redundant code
        if is_comm:
            assert self.graph.vs[src]["type"] == VertexType.SEND and \
                self.graph.vs[dst]["type"] == VertexType.RECV
            self.graph.vs[dst]["src_idx"] = src
            if self.is_ucx:
                src_rank = self.graph.vs[src]["r"]
                dst_rank = self.graph.vs[dst]["r"]
                if not self.wired_up[src_rank, dst_rank]:
                    # Adds an additional attribute "w" to the dst vertex
                    # to indicate that it contains a wire up operation
                    self.graph.vs[src]["w"] = True
                    self.graph.vs[dst]["w"] = True
                    self.wired_up[src_rank, dst_rank] = True
        else:
            # Stores the global index of the src vertex
            # in the dst vertex as its local computation dependency
            if self.graph.vs[dst]["type"] == VertexType.RECV:
                assert self.graph.vs[src]["type"] == VertexType.CALC
                self.graph.vs[dst]["loc_idx"] = src
        
        if immediate:
            self.graph.add_edge(src, dst, is_irequires=is_irequires)
        else:
            self.pending_edges.append(((src, dst), is_irequires))

    def print_vertex_info(self, v: int) -> None:
        """
        Prints the information of the vertex specified by its given
        global index.
        """
        print(f"Vertex {v}:")
        print(f"Type: {self.graph.vs[v]['type']}")
        print(f"Rank: {self.graph.vs[v]['r']}")
        print(f"Local index: {self.graph.vs[v]['l']}")
        print(f"Successors: {self.graph.successors(v)}")
        print(f"Predecessors: {self.graph.predecessors(v)}")
        if self.graph.vs[v]["type"] == VertexType.SEND:
            print(f"Destination rank: {self.graph.vs[v]['dst_r']}")
            print(f"Send size: {self.graph.vs[v]['cost']}")
        elif self.graph.vs[v]["type"] == VertexType.RECV:
            print(f"Source vertex: {self.graph.vs[v]['src_idx']}")
            print(f"Source rank: {self.graph.vs[v]['src_r']}")
            print(f"Recv size: {self.graph.vs[v]['cost']}")
        elif self.graph.vs[v]["type"] == VertexType.CALC:
            print(f"Cost: {self.graph.vs[v]['cost']}")
        else:
            raise ValueError(f"[ERROR] Invalid vertex type: {self.graph.vs[v]['type']}")

    def export(self, out_path: str) -> None:
        """
        Exports the dependency graph to the given path.
        @param out_path: The path to the output file.
        """
        self.graph.write_picklez(out_path)

    def load(self, in_path: str) -> None:
        """
        Loads the dependency graph from the given path, and automatically
        derive the number of ranks from the dependency graph and
        the starting and ending vertices.
        """
        self.graph = igraph.Graph.Read_Picklez(in_path)
        # start_vs = self.get_starting_vertices()
        start_v = self.graph["start_v"]
        assert start_v is not None, \
            f"[ERROR] Dependency graph must have exactly one starting vertex."
        end_v = self.graph["end_v"]
        assert end_v is not None, \
            f"[ERROR] Dependency graph must have exactly one ending vertex."

        # Restores the attributes of the graph
        self.start_v = start_v
        self.end_v = end_v
        self.num_ranks = len(self.get_successors(self.start_v))
        self.rank_to_start_v = self.graph["rank_start"]
        self.rank_to_end_v = self.graph["rank_end"]
        self.is_loggps = self.graph["is_loggps"]

    def visualize(self, out_path: str, rank: Optional[int] = None) -> None:
        """
        Visualize the dependency graph. The colors of the 
        vertices will depend on their types. irequire edges will be red.
        In addition, vertices will be clustered based on their ranks.
        @param out_path: The path to the output file.
        @param rank: If given, will only visualize the vertices
        that belong to the given rank.
        """
        # Visualize the dependency graph
        visual_style = {}
        visual_style["vertex_size"] = 20
        # Adjusts the plot size
        # So that its height is larger than its width
        # visual_style["bbox"] = (2000, 2000)
        # Adds label to each vertex
        visual_style["vertex_label"] = \
            [f"l{v['l']}, {v.index}" for v in self.graph.vs]

        # Edges whose irequires attribute is True will have a different color
        visual_style["edge_color"] = \
            ["red" if e["i"] else "black" for e in self.graph.es]

        # Removes all the vertices not belonging to the given rank
        if rank is not None:
            self.graph.delete_vertices(self.graph.vs.select(rank_ne=rank))
        # visual_style["vertex_label"] = [v["local_index"] for v in self.graph.vs]
        # Obtains a list of unique colors for each rank
        rank_colors = \
            igraph.drawing.colors.ClusterColoringPalette(self.num_ranks)
        vertex_colors = ["lightgreen", "lightgreen", "salmon3"]
        # Assigns a color to each vertex based on its type
        visual_style["vertex_color"] = \
            [vertex_colors[v["type"].value] for v in self.graph.vs]
        visual_style["vertex_label_size"] = 10
        visual_style["vertex_label_dist"] = 1.5
        # irequire edges will have less width than normal edges
        # visual_style["edge_width"] = \
        #     [1 if e["is_irequire"] else 2 for e in self.graph.es]
        # Finds the indices of the vertices that belong to the same rank
        clusters = [[] for _ in range(self.num_ranks)]
        for v in self.graph.vs:
            if v["r"] != -1:
                clusters[v["r"]].append(v.index)
        cover = igraph.VertexCover(self.graph, clusters)
        
        visual_style["layout"] = self.graph.layout("reingold_tilford")
        visual_style["margin"] = 100
        visual_style["vertex_label_color"] = "black"
        visual_style["vertex_label_angle"] = 0
        # Save the plot to the output file
        igraph.plot(cover, out_path, mark_groups=True,
                    palette=rank_colors,
                    **visual_style)
        print(f"[INFO] Saved visualization of the dependency graph to {out_path}")
        # Saves the graph as a dot file
        # Fills the calc vertices as green in the dot file
        for v in self.graph.vs:
            v["style"] = "filled"
            if v["type"] == VertexType.CALC and v["cost"] > 0:
                color = "red"
            elif v["cost"] == 0:
                color = "white"
            else:
                color = "green"
            v["fillcolor"] = color
        
        G = self.graph.to_networkx()
        # Saves the networkx graph as dot file
        networkx.drawing.nx_agraph.write_dot(G, "test.dot")

        # print(self.graph)
        # self.graph.write("test.dot", "dot")
        # self.graph.write_dot(f="test.dot")
        # igraph.write(self.graph, "test.dot", format="dot")