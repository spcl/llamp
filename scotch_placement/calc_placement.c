#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "scotch.h"
#include "argparse.h"


/**
 * @brief Builds the source graph from the given file.
 * @param graph The graph to be built.
 * @param grf_file_path The path to the grf file.
 **/
int load_source_graph(SCOTCH_Graph *graph, char *grf_file_path)
{
    printf("[INFO] Loading source graph from %s\n", grf_file_path);
    /* Initializes the graph */
    if (SCOTCH_graphInit(graph) != 0)
    {
        fprintf(stderr, "[ERROR] Could not initialize graph.\n");
        return 1;
    }
    
    FILE *file_ptr;

    if ((file_ptr = fopen(grf_file_path, "r")) == NULL)
    {
        fprintf(stderr, "[ERROR] Could not open file %s.\n", grf_file_path);
        return 1;
    }

    /* Loads the graph */
    if ((SCOTCH_graphLoad(graph, file_ptr, -1, 0)) != 0)
    {
        fprintf(stderr, "[ERROR] Could not load graph.\n");
        return 1;
    }
    printf("[INFO] Load source graph: SUCCESS\n");
    return 0;
}

/**
 * @brief Builds the TLeaf hierarchical architecture graph that
 * consists of two levels based on the given parameters,
 * where the number of cores per node is the number of vertices
 * in the second level. The first level is the number of nodes.
 * @param arch The architecture graph to be built.
 * @param num_nodes The number of nodes in the architecture.
 * @param num_cores_per_node The number of cores per node.
 * @param intra_node_cost The cost of intra-node communication.
 * @param inter_node_cost The cost of inter-node communication.
 **/
int build_arch_graph(SCOTCH_Arch *arch, int num_nodes, int num_cores_per_node,
                     int intra_node_cost, int inter_node_cost)
{   
    printf("[INFO] Building architecture graph\n");
    printf("[INFO] Number of nodes: %d\n", num_nodes);
    printf("[INFO] Number of cores per node: %d\n", num_cores_per_node);
    printf("[INFO] Intra-node cost: %d\n", intra_node_cost);
    printf("[INFO] Inter-node cost: %d\n", inter_node_cost);

    /* Initializes the parameters */
    const SCOTCH_Num NUM_LEVELS = 2;
    SCOTCH_Num *size_tab = 
        (SCOTCH_Num *) malloc(sizeof(SCOTCH_Num) * NUM_LEVELS);
    assert (size_tab != NULL);
    size_tab[0] = num_nodes;
    size_tab[1] = num_cores_per_node;

    SCOTCH_Num *link_tab = 
        (SCOTCH_Num *) malloc(sizeof(SCOTCH_Num) * NUM_LEVELS);
    assert (link_tab != NULL);
    link_tab[0] = inter_node_cost;
    link_tab[1] = intra_node_cost;

    if (SCOTCH_archTleaf(arch, NUM_LEVELS, size_tab, link_tab) != 0)
    {
        fprintf(stderr, "[ERROR] Could not build architecture graph.\n");
        return 1;
    }
    printf("[INFO] Build architecture graph: SUCCESS\n");
    return 0;
}


/**
 * @brief Writes the result of the mapping to a given file in the 
 * cray MPICH format. For example, if the mapping is [[0,2,4,6],[1,3,5,7]]
 * the output will be:
 * 0,2,4,6,1,3,5,7
 * @param part_tab The array that holds the mapping.
 * @param num_ranks The number of ranks.
*/
int write_rankmap_file(SCOTCH_Num *part_tab, int num_ranks, char *file_path)
{
    printf("[INFO] Writing the mapping result to %s\n", file_path);

    // Converts the mapping to an array in which the index is the position
    // The value is the mapped rank
    int res[num_ranks];
    for (int i = 0; i < num_ranks; i++)
    {
        res[part_tab[i]] = i;
    }

    FILE *file_ptr;
    if ((file_ptr = fopen("rankmap.txt", "w")) == NULL)
    {
        fprintf(stderr, "[ERROR] Could not open file rankmap.txt.\n");
        return 1;
    }

    for (int i = 0; i < num_ranks; i++)
    {
        fprintf(file_ptr, "%d", res[i]);
        if (i != num_ranks - 1)
        {
            fprintf(file_ptr, ",");
        }
    }
    fclose(file_ptr);
    printf("[INFO] Write %s: SUCCESS\n", file_path);
    return 0;
}

/**
 * @brief Calculates the placement of the given graph onto the given
 * architecture.
 **/
int main(int argc, const char **argv)
{
    /* Parse the arguments */
    char *source_path = NULL;
    char *rankmap_file = "rankmap.txt";
    // Default values for the architecture
    int num_nodes = 2;
    int inter_node_cost = 100;
    int intra_node_cost = 10;

    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Options"),
        OPT_STRING('s', "source_graph", &source_path, "The path to the source graph. [REQUIRED]", NULL, 0, 0),
        OPT_INTEGER('n', "num_nodes", &num_nodes, "The number of nodes in the architecture. [DEFAULT: 2]", NULL, 0, 0),
        OPT_INTEGER('i', "inter_node_cost", &inter_node_cost, "The cost of inter-node communication. [DEFAULT: 100]", NULL, 0, 0),
        OPT_INTEGER('r', "intra_node_cost", &intra_node_cost, "The cost of intra-node communication. [DEFAULT: 10]", NULL, 0, 0),
        OPT_STRING('o', "rankmap_file", &rankmap_file, "The path to the file that will hold the mapping result. [DEFAULT: rankmap.txt]", NULL, 0, 0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, NULL, 0);

    argc = argparse_parse(&argparse, argc, argv);

    if (source_path == NULL)
    {
        fprintf(stderr, "[ERROR] Path to the source graph must be provided.\n");
        return 1;
    }
    printf("[INFO] Source graph path: %s\n", source_path);
    printf("[INFO] Number of nodes: %d\n", num_nodes);
    printf("[INFO] Inter-node cost: %d\n", inter_node_cost);
    printf("[INFO] Intra-node cost: %d\n", intra_node_cost);

    SCOTCH_Graph graph;
    SCOTCH_Strat strat;
    SCOTCH_Arch arch;
    
    /* Initialize the graph */
    if (load_source_graph(&graph, source_path) != 0)
    {
        fprintf(stderr, "[ERROR] Could not load source graph.\n");
        return 1;
    }
    // Gets the number of ranks from the graph
    SCOTCH_Num num_ranks;
    SCOTCH_graphSize(&graph, &num_ranks, NULL);
    printf("[INFO] Number of ranks: %d\n", num_ranks);
    int num_cores_per_node = num_ranks / num_nodes;
    printf("[INFO] Number of cores per node: %d\n", num_cores_per_node);
    
    /* Initialize the strategy */
    if (SCOTCH_stratInit(&strat) != 0)
    {
        fprintf(stderr, "[ERROR] Could not initialize strategy.\n");
        return 1;
    }

    /* Initialize the architecture */
    if (build_arch_graph(&arch, num_nodes, num_cores_per_node,
                         intra_node_cost, inter_node_cost) != 0)
    {
        fprintf(stderr, "[ERROR] Could not build architecture graph.\n");
        return 1;
    }

    // Allocates memory for the array that will hold the mapping
    SCOTCH_Num *part_tab = 
        (SCOTCH_Num *) malloc(sizeof(SCOTCH_Num) * num_ranks);

    /* Maps the source graph onto the architecture graph */
    if (SCOTCH_graphMap(&graph, &arch, &strat, part_tab) != 0)
    {
        fprintf(stderr, "[ERROR] Could not map the source graph to the architecture.\n");
        return 1;
    }
    
    /* Prints the result */
    printf("================== Mapping Result ==================\n");
    for (int i = 0; i < num_ranks; i++)
    {
        printf("Rank %d is mapped to %d\n", i, part_tab[i]);
    }
    printf("====================================================\n");
    
    // Writes the result to a file
    if (write_rankmap_file(part_tab, num_ranks, rankmap_file) != 0)
    {
        fprintf(stderr, "[ERROR] Could not write the mapping result to a file.\n");
        return 1;
    }
    /* Clean up */
    free(part_tab);
    SCOTCH_graphExit(&graph);
    SCOTCH_stratExit(&strat);
}