import sys,argparse
import numpy as np
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add
from helpers import safeWrite
from preprocessor import readSnap
from time import time

def get_neighborhood(graph, N, k):
    """takes rdd of edges in graph and a maximum distance integer k, and returns the number of neighbors
        at a distance from 1 to k for each node"""

    # calculates the number of neighbors in the 1st neighborhood
    neighborhood1 = graph.mapValues(lambda node2: 1).reduceByKey(add)

    # initializes an empty list of the node and the number of neighbors in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    all_neighborhoods = nodes.map(lambda u: (u, [])).partitionBy(N).cache()
    all_neighborhoods = all_neighborhoods.leftOuterJoin(neighborhood1) \
        .mapValues(lambda (list, val): list + [val])

    # keeps track of all the visited nodes for each node so far
    visited_nodes = graph.mapValues(lambda x: {x}).reduceByKey(lambda x, y: x.union(y)).cache()

    current_k_edges = graph

    for i in range(2, k + 1):
        # finds edges at distance k and filters out nodes that were already found to be in
        # other neighborhoods
        current_k_edges = current_k_edges.join(graph).values().distinct() \
            .filter(lambda (n1, n2): n1 != n2).partitionBy(N)\
            .mapValues(lambda x: {x})\
            .reduceByKey(lambda x, y: x.union(y))\
            .leftOuterJoin(visited_nodes)\
            .mapValues(lambda (set_current_k, set_visited_nodes): set_current_k - set_visited_nodes)\
            .flatMapValues(lambda x: x).cache()

        k_neighbors = current_k_edges.mapValues(lambda node2: 1) \
            .reduceByKey(add)
        def convert_none2zero(val):
            if val==None:
                return 0.
            else:
                return val
        all_neighborhoods = all_neighborhoods.leftOuterJoin(k_neighbors) \
            .mapValues(lambda (List, val): (List,convert_none2zero(val))).mapValues(lambda (List, val): List + [val]).cache()

        # updates visited nodes
        visited_nodes = current_k_edges.mapValues(lambda x: {x}).union(visited_nodes)\
            .reduceByKey(lambda set1, set2: set1.union(set2)).cache()

    num_nodes = float(nodes.count())
    print("average number of neighbors: ", all_neighborhoods.values().map(lambda nhList: np.array(nhList) / num_nodes)\
          .reduce(lambda list1, list2: list1 + list2))

    return all_neighborhoods

def update(d,key,val):
    if val is None:
	    val = 0
    d[key]=d[key]+[val]
    return d

def get_paths_and_cycles(graph, N, k):
    """takes rdd of edges in graph and a maximum distance integer k, and calculates the paths and cycles of
    length 1 to k for each node"""

    # initializes an empty list of the node and the number of cycles and paths in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    number_of_nodes = float(nodes.count())
    all_cycles_and_paths = nodes \
        .map(lambda u: (u, {'cycles': [], 'paths': []})) \
        .partitionBy(N).cache()

    # initializes the current paths to be the graph
    current_paths = graph

    i = 0
    while i < k:
        # calculates the number of paths and cycles
        checkpnt = i!=0 and i%4==0
        paths = current_paths.mapValues(lambda n:1).reduceByKey(add)

        cycles = current_paths.filter(lambda (n1, n2): n1 == n2).mapValues(lambda n: 1)\
            .reduceByKey(add)

        all_cycles_and_paths = all_cycles_and_paths \
            .leftOuterJoin(paths) \
            .mapValues(lambda (d, val): update(d, 'paths', val)) \
            .leftOuterJoin(cycles) \
            .mapValues(lambda (d, val): update(d, 'cycles', val)) \
            .cache()

        if checkpnt:
            all_cycles_and_paths.checkpoint()
        # update the current paths
        current_paths = current_paths.leftOuterJoin(graph).values().cache()

        if checkpnt:
            current_paths.checkpoint()
        i += 1

    print ("Avg. number of paths and cycles are", all_cycles_and_paths.values()\
           .map(lambda d: tuple([d[key] for key in d])).map(lambda (paths_list, cycles_list):\
            (np.array(paths_list) / number_of_nodes, np.array(cycles_list) / number_of_nodes))\
           .reduce(lambda (paths_array1, cycles_array1), (paths_array2, cycles_array2): \
            (paths_array1 + paths_array2, cycles_array1 + cycles_array2)))

    return all_cycles_and_paths


def get_page_rank(graph, N, eps, max_iterations, gamma):
    """Calculates the page rank for each node"""
    graph_rdd = graph.groupByKey() \
        .mapValues(list) \
        .partitionBy(N) \
        .cache()

    # Discover all nodes; this finds node with no outgoing edges as well
    nodes = graph_rdd.flatMap(lambda (i, edgelist): edgelist + [i]) \
        .distinct() \
        .cache()

    # Initialize scores
    size = nodes.count()
    scores = nodes.map(lambda i: (i, 1.0 / size)).partitionBy(N).cache()

    # Main iterations
    i = 0
    err = eps + 1.0
    while i < max_iterations and err > eps:
        i += 1
        old_scores = scores
        joined = graph_rdd.join(scores)
        scores = joined.values() \
            .flatMap(lambda (neighborlist, score): [(x, 1.0 * score / len(neighborlist)) for x in neighborlist]) \
            .reduceByKey(lambda x, y: x + y, numPartitions=N) \
            .mapValues(lambda x: (1 - gamma) * x + gamma * 1 / size) \
            .cache()
        err = old_scores.join(scores).values() \
            .map(lambda (old_val, new_val): abs(old_val - new_val)) \
            .reduce(lambda x, y: x + y)

        old_scores.unpersist()
        print '### Iteration:', i, '\terror:', err

        # Give score to nodes having no incoming edges. All such nodes
    # should get score gamma / size
    remaining_nodes = nodes.map(lambda x: (x, gamma / size)).subtractByKey(scores)
    scores = scores.union(remaining_nodes)

    return scores

def merge_attributes(neighborhood_rdd, cycles_paths_rdd, pagerank, N):
    def merger(val):
        ((neighborhood_list, cycles_paths_dict), rank) = val
        merged_attribute =  neighborhood_list
        for path_number in cycles_paths_dict['paths']:
            merged_attribute.append(path_number)
        for cycles_number in cycles_paths_dict['cycles']:
            merged_attribute.append(cycles_number)
        merged_attribute.append(rank)
        return merged_attribute


    all = neighborhood_rdd.partitionBy(N).join(cycles_paths_rdd.partitionBy(N)).join(pagerank.partitionBy(N))\
        .mapValues(lambda val:merger(val)).cache()

    number_nodes = float(all.count())

    print("average number of neighbors, cycles and paths, and pagerank: ", all.values()\
          .map(lambda attr_list: np.array(attr_list) / number_nodes).reduce(lambda arr1, arr2: arr1 + arr2))

    return all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Finds number of neighbors in neighborhood k or the number '
                                                 'of cycles and paths of length k',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help = 'Option of what is being calculated (i.e. neighborhood size (NH), cycles and '
                                     'paths (CP), page rank (PR), all attributes merged (M).', choices = ['NH','CP', 'PR', 'M'])
    parser.add_argument('graph',
                        help ='Input Graph. The input should be a file containing one edge per line, '
                             'with each edge represented as a tuple of the form: (from_node,to_node).')
    parser.add_argument('k', type = int,
                        help = 'Distance of largest path computed. E.g. k = 2 computes the size of the '
                             '2-degree neighborhood and the cycles and paths of length 2')
    parser.add_argument('outputfile', type = str, help = "The directory to save the results.")
    parser.add_argument('--timeoutputfile', default = None, help = "File to store runtime data")
    parser.add_argument('--graphName', default = None, help = 'name of graph for runtime file')
    parser.add_argument('--N', type = int, default = 20, help = 'Level of Parallelism')
    parser.add_argument('--gamma',type = float, default = 0.15, help = 'Interpolation parameter')
    parser.add_argument('--max_iterations', type = int, default = 50, help = 'Maximum number of Iterations')
    parser.add_argument('--eps', type = float, default = 0.01, help = 'Desired accuracy/epsilon value')
    parser.add_argument('--fromSnap', action = 'store_true', help = 'Input file is from Snap')
    parser.add_argument('--directed', action = 'store_true', help = 'Input file is directed')

    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism', args.N)
    sc = SparkContext(appName='Characteristics', conf=configuration)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("checkpointdir")
    if args.fromSnap:
        graph = readSnap(args.graph, sc, minPartitions = args.N).partitionBy(args.N).cache()

    else:
        lines = sc.textFile(args.graph)
        graph = lines.map(eval).partitionBy(args.N).cache()
    if args.directed:
        graph = graph.flatMap(lambda (u, v): [(u, v), (v, u)]).partitionBy(args.N).cache()

    # returns the neighborhood sizes
    if args.mode == 'NH':
        neighbors = get_neighborhood(graph, args.N, args.k)
        safeWrite(neighbors, args.outputfile)

    #returns the number of cycles and paths
    if args.mode == 'CP':
        t1 = time()
        cycle_paths = get_paths_and_cycles(graph, args.N, args.k)
        t2 = time()
        print('Cycles and paths found in %f (s) for depth %d' %(t2-t1, args.k))
        safeWrite(cycle_paths, args.outputfile)

    # returns the page rank
    if args.mode == 'PR':
        pageRank = get_page_rank(graph, args.N, args.eps, args.max_iterations, args.gamma)
        pageRank.sortBy(lambda (key, val): -val)
        safeWrite(pageRank, args.outputfile)

    if args.mode == "M":
        start_time = time()
        neighbors = get_neighborhood(graph, args.N, args.k)
        nh_time = time() - start_time

        t1cp = time()
        cycles_paths = get_paths_and_cycles(graph, args.N, args.k)
        cp_time = time() - t1cp

        t1pr = time()
        pageRank = get_page_rank(graph, args.N, args.eps, args.max_iterations, args.gamma)
        pagerank_time = time() - t1pr

        all_attributes = merge_attributes(neighbors, cycles_paths, pageRank, args.N)
        tot_time = time() - start_time

        safeWrite(all_attributes, args.outputfile)

        num_nodes = graph.mapValues(lambda n2: 1).distinct().count()
        num_edges = graph.count() / 2.0

        if args.timeoutputfile:
            with open(args.timeoutputfile, 'a') as file:
                file.write("%s \t %d \t %d \t %f \t %f \t %f \t %f \n"\
                           % (args.graphName, num_nodes, num_edges, nh_time, cp_time, pagerank_time, tot_time))