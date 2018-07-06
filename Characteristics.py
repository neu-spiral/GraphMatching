import sys,argparse
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add

def neighborhood(graph, N, k):
    """takes rdd of edges in graph and a maximum distance integer k, and returns the number of neighbors
        at a distance from 1 to k for each node"""

    # calculates the number of neighbors in the 1st neighborhood
    neighborhood1 = graph.mapValues(lambda node2: 1).reduceByKey(add)

    # initializes an empty list of the node and the number of neighbors in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    all_neighborhoods = nodes.map(lambda u: (u, [])).partitionBy(N).cache()

    # updates the list of number of neighbors
    all_neighborhoods = all_neighborhoods.leftOuterJoin(neighborhood1) \
        .mapValues(lambda (list, val): list + [val])

    # keeps track of all the visited nodes for each node so far
    visited_nodes = graph.mapValues(lambda x: {x}).reduceByKey(lambda x, y: x.union(y)).cache()

    # edges in current k neighborhood
    # this is initialized to the rdd of edges at distance 2
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

        all_neighborhoods = all_neighborhoods.leftOuterJoin(k_neighbors) \
            .mapValues(lambda (list, val): list + [val]).cache()

        # updates visited nodes
        visited_nodes = current_k_edges.mapValues(lambda x: {x}).union(visited_nodes)\
            .reduceByKey(lambda set1, set2: set1.union(set2)).cache()

    return all_neighborhoods

def update(d,key,val):
    if val is None:
	    val = 0
    d[key]=d[key]+[val]
    return d

def paths_and_cycles(graph, N, k):
    """takes rdd of edges in graph and a maximum distance integer k, and calculates the paths and cycles of length 1 to k for each node"""

    # initializes an empty list of the node and the number of cycles and paths in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    all_cycles_and_paths = nodes \
        .map(lambda u: (u, {'cycles': [], 'paths': []})) \
        .partitionBy(N).cache()

    # initializes the current paths to be the graph
    current_paths = graph

    i = 0
    while i < k:
        # calculates the number of paths and cycles
        paths = current_paths.mapValues(lambda n:1).reduceByKey(add)

        cycles = current_paths.filter(lambda (n1, n2): n1 == n2).mapValues(lambda n: 1)\
            .reduceByKey(add)

        all_cycles_and_paths = all_cycles_and_paths \
            .leftOuterJoin(paths) \
            .mapValues(lambda (d, val): update(d, 'paths', val)) \
            .leftOuterJoin(cycles) \
            .mapValues(lambda (d, val): update(d, 'cycles', val)) \
            .cache()

        # update the current paths
        current_paths = current_paths.leftOuterJoin(graph).values().cache()

        i += 1

    return all_cycles_and_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds number of neighbors in neighborhood k',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',
                        help ='Input Graph. The input should be a file containing one edge per line, '
                             'with each edge represented as a tuple of the form: (from_node,to_node).')
    parser.add_argument('k', type=int,
                        help ='Distance of largest path computed. E.g. k=2 computes the size of the '
                             '2-degree neighborhood.')
    parser.add_argument('--N',type=int, default=20, help = 'Level of Parallelism')
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism', args.N)
    sc = SparkContext(appName='Neighbors', conf=configuration)
    sc.setLogLevel("ERROR")

    lines = sc.textFile(args.graph)
    graph = lines.map(eval).partitionBy(args.N).cache()

    neighbors = neighborhood(graph, args.N, args.k).collect()
    print(neighbors)

    cycle_paths = paths_and_cycles(graph, args.N, args.k).collect()
    print(cycle_paths)
