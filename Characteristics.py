import sys,argparse
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add

def neighborhood(graph, N, k):
    """takes rdd of edges in graph and a maximum distance integer k, and returns a list of
        tuples consisting of the node label and a list of the number of neighbors
        at a distance from 1 to k"""

    # calculates the number of neighbors in the 1st neighborhood
    neighborhood1 = graph.mapValues(lambda node2: 1).reduceByKey(add)

    # initializes an empty list of the node and the number of neighbors in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    all_neighborhoods = nodes.map(lambda u: (u, [])).partitionBy(N).cache()

    # updates the list of number of neighbors
    all_neighborhoods = all_neighborhoods.leftOuterJoin(neighborhood1) \
        .mapValues(lambda (list, val): list + [val])

    if k > 1:
        # keeps track of all the edges found in neighborhoods so far
        found_edges = graph.map(lambda edge: (edge, 1))

        # edges in current k neighborhood
        # this is initialized to the rdd of edges at distance 2
        # look in for loop comments for explanation of filtering
        current_k_edges = graph.join(graph).values().distinct() \
            .map(lambda edge: (edge, 1)).leftOuterJoin(found_edges) \
            .filter(lambda (edge, count): count[1] == None) \
            .map(lambda (edge, count): edge).filter(lambda (n1, n2): n1 != n2)


        # calculates the number of neighbors in the 2nd neighborhood
        neighborhood2 = current_k_edges.mapValues(lambda node2: 1) \
            .reduceByKey(add)

        all_neighborhoods = all_neighborhoods.leftOuterJoin(neighborhood2) \
            .mapValues(lambda (list, val): list + [val])

        if k > 2:
            # starts iterating from k = 3
            for i in range(3, k + 1):
                # updates found edges
                found_edges = found_edges.union(current_k_edges.map(lambda edge: (edge, 1)))

                # finds edges at distance k and filters out edges that were already found to be in
                # other neighborhoods
                current_k_edges = current_k_edges.join(graph).values().distinct() \
                    .map(lambda edge: (edge, 1)).leftOuterJoin(found_edges) \
                    .filter(lambda (edge, count): count[1] == None) \
                    .map(lambda (edge, count): edge)

                k_neighbors = current_k_edges.mapValues(lambda node2: 1) \
                    .reduceByKey(add)

                all_neighborhoods = all_neighborhoods.leftOuterJoin(k_neighbors) \
                    .mapValues(lambda (list, val): list + [val])

    return all_neighborhoods

def update(d,key,val):
    if val is None:
	    val = 0
    d[key]=d[key]+[val]
    return d

def paths_and_cycles(graph, N, k):
    """calculates the paths of length 1 to k and the cycles of length 3 to k"""

    # initializes an empty list of the node and the number of neighbors in each neighborhood from 1 to k
    nodes = graph.flatMap(lambda (u, v): [u, v]).distinct()
    all_cycles_and_paths = nodes \
        .map(lambda u: (u, {'cycles': [], 'paths': []})) \
        .partitionBy(N).cache()

    # keeps track of all the paths
    current_paths = graph.map(lambda (x, y): (x, [x, y]))

    # calculates the number of paths and updates the dictionary
    paths = current_paths.mapValues(lambda path_list:1).reduceByKey(lambda x,y:x+y)
    all_cycles_and_paths = all_cycles_and_paths \
        .leftOuterJoin(paths) \
        .mapValues(lambda (d, val): update(d, 'paths', val))\
        .cache()

    #adds in paths of length 2
    current_paths = current_paths.leftOuterJoin(graph) \
        .map(lambda (n1, (path_list, n2)): (n2, [n2] + path_list))\
        .filter(lambda (n1, path_list): path_list[0] != path_list[-1])\
        .cache()

    paths = current_paths.mapValues(lambda path_list: 1).reduceByKey(lambda x, y: x + y)
    all_cycles_and_paths = all_cycles_and_paths \
        .leftOuterJoin(paths) \
        .mapValues(lambda (d, val): update(d, 'paths', val)) \
        .cache()

    # starts with k = 3
    i=2
    while i<k:
        current_paths = current_paths.leftOuterJoin(graph)\
            .map(lambda (n1, (path_list, n2)): (n2, [n2] + path_list))\
            .filter(lambda (n1, path_list): path_list[0] not in path_list[1:-1])\
            .cache()

        paths = current_paths.mapValues(lambda path_list: 1).reduceByKey(lambda x, y: x + y)

        # calculates the number of cycles
        cycles = current_paths.filter(lambda (n1, path_list): path_list[0] == path_list[-1])\
            .mapValues(lambda x: 1).reduceByKey(lambda x, y: x + y).mapValues(lambda x: x/2)

        all_cycles_and_paths = all_cycles_and_paths \
            .leftOuterJoin(paths) \
            .mapValues(lambda (d, val): update(d, 'paths', val))\
            .leftOuterJoin(cycles)\
            .mapValues(lambda (d, val): update(d, "cycles", val))\
            .cache()

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
    graph = lines.map(eval)
    #neighbors = neighborhood(graph, args.N, args.k).collect()

    #print(neighbors)

    cycle_paths = paths_and_cycles(graph, args.N, args.k).collect()
    print(cycle_paths)