import sys,argparse
import numpy as np
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add
from Characteristics import paths_and_cycles

def distance(characteristics1, characteristics2, graph, N):
    """Takes rdds of the number of cycles and paths for each node of two different graphs and computes the distance
    between each node related by a third graph"""

    distances = graph.join(characteristics1).map(lambda (n1, (n2, dict)): (n2, (n1, dict)))\
        .join(characteristics2).map(lambda (n2, ((n1, dict1), dict2)): ((n1, n2), (dict1, dict2)))\
        .partitionBy(N)\
        .flatMapValues(lambda (dict1, dict2): [(dict1['cycles'][i] - dict2['cycles'][i])**2 \
                                              + (dict1['paths'][i] - dict2['paths'][i])**2 \
                                             for i in range(len(dict1['paths']))]) \
        .reduceByKey(add).mapValues(lambda x: np.sqrt(x))

    return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds distance between nodes of two graphs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('connection_graph',
                        help ='Input Graph. The input should be a file containing one match per line, '
                             'with each match represented as a tuple of the form: (graph1_node, graph2_node).')
    parser.add_argument('graph1',
                        help ='Input graph. The input should be a file containing one edge per line, '
                            'with each edge represented as a tuple of the form: (from_node, to_node).')
    parser.add_argument('graph2',
                        help ='Input graph. The input should be a file containing one edge per line, '
                            'with each edge represented as a tuple of the form: (from_node, to_node).')
    parser.add_argument('k', type=int,
                        help='Distance of largest path computed. E.g. k=2 computes the numbere of cycles and paths '
                             'of length k.')
    parser.add_argument('--N',type=int, default=20, help = 'Level of Parallelism')
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism', args.N)
    sc = SparkContext(appName='Distance', conf=configuration)
    sc.setLogLevel("ERROR")

    connection_graph = sc.textFile(args.connection_graph).map(eval).partitionBy(args.N).cache()
    graph1 = sc.textFile(args.graph1).map(eval).partitionBy(args.N).cache()
    graph2 = sc.textFile(args.graph2).map(eval).partitionBy(args.N).cache()

    characteristics1 = paths_and_cycles(graph1, args.k, args.N)
    characteristics2 = paths_and_cycles(graph2, args.k, args.N)

    distance = distance(characteristics1, characteristics2, connection_graph, args.N).collect()
    print(distance)