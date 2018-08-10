import sys,argparse
import numpy as np
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add
from helpers import safeWrite
from Characteristics import paths_and_cycles

def euclidean_dist(dict1, dict2):
    sum = 0.0
    for key in dict1:
        for i in range(len(dict1[key])):
            sum += (dict1[key][i] - dict2[key][i])**2
    return np.sqrt(sum)

def distance(characteristics1, characteristics2, graph, N):
    """Takes rdds of the number of cycles and paths for each node of two different graphs and computes the distance
    between each node related by a third graph"""

    distances = graph.join(characteristics1).map(lambda (n1, (n2, dict)): (n2, (n1, dict)))\
        .join(characteristics2).map(lambda (n2, ((n1, dict1), dict2)): ((n1, n2), (dict1, dict2)))\
        .partitionBy(N)\
        .mapValues(lambda (dict1, dict2): euclidean_dist(dict1, dict2))

    return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds distance between nodes of two graphs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('connection_graph',
                        help ='Input Graph. The input should be a file containing one match per line, '
                             'with each match represented as a tuple of the form: (graph1_node, graph2_node).')
    parser.add_argument('chars1',
                        help ='Input rdd. The input files containing tuples of each node in graph 1 with a dictionary '
                              'of that node\'s attributes.')
    parser.add_argument('chars2',
                        help ='Input rdd. The input files containing tuples of each node in graph 2 with a dictionary '
                              'of that node\'s attributes.')
    parser.add_argument('outputfile', help = 'Directory to store output distances.')
    parser.add_argument('--N',type=int, default=20, help = 'Level of Parallelism')
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism', args.N)
    sc = SparkContext(appName='Distance', conf=configuration)
    sc.setLogLevel("ERROR")

    connection_graph = sc.textFile(args.connection_graph).map(eval).partitionBy(args.N).cache()
    characteristics1 = sc.textFile(args.chars1).map(eval).partitionBy(args.N).cache()
    characteristics2 = sc.textFile(args.chars2).map(eval).partitionBy(args.N).cache()

    distance = distance(characteristics1, characteristics2, connection_graph, args.N)
    safeWrite(distance, args.outputfile)