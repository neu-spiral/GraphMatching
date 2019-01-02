import sys,argparse
import numpy as np
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add
from helpers import safeWrite
from time import time

def euclidean_dist(attr1, attr2):
    return np.sqrt(np.sum((attr1 - attr2)**2))

def distance(characteristics1, characteristics2, constraints=None, N=20, Equalize=False):
    """Takes rdds of all the attributes of two different graphs and computes the distance
    between each node related by a bipartite constraint graph"""

    characteristics1 = characteristics1.mapValues(lambda attributes_list: np.array(attributes_list))
    characteristics2 = characteristics2.mapValues(lambda attributes_list: np.array(attributes_list))
    if Equalize:
        n1 = characteristics1.count()
        n2 = characteristics2.count()
        
        charactrosticLen = characteristics1.values().map(lambda attr: len(attr)).take(1)[0]
        n_min = min(n1, n2)
        n_max = max(n1, n2)
        dummy = [(str(node), np.array( [0.]*charactrosticLen )) for node in range(n_min, n_max)]
        dummy = sc.parallelize(dummy).partitionBy(N).cache()
        if n1<n2:
            characteristics1 = characteristics1.union(dummy).cache()
        else:
            characteristics2 = characteristics2.union(dummy).cache()
            

    if constraints is None:
        distances = characteristics1.cartesian(characteristics2).map(lambda ((n1, attr1), (n2, attr2)):((n1, n2), euclidean_dist(attr1, attr2)))\
            .partitionBy(N)\
            .cache()                 
    else:
        distances = constraints.join(characteristics1).map(lambda (n1, (n2, attributes)): (n2, (n1, attributes)))\
            .join(characteristics2).map(lambda (n2, ((n1, attr1), attr2)): ((n1, n2), (attr1, attr2)))\
            .partitionBy(N)\
            .mapValues(lambda (attr1, attr2): euclidean_dist(attr1, attr2))\
            .cache()

    return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds distance between nodes of two graphs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('chars1',
                        help ='Input rdd. The input files containing tuples of each node in graph 1 with a list '
                              'of all of the attributes belonging to that node.')
    parser.add_argument('chars2',
                        help ='Input rdd. The input files containing tuples of each node in graph 2 with a list '
                              'of all of the attributes belonging to that node.')
    parser.add_argument('outputfile', help = 'Directory to store output distances.')
    parser.add_argument('--connection_graph',
                        help ='Input constraints bipartite connections graph. The input should be files containing '
                              'one match per line, with each match represented as a tuple of the form: '
                              '(graph1_node, graph2_node).', default=None)
    parser.add_argument('--time_outputfile', default = None, help = 'file to store runtime')
    parser.add_argument('--equalize',action='store_true', help='If passed makes sure that the graphs have the same number of nodes by randomly ignoring some  nodes from the larger graph.')
    parser.add_argument('--graphName', default = None, help = 'name of graph for runtime file')
    parser.add_argument('--N',type=int, default=20, help = 'Level of Parallelism')
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism', args.N)
    sc = SparkContext(appName='Distance', conf=configuration)
    sc.setLogLevel("ERROR")

    if args.connection_graph is not None:
        constraints = sc.textFile(args.connection_graph).map(eval).partitionBy(args.N).cache()
    else:
        constraints = None
    characteristics1 = sc.textFile(args.chars1).map(eval).partitionBy(args.N).cache()
    characteristics2 = sc.textFile(args.chars2).map(eval).partitionBy(args.N).cache()

    start_time = time()
    distance = distance(characteristics1=characteristics1, characteristics2=characteristics2,constraints=constraints, N=args.N, Equalize=args.equalize)
    pairs = float(distance.count())
    tot_time = time() - start_time

    safeWrite(distance, args.outputfile)

    if args.time_outputfile:
        with open(args.time_outputfile, 'a') as file:
            file.write("%s \t %f \n" %(args.graphName, tot_time))
