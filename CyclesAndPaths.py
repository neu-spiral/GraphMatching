import sys,argparse
from pyspark import SparkContext,SparkConf,StorageLevel
from operator import add
from Attributes import get_cycles_and_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cycles and Paths Algorithm',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('graph', help='Input Graph. The input should be a file containing one edge per line, with each edge represented as a tuple of the form: (from_node,to_node)')
    parser.add_argument('k',type=int,default=2,help='Distance of largest path computed. E.g. k=2 computes the size of the 2-degree neibhborhood and the number of cycles of length 2.')
    parser.add_argument('--output', default="output",help='File in which PageRank scores are stored')
    parser.add_argument('--N',type=int,default=20,help = 'Level of Parallelism')
    parser.add_argument('--undirected', dest='undirected', action='store_true',help='Treat inputs as undirected graphs; this is the default behavior.')
    parser.add_argument('--directed', dest='undirected', action='store_false',help='Treat inputs as directed graphs. Edge (i,j) does not imply existence of (j,i).')
    parser.add_argument('--storage_level', default="MEMORY_ONLY",help='Control Spark caching/persistence behavrior',choices=['MEMORY_ONLY','MEMORY_AND_DISK','DISK_ONLY'])
    parser.set_defaults(undirected=True)
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Cycles and Paths', conf=configuration)
    storage_level=eval("StorageLevel."+args.storage_level)

    # Read graph and generate rdd containing node and outgoing edges 
    # from file containing edges
    lines = sc.textFile(args.graph)
    graph=lines.map(eval)

    if args.undirected:
        graph = graph.flatMap(lambda (u,v):[ (u,v),(v,u)])

    graph = graph.partitionBy(args.N).persist(storage_level)
    all_cycles_and_paths = get_cycles_and_paths(graph,args.N,args.k)
 
    all_cycles_and_paths.saveAsTextFile(args.output)
