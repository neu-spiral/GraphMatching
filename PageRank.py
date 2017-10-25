import sys,argparse
from pyspark import SparkContext
from Attributes import PageRank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'PageRank Algorithm',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('graph', help='Input Graph. The input should be a file containing one edge per line, with each edge represented as a tuple of the form: (from_node,to_node)')
    parser.add_argument('--output', default="output",help='File in which PageRank scores are stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--N',type=int,default=20,help = 'Level of Parallelism')
    parser.add_argument('--gamma',type=float,default=0.15,help = 'Interpolation parameter')
    parser.add_argument('--max_iterations',type=int,default=50,help = 'Maximum number of Iterations')
    parser.add_argument('--eps',type=float,default=0.01,help = 'Desired accuracy/epsilon value')
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Page Rank')

    lines = sc.textFile(args.graph)
  
    # Read graph and generate rdd containing node and outgoing edges 
    # from file containing edges
    graph_rdd=lines.map(eval).cache()
 
    scores = get_page_rank(graph,args.N,args.eps,args.max_iterations,args.gamma):

    scores.sortBy(lambda (key,val):-val).saveAsTextFile(args.output)
