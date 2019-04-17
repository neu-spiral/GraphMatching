import argparse
import math
from pyspark import SparkConf, SparkContext
from helpers import swap 
from operator import add


def readSnap(file,sc,minPartitions=10):
    '''Read a file in a format used by SNAP'''
    return sc.textFile(file,minPartitions=minPartitions)\
                .filter(lambda x: '#' not in x and '%' not in x)\
                .map(lambda x: x.split())\
                .filter(lambda edge:len(edge)==2)
                #.map(lambda (u,v):(hash(u),hash(v)))

def convertToNumber (tpl, m):
    """Encode the tuple of an objective pair to  an integer, as follows
       Given tpl = (v1, v2)
       Coded tpl = v1*m + v2
    """
    (v1, v2) = tpl
    v1 = eval(v1)
    v2 = eval(v2)
    return v1*m + v2

def convertFromNumber (n):
    "Decode integer n back to the string"
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()
def None2None(tupl):
    l = []
    if tupl[0] != None:
        l.append(tupl[0])
    if tupl[1] != None:
        l.append(tupl[1])
    return l

def SijGenerator(graph1,graph2,G,N):
    """Return the objcetives (Sij) and  variablesSqaured (Vij2), defined as follows:
           Sij = RDD of the form (obj_id:VarList)
           Vij2 = RDD of the form ((var1, var2): w12), where w12 is the number of objectives that depend on both var1 and var2.
    """
    #Compute the support of objectives 
    Sij1 = G.join(graph1.map(swap) ).map(lambda (k,(j,i)): ((i,j),(k,j))).partitionBy(N)
    Sij2 = G.map(swap).join(graph2).map(lambda (k,(i,j)): ((i,j),(i,k))).partitionBy(N)

    #Objectives
    Sij = Sij1.fullOuterJoin(Sij2,N)
    
    print Sij.join(Sij).count()
    Vij2 = Sij.join(Sij).filter(lambda (key, (var1, var2)):var1 != var2)\
              .map(lambda (obj, var_pair): (var_pair, 1))\
              .reduceByKey(add, N)
    print Vij2.count() 
    


    return Sij
    




def streamPartition(biGraph,K,N=100,c=lambda x:x^2):
     """Given a bipartite graph G(L,R,E), produce a streamed balanced partition of nodes on L. The bipartite graph is represented as an rdd of edges E=(i,j), with i in L and j in R.
     """
     invGraph = biGraph.map(swap)\
                        .partitionBy(N)
 
 
     L = biGraph.map(lambda (x,y):x).distinct(numPartitions=N).collect()
     #R = biGraph.map(lambda (x,y):y).distinct(numParitions=N).cache()

     partition = {}
     partitionSizes =  dict([(part,0) for part in range(K)])
     for i in L:
         gain = dict([(part,0) for part in range(K)])
         neighbors = invGraph.join(
                              biGraph.filter(lambda (x,y):x==i)
                               .map(swap)
                               .partitionBy(N))\
                            .map(lambda (y,(z,anchor)):z).collect() 
         for z in neighbors:
             if z in partition:
                gain[partition[z]] += 1 
             
         for part in gain:
               gain[part] -= c(partitionSizes[part]+1)- c(partitionSizes[part])          

         (opt_part,val) = max(gain.iteritems(), key=lambda (part,val):val)
         partition[i] = opt_part
         partitionSizes[opt_part] += 1           

     return partition

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Module to create objectives and objectives_squared.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('G',help = 'File containing the objectives')
    parser.add_argument('graph1',help = 'File containing the first graph')
    parser.add_argument('graph2',help = 'File containing the second graph')
    parser.add_argument('outfile', help='The file to write the outputfile',type=str)
    parser.add_argument('--N', help='Number of partitions',type=int, default=10)
  
    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=True)
    args = parser.parse_args()    

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
    #sc.setLogLevel('OFF')

    G = sc.textFile(args.G, minPartitions=args.N).map(eval).cache()

    if args.fromsnap:
        graph1 = readSnap(args.graph1,sc,minPartitions=args.N)
        graph2 = readSnap(args.graph2,sc,minPartitions=args.N)

    else:
        graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
        graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)



    SijGenerator(graph1,graph2,G,args.N)

    #Write the obtained graph to a file
    
               
        
    
