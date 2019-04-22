import argparse
import json
import math
from pyspark import SparkConf, SparkContext
from helpers import swap, identityHash
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
    
    #Vij2 = Sij.join(Sij).filter(lambda (key, (var1, var2)):var1 != var2)\
    #          .map(lambda (obj, var_pair): (var_pair, 1))\
    #          .reduceByKey(add, N)
    


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
     cnt = 0
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
         cnt += 1
         if cnt % 20 ==1:
             print "Done assigining for %d nodes" %cnt  
     return partition
def partition2RDD(partition, sc):
    """
        Given partitioning as partition, return the RDD.
    """
    numParts = len(set([partition[node] for node in partition]) )
    return sc.paralelize([(partition[node], node) for node in partition]).partitionBy(numParts, partitionFunc=identityHash)
def partitionOtherSide(biGraph, partition, N=10):
    """
        Gievn partitioning for L, generate partitioning for R.
    """
    def keepMax((part_id1, part_id_cnt1), (part_id2, part_id_cnt2)):
        if part_id_cnt1>part_id_cnt2:
            keep_part = part_id1
            keep_part_cnt = part_id_cnt1
        else:
            keep_part = part_id2
            keep_part_cnt = part_id_cnt2
        return (keep_part, keep_part_cnt)
        
    return biGraph.map(swap)\
            .join(partition.map(swap))\
            .map(lambda (var, (obj, part_id)): ((obj, part_id), 1))\
            .reduceByKey(add, N)\
            .map(lambda ((obj, part_id), part_id_cnt):(obj, (part_id, part_id_cnt)))\
            .reduceByKey(keepMax, N)\
            .map(swap)
                
   
  
    
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Module to create objectives and objectives_squared.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('G',help = 'File containing the objectives')
    #parser.add_argument('graph1',help = 'File containing the first graph')
    #parser.add_argument('graph2',help = 'File containing the second graph')
    parser.add_argument('outfile', help='The file to write the outputfile',type=str)
    parser.add_argument('--N', help='Number of partitions',type=int, default=10)
    parser.add_argument('--K', help='Desired number of partitions',type=int, default=10)
  
    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=True)
    args = parser.parse_args()    

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
    sc.setLogLevel('OFF')

    G = sc.textFile(args.G, minPartitions=args.N).map(eval).cache()
    G = G.flatMapValues(lambda (LL, RL):[elem for elem in LL] + [elem for elem in RL]).cache() 

   # if args.fromsnap:
   #     graph1 = readSnap(args.graph1,sc,minPartitions=args.N)
   #     graph2 = readSnap(args.graph2,sc,minPartitions=args.N)

    #else:
    #    graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
    #    graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)


    #Get partitions for L
    L_partitions = streamPartition(biGraph=G.map(swap),K=args.K , N=args.N)
    with open(args.outfile + '.json', 'w') as outfile:
        json.dump(partitions, outfile)
    R_partitions = partitionOtherSide(biGraph, L_partitions , N=args.N)
    
    
   # SijGenerator(graph1,graph2,G,args.N)

    #Write the obtained graph to a file
    
               
        
    
