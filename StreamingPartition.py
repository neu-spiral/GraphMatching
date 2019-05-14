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
    




def streamPartition(biGraph,K,N=100,c=lambda x:x**2):
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

def SequentialStreamPartition(biGraph,K,c=lambda x:x**2):
    """Given a bipartite graph G(L,R,E), produce a streamed balanced partition of nodes on L. The bipartite graph is represented as an rdd of edges E=(i,j), with i in L and j in R.
    """
    #Make inverse graph
    invGraph = {}
    for obj in biGraph:
        for var in biGraph[obj]:
            if var not in invGraph:
                invGraph[var] = [obj]
            else:
                invGraph[var].append(obj)

     
    partition = {}
    partitionSizes =  dict([(part,0) for part in range(K)])
    cnt = 0
    innerEdges = dict([(part,0) for part in range(K)])
    allEdges = 0
    for node in biGraph:
        gain = dict([(part,0) for part in range(K)])
        neighbors = []
        for node_R in biGraph[node]:
            for node_L_neigh in invGraph[node_R]:
                if node_L_neigh != node:
                    neighbors.append(node_L_neigh)       
        allEdges += len(neighbors)
        for z in neighbors:
            if z in partition:
                gain[partition[z]] += 1

        for part in range(K):
            gain[part] +=  c(partitionSizes[part]) -  c(partitionSizes[part]+1) 
       

        (opt_part,val) = max(gain.iteritems(), key=lambda (part,val):val)
        innerEdges[opt_part] += gain[opt_part] - c(partitionSizes[opt_part]) + c(partitionSizes[opt_part]+1)
        partition[node] = opt_part
        partitionSizes[opt_part] += 1
        
        cnt += 1
        if cnt % 1000 ==1:
            print "Done assigining for %d nodes" %cnt

    maxEdges_part, maxEdges = max(innerEdges.iteritems(), key=lambda (part,val):val)
    minEdges_part, minEdges = min(innerEdges.iteritems(), key=lambda (part,val):val)
    maxSize_part, maxSize =  max(partitionSizes.iteritems(), key=lambda (part,val):val)
    minSize_part, minSize =  min(partitionSizes.iteritems(), key=lambda (part,val):val)
    print "The min inner edges is {} the max inner edges is {}, and all edges {}".format(minEdges, maxEdges, allEdges) 
    print "The min partition size is {} the max partition size is {} the average size is {}\n".format(minSize, maxSize, len(biGraph.keys())/K)
    
    return partition
def partition2RDD(partition, sc):
    """
        Given partitioning as partition, return the RDD.
    """
    numParts = len(set([partition[node] for node in partition]) )
    return sc.paralelize([(partition[node], node) for node in partition]).partitionBy(numParts, partitionFunc=identityHash)
def partitionRightSide(biGraph, partition, N=10):
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
                
def SeqPartitionRightSide(biGraph, partition):
    """
        Gievn partitioning for L, generate partitioning for R.
    """   
    partition_R = {}
    final_partition_R = {}
    for node_L in partition:
        current_id = partition[node_L]
        for node_R in biGraph[node_L]:
            if node_R not in partition_R:
                partition_R[node_R] = dict([(current_id, 1)])
            elif current_id not in partition_R[node_R]:
                partition_R[node_R][current_id] = 1
            else:
                partition_R[node_R][current_id] += 1
    partitionSizes = {}
    for node_R in  partition_R:
        opt_part, opt_part_size = max(partition_R[node_R].iteritems(), key=lambda (part,val):val)
        final_partition_R[node_R] = opt_part
        if opt_part not in partitionSizes:
            partitionSizes[opt_part] = 1
        else:
            partitionSizes[opt_part] += 1  

    maxSize_part, maxSize =  max(partitionSizes.iteritems(), key=lambda (part,val):val)
    minSize_part, minSize =  min(partitionSizes.iteritems(), key=lambda (part,val):val)
    print "The min partition size is {} the max partition size is {} the average size is {}\n".format(minSize, maxSize, len(biGraph.keys())/len(partitionSizes))
    
    return final_partition_R
       
 
  
    
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Module to create objectives and objectives_squared.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('G',help = 'File containing the objectives')
    #parser.add_argument('graph1',help = 'File containing the first graph')
    #parser.add_argument('graph2',help = 'File containing the second graph')
    parser.add_argument('outfile', help='The file to write the outputfile',type=str)
    parser.add_argument('--N', help='Number of partitions',type=int, default=10)
    parser.add_argument('--K', help='Desired number of partitions',type=int, default=10)
  
    dirgroup = parser.add_mutually_exclusive_group(required=False)
    dirgroup.add_argument('--inverse', dest='inverse', action='store_true',help='Partition the inverse of the given bi-partite graph (default).')
    dirgroup.add_argument('--forward', dest='inverse', action='store_false',help='Partition the given bi-partite graph.')
    parser.set_defaults(inverse=True)
    args = parser.parse_args()    

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
    sc.setLogLevel('OFF')

    if args.inverse:
        G = sc.textFile(args.G, minPartitions=args.N).map(eval)
        Ginv = G.flatMapValues(lambda (LL, RL):LL + RL)\
                .map(swap).partitionBy(args.N)\
                .groupByKey()\
                .collect()
        Ginv = dict(Ginv)
   
        #Get partitions for LHS (vars) of the inverse 
        L_partitions = SequentialStreamPartition(biGraph=Ginv, K=args.K)

        #Get partitions for RHS of the inverse (objs) according to LHS 
        R_partitions = SeqPartitionRightSide(biGraph=Ginv, partition=L_partitions)
    
        #Write the obtained partitions
        with open(args.outfile + '_VAR.json', 'w') as fout:
            L_partitions = dict([(str(key), L_partitions[key]) for key in L_partitions])
            json.dump(L_partitions, fout)
        with open(args.outfile + '_OBJ.json', 'w') as fout:
            R_partitions = dict([(str(key), R_partitions[key]) for key in R_partitions])
            json.dump(R_partitions, fout)
        
    else:

        G = sc.textFile(args.G, minPartitions=args.N).map(eval).cache()
        G = G.mapValues(lambda (LL, RL):LL + RL).collect()
        G = dict(G)


        #Get partitions for LHS of the forward (objs)
        L_partitions = SequentialStreamPartition(biGraph=G, K=args.K)
    
        #Get partitions for RHS of the forward (vars) according to LHS 
        R_partitions = SeqPartitionRightSide(biGraph=G, partition=L_partitions)
    
    
        #Write the obtained partitions
        with open(args.outfile + '_OBJ.json', 'w') as fout:
            L_partitions = dict([(str(key), L_partitions[key]) for key in L_partitions])
            json.dump(L_partitions, fout)
        with open(args.outfile + '_VAR.json', 'w') as fout:
            R_partitions = dict([(str(key), R_partitions[key]) for key in R_partitions])
            json.dump(R_partitions, fout)
       
    
               
        
    
