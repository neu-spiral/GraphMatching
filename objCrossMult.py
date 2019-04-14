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
    """Return the objcetives (Sij) and and  objectivesSqaured (Sij2), defined as follows:
           Sij = RDD of the form (obj_id:VarList)
           Sin2 = RDD of the form ((obj_id1, obj_id2): wieght_12), where wieght_12 is the number of common variables between obj_id1 and obj_id2, i.e., the size of the intersection of VarList1 and VarLit2.
    """
    #Compute S_ij^1 and S_ij^2
    Sij1 = G.join(graph1.map(swap) ).map(lambda (k,(j,i)): ((i,j),(k,j))).partitionBy(N)
    Sij2 = G.map(swap).join(graph2).map(lambda (k,(i,j)): ((i,j),(i,k))).partitionBy(N)

    Sij = Sij1.fullOuterJoin(Sij2,N)
    print Sij.mapValues(None2None).take(1)


    return Sij
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Objective postprocessor.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    sc.setLogLevel('OFF')

    G = sc.textFile(args.G, minPartitions=args.N).map(eval).cache()

    if args.fromsnap:
        graph1 = readSnap(args.graph1,sc,minPartitions=args.N)
        graph2 = readSnap(args.graph2,sc,minPartitions=args.N)

    else:
        graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
        graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)



    SijGenerator(graph1,graph2,G,args.N)

    #Write the obtained graph to a file
    
               
        
    
