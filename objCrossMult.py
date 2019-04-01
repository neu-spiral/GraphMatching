import argparse
import math
from pyspark import SparkConf, SparkContext
from operator import add

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
def reOrdering_dict(G):
    """
        Assign integers [1,...,NUMBER_OF_OBJECTIVES] to the objectives, whcih are pairs of the form (v, u) and v and u are nodes of the graph.
        Return a dictionary of the translations as well as its ineverse.
    """
    obj2int = {}
    G = G.keys().collect()
    val  = 1
    for obj in G:
        if obj not in obj2int:
            obj2int[obj] = val
            val += 1
    int2obj = dict( [(obj2int[key], key) for key in obj2int])
    return obj2int, int2obj     
    
def iter2choose(objList):
    L = []
    #i = 1
    #objList = list(objList)
    for obj1 in objList:
        for obj2 in objList:
            L.append( ((obj1, obj2), 1) )
        #i = i+1
    return L
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Objective postprocessor.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('G',help = 'File containing the objectives')
    parser.add_argument('outfile', help='The file to write the outputfile',type=str)
    parser.add_argument('--N', help='Number of partitions',type=int, default=10)
    args = parser.parse_args()    

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
    sc.setLogLevel('OFF')

    G = sc.textFile(args.G, minPartitions=args.N).map(eval).cache()

    obj2int, int2obj = reOrdering_dict(G)
    maxVal = G.keys().flatMap(lambda tpl: [eval(elem) for elem in tpl]).reduce(max) + 1


    Ginv = G.flatMap(lambda  (obj, (Rlist1, Rlist2)): [(var, obj) for var in Rlist1] +  [(var, obj) for var in Rlist2])
    print Ginv.groupByKey().values().flatMap(iter2choose).count()
    GmultG = Ginv.groupByKey().values().flatMap(iter2choose)\
               .reduceByKey(add, args.N)
    GmultG = GmultG.flatMap(lambda  ((obj1, obj2), weight):  [(obj1, (obj2, weight)), (obj2, (obj1, weight))]).groupByKey().mapValues(list)
    GmultG_dict = dict( GmultG.collect() )
    print len(GmultG_dict)


    #Write the obtained graph to a file
    f  = open(args.outfile, 'w')
    for i in range(1, len(obj2int)+1):
        l = ""
        for (obj, weight) in GmultG_dict[int2obj[i]]:
            l += "%d %d " %(obj2int[obj], weight)
        l += "\n"
    f.close()
    
               
        
    
