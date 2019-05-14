import numpy as np
import sys,argparse,logging,datetime
import networkx.algorithms.matching as mm
import networkx as nx

from pyspark import SparkContext,StorageLevel,SparkConf
from Characteristics import get_neighborhood
from LocalSolvers import SijGenerator
from operator import add
from helpers import swap,safeWrite,clearFile, NoneToZero, Loop2Zero

def readSnap(file,sc,minPartitions=10):
    '''Read a file in a format used by SNAP'''
    return sc.textFile(file,minPartitions=minPartitions)\
                .filter(lambda x: '#' not in x and '%' not in x)\
		.map(lambda x: x.split())\
                .filter(lambda edge:len(edge)==2)
		#.map(lambda (u,v):(hash(u),hash(v)))
	

def WL(graph,logger,depth = 10,numPartitions=10,storage_level=StorageLevel.MEMORY_ONLY):
    '''Implementation of the Weisfeiler-Leman Algorithm.
    '''
    graphout=graph.partitionBy(numPartitions).persist(storage_level)
    graphin=graph.map(swap).partitionBy(numPartitions).persist(storage_level)
    colors = graphout.flatMap(lambda (u,v):[u,v]).distinct().map(lambda u: (u,1)).partitionBy(numPartitions).persist(storage_level)
    
    for i in range(depth):
	oldcolors = colors
	outsig = graphin.join(colors).values().groupByKey(numPartitions).mapValues(lambda x:tuple(sorted(list(x))))
	insig = graphout.join(colors).values().groupByKey(numPartitions).mapValues(lambda x:tuple(sorted(list(x))))
      #  colors = outsig.join(insig).mapValues(hash).persist(storage_level)
        colors = outsig.fullOuterJoin(insig).mapValues(lambda (c1, c2): (NoneToZero(c1), NoneToZero(c2))).mapValues(hash).persist(storage_level)
        #oldcolors.unpersist()
	numcolors = colors.values().distinct().count()
	logger.info("IT = "+str(i)+"\t#colors = "+str(numcolors))
    return colors
    

def degrees(graph,offset = 0,numPartitions=10):
    outdegrees=graph.map(lambda (u,v): Loop2Zero(u, v) ).reduceByKey(add,numPartitions=numPartitions).flatMapValues(lambda x: [ x+i for i in range(-offset,+offset+1) ])
    indegrees=graph.map(lambda (u,v): Loop2Zero(u, v) ).reduceByKey(add,numPartitions=numPartitions).flatMapValues(lambda x: [ x+i for i in range(-offset,+offset+1) ])
    degrees = outdegrees.join(indegrees,numPartitions=numPartitions)
    return degrees

def hashNodes(color):
     '''Return an RDD where each key is mapped to a unique number in {0,...,# of nodes-1}.

        Starting at i=0, it iteratively applies the following hashing function on each node (key):
            h: Nodes * {0,...,# of nodes-1} -> {0,...,# of nodes-1},
            h(v, i) = [hash(v) + i] % (# of nodes); until h(v, i) is not taken by another node. It assigns h(v,i) to v. 
     '''
     def h(v, i, M):
          return (hash(v) + i) % M
     M = color.count()
     
     nodes = color.keys().collect()
     permutate = {}
     reserved = []
     for v in nodes:
         for i in range(M):
             new_i =   h(v, i, M)
             if new_i not in reserved:
                  permutate[v] = new_i
                  reserved.append(new_i)
                  break 
     return color.map(lambda (key, val): (permutate[key], val)).persist(storage_level)
     
    
def matchColors(color1,color2,numPartitions=10):
    '''Constructs constraint graph by matching classes indicated by colors.
    '''
<<<<<<< HEAD
   # return color1.map(swap).join(color2.map(swap),numPartitions=numPartitions).values().distinct().partitionBy(numPartitions)
#    return color1.map(swap).join(color2.map(swap),numPartitions=numPartitions).values().partitionBy(numPartitions)
=======
    #return color1.map(swap).join(color2.map(swap),numPartitions=numPartitions).values().partitionBy(numPartitions).distinct()
>>>>>>> 027b5c9481fda2434e700d1ef96836be57f898c6
    return color1.map(swap).join(color2.map(swap),numPartitions=numPartitions).values().partitionBy(numPartitions).distinct()



    #Note that two graphs do not necessarily have the same node numbers or node names
   # N1 = color1.count()
   # N2 = color2.count()
   # N_common = max(N1, N2)

   # color1 = color1.map(lambda (i, color_i): (hash(i)%N_common, color_i)).distinct()
   # color2 = color2.map(lambda (i, color_i): (hash(i)%N_common, color_i)).distinct()
    
    #Make sure that the graphs have the same node numbers. 
   # jointColors = color1.join(color2, numPartitions=numPartitions)
   # color1 = jointColors.mapValues(lambda (colori_1, colori_2): colori_1)
   # color2 = jointColors.mapValues(lambda (colori_1, colori_2): colori_2)
     
   # return color1.map(swap).join(color2.map(swap),numPartitions=numPartitions).values().partitionBy(numPartitions)
         
    



def cartesianProduct(graph1,graph2):
    nodes1 = graph1.flatMap(lambda (u,v):[u,v]).distinct()
    nodes2 = graph2.flatMap(lambda (u,v):[u,v]).distinct()
    
    G = nodes1.cartesian(nodes2)
    return G





   
if __name__=="__main__": 
    parser = argparse.ArgumentParser(description = 'Graph Preprocessor .',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph1',help = 'File containing first graph')
    parser.add_argument('graph2',help = 'File containing second graph')
    parser.add_argument('--outputconstraintfile',default='None',help ="File for constraint graph. ")
    parser.add_argument('--constraintmethod',default='degree', help='Constraint generation method',choices=['degree','all','WL','neighbor'])
    parser.add_argument('--debug',default='INFO', help='Debug level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--N',default=8,type=int, help='Number of partitions')
    parser.add_argument('--degreedistance',default=0,type=int, help='Distance of degrees')

    parser.add_argument('--k',type=int,default=10, help='Number of iterations')
    parser.add_argument('--outputobjectivefile',default=None,help="Output file for objectives")
    parser.add_argument('--objectivemethod',default="AP-PB",help="Objective type")
    parser.add_argument('--storage_level', default="MEMORY_ONLY",help='Control Spark caching/persistence behavrior',choices=['MEMORY_ONLY','MEMORY_AND_DISK','DISK_ONLY'])
    parser.add_argument('--inputconstraintfile',default=None,help ="Input file for constraints. If not given, constraints are generated and stored in file named as specified by ---constrainfile")
    parser.add_argument('--checkpointdir',default='checkpointdir',type=str,help='Directory to be used for checkpointing')
    parser.add_argument('--logfile',default='preprocessor.log',help='Log file')

    parser.add_argument('--equalize',action='store_true', help='If passed makes sure that the graphs have the same number of nodes by randomly ignoring some  nodes from the larger graph.')


    dumpgroup = parser.add_mutually_exclusive_group(required=False)
    dumpgroup.add_argument('--driverdump',dest='driverdump',action='store_true', help='Collect output and dump it from driver')
    dumpgroup.add_argument('--slavedump',dest='driverdump',action='store_false', help='Dump output directly from slaves')
    parser.set_defaults(driverdump=False)



    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=False)

    dirgroup = parser.add_mutually_exclusive_group(required=False)
    dirgroup.add_argument('--undirected', dest='undirected', action='store_true',help='Treat inputs as undirected graphs; this is the default behavior.')
    dirgroup.add_argument('--directed', dest='undirected', action='store_false',help='Treat inputs as directed graphs. Edge (i,j) does not imply existence of (j,i).')
    parser.set_defaults(undirected=True)





    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
   # sc.setCheckpointDir(args.checkpointdir)
 

    level = "logging."+args.debug
   
 
    logger = logging.getLogger('Preprocessor')
    logger.setLevel(eval(level))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval(level))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)	
   
    DEBUG = logger.getEffectiveLevel()==logging.DEBUG
    logger.info('Level set to: '+str(level)) 
    logger.info('Debug mode is on? '+str(DEBUG)) 

    if not DEBUG:
        sc.setLogLevel("ERROR")	
  
 

   
    storage_level=eval("StorageLevel."+args.storage_level)

    logger.info("Starting with arguments: "+str(args))

    #Read Graphs		
    logger.info('Read Graphs')
    if args.fromsnap:
        graph1 = readSnap(args.graph1,sc,minPartitions=args.N)
        graph2 = readSnap(args.graph2,sc,minPartitions=args.N)
        
    else:
        graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
        graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)


 
    #make sure that graphs have the same node numbers
    if args.equalize:
        #Add dummy nodes to the smaller graph (i.e., the graph with lower number of nodes).
        n1 = graph1.flatMap(lambda (u,v):[u,v]).distinct().count()
        n2 = graph2.flatMap(lambda (u,v):[u,v]).distinct().count()

        n_max = max(n1, n2)
        n_min = min(n1, n2)


    #    dummyNodes = [(str(dum), str(node_i) ) for dum in range(n_min, n_max) for node_i in range(n_max)]
        dummyNodes = [(str(dum), str(dum) ) for dum in range(n_min, n_max)]
        dummyNodes = sc.parallelize(dummyNodes).partitionBy(args.N)
        
        if n1<n2:
           # graph1 = graph1.filter(lambda (u,v): eval(u)<n2 and eval(v)<n2).cache()
            graph1 = graph1.union(dummyNodes).partitionBy(args.N).cache()
        else:
            graph2 = graph2.union(dummyNodes).partitionBy(args.N).cache()
           # graph2  = graph2.filter(lambda (u,v): eval(u)<n1 and eval(v)<n1).cache()
    
     #repeat edges if graph is undirected
    if args.undirected:
        graph1 = graph1.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()
        graph2 = graph2.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()

    numb_Nodes1 =  graph1.flatMap(lambda (u,v):[u,v]).distinct().count()#, graph2.flatMap(lambda (u,v):[u,v]).distinct().count()
    numb_Nodes2 =  graph2.flatMap(lambda (u,v):[u,v]).distinct().count()
	
    logger.info("The graphs sizes are %d and %d" %(numb_Nodes1, numb_Nodes2))

    #Generate/Read Constraints
    if  not args.inputconstraintfile:
        logger.info('Generate  constraints')
        if args.constraintmethod == 'all':
            G = cartesianProduct(graph1,graph2).persist(storage_level)
        elif args.constraintmethod == 'degree':
            offset = args.degreedistance
      #      while True:
      #          logger.info("Running for offset %d" %offset)    
      #          degree1=degrees(graph1,offset=offset,numPartitions=args.N).persist(storage_level)
      #          degree2=degrees(graph2,offset=offset,numPartitions=args.N).persist(storage_level)
      #          G = matchColors(degree1,degree2,numPartitions=args.N).persist(storage_level)	
      #          
      #         
      #          G_collected = G.collect()
      #          bipartG = nx.Graph()                
      #          for (i,j) in G_collected:
      #              bipartG.add_edge("GA"+i, "GB"+j)
      #          maxG = mm.maximal_matching(bipartG)
      #          coveredNodes = len(maxG)
      #          logger.info("Number of the covered nodes is:  %d" %coveredNodes)
      #          if numb_Nodes1 == coveredNodes:
      #              break
      #          offset = offset + 1
            degree1=degrees(graph1,offset=offset,numPartitions=args.N).persist(storage_level)
            degree2=degrees(graph2,offset=offset,numPartitions=args.N).persist(storage_level)
            G = matchColors(degree1,degree2,numPartitions=args.N).persist(storage_level)  
               
                
               
                
	    #G.checkpoint()
        elif args.constraintmethod == 'WL':
	    color1 = WL(graph1,logger,depth=args.k,numPartitions=args.N) 
	    color2 = WL(graph2,logger,depth=args.k,numPartitions=args.N) 
	    G = matchColors(color1,color2, numPartitions=args.N).persist(storage_level) 	
        elif args.constraintmethod == 'neighbor':
            neighbors1 = get_neighborhood(graph1, args.N, args.k).mapValues(lambda l:hash( tuple(l) ) )
            neighbors2 = get_neighborhood(graph2, args.N, args.k).mapValues(lambda l:hash( tuple(l) ) )
            G = matchColors(neighbors1, neighbors2, numPartitions=args.N).persist(storage_level)
        logger.info('Number of constraints is %d' %G.count())
        if args.equalize:
            idnetityMap  = [(str(node), str(node)) for node  in range(n_max)]
            idnetityMap = sc.parallelize(idnetityMap).partitionBy(args.N).cache()
            G = G.union(idnetityMap).distinct()
        if args.outputconstraintfile:
            logger.info('Write  constraints')
            logger.info('Number of constraints is %d' %G.count())
            safeWrite(G, args.outputconstraintfile, args.driverdump)
            #safeWrite(G,args.outputconstraintfile,args.driverdump)
    else:
        logger.info('Read  constraints')
        G=sc.textFile(args.inputconstraintfile,minPartitions=args.N).map(eval)

    if args.outputobjectivefile:
        logger.info('Generate objectives')
	objectives = SijGenerator(graph1,graph2,G,args.N)
        logger.info('Number of objectives is %d' %objectives.count() )
        logger.info('Write  objectives')
        safeWrite(objectives,args.outputobjectivefile,args.driverdump)
    
    logger.info('...done')
 
