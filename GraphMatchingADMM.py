import numpy as np
import sys,argparse,logging,datetime,pickle,time
from pyspark import SparkContext,StorageLevel,SparkConf
from operator import add,and_
from LocalSolvers import LocalL1Solver,LocalL2Solver,FastLocalL2Solver,SijGenerator,LocalRowProjectionSolver,LocalColumnProjectionSolver
from ParallelSolvers import ParallelSolver
from helpers import swap,clearFile,identityHash,pretty,projectToPositiveSimplex,mergedicts,safeWrite,NoneToZero
from debug import logger,dumpPPhiRDD,dumpBasic
from pprint import pformat
import os
import shutil
def testPositivity(rdd):
    '''Test whether vals are positive
    '''
    val,count= rdd.map(lambda ((i,j),val): (val>=0.0,1 )).reduce(lambda (val1,ct1),(val2,ct2) : (val1+val2,ct1+ct2)  )
    return val/count


def testSimplexCondition(rdd,dir='row'):
    '''Tests whether the row condition holds
    '''
    if dir=='row':
        sums=rdd.map(lambda ((i,j),val): (i,val) ).reduceByKey(add).values()
    elif dir=='column':
        sums=rdd.map(lambda ((i,j),val): (j,val) ).reduceByKey(add).values()
    minsum = sums.reduce(min)
    maxsum = sums.reduce(max)
    return minsum, maxsum

def evalSolvers(cls_args, P_vals, Phi_vals, stats, dumped_cls):
    solvers_cls = pickle.loads(dumped_cls)
    return solvers_cls(cls_args[0], cls_args[1]), P_vals, Phi_vals, stats
    


if __name__=="__main__":
	

    start_timing = time.time()
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('constraintfile',default=None,help ="File containing graph of constraints. ") 
    parser.add_argument('outputfile',help = 'Output file storing the trace, which includes PR, DR, and OBJ.')
    parser.add_argument('outputfileZRDD',help = 'Output file storing learned doubly stochastic matrix Z')
    parser.add_argument('--graph1',default=None,help = 'File containing first graph (optional).')
    parser.add_argument('--graph2',default=None,help = 'File containing second graph (optional).')
    parser.add_argument('--objectivefile',default=None,help="File containing pre-computed objectives. Graphs  need not be given if this argument is set.")
    parser.add_argument('--linear_term',default=None,help="Linear term to be added in the objective")
    parser.add_argument('--problemsize',default=1000,type=int, help='Problem size. Used to initialize uniform allocation, needed when objectivefile is passed')
    parser.add_argument('--solver',default='LocalL1Solver', help='Local Solver',choices=['LocalL1Solver','LocalL2Solver','FastLocalL2Solver'])
    parser.add_argument('--debug',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--logfile',default='graphmatching.log',help='Log file')
    parser.add_argument('--maxiter',default=5,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=8,type=int, help='Number of partitions')
    parser.add_argument('--rhoP',default=1.0,type=float, help='Rho value, used for primal variables P')
    parser.add_argument('--rhoQ',default=1.0,type=float, help='Rho value, used for primal variables Q')
    parser.add_argument('--rhoT',default=1.0,type=float, help='Rho value, used for primal variables T')
    parser.add_argument('--alpha',default=0.05,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--dump_trace_freq',default=10,type=int,help='Number of iterations between trace dumps')
    parser.add_argument('--checkpoint_freq',default=10,type=int,help='Number of iterations between check points')
    parser.add_argument('--checkpointdir',default='checkpointdir',type=str,help='Directory to be used for checkpointing')

    parser.add_argument('--dumpRDDs', dest='dumpRDDs', action='store_true',help='Dump auxiliary RDDs beyond Z')
    parser.add_argument('--no-dumpRDDs', dest='dumpRDDs', action='store_false',help='Do not dump auxiliary RDDs beyond Z')
    parser.set_defaults(dumpRDDs=True)

    parser.add_argument('--init',default="", help="Initialization files for RDDs. It is assumed all files have the same name, passed as argument, with a _XXX indicating the RDD in the end, e.g.,filaname_ZRDD for ZRDD.")
	
    parser.set_defaults(silent=False)
    parser.add_argument('--silent',dest="silent",action='store_true',help='Run in efficient, silent mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. Overwrites verbosity level to ERROR')

    parser.set_defaults(lean=False)
    parser.add_argument('--lean',dest="lean",action='store_true',help='Run in efficient, ``lean'' mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. It is the same as --silent, though it still prints some basic output messages, and does not effect verbosity level.')

   

    parser.set_defaults(directed=False)
    parser.add_argument('--directed', dest='directed', action='store_true',help='Input graphs are directed, i.e., (a,b) does not imply the presense of (b,a).')

    parser.set_defaults(driverdump=False)
    parser.add_argument('--driverdump',dest='driverdump',action='store_true', help='Dump final output Z after first collecting it at the driver, as opposed to directly from workers.')


    args = parser.parse_args()

    SolverClass = eval(args.solver)	

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Matching with  %s  at %d iterations over %d partitions' % (args.solver,args.maxiter,args.N),conf=configuration)
    sc.setCheckpointDir(args.checkpointdir)
    


    level = "logging."+args.debug
    if args.silent:
	level = "logging.ERROR"

    logger.setLevel(eval(level))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval(level))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)	
   
    DEBUG = logger.getEffectiveLevel()==logging.DEBUG
    logger.info("Starting with arguments: "+str(args))
    logger.info('Level set to: '+str(level)) 

    if not DEBUG:
        sc.setLogLevel("OFF")

    has_linear = args.linear_term is not None
    if not has_linear:
        oldLinObjective = 0.
    
    #Read constraint graph
    G = sc.textFile(args.constraintfile).map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)




    if not args.objectivefile:
        #Read Graphs		
        graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval).partitionBy(args.N)
        graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval).partitionBy(args.N)

	#Extract nodes 
	nodes1 = graph1.flatMap(lambda (u,v):[u,v]).distinct()
	nodes2 = graph2.flatMap(lambda (u,v):[u,v]).distinct()

	uniformweight= 1.0/nodes2.count()


	#repeat edges if graph is undirected
	if not args.directed:
	    graph1 = graph1.flatMap(lambda (u,v):[ (u,v),(v,u)]).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	    graph2 = graph2.flatMap(lambda (u,v):[ (u,v),(v,u)]).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	    
	if DEBUG:
	    dumpBasic(G,graph1,graph2)

	objectives = SijGenerator(graph1,graph2,G,args.N).persist(StorageLevel.MEMORY_ONLY)

    else:
	objectives = sc.textFile(args.objectivefile).map(lambda x:eval(x)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	uniformweight= 1.0/float(args.problemsize)
	#Create local primal and dual variables  


    #Create primal and dual variables and associated solvers
    PPhi=ParallelSolver(SolverClass,objectives,uniformweight,args.N,args.rhoP,args.alpha,lean=args.lean)
    logger.info('Partitioned data (P/Phi) RDD stats: '+PPhi.logstats() )    
      
    QXi = ParallelSolver(LocalRowProjectionSolver,G,uniformweight,args.N,args.rhoQ,args.alpha,lean=args.lean)
    logger.info('Row RDD (Q/Xi) RDD stats: '+QXi.logstats() )

    TPsi = ParallelSolver(LocalColumnProjectionSolver, G, uniformweight,args.N,args.rhoT,args.alpha,lean=args.lean)
    logger.info('Column RDD (T/Psi) RDD stats: '+TPsi.logstats() )


    #Create consensus variable, initialized to uniform assignment ignoring constraints
    ZRDD = G.map(lambda (i,j):((i,j),uniformweight)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)

    	
    end_timing = time.time()
    logger.info("Data input and preprocessing done in %f seconds, starting main ADMM iterations " % ( end_timing -start_timing))

    start_timing = time.time()	
    last_time = start_timing
    trace = {}
    for iteration in range(args.maxiter):

        (oldPrimalResidualQ,oldObjQ)=QXi.joinAndAdapt(ZRDD, args.alpha, args.rhoQ)
        logger.info("Iteration %d row (Q/Xi) stats: %s" % (iteration,QXi.logstats())) 
        (oldPrimalResidualT,oldObjT)=TPsi.joinAndAdapt(ZRDD, args.alpha, args.rhoT)
        logger.info("Iteration %d column (T/Psi) stats: %s" % (iteration,TPsi.logstats())) 
        (oldPrimalResidualP,oldObjP)=PPhi.joinAndAdapt(ZRDD, args.alpha, args.rhoP)
        logger.info("Iteration %d solver (P/Phi) stats: %s" % (iteration,PPhi.logstats())) 

      
	oldZ = ZRDD
	rowvars = QXi.getVars(args.rhoQ)
        columnvars = TPsi.getVars(args.rhoT)
	localvars = PPhi.getVars(args.rhoP)
	allvars = localvars.union(rowvars).union(columnvars)

	if DEBUG:
	   logger.debug("Iteration %d all var pairs is:\n%s" %(iteration,pformat(list(allvars.sortByKey().collect()),width=30)) )
	
	ZRDD = allvars.reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)  ).mapValues(lambda (value,count): 1.0*value/count).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	if iteration % args.checkpoint_freq == 0 and iteration != 0:
	    ZRDD.checkpoint()
	
	if DEBUG:
	   logger.debug("Iteration %d Z is:\n%s" %(iteration,pformat(list(ZRDD.collect()),width=30)) )

       	Zstats ={}
	if not (args.silent or args.lean):
	
           #Z feasibility
	   Zstats['POS'] = testPositivity(ZRDD)
	   Zstats['RSUMS'] = tuple(testSimplexCondition(ZRDD))
	   Zstats['CSUMS'] = tuple(testSimplexCondition(ZRDD,dir='column'))

           #Evaluate dual residuals:
	   dualresidual = np.sqrt(ZRDD.join(oldZ,numPartitions=args.N).values().map(lambda (v1,v2):(v1-v2)**2).reduce(add)) 
	   Zstats['DRES'] = dualresidual

	   #Store primal residuals:
	   Zstats['PRES'] = oldPrimalResidualP
	   Zstats['QRES'] = oldPrimalResidualQ
	   Zstats['TRES'] = oldPrimalResidualT
           #Z obj
	   Zstats['OLDOBJ'] = oldObjP + oldLinObjective
	   Zstats['OLDNOLIN'] = oldObjP
	   Zstats['OLDLIN'] = oldLinObjective
   

           logger.info("Iteration %d Z positivity is: %s " % (iteration,str(Zstats['POS']*100)+'%')) 
	   logger.info("Iteration %d Z row sums are: Min %s Max %s " % ((iteration,)+ Zstats['RSUMS'] ) )
	   logger.info("Iteration %d Z column sums are: Min %s Max %s " % ((iteration,)+ Zstats['CSUMS']) )
	   logger.info("Iteration %d-1 Z objective value: %s  (= %s + %s) " % (iteration, Zstats['OLDOBJ'],Zstats['OLDNOLIN'],Zstats['OLDLIN']) )
	   logger.info("Iteration %d-1 Z residuals: " % iteration+ "\t".join( [ key+":"+str(Zstats[key])  for key in ['DRES','PRES','QRES','TRES']] ) )
             
  
	if not args.silent: #under "lean", still output some basic stats
	   now = time.time()
           Zstats['TIME'] = now-start_timing
           Zstats['IT_TIME'] = now-last_time
	   last_time = now
           logger.info("Iteration %d  time is %f sec, average time per iteration is %f sec, total time is %f " % (iteration,Zstats['IT_TIME'],Zstats['TIME']/(iteration+1.0),Zstats['TIME'])) 
	   trace[iteration] = Zstats
	
        #if not (args.silent or args.lean):
        if not (args.silent):
	   if iteration % args.dump_trace_freq == 0 and iteration != 0:
                with open(args.outputfile+"_trace",'wb') as f:
            	    pickle.dump((args,trace),f)
    	        safeWrite(ZRDD,args.outputfileZRDD+"_ZRDD_%diterations" %iteration,args.driverdump)
		
                if args.dumpRDDs:
		    safeWrite(PPhiRDD,args.outputfileZRDD+"_PPhiRDD_%diterations" %iteration,args.driverdump)
		    safeWrite(QXiRDD,args.outputfileZRDD+"_QXiRDD_%diterations" %iteration,args.driverdump)
		    safeWrite(TPsiRDD,args.outputfileZRDD+"_TPsiRDD_%diterations" %iteration,args.driverdump)
		#log.info("ZRDD is "+str(ZRDD.collect()))
 
	oldZ.unpersist()
	
     
    end_timing = time.time()
    logger.info("Finished ADMM iterations in %f seconds." % (end_timing-start_timing))

    if not args.silent:
        with open(args.outputfile+"_trace",'wb') as f:
	    pickle.dump((args,trace),f)

    safeWrite(ZRDD,args.outputfileZRDD+"_ZRDD",args.driverdump)
    

    
