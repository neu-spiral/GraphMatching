import numpy as np
import sys,argparse,logging,datetime,pickle,time
from pyspark import SparkContext,StorageLevel,SparkConf
from operator import add,and_
from LocalSolvers import LocalL1Solver,LocalL2Solver,FastLocalL2Solver,SijGenerator,LocalRowProjectionSolver,LocalColumnProjectionSolver,LocalLSSolver, LocalL1Solver_Old
from ParallelSolvers import ParallelSolver, ParallelSolver1norm, ParallelSolverPnorm, ParallelSolver2norm
from helpers import swap,clearFile,identityHash,pretty,projectToPositiveSimplex,mergedicts,safeWrite,NoneToZero, adaptRho
from helpers_GCP import safeWrite_GCP,upload_blob, download_blob
from debug import logger,dumpPPhiRDD,dumpBasic
from pprint import pformat
import os
import shutil
def testPositivity(rdd):
    '''Test whether vals are positive
    '''
    def fun_test_positivity(val):
        if val>=0.0:
            out = 1.
        else:
            out = 0.
        return (out, 1.)
    val,count= rdd.map(lambda ((i,j),val): fun_test_positivity(val)).reduce(lambda (val1,ct1),(val2,ct2) : (val1+val2,ct1+ct2)  )
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
    if len(cls_args) == 2:
        print "NOT has the lin"
        return solvers_cls(cls_args[0], cls_args[1]), P_vals, Phi_vals, stats
    elif len(cls_args) == 4:
        print "has the lin"
        return solvers_cls(cls_args[0], cls_args[1], cls_args[2], cls_args[3]), P_vals, Phi_vals, stats
def evalSolversY(cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats, dumped_cls, rho_inner):
    solvers_cls = pickle.loads(dumped_cls)
    return solvers_cls(cls_args[0], cls_args[1], rho_inner), P_vals, Y_vals, Phi_vals, Upsilon_vals, stats
    


if __name__=="__main__":
	

    start_timing = time.time()
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('constraintfile',default=None,help ="File containing graph of constraints. ") 
    parser.add_argument('outputfile',help = 'Output file storing the trace, which includes PR, DR, and OBJ.')
    parser.add_argument('outputfileZRDD',help = 'Output file storing learned doubly stochastic matrix Z')
    parser.add_argument('--graph1',default=None,help = 'File containing first graph (optional).')
    parser.add_argument('--graph2',default=None,help = 'File containing second graph (optional).')
    parser.add_argument('--objectivefile',default=None,help="File containing pre-computed objectives. Graphs  need not be given if this argument is set.")
    parser.add_argument('--problemsize',default=1000,type=int, help='Problem size. Used to initialize uniform allocation, needed when objectivefile is passed')
    parser.add_argument('--parallelSolver',default='ParallelSolver', choices=['ParallelSolver', 'ParallelSolver1norm', 'ParallelSolverPnorm', 'ParallelSolver2norm'],help='Parallel Solver')
    parser.add_argument('--solver',default='LocalLSSolver', help='Local Solver',choices=['LocalL1Solver','LocalL2Solver','FastLocalL2Solver','LocalColumnProjectionSolver','LocalRowProjectionSolver','LocalLSSolver','LocalL1Solver_Old'])
    parser.add_argument('--debug',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--logLevel',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--logfile',default='graphmatching.log',help='Log file')
    parser.add_argument('--maxiter',default=5,type=int, help='Maximum number of iterations')
    parser.add_argument('--maxInnerADMMiter',default=40,type=int, help='Maximum number of Inner ADMM iterations')
    parser.add_argument('--N',default=8,type=int, help='Number of partitions')
    parser.add_argument('--Nrowcol',default=1,type=int, help='Level of parallelism for Row/Col RDDs')
    parser.add_argument('--p', default=1.5, type=float, help='p parameter in p-norm')
    parser.add_argument('--distfile',type=str,help='File that stores distances the distance matrix D.', default=None)
    
    parser.add_argument('--rhoP',default=1.0,type=float, help='Rho value, used for primal variables P')
    parser.add_argument('--rhoQ',default=1.0,type=float, help='Rho value, used for primal variables Q')
    parser.add_argument('--rhoT',default=1.0,type=float, help='Rho value, used for primal variables T')
    parser.add_argument('--rho_inner',default=1.0,type=float, help='Rho paramter for Inner ADMM')
    parser.add_argument('--alpha',default=0.05,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--lambda_linear',default=1.0,type=float, help='Regularization parameter for linear term.')
    parser.add_argument('--dump_trace_freq',default=10,type=int,help='Number of iterations between trace dumps')
    parser.add_argument('--checkpoint_freq',default=10,type=int,help='Number of iterations between check points')
    parser.add_argument('--checkpointdir',default='checkpointdir',type=str,help='Directory to be used for checkpointing')
    parser.add_argument('--initRDD',default=None, type=str, help='File name, where the RDDs are dumped.')
    parser.add_argument('--GCP',action='store_true', help='Pass if running on  Google Cloud Platform')
    parser.add_argument('--hasLinear',action='store_true', help='Pass if adding the linear term.')
    parser.add_argument('--bucketname',type=str,default='armin-bucket',help='Bucket name for storing RDDs on Google Cloud Platform, pass if running on GCP')

    parser.add_argument('--dumpRDDs', dest='dumpRDDs', action='store_true',help='Dump auxiliary RDDs beyond Z')
    parser.add_argument('--no-dumpRDDs', dest='dumpRDDs', action='store_false',help='Do not dump auxiliary RDDs beyond Z')
    parser.set_defaults(dumpRDDs=True)

	
    parser.set_defaults(silent=False)
    parser.add_argument('--silent',dest="silent",action='store_true',help='Run in efficient, silent mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. Overwrites verbosity level to ERROR')

    parser.set_defaults(lean=False)
    parser.add_argument('--lean',dest="lean",action='store_true',help='Run in efficient, ``lean'' mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. It is the same as --silent, though it still prints some basic output messages, and does not effect verbosity level.')

    parser.add_argument('--leanInner',dest="leanInner",action='store_true',help='Run in efficient, ``lean'' mode, with final P after a fixed number of Inner ADMM iterations. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress in Inrre ADMM steps.')
    parser.set_defaults(leanInner=False)
   

    parser.set_defaults(directed=True)
    parser.add_argument('--directed', dest='directed', action='store_true',help='Input graphs are directed, i.e., (a,b) does not imply the presense of (b,a).')

    parser.set_defaults(driverdump=False)
    parser.add_argument('--driverdump',dest='driverdump',action='store_true', help='Dump final output Z after first collecting it at the driver, as opposed to directly from workers.')

    parser.set_defaults(adaptrho=False)
    parser.add_argument('--adaptrho',dest='adaptrho',action='store_true', help='Adapt the rho parameter throught ADMM iterations.')

    args = parser.parse_args()

    SolverClass = eval(args.solver)	
    ParallelSolverClass = eval(args.parallelSolver)

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Matching with  %s  at %d iterations over %d partitions' % (args.solver,args.maxiter,args.N),conf=configuration)
    #Setting checkpoint dir is nor reuired for localCheckpoint!?
    sc.setCheckpointDir(args.checkpointdir)
    


    level = "logging."+args.debug
    if args.silent:
	level = "logging.ERROR"

    logger.setLevel(eval("logging."+args.logLevel))
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

    has_linear = args.distfile  is not None or args.hasLinear
    if not has_linear:
        oldLinObjective = 0.

    if args.distfile is not None:
        D =  sc.textFile(args.distfile).map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
    else:
        D = None
    
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



    if args.initRDD != None:


        if args.GCP:
            #Download the previous trace file from the bucket
            outfile_name = 'profiling/GM/' + args.outputfile.split('/')[-1] 
            download_blob(args.bucketname, outfile_name,args.outputfile )
        #Load the previous trace file
        fTrace = open(args.outputfile, 'r')
        (prevArgs, trace) = pickle.load(fTrace)
        fTrace.close()
       # numb_of_prev_iters = len(trace)
        numb_of_prev_iters = max(trace.keys())+1

       #Resume iterations from prevsiously dumped iterations. 
        ZRDD = sc.textFile(args.initRDD+"_ZRDD").map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        if ParallelSolverClass == ParallelSolver:
            PPhi_RDD = sc.textFile(args.initRDD+"_PPhiRDD").map(eval).partitionBy(args.N, partitionFunc=identityHash).mapValues(lambda (cls_args, P_vals, Phi_vals, stats): evalSolvers(cls_args, P_vals, Phi_vals, stats, pickle.dumps(SolverClass))).persist(StorageLevel.MEMORY_ONLY)
            PPhi = ParallelSolver(LocalSolverClass=SolverClass, data=objectives, initvalue=uniformweight, N=args.N, rho=args.rhoP, lean=args.lean,silent=args.silent, RDD=PPhi_RDD)
        else:
            PPhi_RDD = sc.textFile(args.initRDD+"_PPhiRDD").map(eval).partitionBy(args.N, partitionFunc=identityHash).mapValues(lambda (cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats): evalSolversY(cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats, pickle.dumps(SolverClass), args.rho_inner)).persist(StorageLevel.MEMORY_ONLY)
            PPhi = ParallelSolverClass(LocalSolverClass=SolverClass, data=objectives, initvalue=uniformweight, N=args.N, rho=args.rhoP, p=args.p, rho_inner=args.rho_inner, lean=args.leanInner, silent=args.silent, RDD=PPhi_RDD)
        logger.info('From the last iteration solver (P/Phi) RDD stats: '+PPhi.logstats() )

        QXi_RDD = sc.textFile(args.initRDD+"_QXiRDD").map(eval).partitionBy(args.N, partitionFunc=identityHash).mapValues(lambda (cls_args, P_vals, Phi_vals, stats): evalSolvers(cls_args, P_vals, Phi_vals, stats, pickle.dumps(LocalRowProjectionSolver))).persist(StorageLevel.MEMORY_ONLY)
        QXi = ParallelSolver(LocalSolverClass=LocalRowProjectionSolver, data=G, initvalue=uniformweight, N=args.Nrowcol, rho=args.rhoQ, D=D, lambda_linear=args.lambda_linear, lean=args.lean, silent=args.silent, RDD=QXi_RDD)
        logger.info('From the last iteration row (Q/Xi) RDD stats: '+QXi.logstats() )

        TPsi_RDD = sc.textFile(args.initRDD+"_TPsiRDD").map(eval).partitionBy(args.N, partitionFunc=identityHash).mapValues(lambda (cls_args, P_vals, Phi_vals, stats): evalSolvers(cls_args, P_vals, Phi_vals, stats, pickle.dumps(LocalColumnProjectionSolver))).persist(StorageLevel.MEMORY_ONLY)
        TPsi = ParallelSolver(LocalSolverClass=LocalColumnProjectionSolver, data=G, initvalue=uniformweight, N=args.Nrowcol, rho=args.rhoT, D=D, lambda_linear=args.lambda_linear, lean=args.lean, silent=args.silent, RDD=TPsi_RDD)

       
        logger.info('From the last iteration column (T/Psi) RDD stats: '+TPsi.logstats() )


    else:

       #Create primal and dual variables and associated solvers
        if ParallelSolverClass == ParallelSolver:
             PPhi = ParallelSolverClass(LocalSolverClass=SolverClass, data=objectives, initvalue=uniformweight, N=args.N, rho=args.rhoP, lean=args.lean,silent=args.silent)
        else:
             PPhi = ParallelSolverClass(LocalSolverClass=SolverClass, data=objectives, initvalue=uniformweight, N=args.N, rho=args.rhoP, p=args.p, rho_inner=args.rho_inner,lean=args.leanInner, silent=args.silent)
        logger.info('Partitioned data (P/Phi) RDD stats: '+PPhi.logstats() )    
      
        QXi = ParallelSolver(LocalSolverClass=LocalRowProjectionSolver, data=G, initvalue=uniformweight, N=args.Nrowcol, rho=args.rhoQ, D=D, lambda_linear=args.lambda_linear, lean=args.lean, silent=args.silent)
        logger.info('Row RDD (Q/Xi) RDD stats: '+QXi.logstats() )

        TPsi = ParallelSolver(LocalSolverClass=LocalColumnProjectionSolver, data=G, initvalue=uniformweight, N=args.Nrowcol, rho=args.rhoT, D=D, lambda_linear=args.lambda_linear, lean=args.lean, silent=args.silent)
        logger.info('Column RDD (T/Psi) RDD stats: '+TPsi.logstats() )


        #Create consensus variable, initialized to uniform assignment ignoring constraints
        ZRDD = G.map(lambda (i,j):((i,j),uniformweight)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        
        #Initialize trace to an empty dict
        trace = {}
        numb_of_prev_iters = 0
    	
    end_timing = time.time()
    logger.info("Data input and preprocessing done in %f seconds, starting main ADMM iterations " % ( end_timing -start_timing))

   #initialize rho parameters (if adaptrho not passed these value stay fixed).
    rhoP = args.rhoP
    rhoQ = args.rhoQ
    rhoT = args.rhoT
    start_timing = time.time()	
    last_time = start_timing
    dump_time = 0.
    for iteration in range(args.maxiter):
        #checkpoint he RDDs preiodically 
        chckpnt = (iteration!=0 and iteration % args.checkpoint_freq==0)
        #in Silent mode forceComp will force computation of the stats, e.g., in the last iteration
        forceComp = iteration==args.maxiter-1

        if not args.silent or forceComp:
            (oldPrimalResidualQ,oldObjQ)=QXi.joinAndAdapt(ZRDD, args.alpha, rhoQ, checkpoint=chckpnt, forceComp=forceComp)
            logger.info("Iteration %d row (Q/Xi) stats: %s" % (iteration,QXi.logstats())) 
            (oldPrimalResidualT,oldObjT)=TPsi.joinAndAdapt(ZRDD, args.alpha, rhoT, checkpoint=chckpnt, forceComp=forceComp)
            logger.info("Iteration %d column (T/Psi) stats: %s" % (iteration,TPsi.logstats())) 
            if ParallelSolverClass == ParallelSolver:
                (oldPrimalResidualP,oldObjP)=PPhi.joinAndAdapt(ZRDD, args.alpha, rhoP, checkpoint=chckpnt, forceComp=forceComp)
            else:
                (oldPrimalResidualP,oldObjP)=PPhi.joinAndAdapt(ZRDD, args.alpha, rhoP, checkpoint=chckpnt, residual_tol=1.e-06, logger=logger, maxiters=args.maxInnerADMMiter, forceComp=forceComp)
            logger.info("Iteration %d solver (P/Phi) stats: %s" % (iteration,PPhi.logstats())) 
        else:
            QXi.joinAndAdapt(ZRDD, args.alpha, rhoQ, checkpoint=chckpnt, forceComp=forceComp)
            TPsi.joinAndAdapt(ZRDD, args.alpha, rhoT, checkpoint=chckpnt, forceComp=forceComp)
            if ParallelSolverClass == ParallelSolver:
                PPhi.joinAndAdapt(ZRDD, args.alpha, rhoP, checkpoint=chckpnt, forceComp=forceComp)
            else:
                PPhi.joinAndAdapt(ZRDD, args.alpha, rhoP, checkpoint=chckpnt, residual_tol=1.e-02, logger=logger, maxiters=args.maxInnerADMMiter, forceComp=forceComp)

       #Check row/col sums:
        #QRDD = QXi.PrimalDualRDD.flatMapValues(lambda (solver, Primal, Dual, stats): [(key, Primal[key]) for key in Primal] ).values()
        #Qsums = tuple(testSimplexCondition(QRDD) )
        #logger.info("Iteration %d Q row sums are: Min %s Max %s " % ((iteration,)+ Qsums ) )
        #logger.info("Iteration %d Q posivity is %f" %(iteration, testPositivity(QRDD) ) )
       ##
        #TRDD = TPsi.PrimalDualRDD.flatMapValues(lambda (solver, Primal, Dual, stats): [(swap(key), Primal[key]) for key in Primal] ).values()
        #Tsums = tuple(testSimplexCondition(TRDD) )
        #logger.info("Iteration %d T col sums are: Min %s Max %s " % ((iteration,)+ Tsums ) )
        #logger.info("Iteration %d T posivity is %f" %(iteration, testPositivity(TRDD) ) )
        

      
	oldZ = ZRDD
	rowvars = QXi.getVars(rhoQ)
        columnvars = TPsi.getVars(rhoT)
	localvars = PPhi.getVars(rhoP)
	allvars = localvars.union(rowvars).union(columnvars)

	if DEBUG:
	   logger.debug("Iteration %d all var pairs is:\n%s" %(iteration,pformat(list(allvars.sortByKey().collect()),width=30)) )
	
	ZRDD = allvars.reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)  ).mapValues(lambda (value,count): 1.0*value/count).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
       #Maybe this is more efficient!
      #  ZRDD = allvars.partitionBy(args.N).reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)).mapValues(lambda (value,count): 1.0*value/count).persist(StorageLevel.MEMORY_ONLY)
	if chckpnt:
	    ZRDD.localCheckpoint()
	
	if DEBUG:
	   logger.debug("Iteration %d Z is:\n%s" %(iteration,pformat(list(ZRDD.collect()),width=30)) )

       	Zstats ={}
	if (not (args.silent or args.lean)) or (args.silent and forceComp):


           #Z feasibility
	   Zstats['POS'] = testPositivity(ZRDD)
	   Zstats['RSUMS'] = tuple(testSimplexCondition(ZRDD))
	   Zstats['CSUMS'] = tuple(testSimplexCondition(ZRDD,dir='column'))

           #Evaluate dual residuals:
           ZRDDjoinedOldZRDD = ZRDD.join(oldZ,numPartitions=args.N).cache()
           dualresidualP = rhoP * PPhi.computeDualResidual(ZRDDjoinedOldZRDD)
           dualresidualQ = rhoQ * QXi.computeDualResidual(ZRDDjoinedOldZRDD)
           dualresidualT = rhoT * TPsi.computeDualResidual(ZRDDjoinedOldZRDD)
           dualresidual = dualresidualP+dualresidualQ+dualresidualT

           #Adapt rhos
           if args.adaptrho:
               rhoP = adaptRho(rhoP, oldPrimalResidualP, dualresidualP)
               rhoQ = adaptRho(rhoQ, oldPrimalResidualQ, dualresidualQ)
               rhoT = adaptRho(rhoT, oldPrimalResidualT, dualresidualT)

           if has_linear:
               oldLinObjective = oldObjQ 

	   #dualresidual = np.sqrt(ZRDD.join(oldZ,numPartitions=args.N).values().map(lambda (v1,v2):(v1-v2)**2).reduce(add)) 
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
             
  
	if not args.silent or forceComp: #under "lean", still output some basic stats. Also output information for the last iteration. 
	   now = time.time() 
           Zstats['TIME'] = now-start_timing-dump_time
           Zstats['IT_TIME'] = now-last_time-dump_time
	   last_time = now
           logger.info("Iteration %d  time is %f sec, average time per iteration is %f sec, total time is %f " % (iteration,Zstats['IT_TIME'],Zstats['TIME']/(iteration+1.0),Zstats['TIME'])) 
	   trace[iteration + numb_of_prev_iters] = Zstats
	
        #if not (args.silent or args.lean):
        dump_time = 0.
        if not (args.silent):
	    if iteration % args.dump_trace_freq == 1 or iteration == args.maxiter-1:
                dump_st_time = time.time()
		
                if args.dumpRDDs:
                    with open(args.outputfile,'wb') as f:
                        pickle.dump((args,trace),f)
                    if not args.GCP:
                        #If not running on GCP, save the RDDs and the trace. 
                        safeWrite(ZRDD,args.outputfileZRDD+"_ZRDD",args.driverdump)
	                safeWrite(PPhi.PrimalDualRDD,args.outputfileZRDD+"_PPhiRDD" ,args.driverdump)
		        safeWrite(QXi.PrimalDualRDD,args.outputfileZRDD+"_QXiRDD",args.driverdump)
		        safeWrite(TPsi.PrimalDualRDD,args.outputfileZRDD+"_TPsiRDD",args.driverdump)
                    else:
                        #If running on GCP, upload the log and the trace to the bucket. Also, save RDDs on the bucket. 
                        outfile_name = 'profiling/GM/' + args.outputfile.split('/')[-1]
                        logfile_name = 'profiling/GM/' + args.logfile.split('/')[-1]

                        upload_blob(args.bucketname, args.outputfile, outfile_name)
                        upload_blob(args.bucketname, args.logfile, logfile_name)
                        safeWrite_GCP(ZRDD,args.outputfileZRDD+"_ZRDD",args.bucketname)
                        safeWrite_GCP(PPhi.PrimalDualRDD,args.outputfileZRDD+"_PPhiRDD",args.bucketname)
                        safeWrite_GCP(QXi.PrimalDualRDD,args.outputfileZRDD+"_QXiRDD",args.bucketname)
                        safeWrite_GCP(TPsi.PrimalDualRDD,args.outputfileZRDD+"_TPsiRDD",args.bucketname)
            
		#log.info("ZRDD is "+str(ZRDD.collect()))
                dump_end_time = time.time()
                dump_time = dump_end_time - dump_st_time
      
	#oldZ.unpersist()
	
     
    end_timing = time.time()
    logger.info("Finished ADMM iterations in %f seconds." % (end_timing-start_timing))


    

    
