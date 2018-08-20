import time
import argparse,logging
from LocalSolvers import LocalL1Solver, LocalRowProjectionSolver, LocalL1Solver_Old, LocalLpSolver
from ParallelSolvers import ParallelSolver, ParallelSolverPnorm
from pyspark import SparkContext, StorageLevel
from debug import logger
from helpers import clearFile
import helpers_GCP
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching Test.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outfile', type=str, help='File to store running ansd time.')
    parser.add_argument('data',type=str,help ="File containing data, either constraints or objectives.")
    parser.add_argument('--G', type=str,help="File containing the variables.")
    parser.add_argument('--solver',default='LocalL1Solver', help='Local Solver')
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--alpha',default=1.0,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--maxiters',default=20, type=int, help='Max iterations to run the algorithm.')
    parser.add_argument('--p', default=1.5, type=float, help='p parameter in p-norm')
    parser.add_argument('--logfile',type=str,help='Log file to keep track of the stats.')
    parser.add_argument('--checkpoint_freq',default=15,type=int,help='Number of iterations between check points')
    parser.add_argument('--checkpointdir',default='checkpointdir',type=str,help='Directory to be used for checkpointing')
    parser.add_argument('--bucket_name',default='armin-bucket',type=str,help='Bucket name, specify when running on google cloud. Outfile and logfile will be uploaded here.')

    parser.add_argument('--GCP',action='store_true',help='Pass if running on Google Cloud Platform.')
    parser.set_defaults(GCP=False)


    parser.set_defaults(undirected=True)

    args = parser.parse_args()


    #Log file 
    level = "logging.INFO"

    logger.setLevel(eval(level))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval(level))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    DEBUG = logger.getEffectiveLevel()==logging.DEBUG
    logger.info("Starting with arguments: "+str(args))
    logger.info('Level set to: '+str(level))



    sc = SparkContext(appName="Parallel Tester for %s using %d partitions" %(args.solver, args.N))
 #   sc.setLogLevel("OFF")
    sc.setCheckpointDir(args.checkpointdir)
    sc.setLogLevel('OFF')


    SolverClass = eval(args.solver)
    N = args.N
    uniformweight = 1/2000.
    alpha = args.alpha
    rho = args.rho
    p = args.p

    data = sc.textFile(args.data).map(lambda x:eval(x)).partitionBy(N).persist(StorageLevel.MEMORY_ONLY)
 
    
    G = sc.textFile(args.G).map(eval)
  
  
    tstart = time.time()
    tlast = tstart
    #Initiate the ParallelSolver object
    if SolverClass == LocalLpSolver:
        RDDSolver_cls = ParallelSolverPnorm(LocalSolverClass=SolverClass, data=data, initvalue=uniformweight*2, N=N, rho=rho, p=p, rho_inner=rho)
    else:
        RDDSolver_cls = ParallelSolverPnorm(LocalSolverClass=SolverClass, data=data, initvalue=uniformweight*2, N=N, rho=rho)


    #Create consensus variable, initialized to uniform assignment ignoring constraints
    ZRDD = G.map(lambda var:(var,uniformweight)).partitionBy(N).persist(StorageLevel.MEMORY_ONLY)

    logger.info("Iinitial row (Q/Xi) stats: %s" %RDDSolver_cls.logstats())

    for i in range(args.maxiters):
        chckpnt = (i!=0 and i % args.checkpoint_freq==0)
        OldZ=ZRDD
        (oldPrimalResidualQ,oldObjQ) = RDDSolver_cls.joinAndAdapt(ZRDD, alpha, rho, checkpoint=chckpnt)

        allvars = RDDSolver_cls.getVars(rho)

        ZRDD = allvars.reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)  ).mapValues(lambda (value,count): 1.0*value/count).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
    #    OldZ.unpersist()
        if chckpnt:
           ZRDD.localCheckpoint()
        #OldZ.unpersist()
        now = time.time()
        logger.info("Iteration %d row (Q/Xi) stats: %s, objective value is %f residual is %f, time is %f, iteration time is %f" % (i,RDDSolver_cls.logstats(),oldObjQ, oldPrimalResidualQ, now-tstart,now-tlast))
	tlast=now
       
    tend = time.time()


    #Write the results to file.
    Num_vars = G.count()
    fP = open(args.outfile, 'w')
    fP.write('(veriables, time)\n')
    fP.write('(%d, %f)' %(Num_vars, tend-tstart))
    fP.close() 
  
    #If running on google cloud upload the outfile and logfile to the bucket
    if args.GCP:
       #File names are specified by 
        outfile_name = "profiling/L1.5/" + args.outfile.split('/')[-1]
        logfile_name = "profiling/L1.5/" + args.logfile.split('/')[-1]
         
        helpers_GCP.upload_blob(args.bucket_name, args.outfile, outfile_name)
        helpers_GCP.upload_blob(args.bucket_name, args.logfile, logfile_name)
    #    helpers_GCP.safeWrite_GCP(ZRDD,args.outfile+"_ZRDD",args.bucket_name)
    #    helpers_GCP.safeWrite_GCP(RDDSolver_cls.PrimalDualRDD,args.outfile+"_PPhiRDD",args.bucket_name)
        
    
     
