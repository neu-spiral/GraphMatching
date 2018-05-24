import time
import argparse
from LocalSolvers import LocalL1Solver, LocalRowProjectionSolver, LocalL1Solver_Old
from ParallelSolvers import ParallelSolver
from pyspark import SparkContext, StorageLevel
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching Test.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data',type=str,help ="File containing data, either constraints or objectives.")
    parser.add_argument('--G', type=str,help="File containing the variables.")
    parser.add_argument('--solver',default='LocalL1Solver', help='Local Solver')
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--alpha',default=1.0,type=float, help='Alpha value, used for dual variables')
    parser.set_defaults(undirected=True)

    args = parser.parse_args()




    sc = SparkContext()
    sc.setLogLevel("OFF")
    SolverClass = eval(args.solver)
    N = args.N
    uniformweight = 1/2000.
    alpha = args.alpha
    rho = args.rho

    data = sc.textFile(args.data).map(lambda x:eval(x)).partitionBy(N).persist(StorageLevel.MEMORY_ONLY)
 
    
    G = sc.textFile(args.G).map(eval)
  
  
    tstart = time.time()
    #Initiate the ParallelSolver object
    RDDSolver_cls = ParallelSolver(LocalSolverClass=SolverClass, data=data, initvalue=uniformweight, N=N, rho=rho)


    #Create consensus variable, initialized to uniform assignment ignoring constraints
    ZRDD = G.map(lambda var:(var,uniformweight)).partitionBy(N).persist(StorageLevel.MEMORY_ONLY)

    print "Iinitial row (Q/Xi) stats: %s" %RDDSolver_cls.logstats()

    for i in range(20):
        print RDDSolver_cls.PrimalDualRDD.take(1)
        (oldPrimalResidualQ,oldObjQ) = RDDSolver_cls.joinAndAdapt(ZRDD, alpha, rho)

        allvars = RDDSolver_cls.getVars(rho)

        ZRDD = allvars.reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)  ).mapValues(lambda (value,count): 1.0*value/count).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        print "Iteration %d row (Q/Xi) stats: %s" % (i,RDDSolver_cls.logstats())

       
    tend = time.time()
     
