import time
import argparse,logging
from LocalSolvers import LocalL1Solver, LocalRowProjectionSolver, LocalL1Solver_Old, LocalLpSolver
from ParallelSolvers import ParallelSolver, ParallelSolverPnorm
from pyspark import SparkContext, StorageLevel
from debug import logger
from helpers import clearFile
#import helpers_GCP
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching Test.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objectives',type=str,help ="File containing data, either constraints or objectives.")
    parser.add_argument('--outfile', type=str, help='File to store running ansd time.')
    parser.add_argument('--G', type=str,help="File containing the variables.")
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--alpha',default=1.0,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--maxiters',default=20, type=int, help='Max iterations to run the algorithm.')

    args = parser.parse_args()

    sc = SparkContext(appName="Inner ADMM Tester for using %d partitions" %args.N)

    uniformweight = 1./args.N
    data = sc.textFile(args.objectives).map(lambda x:eval(x)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)

    RDDSolver_cls = ParallelSolverPnorm(LocalSolverClass=LocalLpSolver, data=data, initvalue=uniformweight*2, N=args.N, rho=args.rho, rho_inner=args.rho, p=1.5)
    print RDDSolver_cls.PrimalDualRDD.take(1)
    
    
