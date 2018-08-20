import time
import argparse,logging
from LocalSolvers import LocalL1Solver, LocalRowProjectionSolver, LocalL1Solver_Old, LocalLpSolver
from ParallelSolvers import ParallelSolver, ParallelSolverPnorm
from pyspark import SparkContext, StorageLevel
from debug import logger
from helpers import clearFile, writeMat2File
import numpy as np
#import helpers_GCP
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching Test.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objectives',type=str,help ="File containing data, either constraints or objectives.")
    parser.add_argument('G', type=str,help="File containing the variables.")
    parser.add_argument('--outfile', type=str, help='File to store running ansd time.')
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--rho_inner',default=1.0,type=float, help='Rho value, used for inner ADMM')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--alpha',default=1.0,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--maxiters',default=20, type=int, help='Max iterations to run the algorithm.')

    args = parser.parse_args()

    sc = SparkContext(appName="Inner ADMM Tester for using %d partitions" %args.N)
    sc.setLogLevel('OFF')

    uniformweight = 1./args.N
    data = sc.textFile(args.objectives).map(lambda x:eval(x)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)

    RDDSolver_cls = ParallelSolverPnorm(LocalSolverClass=LocalLpSolver, data=data, initvalue=uniformweight, N=args.N, rho=args.rho, rho_inner=args.rho_inner, p=1.5)

    G = sc.textFile(args.G).map(eval)
    #Create consensus variable, initialized to uniform assignment ignoring constraints
    ZRDD = G.map(lambda var:(var,0.3)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
    
    (splitID, (solver_args, P, Y, Phi, Upsilon, stats) ) = RDDSolver_cls.PrimalDualRDD.collect()[0]
    Z = dict( ZRDD.collect() )

  
    local_solver_cls = LocalLpSolver(dict( data.collect() ), args.rho, args.rho_inner)
    
    D_mat = local_solver_cls.D
    
    (pi, ni) = D_mat.shape
    P_vec = np.matrix( np.zeros((ni,1)))
    Phi_vec =  np.matrix( np.zeros((ni,1)))
    Z_vec =  np.matrix( np.zeros((ni,1)))
    for ind in range(ni):
        P_vec[ind] = P[ local_solver_cls.translate_coordinates2ij[ind]] 
        Phi_vec[ind] = Phi[ local_solver_cls.translate_coordinates2ij[ind]]
        Z_vec[ind] = Z[ local_solver_cls.translate_coordinates2ij[ind]]
    writeMat2File('data/D', D_mat)
    writeMat2File('data/Zbar', Z_vec-Phi_vec)


    RDDSolver_cls.joinAndAdapt(ZRDD = ZRDD, alpha = args.alpha, rho = args.rho, maxiters=args.maxiters)

    (splitID, (solver_args, P, Y, Phi, Upsilon, stats) ) = RDDSolver_cls.PrimalDualRDD.collect()[0]
    Y_vec  = np.matrix( np.zeros((pi,1)))
    Upsilon_vec =  np.matrix( np.zeros((pi,1)))
    for ind in range(pi):
        Y_vec[ind] = Y[local_solver_cls.translate_coordinates2ij_Y[ind]]
        Upsilon_vec[ind] = args.rho_inner * Upsilon[local_solver_cls.translate_coordinates2ij_Y[ind]]
    for ind in range(ni):
        P_vec[ind] = P[ local_solver_cls.translate_coordinates2ij[ind]]
        Phi_vec[ind] = Phi[ local_solver_cls.translate_coordinates2ij[ind]]

    print Upsilon_vec
    Grad_P = args.rho * (P_vec - (Z_vec-Phi_vec)) - D_mat.T * Upsilon_vec
    print Y_vec - D_mat * P_vec
    print Grad_P


    
    
