import time
import pickle
import argparse,logging
from LocalSolvers import LocalL1Solver, LocalRowProjectionSolver, LocalL1Solver_Old, LocalLSSolver
from ParallelSolvers import ParallelSolver, ParallelSolverPnorm, ParallelSolver1norm, ParallelSolver2norm
from pyspark import SparkContext, StorageLevel
from debug import logger
from helpers import clearFile, writeMat2File, identityHash
import numpy as np
from debug import logger 
from helpers_GCP import upload_blob, safeWrite_GCP, download_blob
def norm_p(Y, p):
    "Compute p-norm of the vector Y"
    return ( sum([abs(float(y))**p for y in Y]))**(1./p)


def evalSolversY(cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats, dumped_cls, rho_inner):
    solvers_cls = pickle.loads(dumped_cls)
    return solvers_cls(cls_args[0], cls_args[1], rho_inner), P_vals, Y_vals, Phi_vals, Upsilon_vals, stats
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching Test.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objectives',type=str,help ="File containing data, either constraints or objectives.")
    parser.add_argument('G', type=str,help="File containing the variables.")
    parser.add_argument('--outfile', type=str, help='File to store running ansd time.')
    parser.add_argument('--logfile',default='graphmatching.log',help='Log file')
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--rho_inner',default=1.0,type=float, help='Rho value, used for inner ADMM')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--alpha',default=0.0,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--maxiters',default=40, type=int, help='Max iterations to run the algorithm.')
    parser.add_argument('--ParallelSolver', default='ParallelSolverPnorm', type=str, help='Parallel solver class')
    parser.add_argument('--p', default=1.5,type=float,help='p in p-norm')
    parser.add_argument('--initfile', default=None, type=str, help='File that stores the prev. values of P and Y variables.')
    parser.add_argument('--tracefile', default=None, type=str, help="File storing trace")
    parser.add_argument('--RDDfile',  type=str, help='File to store the values of P and Y variables.')
     
    parser.add_argument('--bucketname',type=str,default='armin-bucket',help='Bucket name for storing RDDs omn Google Cloud Platform, pass if running pn the platform')
    args = parser.parse_args()

    sc = SparkContext(appName="Inner ADMM Tester for using %d partitions" %args.N)
    sc.setLogLevel('OFF')


    logger.setLevel(logging.INFO)
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    uniformweight = 1./20
    ParallelSolverCls = eval(args.ParallelSolver)
    data = sc.textFile(args.objectives).map(lambda x:eval(x)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)


    if args.initfile != None:
        PY_RDD = sc.textFile(args.initfile).map(eval).partitionBy(args.N, partitionFunc=identityHash).mapValues(lambda (cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats): evalSolversY(cls_args, P_vals, Y_vals, Phi_vals, Upsilon_vals, stats, pickle.dumps(LocalLSSolver), args.rho_inner)).persist(StorageLevel.MEMORY_ONLY)
        RDDSolver_cls = ParallelSolverCls(LocalSolverClass=LocalLSSolver, data=data, initvalue=uniformweight, N=args.N, rho=args.rho, rho_inner=args.rho_inner, p=args.p, lean=False, debug=True, RDD=PY_RDD)
    else:
        RDDSolver_cls = ParallelSolverCls(LocalSolverClass=LocalLSSolver, data=data, initvalue=uniformweight, N=args.N, rho=args.rho, rho_inner=args.rho_inner, p=args.p, lean=False, debug=True)
    
    
    G = sc.textFile(args.G).map(eval)
    #Create consensus variable, initialized to uniform assignment ignoring constraints
    ZRDD = G.map(lambda var:(var,uniformweight+0.2)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
    
    P = {}
    Y = {}
    Phi ={}
    Upsilon = {}
    (splitID, (solver_args, P, Y, Phi, Upsilon, stats) ) =  RDDSolver_cls.PrimalDualRDD.collect()[0]
    Z = dict( ZRDD.collect() )

  
    local_solver_cls = LocalLSSolver(dict( data.collect() ), args.rho, args.rho_inner)
    
    D_mat = local_solver_cls.D
    
    (pi, ni) = D_mat.shape
    P_vec = np.matrix( np.zeros((ni,1)))
    Phi_vec =  np.matrix( np.zeros((ni,1)))
    Z_vec =  np.matrix( np.zeros((ni,1)))
    for ind in range(ni):
        P_vec[ind] = P[ local_solver_cls.translate_coordinates2ij[ind]] 
        Phi_vec[ind] = Phi[ local_solver_cls.translate_coordinates2ij[ind]]
        Z_vec[ind] = Z[ local_solver_cls.translate_coordinates2ij[ind]]

    
   
    G1G2 = args.G.split('/')[-1]
    writeMat2File('/home/armin_mrm93/D%s' %G1G2, D_mat)
    writeMat2File('/home/armin_mrm93/Zbar%s' %G1G2, Z_vec-Phi_vec)
    D_name = "outputZs_matrix/D%s" %G1G2
    Zbar_name = "outputZs_matrix/Zbar%s" %G1G2
    upload_blob('armin-bucket', '/home/armin_mrm93/D%s' %G1G2, D_name)
    upload_blob('armin-bucket', '/home/armin_mrm93/Zbar%s' %G1G2, Zbar_name)


    if args.initfile!= None:
        file_name = 'profiling/InnerADMM/' + args.tracefile.split('/')[-1]
        download_blob(args.bucketname, file_name,args.tracefile )
        fp = open(args.tracefile, 'r')
        (args_old, trace) = pickle.load(fp)
        prev_iters = max( trace.keys()) + 1
        
    else:    
        trace = {}
        prev_iters = 0
    new_trace = RDDSolver_cls.joinAndAdapt(ZRDD = ZRDD, alpha = args.alpha, rho = args.rho, maxiters=args.maxiters, logger=logger)

    if prev_iters>0:
        for key in new_trace:
            val = new_trace[key]
            trace[key+prev_iters] = val
    else:
        trace = new_trace 
    #Save the trace
    with open(args.tracefile,'w') as f:
        pickle.dump((args,trace),f)

    #Upload the trace
    file_name = 'profiling/InnerADMM/' + args.tracefile.split('/')[-1]
    upload_blob(args.bucketname, args.tracefile, file_name)    
    

    #Save the variables 
    safeWrite_GCP(RDDSolver_cls.PrimalDualRDD,args.RDDfile,args.bucketname)

    (splitID, (solver_args, P, Y, Phi, Upsilon, stats) ) = RDDSolver_cls.PrimalDualRDD.collect()[0]
    Y_vec  = np.matrix( np.zeros((pi,1)))
    Upsilon_vec =  np.matrix( np.zeros((pi,1)))
    for ind in range(pi):
        Y_vec[ind] = Y[local_solver_cls.translate_coordinates2ij_Y[ind]]
        Upsilon_vec[ind] = args.rho_inner * Upsilon[local_solver_cls.translate_coordinates2ij_Y[ind]]
    for ind in range(ni):
        P_vec[ind] = P[ local_solver_cls.translate_coordinates2ij[ind]]
        Phi_vec[ind] = Phi[ local_solver_cls.translate_coordinates2ij[ind]]

    Grad_P = args.rho * (P_vec - (Z_vec-Phi_vec)) - D_mat.T * Upsilon_vec
    grad_Y = np.matrix( np.zeros((pi,1)))
    Y_norm = norm_p(Y_vec, args.p)
    if Y_norm>0.0:
        for i in range(pi):
            grad_Y[i] = np.sign(float(Y_vec[i]))  * (abs(float(Y_vec[i]))/Y_norm)**(args.p-1.) + Upsilon_vec[i]
    print np.linalg.norm(grad_Y)#, Y_vec
    print np.linalg.norm(Y_vec - D_mat * P_vec)
    print np.linalg.norm(Grad_P)
   #Write P and Y
    Pfile = 'P%s_%s_p%0.1f' %(G1G2, args.ParallelSolver, args.p)
    Yfile = 'Y%s_%s_p%0.1f' %(G1G2, args.ParallelSolver, args.p)
    writeMat2File('/home/armin_mrm93/%s' %Pfile, P_vec)
    writeMat2File('/home/armin_mrm93/%s' %Yfile, Y_vec)
    upload_blob('armin-bucket', '/home/armin_mrm93/%s' %Pfile, 'PY_vars/%s' %Pfile) 
    upload_blob('armin-bucket', '/home/armin_mrm93/%s' %Yfile, 'PY_vars/%s' %Yfile) 
   
   


    
    
