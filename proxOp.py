import argparse 
from pyspark import SparkContext
from math import sqrt
from  scipy.optimize import newton
from numpy import sign 

from numpy import linalg as LA
from sympy import Symbol, solve
import time
from helpers import safeWrite

def solve_ga(a, p):
    """Return the solution of (x/a)^(p-1)+x=1"""
    if a>0:
        x = Symbol("x", positive=True)
        sols = solve((x/a)**(p-1)+x-1)
        if len(sols) == 0:
            sol = 0.
        else:
            sol = max(sols)
        return sol
    else:
        return  0.

def normp(RDD, p):
    """Compute p-norm of a vector stored as an RDD."""
    norm_to_P =  RDD.values().map(lambda x:abs(x)**p).reduce(lambda x,y:x+y)
    return norm_to_P**(1./p)
def pnorm_proxop(N, p, rho, epsilon):
    """Solve prox operator for vector N and p-norm"""

    t_start = time.time()

    #Normalize N
    S = N.mapValues(lambda nr: sign(nr)).cache()
    N = N.mapValues(lambda nr: rho*abs(nr)).cache()
    
    N_norm = normp(N, p)

    #Initial lower and upper norm
    Z_norm_L = 0.
    Z_norm_U = N_norm
    #Make sure Alg. eneters the iterations
    error = epsilon + 1
    while error> epsilon:
        Z_norm = (Z_norm_L + Z_norm_U)/2
        Z = N.mapValues(lambda Nr: Nr*solve_ga(Z_norm * Nr**((2-p)/(p-1)), p)) 
        Z_norm_current = normp(Z, p)
        if Z_norm_current<Z_norm:
            Z_norm_U = Z_norm_current
        else:
            Z_norm_L = Z_norm_current
        error = (Z_norm_U-Z_norm_L)/N_norm
        print "Error is %f, time is %f" %(error, time.time()-t_start)
    Z = Z.join(S).mapValues(lambda (zr, s):zr*s/rho).cache()
    t_end = time.time()
    return Z, Z_norm, t_end-t_start     
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Proximal Operator for p-norms over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile',default=None,help ="File containing N.")
    parser.add_argument('outputfile',help = 'Output the result of the prox op Z.')
    parser.add_argument('--p',help='The norm p',type=float)
    parser.add_argument('--rho',help='The rho in prox. operator',type=float)
    parser.add_argument('--epsilon',help='desired accuracy',default=1.e-6,type=float)
    parser.add_argument('--parts',help='partitions',default=10,type=int)
    args = parser.parse_args() 
    

    sc = SparkContext()
    sc.setLogLevel("OFF")
    p = args.p
    rho = args.rho
    epsilon = args.epsilon
    N = sc.textFile(args.inputfile).map(eval).partitionBy(args.parts).cache()
    #N = sc.parallelize([(1,22.9),(2,33.8),(3,98.2)]).cache()
    Z, Z_norm, t_running = pnorm_proxop(N, p, rho, epsilon)
    safeWrite(Z, args.outputfile)
    fp = open(args.outputfile + "_norm_time_info",'w')
    fp.write("(%f, %f)" %(Z_norm, t_running))
    fp.close()



    
