import argparse 
from pyspark import SparkContext
from math import sqrt
from  scipy.optimize import newton

def solve_ga(a, p):
    """Return the solution of (x/a)^(p-1)+x=1"""
    Func = lambda x:(x/a)**(p-1) + x - 1
    Func_prime = lambda x:(p-1)/a * (x/a)**(p-2) + 1
    x0 = 0.5
    try: 
        x_sol = newton(func=Func, x0=x0, fprime=Func_prime, tol=1.48e-08, maxiter=50, fprime2=None)
    except RuntimeError:
        x_sol = 0.
    return x_sol

def normp(RDD, p):
    """Compute p-norm of a vector stored as an RDD."""
    norm_to_P =  RDD.values().map(lambda x:x**p).reduce(lambda x,y:x+y)
    return norm_to_P**(1./p)
def pnorm_proxop(N, p, epsilon):
    """Solve prox operator for vector N and p-norm"""


    N_norm = normp(N, p)

    #Initial lower and upper norm
    Z_norm_L = 0.
    Z_norm_U = N_norm
    #Make sure Alg. eneters the iterations
    accuracy = epsilon + 1
    while accuracy> epsilon:
        Z_norm = (Z_norm_L + Z_norm_U)/2
        print "Z norm is %f" %Z_norm, N_norm
        Z = N.mapValues(lambda Nr: Nr*solve_ga(Z_norm * Nr**((2-p)/(p-1)), p)) 
        Z_norm_current = normp(Z, p)
        if Z_norm_current<Z_norm:
            Z_norm_U = Z_norm_current
        else:
            Z_norm_L = Z_norm_current
        accuracy = (Z_norm_U-Z_norm_L)/N_norm
    return Z, Z_norm     
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Proximal Operator for p-norms over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('inputfile',default=None,help ="File containing N. ")
    #parser.add_argument('outputfile',help = 'Output file storing learned doubly stochastic matrix Z')
    parser.add_argument('--p',help='The norm p',type=float)
    parser.add_argument('--epsilon',help='desired accuracy',default=1.e-2,type=float)
    args = parser.parse_args() 
    

    sc = SparkContext()
    sc.setLogLevel("OFF")
    p = args.p
    epsilon = args.epsilon
    #N = sc.textFile(args.inputfile).map(eval).partitionBy(10).cache()
    N = sc.parallelize([(1,22.9),(2,33.8),(3,98.2)]).cache()
    
    pnorm_proxop(N, p, epsilon)
