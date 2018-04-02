import argparse 
from pyspark import SparkContext
from math import sqrt
from  scipy.optimize import newton

def solve_ga(a, p):
    """Return the solution of (x/a)^(p-1)+x=1"""
    Func = lambda x:(x/a)**(p-1) + x - 1
    Func_prime = lambda x:(p-1)/a * (x/a)**(p-2) + 1
    x0 = 0.5
    
    x_sol = newton(func=Func, x0=x0, fprime=Func_prime, tol=1.48e-08, maxiter=50, fprime2=None)
    return x_sol


def pnorm_proxop(N, p, epsilon):
    """Solve prox operator for vector N and p-norm"""
    #Initializing the vector Z

    #The lenght of the vector
    R = N.count()

    N_norm = sqrt(N.values().map(lambda x:x**2).reduce(lambda x,y:x+y))

    #Initial lower and upper norm
    Z_norm_L = 0.
    Z_norm_U = N_norm
    #Make sure Alg. eneters the iterations
    accuracy = epsilon + 1
    while accuracy> epsilon:
        Z_norm = (Z_norm_L + Z_norm_U)/2
        print "Z norm is %f" %Z_norm
        Z = N.mapValues(lambda Nr: Nr*solve_ga(Z_norm * Nr**((2-p)/(p-1)), p)) 
        Z_norm_current = sqrt(Z.values().map(lambda x:x**2).reduce(lambda x,y:x+y))
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
