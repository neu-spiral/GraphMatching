import argparse 
from pyspark import SparkContext
from math import sqrt
from numpy import sign 

from numpy import linalg as LA
import time
#from helpers_GCP import safeWrite_GCP, upload_blob
from helpers import safeWrite

def solve_ga_bisection(a, p):
    """Return the solution of (x/a)^(p-1)+x=1, via bi-section method."""
    if a>0.:
        U = a
        L = 0.
        epsilon = 1.e-6
        error = epsilon + 1
        f = lambda x:(x/a)**(p-1)+x-1
        while error>epsilon:    
            C = (L+U)/2.
            if f(C)*f(U)>0:
                U = C
            else:
                L = C
            error = (U-L)/a
        return C
    else:
        return 0.
         

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
    """Compute p-norm of a vector stored as an RDD with its sign."""
    norm_to_P =  RDD.values().map(lambda (x, x_sign):abs(x)**p).reduce(lambda x,y:x+y)
    return norm_to_P**(1./p)
def pnormOp(NothersRDD,p, rho, epsilon):
    """Prox. operator for p-norm, i.e., solve the following problem:
           min_Y \|Y\|_p + rho/2*\|Y-N\|_2^2,
        where N values are given in NothersRDD, s.t., each partition i contains (N_i, Others_i) and N_i is a dictionary.
    """
    def pnorm(RDD, p):
        return (  RDD.values().flatMap(lambda (Nm, Others):[abs(Nm[key][0])**p for key in Nm]).reduce(lambda x,y:x+y) )**(1./p)

     #Normalize N
    NothersRDD = NothersRDD.mapValues(lambda (Nm, Others): ( dict([ (key, (rho*abs(Nm[key]),sign(Nm[key]))) for key in Nm]), Others)).cache() 
    
    N_norm = pnorm(NothersRDD, p)
    
    Y_norm_L = 0.
    Y_norm_U = N_norm

    error = epsilon + 1
    while error>epsilon:
        Y_norm = (Y_norm_L + Y_norm_U)/2   
        TempRDD = NothersRDD.mapValues(lambda (Nm, Others): ( dict([(key, ( Nm[key][0]*solve_ga_bisection(Y_norm *Nm[key][0] **((2-p)/(p-1)),  p ), Nm[key][1]) ) for key in Nm]) ,Others) )
      
        Y_norm_current = pnorm(TempRDD, p)
        if Y_norm_current<Y_norm:
            Y_norm_U = Y_norm
        else:
            Y_norm_L = Y_norm
        error = (Y_norm_U-Y_norm_L)/N_norm
    #    print "Error in p-norm Prox. Op. is %f" %error
    #Denormalize the solution 
    YothersRDD = NothersRDD.mapValues(lambda (Nm, Others):(dict([(key, Nm[key][1]*Nm[key][0]/rho) for key in Nm]), Others))
    Ypnorm =  YothersRDD.values().flatMap(lambda (Y, Others): [abs(Y[key])**p for key in Y]).reduce(lambda x,y:x+y)
    Ypnorm = Ypnorm**(1./p)
    return (YothersRDD, Ypnorm) 

    
def pnorm_proxop(N, p, rho, epsilon):
    """Solve prox operator for vector N and p-norm"""

    t_start = time.time()

    #Normalize N
   # S = N.mapValues(lambda nr: sign(nr)).cache()
    N = N.mapValues(lambda nr: (rho*abs(nr), sign(nr))).cache()
    
    N_norm = normp(N, p)

    #Initial lower and upper norm
    Z_norm_L = 0.
    Z_norm_U = N_norm
    #Make sure Alg. eneters the iterations
    error = epsilon + 1
    while error>epsilon:
        Z_norm = (Z_norm_L + Z_norm_U)/2
        Z = N.mapValues(lambda (Nr, Nr_sign):  (Nr*solve_ga_bisection(Z_norm * Nr**((2-p)/(p-1)), p), Nr_sign)) 
        Z_norm_current = normp(Z, p)
        if Z_norm_current<Z_norm:
            Z_norm_U = Z_norm
        else:
            Z_norm_L = Z_norm
        error = (Z_norm_U-Z_norm_L)/N_norm
     #   print "Error is %f, time is %f" %(error, time.time()-t_start)
    Z = Z.mapValues(lambda (zi, zi_sign):zi*zi_sign/rho).cache()
    t_end = time.time()
    return Z, Z_norm, t_end-t_start     
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Proximal Operator for p-norms over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfile',default=None,help ="File containing N.")
    parser.add_argument('outputfile',help = 'Output the result of the prox op Z.')
    parser.add_argument('infofile',default=None,help="File to store norm and time")
    parser.add_argument('BUCKinfofile',default=None,help="File to store norm and time on the bucket")
    parser.add_argument('--p',help='The norm p',type=float)
    parser.add_argument('--rho',help='The rho in prox. operator',type=float)
    parser.add_argument('--epsilon',help='desired accuracy',default=1.e-6,type=float)
    parser.add_argument('--parts',help='partitions',default=10,type=int)
    parser.add_argument('--BUCKET', type=str, help='The google cloud bucket you are interacting with. Only pass if you are running the code on Google Cloud.')
    parser.add_argument('--GC',action='store_true', help = 'Pass if you run on Google Cloud.')
    args = parser.parse_args() 
    
    if args.GC:
        BUCKET = args.BUCKET
    sc = SparkContext()
    sc.setLogLevel("OFF")
    p = args.p
    rho = args.rho
    epsilon = args.epsilon
    N = sc.textFile(args.inputfile).map(eval).partitionBy(args.parts).cache()
    Z, Z_norm, t_running = pnorm_proxop(N, p, rho, epsilon)
    if args.GC:
        safeWrite_GCP(Z, args.outputfile,  BUCKET)
    else:
        safeWrite(Z, args.outputfile)

    fp = open(args.infofile,'w')
    fp.write("(%f, %f)" %(Z_norm, t_running))
    fp.close()
    if args.GC:
        upload_blob(BUCKET, args.infofile,  args.BUCKinfofile)



    
