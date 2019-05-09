import argparse
import glob
import numpy as np
from numpy import sign
from pyspark import SparkContext
from time import time
from numpy.linalg import matrix_rank
def norm(x,p):
    return ( sum( abs(x)**p)) **(1./p)
def grad_norm(x,p):
    """Return gradient for p-norm function g(x)=\|x\|_p"""
    m, one = x.size
    norm_p = norm(x,p) 
    grad = matrix(0.0, (m,1))
    for i in range(m):
        grad[i] = sign(x[i]) * (abs(x[i])/norm_p)**(p-1.)
    return grad
def Hessian_norm(x,p):
    """Return the Hessian matrix for p-norm function g(x)=\|x\|_p"""
    m, one = x.size
    Hessian = matrix(0.0, (m,m))
    norm_p = norm(x,p)
    for i in range(m):
        for j in range(i+1):
            if i!=j:
                Hessian[i,j] = (1.-p) * sign(x[i]) * sign(x[j]) * abs(x[i])**(p-1.) * abs(x[j])**(p-1.)/(norm_p**(2*p-1.))
            else:
                Hessian[i,j] = (p-1.) * abs(x[i])**(p-2.)/(norm_p**(p-1.)) + (1.-p) * abs(x[i])**(2*p-2.)/(norm_p**(2*p-1.)) 
    return Hessian 
           
def grad_Fij(transl_vars2i,transl_objs2i):
    """Compute the \nablaF, where F=[f_1(p), ..., f_m(p)]^T; it returns an n by m matrix."""
    m = len(transl_objs2i)
    n = len(transl_vars2i)
    grad = matrix(0.0, (n,m)) 
    for key in objectives:
        col = transl_objs2i[key]
        (s1, s2) = objectives[key]
        for var in s1:
            row = transl_vars2i[var]
            grad[row, col] = 1.
        for var in s2:
            row = transl_vars2i[var]
            if grad[row, col] == 0:
                col = transl_vars2i[var]
                grad[row, col] = -1.
            else:
                grad[row, col] = 0.0
    return grad
def eval_Fij(objectives, transl_vars2i,transl_objs2i, p):
    """Given the vector p return the functions f_1(p), ..., f_m(p) as an m-dimensional vector F"""
    m = len(transl_objs2i)
    F = matrix(0.0, (m,1))
    for key in objectives:
        (s1, s2) = objectives[key]
        row = transl_objs2i[key]
        tmpVal = 0.0
        for var in s1:
            var_ind = transl_vars2i[var]
            tmpVal += p[var_ind]
        for var in s2:
            var_ind = transl_vars2i[var]
            tmpVal -= p[var_ind]
        F[row] = tmpVal
    return F
    
def get_translators(objectives):
    """Return two dictionaries, transl_vars2i and transl_objs2i, that map variables and objectives to integers that correspond to elemnts in vector.""" 
    m = len(objectives)
    variable_set = set()

    transl_vars2i = {}
    transl_objs2i = {}
    i = 0
    for key in objectives:
        (s1, s2) = objectives[key]
        transl_objs2i[key] = i
        i += 1
        variable_set.update(set(s1))
        variable_set.update(set(s2))
    i = 0
    for var in variable_set:
        transl_vars2i[var] = i
        i += 1
    transl_i2vars = dict( [(transl_vars2i[key],key) for key in transl_vars2i])
    transl_i2objs = dict( [(transl_objs2i[key],key) for key in transl_objs2i])
    return transl_vars2i, transl_objs2i, transl_i2vars, transl_i2objs
        
def get_row_col_objs(transl_vars2i):
    rowObjs = {}
    colObjs = {}
    ii = 0
    for var in transl_vars2i:
        (i,j) = var
        if i not in rowObjs:
            rowObjs[i] = [(i,j)]
        else:
            rowObjs[i].append((i,j))
        if j not in colObjs: 
            colObjs[j] = [(i,j)]
        else:
            colObjs[j].append((i,j))
    return rowObjs, colObjs
def build_constraints(transl_vars2i, rowObjs, colObjs):
    n = len(transl_vars2i)
    col_constraints = len(colObjs)
    row_constraints  = len(rowObjs)
    A = matrix(0.0, (row_constraints+col_constraints , n))
    b = matrix(1., (row_constraints+col_constraints, 1))
    cnt = 0
    for row in rowObjs:
        for var in rowObjs[row]:
            coord = transl_vars2i[var]
            A[cnt, coord] = 1.
        cnt += 1
    for col in colObjs:
        for var in colObjs[col]:
            coord = transl_vars2i[var]
            A[cnt, coord] = 1.
        cnt += 1
    return A, b
def FW(objs, N, maxiters=100):

    
    
    #Find dictionaries to trnaslate matrix elements to the corresponding coordinate in its vectorization.
    transl_vars2i, transl_objs2i, transl_i2vars, transl_i2objs = get_translators(objectives)

    #Initial value 
    P = np.zeros(N*N)
    for i in range(N):
         coord = transl_vars2i[(unicode(i), unicode(i))]
         P[coord] = 1.0

    grad_APminusPB = grad_Fij(transl_vars2i,transl_objs2i)
    for t in range(maxiters):
        print t
        APminusBP = eval_Fij(objectives, transl_vars2i,transl_objs2i, P)
        Df = (grad_APminusPB *  grad_norm(APminusBP,args.p)).T 
        

    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'CVXOPT Solver for Graph Matching',formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('objectives',type=str,help='File containing objectives.')
    parser.add_argument('N',type=int,help='File to store the results.')
    parser.add_argument('--maxiters',default=100,type=int, help='Maximum number of iterations')
    parser.add_argument('--epsilon', default=1.e-3, type=float, help='The accuracy for cvxopt solver.')
    parser.add_argument('--p', default=2.5, type=float, help='p parameter in p-norm')
    parser.add_argument('--bucket_name',default='armin-bucket',type=str,help='Bucket name, specify when running on google cloud. Outfile and logfile will be uploaded here.')
    parser.add_argument('--GCP',action='store_true',help='Pass if running on Google Cloud Platform.')
    parser.set_defaults(GCP=False)

    args = parser.parse_args()
    sc = SparkContext(appName='CVX GM',master='local[40]')
    objectives = dict(sc.textFile(args.objectives).map(eval).collect())

    sol, trace = FW(objectives, args.N)

    
    
    
    
