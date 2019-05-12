import argparse
from scipy.optimize import linear_sum_assignment
import glob
import numpy as np
from numpy import sign
from pyspark import SparkContext
from time import time
from numpy.linalg import matrix_rank


def vec_norm(x,p):
    tmp = 0.0
    for key in x:
        tmp += abs(x[key])**p
    return tmp **(1./p)

def grad_norm(F,p):
    """Return gradient for p-norm function g(x)=\|F\|_p, F is a dict"""
    grad = {}
    norm_p = vec_norm(F,p)
    for key in F:
        grad[key] = sign(F[key]) * (abs(F[key])/norm_p)**(p-1.)
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
           

def dict_grad_Fij(objectives):
    '''Return the gradianet of AP-PB w.r.t. P as a dict, with keys (obj, var)'''
    grad = {}

    for obj in objectives:
        (s1, s2) = objectives[obj]
        for var in s1:
            grad[(obj, var)] = 1.
        for var in s2:
            if (obj, var) not in grad:
                grad[(obj, var)] = -1.
            else:
                grad[(obj, var)] = 0.0
    return grad
def eval_Fij(objectives, P):
    """Given the vector p return the functions f_1(p), ..., f_m(p) as an m-dimensional vector F"""
    F = {}
    for obj  in objectives:
        tmpVal = 0.0
        (s1, s2) = objectives[obj]
        for var in s1:
            tmpVal += P[var]
        for var in s2:
            tmpVal -= P[var]
        F[obj] = tmpVal
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
        (node1, node2) = var
        node1 = int(node1)
        node2 = int(node2)
        transl_vars2i[(node1, node2)] = i
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
def vec2mat(vec, trans_dict, dim):
    m, one = vec.shape
    mat = np.matrix(np.zeros((dim, dim)) )
    for i in range(m):
        r,c = trans_dict[i]
        mat[r,c] = vec[i,0]
    return mat
def ComposGrad(VARS2objectivesPlus, VARS2objectivesMinus, g_norm):
    grad = {}
    for var in VARS2objectivesPlus:
        grad[var] = 0.0
        for obj in VARS2objectivesPlus[var]:
            grad[var] += g_norm[obj]
    for var in VARS2objectivesMinus:
        if var not in grad:
            grad[var] = 0.0
        for obj in VARS2objectivesMinus[var]: 
            grad[var] -= g_norm[obj]
    return grad
def dictTo2darray(grad, N):
    M = np.zeros((N, N))
    for var in grad:
       i,j = evalPair(var)
       M[i][j] = grad[var]

    return M
def SumRowCol(P, N):
    rowSum = {}
    colSum = {}
    for i in range(N):
        tmp = 0.0 
        for j in range(N):
            if (unicode(i), unicode(j)) in P:
                tmp += P[(unicode(i), unicode(j))]      
        rowSum[i] = tmp
    for j in range(N):
        tmp = 0.0
        for i in range(N):
            if (unicode(i), unicode(j)) in P:
                tmp += P[(unicode(i), unicode(j))]
        colSum[j] = tmp
    return rowSum, colSum
    
      
        
def FW(objectives, VARS2objectivesPlus, VARS2objectivesMinus, N, maxiters=100):
    
    
    trace = {} 
    tSt = time() 
    #Find dictionaries to trnaslate matrix elements to the corresponding coordinate in its vectorization.
   # transl_vars2i, transl_objs2i, transl_i2vars, transl_i2objs = get_translators(objectives)

    
    #Initial value 
    P = {}
    for i in range(N):
        for j in range(N):
            if i==j:
                P[(unicode(i), unicode(j))] = 1.0
            else:
                P[(unicode(i), unicode(j))] = 0.0
              

    last = time()
    print "Done preprocessing...\n Time is: %f (s)" %(last-tSt)
    for t in range(maxiters):
        trace[t] = {}
        step_size = 1./(t+2)
        APminusBP = eval_Fij(objectives, P)
        OBJ = vec_norm(APminusBP, args.p)
         
        Df = ComposGrad(VARS2objectivesPlus, VARS2objectivesMinus, grad_norm(APminusBP,args.p) )
        Df_mat = dictTo2darray(Df, N)
        
        row_ind, col_ind = linear_sum_assignment(Df_mat)

        for var in P:
            P[var] = (1-step_size) * P[var]
        for (r, c) in enumerate(col_ind):
            P[(unicode(r), unicode(c))] += step_size
        now = time()
        trace[t]['OBJ'] = OBJ
        trace[t]['IT_TIME'] = now - last
        trace[t]['TIME'] = now - tSt
        print "Iteration %d, iteration time is %f Objective is %f" %(t, now - last, OBJ)
        last = time()
    return P, trace 
    
def evalPair(pair):
    return (eval(pair[0]), eval(pair[1]))
    
    
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
    sc.setLogLevel('OFF')

    objectives = {}
    VARS2objectivesPlus = {}
    VARS2objectivesMinus = {}
    for partFile in  glob.glob(args.objectives + "/part*"):
        print "Now readaiang" + partFile
        with open(partFile, 'r') as pF:
            for obj_line in pF:
                (obj, VARS) = eval(obj_line)
                objectives[obj] = VARS
                for var in VARS[0]:
                    if var not in VARS2objectivesPlus:
                        VARS2objectivesPlus[var] = [obj]
                    else:
                        VARS2objectivesPlus[var].append(obj)
                for var in VARS[1]:
                    if var not in VARS2objectivesMinus:
                        VARS2objectivesMinus[var] = [obj]
                    else:
                        VARS2objectivesMinus[var].append(obj)
                   

    sol, trace = FW(objectives, VARS2objectivesPlus, VARS2objectivesMinus, args.N)

    
    
    
    
