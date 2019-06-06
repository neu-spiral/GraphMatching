import argparse
import pickle
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
           

def dict_grad_Fij(objectives, Wb=None, N=64):
    '''Return the gradianet of AP-PB w.r.t. P as a dict, with keys (obj, var), B matrix could be weighted.'''
    grad = {}

    for obj in objectives:
        (s1, s2) = objectives[obj]
        for var in s1:
            grad[(obj, var)] = 1.
        if Wb==None:
            for var in s2:
                if (obj, var) not in grad:
                    grad[(obj, var)] = -1.
                else:
                    grad[(obj, var)] = 0.0
        else:
            for k in range(N):
                node_k = unicode(k)
                row, col = obj
                var = (row, node_k)
                if (obj, var) not in grad:
                    grad[(obj, var)] = -1.0* Wb[(node_k, col)]
                else:
                    grad[(obj, var)] += -1.0* Wb[(node_k, col)]
    return grad
def eval_Fij(objectives, P, Wb=None, N=64):
    """Given the vector p return the functions f_1(p), ..., f_m(p) as an m-dimensional vector F"""
    F = {}
    for obj  in objectives:
        tmpVal = 0.0
        (s1, s2) = objectives[obj]
        for var in s1:
            tmpVal += P[var]
        if Wb==None:
            for var in s2:
                tmpVal -= P[var]
            F[obj] = tmpVal
        else:
            for  k in range(N):
                node_k = unicode(k)
                row, col = obj
                var = (row, node_k)
                tmpVal -=  Wb[(node_k, col)]*P[var]
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
def ComposGrad(VARS2objectivesPlus, VARS2objectivesMinus, g_norm, Wb=None, N=64):
    grad = {}
    for var in VARS2objectivesPlus:
        grad[var] = 0.0
        for obj in VARS2objectivesPlus[var]:
            grad[var] += g_norm[obj]
    if Wb==None:
        for var in VARS2objectivesMinus:
            if var not in grad:
                grad[var] = 0.0
            for obj in VARS2objectivesMinus[var]: 
                grad[var] -= g_norm[obj]
    else:
        for row in range(N):
            r = unicode(row)
            for k in range(N):
                node_k = unicode(k)
                var = (r, node_k)
                if var not in grad:
                    grad[var] = 0.0
                for col in range(N):
                    c = unicode(col)
                    obj = (r,c)
                    grad[var] -= g_norm[obj]*Wb[(node_k,c)]
                           
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
    
def  DualGap(Var, S, Grad):
    gap = 0.0
    for key in Var:
######################
        try: 
            gap += Var[key]*Grad[key]
        except KeyError:
            gap += 0.0
    for  (r, c) in enumerate(S):
        try:
            gap -= Grad[(unicode(r), unicode(c))]
        except:
            gap += 0.0
    return gap
      
        
def FW(objectives, VARS2objectivesPlus, VARS2objectivesMinus, N, p,  D=None, lamb=0.0, Wb=None, maxiters=100, epsilon=1.e-2, ONLY_lin=False, initP=None):
    
    
    trace = {} 
    tSt = time() 
    #Find dictionaries to trnaslate matrix elements to the corresponding coordinate in its vectorization.
   # transl_vars2i, transl_objs2i, transl_i2vars, transl_i2objs = get_translators(objectives)

    
    #Initial value 
    if initP==None:
        P = {}
        for i in range(N):
            for j in range(N):
  ##########################
                if j==i+1 or (j==0 and j==N-1):
                    P[(unicode(i), unicode(j))] = 1.0
                else:
                    P[(unicode(i), unicode(j))] = 0.0
    else:
        P = initP
              

   
    last = time()
    print "Done preprocessing...\n Time is: %f (s)" %(last-tSt)
    for t in range(maxiters):
        trace[t] = {}
        step_size = 1./(t+2)


        #Compute objective
        if not ONLY_lin:
            APminusBP = eval_Fij(objectives, P, Wb, N)
            OBJNOLIN = vec_norm(APminusBP, args.p)
        else:
            OBJNOLIN = 0.0
         
       #find the grad. of ||AP-PB|| w.r.t. P
        if not ONLY_lin:
            Df = ComposGrad(VARS2objectivesPlus, VARS2objectivesMinus, grad_norm(APminusBP,args.p), Wb, N) 
        #If ONLY_lin passed, ignore the term ||AP-PB||_p
        else:
            Df = dict([(key, 0.0) for key in P])
        
        #Add the grad. of the linear term (if given)
        if D != None:
            LIN_OBJ = 0.0
            for key in D:
                Df[key] += lamb*D[key]
                LIN_OBJ += P[key]*D[key]
        else:
            LIN_OBJ = 0.0

                 
        Df_mat = dictTo2darray(Df, N)

        #Solve the linear problem via Hungarian
        row_ind, col_ind = linear_sum_assignment(Df_mat)
    
        #Compute the duality gap
        dual_gap = DualGap(P, col_ind, Df)

        #Update variables by taking aconvex combination
        for var in P:
            P[var] = (1-step_size) * P[var]
        for (r, c) in enumerate(col_ind):
            P[(unicode(r), unicode(c))] += step_size
        now = time()
        trace[t]['OBJNOLIN'] = OBJNOLIN
        trace[t]['OBJ'] = OBJNOLIN + lamb*LIN_OBJ
        trace[t]['GAP'] = dual_gap
        trace[t]['IT_TIME'] = now - last
        trace[t]['TIME'] = now - tSt
        print "Iteration %d, iteration time is %f Objective is %f, Norm is %f, duality gap is %f" %(t, now - last, OBJNOLIN + lamb*LIN_OBJ, OBJNOLIN, dual_gap)
        last = time()
    return P, trace 
    
def evalPair(pair):
    return (eval(pair[0]), eval(pair[1]))
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'CVXOPT Solver for Graph Matching',formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('objectives',type=str,help='File containing objectives.')
    parser.add_argument('N',type=int,help='File to store the results.')
    parser.add_argument('outfile',type=str,help='Output file')
    parser.add_argument('--dist',type=str,help='File containing distace file.')
    parser.add_argument('--weights',type=str,help='File containing distace file.')
    parser.add_argument('--lamb', default=0.0, type=float, help='lambda parameter regularizing the linear term.')
    parser.add_argument('--maxiters',default=100,type=int, help='Maximum number of iterations')
    parser.add_argument('--epsilon', default=1.e-2, type=float, help='The accuracy for cvxopt solver.')
    parser.add_argument('--p', default=2.5, type=float, help='p parameter in p-norm')
    parser.add_argument('--initP',default=None, type=str, help="Initial solution P.")
    parser.add_argument('--bucket_name',default='armin-bucket',type=str,help='Bucket name, specify when running on google cloud. Outfile and logfile will be uploaded here.')
    parser.add_argument('--ONLY_lin',action='store_true',help='Pass to ignore ||AP-PB||_p')
    parser.set_defaults(ONLY_lin=False)

    args = parser.parse_args()

    objectives = {}
    VARS2objectivesPlus = {}
    VARS2objectivesMinus = {}
    stLoad = time()
    for partFile in  glob.glob(args.objectives + "/part*"):
        print "Now readaiang " + partFile
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
          
    if args.dist != None:         
        D = {}
        for partFile in  glob.glob(args.dist + "/part*"):
            print "Now readaiang " + partFile
            with open(partFile, 'r') as pF:
                for dist_line in pF: 
                    (var, dist) = eval(dist_line)
                    D[var]  = dist
    else:
        D = None
              
    if args.weights != None:
        with open(args.weights) as weightF:
            Wb = pickle.load(weightF)
    else:
        Wb = None
   
    endLoad = time()

    print len(VARS2objectivesPlus), len(VARS2objectivesMinus)
    if args.ONLY_lin:
        ONLY_lin= True
    else:
        ONLY_lin = False

    if args.initP != None:
        with open(args.initP,'r') as trF:
            initP = pickle.load(trF)
    else:
        initP = None
        

    sol, trace = FW(objectives=objectives, VARS2objectivesPlus=VARS2objectivesPlus, VARS2objectivesMinus=VARS2objectivesMinus, N=args.N, p=args.p, D=D, lamb = args.lamb, Wb=Wb, maxiters=args.maxiters, epsilon=args.epsilon, ONLY_lin=ONLY_lin, initP=initP)
    
    
    with open(args.outfile + '_trace', 'wb') as fTrace:
        pickle.dump((endLoad-stLoad, trace), fTrace)
    with open(args.outfile + '_P', 'wb') as fSol:
        pickle.dump(sol, fSol)
       

    
    
    
    
