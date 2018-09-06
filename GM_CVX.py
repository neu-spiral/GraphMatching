
import cvxopt
from cvxopt import matrix
from pyspark import SparkContext
from scipy.stats import norm

def grad_norm(x,p):
    """Return gradient for p-norm"""
    m, one = x.size
    norm_p = norm(x,p) 
    grad = matrix(0.0, (m,1))
    for i in range(m):
        grad[i] = (x[i]/norm_p)***(p-1)
    return grad
def grad_Fij(transl_vars2i,transl_objs2i)
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
def get_translators(objectives):
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
        (i,j) = var:
        if i not in rowObjs:
            rowObjs[i] = [(i,j)]
        else:
            rowObjs[i].append((i,j))
        if j not in colObjs: 
            colObjs[j] = [(i,j)]
        else:
            colObjs[j].apppend((i,j))
    return rowObjs, colObjs
def build_constraints(transl_vars2i, rowObjs, colObjs):
    n = len(transl_vars2i)
    col_constraints = len(colObjs)
    row_constraints  = len(rowObjs)
    A = matrix(0.0, (row_constraints+col_constraints , n))
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
    return A
    
    
if __name__=="__main__":
    


