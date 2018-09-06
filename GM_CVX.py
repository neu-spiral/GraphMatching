
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
def grad_Fij(objectives)
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
    n = len(variable_set)
    i = 0
    for var in variable_set:
        if var not in transl_vars2i.keys():
            transl_vars2i[var] = i
            i += 1
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
           
    
    
if __name__=="__main__":
    


