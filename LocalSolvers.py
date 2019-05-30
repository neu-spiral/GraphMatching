#from cvxopt import spmatrix,matrix,solvers
#from cvxopt.solvers import qp,lp
#from scipy.sparse import coo_matrix,csr_matrix
import scipy.linalg
from helpers import cartesianProduct



from helpers import identityHash,swap,mergedicts,identityHash,projectToPositiveSimplex,readfile, writeMat2File,NoneToZero, vec_norm
import numpy as np
from numpy.linalg import solve as linearSystemSolver,inv
import logging
from debug import logger,Sij_test
from numpy.linalg import matrix_rank
#from scipy.sparse import coo_matrix,csr_matrix
from pprint import pformat
from time import time
import argparse

from pyspark import SparkContext,StorageLevel,SparkConf

def SijGenerator(graph1,graph2,G,N):
    #Compute S_ij^1 and S_ij^2
   # Sij1 = G.join(graph1).map(lambda (k,(j,i)): ((i,j),(k,j))).groupByKey().flatMapValues(list).partitionBy(N)
    Sij1 = G.join(graph1.map(swap) ).map(lambda (k,(j,i)): ((i,j),(k,j))).groupByKey().flatMapValues(list).partitionBy(N)
    Sij2 = G.map(swap).join(graph2).map(lambda (k,(i,j)): ((i,j),(i,k))).groupByKey().flatMapValues(list).partitionBy(N) 

    #Do an "outer join"
    Sij = Sij1.cogroup(Sij2,N).mapValues(lambda (l1,l2):(list(set(l1)),list(set(l2))))
    return Sij

def BuilCoeeficientsMat(objectives, variables, Wa=None, Wb=None, N=64):
    objectives = dict(objectives)
    translate_ij2coordinates = {}
    P = len(objectives)


    
    #Create a dictioanry for translating variables to coordinates.
    translate_ij2coordinates = dict([(var, i) for (i,var) in enumerate(variables)] )
    n_i = len(translate_ij2coordinates)
    #Create the structure matrix D.
    D = np.matrix( np.zeros((P, n_i)))
    row = 0
    for key in objectives:
        obj_i, obj_j = key
        if Wa ==None:
            [S1, S2] = objectives[key]
            for var in S1:
                var_k, var_j = var
                D[row, translate_ij2coordinates[var]] = +1.
        else:
            for k in range(N):
                var_k = unicode(k)
                var = var_k, obj_j
                D[row, translate_ij2coordinates[var]] = Wa[(obj_i, var_k)]
        if Wb==None:
            for var in  S2:
                var_i, var_k = var
                if  D[row, translate_ij2coordinates[var]]==0:
                    D[row, translate_ij2coordinates[var]] = -1.
                else:
                    D[row, translate_ij2coordinates[var]] -= 1.0
        else:
            for k in range(N):
                var_k = unicode(k)
                var = obj_i, var_k
                D[row, translate_ij2coordinates[var]] -= Wb[(var_k, obj_j)]
        row = row+1
    translate_coordinates2ij = dict([(translate_ij2coordinates[ij], ij) for ij in translate_ij2coordinates])
    return D, translate_ij2coordinates, translate_coordinates2ij
def compNormAPminusPB(objectives, P, Wa=None, Wb=None):
    norm_1 = 0.0
    for key in objectives:
        tmp = 0.0
        obj_i, obj_j = key
        [S1, S2] = objectives[key]
        if Wa == None:
            for var in S1:
                var_k, var_j = var
                tmp += P[var]
        else:
            for k in range(N):
                var_k = unicode(k)
                var = var_k, obj_j
                tmp +=  Wa[(obj_i, var_k)]*P[var]
        if Wb==None:
            for var in  S2:
                var_i, var_k = var
                tmp -= P[var]
        else:
            for k in range(N):
                var_k = unicode(k)
                var = obj_i, var_k
                tmp -= P[var]*Wb[(var_k, obj_j)] 
        norm_1 += abs(tmp)
    return norm_1
                              
         
def RowColObjectivesGenerator(G, initvalue, R=True):
    Primal = dict()
    Dual = dict()
    objectives = dict()

    for edge in G:
        row,column = edge
        Primal[edge] = initvalue
        Dual[edge] = 0.0
        if R:
            if row in objectives:
                objectives[row].append(column)
            else:
                objectives[row] = [column]
        else:
            if column in objectives:
                objectives[column].append(row)
            else:
                objectives[column] = [row]

    return objectives, Primal, Dual
def General_LASSO(D, y, rho):
    """
        Solve the problm:
             min_beta 0.5\|y-beta\|_2^2 + rho \|D beta\|_1
        It is the implementation  of Alg. 2 in THE SOLUTION PATH OF THE GENERALIZED LASSO, Tibshirani and Taylor.
        D is a numpy matrix. 
    """
    def find_DminusB(D, B_S):
        """Given the matrix D and the coordinates and signs in B_S return the matrix for -B set"""
        P, N = D.shape
        B_corrdinates  = [i for (i,sgn_i) in B_S]
        D_minusB = np.matrix( np.zeros((P-len(B_corrdinates), N)))
        j = 0
        trasnslate_itoj = {}
        for i in range(P):
            if i not in B_corrdinates:
                D_minusB[j,:] = D[i,:]
                trasnslate_itoj[j] = i
                j = j+1
        return trasnslate_itoj, D_minusB
                
    def find_D_B(D, B_S):
        """Given the matrix D and the coordinates and signs in B_S return the matrix for B set"""
        P, N = D.shape
        D_B = np.matrix(np.zeros((len(B_S), N)))
        j = 0
        trasnslate_itoj = {}
        for (i,sgn_i) in B_S:
            D_B[j,:] = D[i,:]
            trasnslate_itoj[j] = i
            j = j+1
        return trasnslate_itoj, D_B
    def HitTimes(D_minusB_sqrd, D_minusB_sqrd_psudoinv, D_minusB, D_B_times_sgn,  y, B_S, lambda_k, trasnslate_itoj_minusB):
        """
            Compute the hitting times (see Eq. (27))
        """
        P_minus, N = D_minusB.shape
        t_hit = {}
        
        a_i_s = D_minusB_sqrd_psudoinv*D_minusB*y 
        b_i_s = D_minusB_sqrd_psudoinv*D_minusB*D_B_times_sgn
        for i in range(P_minus):
            coordinate_i = trasnslate_itoj_minusB[i]
            a_i = float( a_i_s[i])
            b_i = float( b_i_s[i])
            DEN_plus = b_i + 1.
            DEN_minus = b_i - 1.
            
            #Make sure that the denominator is non-zero, otherwise, set it the hitting time to a negative value. 
            try:     
                t_plus = a_i/DEN_plus
            except ZeroDivisionError:
                t_plus = float('Inf')
            try:
                t_minus = a_i/DEN_minus
            except ZeroDivisionError:
                t_minus = float('Inf')
   
            #Pick the value which is positive and less than the current lambda_k            
            if t_minus<0 and t_plus>=0:
                t_hit[coordinate_i] = t_plus
            elif t_minus>=0 and t_plus<0:            
                t_hit[coordinate_i] = t_minus
            elif t_minus<0 and t_plus<0:
                print 'ERROR'
                break
            else:
                t_hit[coordinate_i] =  min(t_minus, t_plus)
        return t_hit
    def LeaveTimes(D_B, D_minus_B, D_minusB_sqrd_psudoinv, y, D_B_times_sgn, B_S, trasnslate_itoj_B):
        """Compute leaving times (see Eq. (29))"""
        P_minus, N = D_minusB.shape
        t_leave = {}
        if P_minus>0:
            precomp_c_i_s = (np.matrix(np.identity(N)) -D_minus_B.transpose()*D_minusB_sqrd_psudoinv * D_minus_B)*y
            precomp_d_i_s = (np.matrix(np.identity(N))  - D_minus_B.transpose()*D_minusB_sqrd_psudoinv * D_minus_B)*D_B_times_sgn
        else:
            precomp_c_i_s = y
            precomp_d_i_s = D_B_times_sgn
        trasnslate_jtoi_B = dict([(trasnslate_itoj_B[key], key) for key in trasnslate_itoj_B])
        for (i, sgn_i) in B_S:
            i_coordinate = trasnslate_jtoi_B[i]
            c_i = sgn_i * float( D_B[i_coordinate,:]*precomp_c_i_s)
            d_i = sgn_i * float(D_B[i_coordinate,:]*precomp_d_i_s)
            if c_i<0 and d_i<0:
                t_leave[i] = c_i/d_i
            else:
                t_leave[i] = 0.
        return t_leave
    def next_kink(t_hit, t_leave, lambda_k):
        """Given hitting and leaving times find the next lambda, corresponding to a kink. Make sure that leaving time is smaller than the current lambda."""
        max_t = -1.0
        leaving_coordinate = None
        for j in t_hit:
            if t_hit[j]>max_t:
                max_t = t_hit[j]
                coordinate = j
        for j in t_leave:
            if t_leave[j]>max_t and t_leave[j]<lambda_k:
                max_t = t_leave[j]
                leaving_coordinate = j
        lambda_k = max_t
        if leaving_coordinate == None:
            status = 'hit'
            coord = coordinate
        else:
            status = 'leave'
            coord = leaving_coordinate
        return status, coord, lambda_k
    def update_boundary_set(B_S, status, coord, sgn_coord):
        """Add or remove coord according to status to or form B_S"""
        if status == 'hit':
            B_S.append((coord, sgn_coord))
        else:
            B_S_dict = dict(B_S)
            B_S_dict.pop( coord)
            B_S = B_S_dict.items()
        return B_S
    def eval_dual(u, D, y):
        """"Evaluate the dual objective."""
        return 0.5 * np.linalg.norm(y-D.transpose()*u,2)**2 
    def eval_prim(sol, D, y, rho):
        return 0.5*np.linalg.norm(y-sol,2)**2+ rho*np.linalg.norm(D*sol,1)
        
    P, N = D.shape
    k = 0
    lambda_0 = float('Inf')
    #B_S keeps track of the coordinates on the boundary and their signs, i.e., it is a list of tuples (i th coordinate, sign of i th coordinate)
    B_S = []
    lambda_k = lambda_0
    while lambda_k>rho:
        trasnslate_itoj_minusB, D_minusB = find_DminusB(D, B_S)
       

        trasnslate_itoj_B, D_B = find_D_B(D, B_S)
        #If B is empty set D_B_times_sgn to zero.
        if len(B_S)>0: 
            D_B_times_sgn = sum([D[i,:]*s for (i,s) in B_S]).transpose()
        elif len(B_S)==0:
            D_B_times_sgn = np.matrix( np.zeros((N,1)))

            
        #If all coordinates are on the boundary, skip computation  of hitting times.   
        if len(B_S)<P:
            D_minusB_sqrd = D_minusB*D_minusB.transpose()
            D_minusB_sqrd_psudoinv = np.linalg.pinv(D_minusB_sqrd)

            #Compute hitting and leaving times
            t_hit = HitTimes(D_minusB_sqrd, D_minusB_sqrd_psudoinv, D_minusB, D_B_times_sgn, y, B_S, lambda_k, trasnslate_itoj_minusB)
        elif len(B_S)==P:
            D_minusB_sqrd = np.matrix (np.zeros((0,0)))
            D_minusB_sqrd_psudoinv = np.matrix (np.zeros((0,0)))
            t_hit = {}



        t_leave = LeaveTimes(D_B, D_minusB, D_minusB_sqrd_psudoinv, y, D_B_times_sgn, B_S, trasnslate_itoj_B)
        #Find the next hitting or leaving time
        status, coord, lambda_k = next_kink(t_hit, t_leave, lambda_k)
        if status=='hit':
            #Compute the least suare solution, see Eq. (26). It`s a linear functions of lambda
            LSQ = lambda lam: D_minusB_sqrd_psudoinv*D_minusB*(y-lam*D_B_times_sgn)
            trasnslate_jtoi = dict([(trasnslate_itoj_minusB[val], val) for val in trasnslate_itoj_minusB])
            i_coord = trasnslate_jtoi[coord]
            u_hat = LSQ(lambda_k)
            sgn_coord = np.sign( float( u_hat[i_coord]) )
        else:
            sgn_coord = 0.
        if lambda_k< rho:
            break
        #u_hat = u_hat_minusB(lambda_k)
        B_S = update_boundary_set(B_S, status, coord, sgn_coord)
        k = k+1 
    sol = np.zeros((N,1))
    u = np.zeros((P,1))

    #Compute the least suare solution, see Eq. (26). It`s a linear functions of lambda
    LSQ = lambda lam: D_minusB_sqrd_psudoinv*D_minusB*(y-lam*D_B_times_sgn)

    u_minus_B = LSQ(rho)
    p_minus, N = u_minus_B.shape

    for i in range(p_minus):
        i_real = trasnslate_itoj_minusB[i]
        u[i_real] = u_minus_B[i]
    for (i, sgn_i) in B_S:
        u[i] = sgn_i * rho

    u = np.matrix(u)
    sol = y- D.transpose()*u
    return sol
    
        
    

def General_LASSO_test(D, y, rho):
    """
        Solve the problm:
             min_beta 0.5\|y-beta\|_2^2 + rho \|D beta\|_1
        It is the implementation  of Alg. 2 in THE SOLUTION PATH OF THE GENERALIZED LASSO, Tibshirani and Taylor.
        D is a numpy matrix. 
    """
    def find_DminusB(D, B_S):
        """Given the matrix D and the coordinates and signs in B_S return the matrix for -B set"""
        P, N = D.shape
        B_corrdinates  = [i for (i,sgn_i) in B_S]
        D_minusB = np.matrix( np.zeros((P-len(B_corrdinates), N)))
        j = 0
        trasnslate_itoj = {}
        for i in range(P):
            if i not in B_corrdinates:
                D_minusB[j,:] = D[i,:]
                trasnslate_itoj[j] = i
                j = j+1
        return trasnslate_itoj, D_minusB

    def find_D_B(D, B_S):
        """Given the matrix D and the coordinates and signs in B_S return the matrix for B set"""
        P, N = D.shape
        D_B = np.matrix(np.zeros((len(B_S), N)))
        j = 0
        trasnslate_itoj = {}
        for (i,sgn_i) in B_S:
            D_B[j,:] = D[i,:]
            trasnslate_itoj[j] = i
            j = j+1
        return trasnslate_itoj, D_B
    def HitTimes(D_minusB_psudoinv, D_minusB, D_B_times_sgn,  y, B_S, lambda_k, trasnslate_itoj_minusB):
        """
            Compute the hitting times (see Eq. (27))
        """
        P_minus, N = D_minusB.shape
        t_hit = {}

        a_i_s = D_minusB_psudoinv*y
        b_i_s = D_minusB_psudoinv*D_B_times_sgn
        for i in range(P_minus):
            coordinate_i = trasnslate_itoj_minusB[i]
            a_i = float( a_i_s[i])
            b_i = float( b_i_s[i])
            DEN_plus = b_i + 1.
            DEN_minus = b_i - 1.

            t_plus = a_i/DEN_plus
            t_minus = a_i/DEN_minus
         
            if t_minus<0 and t_plus>=0:
                t_hit[coordinate_i] = t_plus
            elif t_minus>=0 and t_plus<0:
                t_hit[coordinate_i] = t_minus
            elif t_minus<0 and t_plus<0:
                print 'ERROR'
                break
            else:
                sgn_min = min(t_minus, t_plus)
                if sgn_min<lambda_k:
                    t_hit[coordinate_i] = sgn_min
                else:
                    print 'ERROR'
                    break

        return t_hit
    def LeaveTimes(D_B, D_minus_B, D_minusB_psudoinv, y, D_B_times_sgn, B_S, trasnslate_itoj_B):
        """Compute leaving times (see Eq. (29))"""
        P_minus, N = D_minusB.shape
        t_leave = {}
        if P_minus>0:
            precomp_c_i_s = (np.matrix(np.identity(N)) -D_minus_B.transpose()*D_minusB_psudoinv)*y
            precomp_d_i_s = (np.matrix(np.identity(N))  - D_minus_B.transpose()*D_minusB_psudoinv)*D_B_times_sgn
        else:
            precomp_c_i_s = y
            precomp_d_i_s = D_B_times_sgn
        trasnslate_jtoi_B = dict([(trasnslate_itoj_B[key], key) for key in trasnslate_itoj_B])
        for (i, sgn_i) in B_S:
            i_coordinate = trasnslate_jtoi_B[i]
            c_i = sgn_i * float( D_B[i_coordinate,:]*precomp_c_i_s)
            d_i = sgn_i * float(D_B[i_coordinate,:]*precomp_d_i_s)
            if c_i<0 and d_i<0:
                t_leave[i] = c_i/d_i
            else:
                t_leave[i] = 0.
        return t_leave
    def next_kink(t_hit, t_leave, lambda_k):
        """Given hitting and leaving times find the next lambda, corresponding to a kink. Make sure that leaving time is smaller than the current lambda."""
        max_t = -1.0
        leaving_coordinate = None
        for j in t_hit:
            if t_hit[j]>max_t and t_hit[j]<lambda_k:
                max_t = t_hit[j]
                coordinate = j
        for j in t_leave:
            if t_leave[j]>max_t and t_leave[j]<lambda_k:
                max_t = t_leave[j]
                leaving_coordinate = j
        lambda_k = max_t
        if leaving_coordinate == None:
            status = 'hit'
            coord = coordinate
        else:
            status = 'leave'
            coord = leaving_coordinate
        return status, coord, lambda_k
    def update_boundary_set(B_S, status, coord, sgn_coord):
        """Add or remove coord according to status to or form B_S"""
        if status == 'hit':
            B_S.append((coord, sgn_coord))
        else:
            B_S_dict = dict(B_S)
            B_S_dict.pop( coord)
            B_S = B_S_dict.items()
        return B_S
    def eval_dual(u, D, y):
        """"Evaluate the dual objective."""
        return 0.5 * np.linalg.norm(y-D.transpose()*u,2)**2
    def eval_prim(sol, D, y, rho):
        return 0.5*np.linalg.norm(y-sol,2)**2+ rho*np.linalg.norm(D*sol,1)

    P, N = D.shape
    k = 0
    lambda_0 = float('Inf')
    #B_S keeps track of the coordinates on the boundary and their signs, i.e., it is a list of tuples (i th coordinate, sign of i th coordinate)
    B_S = []
    lambda_k = lambda_0
    while lambda_k>rho:
        trasnslate_itoj_minusB, D_minusB = find_DminusB(D, B_S)
 


        trasnslate_itoj_B, D_B = find_D_B(D, B_S)

        #If B is empty set D_B_times_sgn to zero.
        if len(B_S)>0:
            D_B_times_sgn = sum([D[i,:]*s for (i,s) in B_S]).transpose()
        elif len(B_S)==0:
            D_B_times_sgn = np.matrix( np.zeros((N,1)))


        #If all coordinates are on the boundary, skip computation  of hitting times.   
        if len(B_S)<P:
            D_minusB_psudoinv = np.linalg.pinv(D_minusB).transpose()

            #Compute hitting and leaving times
            t_hit = HitTimes(D_minusB_psudoinv, D_minusB, D_B_times_sgn, y, B_S, lambda_k, trasnslate_itoj_minusB)
        elif len(B_S)==P:
            D_minusB_psudoinv = np.matrix (np.zeros((0,0)))
            t_hit = {}



        t_leave = LeaveTimes(D_B, D_minusB, D_minusB_psudoinv, y, D_B_times_sgn, B_S, trasnslate_itoj_B)
        #Find the next hitting or leaving time
        status, coord, lambda_k = next_kink(t_hit, t_leave, lambda_k)
        if status=='hit':
            #Compute the least suare solution, see Eq. (26). It`s a linear functions of lambda
            LSQ = lambda lam: D_minusB_psudoinv*(y-lam*D_B_times_sgn)
            trasnslate_jtoi = dict([(trasnslate_itoj_minusB[val], val) for val in trasnslate_itoj_minusB])
            i_coord = trasnslate_jtoi[coord]
            u_hat = LSQ(lambda_k)
            sgn_coord = np.sign( float( u_hat[i_coord]) )
        else:
            sgn_coord = 0.
       # print lambda_k,coord, t_hit, t_leave
        if lambda_k< rho:
            break
        #u_hat = u_hat_minusB(lambda_k)
        B_S = update_boundary_set(B_S, status, coord, sgn_coord)
        k = k+1
    sol = np.zeros((N,1))
    u = np.zeros((P,1))
    u_minus_B = LSQ(rho)
    p_minus, N = u_minus_B.shape

    for i in range(p_minus):
        i_real = trasnslate_itoj_minusB[i]
        u[i_real] = u_minus_B[i]
    for (i, sgn_i) in B_S:
        u[i] = sgn_i * rho
    sol = y- D.transpose()*u
    return sol
        
class LocalSolver():

    @classmethod
    def initializeLocalVariables(cls,Sij,initvalue,N,rho,prePartFunc=None):

#        if logger.getEffectiveLevel()==logging.DEBUG:
#	     logger.debug(Sij_test(G,graph1,graph2,Sij))

    	#Create local P and Phi variables
    	def createLocalPandPhiVariables(splitIndex, iterator):
    	    P = dict()
    	    Phi =dict()
	    stats = {}
    	    objectives = []

	    for ( edge, (l1,l2)) in iterator:
	        objectives.append( (edge, (l1,l2)) )	
	        for varindex in l1:
		    P[varindex] = initvalue
		    Phi[varindex] = 0.0
	        for varindex in l2:
		    P[varindex] = initvalue
		    Phi[varindex] = 0.0

	    stats['variables'] = len(P)
	    stats['objectives'] = len(objectives)
   
            return [(splitIndex,(cls(objectives,rho),P,Phi,stats))]

        if prePartFunc==None:
            partitioned =  Sij.partitionBy(N)
        else:
            partitioned =  Sij.partitionBy(N, partitionFunc=prePartFunc)
	createVariables = partitioned.mapPartitionsWithIndex(createLocalPandPhiVariables, preservesPartitioning=True)
        PPhiRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
	#PPhiRDD = Sij.mapPartitionsWithIndex(createLocalPandPhiVariables).cache()

	return PPhiRDD

    @classmethod
    def getclsname(cls):
        junk, clsname = str(cls).split('.')
        return clsname

    def __init__(self,objectives,rho= None):
	self.objectives = dict(objectives)
	self.rho = rho

    def __repr__(self):
        return '(' + str(self.objectives) + ',' + str(self.rho) + ')'
 
    def solve(self,zbar,rho):
    	pass

    def variables(self):
	pass

    def __str__(self):
	return self.__repr__()





class FastLocalL2Solver(LocalSolver):

    def __init__(self,objectives,rho):
	objectives = dict(objectives)
	self.rho = rho

	pvariables = set()
	for key in objectives:
	    (l1,l2) = objectives[key] 
	    pvariables.update(l1)
	    pvariables.update(l2)
	numo = len(objectives)
	nump = len(pvariables)
        pvarpos = dict(zip(list(pvariables),range(nump)))
	objectpos = dict(zip(list(objectives.keys()),range(numo)))

	W = coo_matrix((numo,nump))
	#add objectives to quadratic matrix

	data = []
        ilist =[]
        jlist =[]
	for key in objectives:
	    (l1,l2) = objectives[key] 
	
	    for pvar in l1:
		data.append(1.0)
		ilist.append(objectpos[key])
		jlist.append(pvarpos[pvar])
	    for pvar in l2:
		data.append(-1.0)
		ilist.append(objectpos[key])
		jlist.append(pvarpos[pvar])
	   
	W = coo_matrix((data,(ilist,jlist)))

	Wf = W.tocsr()
	WfT = W.T.tocsr()
	
	#M = (I_o + rho^{-1}  * WW^T)^-1
        M = inv(np.eye(numo) + Wf * WfT/rho) 
	
	self.objectives = objectives
	self.objectpos = objectpos
	self.pvarpos = pvarpos
	self.nump = nump
	self.numo = numo
	
	self.M = M
	self.Wf = Wf
	self.WfT = WfT

    def solve(self,zbar=None): 
 
#	q = matrix(0.0,size=(self.nump, 1))
        q = np.matrix(np.zeros((self.nmp, 1)))
	if zbar!= None:
	    for key,ppos in self.pvarpos.iteritems():
		q[ppos] = zbar[key]

	
        sol = q - 1/self.rho *(self.WfT  * (self.M * (self.Wf * q)))
	
	newp= dict( ( (key, float(sol[self.pvarpos[key]])  )  for  key in self.pvarpos ))	

	stats = {}
	localval = self.evaluate(newp)
	stats['pobj'] = localval
	
	if zbar!= None:
	   rhoerr = self.rho*sum( (pow(float(zbar[key]-newp[key]),2)  for key in self.pvarpos ) ) 
	   stats['rhoerr'] = rhoerr
           stats['proximalobj'] = localval+rhoerr
	return newp,stats

    def evaluate(self,z):
	#Objective value without penalty
	result = 0.0
	for key in self.objectives:
	    l1,l2 = self.objectives[key]
	    tmp = 0.0
	    for zkey in l1:
		tmp += z[zkey]
	    for zkey in l2:
		tmp -= z[zkey]
	    result += float(tmp)**2 
	return 0.5*result 

    def variables(self):
	return self.pvariables.keys()

    




class LocalL2Solver(LocalSolver):

    def __init__(self,objectives,rho=None):
	objectives = dict(objectives)
	self.rho = rho

	pvariables = set()
	for key in objectives:
	    (l1,l2) = objectives[key] 
	    pvariables.update(l1)
	    pvariables.update(l2)
	nump = len(pvariables)
        pvariables = dict(zip(list(pvariables),range(nump)))


	Qmat = np.zeros((nump,nump))
	#add objectives to quadratic matrix
	for key in objectives:
	    (l1,l2) = objectives[key] 
	
	    u = np.zeros((nump,1))
	    for pvar in l1:
		ppos = pvariables[pvar]
		u[ppos] = 1.0
	    for pvar in l2:
		ppos = pvariables[pvar]
		u[ppos] = -1.0
	   
            Qmat +=  np.outer(u,u)

	self.Qmat = Qmat
	
	self.objectives = objectives
	self.pvariables = pvariables
	self.nump = nump
	
    def solve(self,zbar=None,rho=None): 
	if rho is None:
	   rho = self.rho  
	#q = np.matrix(0.0,size=(self.nump, 1))
        q = np.matrix(np.zeros((self.nump, 1)))
	if zbar!= None:
	    for key,ppos in self.pvariables.iteritems():
		q[ppos] = rho*zbar[key]

	newQmat = self.Qmat+rho*np.eye(self.nump)
	
        sol = linearSystemSolver(newQmat,q)
	
	newp= dict( ( (key, float(sol[self.pvariables[key]])  )  for  key in self.pvariables ))	

	stats = {}
	localval = self.evaluate(newp)
	stats['pobj'] = localval
	
	if zbar!= None:
	   rhoerr = rho*sum( (pow(float(zbar[key]-newp[key]),2)  for key in self.pvariables ) ) 
	   stats['rhoerr'] = rhoerr
           stats['proximalobj'] = localval+rhoerr
	return newp,stats

    def evaluate(self,z):
	#Objective value without penalty
	result = 0.0
	for key in self.objectives:
	    l1,l2 = self.objectives[key]
	    tmp = 0.0
	    for zkey in l1:
		tmp += z[zkey]
	    for zkey in l2:
		tmp -= z[zkey]
	    result += float(tmp)**2 
	return 0.5*result 

    def variables(self):
	return self.pvariables.keys()

    

class LocalL1Solver(LocalSolver):
    def __init__(self, objectives, rho=None):
        objectives = dict(objectives)
        #Create a dictioanry for translating variables to coordinates.
        n_i = 0
        translate_ij2coordinates = {}
        P = len(objectives)
        for key in objectives:
            [S1, S2] = objectives[key]
            for var in S1:
                if var  not in translate_ij2coordinates:
                    translate_ij2coordinates[var] = n_i
                    n_i = n_i+1
            for var in S2:
                if var not in translate_ij2coordinates:
                    translate_ij2coordinates[var] = n_i
                    n_i = n_i+1
        #Create the structure matrix D.
        D = np.matrix( np.zeros((P, n_i)))
        row = 0
        for key in objectives:
            [S1, S2] = objectives[key]
            for var in S1:
                D[row, translate_ij2coordinates[var]] = +1.
            for var in  S2:
                if  D[row, translate_ij2coordinates[var]]==0:
                    D[row, translate_ij2coordinates[var]] = -1.
                else:
                    D[row, translate_ij2coordinates[var]] = 0.
            row = row+1
        self.objectives = objectives
        self.D = D  
        self.translate_ij2coordinates = translate_ij2coordinates
        self.translate_coordinates2ij = dict([(translate_ij2coordinates[key], key) for key in translate_ij2coordinates])
        self.num_variables = len(translate_ij2coordinates)
        self.rho = rho
    def solve(self,zbar=None,rho=None):
        y = np.matrix( np.zeros((self.num_variables,1)))
        for var in zbar:
            y[self.translate_ij2coordinates[var]] = zbar[var]    
        sol = General_LASSO(self.D, y, 1./rho)
        #### 
        newp = dict(  [(self.translate_coordinates2ij[i], float(sol[i])) for i in self.translate_coordinates2ij] )
        stats = {}
        localval = self.evaluate(newp)
        stats['pobj'] = localval
        if zbar!= None:
           rhoerr = rho*sum( (pow((zbar[key]-newp[key]),2)  for key in newp ) )
           stats['rhoerr'] = rhoerr
           stats['proximalobj'] = localval+rhoerr
        return newp,stats
        
            
        
        
    def evaluate(self,z):
        #Objective value without penalty
        result = 0.0
        for key in self.objectives:
            l1,l2 = self.objectives[key]
            tmp = 0.0
            for zkey in l1:
                tmp += z[zkey]
            for zkey in l2:
                tmp -= z[zkey]
            result += np.abs(tmp)
        return result
    def variables(self):
        return self.translate_ij2coordinates.keys()
class LocalLSSolver(LocalSolver):
    """A class for updating P variables in the inner ADMM"""
    @classmethod
    def initializeLocalVariables(cls,Sij,initvalue,N,rho,rho_inner,prePartFunc=None):

        if logger.getEffectiveLevel()==logging.DEBUG:
             logger.debug(Sij_test(G,graph1,graph2,Sij))

        #Create local P and Phi variables
        def createLocalPandPhiVariables(splitIndex, iterator):
            P = dict()
            Y = dict()
            Phi =dict()
            Upsilon = dict()
            stats = {}
            objectives = []

            for ( edge, (l1,l2)) in iterator:
                objectives.append( (edge, (l1,l2)) )
                Upsilon[edge] = 0.0
                tmp_edge_val = 0. 
                for varindex in l1:
                    P[varindex] = initvalue
                    Phi[varindex] = 0.0
                    tmp_edge_val = tmp_edge_val+P[varindex]
                for varindex in l2:
                    P[varindex] = initvalue
                    Phi[varindex] = 0.0
                    tmp_edge_val = tmp_edge_val-P[varindex]
                Y[edge] = tmp_edge_val
            stats['variables'] = len(P)
            stats['objectives'] = len(objectives)

            return [(splitIndex,(cls(objectives,rho,rho_inner),P,Y,Phi,Upsilon,stats))]

        if prePartFunc==None:
            partitioned =  Sij.partitionBy(N)
        else:
            partitioned =  Sij.partitionBy(N, partitionFunc=prePartFunc)
        createVariables = partitioned.mapPartitionsWithIndex(createLocalPandPhiVariables, preservesPartitioning=True)
        PYPhiUpsilonRDD = createVariables.cache()
       # PYPhiUpsilonRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
        #PPhiRDD = Sij.mapPartitionsWithIndex(createLocalPandPhiVariables).cache()
        return PYPhiUpsilonRDD
        

    def __init__(self, objectives,rho=None, rho_inner=None):
        objectives = dict(objectives)
        #Create a dictioanry for translating variables to coordinates.
        n_i = 0
        p_i = 0
        translate_ij2coordinates = {}
        translate_ij2coordinates_Y = {}
        P = len(objectives)
        for key in objectives:
            [S1, S2] = objectives[key]
            for var in S1:
                if var  not in translate_ij2coordinates:
                    translate_ij2coordinates[var] = n_i
                    n_i = n_i+1
            for var in S2:
                if var not in translate_ij2coordinates:
                    translate_ij2coordinates[var] = n_i
                    n_i = n_i+1
            translate_ij2coordinates_Y[key] = p_i
            p_i = p_i +1
        #Create the structure matrix D and the pre-compute the A matrix in the leat-square problem. 
        D = np.matrix( np.zeros((P, n_i)))
        row = 0
        for key in objectives:
            [S1, S2] = objectives[key]
            for var in S1:
                D[row, translate_ij2coordinates[var]] = +1.
            for var in  S2:
                if  D[row, translate_ij2coordinates[var]]==0:
                    D[row, translate_ij2coordinates[var]] = -1.
                else:
                    D[row, translate_ij2coordinates[var]] = 0.
            row = row+1
        #invA = (rho I + rho_inner D.T * D)^-1
        invA = inv(rho* np.matrix(np.identity(n_i)) + rho_inner * D.T * D)
       
        self.D = D
        self.invA =invA
        self.objectives = objectives
        self.translate_ij2coordinates = translate_ij2coordinates
        self.translate_coordinates2ij = dict([(translate_ij2coordinates[key], key) for key in translate_ij2coordinates])
        self.translate_ij2coordinates_Y = translate_ij2coordinates_Y
        self.translate_coordinates2ij_Y = dict([(translate_ij2coordinates_Y[key], key) for key in translate_ij2coordinates_Y])
        self.num_variables = len(translate_ij2coordinates)
        self.rho = rho
        self.rho_inner = rho_inner
    def solve(self, Y, zbar, Upsilon, rho, rho_inner):
        """Solve the follwing quadratic problem:
               min_P 0.5*rho \|P-zbar\|_2^2 + 0.5*rho_inner \| D*P - (Y+Upsilon)\|_2^2 
        """
        N_i = len(self.translate_ij2coordinates)
        P_i = len(self.translate_ij2coordinates_Y)
        zbar_vec = np.matrix(np.zeros((N_i,1)))
        Y_vec = np.matrix(np.zeros((P_i,1)))
        Upsilon_vec = np.matrix(np.zeros((P_i,1)))

        for i in range(N_i):
            zbar_vec[i] = zbar[self.translate_coordinates2ij[i]]
        for i in range(P_i):
            Y_vec[i] = Y[self.translate_coordinates2ij_Y[i]]
            Upsilon_vec[i] = Upsilon[self.translate_coordinates2ij_Y[i]]
        p_vec = self.invA *(rho * zbar_vec + rho_inner * self.D.T *(Y_vec+Upsilon_vec))
        newp = dict( [(self.translate_coordinates2ij[i], float(p_vec[i]) ) for i in range(N_i)])
        rhoerr = rho/2.* float( (p_vec-zbar_vec).T * (p_vec-zbar_vec))
        stats = {'rhoerr':rhoerr}
        return (newp, stats)
    def variables(self):
        return (self.translate_ij2coordinates.keys(), self.translate_ij2coordinates_Y.keys())
    def evaluate(self, z, p):
        """Return p-norm of Y^(m) to the power of p"""
        s = 0.
        for edge in self.objectives:
            Y_elem = 0.0
            (S1, S2) = self.objectives[edge]
            for key in S1:
                Y_elem += z[key]
            for key in S2:
                Y_elem -= z[key]
            s += abs(Y_elem)**p
        return s
    def __repr__(self):
        return '(' + str(self.objectives) + ',' + str(self.rho) +  ',' + str(self.rho_inner) + ')'
    def __str__(self):
        return self.__repr__()
                        
        
        
        
        
         
        
        


class LocalL1Solver_Old(LocalSolver):

    def __init__(self,objectives,rho=None):
	objectives = dict(objectives)
	
        tvariables = objectives.keys()
	numt = len(tvariables)
        tvariables = dict(zip(tvariables,range(numt)))

	pvariables = set()
	for key in objectives:
	    (l1,l2) = objectives[key] 
	    pvariables.update(l1)
	    pvariables.update(l2)
	nump = len(pvariables)
        pvariables = dict(zip(list(pvariables),range(numt,numt+nump)))


	#create constraint matrix
	row = 0
	tuples = []
	for key in objectives:
	    tpos = tvariables[key]
	    (l1,l2) = objectives[key] 
	
	    tuples.append( (row,tpos,-1.0))	
	    tuples.append( (row+1,tpos,-1.0))	
	    for pvar in l1:
		ppos = pvariables[pvar]
		tuples.append( (row,ppos,1.0) )
		tuples.append( (row+1,ppos,-1,0) )
	    for pvar in l2:
		ppos = pvariables[pvar]
		tuples.append( (row,ppos,-1.0) )
		tuples.append( (row+1,ppos,1,0) )
	    row = row+2
	
        # Add [0,1] constraints
        #for key in pvariables:
        #    ppos = pvariables[key]
        #    tuples.append( (row,ppos,-1.0)  )
        #    row += 1

        #for key in pvariables:
        #    ppos = pvariables[key]
        #    tuples.append( (row,ppos,1.0)  )
        #    row += 1

        # Matrices named as in in cvxopt.solver.qp
	I,J,vals = zip(*tuples)
	self.G = spmatrix(vals,I,J)  
	#self.h = matrix([0.0]*(row-nump)+[1.0]*nump,size=(row,1))
        ## Try without [0,1] constraint
        self.h = matrix([0.0]*row, size=(row,1))
	self.P = spmatrix(1.0,range(numt,numt+nump),range(numt,numt+nump))
	
	self.objectives = objectives
	self.rho = rho
	self.tvariables = tvariables
	self.pvariables = pvariables
	self.numt = numt
	self.nump = nump
	
    def solve(self,zbar=None,rho=None):
	if rho is None:
	    rho = self.rho
   
	q = matrix(0.0,size=(self.numt+self.nump, 1))
        solvers.options['show_progress'] = False
        for i in range(self.numt):
	    q[i] = 1.0
	for key,ppos in self.pvariables.iteritems():
	    if zbar!=None:
		q[ppos] = -rho*zbar[key]
	    else:
		q[ppos] = 0.0
	    result = solvers.qp(rho*self.P,q,self.G,self.h)
	
	sol = result['x']
	newp= dict( ( (key, sol[self.pvariables[key]]  )  for  key in self.pvariables ))	

	stats = {}
	stats['status:'+result['status']]=1.0
	localval = self.evaluate(newp)
	stats['pobj'] = localval
	
	if zbar!= None:
	   rhoerr = rho*sum( (pow((zbar[key]-newp[key]),2)  for key in self.pvariables ) ) 
	   stats['rhoerr'] = rhoerr
           stats['proximalobj'] = localval+rhoerr
	return newp,stats

    def evaluate(self,z):
	#Objective value without penalty
	result = 0.0
	for key in self.objectives:
	    l1,l2 = self.objectives[key]
	    tmp = 0.0
	    for zkey in l1:
		tmp += z[zkey]
	    for zkey in l2:
		tmp -= z[zkey]
	    result += np.abs(tmp) 
	return result 

    def variables(self):
	return self.pvariables.keys()

    

#    def solveWithRowColumnConstraints(self):
#	variables = self.pvariables.keys()
#	v1,v2 = zip(*variables)
#	v1 = set(v1)
#	v2 = set(v2)
#	
#	row = 0
#        tuples = []
#	for v in v1:
#	    rowv = [  (i,j) for (i,j) in variables if i==v ]
#	    print rowv
#	    tuples += [(row,self.pvariables[x],1.0)  for x in rowv]
#	    row += 1
#
#	for v in v2:
#	    colv = [  (i,j) for (i,j) in variables if j==v ]
#	    print colv
#	    tuples += [(row,self.pvariables[x],1.0)  for x in colv]
#	    row += 1
#
#	#print tuples
#	I,J,V = zip(*tuples)
#	print I,J,V
#	A = spmatrix(V,I,J)
#	b = matrix(1.0,size=(row,1))
#	#print(np.matrix(matrix(A)))
#
#	print('p='+str(row)+' rank(A)='+str(matrix_rank(np.matrix(matrix(A)))))
#        q = [0.5] * self.numt
#	q += [0.0 for key in self.pvariables]
#	q = matrix(q,size=(self.numt+self.nump, 1))
#	result = lp(q,self.G,self.h,A,b)	
#	
#	sol = result['x']
#	newp= dict( ( (key, sol[self.pvariables[key]]  )  for  key in self.pvariables ))	
#
#	stats = {}
#	stats[result['status']]=1.0
#	localval = self.evaluate(newp)
#	stats['obj'] = localval
#
#	return newp,stats


class LocalRowProjectionSolver(LocalSolver):
    """ A class for projecting rows to the simplex."""
    @classmethod


    def initializeLocalVariables(SolverClass,G,initvalue,N,rho,D=None,lambda_linear=1.0):
        """ Produce an RDD containing solver, primal-dual variables, and some statistics.  
        """
        def createLocalPrimalandDualRowVariables(splitIndex, iterator):
            Primal = dict()
            Dual = dict()
            objectives = dict()
            stats = dict()

            for edge in iterator:
                row,column = edge
                Primal[edge] = initvalue
                Dual[edge] = 0.0
                if row in objectives:
                    objectives[row].append(column)
                else:
                    objectives[row] = [column]

            stats['variables'] = len(Primal)
            stats['objectives'] = len(objectives)
            return [(splitIndex,(SolverClass(objectives,rho),Primal,Dual,stats))]
        def createLocalPrimalandDualRowVariables_withD(splitIndex, iterator):
            Primal = dict()
            Dual = dict()
            objectives = dict()
            stats = dict()
            D_local = dict()

            for (row, (column, d_rowcol)) in iterator:
                edge = (row, column)
                Primal[edge] = initvalue
                Dual[edge] = 0.0
                D_local[edge] = d_rowcol
                if row in objectives:
                    objectives[row].append(column)
                else:
                    objectives[row] = [column]

            stats['variables'] = len(Primal)
            stats['objectives'] = len(objectives)
            return [(splitIndex,(SolverClass(objectives,rho,D_local,lambda_linear),Primal,Dual,stats))]

        partitioned = G.partitionBy(N)
        if D == None: 
            createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualRowVariables)
        else:
            D = D.rightOuterJoin(partitioned.map(lambda pair: (pair, 1)).partitionBy(N) ).mapValues(lambda (val, dummy ): NoneToZero(val))
            partitioned = D.map(lambda ((row, col), d_rowcol): (row, (col, d_rowcol)) ).partitionBy(N)
            createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualRowVariables_withD)

        PrimalDualRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
        return PrimalDualRDD
    def __init__(self, objectives,rho=None,D_local=None,lambda_linear=1.0):
        self.objectives = objectives
        self.rho = rho
        self.D_local = D_local
        self.lambda_linear = lambda_linear
    def variables(self):
        """Return variables participating in local optimization"""
	varbles = []
        for row in self.objectives:
            for col in self.objectives[row]:
                varbles.append((row,col))
        return varbles


    def solve(self,zbar,rho=None):
        """ Solve optimization problems (for each relevant row):

                  min_x ||x -z[row]||_2^2 s.t. 
                        \sum x_i=1, x_i >=0 for all i 

           This should return the "unfolded" x, along with some statistics. This is basically projection of each row onto the simplex.     
        """
        res=[]
        stats={}

        stats['rows_gt_one']=0
        stats['vars_lt_zero']=0
        stats['vars_gt_one']=0
        for row in self.objectives:       
            zrow = {}
            for col in self.objectives[row]:
                zrow[(row,col)] = zbar[(row,col)]
                if self.D_local != None:
                    zrow[(row,col)] -= 0.5 * self.D_local[(row,col)]*self.lambda_linear/rho
            xrow = projectToPositiveSimplex(zrow,1.0)
            stats['vars_lt_zero']  += sum( (val<0.0 for val in xrow.values()) )
            stats['vars_gt_one']  += sum( (val>1.0 for val in xrow.values()) )
            stats['rows_gt_one']  += sum( xrow.values())> 1.0 
            res = res + xrow.items()
        return (dict(res),stats) 

    def evaluate(self,z):
        """ Evaluate objective. This returns True if z is feasible, False otherwise. 

        """
        if self.D_local != None:
            return self.evaluateLinear(z)
        for row in self.objectives:       
            for col in self.objectives[row]:
                if z[(row,col)]<0.0 or z[(row,col)]>1.0:
                    return False
            totsum = sum( (z[(row,col)] for col in self.objectives[row] ) )
            if totsum > 1.0:
                    return False
        return True
    def evaluateLinear(self, z):
        """Evaluate the linear term in case it is passed.
    
        """
        res = 0.0
        for edge in self.D_local:
            res += self.D_local[edge]*z[edge]
        return res
    def __repr__(self):
        return '(' + str(self.objectives) + ',' + str(self.rho) + ',' + str(self.D_local) + ',' + str(self.lambda_linear) + ')'

class LocalColumnProjectionSolver(LocalSolver):
    """ A class for projecting rows to the simplex."""
    @classmethod


    def initializeLocalVariables(SolverClass,G,initvalue,N,rho,D=None,lambda_linear=0.0):
        """ Produce an RDD containing solver, primal-dual variables, and some statistics.  
        """
        def createLocalPrimalandDualColumnVariables(splitIndex, iterator):
            Primal = dict()
            Dual = dict()
            objectives = dict()
            stats = dict()

            for edgeInv in iterator:
                column,row = edgeInv
                edge = (row,column)
                Primal[edge] = initvalue
                Dual[edge] = 0.0
                if column in objectives:
                    objectives[column].append(row)
                else:
                    objectives[column] = [row]

            stats['variables'] = len(Primal)
            stats['objectives'] = len(objectives)
            return [(splitIndex,(SolverClass(objectives,rho),Primal,Dual,stats))]
        def createLocalPrimalandDualColumnVariables_withD(splitIndex, iterator):
            Primal = dict()
            Dual = dict()
            objectives = dict()
            stats = dict()
            D_local = dict()

            for (column, (row, d_rowcol)) in iterator:
                edge = (row,column)
                Primal[edge] = initvalue
                Dual[edge] = 0.0
                D_local[edge] = d_rowcol
                if column in objectives:
                    objectives[column].append(row)
                else:
                    objectives[column] = [row]

            stats['variables'] = len(Primal)
            stats['objectives'] = len(objectives)
            return [(splitIndex,(SolverClass(objectives,rho,D_local,lambda_linear),Primal,Dual,stats))]

            
        if D == None:
            partitioned = G.map(swap).partitionBy(N)
            createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualColumnVariables)
        else:
            partitioned = D.rightOuterJoin(G.map(lambda pair: (pair, 1))).mapValues(lambda (val, dummy ): NoneToZero(val))\
                .map(lambda ((row, col), val): (col, (row, val))).partitionBy(N)
          #  D = D.partitionBy(N, partitionFunc=lambda rowcol: rowcol[0])
            createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualColumnVariables_withD)
        PrimalDualRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
        return PrimalDualRDD
    def __init__(self, objectives,rho=None,D_local=None, lambda_linear=1.0):
        self.objectives = objectives
        self.rho = rho
        self.D_local = D_local
        self.lambda_linear = lambda_linear

    def variables(self):
        """Return variables participating in local optimization"""
	varbles = []
        for col in self.objectives:
            for row in self.objectives[col]:
                varbles.append((row,col))
        return varbles


    def solve(self,zbar,rho=None):
        """ Solve optimization problems (for each relevant col): 

                  min_x ||x -z[col]||_2^2 s.t. 
                        \sum x_i=1, x_i >=0 for all i 

           This should return the "unfolded" x, along with some statistics. This is basically projection of each column onto the simplex.     
        """
        res=[]
        stats={}
        stats['vars_lt_zero']=0
        stats['vars_gt_one']=0
        stats['cols_gt_one']=0
        for col in self.objectives:       
            zcol = {}
            for row in self.objectives[col]:
                zcol[(row,col)] = zbar[(row,col)]
                if self.D_local != None:
                    zcol[(row,col)] -= 0.5 * self.lambda_linear * self.D_local[(row,col)]/rho
            xcol = projectToPositiveSimplex(zcol,1.0)
            stats['vars_lt_zero']  += sum( (val<0.0 for val in xcol.values()) )
            stats['vars_gt_one']  += sum( (val>1.0 for val in xcol.values()) )
            stats['cols_gt_one']  += sum(xcol.values()) > 1.0
            res = res + xcol.items()
        return (dict(res),stats) 

    def evaluate(self,z):
        """ Evaluate objective. This returns True if z is feasible, False otherwise. 

        """
        for col in self.objectives:       
            for row in self.objectives[col]:
                if z[(row,col)]<0.0 or z[(row,col)]>1.0:
                    return False
            totsum = sum( (z[(row,col)] for row in self.objectives[col] ) )
            if totsum > 1.0:
                    return False
        return True
    def __repr__(self):
        return '(' + str(self.objectives) + ',' + str(self.rho) + ',' + str(self.D_local) + ',' + str(self.lambda_linear) + ')'

 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Serial ADMM Solver for Graph Matching (p=1)',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objectives',type=str,help='File containing objectives.')
    parser.add_argument('variables',type=str,help='File containing the variables support.')
    parser.add_argument('outfile',type=str,help='Output file')
     
    
    parser.add_argument('--N',default=64,type=int,help='Graph size.')
    parser.add_argument('--dist',type=str,help='File containing distace file.')
    parser.add_argument('--weights',type=str,help='File containing distace file.')
    parser.add_argument('--lamb', default=0.0, type=float, help='lambda parameter regularizing the linear term.')
    parser.add_argument('--maxiters',default=100,type=int, help='Maximum number of iterations')
    parser.add_argument('--epsilon', default=1.e-2, type=float, help='The accuracy for cvxopt solver.')
    parser.add_argument('--rho', default=5.0, type=float, help='Rho parameter in ADMM.')
    args = parser.parse_args()
  
    sc = SparkContext()
    sc.setLogLevel('OFF')
     
    tSt = time()
     #Generate objectives
    objectives = sc.textFile(args.objectives).map(eval).collect() 
    variables = sc.textFile(args.variables).map(eval).collect()


    #Initialize row and col variables.
    row_objectives, Q, Xi  = RowColObjectivesGenerator(variables, 1.0/args.N, True)
    col_objectives, T, Psi = RowColObjectivesGenerator(variables, 1.0/args.N, False)
    P = dict([(key, 1.0/args.N) for key in variables ])
    Phi = dict([(key, 0.0) for key in variables ])
    Z  = dict([(key, 1.0/args.N) for key in variables ])    

    #Load distance for the linear term.
    if args.dist != None:
        D = dict( sc.textFile(args.dist).map(eval).collect())
    else:
        D = None

    #load weights
    if args.weights != None:
        with open(args.weights) as weightF:
            Wb = pickle.load(weightF)
    else:
        Wb = None
    
    #Build the matrix for generlized LASSO problem 
    D_LASSO, trans_var2coord, trans_coord2var = BuilCoeeficientsMat(objectives=objectives, variables=variables, Wa=None, Wb=Wb, N=args.N)


    trace = {}
    print "Starting main ADMM iterations. Preprocessing done in %.2f (s)" %(time()-tSt)
  
    last = time()
    for t in  range(args.maxiters):
        trace[t] = {}
        Zbar_P = {}
        Zbar_Q = {}
        Zbar_T = {}
        #Adapt dual variables
        for key in Z:
            Phi[key] +=  P[key]-Z[key]
            Xi[key] += Q[key]-Z[key]
            Psi[key] += T[key]-Z[key]
            Zbar_P[key] = Z[key] - Phi[key]
            Zbar_Q[key] = Z[key] - Xi[key]
            Zbar_T[key] = Z[key] - Psi[key]
            if D != None: 
                Zbar_Q[key] -= 0.5 * args.lamb * D[key] / args.rho
                Zbar_T[key] -= 0.5 * args.lamb * D[key] /args.rho

  
        #Update P via generlaized LASSO
        y = np.matrix( np.zeros((len(variables),1)))
        for key in Z:
            y[trans_var2coord[key]] = Z[key]-Phi[key]
        sol_P = General_LASSO(D_LASSO, y, 1./args.rho)
        OBJNOLIN = np.linalg.norm(D_LASSO*sol_P, 1)
        
        LINOBJ = 0.0
        if D != None:
             for key in D:
                 LINOBJ += D[key]*P[key]
        P = dict([(trans_coord2var[coord], float(sol_P[coord])) for coord in trans_coord2var])
        
 
        #Update Q via projection
        Q = {}
        for row in row_objectives:
            zrow = {}
            for col in row_objectives[row]:
                var = (row, col)
                zrow[var] = Zbar_Q[var] 
            sol_row = projectToPositiveSimplex(zrow, 1.0)
            for key in sol_row:
               Q[key] = sol_row[key] 

        #Update T via projection
        T = {}
        for col in col_objectives:
            zcol = {}
            for row in col_objectives[col]:
                var = (row, col)
                zcol[var] = Zbar_T[var]
            sol_col = projectToPositiveSimplex(zcol, 1.0)
            for key in sol_col:
                T[key] = sol_col[key]

        #Update Z via averaging
        OldZ = Z
        Z = {}
        for key in OldZ: 
            Z[key] = P[key]+Phi[key] + Q[key]+Xi[key] + T[key]+Psi[key] 
            Z[key] = Z[key]/3.0

        #Compute primal and dual residuals
        primal_resid = 0.0
        dual_resid = 0.0
        for key in Z:
            primal_resid += (Z[key]-P[key])**2
            primal_resid += (Z[key]-Q[key])**2
            primal_resid += (Z[key]-T[key])**2
            dual_resid += (Z[key]-OldZ[key])**2
        primal_resid  = np.sqrt(primal_resid)
        dual_resid = np.sqrt(dual_resid)
        now = time()
        trace[t]['PRES'] = primal_resid
        trace[t]['DRES'] = dual_resid
        trace[t]['OBJNOLIN'] = OBJNOLIN
        trace[t]['OBJ'] = OBJNOLIN + args.lamb*LINOBJ
        print "Iteration %d, PRES is %.4f DRES is %.4f, OBJ is %.4f, norm is %.4f, iteration time is %f" %(t, primal_resid, dual_resid, OBJNOLIN + args.lamb*LINOBJ, OBJNOLIN, now-last)
        last = time()
        
    print "Finished ADMM iterations, saving the results."
    with open(args.outfile + '_trace', 'w') as outF:
        pickle.dump(trace, outF)
    with open(args.outfile + '_P', 'w') as outF:
        pickle.dump(P, outF)
        
            
       
            
                 
        

        
    
    
    

    
     

      

      
     


    

    
