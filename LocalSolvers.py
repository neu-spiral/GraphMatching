from cvxopt import spmatrix,matrix
from cvxopt.solvers import qp,lp
from helpers import identityHash,swap,mergedicts,identityHash,projectToPositiveSimplex,readfile
import numpy as np
from numpy.linalg import solve as linearSystemSolver,inv
import logging
from debug import logger,Sij_test
from numpy.linalg import matrix_rank
from scipy.sparse import coo_matrix,csr_matrix
from pprint import pformat
from time import time
import argparse
from pyspark import SparkContext

def SijGenerator(graph1,graph2,G,N):
    #Compute S_ij^1 and S_ij^2
    Sij1 = G.join(graph1).map(lambda (k,(j,i)): ((i,j),(k,j))).groupByKey().flatMapValues(list).partitionBy(N)
    Sij2 = G.map(swap).join(graph2).map(lambda (k,(i,j)): ((i,j),(i,k))).groupByKey().flatMapValues(list).partitionBy(N) 

    #Do an "outer join"
    Sij = Sij1.cogroup(Sij2,N).mapValues(lambda (l1,l2):(list(set(l1)),list(set(l2))))
    return Sij
def General_LASSO(D, y, rho):
    """
        Solve the problm:
             min_beta \|y-beta\|_2^2 + rho \|D beta\|_1
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
            if a_i/DEN_plus>=0 and a_i/DEN_plus<=lambda_k:
                t_hit[coordinate_i] = a_i/DEN_plus
            else:
                t_hit[coordinate_i] = a_i/DEN_minus
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
    def next_kink(t_hit, t_leave):
        """Given hitting and leaving times find the next lambda, corresponding to a kink"""
        max_t = -1.0
        leaving_coordinate = None
        for j in t_hit:
            if t_hit[j]>max_t:
                max_t = t_hit[j]
                coordinate = j
        for j in t_leave:
            if t_leave[j]>max_t:
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
        status, coord, lambda_k = next_kink(t_hit, t_leave)
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
    u_minus_B = LSQ(rho)
    p_minus, N = u_minus_B.shape

    for i in range(p_minus):
        i_real = trasnslate_itoj_minusB[i]
        u[i_real] = u_minus_B[i]
    for (i, sgn_i) in B_S:
        u[i] = sgn_i * rho
    sol = y- D.transpose()*u
    return sol, u, eval_dual(u, D, y) 
    
        
    

        
class LocalSolver():

    @classmethod
    def initializeLocalVariables(cls,Sij,initvalue,N,rho):

        if logger.getEffectiveLevel()==logging.DEBUG:
	     logger.debug(Sij_test(G,graph1,graph2,Sij))

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

        partitioned =  Sij.partitionBy(N)
	createVariables = partitioned.mapPartitionsWithIndex(createLocalPandPhiVariables)
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
 
	q = matrix(0.0,size=(self.nump, 1))
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
	q = matrix(0.0,size=(self.nump, 1))
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
        for key in pvariables:
	    ppos = pvariables[key]
	    tuples.append( (row,ppos,-1.0)  )
	    row += 1

        for key in pvariables:
	    ppos = pvariables[key]
	    tuples.append( (row,ppos,1.0)  )
	    row += 1

        # Matrices named as in in cvxopt.solver.qp
	I,J,vals = zip(*tuples)
	self.G = spmatrix(vals,I,J)  
	self.h = matrix([0.0]*(row-nump)+[1.0]*nump,size=(row,1))
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
        for i in range(self.numt):
	    q[i] = 0.5
	for key,ppos in self.pvariables.iteritems():
	    if zbar!=None:
		q[ppos] = -rho*zbar[key]
	    else:
		q[ppos] = 0.0
	    result = qp(rho*self.P,q,self.G,self.h)
	
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


    def initializeLocalVariables(SolverClass,G,initvalue,N,rho):
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

            
        partitioned = G.partitionBy(N)
        createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualRowVariables)
        PrimalDualRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
        return PrimalDualRDD

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
        stats['vars_lt_zero']=0
        stats['vars_gt_one']=0
        stats['rows_gt_one']=0
        for row in self.objectives:       
            zrow = {}
            for col in self.objectives[row]:
                zrow[(row,col)] = zbar[(row,col)]
            xrow = projectToPositiveSimplex(zrow,1.0)
            stats['vars_lt_zero']  += sum( (val<0.0 for val in xrow.values()) )
            stats['vars_gt_one']  += sum( (val>1.0 for val in xrow.values()) )
            stats['rows_gt_one']  += sum( xrow.values())> 1.0 
            res = res + xrow.items()
        return (dict(res),stats) 

    def evaluate(self,z):
        """ Evaluate objective. This returns True if z is feasible, False otherwise. 

        """
        for row in self.objectives:       
            for col in self.objectives[row]:
                if z[(row,col)]<0.0 or z[(row,col)]>1.0:
                    return False
            totsum = sum( (z[(row,col)] for col in self.objectives[row] ) )
            if totsum > 1.0:
                    return False
        return True

class LocalColumnProjectionSolver(LocalSolver):
    """ A class for projecting rows to the simplex."""
    @classmethod


    def initializeLocalVariables(SolverClass,Ginv,initvalue,N,rho):
        """ Produce an RDD containing solver, primal-dual variables, and some statistics.  
        """
        def createLocalPrimalandDualColVariables(splitIndex, iterator):
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

            
        partitioned = G.map(swap).partitionBy(N)
        createVariables = partitioned.mapPartitionsWithIndex(createLocalPrimalandDualColumnVariables)
        PrimalDualRDD = createVariables.partitionBy(N,partitionFunc=identityHash).cache()
        return PrimalDualRDD

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
            xcol = projectToPositiveSimplex(zcol,1.0)
            stats['vars_lt_zero']  += sum( (val<0.0 for val in xcol.values()) )
            stats['vars_gt_one']  += sum( (val>1.0 for val in xcol.values()) )
            stats['cols_gt_one']  += sum(xcol.values()) > 1.0
            res = res.extend(xcol.items())
        return (dict(res),stats) 

    def evaluate(self,z):
        """ Evaluate objective. This returns True if z is feasible, False otherwise. 

        """
        for col in self.objectives:       
            for row in self.objectives[row]:
                if z[(row,col)]<0.0 or z[(row,col)]>1.0:
                    return False
            totsum = sum( (z[(row,col)] for row in self.objectives[col] ) )
            if totsum > 1.0:
                    return False
        return True


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Local Solver Test',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('D', help='File contatining the matrix D')
    parser.add_argument('y', help='File containing the vector y')
    parser.add_argument('outfile', help='File to store the sol')
    parser.add_argument('--rho', help='Values of rho', type=float,default=1.) 
#    parser.add_argument('graph1',help = 'File containing first graph')
#    parser.add_argument('graph2',help = 'File containing second graph')
#    parser.add_argument('G',help = 'Constraint graph')
#    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
#    parser.add_argument('--rho',default=1.0,type=float, help='rho')
#
#
    args = parser.parse_args()
#    sc = SparkContext(appName='Local Solver Test')
#    
#    graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
#    graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)
#    G = sc.textFile(args.G,minPartitions=args.N).map(eval)
#
#    objectives = list(SijGenerator(graph1,graph2,G,args.N).collect())
#    
#    start = time()	
#    L2 = LocalL2Solver(objectives,args.rho)
#    end = time()
#    print "L2 initialization in ",end-start,'seconds.'
#    
#    start = time()	
#    FL2 = FastLocalL2Solver(objectives,args.rho)
#    end = time()
#    print "FL2 initialization in ",end-start,'seconds.'
#
#    n = len(L2.variables())
#    Z = dict([ (var,1.0/n) for var in L2.variables()])
#    
#    start = time()	
#    newp,stats= L2.solve(Z)
#    end = time()
#    print 'L2 solve in',end-start,'seconds, stats:',stats
#	
#	
#    start = time()	
#    newp,stats= FL2.solve(Z)
#    end = time()
#    print 'FL2 solve in',end-start,'seconds, stats:',stats
#    np.random.seed(1993)
#    P = 7
#    N = 8
#    D = np.matrix(np.random.random(P*N)).reshape(P,N)
#    y = np.matrix( np.random.random(N) ).reshape(N,1)
    D, p, N = readfile( args.D)
    y, N_1, one = readfile( args.y)
    rho = args.rho
    if N != N_1:
        print "Dimensions do not match."  
    D = np.matrix(D).reshape((p,N))
    y = np.matrix(y).reshape((N, 1))
    sol, u, dual_obj =   General_LASSO(D, y, rho)
    fp = open(args.outfile,'w')
    fp.write(str(dual_obj))
    fp.close() 


