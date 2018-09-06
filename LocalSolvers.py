from cvxopt import spmatrix,matrix
from cvxopt.solvers import qp,lp
from helpers import identityHash,swap
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


class LocalSolver():

    @classmethod
    def initializeLocalVariables(SolverClass,Sij,initvalue,N,rho):

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
   
            return [(splitIndex,(SolverClass(objectives,rho),P,Phi,stats))]

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

    

    def solveWithRowColumnConstraints(self):
	variables = self.pvariables.keys()
	v1,v2 = zip(*variables)
	v1 = set(v1)
	v2 = set(v2)
	
	row = 0
        tuples = []
	for v in v1:
	    rowv = [  (i,j) for (i,j) in variables if i==v ]
	    print rowv
	    tuples += [(row,self.pvariables[x],1.0)  for x in rowv]
	    row += 1

	for v in v2:
	    colv = [  (i,j) for (i,j) in variables if j==v ]
	    print colv
	    tuples += [(row,self.pvariables[x],1.0)  for x in colv]
	    row += 1

	#print tuples
	I,J,V = zip(*tuples)
	print I,J,V
	A = spmatrix(V,I,J)
	b = matrix(1.0,size=(row,1))
	#print(np.matrix(matrix(A)))

	print('p='+str(row)+' rank(A)='+str(matrix_rank(np.matrix(matrix(A)))))
        q = [0.5] * self.numt
	q += [0.0 for key in self.pvariables]
	q = matrix(q,size=(self.numt+self.nump, 1))
	result = lp(q,self.G,self.h,A,b)	
	
	sol = result['x']
	newp= dict( ( (key, sol[self.pvariables[key]]  )  for  key in self.pvariables ))	

	stats = {}
	stats[result['status']]=1.0
	localval = self.evaluate(newp)
	stats['obj'] = localval

	return newp,stats


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Local Solver Test',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph1',help = 'File containing first graph')
    parser.add_argument('graph2',help = 'File containing second graph')
    parser.add_argument('G',help = 'Constraint graph')
    parser.add_argument('--N',default=1,type=int, help='Level of parallelism')
    parser.add_argument('--rho',default=1.0,type=float, help='rho')


    args = parser.parse_args()
    sc = SparkContext(appName='Local Solver Test')
    
    graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval)
    graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval)
    G = sc.textFile(args.G,minPartitions=args.N).map(eval)

    objectives = list(SijGenerator(graph1,graph2,G,args.N).collect())
    
    start = time()	
    L2 = LocalL2Solver(objectives,args.rho)
    end = time()
    print "L2 initialization in ",end-start,'seconds.'
    
    start = time()	
    FL2 = FastLocalL2Solver(objectives,args.rho)
    end = time()
    print "FL2 initialization in ",end-start,'seconds.'

    n = len(L2.variables())
    Z = dict([ (var,1.0/n) for var in L2.variables()])
    
    start = time()	
    newp,stats= L2.solve(Z)
    end = time()
    print 'L2 solve in',end-start,'seconds, stats:',stats
	
	
    start = time()	
    newp,stats= FL2.solve(Z)
    end = time()
    print 'FL2 solve in',end-start,'seconds, stats:',stats
