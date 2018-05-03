from cvxopt import spmatrix,matrix
from cvxopt.solvers import qp,lp
from helpers import identityHash,swap,mergedicts,identityHash
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


class ParallelSolver():
    """ A class for a parallel solver object. This object stores an RDD containing "local" data per partition, captured via a local solver object.
       The RDD also stores primal and dual variables associated with the arguments of this local solver function, as well as statistics reported by 
       the last computation of the local solver. The class can be used as an interface to add "homogeneous" objectives in the consensus admm algorithm,
       that can be executed in parallel
    """
    def __init__(self,LocalSolverClass,data,initvalue,N,rho,alpha,lean=False):
        """Class constructor. It takes as an argument a local solver class, data (of a form understandable by the local solver class), an initial value for the primal variables, and a boolean value; the latter can be used to suppress the evaluation of the objective. 
        """
        self.SolverClass=LocalSolverClass
        self.PrimalDualRDD =  LocalSolver.initializeLocalVariables(data,initvalue,N,rho).cache()    #LocalSolver class should implement class method initializeLocalVariables
        self.rho = rho
        self.alpha = alpha
        self.N = N
        self.silent=silent
        self.lean=lean
        self.varsToPartitions = self.PrimalDualRDD.flatMapValues( lambda  (solver,P,Phi,stats) : P.keys()).map(swap).partitionBy(self.N).cache() 


    def joinAndAdapt(self,ZRDD):
        """ Given a ZRDD, adapt the local primal and dual variables. The former are updated via the proximal operator, the latter via gradient ascent.
        """
        toUnpersist = self.PrimalDualRDD         #Old RDD is to be uncached

        #Send z to the appropriate partitions
        ZtoPartitions = ZRDD.join(self.varsToPartitions,numPartitions=self.N).map(lambda (key,(z,splitIndex)): (splitIndex, (key,z))).partitionBy(self.N,partitionFunc=identityHash).groupByKey().mapValues(list).mapValues(dict)
        PrimalDualOldZ=self.PrimalDualRDD.join(ZtoPartitions,numPartitions=self.N)


        if not self.lean:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z): solver.evaluate(Z)).reduce(add)  #local solver should implement evaluate



        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Phi,stats),Z): ( solver, P, dict( [ (key,Phi[key]+args.alpha *(P[key]-Z[key]))  for key in Phi  ]  ), Z))
        ZbarAndNewDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Phi,Z): ( solver, dict( [(key, Z[key]-Phi[key]) for key in Z]), Phi ))
        self.PrimalDualRDD = ZbarAndNewDual.mapValues( lambda  (solver,Zbar,Phi) : (solver,solver.solve(Zbar),Phi)).mapValues(lambda (solver,(P,stats),Phi): (solver,P,Phi,stats)).partitionBy(self.N,partitionFunc=identityHash).cache() #Solver should implement solve
        #Maybe partitioning is not needed?
        toUnpersist.unpersist()
  
        if not self.lean:
	    return (oldPrimalresidual,oldObjValue)
        else:
            return None

    def logstats(self):
        """  Return statistics from PrimalDualRDD. In particular, this returns the average, min, and maximum value of each statistic.
        """
        rdd = self.PrimalDualRDD
        
        statsonly =rdd.map(lambda (partitionid, (solver,P,Phi,stats)): stats)
        stats = statsonly.reduce(lambda x,y:  mergedicts(x,y))
        minstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,min))
        maxstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,max))
        return " ".join([ key+"= %s (%s/%s)" % (str(1.0*stats[key]/self.N),str(minstats[key]),str(maxstats[key]))   for key in stats])   	

    def getVars(self):
        """Return the primal variables associated with this RDD. To be used to compute the new consensus variable"""
        return self.PrimalDualRDD.flatMap(lambda (partitionId,(solver,P,Phi,stats)): [ (key, ( args.rhoP*( P[key]+Phi[key]),args.rhoP))    for key in P ]  )

