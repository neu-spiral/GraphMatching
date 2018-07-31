#from cvxopt import spmatrix,matrix
#from cvxopt.solvers import qp,lp
from helpers import identityHash,swap,mergedicts,identityHash
import numpy as np
from numpy.linalg import solve as linearSystemSolver,inv
import logging
from debug import logger,Sij_test
from numpy.linalg import matrix_rank
from pprint import pformat
from time import time
import argparse
from pyspark import SparkContext
from operator import add,and_
from LocalSolvers import LocalLpSolver
from proxOp import pnormOp,pnorm_proxop


class ParallelSolver():
    """ A class for a parallel solver object. This object stores an RDD containing "local" data per partition, captured via a local solver object.
       The RDD also stores primal and dual variables associated with the arguments of this local solver function, as well as statistics reported by 
       the last computation of the local solver. The class can be used as an interface to add "homogeneous" objectives in the consensus admm algorithm,
       that can be executed in parallel
    """
    def __init__(self,LocalSolverClass,data,initvalue,N,rho,silent=True,lean=False, RDD=None):
        """Class constructor. It takes as an argument a local solver class, data (of a form understandable by the local solver class), an initial value for the primal variables, and a boolean value; the latter can be used to suppress the evaluation of the objective. 
        """
        self.SolverClass=LocalSolverClass
        if RDD==None:
            self.PrimalDualRDD =  LocalSolverClass.initializeLocalVariables(data,initvalue,N,rho).cache()    #LocalSolver class should implement class method initializeLocalVariables
        else:
            self.PrimalDualRDD = RDD
        self.N = N
        self.silent=silent
        self.lean=lean
        self.varsToPartitions = self.PrimalDualRDD.flatMapValues( lambda  (solver,P,Phi,stats) : P.keys()).map(swap).partitionBy(self.N).cache() 

    def joinAndAdapt(self,ZRDD, alpha, rho,checkpoint = False):
        """ Given a ZRDD, adapt the local primal and dual variables. The former are updated via the proximal operator, the latter via gradient ascent.
        """
        toUnpersist = self.PrimalDualRDD         #Old RDD is to be uncached

        def adaptDual(solver, P, Phi, stats, Z, alpha):
            """Update the dual variables."""
            return ( solver, P, dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ), Z)

        #Send z to the appropriate partitions
        ZtoPartitions = ZRDD.join(self.varsToPartitions,numPartitions=self.N).map(lambda (key,(z,splitIndex)): (splitIndex, (key,z))).partitionBy(self.N,partitionFunc=identityHash).groupByKey().mapValues(list).mapValues(dict)
        PrimalDualOldZ=self.PrimalDualRDD.join(ZtoPartitions,numPartitions=self.N)

        if not self.lean:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z): solver.evaluate(Z)).reduce(add)  #local solver should implement evaluate

        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Phi,stats),Z): adaptDual(solver, P, Phi, stats, Z, alpha))
        ZbarAndNewDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Phi,Z): ( solver, dict( [(key, Z[key]-Phi[key]) for key in Z]), Phi ))
        self.PrimalDualRDD = ZbarAndNewDual.mapValues( lambda  (solver,Zbar,Phi) : (solver,solver.solve(Zbar, rho),Phi)).mapValues(lambda (solver,(P,stats),Phi): (solver,P,Phi,stats)).cache() #Solver should implement solve
        #Maybe partitioning is not needed?

        if checkpoint:
            self.PrimalDualRDD.localCheckpoint()
        ##Unpersisit commented for now because running time increases.
       # toUnpersist.unpersist()
  
        if not self.lean:
	    return (oldPrimalResidual,oldObjValue)
        else:
            return None

    def logstats(self):
        """  Return statistics from PrimalDualRDD. In particular, this returns the average, min, and maximum value of each statistic.
        """
        rdd = self.PrimalDualRDD
        
        statsonly =rdd.map(lambda (partitionid, (solver,P,Phi,stats)): stats).cache()
        #Checkpoint the RDD
       # if iteration!=0 and iteration % checkointing_freq == 0:
       #     statsonly.checkpoint()
        stats = statsonly.reduce(lambda x,y:  mergedicts(x,y))
        minstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,min))
        maxstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,max))
        return " ".join([ key+"= %s (%s/%s)" % (str(1.0*stats[key]/self.N),str(minstats[key]),str(maxstats[key]))   for key in stats])   	

    def getVars(self, rho):
        """Return the primal variables associated with this RDD. To be used to compute the new consensus variable"""
        return self.PrimalDualRDD.flatMap(lambda (partitionId,(solver,P,Phi,stats)): [ (key, ( rho*( P[key]+Phi[key]), rho))    for key in P ]  )

class ParallelSolverPnorm(ParallelSolver):
    """This class is inheritted from ParallelSolver, it updates P and Y vriables for a general p-norm solver via inner ADMM."""
    def __init__(self,LocalSolverClass,data,initvalue,N,rho,rho_inner, p, silent=True,lean=False, RDD=None):
        """Class constructor. It takes as an argument a local solver class, data (of a form understandable by the local solver class), an initial value for the primal variables, and a boolean value; the latter can be used to suppress the evaluation of the objective. 
        """
        self.SolverClass=LocalSolverClass
        if RDD==None:
            self.PrimalDualRDD =  LocalSolverClass.initializeLocalVariables(data,initvalue,N,rho).cache()    #LocalSolver class should implement class method initializeLocalVariables
        else:
            self.PrimalDualRDD = RDD
        self.N = N
        self.silent=silent
        self.lean=lean
        self.rho_inner = rho_inner
        self.p = p
        self.varsToPartitions = self.PrimalDualRDD.flatMapValues( lambda  (solver,P,Y,Phi,Upsilon, stats) : P.keys()).map(swap).partitionBy(self.N).cache()
    def joinAndAdapt(self,ZRDD, alpha, rho, checkpoint = False):
        rho_inner = self.rho_inner
        p_param = self.p
        #Send z to the appropriate partitions
        def Fm(objs,P):
            """
                Compute the FPm functions, i.e., FPm = \sum_{ij\in S1} P[(i,j)]-\sum_{ij \in S2} P[(i,j)]
            """
            FPm = {}
            for edge  in objs:
                (set1, set2) = objs[edge]
                tmp_val = 0.0
                for key in set1:
                    tmp_val += P[key]
                for key in set2:
                    tmp_val -= P[key]
                FPm[edge] = tmp_val
            return FPm

        ZtoPartitions = ZRDD.join(self.varsToPartitions,numPartitions=self.N).map(lambda (key,(z,splitIndex)): (splitIndex, (key,z))).partitionBy(self.N,partitionFunc=identityHash).groupByKey().mapValues(list).mapValues(dict)
        PrimalDualOldZ=self.PrimalDualRDD.join(ZtoPartitions,numPartitions=self.N)
        if not self.lean:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = (PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): solver.evaluate(Z, p_param)).reduce(add))**(1./p_param)  #local solver should compute p-norm to the power p. 
        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): ( solver, P, Y,dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ),Upsilon, stats,  Z))
        ZbarPrimalDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Z): ( solver,P,Y,Phi,Upsilon,stats,dict( [(key, Z[key]-Phi[key]) for key in Z])))
        
        #Start the inner ADMM iterations
        for i in range(100):
            #Compute vectors Fm(Pm)
            FmZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, Fm(solver.objectives,P),Y,Phi,Upsilon,stats,Zbar))
            if not self.lean:
               #Compute the residual 
                OldinnerResidual = np.sqrt(FmZbarPrimalDual.values().flatMap(lambda (solver, FPm,Y,Phi,Upsilon,stats,Zbar): [(Y[key]-FPm[key])**2 for key in Y]).reduce(add) )


            ##ADMM steps
            #Adapt the dual varible Upsilon
            FmYNewUpsilonPPhi = FmZbarPrimalDual.mapValues(lambda (solver, FPm, Y,Phi,Upsilon,stats,Zbar): (solver, FPm, Y, Phi, dict( [(key,Upsilon[key]+alpha*(Y[key]-FPm[key])) for key in Y]),stats,Zbar))
            #Update Y via prox. op. for p-norm
            NewYUpsilonPhi, Ynorm = pnormOp(FmYNewUpsilonPPhi.mapValues(lambda (solver, FPm, Y, Phi, Upsilon, stats, Zbar):(dict([(key,FPm[key]-Upsilon[key]) for key in Upsilon]), (solver, Y, Phi, Upsilon,stats,Zbar)  ) ), p_param, rho_inner, 1.e-6 )
            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (Y, (solver, OldY, Phi, Upsilon, stats, Zbar)): (solver, Y, OldY, Phi, Upsilon,stats, Zbar) )


            if not self.lean:
               #Compute the dual residual 
                DualInnerResidual = np.sqrt( NewYUpsilonPhi.values().flatMap(lambda (solver, Y, OldY, Phi, Upsilon,stats, Zbar): [ (Y[key] -OldY[key])**2 for key in Y]).reduce(add) )

            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (solver, Y, OldY, Phi, Upsilon,stats, Zbar):(solver, Y, Phi, Upsilon,stats, Zbar) )
            print "Iteration %d, p-norm is %f residual is %f, dual residual is %f" %(i, Ynorm, OldinnerResidual, DualInnerResidual)
           
            #Update P via solving a least-square problem
            ZbarPrimalDual = NewYUpsilonPhi.mapValues(lambda (solver,  Y, Phi,Upsilon,stats,Zbar): (solver,solver.solve(Y, Zbar, Upsilon, rho, rho_inner), Y, Phi, Upsilon, stats, Zbar)).mapValues(lambda (solver, (P, stats), Y, Phi, Upsilon, stats_old, Zbar): (solver,P,Y,Phi,Upsilon, stats, Zbar))
        
        self.PrimalDualRDD = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Zbar): (solver,P,Y,Phi,Upsilon,stats)).cache() 
        if not self.lean:
            return (oldPrimalResidual,oldObjValue)
        else:
            return None
           
    def getVars(self, rho):
        return self.PrimalDualRDD.flatMap(lambda (partitionId,(solver,P,Y,Phi,Upsilon,stats)): [ (key, ( rho*( P[key]+Phi[key]), rho))    for key in P ]  )
           
           
                                                     
 
