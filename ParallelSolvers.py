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
from proxOp import pnormOp,pnorm_proxop, L1normOp, EuclidiannormOp


class ParallelSolver():
    """ A class for a parallel solver object. This object stores an RDD containing "local" data per partition, captured via a local solver object.
       The RDD also stores primal and dual variables associated with the arguments of this local solver function, as well as statistics reported by 
       the last computation of the local solver. The class can be used as an interface to add "homogeneous" objectives in the consensus admm algorithm,
       that can be executed in parallel
    """
    def __init__(self,LocalSolverClass,data,initvalue,N,rho,silent=False,lean=False, RDD=None, D=None, lambda_linear=1.0, prePartFunc=None):
        """Class constructor. It takes as an argument a local solver class, data (of a form understandable by the local solver class), an initial value for the primal variables, and a boolean value; the latter can be used to suppress the evaluation of the objective. 
        """
        self.SolverClass=LocalSolverClass
        if RDD==None:
            if D==None:
                self.PrimalDualRDD =  LocalSolverClass.initializeLocalVariables(Sij=data,initvalue=initvalue,N=N,rho=rho, prePartFunc=prePartFunc).cache()    #LocalSolver class should implement class method initializeLocalVariables
            else:
                self.PrimalDualRDD =  LocalSolverClass.initializeLocalVariables(data,initvalue,N,rho,D,lambda_linear).cache()
        else:
            self.PrimalDualRDD = RDD
        self.N = N
        self.silent=silent
        self.lean=lean
        self.varsToPartitions = self.PrimalDualRDD.flatMapValues( lambda  (solver,P,Phi,stats) : P.keys()).map(swap).partitionBy(self.N).cache() 

    def joinAndAdapt(self,ZRDD, alpha, rho,checkpoint = False, forceComp=False):
        """ Given a ZRDD, adapt the local primal and dual variables. The former are updated via the proximal operator, the latter via gradient ascent.
        """
        toUnpersist = self.PrimalDualRDD         #Old RDD is to be uncached

        def adaptDual(solver, P, Phi, stats, Z, alpha):
            """Update the dual variables."""
            return ( solver, P, dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ), Z)

        #Send z to the appropriate partitions
        ZtoPartitions = ZRDD.join(self.varsToPartitions,numPartitions=self.N).map(lambda (key,(z,splitIndex)): (splitIndex, (key,z))).partitionBy(self.N,partitionFunc=identityHash).groupByKey().mapValues(list).mapValues(dict)
        PrimalDualOldZ=self.PrimalDualRDD.join(ZtoPartitions,numPartitions=self.N)

        if not self.silent or forceComp:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = PrimalDualOldZ.values().map(lambda ((solver,P,Phi,stats),Z): solver.evaluate(Z)).reduce(add)  #local solver should implement evaluate

        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Phi,stats),Z): adaptDual(solver, P, Phi, stats, Z, alpha))
        ZbarAndNewDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Phi,Z): ( solver, dict( [(key, Z[key]-Phi[key]) for key in Z]), Phi ))
        self.PrimalDualRDD = ZbarAndNewDual.mapValues( lambda  (solver,Zbar,Phi) : (solver,solver.solve(Zbar, rho),Phi)).mapValues(lambda (solver,(P,stats),Phi): (solver,P,Phi,stats)).cache() #Solver should implement solve
        #Maybe partitioning is not needed?

        if checkpoint:
            self.PrimalDualRDD.localCheckpoint()
        ##Unpersisit commented for now because running time increases.
        #toUnpersist.unpersist()
  
        if not self.silent or forceComp:
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
    def computeDualResidual(self, ZRDDjoinedOldZRDD):
        '''Return the squared norm of the dual residual, which is computed as:
                S = A^TB(Z^(k+1)-Z^(k))
        '''
        ZRDDjoinedOldZRDD = ZRDDjoinedOldZRDD.mapValues(lambda (z, zOld): (z-zOld)**2)
        return np.sqrt( self.varsToPartitions.join(ZRDDjoinedOldZRDD).mapValues(lambda (splitID, deltaZ): deltaZ).values().reduce(add) )

class ParallelSolverPnorm(ParallelSolver):
    """This class is inheritted from ParallelSolver, it updates P and Y vriables for a general p-norm solver via inner ADMM."""
    def __init__(self,LocalSolverClass,data,initvalue,N,rho,rho_inner, p, silent=False,lean=False, RDD=None, debug=False, prePartFunc=None):
        """Class constructor. It takes as an argument a local solver class, data (of a form understandable by the local solver class), an initial value for the primal variables, and a boolean value; the latter can be used to suppress the evaluation of the objective. 
        """
        self.SolverClass=LocalSolverClass
        if RDD==None:
            self.PrimalDualRDD =  LocalSolverClass.initializeLocalVariables(Sij=data,initvalue=initvalue,N=N,rho=rho, rho_inner=rho_inner, prePartFunc=prePartFunc).cache()    #LocalSolver class should implement class method initializeLocalVariables
        else:
            self.PrimalDualRDD = RDD
        self.N = N
        self.silent=silent
        self.lean=lean
        self.debug = debug #In debug mode keep track of the obj. val. and residuals
        self.rho_inner = rho_inner
        self.p = p
        self.varsToPartitions = self.PrimalDualRDD.flatMapValues( lambda  (solver,P,Y,Phi,Upsilon, stats) : P.keys()).map(swap).partitionBy(self.N).cache()
    def joinAndAdapt(self,ZRDD, alpha, rho, alpha_inner=1.0, maxiters = 100, residual_tol = 1.e-06, checkpoint = False, logger=None, forceComp=False):
        rho_inner = self.rho_inner
        p_param = self.p
        #In debug mode keep track of the obj. val. and residuals
        if self.debug:
            trace = {}
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
        if not self.silent or forceComp:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = (PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): solver.evaluate(Z, p_param)).reduce(add))**(1./p_param)  #local solver should compute p-norm to the power p. 
        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): ( solver, P, Y,dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ),Upsilon, stats,  Z))
        ZbarPrimalDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Z): ( solver,P,Y,Phi,Upsilon,stats,dict( [(key, Z[key]-Phi[key]) for key in Z])))
        
 #       print ZbarPrimalDual.values().map(lambda (solver,P,Y,Phi,Upsilon,stats, Z): Z).take(1)
        last = time()
        start_time = time()
        #Start the inner ADMM iterations
        for i in range(maxiters):
            #Compute vectors Fm(Pm)
            FmZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, Fm(solver.objectives,P),P,Y,Phi,Upsilon,stats,Zbar))
            if not self.lean or (self.lean and i==maxiters-1):
               #Compute the residual 
                OldinnerResidual = np.sqrt(FmZbarPrimalDual.values().flatMap(lambda (solver, FPm,OldP,Y,Phi,Upsilon,stats,Zbar): [(Y[key]-FPm[key])**2 for key in Y]).reduce(add) )
                


            ##ADMM steps
            #Adapt the dual varible Upsilon
            FmYNewUpsilonPPhi = FmZbarPrimalDual.mapValues(lambda (solver, FPm,OldP, Y,Phi,Upsilon,stats,Zbar): (solver, FPm, OldP, Y, Phi, dict( [(key,Upsilon[key]+alpha_inner*(Y[key]-FPm[key])) for key in Y]),stats,Zbar))
 

            #Update Y via prox. op. for p-norm
            NewYUpsilonPhi, Ynorm = pnormOp(FmYNewUpsilonPPhi.mapValues(lambda (solver, FPm, OldP, Y, Phi, Upsilon, stats, Zbar):(dict([(key,FPm[key]-Upsilon[key]) for key in Upsilon]), (solver, OldP, Y, Phi, Upsilon,stats,Zbar)  ) ), p_param, rho_inner, 1.e-6,  self.lean and i<maxiters-1 )
            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (Y, (solver, OldP, OldY, Phi, Upsilon, stats, Zbar)): (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar) )


            if not self.lean or (self.lean and i==maxiters-1):
               #Compute the dual residual for Y
                DualInnerResidual_Y = np.sqrt( NewYUpsilonPhi.values().flatMap(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar): [ (Y[key] -OldY[key])**2 for key in Y]).reduce(add) )

            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar):(solver, OldP, Y, Phi, Upsilon,stats, Zbar) )

           
            #Update P via solving a least-square problem
            ZbarPrimalDual = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, Phi,Upsilon,stats,Zbar): (solver,solver.solve(Y, Zbar, Upsilon, rho, rho_inner),OldP, Y, Phi, Upsilon, stats, Zbar)).mapValues(lambda (solver, (P, stats),OldP, Y, Phi, Upsilon, stats_old, Zbar): (solver,P,OldP,Y,Phi,Upsilon, stats, Zbar))


            if not self.lean or (self.lean and i==maxiters-1):
                #Compute the dual residual for P
                DualInnerResidual_P = np.sqrt( ZbarPrimalDual.values().flatMap(lambda (solver,P,OldP,Y,Phi,Upsilon, stats, Zbar):  [ (P[key] -OldP[key])**2 for key in P]).reduce(add) )
                #Total dual residual 
                DualInnerResidual = DualInnerResidual_P + DualInnerResidual_Y

            ZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,OldP,Y,Phi,Upsilon, stats, Zbar): (solver,P,Y,Phi,Upsilon, stats, Zbar))
            if not self.lean  or (self.lean and i==maxiters-1): 
                objval = ZbarPrimalDual.values().flatMap(lambda (solver,P,Y,Phi,Upsilon, stats, Zbar):[(P[key]-Zbar[key])**2 for key in P]).reduce(lambda x,y:x+y) + Ynorm
            now = time()
            if logger != None and ( not self.lean or (self.lean and i==maxiters-1) ):
                logger.info("Inner ADMM iteration %d, p-norm is %f, objective is %f, residual is %f, dual residual is %f, time is %f" %(i, Ynorm, objval, OldinnerResidual, DualInnerResidual, now-last))
            if (not self.lean or (self.lean and i==maxiters-1)) and self.debug:
                trace[i] = {}
                trace[i]['OBJ'] = objval
                trace[i]['PRES'] = OldinnerResidual
                trace[i]['DRES'] = DualInnerResidual
                trace[i]['IT_TIME'] = now-last 
                trace[i]['TIME'] = now-start_time
                 
            last = time()
            if not self.lean and DualInnerResidual<residual_tol and OldinnerResidual<residual_tol:
                break
        self.PrimalDualRDD = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Zbar): (solver,P,Y,Phi,Upsilon,stats)).cache() 

        #Checkpointing
        if checkpoint:
            self.PrimalDualRDD.localCheckpoint()

        if self.debug:
            return trace

        if not self.silent or forceComp:
            return (oldPrimalResidual,oldObjValue)
        else:
            return None
           
    def logstats(self):
        """  Return statistics from PrimalDualRDD. In particular, this returns the average, min, and maximum value of each statistic.
        """
        rdd = self.PrimalDualRDD

        statsonly =rdd.map(lambda (partitionid, (solver,P,Y,Phi,Upsilon,stats)): stats).cache()
        #Checkpoint the RDD
       # if iteration!=0 and iteration % checkointing_freq == 0:
       #     statsonly.checkpoint()
        stats = statsonly.reduce(lambda x,y:  mergedicts(x,y))
        minstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,min))
        maxstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,max))
        return " ".join([ key+"= %s (%s/%s)" % (str(1.0*stats[key]/self.N),str(minstats[key]),str(maxstats[key]))   for key in stats])
    def getVars(self, rho):
        return self.PrimalDualRDD.flatMap(lambda (partitionId,(solver,P,Y,Phi,Upsilon,stats)): [ (key, ( rho*( P[key]+Phi[key]), rho))    for key in P ]  )
           
           
                                                     
class ParallelSolver1norm(ParallelSolverPnorm):
    def joinAndAdapt(self,ZRDD, alpha, rho,alpha_inner=1.0,  maxiters = 100, residual_tol = 1.e-06, checkpoint = False, logger = None, forceComp=False):
        rho_inner = self.rho_inner
        p_param = 1
        if self.debug:
            trace = {}
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
        if not self.silent or forceComp:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = (PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): solver.evaluate(Z, p_param)).reduce(add))**(1./p_param)  #local solver should compute p-norm to the power p. 
        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): ( solver, P, Y,dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ),Upsilon, stats,  Z))
        ZbarPrimalDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Z): ( solver,P,Y,Phi,Upsilon,stats,dict( [(key, Z[key]-Phi[key]) for key in Z])))


        #Initialization for Inner ADMM
        #initialize Upsilon to 0
   #     ZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, P, Y, Phi, dict([(key,0.0) for key in Upsilon]), stats, Zbar))
        #initialize P by solving         
      #  ZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, solver.solve(Y, Zbar, Upsilon, rho, rho_inner), Y, Phi,Upsilon,stats,Zbar)).mapValues(lambda (solver, (P, stats0), Y, Phi,Upsilon,stats,Zbar): (solver,P,Y,Phi,Upsilon,stats,Zbar))

        last = time()
        start_time  = last
        #Start the inner ADMM iterations
        for i in range(maxiters):
            #Compute vectors Fm(Pm)
            FmZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, Fm(solver.objectives,P),P,Y,Phi,Upsilon,stats,Zbar))
            if not self.lean  or (self.lean and i==maxiters-1):
               #Compute the residual 
                OldinnerResidual = np.sqrt(FmZbarPrimalDual.values().flatMap(lambda (solver, FPm,OldP,Y,Phi,Upsilon,stats,Zbar): [(Y[key]-FPm[key])**2 for key in Y]).reduce(add) )


            ##ADMM steps
            #Adapt the dual varible Upsilon
            FmYNewUpsilonPPhi = FmZbarPrimalDual.mapValues(lambda (solver, FPm, OldP,Y,Phi,Upsilon,stats,Zbar): (solver, FPm,OldP, Y, Phi, dict( [(key,Upsilon[key]+alpha_inner*(Y[key]-FPm[key])) for key in Y]),stats,Zbar))


            #Update Y via prox. op. for ell_1 norm
            NewYUpsilonPhi, Ynorm = L1normOp(FmYNewUpsilonPPhi.mapValues(lambda (solver, FPm,OldP, Y, Phi, Upsilon, stats, Zbar):(dict([(key,FPm[key]-Upsilon[key]) for key in Upsilon]), (solver, OldP,Y, Phi, Upsilon,stats,Zbar)  ) ), rho_inner , self.lean and i<maxiters-1)
            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (Y, (solver, OldP, OldY, Phi, Upsilon, stats, Zbar)): (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar) )


            if not self.lean  or (self.lean and i==maxiters-1):
               #Compute the dual residual for Y
                DualInnerResidual_Y = np.sqrt( NewYUpsilonPhi.values().flatMap(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar): [ (Y[key] -OldY[key])**2 for key in Y]).reduce(add) )

            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar):(solver, OldP, Y, Phi, Upsilon,stats, Zbar) )


            #Update P via solving a least-square problem
            ZbarPrimalDual = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, Phi,Upsilon,stats,Zbar): (solver,solver.solve(Y, Zbar, Upsilon, rho, rho_inner), OldP, Y, Phi, Upsilon, stats, Zbar)).mapValues(lambda (solver, (P, stats), OldP, Y, Phi, Upsilon, stats_old, Zbar): (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar))



            if not self.lean  or (self.lean and i==maxiters-1):
                #Compute the dual residual for P
                DualInnerResidual_P = np.sqrt( ZbarPrimalDual.values().flatMap(lambda (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar):  [ (P[key] -OldP[key])**2 for key in P]).reduce(add) )
                #Total dual residual 
                DualInnerResidual = DualInnerResidual_P + DualInnerResidual_Y

            ZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar): (solver,P,Y,Phi,Upsilon, stats, Zbar))

            if not self.lean  or (self.lean and i==maxiters-1):
                objval = ZbarPrimalDual.values().flatMap(lambda (solver,P,Y,Phi,Upsilon, stats, Zbar):[(P[key]-Zbar[key])**2 for key in P]).reduce(lambda x,y:x+y) + Ynorm
            now = time()
            if logger != None and (not self.lean  or (self.lean and i==maxiters-1)):
                logger.info("Inner ADMM iteration %d, p-norm is %f, objective is %f, residual is %f, dual residual is %f, iteration time is %f" %(i, Ynorm, objval, OldinnerResidual, DualInnerResidual, now-last))

            if (not self.lean or (self.lean and i==maxiters-1)) and self.debug:
                trace[i] = {}
                trace[i]['OBJ'] = objval
                trace[i]['PRES'] = OldinnerResidual
                trace[i]['DRES'] = DualInnerResidual
                trace[i]['IT_TIME'] = now-last
                trace[i]['TIME'] = now-start_time
            last = time()
            

            if not self.lean and DualInnerResidual<residual_tol and OldinnerResidual<residual_tol:
                break
        self.PrimalDualRDD = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Zbar): (solver,P,Y,Phi,Upsilon,stats)).cache()

        #Checkpointing
        if checkpoint:
            self.PrimalDualRDD.localCheckpoint()


        if self.debug:
            return trace

        if not self.silent or forceComp:
            return (oldPrimalResidual,oldObjValue)
        else:
            return None

class ParallelSolver2norm(ParallelSolverPnorm):
    def joinAndAdapt(self,ZRDD, alpha, rho, alpha_inner=1.0, maxiters = 100, residual_tol = 1.e-06, checkpoint = False, logger = None, forceComp=False):
        rho_inner = self.rho_inner
        p_param = 2
        if self.debug:
            trace = {}
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
        if not self.silent or forceComp:
            oldPrimalResidual = np.sqrt(PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
            oldObjValue = (PrimalDualOldZ.values().map(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): solver.evaluate(Z, p_param)).reduce(add))**(1./p_param)  #local solver should compute p-norm to the power p. 
        PrimalNewDualOldZ = PrimalDualOldZ.mapValues(lambda ((solver,P,Y,Phi,Upsilon,stats),Z): ( solver, P, Y,dict( [ (key,Phi[key]+alpha*(P[key]-Z[key]))  for key in Phi  ]  ),Upsilon, stats,  Z))
        ZbarPrimalDual = PrimalNewDualOldZ.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Z): ( solver,P,Y,Phi,Upsilon,stats,dict( [(key, Z[key]-Phi[key]) for key in Z])))



   
        last = time()
        start_time = last
        #Start the inner ADMM iterations
        for i in range(maxiters):
            #Compute vectors Fm(Pm)
            FmZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats,Zbar):(solver, Fm(solver.objectives,P),P,Y,Phi,Upsilon,stats,Zbar))
            if not self.lean or (self.lean and i==maxiters-1):
               #Compute the residual 
                OldinnerResidual = np.sqrt(FmZbarPrimalDual.values().flatMap(lambda (solver, FPm,OldP,Y,Phi,Upsilon,stats,Zbar): [(Y[key]-FPm[key])**2 for key in Y]).reduce(add) )


            ##ADMM steps
            #Adapt the dual varible Upsilon
            FmYNewUpsilonPPhi = FmZbarPrimalDual.mapValues(lambda (solver, FPm,OldP, Y,Phi,Upsilon,stats,Zbar): (solver, FPm, OldP, Y, Phi, dict( [(key,Upsilon[key]+alpha_inner*(Y[key]-FPm[key])) for key in Y]),stats,Zbar))


            #Update Y via prox. op. for ell_2 norm
            NewYUpsilonPhi, Ynorm = EuclidiannormOp(FmYNewUpsilonPPhi.mapValues(lambda (solver, FPm,OldP, Y, Phi, Upsilon, stats, Zbar):(dict([(key,FPm[key]-Upsilon[key]) for key in Upsilon]), (solver, OldP, Y, Phi, Upsilon,stats,Zbar)  ) ),  rho_inner,  self.lean and i<maxiters-1)
            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (Y, (solver, OldP, OldY, Phi, Upsilon, stats, Zbar)): (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar) )
        

            if not self.lean or (self.lean and i==maxiters-1):
               #Compute the dual residual for Y
                DualInnerResidual_Y = np.sqrt( NewYUpsilonPhi.values().flatMap(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar): [ (Y[key] -OldY[key])**2 for key in Y]).reduce(add) )

            NewYUpsilonPhi = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, OldY, Phi, Upsilon,stats, Zbar):(solver, OldP, Y, Phi, Upsilon,stats, Zbar) )


            #Update P via solving a least-square problem
            ZbarPrimalDual = NewYUpsilonPhi.mapValues(lambda (solver, OldP, Y, Phi,Upsilon,stats,Zbar): (solver,solver.solve(Y, Zbar, Upsilon, rho, rho_inner),OldP, Y, Phi, Upsilon, stats, Zbar)).mapValues(lambda (solver, (P, stats), OldP, Y, Phi, Upsilon, stats_old, Zbar): (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar))


            if not self.lean or (self.lean and i==maxiters-1):
                #Compute the dual residual for P
                DualInnerResidual_P = np.sqrt( ZbarPrimalDual.values().flatMap(lambda (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar):  [ (P[key] -OldP[key])**2 for key in P]).reduce(add) )
                #Total dual residual 
                DualInnerResidual = DualInnerResidual_P + DualInnerResidual_Y

            ZbarPrimalDual = ZbarPrimalDual.mapValues(lambda (solver,OldP,P,Y,Phi,Upsilon, stats, Zbar): (solver,P,Y,Phi,Upsilon, stats, Zbar))

            if not self.lean or (self.lean and i==maxiters-1):
                objval = ZbarPrimalDual.values().flatMap(lambda (solver,P,Y,Phi,Upsilon, stats, Zbar):[(P[key]-Zbar[key])**2 for key in P]).reduce(lambda x,y:x+y) + Ynorm
            now = time()
            if logger != None and (not self.lean  or (self.lean and i==maxiters-1)):
                logger.info("Inner ADMM iteration %d, p-norm is %f, objective is %f, residual is %f, dual residual is %f, iteration time is %f" %(i, Ynorm, objval, OldinnerResidual, DualInnerResidual, now-last))
            if (not self.lean or (self.lean and i==maxiters-1)) and self.debug:
                trace[i] = {}
                trace[i]['OBJ'] = objval
                trace[i]['PRES'] = OldinnerResidual
                trace[i]['DRES'] = DualInnerResidual
                trace[i]['IT_TIME'] = now-last
                trace[i]['TIME'] = now-start_time
            last = time()

            if  not self.lean and DualInnerResidual<residual_tol and OldinnerResidual<residual_tol:
                break
        self.PrimalDualRDD = ZbarPrimalDual.mapValues(lambda (solver,P,Y,Phi,Upsilon,stats, Zbar): (solver,P,Y,Phi,Upsilon,stats)).cache()

        #Checkpointing
        if checkpoint:
            self.PrimalDualRDD.localCheckpoint()

        if self.debug:
            return trace

        if not self.silent or forceComp:
            return (oldPrimalResidual,oldObjValue)
        else:
            return None
