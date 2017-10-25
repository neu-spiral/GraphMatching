import numpy as np
import sys,argparse,logging,datetime,pickle,time
from pyspark import SparkContext,StorageLevel,SparkConf
from operator import add,and_
from LocalSolvers import LocalL1Solver,LocalL2Solver,FastLocalL2Solver
from helpers import swap,clearFile,identityHash,pretty,projectToPositiveSimplex,mergedicts,safeWrite,NoneToZero
from debug import logger,dumpPPhiRDD,dumpBasic
from pprint import pformat
import os
import shutil



def logstats(rdd,N):
    #stats,minstats,maxstats=rdd.map(lambda (partitionid, (solver,P,Phi,stats)): (stats,stats,stats)).reduce(lambda x,y: ( mergedicts(x[0],y[0]),
    #													mergedicts(x[1],y[1],min),
    #													mergedicts(x[2],y[2],max)))

    #return " ".join([ key+"= %s (%s/%s)" % (str(1.0*stats[key]/N),str(minstats[key]),str(maxstats[key]))   for key in stats])   	
 
    statsonly =rdd.map(lambda (partitionid, (solver,P,Phi,stats)): stats)
    stats = statsonly.reduce(lambda x,y:  mergedicts(x,y))
    minstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,min))
    maxstats = statsonly.reduce(lambda x,y:  mergedicts(x,y,max))
    return " ".join([ key+"= %s (%s/%s)" % (str(1.0*stats[key]/N),str(minstats[key]),str(maxstats[key]))   for key in stats])   	

def testPositivity(rdd):
    '''Test whether vals are positive
    '''
    val,count= rdd.map(lambda ((i,j),val): (val>=0.0,1 )).reduce(lambda (val1,ct1),(val2,ct2) : (val1+val2,ct1+ct2)  )
    return val/count 

def testSimplexCondition(rdd,dir='row'):
    '''Tests whether the row condition holds
    '''
    if dir=='row':
        sums=rdd.map(lambda ((i,j),val): (i,val) ).reduceByKey(add).values()
    elif dir=='column':
        sums=rdd.map(lambda ((i,j),val): (j,val) ).reduceByKey(add).values()
    minsum = sums.reduce(min)
    maxsum = sums.reduce(max)
    return minsum, maxsum


if __name__=="__main__":
	

    start_timing = time.time()
    parser = argparse.ArgumentParser(description = 'Parallel Graph Matching over Spark.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('constraintfile',default=None,help ="File containing graph of constraints. ") 
    parser.add_argument('outputfile',help = 'Output file storing learned doubly stochastic matrix Z')
    parser.add_argument('--graph1',default=None,help = 'File containing first graph (optional).')
    parser.add_argument('--graph2',default=None,help = 'File containing second graph (optional).')
    parser.add_argument('--objectivefile',default=None,help="File containing pre-computed objectives. Graphs  need not be given if this argument is set.")
    parser.add_argument('--linear_term',default=None,help="Linear term to be added in the objective")
    parser.add_argument('--problemsize',default=1000,type=int, help='Problem size. Used to initialize uniform allocation, needed when objectivefile is passed')
    parser.add_argument('--solver',default='LocalL1Solver', help='Local Solver',choices=['LocalL1Solver','LocalL2Solver','FastLocalL2Solver'])
    parser.add_argument('--debug',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--logfile',default='graphmatching.log',help='Log file')
    parser.add_argument('--maxiter',default=5,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=8,type=int, help='Number of partitions')
    parser.add_argument('--rhoP',default=1.0,type=float, help='Rho value, used for primal variables P')
    parser.add_argument('--rhoQ',default=1.0,type=float, help='Rho value, used for primal variables Q')
    parser.add_argument('--rhoT',default=1.0,type=float, help='Rho value, used for primal variables T')
    parser.add_argument('--alpha',default=0.05,type=float, help='Alpha value, used for dual variables')
    parser.add_argument('--dump_trace_freq',default=10,type=int,help='Number of iterations between trace dumps')
    parser.add_argument('--checkpoint_freq',default=10,type=int,help='Number of iterations between check points')
    parser.add_argument('--checkpointdir',default='checkpointdir',type=str,help='Directory to be used for checkpointing')

    parser.add_argument('--dumpRDDs', dest='dumpRDDs', action='store_true',help='Dump auxiliary RDDs beyond Z')
    parser.add_argument('--no-dumpRDDs', dest='dumpRDDs', action='store_false',help='Do not dump auxiliary RDDs beyond Z')
    parser.set_defaults(dumpRDDs=True)

    parser.add_argument('--init',default="", help="Initialization files for RDDs. It is assumed all files have the same name, passed as argument, with a _XXX indicating the RDD in the end, e.g.,filaname_ZRDD for ZRDD.")
	
    parser.set_defaults(silent=False)
    parser.add_argument('--silent',dest="silent",action='store_true',help='Run in efficient, silent mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. Overwrites verbosity level to ERROR')

    parser.set_defaults(lean=False)
    parser.add_argument('--lean',dest="lean",action='store_true',help='Run in efficient, ``lean'' mode, with final Z as sole output. Skips  computation of objective value and residuals duning exection and supresses both monitoring progress logging and trace dumping. It is the same as --silent, though it still prints some basic output messages, and does not effect verbosity level.')

   

    parser.set_defaults(directed=False)
    parser.add_argument('--directed', dest='directed', action='store_true',help='Input graphs are directed, i.e., (a,b) does not imply the presense of (b,a).')

    parser.set_defaults(driverdump=False)
    parser.add_argument('--driverdump',dest='driverdump',action='store_true', help='Dump final output Z after first collecting it at the driver, as opposed to directly from workers.')


    args = parser.parse_args()

    SolverClass = eval(args.solver)	

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',args.N)
    sc = SparkContext(appName='Parallel Graph Matching with  %s  at %d iterations over %d partitions' % (args.solver,args.maxiter,args.N),conf=configuration)
    sc.setCheckpointDir(args.checkpointdir)
    


    level = "logging."+args.debug
    if args.silent:
	level = "logging.ERROR"

    logger.setLevel(eval(level))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval(level))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)	
   
    DEBUG = logger.getEffectiveLevel()==logging.DEBUG
    logger.info("Starting with arguments: "+str(args))
    logger.info('Level set to: '+str(level)) 

    if not DEBUG:
        sc.setLogLevel("OFF")

    has_linear = args.linear_term is not None
    
    if args.init!="":
       

        ZRDD = sc.textFile(args.init+"_ZRDD").map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        
	# TODO: pickle old PPhiRDD solvers or have a constructor?
	PPhiRDD = sc.textFile(args.init+"_PPhiRDD").map(pickle.loads).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        
	QXiRDD = sc.textFile(args.init+"_QXiRDD").map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        TPsiRDD = sc.textFile(args.init+"_TPsiRDD").map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
        print "PPhiRDD is:",PPhiRDD.take(1)
     
    else:
	#Read constraint graph
	G = sc.textFile(args.constraintfile).map(eval).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)


	#Load linear term, if necesary:
	#has_linear =  args.linear_term is not None
	if has_linear:
	    D = sc.textFile(args.linear_term).map(eval)   
	    #print 'G=',sorted(G.collect())
	    D = D.rightOuterJoin(G.map(lambda pair: (pair, 1))).mapValues(lambda (val, dummy ): NoneToZero(val)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	    #print 'D=',sorted(D.collect())


	if not args.objectivefile:
	    #Read Graphs		
	    graph1 = sc.textFile(args.graph1,minPartitions=args.N).map(eval).partitionBy(args.N)
	    graph2 = sc.textFile(args.graph2,minPartitions=args.N).map(eval).partitionBy(args.N)

	    #Extract nodes 
	    nodes1 = graph1.flatMap(lambda (u,v):[u,v]).distinct()
	    nodes2 = graph2.flatMap(lambda (u,v):[u,v]).distinct()

	    uniformweight= 1.0/nodes2.count()


	    #repeat edges if graph is undirected
	    if not args.directed:
		graph1 = graph1.flatMap(lambda (u,v):[ (u,v),(v,u)]).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
		graph2 = graph2.flatMap(lambda (u,v):[ (u,v),(v,u)]).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	    
	    if DEBUG:
		dumpBasic(G,graph1,graph2)

	    objectives = SijGenerator(graph1,graph2,G,args.N).persist(StorageLevel.MEMORY_ONLY)

	else:
	    objectives = sc.textFile(args.objectivefile).map(lambda x:eval(x)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	    uniformweight= 1.0/float(args.problemsize)
	#Create local primal and dual variables  

	PPhiRDD = SolverClass.initializeLocalVariables(objectives,uniformweight,args.N,args.rhoP)
	
	if DEBUG:
	    dumpPPhiRDD(PPhiRDD)
     
	logger.info('Partitioned data rdd stats: '+logstats(PPhiRDD,args.N))     
      
	#Create remaining primal and dual variables
	QXiRDD = G.map(lambda (i,j): ((i,j),(uniformweight,0.0))).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	TPsiRDD = G.map(lambda (i,j): ((i,j),(uniformweight,0.0))).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)

	#Create consensus variable, initialized to uniform assignment ignoring constraints
	ZRDD = G.map(lambda (i,j):((i,j),uniformweight)).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)

    	
    #Variables to partitions
    varsToPartitions = PPhiRDD.flatMapValues( lambda  (solver,P,Phi,stats) : P.keys() ).map(swap).persist(StorageLevel.MEMORY_ONLY) 


    
    def projectAndUnfold( lst  ):
	zbar = dict( [(j,zbarval) for (j,zbarval,dualval) in lst])
	dual = dict( [(j,dualval) for (j,zbarval,dualval) in lst])
		
    	newprimal = projectToPositiveSimplex(zbar,1.0)
	return [ (j,newprimal[j],dual[j]) for j in newprimal]

    end_timing = time.time()

    logger.info("Data input and preprocessing done in %f seconds, starting main ADMM iterations " % ( end_timing -start_timing))

    start_timing = time.time()	
    last_time = start_timing
    trace = {}
    for iteration in range(args.maxiter):
	toUnpersist = QXiRDD
	#Compute Linear Objecti
	if not (args.silent or args.lean): 
	    if has_linear:
	        oldLinObjective= D.join(ZRDD).values().map(lambda (dij,z):dij*z).reduce(add) 
	        #logger.info("D,join(RDD) %s" % str(D.join(ZRDD).collect()))
	        #logger.info("OldLin %f" % oldLinObjective)
	    else:
	        oldLinObjective=0.0

	#Send Z to rows
	QXiandOldZ = QXiRDD.join(ZRDD)
	
	#Compute old primal residual
	if not (args.silent or args.lean):
	    oldprimalresidualQ =  np.sqrt(QXiandOldZ.values().map(lambda ((q,xi),z):(q-z)**2).reduce(add))

	#Update dual and primal variables for row projection
	
	#Fudge z if linear term exists
	if has_linear:
	    #print "Dkeys=",sorted(D.keys().collect())
	    #print 'QXIKEYS=',sorted(QXiandOldZ.keys().collect())
	    QXiandOldZD = QXiandOldZ.join(D).mapValues(lambda (((q,xi),z), dij): (q,xi,z,0.5*dij/args.rhoQ ) )
	else:
            QXiandOldZD = QXiandOldZ.mapValues(lambda ((q,xi),z):(q,xi,z,0))

	ZbarAndXi = QXiandOldZD.mapValues(lambda (q,xi,z,d) : (q,xi+args.alpha*(q-z),z,d)).map(lambda ((i,j), (q,xi,z,d)): (i, (j,z-xi-d,xi))) 
	
	    
        ZbarAndXiCombined = ZbarAndXi.combineByKey( 
 					lambda (j,zbar,xi) : [(j,zbar,xi) ], #createCombiner
					lambda listsofar,(j,zbar,xi) : listsofar+[(j,zbar,xi)], #mergeValue
         				lambda list1,list2: list1+list2  #mergeCombiners        
					)
	QXiRDD = ZbarAndXiCombined.flatMapValues(projectAndUnfold).map(lambda (i,(j,q,xi)): ((i,j),(q,xi)) ).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	if iteration % args.checkpoint_freq == 1:
	    QXiRDD.checkpoint()
	#logger.info("Iteration %d QXi  has %d partitions" % (iteration,QXiRDD.getNumPartitions()))
	toUnpersist.unpersist()

        if DEBUG:
            Q = QXiRDD.mapValues(lambda (q,xi):q)
	    logger.debug("Iteration %d Q positivity is: %s " % (iteration,str(testPositivity(Q)*100)+'%%')) 
	    logger.debug("Iteration %d Q positivity row sums are: Min %s Max %s " % ((iteration,)+tuple(testSimplexCondition(Q)) )) 


	toUnpersist = TPsiRDD
	#Send Z to columns
	TPsiandOldZ = TPsiRDD.join(ZRDD,numPartitions=args.N)

	#Compute old primal residual
	if not (args.silent or args.lean):
	    oldprimalresidualT =  np.sqrt(TPsiandOldZ.values().map(lambda ((t,psi),z):(t-z)**2).reduce(add))


	#Update dual and primal variables for column projection

	#Fudge z if linear term exists
	if has_linear:
	    TPsiandOldZD = TPsiandOldZ.join(D).mapValues(lambda ( ((t,psi),z), dij): (t,psi,z,0.5*dij/args.rhoT) ) 
	else:
	    TPsiandOldZD = TPsiandOldZ.mapValues( lambda  ((t,psi),z): (t,psi,z,0)   )

        ZbarAndPsi = TPsiandOldZD.mapValues(lambda (t,psi,z,d) : (t,psi+args.alpha*(t-z),z,d) ).map(lambda ((i,j), (t,psi,z,d)): (j, (i,z-psi-d,psi))) 
        ZbarAndPsiCombined = ZbarAndPsi.combineByKey( 
 					lambda (j,zbar,psi) : [(j,zbar,psi) ], #createCombiner
					lambda listsofar,(j,zbar,psi) : listsofar+[(j,zbar,psi)], #mergeValue
         				lambda list1,list2: list1+list2  #mergeCombiners        
					)
	TPsiRDD = ZbarAndPsiCombined.flatMapValues(projectAndUnfold).map(lambda (j,(i,t,psi)): ((i,j),(t,psi)) ).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	if iteration % args.checkpoint_freq == 1:
	    TPsiRDD.checkpoint()
	#logger.info("Iteration %d TPsi  has %d partitions" % (iteration,TPsiRDD.getNumPartitions()))
	toUnpersist.unpersist()

	if DEBUG:
            T = TPsiRDD.mapValues(lambda (t,psi):t)
	    logger.debug("Iteration %d T positivity is: %s " % (iteration,str(testPositivity(T)*100)+'%%')) 
	    logger.debug("Iteration %d T column sums are: Min %s Max %s " % ((iteration,)+tuple(testSimplexCondition(T,dir='column')) ) )
	
	toUnpersist = PPhiRDD 
        #Send Z to Local P Variables
        ZtoPartitions = ZRDD.join(varsToPartitions,numPartitions=args.N).map(lambda ((i,j),(z,splitIndex)): (splitIndex, ((i,j),z))).groupByKey().mapValues(list).mapValues(dict).partitionBy(args.N,partitionFunc=identityHash)
	PPhiOldZ =PPhiRDD.join(ZtoPartitions,numPartitions=args.N)
	
	#Compute old primal residual and old objective value
	if not (args.silent or args.lean):
	    oldprimalresidualP =  np.sqrt(PPhiOldZ.values().map(lambda ((solver,P,Phi,stats),Z):  sum( ( (P[key]-Z[key])**2    for key in Z) )    ).reduce(add))
	    oldObjValue = PPhiOldZ.values().map(lambda ((solver,P,Phi,stats),Z): solver.evaluate(Z)).reduce(add)

	#Update dual and primal variables for Local P optimization
	PPhiZ = PPhiOldZ.mapValues(lambda ((solver,P,Phi,stats),Z): ( solver, P, dict( [ (key,Phi[key]+args.alpha *(P[key]-Z[key]))  for key in Phi  ]  ), Z))
        ZbarAndPhi = PPhiZ.mapValues(lambda (solver,P,Phi,Z): ( solver, dict( [(key, Z[key]-Phi[key]) for key in Z]), Phi ))
	PPhiRDD = ZbarAndPhi.mapValues( lambda  (solver,Zbar,Phi) : (solver,solver.solve(Zbar),Phi)).mapValues(lambda (solver,(P,stats),Phi): (solver,P,Phi,stats)).partitionBy(args.N,partitionFunc=identityHash).persist(StorageLevel.MEMORY_ONLY)
	if iteration % args.checkpoint_freq == 1:
	    PPhiRDD.checkpoint()
	toUnpersist.unpersist()	
	
	if not (args.silent or args.lean):
	   logger.info("Iteration %d local stats: " % iteration + logstats(PPhiRDD,args.N))

	#Aggregate variables to construct new Z
	if DEBUG:
	   logger.debug("Iteration %d QXi is:\n%s " %(iteration,pformat(list(QXiRDD.collect()),width=30)))
	   logger.debug("Iteration %d TPsi is:\n%s " %(iteration,pformat(list(TPsiRDD.collect()),width=30)))
	   logger.debug("Iteration %d PPhi is:\n%s " %(iteration,pformat(list(PPhiRDD.map(lambda (partitionID,(solver,P,Phi,stats)): (partitionID,(stats,P,Phi))).collect()),width=30)))
	   
	oldZ = ZRDD
	rowvars = QXiRDD.mapValues( lambda (q,xi): (args.rhoQ*(q+xi),args.rhoQ) )
	columnvars = TPsiRDD.mapValues( lambda (t,psi): (args.rhoT*(t+psi),args.rhoT)  )
	localvars = PPhiRDD.flatMap(lambda (partitionId,(solver,P,Phi,stats)): [ (key, ( args.rhoP*( P[key]+Phi[key]),args.rhoP))    for key in P ]  )
	allvars = localvars.union(rowvars).union(columnvars)

	if DEBUG:
	   logger.debug("Iteration %d all var pairs is:\n%s" %(iteration,pformat(list(allvars.sortByKey().collect()),width=30)) )
	
	ZRDD = allvars.reduceByKey(lambda (value1,count1),(value2,count2) : (value1+value2,count1+count2)  ).mapValues(lambda (value,count): 1.0*value/count).partitionBy(args.N).persist(StorageLevel.MEMORY_ONLY)
	if iteration % args.checkpoint_freq == 1:
	    ZRDD.checkpoint()
	
	if DEBUG:
	   logger.debug("Iteration %d Z is:\n%s" %(iteration,pformat(list(ZRDD.collect()),width=30)) )

       	Zstats ={}
	if not (args.silent or args.lean):
	
           #Z feasibility
	   Zstats['POS'] = testPositivity(ZRDD)
	   Zstats['RSUMS'] = tuple(testSimplexCondition(ZRDD))
	   Zstats['CSUMS'] = tuple(testSimplexCondition(ZRDD,dir='column'))

           #Evaluate dual residuals:
	   dualresidual = np.sqrt(ZRDD.join(oldZ,numPartitions=args.N).values().map(lambda (v1,v2):(v1-v2)**2).reduce(add)) 
	   Zstats['DRES'] = dualresidual

	   #Store primal residuals:
	   Zstats['PRES'] = oldprimalresidualP
	   Zstats['QRES'] = oldprimalresidualQ
	   Zstats['TRES'] = oldprimalresidualT
           #Z obj
	   Zstats['OLDOBJ'] = oldObjValue + oldLinObjective
	   Zstats['OLDNOLIN'] = oldObjValue
	   Zstats['OLDLIN'] = oldLinObjective
   

           logger.info("Iteration %d Z positivity is: %s " % (iteration,str(Zstats['POS']*100)+'%')) 
	   logger.info("Iteration %d Z row sums are: Min %s Max %s " % ((iteration,)+ Zstats['RSUMS'] ) )
	   logger.info("Iteration %d Z column sums are: Min %s Max %s " % ((iteration,)+ Zstats['CSUMS']) )
	   logger.info("Iteration %d-1 Z objective value: %s  (= %s + %s) " % (iteration, Zstats['OLDOBJ'],Zstats['OLDNOLIN'],Zstats['OLDLIN']) )
	   logger.info("Iteration %d-1 Z residuals: " % iteration+ "\t".join( [ key+":"+str(Zstats[key])  for key in ['DRES','PRES','QRES','TRES']] ) )
  
  
	if not args.silent: #under "lean", still output some basic stats
	   now = time.time()
           Zstats['TIME'] = now-start_timing
           Zstats['IT_TIME'] = now-last_time
	   last_time = now
           logger.info("Iteration %d  time is %f sec, average time per iteration is %f sec, total time is %f " % (iteration,Zstats['IT_TIME'],Zstats['TIME']/(iteration+1.0),Zstats['TIME'])) 
	   trace[iteration] = Zstats
	
        #if not (args.silent or args.lean):
        if not (args.silent):
	   if iteration % args.dump_trace_freq == 1:
                with open(args.outputfile+"_trace",'wb') as f:
            	    pickle.dump((args,trace),f)
    	        safeWrite(ZRDD,args.outputfile+"_ZRDD",args.driverdump)
		
                if args.dumpRDDs:
		    safeWrite(PPhiRDD.map(pickle.dumps),args.outputfile+"_PPhiRDD",args.driverdump)
		    safeWrite(QXiRDD,args.outputfile+"_QXiRDD",args.driverdump)
		    safeWrite(TPsiRDD,args.outputfile+"_TPsiRDD",args.driverdump)
		#log.info("ZRDD is "+str(ZRDD.collect()))
 
	oldZ.unpersist()
	
     
    end_timing = time.time()
    logger.info("Finished ADMM iterations in %f seconds." % (end_timing-start_timing))

    if not args.silent:
        with open(args.outputfile+"_trace",'wb') as f:
	    pickle.dump((args,trace),f)

    safeWrite(ZRDD,args.outputfile+"_ZRDD",args.driverdump)
    

    
