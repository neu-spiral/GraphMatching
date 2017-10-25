Graph Matching
===========================

The present code computes the distances between two graphs using ADMM.





Usage
-----
Execution requires first preprocessing the data to create objectives, constraints, etc.

	spark-submit --master spark://compute-3-103:7077 --total-executor-cores 100  --executor-memory 50G  --py-files "helpers.py,LocalSolvers.py,debug.py" preprocessor.py data/ER10000_0.3/ER10000_0.3 data/ER10000_0.3/ER10000_0.3  data/ER10000_0.3/G --N 100


Once these have been created, the code can be executed as follows:

	spark-submit --master spark://10.100.9.167:7077   --executor-memory 450G --driver-memory 100G --py-files "LocalSolvers.py,helpers.py,debug.py"  --conf "spark.akka.frameSize=1024" --conf "spark.driver.maxResultSize=20G" GraphMatching.py data/slashdot/G_WL5    data/slashdot/output_WL5  --problemsize 80000 --solver LocalL2Solver --logfile data/slashdot/logs/matching_WL5_massive.log --N 20000 --rhoP 10.0 --rhoQ 5.0 --rhoT 5.0 --objectivefile data/slashdot/objectives_WL5 --maxiter 1000 --dump_trace_freq 5 --checkpoint_freq 5


Preprocessor
------------

The Graph Preprocessor takes the following arguments

	positional arguments:
	  graph1                File containing first graph
	  graph2                File containing second graph
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --outputconstraintfile OUTPUTCONSTRAINTFILE
	                        File for constraint graph. (default: None)
	  --constraintmethod {degree,all,WL}
	                        Constraint generation method (default: degree)
	  --debug {DEBUG,INFO,WARNING,ERROR,CRITICAL}
	                        Debug level (default: INFO)
	  --N N                 Number of partitions (default: 8)
	  --degreedistance DEGREEDISTANCE
				Distance of degrees (default: 0)
	  --k K                 Number of iterations (default: 10)
	  --outputobjectivefile OUTPUTOBJECTIVEFILE
				Output file for objectives (default: None)
	  --objectivemethod OBJECTIVEMETHOD
				Objective type (default: AP-PB)
	  --storage_level {MEMORY_ONLY,MEMORY_AND_DISK,DISK_ONLY}
				Control Spark caching/persistence behavrior (default:
				MEMORY_ONLY)
	  --inputconstraintfile INPUTCONSTRAINTFILE
				Input file for constraints. If not given, constraints
				are generated and stored in file named as specified by
				---constrainfile (default: None)
	  --checkpointdir CHECKPOINTDIR
				Directory to be used for checkpointing (default:
				checkpointdir)
	  --logfile LOGFILE     Log file (default: preprocessor.log)
	  --driverdump          Collect output and dump it from driver (default:
				False)
	  --slavedump           Dump output directly from slaves (default: False)
	  --fromsnap            Inputfiles are from SNAP (default: False)
	  --notfromsnap         Inputfiles are pre-formatted (default: False)
	  --undirected          Treat inputs as undirected graphs; this is the default
				behavior. (default: True)
	  --directed            Treat inputs as directed graphs. Edge (i,j) does not
				imply existence of (j,i). (default: True)

GraphMatching
-------------

The graph matching python program takes the following arguments:

	usage: GraphMatching.py [-h] [--graph1 GRAPH1] [--graph2 GRAPH2]
				[--objectivefile OBJECTIVEFILE]
				[--linear_term LINEAR_TERM]
				[--problemsize PROBLEMSIZE]
				[--solver {LocalL1Solver,LocalL2Solver,FastLocalL2Solver}]
				[--debug {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
				[--logfile LOGFILE] [--maxiter MAXITER] [--N N]
				[--rhoP RHOP] [--rhoQ RHOQ] [--rhoT RHOT]
				[--alpha ALPHA] [--dump_trace_freq DUMP_TRACE_FREQ]
				[--checkpoint_freq CHECKPOINT_FREQ]
				[--checkpointdir CHECKPOINTDIR] [--silent] [--lean]
				[--directed] [--driverdump]
				constraintfile outputfile

	Parallel Graph Matching over Spark.

	positional arguments:
	  constraintfile        File containing graph of constraints.
	  outputfile            Output file storing learned doubly stochastic matrix Z

	optional arguments:
	  -h, --help            show this help message and exit
	  --graph1 GRAPH1       File containing first graph (optional). (default:
				None)
	  --graph2 GRAPH2       File containing second graph (optional). (default:
				None)
	  --objectivefile OBJECTIVEFILE
				File containing pre-computed objectives. Graphs need
				not be given if this argument is set. (default: None)
	  --linear_term LINEAR_TERM
				Linear term to be added in the objective (default:
				None)
	  --problemsize PROBLEMSIZE
				Problem size. Used to initialize uniform allocation,
				needed when objectivefile is passed (default: 1000)
	  --solver {LocalL1Solver,LocalL2Solver,FastLocalL2Solver}
				Local Solver (default: LocalL1Solver)
	  --debug {DEBUG,INFO,WARNING,ERROR,CRITICAL}
				Verbosity level (default: INFO)
	  --logfile LOGFILE     Log file (default: graphmatching.log)
	  --maxiter MAXITER     Maximum number of iterations (default: 5)
	  --N N                 Number of partitions (default: 8)
	  --rhoP RHOP           Rho value, used for primal variables P (default: 1.0)
	  --rhoQ RHOQ           Rho value, used for primal variables Q (default: 1.0)
	  --rhoT RHOT           Rho value, used for primal variables T (default: 1.0)
	  --alpha ALPHA         Alpha value, used for dual variables (default: 0.05)
	  --dump_trace_freq DUMP_TRACE_FREQ
				Number of iterations between trace dumps (default: 10)
	  --checkpoint_freq CHECKPOINT_FREQ
				Number of iterations between check points (default:
				10)
	  --checkpointdir CHECKPOINTDIR
				Directory to be used for checkpointing (default:
				checkpointdir)
	  --silent              Run in efficient, silent mode, with final Z as sole
				output. Skips computation of objective value and
				residuals duning exection and supresses both
				monitoring progress logging and trace dumping.
				Overwrites verbosity level to ERROR (default: False)
	  --lean                Run in efficient, ``lean mode, with final Z as sole
				output. Skips computation of objective value and
				residuals duning exection and supresses both
				monitoring progress logging and trace dumping. It is
				the same as --silent, though it still prints some
				basic output messages, and does not effect verbosity
				level. (default: False)
	  --directed            Input graphs are directed, i.e., (a,b) does not imply
				the presense of (b,a). (default: False)
	  --driverdump          Dump final output Z after first collecting it at the
				driver, as opposed to directly from workers. (default:
				False)

