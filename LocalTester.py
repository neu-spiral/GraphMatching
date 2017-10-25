from LocalSolvers import LocalL1Solver
import argparse
import logging

def readlinelist(file):
    with open(file,'r') as f:
	l=map(eval,list(f.readlines()))
    return l

def countfun(i,j,n1,n2):
        return (j-1)*n2+i

    
def generateZbar(objectives):
    objectives = dict(objectives)
    Zbar = {}
    for key in objectives:
	l1,l2 = objectives[key]
	for x in l1+l2:
	    Zbar[x]=0.0

    n1 = len(set([ i for (i,j) in Zbar.keys()]))
    n2 = len(set([ j for (i,j) in Zbar.keys()]))
    for (i,j) in Zbar:
	Zbar[(i,j)] = countfun(i,j,n1,n2)/(32)

    return Zbar

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Serial Graph Matching.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objectives',help ="File containing objectives")
    parser.add_argument('--solver',default='LocalL1Solver', help='Local Solver')
    parser.add_argument('--debug',default='INFO', help='Debug level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument('--maxiter',default=100,type=int, help='Maximum number of iterations')
    parser.add_argument('--rho',default=1.0,type=float, help='Rho value, used for primal variables')
    parser.add_argument('--alpha',default=1.0,type=float, help='Alpha value, used for dual variables')
    parser.set_defaults(undirected=True)

    args = parser.parse_args()

    SolverClass = eval(args.solver)	
    logging.basicConfig(level = eval('logging.'+args.debug))
   
    logging.info('Reading objectives...' )
    objectives = readlinelist(args.objectives)
    logging.info('...done')
    logging.debug('Read objectives: %s' % str(objectives))	
 
    logging.info('Generate Zbar...' )
    #Zbar = generateZbar(objectives) 	
    Zbar={}
    Zbar[(1,1)] = 1.0 / 32.0 
    Zbar[(2,1)] = 2.0 / 32.0 
    Zbar[(3,1)] = 3.0 / 32.0 
    Zbar[(4,1)] = 4.0 / 32.0 
    Zbar[(1,2)] = 5.0 / 32.0 
    Zbar[(2,2)] = 6.0 / 32.0 
    Zbar[(3,2)] = 7.0 / 32.0 
    Zbar[(4,2)] = 8.0 / 32.0 
    Zbar[(1,3)] = 9.0 / 32.0 
    Zbar[(2,3)] =10.0 / 32.0 
    Zbar[(3,3)] = 100.0 #11.0 / 32.0 
    Zbar[(4,3)] =12.0 / 32.0 
    Zbar[(1,4)] =13.0 / 32.0 
    Zbar[(2,4)] =14.0 / 32.0 
    Zbar[(3,4)] =15.0 / 32.0 
    Zbar[(4,4)] =16.0 / 32.0 
    logging.info('...done')
    logging.info('Zbar is: '+str(Zbar))

    solver = SolverClass(objectives)
    sol,stats = solver.solve(Zbar,args.rho)
    print sol
    print stats


