import cvxpy as cp
import glob

import pickle
import argparse
import numpy as np

def read_dist_file( dirname, graph_size, prefix="part"):
    """
        Read distnace file, each line is a key valued pair; keys are node pairs and values are the corresponding distance between the nodes.
    """
    D = np.zeros((graph_size, graph_size))
    for partFile in  glob.glob(dirname + "/{}*".format(prefix) ):
        print( "Now readaiang " + partFile)
        with open(partFile, 'r') as pF:
            for vals_line in pF:
                (node_pair, dist) = eval(vals_line)
                row = eval(node_pair[0])
                col = eval(node_pair[1])
                D[(row, col)] = dist
    return D

def readGraph(gfile, graph_size):
    """
        Read graph stored in gfile, and return adjacancy matrix as np array.
    """
    A  = np.zeros((graph_size, graph_size))
    with open(gfile) as graph_file:
        for g_line in graph_file:
            i, j = g_line.split()
            A[eval(i), eval(j)] = 1
    
    #make sure the adjacancy is symmetric
    A_symm = np.maximum( A, A.transpose() )
    return A_symm

def dict2array(a_dict, size):
    out = np.zeros((size, size))
    for key in a_dict:
        out[key] = a_dict[key]
    return out
    

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'CVXOPT Solver for Graph Matching',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph1', help="Text file containing the first graph.")
    parser.add_argument('graph2', help="Text file containing the second graph.")
    parser.add_argument('graph_size', type=int, help="Number of nodes in the graph.")
    parser.add_argument('outfile',type=str,help='File to store results.')

    parser.add_argument('--dist_file',default=None, type=str,help='File containing distace file.')
    parser.add_argument('--weights', default=None, type=str,help='File containing distace file.')
    parser.add_argument('--lamb', default=0.0, type=float, help='lambda parameter regularizing the linear term.')
    parser.add_argument('--epsilon', default=1.e-2, type=float, help='The accuracy for cvxopt solver.')
    parser.add_argument('--p', default=2.5, type=float, help='p parameter in p-norm')
    parser.add_argument('--ONLY_lin',action='store_true',help='Pass to ignore ||AP-PB||_p')
    parser.set_defaults(ONLY_lin=False)

    args = parser.parse_args()


    # Load problem data: graphs and distance file (if passed)
    A = readGraph(args.graph1, args.graph_size)
    B = readGraph(args.graph2, args.graph_size)
    #add weights (if given)
    if args.weights:
        W_dict = pickle.load(open(args.weights, 'rb')) 
        B += dict2array( W_dict, args.graph_size)

    if args.dist_file:
        D = read_dist_file(args.dist_file, args.graph_size)

    print(D)

    # Construct the problem.
    one_vec = np.ones(args.graph_size)

    #optimization variables
    P = cp.Variable( (args.graph_size, args.graph_size) )

    #if args.p == 2:
    #    loss_term = cp.norm2( cp.reshape( A @ P - P @ B , args.graph_size ** 2) ) 
    #else:
    loss_term  = cp.atoms.pnorm(A @ P - P @ B, p=args.p)
    if args.dist_file:
    	loss_term += args.lamb * cp.sum( cp.multiply(D, P) )

    objective = cp.Minimize(loss_term )
    constraints = [P @ one_vec == one_vec, P.T @ one_vec == one_vec, P >= 0]
    prob = cp.Problem(objective, constraints)

    if  args.p == 2:
        prob.solve(solver='SCS', verbose=True)
    else:
    	prob.solve(verbose=True)

    print("status:", prob.status)
    print("optimal value", prob.value)
    print("Constraints are ", np.sum(P.value, axis=0), np.sum(P.value, axis=1) )
    with open(args.outfile + "_P", 'wb') as sol_file:
        pickle.dump(P.value, sol_file)
    


