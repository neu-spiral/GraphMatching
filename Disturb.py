import numpy as np
import pickle
import random
import argparse


def Graph2AdjcanecyMat(G, nodes):
    A = np.zeros((nodes, nodes))
    A = np.matrix(A)
    for (i,j) in G:
        A[eval(i), eval(j)] = 1.
    return A
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Graph Preprocessor .',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',help = 'File containing the graph')
    parser.add_argument('out',help = 'File to store the permutated graph')
    parser.add_argument('size',type=int,help='Graph size')
    parser.add_argument('outliers_nodes', type=int, help="Number of nodes that are outliers.")
    parser.add_argument('--seed', type=int, default=1993, help="Seed to random generation.")
    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=True)



    args = parser.parse_args()
    
    perm = np.random.RandomState(seed=1993).permutation(args.size)
    print(perm)
    
    #Map each node in GA (key) to a node in GBi (value).
    permDict = dict([(str(ii), str(perm[ii])) for ii in range(len(perm))]) 
  

    outlier_nodes = np.random.choice(args.size, args.outliers_nodes)
    print(outlier_nodes)

    #Get curret list of edges 
    G_disturbed = []
    with open(args.graph, 'r') as Gfile:
        for l_ij in Gfile:
            i,j = l_ij.split()
            if eval(i) in outlier_nodes or eval(j) in outlier_nodes:
                continue 
 
            G_disturbed.append( (permDict[i], permDict[j] ) )
            if i==j:
                print ("circle")

            if (permDict[j], permDict[i]) not in G_disturbed:
                G_disturbed.append((permDict[j], permDict[i]  ))

    
        

   #add outliers
    for node in outlier_nodes:
        for node_j in range(args.size):
            if (permDict[str(node)], permDict[str(node_j)] ) not in  G_disturbed:
                G_disturbed.append( (permDict[str(node)], permDict[str(node_j)]) )      

            if (permDict[str(node_j)], permDict[str(node)] ) not in  G_disturbed: 
                G_disturbed.append(  (permDict[str(node_j)], permDict[str(node)])  )

    with open(args.out, 'w') as outF:
        for (i,j) in G_disturbed:
            print(i,j)
            outF.write("%s %s\n" %(i,j))
           
               
