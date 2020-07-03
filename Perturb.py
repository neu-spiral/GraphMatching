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
    parser.add_argument('--keepProb',type=float, default=1.0,help='The probability of keeping an edge.')
    parser.add_argument('--seed', type=int, default=1993, help="Seed to random generation.")
    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=True)



    args = parser.parse_args()
    
    perm = np.random.RandomState(seed=1993).permutation(args.size)
    
    #Map each node in GA (key) to a node in GBi (value).
    permDict = dict([(ii, perm[ii]) for ii in range(len(perm))]) 
    with open(args.out+ "_perm", 'w') as pFile:
        np.save(pFile, perm)
  

    #Get curret list of edges 
    G = []
    
    with open(args.graph, 'r') as Gfile:
        for l_ij in Gfile:
            i,j = l_ij.split()
            G.append((eval(i), eval(j) ))
            if i==j:
                print "circle"
            if (j,i) not in G:
                G.append((eval(j),  eval(i)))

    G_prtb = []
    numb = 0
    for ind, i in enumerate(range(args.size)):
        for j in range(args.size):
            if (i, j) in G:
                numb += 1
                if random.random()<args.keepProb:
                    G_prtb.append((permDict[i], permDict[j]))

            else:
                if random.random()>=args.keepProb:
                    G_prtb.append((permDict[i], permDict[j]))

    print(len(G), len(G_prtb), numb )
    with open(args.out, 'w') as outF:
        for (i,j) in G_prtb:
            outF.write("%s %s\n" %(i,j))
           
               
