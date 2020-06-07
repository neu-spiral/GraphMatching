import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import pickle
from pyspark import SparkConf, SparkContext
from preprocessor import degrees, readSnap
from scipy.optimize import linear_sum_assignment
from LocalSolvers import LocalL1Solver

#plt.rc('font', serif='Times New Roman')
#plt.rc('font', size=15)


def Graph2AdjcanecyMat(G, nodes):
    A = np.zeros((nodes, nodes))
    A = np.matrix(A)
    for (i,j) in G:
        A[eval(i), eval(j)] = 1.
    return A
    
def WeigtGraph2djcanecyMat(G, nodes):
    A = np.zeros((nodes, nodes))
    A = np.matrix(A)
    for ((i,j), w_ij) in G:
         A[eval(i), eval(j)] = w_ij
    return A


def ToDict(orderedDegrees1):
    i = 0
    d = {}
    for (node, deg) in orderedDegrees1:
        d[node] = i
        i += 1
    return d
def readMatlabMat(matFile):
    P = {}
    with open(matFile, 'r') as matF:
        for l in matF:
            col, row, val = l.split()
            row = eval(row)
            col = eval(col)
            row -= 1
            col -= 1
            P[(unicode(row), unicode(col))] = eval(val)
    return P
            
def plotter(Parray, outfile):
    xLabel = 'G1'
    yLabel = 'G2'

    (n1 , n2) = Parray.shape

    #Plot
    fig, ax = plt.subplots()
    im = ax.imshow(Parray)

    ax.set_xticks(np.arange(0, n1, 20))
    ax.set_yticks(np.arange(0, n2, 20))
    ax.tick_params(axis='both',top=False, bottom=False,left=False,labelleft=False,
                   labeltop=False, labelbottom=False)
    #ax.set_xticklabels([str(i) for i in np.arange(0, n1, 10)],fontsize=18)
    #ax.set_yticklabels([str(i) for i in np.arange(0, n2, 10)], fontsize=18)

    cbar = fig.colorbar(im, ax=ax,ticks=[0, np.max(Parray)])
#    cbar.ax.set_ylabel("values", rotation=-90, va="bottom")



# Loop over data dimensions and create text annotations.
#    for i in range(n1):
#        for j in range(n2):
#            text = ax.text(j, i, Parray[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.show()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Extracting a heat map.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mapping',default=None,help ="File containing the mapping of the nodes. ")
    
    parser.add_argument('graph1',default=None,help = 'File containing first graph.')
    parser.add_argument('graph2',default=None,help = 'File containing second graph.')
    parser.add_argument('--size',default=64,type=int,help = 'File containing second graph.')
    parser.add_argument('--obj',default=None,help = 'Ojectives file')
    parser.add_argument('--N',default=60,type=int, help='Number of partitions')
    parser.add_argument('--outfile',default=None,help = 'Figures name')

    parser.add_argument('--readMode',default='pickle', choices={'sc','pickle','matlab'},help = 'Reading mode')
    parser.set_defaults(sc=False)

    args = parser.parse_args() 



    with open(args.graph1+ "_perm", 'r') as permF:
        perm = np.load(permF)

    if args.readMode == 'sc':
        sc = SparkContext()
        P = sc.textFile(args.mapping).map(eval).collect()
        Pdict = dict(P)
       
    elif args.readMode == 'pickle':
        with open(args.mapping,'r') as mapF:
            Pdict = pickle.load(mapF)
    else:
        Pdict = readMatlabMat(args.mapping)
    
    #OBJs =  sc.textFile(args.obj).map(eval).collect() 
    #L1Solver_obj = LocalL1Solver(OBJs, 1.0)
    

   # P = sc.textFile(args.mapping).map(eval).partitionBy(args.N).cache()
   # graph1 = sc.textFile(args.graph1).map(eval).partitionBy(args.N).cache()
   # graph2 = sc.textFile(args.graph1).map(eval).partitionBy(args.N).cache() 
#    graph1 = readSnap(args.graph1, sc, args.N)
#    graph2 = readSnap(args.graph2, sc, args.N)

#    graph1 = graph1.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()
#    graph2 = graph2.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()
#    n1 = graph1.flatMap(lambda (u,v):[u,v]).distinct().count()
#    n2 = graph2.flatMap(lambda (u,v):[u,v]).distinct().count()

#    n_min = min(n1, n2)
#    n_max = max(n1, n2)

#    dummyNodes = [(str(dum), str(dum) ) for dum in range(n_min, n_max)]
#    dummyNodes = sc.parallelize(dummyNodes).partitionBy(args.N)

#    if n1<n2:
           # graph1 = graph1.filter(lambda (u,v): eval(u)<n2 and eval(v)<n2).cache()
#        graph1 = graph1.union(dummyNodes).partitionBy(args.N).cache()
#    else:
#        graph2 = graph2.union(dummyNodes).partitionBy(args.N).cache()

#    graph1 = graph1.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()
#    graph2 = graph2.flatMap(lambda (u,v):[ (u,v),(v,u)]).distinct()


#    n1 = graph1.flatMap(lambda (u,v):[u,v]).distinct().count()
#    n2 = graph2.flatMap(lambda (u,v):[u,v]).distinct().count()
  #  A = Graph2AdjcanecyMat(graph1.collect(), n_max)
  #  B = Graph2AdjcanecyMat(graph2.collect(), n_max)
    
#    orderedDegrees1 = degrees(graph1, 0, args.N).mapValues(lambda (inward, outward): inward+outward).takeOrdered(n_max, key = lambda x:x[1])
#    orderedDegrees2 = degrees(graph2, 0, args.N).mapValues(lambda (inward, outward): inward+outward).takeOrdered(n_max, key = lambda x:x[1])


    orderedDegrees1 = [unicode(i) for i in range(args.size)]
    orderedDegrees2 = [unicode(i) for i in  perm]
    
    

   # Pcolected = P.collect()
   # print "Objective is ", L1Solver_obj.evaluate(dict(Pcolected) )
   # Pmat = WeigtGraph2djcanecyMat(Pcolected, n_max)
   # APmPB = A*Pmat - Pmat*B
   # tmp = 0.
   # p = 2.
   ## for i in range(n_max):
   #     for j in range(n_max):
    #        tmp += (np.abs(APmPB[i,j]))**p
   # print tmp**(1./p)
   # Pdict = dict( Pcolected  )
    

    PheatList = []
    for i in range(args.size):
        row = []
        for j in range(args.size):
            try:
                row.append(Pdict[(orderedDegrees1[i], orderedDegrees2[j])])
                 
            except KeyError:
                 row.append(0.0)
        PheatList.append(row)
    PheatArray = np.array( PheatList  )

    plotter(PheatArray, args.outfile + ".pdf")
    
    
   #Find mathcing (by projection on the set of permutaion matrices via Hungarian method)
    Cost_Mat = np.zeros((args.size, args.size))
    for (node_i, node_j) in Pdict:
        Cost_Mat[eval(node_i)][eval(node_j)] = -Pdict[(node_i, node_j)]
   
    row_ind, col_ind  = linear_sum_assignment(Cost_Mat)
    print row_ind, col_ind
    #Compute matching stats
    ell1_diff = 0.0
    for i in range(args.size):
        for j in range(args.size):
            if i==j:
                ell1_diff += abs(Pdict[(orderedDegrees1[i], orderedDegrees2[j])]-1)
            else:
                ell1_diff += abs(Pdict[(orderedDegrees1[i], orderedDegrees2[j])])
            

    stats = {}
    stats['ell1Diff'] = ell1_diff
    print "The ell1 diff. between the solution and the true perm. matrix is %f\n" %ell1_diff
    stats['diagMass'] = sum([Pdict[(orderedDegrees1[i], orderedDegrees2[i])] for i in range(args.size)] )
    print "Digonal mass is %f" %stats['diagMass']
    stats['mismatch'] = np.linalg.norm(np.array(col_ind) - perm, 0)
    print "mismatch is %d" %stats['mismatch']
    with open(args.outfile + '_stats', 'w') as statsFile:
        pickle.dump(stats, statsFile)
    
