import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import argparse
import pickle
#from pyspark import SparkConf, SparkContext
#from preprocessor import degrees, readSnap
from scipy.optimize import linear_sum_assignment

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
    parser.add_argument('mappings', metavar= 'mapping', nargs='+',  help ="File containing the mapping of the nodes. ")
    parser.add_argument('--outfile', help = "File to store results name")

    parser.add_argument('--perm_file', default=None, help="File storing the true mapping")
    parser.add_argument('--size',default=64,type=int,help = 'File containing second graph.')
    parser.add_argument('--obj',default=None,help = 'Ojectives file')
    parser.add_argument('--N',default=60,type=int, help='Number of partitions')

    parser.add_argument('--readMode',default='np', choices={'sc','pickle','matlab', 'np'},help = 'Reading mode')
    parser.add_argument('--no_cbar', action='store_false', help='Pass to not plot the color-bar', dest='cbar')
    parser.set_defaults(sc=False)

    args = parser.parse_args() 



    
    if args.perm_file:
    	with open(args.perm_file, 'r') as permF:
            perm = np.load(permF)
    else:
        perm = np.random.RandomState(seed=1993).permutation(args.size)

    ordered1 = list(range(args.size))
    ordered2 = list(perm)

    stats = {}
    

    for input_ind, mapping in enumerate(args.mappings):
        if args.readMode == 'sc':
            sc = SparkContext()
            P = sc.textFile(mapping).map(eval).collect()
            Pdict = dict(P)
       
        elif args.readMode in ['pickle', 'np']:
            with open(mapping,'rb') as mapF:
                Pdict = pickle.load(mapF)
        else:
            Pdict = readMatlabMat(mapping)
    
        #permutate columns based on ordering
        if args.readMode == 'np':
            PheatArray = Pdict[:, ordered2]
        else:
            PheatList = []
            for i in range(args.size):
                row = []
                for j in range(args.size):
                    try:
                        row.append(Pdict[(ordered1[i], ordered2[j])])
                 
                    except KeyError:
                        row.append(0.0)
                PheatList.append(row)
            PheatArray = np.array( PheatList  )

       
    
    
       #Find mathcing (by projection on the set of permutaion matrices via Hungarian method)
        Cost_Mat = -1 * PheatArray
        row_ind, col_ind  = linear_sum_assignment(Cost_Mat)

        if input_ind == 0:
            PheatArray_tot = PheatArray

            #Compute matching stats
            stats['ell1Diff'] = np.sum( np.absolute( PheatArray - np.eye(args.size)) ) 
            stats['DPM'] = np.sum( np.diagonal(PheatArray) ) / args.size
            stats['mismatch'] = np.linalg.norm(np.array(col_ind) - np.array(range(args.size)), 0) / args.size
            stats['DPM-P'] = 1 - stats['mismatch']
        else:
            PheatArray_tot += PheatArray

            #Compute matching stats
            stats['ell1Diff'] += np.sum( np.absolute( PheatArray - np.eye(args.size)) )
            stats['DPM'] += np.sum( np.diagonal(PheatArray) ) / args.size
            stats['mismatch'] += np.linalg.norm(np.array(col_ind) - np.array(range(args.size)), 0) / args.size
            stats['DPM-P'] += 1 - np.linalg.norm(np.array(col_ind) - np.array(range(args.size)), 0) / args.size
            print(1 - np.linalg.norm(np.array(col_ind) - np.array(range(args.size)), 0) / args.size, np.sum( np.diagonal(PheatArray) ) / args.size )

    for key in stats:
        stats[key] /= len(args.mappings)
    PheatArray_tot /= len(args.mappings)
    print("Loaded {} solutions".format(len(args.mappings) ) )
    print( "Digonal mass is {}".format(stats['DPM']) )
    print("DPM-P is {}".format(stats['DPM-P']))
       


    with open(args.outfile + '_stats', 'wb') as statsFile:
        pickle.dump(stats, statsFile)
    


    #plot heatmap
    hm_plt = seaborn.heatmap(PheatArray, vmin=0.0, vmax=1.0, cmap='gist_gray', xticklabels=False, yticklabels=False, cbar=args.cbar, square=True)
    hm_plt.get_figure().savefig(args.outfile + ".pdf", bbox_inches='tight')
