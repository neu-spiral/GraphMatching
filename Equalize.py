import argparse
import json
from shutil import copyfile
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Graph Preprocessor .',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',help = 'File containing first graph')
    parser.add_argument('--feat',default=None,help = 'File containing features of the graph.')
    args = parser.parse_args()

    f = open(args.graph, 'r')
    fNew = open('tmp', 'w')
    d = {}
    val = 0
    for l in f:
        try:
            u, v = l.split()
        except ValueError:
            continue 
        
        if u not in d:
            d[u] = val
            val +=1
        if v not in d:
            d[v] = val
            val += 1
        fNew.write("%d %d\n" %(d[u], d[v]))
        

    dInv = dict([(d[key], key) for key in d])
    f.close()
    fNew.close()
        
    copyfile('tmp', args.graph)
    os.remove('tmp')

    
    print "The number of nodes %d" %len(d)
    #Processing features (if given) 
    if args.feat != None:
        #Processing features
        print "Processing features..."
        featDICT = {}
        featFile = open(args.feat, 'r')
        for l in featFile:
            node_id, feat = l.split()[0], [eval(elem) for elem in l.split()[1:]]
            try:
                featDICT[d[node_id]] = feat
            except KeyError:
                continue 
     
        featFile.close()           
        with open(args.feat + '.json', 'w') as outfile:
            json.dump(featDICT, outfile)
    
