import numpy as np
import pickle
import random
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Graph Preprocessor .',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',help = 'File containing the graph')
    parser.add_argument('out',help = 'File to store the permutated graph')
    parser.add_argument('size',type=int,help='Graph size')
    parser.add_argument('--scale',type=float, default=0.001,help='The standard deviation of noise..')
    snap_group = parser.add_mutually_exclusive_group(required=False)
    snap_group.add_argument('--fromsnap', dest='fromsnap', action='store_true',help="Inputfiles are from SNAP")
    snap_group.add_argument('--notfromsnap', dest='fromsnap', action='store_false',help="Inputfiles are pre-formatted")
    parser.set_defaults(fromsnap=True)

    parser.add_argument('--noise', choices=['normal', 'laplace', 'both'], help="Noise type")
    parser.add_argument('--mix_noise_weight', type=float, default=0.5, help="The coeff. of normal distributed weights, only relevant if noise is set to 'both'.")


    args = parser.parse_args()
    
    weights = {}
    
    #generate weights 
    for i in range(args.size):
        for j in range(args.size):
            if (j,i) in weights or (i,j) in weights:
                continue 

            if args.noise == 'normal':
                weights[(i,j)] = np.random.normal(loc=0.0,scale=args.scale)
            elif args.noise == 'laplace':
                weights[(i,j)] = np.random.laplace(loc=0.0,scale=args.scale)
            elif args.noise == 'both':
                weights[(i,j)] = args.mix_noise_weight * np.random.normal(loc=0.0,scale=args.scale) + (1-args.mix_noise_weight) * np.random.laplace(loc=0.0,scale=args.scale) 
                 
            weights[(j,i)] = weights[(i,j)]  
    
    print (weights[(22, 55)], weights[(55, 22)])

    out_file_name = args.out + '_weights_' + args.noise + str(args.scale)
    if args.noise == 'both':
        out_file_name += '_mixcoeff' + str(args.mix_noise_weight)

    with open(out_file_name, 'wb') as fW:
        pickle.dump(weights, fW)
              

