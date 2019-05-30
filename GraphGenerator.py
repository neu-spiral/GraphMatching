from networkx import Graph, DiGraph
import networkx
import logging,argparse
import topologies
import random
import numpy as np

def format(G):
    ''' Returns a sorted list with the edges in G formatted according to the following convention:
	-Nodes are numbered 1,2,...,n, where n is the number of nodes in G
	-Edges are sorted by the second element first and the second element later
    '''
    nodeMap ={}
    i = 1
    for v in sorted(G.nodes()):
	nodeMap[v] = i
	i +=1

    n = len(nodeMap)
    edges = [ (nodeMap[u],nodeMap[v]) for (u,v) in G.edges()]

    return sorted(edges,key=lambda (u,v): (v-1)*n+u-1) 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Topology Generator',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type',default="erdos_renyi",type=str, help='Graph type',choices=['erdos_renyi','balanced_tree','hypercube',"cicular_ladder","cycle","grid_2d",'lollipop','expander','hypercube','star', 'barabasi_albert','watts_strogatz','regular','powerlaw_tree','small_world','geant','abilene','dtelekom','servicenetwork'])
    parser.add_argument('--graph_size',default=100, type=int, help='Network size')
    parser.add_argument('--random_seed',default=123456, type=int, help='Random seed')
    parser.add_argument('--graph_degree',default=4, type=int, help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p',default=0.10, type=float, help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('output',help="outputfile")
    parser.add_argument('--sparseG',action='store_true',help='When passed create graphs with the probability set to 0.3*log(graph_size)/graph_size')
    parser.add_argument('--incPerm',action='store_true',help='Pass if you want to include a permutation matrix in the support.')

    parser.set_defaults(sparseG=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO) 
    random.seed(args.random_seed)
    np.random.seed(args.random_seed+2015 )


    if args.sparseG:
        graph_p = (1.0+args.graph_p)*np.log(args.graph_size)/args.graph_size
    else:
        graph_p = args.graph_p
    
    def graphGenerator():
	if args.graph_type == "erdos_renyi":
	    return networkx.erdos_renyi_graph(args.graph_size,graph_p)
	if args.graph_type == "balanced_tree":
	    ndim = int(np.ceil(np.log(args.graph_size)/np.log(args.graph_degree)))
	    return networkx.balanced_tree(args.graph_degree,ndim)
	if args.graph_type == "cicular_ladder":
	    ndim = int(np.ceil(args.graph_size*0.5))
	    return  networkx.circular_ladder_graph(ndim)
	if args.graph_type == "cycle":
	    return  networkx.cycle_graph(args.graph_size)
	if args.graph_type == 'grid_2d':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim,ndim)
	if args.graph_type == 'lollipop':
	    ndim = int(np.ceil(args.graph_size*0.5))
	    return networkx.lollipop_graph(ndim,ndim)
	if args.graph_type =='expander':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
	    return networkx.margulis_gabber_galil_graph(ndim)
	if args.graph_type =="hypercube":
	    ndim = int(np.ceil(np.log(args.graph_size)/np.log(2.0)))
	    return networkx.hypercube_graph(ndim)
	if args.graph_type =="star":
	    ndim = args.graph_size-1
	    return networkx.star_graph(ndim)
	if args.graph_type =='barabasi_albert':
	    return networkx.barabasi_albert_graph(args.graph_size,args.graph_degree)
	if args.graph_type =='watts_strogatz':
	    return networkx.connected_watts_strogatz_graph(args.graph_size,args.graph_degree,graph_p)
	if args.graph_type =='regular':
	    return networkx.random_regular_graph(args.graph_degree,args.graph_size)
	if args.graph_type =='powerlaw_tree':
	    return networkx.random_powerlaw_tree(args.graph_size)
	if args.graph_type =='small_world':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
	    return networkx.navigable_small_world_graph(ndim)
	if args.graph_type =='geant':
	    return topologies.GEANT()
	if args.graph_type =='dtelekom':
	    return topologies.Dtelekom()
	if args.graph_type =='abilene':
	    return topologies.Abilene()
	if args.graph_type =='servicenetwork':
	    return topologies.ServiceNetwork()
	

    logging.info('Generating graph: ' + args.graph_type )
    G = graphGenerator()

    if args.incPerm:
        for i in range(args.graph_size-1):
            G.add_edge(i, i+1)
        G.add_edge(args.graph_size-1, 0)
    logging.info('Saving to file: ' + args.output )
    with open(args.output,'w') as f:
        for e in format(G):
            
	   # f.write(str(e)+'\n')	
            f.write("%d %d\n" %(e[0], e[1]))
             
