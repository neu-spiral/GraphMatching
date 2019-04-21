import argparse
import random
import networkx as nx
from networkx.readwrite import node_link_data
import json
import numpy as np
from pyspark import SparkContext, SparkConf

#Macros
N_WALKS=50
WALK_LEN=5


def randomWalk(G, k):
    """
    Return random wlaks of length k starting from every node of the graph G. G is a networkx graph.
    """
    random_walks = {}
    for node in G.nodes:
        random_walks[node] = [node]
        for l in range(k):
            random_walks[node].append( np.random.choice(list(G.neighbors(node) ), 1)[0] )
    return random_walks
def get_cooccurences(G, k, t, w):
     """
     Return a list of tuples (u,v), s.t., u and v co-occur on t random walks of length k. w is the size of the sliding window (see Alg. 2 in "DeepWalk: Online Learning of Social Representations" by Perozzi et al).
     """
     co_occurences = {}
     for iteration in range(t):
         random_walks = randomWalk(G, k)
         for src_node in random_walks:
             i = 0
             for node1 in random_walks[src_node]:
                 for node2 in random_walks[src_node][max(0, i-w):i] + random_walks[src_node][i+1:i+w+1]:
                     if node1 not in co_occurences:
                         co_occurences[node1] = set([node2])
                     else:
                         co_occurences[node1].update([node2])
                #     if node2 not in co_occurences: 
                #         co_occurences[node2] = set([node1])
                #     else:
                #         co_occurences[node2].update([node1])
                         
                 i += 1
     return co_occurences
           
def run_random_walks(G, nodes, num_walks=N_WALKS):
    """
        Generete radnom walks, copied from GraphSAGE/utilities.py
    """
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(list(G.neighbors(curr_node)))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        #if count % 1000 == 0:
    print("Done walks for", count, "nodes")
    return pairs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Graph Preprocessor .',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('graph',help = 'File containing first graph') 
    parser.add_argument('prefix',help = 'File to write the networkx formatted graph.')
    parser.add_argument('--feat',default=None,help = 'JSON file storing features.')
    parser.add_argument('--attr',default=None,help = 'File storing attributes.')
    args = parser.parse_args()

    configuration = SparkConf()
    configuration.set('spark.default.parallelism',20)
    sc = SparkContext(appName='Parallel Graph Preprocessing', conf=configuration)
    
    G = nx.Graph()
    attrs = {}
    id_map = {}
    cls_map = {}

    #Processing exogenous features
    if args.feat != None:
        with open(args.feat, 'r') as cls_file:
            all_exogenous_feats = json.load(cls_file)
            exogenous_dict = dict([(node, all_exogenous_feats[node][77:79])  for node in all_exogenous_feats] )

    #Processing attributes
    if args.attr != None:
        attrs_dict = dict(sc.textFile(args.attr).map(eval).collect())
        print attrs_dict.keys()
        feats = []
        order = []
        for node_id in attrs_dict:
             attrs_node = attrs_dict[node_id]
             if args.feat:
                 #Append exogenous features to the attributes 
                 attrs_node += exogenous_dict[node_id]
             feats.append(attrs_node)  
             order.append(eval(node_id))
        feats = np.array(feats)
        feats  = feats[order]
            
          
    with open(args.graph) as Gfile:
        for l in Gfile:
            (u, v) = l.split()
            u = eval(u)
            v = eval(v)
            G.add_edge(u, v)
            if u not in attrs:
                attrs[u] = {}
                #Dummy class labels
                cls_map[u] = 1
                id_map[u] = int(u)
            if v not in attrs:
                attrs[v] = {}
                #Dummy class labels 
                cls_map[v] = 1
                id_map[v] = int(v)


    print "Number of nodes and edges are %d %d" %(len(G.nodes), len(G.edges()) )
    #Writing attributes
    for u in attrs:
        if random.random() < 0.7:
            attrs[u] = {'val':True, 'test':False}
        else:
            attrs[u] = {'val':False, 'test':False}

    # Add attributes to graph
    nx.set_node_attributes(G, attrs)

    #random walks
    train_nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    pairs = run_random_walks(G.subgraph(train_nodes), train_nodes)


    #Convert graph to node_link data form for serialization.
    G_data = node_link_data(G)
    #Write data to files
    with open(args.prefix + '-G.json', 'w') as outFile:
        json.dump(G_data, outFile)
    with open(args.prefix + '-class_map.json', 'w') as outFile:
        json.dump(cls_map, outFile)
    with open(args.prefix + '-id_map.json', 'w') as outFile:
        json.dump(id_map, outFile)
    with open(args.prefix + '-walks.txt', 'w') as outFile:
        outFile.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
    np.save(args.prefix + '-feats', feats)
       
   


