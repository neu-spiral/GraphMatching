import logging
from pprint import pformat

logger = logging.getLogger('Graph Matching')

def Sij_test(G,graph1,graph2,Sij):
    graph1 = set(graph1.collect())
    graph2 =set(graph2.collect())
    G = set(G.collect())
    Sij = Sij.collectAsMap()  
 
    s = ""
    condition = True
 
    for (i,j) in Sij:
	s += "Pair %s has the following edges in Sij1: \n" % str((i,j))
        for (k1,k2) in Sij[(i,j)][0]:
	    inG = (k1,k2) in G
	    isj = k2 == j
	    ingraph1 = (i,k1) in graph1
	    condition = condition and inG and isj and ingraph1
	    s += "\t %s: \n" % str((k1,k2))
	    s += "\t\t%s is in G: %s \n" % (str((k1,k2)),str(inG))
	    s += "\t\t%s is %s: %s \n" % (str(k2),str(j),str(isj))
	    s += "\t\t%s is in graph1: %s \n" % (str((i,k1)),str(ingraph1))
    
	s += "Pair %s has the following edges in Sij2:\n" % str((i,j))
        for (k1,k2) in Sij[(i,j)][1]:
	    inG = (k1,k2) in G
	    isi = k1 == i
	    ingraph2 = (k2,j) in graph2
	    condition = condition and inG and isi and ingraph2
	    s += "\t %s: \n" % str((k1,k2))
	    s += "\t\t%s is in G: %s \n" % (str((k1,k2)),str(inG))
	    s += "\t\t%s is %s: %s \n" % (str(k1),str(i),str(isi))
	    s += "\t\t%s is in graph1: %s \n" % (str((k2,j)),str(ingraph2))

    if condition:
	return "Sij test passed, Sij dump is:\n"+s
    else:
	return "Sij test failed, Sij dump is:\n"+s

def dumpBasic(G,graph1,graph2):
    graph1 = set(graph1.collect())
    graph2 = set(graph2.collect())
    G = set(G.collect())
    logger.debug("----Basic Structures----\n")
    logger.debug("Graph 1: "+pformat(graph1,width=30)+'\n')
    logger.debug("Graph 2: "+pformat(graph2,width=30)+'\n')
    logger.debug("G: "+pformat(G,width=30)+'\n')




def dumpPPhiRDD(PPhiRDD):
    PPhi = PPhiRDD.collectAsMap()
    logger.debug("----PPhiRDD----\n")
    for ind in PPhi:
	    solver,P,Phi,stats = PPhi[ind]
	    logger.debug("Partition %d contains: \n" % ind)
	    logger.debug("\t Solver: "+ str(solver)) 
	    logger.debug("\t P and Phi variables: \n")
  	    for x in P:
	        logger.debug("\t\t%s:%s %s\n" % (str(x),str(P[x]),str(Phi[x])) ) 
	    logger.debug("\t Stats: \n")
	    for x in stats:
	        logger.debug("\t\t%s:%s \n" % (str(x),str(stats[x]))) 
	      

