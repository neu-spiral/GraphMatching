import sys,argparse
from pyspark import SparkContext
from operator import add

def swap((x,y)):
    return (y,x)

def update(d,key,val):
    if val is None:
	val = 0
    d[key]=d[key]+[val]
    return d




def get_all_neighborhoods(graph,N,k):
    nodes = graph.flatMap(lambda (u,v):[u,v]).distinct()

    all_neighborhoods= nodes\
              .map(lambda u:(u,[]))\
              .partitionBy(N).cache()
    
    current_neighborhood = graph.distinct()\
                                .cache()
    
    	
    i = 0
    while i<k:
	i += 1

	#print current_neighborhood.collect()

	neighborhoods = current_neighborhood\
                        .mapValues(lambda v:1)\
                        .reduceByKey(add,numPartitions=N)


	all_neighborhoods=all_neighborhoods\
			.leftOuterJoin(neighborhoods,numPartitions=N)\
			.mapValues(lambda (neighs,val):neighs+[val])
	

	old = current_neighborhood
    	current_neighborhood = current_neighborhood.map(swap)\
			.join(graph,numPartitions=N)\
			.values()\
			.union(current_neighborhood)\
			.distinct()\
			.partitionBy(N)\
			.cache()
   	
        old.unpersist() 

    return all_neighborhoods



def get_cycles_and_paths(graph,N,k):
    nodes = graph.flatMap(lambda (u,v):[u,v]).distinct()

    all_cycles_and_paths= nodes\
              .map(lambda u:(u,{'cycles':[],'paths':[]}))\
              .partitionBy(N).cache()

    current_path = graph.mapValues(lambda x:x)\
                                .cache()
    
    	
    i = 0
    while i<k:
	i += 1
        cycles = current_path\
                     .filter(lambda (u,v):u==v)\
                     .map(lambda (u,v):(u,1))\
                     .reduceByKey(add,numPartitions=N)

	paths = current_path\
                        .mapValues(lambda v:1)\
                        .reduceByKey(add,numPartitions=N)

	all_cycles_and_paths=all_cycles_and_paths\
			.leftOuterJoin(cycles)\
			.mapValues(lambda ( d,val):update(d,'cycles',val) )\
			.leftOuterJoin(paths)\
			.mapValues(lambda (d,val):update(d,'paths',val))

	old = current_path
    	current_path = current_path.map(swap)\
			.join(graph,numPartitions=N)\
			.values()\
			.partitionBy(N)\
			.cache()
   	
        old.unpersist() 

        all_cycles_and_paths.mapValues(lambda d: dict([  (key+'_'+str(i),1.*val/(i+1))     for key in d  for (i,val) in zip(range(len(d[key])),d[key])  ]  ))
 
    return all_cycles_and_paths

def get_page_rank(graph,N,eps,max_iterations,gamma):
    graph_rdd = graph.groupByKey()\
		   .mapValues(list)\
                   .partitionBy(N)\
		   .cache()



    # Discover all nodes; this finds node with no outgoing edges as well	
    nodes = graph_rdd.flatMap(lambda (i,edgelist):edgelist+[i])\
	              .distinct()\
		      .cache()
 
    

    #Initialize scores
    size = nodes.count()
    scores = nodes.map(lambda i: (i, 1.0/size )).partitionBy(N).cache()

    #Main iterations
    i = 0
    err = eps+1.0
    while i<max_iterations and err > eps  :
	 i += 1
	 old_scores = scores
	 joined = graph_rdd.join(scores)
	 scores = joined.values()\
			.flatMap(lambda (neighborlist,score): [  (x,1.0*score/len(neighborlist) )  for x in neighborlist  ] )\
			.reduceByKey(lambda x,y:x+y,numPartitions=N)\
			.mapValues(lambda x: (1-gamma)*x+gamma*1/size)\
			.cache()

	 
         err = old_scores.join(scores).values()\
                         .map(lambda (old_val,new_val): abs(old_val-new_val))\
		         .reduce(lambda x,y:x+y)

	 old_scores.unpersist()
	 print '### Iteration:',i,'\terror:',err 
 
    # Give score to nodes having no incoming edges. All such nodes
    # should get score gamma / size
    remaining_nodes = nodes.map(lambda x: (x,gamma/size)).subtractByKey(scores)	    
    scores = scores.union(remaining_nodes)	

    return scores

