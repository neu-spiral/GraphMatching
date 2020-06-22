import datetime
import json
import os
import shutil
#from google.cloud import storage

def readSnap(file,sc,minPartitions=10):
    '''Read a file in a format used by SNAP'''
    return sc.textFile(file,minPartitions=minPartitions)\
                .filter(lambda x: '#' not in x and '%' not in x)\
                .map(lambda x: tuple(x.split()))\
                .filter(lambda edge:len(edge)==2)
                #.map(lambda (u,v):(hash(u),hash(v)))
def readJSON(fname):
    """
    Read a dictionary stored as a JSON file.
    """
    with open(fname, 'r') as partF:
        partDict_str = json.load(partF)
    return dict( [(eval(key), partDict_str[key]) for key in partDict_str ] ) 
    



def readfile(fname):
    f = open(fname, 'r')
    m = 0 
    n = 0
    mat = []
    for l in f:
        l_row = []
        row = l.split()
        n = len(row)
        for elem in row:
            l_row.append( eval(elem)   )
        mat.append(l_row)
        m = m+1
    return mat, m , n
        
            
def writeMat2File(fname, A):
    m,n = A.shape
    fP = open(fname, 'w')
    for i in range(m):
        l = ""
        for j in range(n):
            if j<n-1:
                fP.write(str(A[i,j])+'\t')
            elif j==n-1:
                fP.write(str(A[i,j])+'\n')          
    fP.close()

def Loop2Zero(u , v):
    if u == v:
       return (u, 0)
    else:
       return (u, 1)
    
        
def NoneToZero(x):
    if x is None:
        return 0.0
    else:
	return x
def NoneToEmpty(x):
    if x is None:
        return set()
    else:
        return x

def safeWrite(rdd,outputfile,dvrdump=False):
    if os.path.isfile(outputfile):
       os.remove(outputfile)	
    elif os.path.isdir(outputfile):
       shutil.rmtree(outputfile)	
 
    if dvrdump:
	rdd_list = rdd.collect()
	with open(outputfile,'wb') as f:
	    count = 0
	    for item in rdd_list:
	        f.write(str(item))   
	        count = count+1
	        if count < len(rdd_list):
		    f.write("\n")  
    else:
       rdd.saveAsTextFile(outputfile)
def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket, used for GCP."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

    print('Blob {} deleted.'.format(blob_name))

 
def mergedicts(d1,d2,binop=lambda x,y:x+y):
    dout = []
    dout += [ (key,binop(d1[key],d2[key])) for key in d1 if key in d2  ]	
    dout += [ (key,d1[key]) for key in d1 if key not in d2  ]	
    dout += [ (key,d2[key]) for key in d2 if key not in d1  ]	
    return dict(dout)

def scaledict(d1,s):
    return dict([(x,s*d1[x]) for x in d1])	
  	
def swap(x):
    "Swap a pair tuple"
    return x[1],x[0] 	

def clearFile(file):
    "Delete all contents of a file"	
    with open(file,'w') as f:
   	f.write("")
   
def identityHash(i):
    "Identity function"	
    return int(i)

def cartesianProduct(graph1,graph2):
    nodes1 = graph1.flatMap(lambda (u,v):[u,v]).distinct()
    nodes2 = graph2.flatMap(lambda (u,v):[u,v]).distinct()

    G = nodes1.cartesian(nodes2)
    return G

def pretty(l):
    return " ".join(map(str,l))

def projectToPositiveSimplex(x,r):
    """ A function that projects a vector x to the face of the positive simplex. 

	Given x as input, where x is a dictionary, and a r>0,  the algorithm returns a dictionary y with the same keys as x such that:
        (1) sum( [ y[key] for key in y] ) == r,
        (2) y[key]>=0 for all key in y
        (3) y is the closest vector to x in the l2 norm that satifies (1) and (2)  

        The algorithm terminates in at most O(len(x)) steps, and is described in:
	 
             Michelot, Christian. "A finite algorithm for finding the projection of a point onto the canonical simplex of R^n." Journal of Optimization Theory and Applications 50.1 (1986): 195-200iii

        and a short summary can be found in Appendix C of:
 
	     http://www.ece.neu.edu/fac-ece/ioannidis/static/pdf/2010/CR-PRL-2009-07-0001.pdf
    """
    def projectToVA(x,A,r):
        Ac = set(x.keys()).difference(set(A))
        offset = 1.0/len(Ac)*(sum([ x[i]  for i in Ac])-r)
        y = dict([ (i,0.0) for i in A]+ [(i,x[i]-offset) for i in Ac])
	return y
   
    A = []
    y = projectToVA(x,A,r)
    B = [ i  for i in y.keys() if y[i]<0.0 ]    
    while len(B)>0:
	A = A+B
        y = projectToVA(y,A,r)
        B = [ i  for i in y.keys() if y[i]<0.0 ]    
    return y

def adaptRho(rho, primalresid, dualresid, mu=10., tau=2.):
    '''
        Adapt rho based in ADMM as follwos (see He et. el.):
            (1) if primalresid > mu*dualresid: rho_new = tau * rho
            (2) if dualresid > mu*primalresid: rho_new = rho/tau 
            (3) o.w.: rho_new = rho
    ''' 
    if primalresid > mu*dualresid:
        return tau * rho
    elif dualresid > mu*primalresid:
        return rho/tau 
    else:
        return rho 

def softThresholding(x, k):
    """Implementation of the soft thresholding operator defined as:
            (1) if  x>k: S_k(x) = x - k
            (2) if  |x|<=k: S_k(x) = 0
            (3) if  x<-k: S_k(x) = x + k
    """
    if x>k:
       out = x - k
    elif abs(x)<=k:
        out = 0.
    elif x<-k:
        out = x+k
    return out
def EuclidianPO(x, k, norm_X):
    """ Proximal operator for Euclidian norm. It returns:
            prox_k(x) = (1 - k/max(k, \|X\|_2))x
    """
    if norm_X<=k:
        out = 0.0
    else:
        out = (1 - k/norm_X) * x
    return out 
        
def vec_norm(vec, p):
    ''' Return the p-noorm of vec which is a dictionary'''
    tmp = 0.0
    for key in vec:
        tmp += abs(vec[key])**p
    return tmp**(1.0/p)
