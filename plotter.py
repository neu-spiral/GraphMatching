import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os


#plt.rc('font', family='serif') 
plt.rc('font', serif='Times New Roman') 
plt.rc('font', size=22) 


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def lcmm(*args):
    """Return lcm of args."""   
    return reduce(lcm, args)

colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
symbols = ['o','.','v','^','*','x','d']
linetypes= ['-','--','-.',':']

max_len = lcmm(len(colors),len(symbols),len(linetypes))
forms = [ x+y+z for (x,y,z) in zip(  
	colors * (max_len/len(colors)),
	symbols * (max_len/len(symbols)),
	linetypes * (max_len/len(linetypes))
      )]

print forms

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process output files.')
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='pickled files to be processed')
    parser.add_argument('--labels',  type=str,default=None ,help='Comma separated labels to be used in figures')
    parser.add_argument('--outputdir', default='./figs', type=str,help='output dir')
    parser.add_argument('--objective',default='$\|AP-PB\|_2$',type=str, help='objective function to be displayed on y-axis.')
    parser.add_argument('--title', default=None, type=str, help='plots title')
    parser.add_argument('--rhos', default=None, type=str, help='comma separated lambda parameter')
    parser.add_argument('--lambdas', default=None, type=str, help='lambda parameter')

    axisgroup = parser.add_mutually_exclusive_group(required=False)
    axisgroup.add_argument('--time', dest='time', action='store_true',help='Plot w.r.t. time.')
    axisgroup.add_argument('--iter', dest='time', action='store_false',help='Plot w.r.t. iteration number.')
    parser.set_defaults(time=True)    
   
    scalegroup = parser.add_mutually_exclusive_group(required=False)
    scalegroup.add_argument('--log', dest='log', action='store_true',help='Use log scale for y axis')
    scalegroup.add_argument('--lin', dest='log', action='store_false',help='Use lin scale for x axis.')
    parser.set_defaults(log=False)

    myargs = parser.parse_args()
    
    if not os.path.exists(myargs.outputdir):
	os.mkdir(myargs.outputdir)


    if myargs.labels:
	labels = dict(zip(myargs.filenames,myargs.labels.split(',')))	
    else:
	labels = dict(zip(myargs.filenames,myargs.filenames))	

    if myargs.lambdas:
        lambdas = dict(zip(myargs.filenames, eval(myargs.lambdas)))
    else:
        lambdas = dict(zip(myargs.filenames, len(myargs.filenames)*[0.0]))
        
    if myargs.rhos:
        rhos = dict(zip(myargs.filenames, eval(myargs.rhos)))
    else:
        rhos = dict(zip(myargs.filenames, len(myargs.filenames)*[1.0]))


    
    data ={}
    data_labels = { 'TIME':'t (min)','ITERATION':'iteration','OBJNOLIN':myargs.objective,'OBJ':'objective','PRES':'PRES','DRES':'DRES'}
    data['TIME']={}
    data['ITERATION']={}
    data['OBJ'] ={}
    data['OBJNOLIN'] = {}
    data['PRES'] ={}
    data['DRES'] ={}

    minITERS = 10**8
    print rhos
    for filename in myargs.filenames:
        #Added to process based on iteration time rather than total time
        cuuernt_time = 0.0
        time_steps = []
        steps = []
	print 'Processing...', filename

	with open(filename,'rb') as f:
	    arg,trace = pickle.load(f)

	print 'Read trace with parameters',arg,'total iterations:',len(trace)
        if len(trace)<minITERS:
            minITERS = len(trace)
	iterations = sorted(trace.keys())
        for iteration in iterations:
            cuuernt_time += trace[iteration]['IT_TIME']/60.0
            time_steps.append(cuuernt_time)
            steps.append(iteration)
            

        data['TIME'][filename] = time_steps
        data['ITERATION'][filename] = steps
       # data['TIME'][filename] = [trace[iteration]['TIME']/60.0 for iteration in iterations]
        data['OBJ'][filename] = [(trace[iteration]['OLDOBJ'] - trace[iteration]['OLDNOLIN'])*lambdas[filename] + trace[iteration]['OLDNOLIN']  for iteration in iterations]
        data['OBJNOLIN'][filename] = [trace[iteration]['OLDNOLIN'] for iteration in iterations]
        data['PRES'][filename] = [ (trace[iteration]['PRES']+ trace[iteration]['QRES']+trace[iteration]['TRES'])/(rhos[filename]*3.) for iteration in iterations]
        data['DRES'][filename] = [ trace[iteration]['DRES']/(3.0*rhos[filename]) for iteration in iterations]

   
    
    
    for data_label in ['OBJ','OBJNOLIN','PRES','DRES']:
    	fig =plt.figure()
        size = fig.get_size_inches()
    	ax = fig.add_subplot(1,1,1)
     #   if data_label == 'OBJ':
        if myargs.log:
            ax.set_yscale('log')
        if myargs.title:
            ax.set_title(myargs.title)
        lines = []
        for (form,filename) in zip(forms[:len(myargs.filenames)],myargs.filenames):
           if myargs.time:
    	       line, =ax.plot(data['TIME'][filename],data[data_label][filename],form,label=labels[filename],markevery=10, linewidth=2)
           else:
               line, =ax.plot(data['ITERATION'][filename],data[data_label][filename],form,label=labels[filename],markevery=10, linewidth=2)
	   lines = lines + [line]
    	ax.set_ylabel(data_labels[data_label])
        if myargs.time:
    	    ax.set_xlabel(data_labels['TIME'])
            #ax.set_xlim([0, 62])
        else:
            ax.set_xlabel(data_labels['ITERATION'])
            ax.set_xlim([0, 52])        
        
    	names= [ labels[filename] for filename in myargs.filenames ]
        if data_label == 'OBJNOLIN':
    	    LGD=plt.legend(lines,names, ncol=2,loc='lower right', bbox_to_anchor=(1,0),fontsize='x-small')
    	#ax.set_title(graph+' '+cache )
	plt.tight_layout()
    	fig.savefig(myargs.outputdir+'/fig_'+data_label+'.pdf', bbox_inches='tight')
    	plt.close(fig)

