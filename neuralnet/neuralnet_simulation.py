
# coding: utf-8

# In[38]:

from neuralnet_function import *
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count
import numpy as np
import itertools
import csv


# In[39]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)

# Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]


# In[40]:

## # Define set of global parameters

# Parameters
prop_train    = np.array([0.50])
method1       = np.array(["Tanh"])#,"Rectifier","Sigmoid","ExpLin"])
neurons1      = np.array([1,2])#12,24,39,47])
method2       = np.array(["Tanh"])#,"Rectifier","Sigmoid","ExpLin"])
neurons2      = np.array([0,])#12,24,39,47])
decay         = np.array([0.0001])
learning_rate = np.array([0.001])
n_iter        = np.array([25])
random_state  = np.array([1])

# Combinations
param = np.array([prop_train,
                       method1,
                       neurons1,
                       method2,
                       neurons2,
                       decay,
                       learning_rate,
                       n_iter,
                       random_state])

parameters = list(itertools.product(*param))

indexes = range(len(parameters))
print "Number of sets of parameters: %s.\n" %len(parameters)

print 'Parameters:\n-----------'
print np.array(parameters)


# In[ ]:

def processInput(index): 
    # Define parameters names
    prop_train,method1,neurons1,method2,neurons2,decay,learning_rate,n_iter,random_state=parameters[index]
    
    # Run nnet
    result = two_layers_nnet(X,
                             Y,
                             prop_train,
                             method1,
                             neurons1,
                             method2,
                             neurons2,
                             decay,
                             learning_rate,
                             n_iter,
                             random_state)
    return result

# Number of clusters
num_cpu = cpu_count()          
print "Number of identified CPUs: %s.\n" %num_cpu
num_clusters = min(num_cpu,len(parameters))


# In[ ]:

## # Parallelization

# Start clusters
print 'Start %s clusters.\n' % num_clusters
pool = Pool(processes=num_clusters)                   
results = pool.map(processInput, indexes) 
pool.terminate()

# Results
print 'Results:\n--------'
print results


# In[62]:

print 'Write into csv...'
## # Write into csv file
with open('data/neuralnet_results.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( ["logloss",
                       "miss_err",
                       "prec",
                       "recall",
                       "f1",
                       "method1",
                       "neurons1",
                       "method2",
                       "neurons2",
                       "decay",
                       "learning_rate",
                       "n_iter",
                       "random_state",
                       "prop_train"] )
    writer.writerows(results)


# In[27]:

print 'Done.'


# In[ ]:



