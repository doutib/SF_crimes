
# coding: utf-8

# In[25]:

from neuralnet_function import *
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count


# In[47]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
## # Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]


# In[44]:

# Import set of global parameters from parameters.csv
parameters_filename = "data/neuralnet_parameters.csv"

# Import parameters
df_parameters = pd.DataFrame.from_csv(parameters_filename, index_col = None)
indexes = np.arange(df_parameters.shape[0]) 
print df_parameters


# In[35]:

def processInput(index): 
    # Load parameters
    parameters = np.array(df_parameters.iloc[[index]])[0]
    # Define parameters
    prop_train,method1,neurons1,method2,neurons2,decay,learning_rate,n_iter,random_state=parameters
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


# In[48]:

## Parallelization parameters
num_cpu = cpu_count()          # Number of clusters
print "Number of identified clusters: %s." %num_cpu


# In[49]:

print 'Start clusters...'
pool = Pool(processes=num_cpu)                 # Start clusters  
results = pool.map(processInput, indexes) 
pool.terminate()


# In[59]:

print 'Results:\n--------'
print results


# In[62]:

# Open file
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


# In[ ]:



