
# coding: utf-8

# In[8]:

from neuralnet_function import *
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count


# In[2]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
## # Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]

## Parallelization parameters
num_cpu = cpu_count()          # Number of clusters
pool = Pool(processes=num_cpu) # Nb of processes
max_t =  10*60                 # Max time to wait for each process


# In[3]:

# Import set of global parameters from parameters.csv
parameters_filename = "data/parameters.csv"

# Import parameters
df_parameters = pd.DataFrame.from_csv(parameters_filename, index_col = None)
indexes = np.arange(df_parameters.shape[0]+1) 

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


# In[ ]:

## # Store result into csv
with open('data/results.csv', 'wb') as f:
    writer = csv.writer(f)
    for index in indexes:
        try:
            print "Start job %s..." % index
            res = pool.apply_async(processInput, [index])      # runs in *only* one process
            writer.writerows(res.get(timeout=max_t))
            print "Job %s done." % index
        except TimeoutError:
            print "We lacked patience and got a multiprocessing.TimeoutError"


# In[ ]:



