
# coding: utf-8

# In[33]:

from neuralnet_function import *
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count


# In[34]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
## # Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]


# In[35]:

# Import set of global parameters from parameters.csv
parameters_filename = "data/parameters.csv"

# Import parameters
df_parameters = pd.DataFrame.from_csv(parameters_filename, index_col = None)
indexes = np.arange(df_parameters.shape[0]+1) 


# In[11]:

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


# In[19]:

## Parallelization parameters
num_cpu = cpu_count()          # Number of clusters
max_t =  10*60                 # Max time to wait for each process


# In[44]:

print 'Starting clusters...'
pool = Pool(processes=num_cpu)                 # Start clusters
print 'Starting process...'
results = pool.map(processInput, indexes)      


# In[ ]:

print 'Starting writing...'
## # Store result into csv
with open('data/results.csv', 'wb') as f:
    writer = csv.writer(f)
    for res in results:
        writer.writerows(res)

