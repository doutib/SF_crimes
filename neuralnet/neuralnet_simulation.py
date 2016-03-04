
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
parameters_filename = "data/neuralnet_parameters.csv"

# Import parameters
df_parameters = pd.DataFrame.from_csv(parameters_filename, index_col = None)
indexes = np.arange(df_parameters.shape[0]+1) 

# Open file
f = open('data/neuralnet_results.csv', 'wb')
writer = csv.writer(f)


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
    writer.writerows(result)


# In[19]:

## Parallelization parameters
num_cpu = cpu_count()          # Number of clusters
max_t =  10*60                 # Max time to wait for each process


# In[44]:

print 'Starting clusters...'
pool = Pool(processes=num_cpu)                 # Start clusters
print 'Starting process...'
pool.map(processInput, indexes)      

