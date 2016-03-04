
# coding: utf-8

# In[69]:

from neuralnet_function import *


# In[70]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)

## # Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]


# In[71]:

## # Define Global Parameters

prop_train    = 0.10
method1       = "Tanh"
neurons1      = 1
method2       = "None"
neurons2      = 0
decay         = 0.0001
learning_rate = 0.001
n_iter        = 2
random_state  = 1


# In[72]:

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


# In[ ]:

print result

