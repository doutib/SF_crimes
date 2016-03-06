
# coding: utf-8

# In[7]:

import numpy as np
import itertools
import csv


# In[8]:

## # Parameters
#prop_train    = np.array([0.50])
#method1       = np.array(["Tanh"])  #,"Rectifier","Sigmoid","ExpLin"])
#neurons1      = np.array([12,24,39,47])
#method2       = np.array(["Tanh"])  #,"Rectifier","Sigmoid","ExpLin"])
#neurons2      = np.array([0,12,24,39,47])
#decay         = np.array([0.0001])
#learning_rate = np.array([0.001])
#n_iter        = np.array([25])
#random_state  = np.array([1,2,3,4,5,6,7,8,9,10])

prop_train    = np.array([0.50])
method1       = np.array(["Tanh"])
neurons1      = np.array([1])
method2       = np.array(["None"])
neurons2      = np.array([0])
decay         = np.array([0.0001])
learning_rate = np.array([0.001])
n_iter        = np.array([25])
random_state  = np.array([1,2])

parameters = np.array([prop_train,
                       method1,
                       neurons1,
                       method2,
                       neurons2,
                       decay,
                       learning_rate,
                       n_iter,
                       random_state])


# In[9]:

## # Export csv of parameters combinations
with open('data/neuralnet_parameters.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(["prop_train",
                       "method1",
                       "neurons1",
                       "method2",
                       "neurons2",
                       "decay",
                       "learning_rate",
                       "n_iter",
                       "random_state"])
    for element in itertools.product(*parameters):
        spamwriter.writerow(element)


# In[ ]:



