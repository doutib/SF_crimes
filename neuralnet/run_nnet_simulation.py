
# coding: utf-8

# In[12]:

from nnet import *


# In[13]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)

# Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]


# In[14]:

## # Lauch simulation

# Parameters
prop_train    = np.array([0.50])
method1       = np.array(["Tanh","Rectifier","Sigmoid","ExpLin"])
neurons1      = np.array([12,24,39,47])
method2       = np.array(["Tanh","Rectifier","Sigmoid","ExpLin"])
neurons2      = np.array([12,24,39,47])
decay         = np.array([0.0001,0.001])
learning_rate = np.array([0.001])
n_iter        = np.array([25])
random_state  = np.array([1])

# Write csv file
results = two_layers_nnet_simulation(X,
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

print 'Write into csv...'
filename = 'data/neuralnet_results.csv'
## # Write into csv file
with open(filename, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( ["logloss",
                       "miss_err",
                       "prec",
                       "recall",
                       "f1",
                       "prop_train",
                       "method1",
                       "neurons1",
                       "method2",
                       "neurons2",
                       "decay",
                       "learning_rate",
                       "n_iter",
                       "random_state"] )
    writer.writerows(results)

