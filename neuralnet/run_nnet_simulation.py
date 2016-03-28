
# coding: utf-8

# In[3]:

from nnet import *
from time import gmtime, strftime


# In[5]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
date = strftime("%Y-%m-%d_%H:%M:%S")
filename = "data/simulations_NNET_"+date+".csv"

# Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]

## # Split data set into train/test
prop_train=0.5
np.random.seed(seed=1)
msk = np.random.rand(len(X)) < prop_train
X_train = np.array(X[msk],dtype='float64')
Y_train = np.array(Y[msk])
X_test =  np.array(X[~msk],dtype='float64')
Y_test =  np.array(Y[~msk])


# In[6]:

## # Lauch simulation

# Parameters
method1       = np.array(["Tanh"])#,"Rectifier","Sigmoid","ExpLin"])
neurons1      = np.array([24,39])#n12,24,39,47])
method2       = np.array(["Tanh"])#,"Rectifier","Sigmoid","ExpLin"])
neurons2      = np.array([39,47])#,12,24,39,47])
decay         = np.array([0.0001,0.001])
learning_rate = np.array([0.001,0.0001])
n_iter        = np.array([25,50])
random_state  = np.array([1])

# Write csv file
results = two_layers_nnet_simulation(X_train,
                                     Y_train,
                                     X_test,
                                     Y_test,
                                     method1,
                                     neurons1,
                                     method2,
                                     neurons2,
                                     decay,
                                     learning_rate,
                                     n_iter,
                                     random_state)


# In[ ]:

## # Write into csv file
with open(filename, 'wb') as f:
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
                       "random_state"] )
    writer.writerows(results)

