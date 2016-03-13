
# coding: utf-8

# In[6]:

from rf import *
from time import gmtime, strftime


# In[7]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
date = strftime("%Y/%m/%d_%H:%M:%S")
filename = "data/results_RF_"+date+".csv"

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


# In[8]:

## # Lauch simulation

# Parameters
n_estimators  = [10,20]
criterion     = ["gini"]
max_features  = ["auto"]
max_depth     = [2,5]

results = rf_simulation(X_train,
                        Y_train,
                        X_test,
                        Y_test,
                        n_estimators,
                        criterion,   
                        max_features,
                        max_depth)


# In[ ]:

## # Write into csv file
with open(filename, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( [  "logloss",
                        "miss_err",
                        "prec",
                        "recall",
                        "f1",
                        "C",
                        "kernel",
                        "degree",
                        "gamma",
                        "probability",
                        "tol",
                        "decision_function_shape"] )
    writer.writerows(results)

