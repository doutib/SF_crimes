
# coding: utf-8

# In[1]:

from svm import *
from time import gmtime, strftime


# In[2]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
date = strftime("%Y/%m/%d_%H:%M:%S")
filename = "data/results_SVM_"+date+".csv"

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


# In[ ]:

## # Lauch simulation

# Parameters
C                       = [1.0]
kernel                  = ['poly'] 
degree                  = [2,3,4]
gamma                   = ['auto']
tol                     = [0.001]
decision_function_shape = ['ovr']

results = svm_simulation(X_train,
                         Y_train,
                         X_test,
                         Y_test,
                         C,
                         kernel,
                         degree,
                         gamma,
                         tol,
                         decision_function_shape)


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
                        "tol",
                        "decision_function_shape"] )
    writer.writerows(results)

