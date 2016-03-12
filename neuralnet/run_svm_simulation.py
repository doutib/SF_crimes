
# coding: utf-8

# In[31]:

from svm import *


# In[ ]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)

# Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]
X


# In[ ]:

## # Lauch simulation

# Parameters
prop_train              = [0.5]
C                       = [1.0]
kernel                  = ['rbf'] 
degree                  = [2]
gamma                   = ['auto']
probability             = [True]
tol                     = [0.001]
decision_function_shape = ['ovr']

results = svm_simulation(X,
                         Y,
                         prop_train,
                         C,
                         kernel,
                         degree,
                         gamma,
                         probability,
                         tol,
                         decision_function_shape)


# In[ ]:

print 'Write into csv...'
filename = 'data/svm_results.csv'
## # Write into csv file
with open(filename, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( ["logloss",
                       "miss_err",
                       "prec",
                       "recall",
                       "f1",
                       "prop_train",
                       "C",
                       "kernel",
                       "degree",
                       "gamma",
                       "probability",
                       "tol",
                       "decision_function_shape"])
    writer.writerows(results)

