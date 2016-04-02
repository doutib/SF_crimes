
# coding: utf-8

# In[2]:

from nnet import *
from time import gmtime, strftime


# In[3]:

## # Collect data
df = pd.DataFrame.from_csv("data/data_train.csv", index_col = None)
df_test = pd.DataFrame.from_csv("data/data_test.csv", index_col = None)
date = strftime("%Y-%m-%d_%H:%M:%S")
filename = "data/predictions_NNET_"+date+".csv"

# Separate labels from data
X = df.drop(['Category'], axis = 1)
Y = df[['Category']]

## # Load train data
np.random.seed(seed=1)
X_train = np.array(X,dtype='float64')
Y_train = np.array(Y)
X_test = np.array(df_test)


# In[ ]:

## # Run algorithm

# Parameters
method1       = "Tanh"
neurons1      = 1#39
method2       = "Tanh"
neurons2      = 0#53
decay         = 0.001
learning_rate = 0.001
n_iter        = 1#500
random_state  = 1

# Write csv file
y_hat, y_probs = two_layers_nnet_predict(X_train,
                                         Y_train,
                                         X_test,
                                         method1,
                                         neurons1,
                                         method2,
                                         neurons2,
                                         decay,
                                         learning_rate,
                                         n_iter,
                                         random_state)
results = y_probs


# In[7]:

with open(filename, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( ["Id",
                      "ARSON",
                      "ASSAULT",
                      "BAD CHECKS",
                      "BRIBERY",
                      "BURGLARY",
                      "DISORDERLY CONDUCT",
                      "DRIVING UNDER THE INFLUENCE",
                      "DRUG/NARCOTIC",
                      "DRUNKENNESS",
                      "EMBEZZLEMENT",
                      "EXTORTION",
                      "FAMILY OFFENSES",
                      "FORGERY/COUNTERFEITING",
                      "FRAUD,GAMBLING",
                      "KIDNAPPING",
                      "LARCENY/THEFT",
                      "LIQUOR LAWS",
                      "LOITERING",
                      "MISSING PERSON",
                      "NON-CRIMINAL",
                      "OTHER OFFENSES",
                      "PORNOGRAPHY/OBSCENE MAT",
                      "PROSTITUTION",
                      "RECOVERED VEHICLE",
                      "ROBBERY",
                      "RUNAWAY",
                      "SECONDARY CODES",
                      "SEX OFFENSES FORCIBLE",
                      "SEX OFFENSES NON FORCIBLE",
                      "STOLEN PROPERTY",
                      "SUICIDE",
                      "SUSPICIOUS OCC",
                      "TREA",
                      "TRESPASS",
                      "VANDALISM",
                      "VEHICLE THEFT",
                      "WARRANTS",
                      "WEAPON LAWS"] )
    # Index
    n = results.shape[0]
    index = np.ndarray((n,1), buffer=np.arange(n+1),dtype=int)
    results = np.hstack((index,results))
    # Results
    writer.writerows(results)


# In[ ]:

labels = ["ARSON",
          "ASSAULT",
          "BAD CHECKS",
          "BRIBERY",
          "BURGLARY",
          "DISORDERLY CONDUCT",
          "DRIVING UNDER THE INFLUENCE",
          "DRUG/NARCOTIC",
          "DRUNKENNESS",
          "EMBEZZLEMENT",
          "EXTORTION",
          "FAMILY OFFENSES",
          "FORGERY/COUNTERFEITING",
          "FRAUD,GAMBLING",
          "KIDNAPPING",
          "LARCENY/THEFT",
          "LIQUOR LAWS",
          "LOITERING",
          "MISSING PERSON",
          "NON-CRIMINAL",
          "OTHER OFFENSES",
          "PORNOGRAPHY/OBSCENE MAT",
          "PROSTITUTION",
          "RECOVERED VEHICLE",
          "ROBBERY",
          "RUNAWAY",
          "SECONDARY CODES",
          "SEX OFFENSES FORCIBLE",
          "SEX OFFENSES NON FORCIBLE",
          "STOLEN PROPERTY",
          "SUICIDE",
          "SUSPICIOUS OCC",
          "TREA",
          "TRESPASS",
          "VANDALISM",
          "VEHICLE THEFT",
          "WARRANTS",
          "WEAPON LAWS"]

results = pd.DataFrame(results)
results.to_csv(path_or_buf=filename, 
               sep=',',
               header=labels, 
               index=True, 
               index_label="Id")


# In[ ]:




# In[ ]:



