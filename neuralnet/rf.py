
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count
from datetime import timedelta

from  sklearn.ensemble import RandomForestClassifier

import sys
import csv
import itertools
import time


# In[13]:

def rf(X_train,
       Y_train,
       X_test,
       Y_test,
       n_estimators=10,
       criterion="gini",
       max_features="auto",
       max_depth=-1,
       n_jobs=1):
    """
    Parameters
    ----------
    X_train       : pandas data frame
        data frame of features for the training set
    Y_train       : pandas data frame
        data frame of labels for the training set
    X_test        : pandas data frame
        data frame of features for the test set
    Y_test        : pandas data frame
        data frame of labels for the test set
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default=”gini”)
        The function to measure the quality of a split. 
        Supported criteria are “gini” for the Gini impurity and “entropy” 
        for the information gain. 
    max_features : int, float, string or None, optional (default=”auto”)
        The number of features to consider when looking for the best split:
        If int, then consider max_features features at each split.
        If float, then max_features is a percentage and int(max_features * n_features) 
        features are considered at each split.
        If “auto”, then max_features=sqrt(n_features).
        If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
        If “log2”, then max_features=log2(n_features).
        If None, then max_features=n_features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. 
        If None, then nodes are expanded until all leaves are pure or 
        until all leaves contain less than min_samples_split samples. 
        Ignored if max_leaf_nodes is not None. 
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both fit and predict. 
        If -1, then the number of jobs is set to the number of cores.
        
    Result:
    -------
    numpy array
        logloss    : averaged logarithmic loss
        miss_err   : missclassification error rate
        prec       : precision
        recall     : recall
        f1         : f1 score
        parameters : previous parameters in the order previously specified
    """
    if max_depth==-1:
        max_depth = None
        
    labels = np.unique(Y_train)
    
    ## # Scale Data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)
    
    ## # Run rf
    # Define classifier
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_features=max_features,
                                max_depth=max_depth,
                                n_jobs=n_jobs)

    # Fit
    rf.fit(X_train, Y_train[:,0])
    # Predict
    Y_hat = rf.predict(X_test)
    Y_probs = rf.predict_proba(X_test)
    
    ## # Misclassification error rate
    miss_err = 1-accuracy_score(Y_test, Y_hat)
    ## # Log Loss
    eps = 10^(-15)
    logloss = log_loss(Y_test, Y_probs, eps = eps)
    
    ## # Precision
    prec = precision_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro')
    ## # Recal
    recall = recall_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro') 
    ## # F1
    f1 = f1_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro')
    
    # Summarized results
    result = np.array([logloss,
                       miss_err,
                       prec,
                       recall,
                       f1,
                       n_estimators,
                       criterion,
                       max_features,
                       max_depth,
                       n_jobs])
    return result


# In[14]:

def processInput((X_train,Y_train,X_test,Y_test,parameters,index)): 
    # Define parameters names
    n_estimators,criterion,max_features,max_depth=parameters[index]
    
    # Run rf
    result = rf(X_train,
                Y_train,
                X_test,
                Y_test,
                n_estimators,
                criterion,
                max_features,
                max_depth,
                n_jobs=1)
    return result


def rf_simulation(X_train,
                  Y_train,
                  X_test,
                  Y_test,
                  n_estimators,
                  criterion,
                  max_features,
                  max_depth):
    """
    Parameters:
    -----------
    Same parameters as rf, in a list format. n_jobs has to be one here.
    
    Result:
    ------
    List of Lists of results from rf.
        One list corresponds to one set of parameters
    """
    
    n_jobs = 1
    
    print('Lauching Simulation...')
    start = time.time()
    
    # Combinations
    param = np.array([n_estimators,
                      criterion,
                      max_features,
                      max_depth])
    
    parameters = list(itertools.product(*param))
    
    indexes = range(len(parameters))
    print "Number of sets of parameters: %s.\n" %len(parameters)
    
    print 'Parameters:\n-----------'
    print np.array(parameters)
    

    # Number of clusters
    num_cpu = cpu_count()          
    print "\nNumber of identified CPUs: %s.\n" %num_cpu
    num_clusters = min(num_cpu,len(parameters))
    
    ## # Parallelization
    tuples_indexes = tuple([(X_train,Y_train,X_test,Y_test,parameters,index) for index in indexes])

    # Start clusters
    print 'Start %s clusters.\n' % num_clusters
    print 'Running...'
    pool = Pool(processes=num_clusters)
    results = pool.map(processInput, tuples_indexes) 
    pool.terminate()
    
    # Results
    print 'Results:\n--------'
    print results
    end = time.time()
    elapsed = end - start
    print 'End of Simulation.\nElapsed time: %s' %str(timedelta(seconds=elapsed))
    print 'Write into csv...'
    
    return results


# In[ ]:

def rf_predict(X_train,
               Y_train,
               X_test,
               n_estimators=10,
               criterion="gini",
               max_features="auto",
               max_depth=-1,
               n_jobs=1):
    """
    Parameters
    ----------
    X_train       : pandas data frame
        data frame of features for the training set
    Y_train       : pandas data frame
        data frame of labels for the training set
    X_test        : pandas data frame
        data frame of features for the test set
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default=”gini”)
        The function to measure the quality of a split. 
        Supported criteria are “gini” for the Gini impurity and “entropy” 
        for the information gain. 
    max_features : int, float, string or None, optional (default=”auto”)
        The number of features to consider when looking for the best split:
        If int, then consider max_features features at each split.
        If float, then max_features is a percentage and int(max_features * n_features) 
        features are considered at each split.
        If “auto”, then max_features=sqrt(n_features).
        If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
        If “log2”, then max_features=log2(n_features).
        If None, then max_features=n_features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. 
        If None, then nodes are expanded until all leaves are pure or 
        until all leaves contain less than min_samples_split samples. 
        Ignored if max_leaf_nodes is not None. 
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both fit and predict. 
        If -1, then the number of jobs is set to the number of cores.
        
    Result:
    -------
    tuple of numpy arrays
        (predicted classes, predicted probabilities)
    """

    labels = np.unique(Y_train)
    
    ## # Scale Data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    ## # Split data set into train/test
        
    ## # Run rf
    # Define classifier
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_features=max_features,
                                max_depth=max_depth,
                                n_jobs=n_jobs)
    # Fit
    rf.fit(X_train, Y_train)
    # Predict
    Y_hat = rf.predict(X_test)
    Y_probs = rf.predict_proba(X_test)
    
    # Summarized results
    result = (Y_hat,Y_probs)
    return result

