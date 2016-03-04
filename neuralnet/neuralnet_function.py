
# coding: utf-8

# In[34]:

import numpy as np
import pandas as pd
from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import sys
import csv


# In[35]:

def two_layers_nnet(X,
                    Y,
                    prop_train=0.5,
                    method1="Tanh",
                    neurons1=5,
                    method2="",
                    neurons2=0,
                    decay=0.0001,
                    learning_rate=0.001,
                    n_iter=25,
                    random_state=1):
    """
    Parameters
    ----------
    X             : pandas data frame
        data frame of features
    Y             : pandas data frame
        data frame of labels
    prop_train    : float
        proportion of the traning set
    method1       : str
        method used for the first layer
    neurons1      : int
        number of neurons of the first layer
    method2       : None
        method used for the first layer
    neurons2      : int
        number of neurons of the first layer
    decay         : float
        weight decay
    learning_rate : float
        learning rate
    n_iter        : int
        number of iterations
    random_state  : int
        seed for weight initialization
        
    Returns
    -------
    numpy array
        logloss    : averaged logarithmic loss
        miss_err   : missclassification error rate
        prec       : precision
        recall     : recall
        f1         : f1 score
        parameters : previous parameters in the order previously specified
    """

    labels = np.unique(Y)
    
    ## # Scale Data
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

    ## # Split data set into train/test
    np.random.seed(seed=1)
    msk = np.random.rand(len(X)) < prop_train
    X_train = np.array(X[msk])
    Y_train = np.array(Y[msk])
    X_test =  np.array(X[~msk])
    Y_test =  np.array(Y[~msk])
    
    # Layers
    if neurons2 == 0 :
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer("Softmax")]
    else:
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer(method2, weight_decay = decay, units = neurons2),
                Layer("Softmax")]
        
    ## # Run nnet
    # Define classifier
    nn = Classifier(layers,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    n_iter=n_iter)
    # Fit
    nn.fit(X_train, Y_train)
    # Predict
    Y_hat = nn.predict(X_test)
    Y_probs = nn.predict_proba(X_test)
    
    ## # Misclassification error rate
    miss_err = float(sum(Y_test[:,0]!=Y_hat[:,0]))/float(len(Y_test[:,0]))
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
                       method1,
                       neurons1,
                       method2,
                       neurons2,
                       decay,
                       learning_rate,
                       n_iter,
                       random_state,
                       prop_train])
    return result

