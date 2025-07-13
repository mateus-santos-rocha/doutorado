import numpy as np

def relevance_function(x,x_0=25,a=2):
    return 1-x_0**a/(x**a+x_0**a)

def loss_function(y_true,y_pred):
    return np.abs(1-y_pred/y_true)

def accuracy(y_true,y_pred,error_threshold):
    return np.where(loss_function(y_true,y_pred)<=error_threshold,1,0)