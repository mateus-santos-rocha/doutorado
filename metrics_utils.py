import numpy as np

def relevance_function(x, x_0=25, a=2):
    x = np.array(x, dtype=float)

    x = np.clip(x, 0, None)

    return 1 - (x_0 ** a) / (x ** a + x_0 ** a)

def loss_function(y_true,y_pred,eps=1e-5):
    return np.abs(y_pred-y_true)/(y_true+eps)

def accuracy(y_true,y_pred,error_threshold):
    return np.where(loss_function(y_true,y_pred)<=error_threshold,1,0)