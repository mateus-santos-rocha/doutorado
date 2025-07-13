from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error
from metrics_utils import relevance_function,accuracy
import numpy as np

def R2_determinacao(y_pred,y_true):
    return r2_score(y_pred,y_true)

def RMSE(y_pred,y_true):
    return root_mean_squared_error(y_pred,y_true)

def MAE(y_pred,y_true):
    return mean_absolute_error(y_pred,y_true)

def PSC(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    
    indices_sem_chuva = y_true == 0
    n_casos_sem_chuva = np.sum(indices_sem_chuva)

    predicoes_corretas = np.sum(y_pred[indices_sem_chuva] == 0)
    psc = (predicoes_corretas / n_casos_sem_chuva) * 100 if n_casos_sem_chuva > 0 else 0.0
    return psc

def PSC_A(y_true,y_pred,erro):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    
    indices_sem_chuva = y_true == 0
    n_casos_sem_chuva = np.sum(indices_sem_chuva)

    predicoes_corretas = np.sum(y_pred[indices_sem_chuva] <= erro)
    psc_a = (predicoes_corretas / n_casos_sem_chuva) * 100 if n_casos_sem_chuva > 0 else 0.0
    return psc_a

def PCC_A(y_true, y_pred, erro):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    
    indices_com_chuva = y_true > 0
    total_chuva = np.sum(indices_com_chuva)
    
    predicoes_corretas = np.sum(
        np.abs(y_true[indices_com_chuva] - y_pred[indices_com_chuva]) <= erro
    )
    pcc_a = (predicoes_corretas / total_chuva) * 100 if total_chuva > 0 else 0.0
    
    return pcc_a

def PMC_A(y_true, y_pred, erro, chuva_minima):
    y_true,y_pred = np.array(y_true),np.array(y_pred)

    indices_chuva_pesada = y_true > chuva_minima
    total_chuva_pesada = np.sum(indices_chuva_pesada)
    
    previsoes_corretas = np.sum(
        np.abs(y_true[indices_chuva_pesada] - y_pred[indices_chuva_pesada]) <= erro
    )
    pmc_a = (previsoes_corretas / total_chuva_pesada) * 100 if total_chuva_pesada > 0 else 0.0
    
    return pmc_a

def recall(y_true,y_pred,error_threshold=0.1,min_relevancia=0.5):
    index_relevant = np.where(relevance_function(y_true)>=min_relevancia)[0]
    return sum(accuracy(y_true[index_relevant],y_pred[index_relevant],error_threshold)*relevance_function(y_true[index_relevant]))/sum(relevance_function(y_true[index_relevant]))

def precision(y_true,y_pred,error_threshold=0.1,min_relevancia=0.5):
    index_relevant = np.where(relevance_function(y_pred)>=min_relevancia)[0]
    return sum(accuracy(y_true[index_relevant],y_pred[index_relevant],error_threshold)*relevance_function(y_pred[index_relevant]))/sum(relevance_function(y_pred[index_relevant]))