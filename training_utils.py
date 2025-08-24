
import numpy as np
from modelling_utils.model_management import save_model_and_comparison
from comparison_utils import compute_comparison_df
import numpy as np
import warnings
from modelling_utils.sampling import undersample_zeros
from modelling_utils.preprocessing import split_com_sem_vizinha,particao_por_estacao
warnings.filterwarnings('ignore')


def generate_X_y_train_test(abt_estacoes_vizinhas,usar_n_estacoes_vizinhas=0,zero_undersampling_ratio = None,smote_oversampling = False,use_bi_model = False,percent_datetime_partitioning_split=0.7):
    abt = abt_estacoes_vizinhas[[c for c in abt_estacoes_vizinhas.columns if 'vizinha' not in c]].copy()
    if usar_n_estacoes_vizinhas > 0:
        vizinhas_columns_prefix = ['vl_correlacao_estacao_vizinha_{i_vizinha}','pct_intersecao_precipitacao_vizinha_{i_vizinha}','vl_distancia_km_vizinha_{i_vizinha}','vl_prioridade_vizinha_{i_vizinha}','vl_precipitacao_vizinha_{i_vizinha}']
        for i in range(1, usar_n_estacoes_vizinhas + 1):
            vizinha_columns = [col.format(i_vizinha=i) for col in vizinhas_columns_prefix]
            for col in vizinha_columns:
                abt.loc[:,col] = abt_estacoes_vizinhas[col]

    training_abt,validation_abt = particao_por_estacao(abt,percent_datetime_partitioning_split)
        
    X_train,y_train = training_abt.drop('vl_precipitacao',axis=1),training_abt['vl_precipitacao']
    X_test,y_test = validation_abt.drop('vl_precipitacao',axis=1),validation_abt['vl_precipitacao']

    if not zero_undersampling_ratio is None:
        X_train, y_train = undersample_zeros(X_train, y_train, zero_ratio=zero_undersampling_ratio)

    if smote_oversampling:
        pass

    if use_bi_model:
        pass
    return X_train,X_test,y_train,y_test


def train_model(abt_estacoes_vizinhas,Model,model_number,usar_n_estacoes_vizinhas,zero_undersampling_ratio=None,smote_oversampling=False,use_bi_model=False,threshold_prioridade=0.5,percent_datetime_partitioning_split=0.7,truncate_to_non_negative_target=True):
    if not use_bi_model:
        X_train,X_test,y_train,y_test = generate_X_y_train_test(abt_estacoes_vizinhas,usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,zero_undersampling_ratio=zero_undersampling_ratio,smote_oversampling=smote_oversampling,use_bi_model=use_bi_model,percent_datetime_partitioning_split=percent_datetime_partitioning_split)
        model = Model()
        model.fit(X_train.drop(['id_estacao','dt_medicao'],axis=1),y_train)

        y_pred = model.predict(X_test.drop(['id_estacao','dt_medicao'],axis=1))
        if truncate_to_non_negative_target: 
            y_pred = np.clip(y_pred, a_min=0, a_max=None)
        comparison = compute_comparison_df(X_test,y_test,y_pred)

    elif use_bi_model:
        abt,X_train,X_test,y_train,y_test,model,y_pred,comparison = {},{},{},{},{},{},{},{}
        abt['com_vizinha'],abt['sem_vizinha'] = split_com_sem_vizinha(abt_estacoes_vizinhas,threshold_prioridade)
        for tipo in ['com_vizinha','sem_vizinha']:
            X_train[tipo],X_test[tipo],y_train[tipo],y_test[tipo] = generate_X_y_train_test(abt[tipo],usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,zero_undersampling_ratio=zero_undersampling_ratio,smote_oversampling=smote_oversampling,use_bi_model=use_bi_model,percent_datetime_partitioning_split=percent_datetime_partitioning_split)
        
        model = {}
        model['com_vizinha'],model['sem_vizinha'] = Model(), Model()
        for tipo in ['com_vizinha','sem_vizinha']:
            model[tipo].fit(X_train[tipo].drop(['id_estacao','dt_medicao'],axis=1),y_train[tipo])
        for tipo in ['com_vizinha','sem_vizinha']:
            y_pred[tipo] = model[tipo].predict(X_test[tipo].drop(['id_estacao','dt_medicao'],axis=1))
            if truncate_to_non_negative_target:
                y_pred[tipo] = np.clip(y_pred[tipo], a_min=0, a_max=None)
            comparison[tipo] = compute_comparison_df(X_test[tipo],y_test[tipo],y_pred[tipo])

    model_path,comparison_path = f'models/model_{model_number}.pkl',f'comparisons/comparison_{model_number}.pkl'
    save_model_and_comparison(model,comparison,model_path,comparison_path)
    
    return model,comparison

