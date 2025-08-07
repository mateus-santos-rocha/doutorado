
from sklearn.utils import resample
import pandas as pd
import numpy as np
import pickle
from comparison_utils import compute_comparison_df


def get_abt_estacoes_vizinhas(modelling_conn,abt_estacoes_vizinhas_table_name = 'abt_estacoes_5_vizinhas',create_dt_columns=True):
    abt_estacoes_vizinhas = modelling_conn.execute(f"""SELECT * FROM {abt_estacoes_vizinhas_table_name}""").fetchdf()
    if create_dt_columns:
        abt_estacoes_vizinhas['dt_medicao_mes'] = abt_estacoes_vizinhas['dt_medicao'].dt.month
        abt_estacoes_vizinhas['dt_medicao_ano'] = abt_estacoes_vizinhas['dt_medicao'].dt.year
    return abt_estacoes_vizinhas

def split_com_sem_vizinha(abt_estacoes_vizinhas,threshold_prioridade):
    abt_com_vizinha = abt_estacoes_vizinhas \
        .loc[abt_estacoes_vizinhas['vl_prioridade_vizinha_1'].fillna(0) >= threshold_prioridade] \
        .sort_values(by=['id_estacao','dt_medicao'])

    abt_sem_vizinha = abt_estacoes_vizinhas.copy()

    return abt_com_vizinha,abt_sem_vizinha

def particao_por_estacao(df,percent_datetime_partitioning_split):
    df_treinamento = df \
        .groupby('id_estacao', as_index=False).apply(lambda g: g.iloc[:max(1, int(len(g) * percent_datetime_partitioning_split))]) \
        .reset_index(drop=True)
    
    df_validacao = df \
        .groupby('id_estacao', as_index=False).apply(lambda g: g.iloc[max(1, int(len(g) * percent_datetime_partitioning_split)):]) \
        .reset_index(drop=True)

    return df_treinamento,df_validacao

def feature_selection(df,feature_list,target):
    X = df[feature_list]
    y = df[target]
    return X,y

import numpy as np
import pandas as pd
from sklearn.utils import resample

def undersample_zeros(X, y, zero_ratio=1.0, random_state=42):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    zero_mask = y == 0
    non_zero_mask = y > 0

    X_zero = X[zero_mask]
    y_zero = y[zero_mask]

    X_non_zero = X[non_zero_mask]
    y_non_zero = y[non_zero_mask]

    n_non_zero = len(y_non_zero)
    n_zero_desired = int(zero_ratio * n_non_zero)
    n_zero_available = len(y_zero)

    n_zero_final = min(n_zero_desired, n_zero_available)

    X_zero_downsampled, y_zero_downsampled = resample(
        X_zero, y_zero,
        replace=False,
        n_samples=n_zero_final,
        random_state=random_state
    )

    X_bal = pd.concat([X_zero_downsampled, X_non_zero], axis=0)
    y_bal = pd.concat([y_zero_downsampled, y_non_zero], axis=0)

    X_bal, y_bal = resample(X_bal, y_bal, random_state=random_state)

    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True)

def drop_estacoes_vizinhas(abt_estacoes_vizinhas):
    dropped_df = abt_estacoes_vizinhas.drop(['vl_precipitacao_vizinha','vl_correlacao_vizinha','pct_intersecao_precipitacao_vizinha','vl_distancia_km_vizinha','estacao_vizinha_escolhida','vl_prioridade_vizinha'],axis=1)
    return dropped_df

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



def import_model_and_comparison(model_path,comparison_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(comparison_path,'rb') as f:
        comparison = pickle.load(f)
    return model,comparison

def save_model_and_comparison(model,comparison,model_path,comparison_path):
    with open(model_path,'wb') as f:
        pickle.dump(model,f)
    with open(comparison_path,'wb') as f:
        pickle.dump(comparison,f)
    return 




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