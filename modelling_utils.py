
from sklearn.utils import resample
import pandas as pd
import numpy as np


def get_abt_estacoes_vizinhas(modelling_conn,abt_estacoes_vizinhas_table_name = 'abt_estacoes_vizinhas',create_dt_columns=True):
    abt_estacoes_vizinhas = modelling_conn.execute(f"""SELECT * FROM {abt_estacoes_vizinhas_table_name}""").fetchdf()
    if create_dt_columns:
        abt_estacoes_vizinhas['dt_medicao_mes'] = abt_estacoes_vizinhas['dt_medicao'].dt.month
        abt_estacoes_vizinhas['dt_medicao_ano'] = abt_estacoes_vizinhas['dt_medicao'].dt.year
    return abt_estacoes_vizinhas

def split_com_sem_vizinha(abt_estacoes_vizinhas,threshold_prioridade):
    abt_com_vizinha = abt_estacoes_vizinhas \
        .loc[abt_estacoes_vizinhas['vl_prioridade_vizinha'].fillna(0) >= threshold_prioridade] \
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

def undersample_zeros(X, y, random_state=42):
    """
    Realiza undersampling das amostras com target igual a zero para balancear com amostras onde y > 0.

    Parâmetros:
        X (pd.DataFrame ou np.ndarray): Features.
        y (pd.Series ou np.ndarray): Target de regressão (espera-se muitos zeros).
        random_state (int): Semente para reprodutibilidade.

    Retorna:
        X_bal, y_bal: X e y balanceados.
    """
    # Converter para DataFrame/Séries se necessário
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Separar amostras zero e não-zero
    zero_mask = y == 0
    non_zero_mask = y > 0

    X_zero = X[zero_mask]
    y_zero = y[zero_mask]

    X_non_zero = X[non_zero_mask]
    y_non_zero = y[non_zero_mask]

    # Fazer undersampling para igualar ao número de amostras não-zero
    X_zero_downsampled, y_zero_downsampled = resample(
        X_zero, y_zero,
        replace=False,
        n_samples=len(y_non_zero),
        random_state=random_state
    )

    # Concatenar dados balanceados
    X_bal = pd.concat([X_zero_downsampled, X_non_zero], axis=0)
    y_bal = pd.concat([y_zero_downsampled, y_non_zero], axis=0)

    # Embaralhar
    X_bal, y_bal = resample(X_bal, y_bal, random_state=random_state)

    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True)