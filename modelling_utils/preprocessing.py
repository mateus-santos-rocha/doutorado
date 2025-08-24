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

def feature_selection(df,feature_list,target):
    X = df[feature_list]
    y = df[target]
    return X,y

def particao_por_estacao(df,percent_datetime_partitioning_split):
    df_treinamento = df \
        .groupby('id_estacao', as_index=False).apply(lambda g: g.iloc[:max(1, int(len(g) * percent_datetime_partitioning_split))]) \
        .reset_index(drop=True)
    
    df_validacao = df \
        .groupby('id_estacao', as_index=False).apply(lambda g: g.iloc[max(1, int(len(g) * percent_datetime_partitioning_split)):]) \
        .reset_index(drop=True)

    return df_treinamento,df_validacao


def drop_estacoes_vizinhas(abt_estacoes_vizinhas):
    dropped_df = abt_estacoes_vizinhas.drop(['vl_precipitacao_vizinha','vl_correlacao_vizinha','pct_intersecao_precipitacao_vizinha','vl_distancia_km_vizinha','estacao_vizinha_escolhida','vl_prioridade_vizinha'],axis=1)
    return dropped_df