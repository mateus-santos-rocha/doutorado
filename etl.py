import pandas as pd
import duckdb

## ------------------------------------------------------------------------------------------------ ##
## --------------------------------------- LANDING TO BRONZE -------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

bronze_conn = duckdb.connect("bronze_db")

# Estações
estacoes_bronze_tables_dict = {
    'dim_estacoes':'bronze/dim_estacoes.csv',
    'fato_estacoes_precipitacao':'bronze/fato_estacoes_precipitacao.csv',
    'fato_estacoes_temperatura_maxima':'bronze/fato_estacoes_temperatura_maxima.csv',
    'fato_estacoes_temperatura_media':'bronze/fato_estacoes_temperatura_media.csv',
    'fato_estacoes_temperatura_minima':'bronze/fato_estacoes_temperatura_minima.csv',
    'fato_estacoes_umidade_relativa_maxima':'bronze/fato_estacoes_umidade_relativa_maxima.csv',
    'fato_estacoes_umidade_relativa_media':'bronze/fato_estacoes_umidade_relativa_media.csv',
    'fato_estacoes_umidade_relativa_minima':'bronze/fato_estacoes_umidade_relativa_minima.csv',
    'fato_estacoes_velocidade_vento_2m_maxima':'bronze/fato_estacoes_velocidade_vento_2m_maxima.csv',
    'fato_estacoes_velocidade_vento_2m_media':'bronze/fato_estacoes_velocidade_vento_2m_media.csv',
    'fato_estacoes_velocidade_vento_10m_media':'bronze/fato_estacoes_velocidade_vento_10m_media.csv',
}

for table_name,table_path in estacoes_bronze_tables_dict.items():
    bronze_conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
            SELECT
                *
            FROM read_csv('{table_path}')
""")
    

## ------------------------------------------------------------------------------------------------ ##
## ---------------------------------------- BRONZE TO PRATA --------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

bronze_conn = duckdb.connect("bronze_db")
bronze_conn.execute("INSTALL spatial; LOAD spatial")
prata_conn = duckdb.connect("prata_db")

## -------------------- ##
## MATRIZ DE DISTÂNCIAS ##
## -------------------- ##

prata_matriz_distancias_table_name  = "fato_estacoes_distancia"

df_matriz_distancias = bronze_conn.execute(f"""
    WITH estacoes AS (
        SELECT DISTINCT
            id_estacao
            ,ST_POINT(longitude,latitude) AS point
        FROM dim_estacoes)
    SELECT 
        estacoes_1.id_estacao as id_estacao_base
        ,CAST(ST_Distance_Sphere(estacoes_1.point,estacoes_2.point)/1000 AS DECIMAL(8,2))  AS vl_distancia_km
    FROM estacoes AS estacoes_1
    CROSS JOIN estacoes AS estacoes_2
    WHERE estacoes_1.id_estacao < estacoes_2.id_estacao
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_matriz_distancias_table_name} AS
    SELECT
        *
    FROM df_matriz_distancias
""")

## ------------- ##
## FATO ESTAÇÕES ##
## ------------- ##

prata_fato_estacoes_table_name = "fato_estacoes"

prata_fato_estacoes = bronze_conn.execute(f"""
SELECT
    precipitacao.dt_medicao
    ,precipitacao.id_estacao
    ,precipitacao.vl_precipitacao
    ,precipitacao.fl_dado_ausente AS fl_precipitacao_dado_ausente
    ,precipitacao.fl_dado_duvidoso AS fl_precipitacao_dado_duvidoso
    ,precipitacao.fl_dado_invalido AS fl_precipitacao_dado_invalido
    ,precipitacao.fl_dado_valido AS fl_precipitacao_dado_valido

    ,temperatura_maxima.vl_temperatura_maxima
    ,temperatura_maxima.fl_dado_ausente AS fl_temperatura_maxima_dado_ausente
    ,temperatura_maxima.fl_dado_duvidoso AS fl_temperatura_maxima_dado_duvidoso
    ,temperatura_maxima.fl_dado_invalido AS fl_temperatura_maxima_dado_invalido
    ,temperatura_maxima.fl_dado_valido AS fl_temperatura_maxima_dado_valido

    ,temperatura_media.vl_temperatura_media
    ,temperatura_media.fl_dado_ausente AS fl_temperatura_media_dado_ausente
    ,temperatura_media.fl_dado_duvidoso AS fl_temperatura_media_dado_duvidoso
    ,temperatura_media.fl_dado_invalido AS fl_temperatura_media_dado_invalido
    ,temperatura_media.fl_dado_valido AS fl_temperatura_media_dado_valido

    ,temperatura_minima.vl_temperatura_minima
    ,temperatura_minima.fl_dado_ausente AS fl_temperatura_minima_dado_ausente
    ,temperatura_minima.fl_dado_duvidoso AS fl_temperatura_minima_dado_duvidoso
    ,temperatura_minima.fl_dado_invalido AS fl_temperatura_minima_dado_invalido
    ,temperatura_minima.fl_dado_valido AS fl_temperatura_minima_dado_valido

    ,umidade_relativa_maxima.vl_umidade_relativa_maxima
    ,umidade_relativa_maxima.fl_dado_ausente AS fl_umidade_relativa_maxima_dado_ausente
    ,umidade_relativa_maxima.fl_dado_duvidoso AS fl_umidade_relativa_maxima_dado_duvidoso
    ,umidade_relativa_maxima.fl_dado_invalido AS fl_umidade_relativa_maxima_dado_invalido
    ,umidade_relativa_maxima.fl_dado_valido AS fl_umidade_relativa_maxima_dado_valido

    ,umidade_relativa_media.vl_umidade_relativa_media
    ,umidade_relativa_media.fl_dado_ausente AS fl_umidade_relativa_media_dado_ausente
    ,umidade_relativa_media.fl_dado_duvidoso AS fl_umidade_relativa_media_dado_duvidoso
    ,umidade_relativa_media.fl_dado_invalido AS fl_umidade_relativa_media_dado_invalido
    ,umidade_relativa_media.fl_dado_valido AS fl_umidade_relativa_media_dado_valido

    ,umidade_relativa_minima.vl_umidade_relativa_minima
    ,umidade_relativa_minima.fl_dado_ausente AS fl_umidade_relativa_minima_dado_ausente
    ,umidade_relativa_minima.fl_dado_duvidoso AS fl_umidade_relativa_minima_dado_duvidoso
    ,umidade_relativa_minima.fl_dado_invalido AS fl_umidade_relativa_minima_dado_invalido
    ,umidade_relativa_minima.fl_dado_valido AS fl_umidade_relativa_minima_dado_valido

    ,velocidade_vento_2m_maxima.vl_velocidade_vento_2m_maxima
    ,velocidade_vento_2m_maxima.fl_dado_ausente AS fl_velocidade_vento_2m_maxima_dado_ausente
    ,velocidade_vento_2m_maxima.fl_dado_duvidoso AS fl_velocidade_vento_2m_maxima_dado_duvidoso
    ,velocidade_vento_2m_maxima.fl_dado_invalido AS fl_velocidade_vento_2m_maxima_dado_invalido
    ,velocidade_vento_2m_maxima.fl_dado_valido AS fl_velocidade_vento_2m_maxima_dado_valido

    ,velocidade_vento_2m_media.vl_velocidade_vento_2m_media
    ,velocidade_vento_2m_media.fl_dado_ausente AS fl_velocidade_vento_2m_media_dado_ausente
    ,velocidade_vento_2m_media.fl_dado_duvidoso AS fl_velocidade_vento_2m_media_dado_duvidoso
    ,velocidade_vento_2m_media.fl_dado_invalido AS fl_velocidade_vento_2m_media_dado_invalido
    ,velocidade_vento_2m_media.fl_dado_valido AS fl_velocidade_vento_2m_media_dado_valido

    ,velocidade_vento_10m_media.vl_velocidade_vento_10m_media
    ,velocidade_vento_10m_media.fl_dado_ausente AS fl_velocidade_vento_10m_media_dado_ausente
    ,velocidade_vento_10m_media.fl_dado_duvidoso AS fl_velocidade_vento_10m_media_dado_duvidoso
    ,velocidade_vento_10m_media.fl_dado_invalido AS fl_velocidade_vento_10m_media_dado_invalido
    ,velocidade_vento_10m_media.fl_dado_valido AS fl_velocidade_vento_10m_media_dado_valido

FROM fato_estacoes_precipitacao AS precipitacao

LEFT JOIN fato_estacoes_temperatura_maxima AS temperatura_maxima
    ON precipitacao.dt_medicao = temperatura_maxima.dt_medicao
    AND precipitacao.id_estacao = temperatura_maxima.id_estacao

LEFT JOIN fato_estacoes_temperatura_media AS temperatura_media
    ON precipitacao.dt_medicao = temperatura_media.dt_medicao
    AND precipitacao.id_estacao = temperatura_media.id_estacao

LEFT JOIN fato_estacoes_temperatura_minima AS temperatura_minima
    ON precipitacao.dt_medicao = temperatura_minima.dt_medicao
    AND precipitacao.id_estacao = temperatura_minima.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_maxima AS umidade_relativa_maxima
    ON precipitacao.dt_medicao = umidade_relativa_maxima.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_maxima.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_media AS umidade_relativa_media
    ON precipitacao.dt_medicao = umidade_relativa_media.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_media.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_minima AS umidade_relativa_minima
    ON precipitacao.dt_medicao = umidade_relativa_minima.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_minima.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_2m_maxima AS velocidade_vento_2m_maxima
    ON precipitacao.dt_medicao = velocidade_vento_2m_maxima.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_2m_maxima.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_2m_media AS velocidade_vento_2m_media
    ON precipitacao.dt_medicao = velocidade_vento_2m_media.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_2m_media.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_10m_media AS velocidade_vento_10m_media
    ON precipitacao.dt_medicao = velocidade_vento_10m_media.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_10m_media.id_estacao
                    
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_fato_estacoes_table_name} AS
    SELECT
        *
    FROM prata_fato_estacoes
""")

## ------------ ##
## DIM ESTAÇÕES ##
## ------------ ##

prata_dim_estacoes_table_name = "dim_estacoes"

bronze_dim_estacoes = bronze_conn.execute("SELECT * FROM dim_estacoes").fetch_df()

prata_dim_estacoes = bronze_dim_estacoes.copy()

prata_dim_estacoes['nm_instituicao'] = prata_dim_estacoes['nm_instituicao'].astype(str) \
    .str.replace('\n','') \
    .replace(r'\s+', ' ', regex=True)

prata_dim_estacoes['nm_estacao'] = prata_dim_estacoes['nm_estacao'].astype(str).replace(r'\s+', ' ', regex=True)
    
prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_dim_estacoes_table_name} AS 
    SELECT
        *
    FROM prata_dim_estacoes
    """)




    