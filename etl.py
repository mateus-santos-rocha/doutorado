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




    