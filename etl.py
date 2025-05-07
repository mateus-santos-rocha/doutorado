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